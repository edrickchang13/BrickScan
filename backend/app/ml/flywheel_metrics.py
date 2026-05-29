"""
FLYWHEEL monitoring — accuracy / coverage snapshots over time.

Complements the weekly accuracy snapshot (POST /feedback/snapshot → the
FeedbackEvalSnapshot top1/top3 trend) with an *operational* health summary of
the flywheel itself:

  flywheel_metrics(db)   A point-in-time summary:
                           - gallery sizes (visual_search rows, EmbeddingLibrary
                             part-prototypes) + whether the encoder is deployed
                           - coverage: distinct parts / colours seen in confirmed
                             feedback, journaled colour exemplars pending an
                             offline colour-gallery rebuild
                           - recent correction volume (last 24h / 7d / 30d)
                           - top confusion pairs (predicted → corrected) so the
                             ops view shows what the model is still getting wrong
                           - the latest accuracy snapshot (top1/top3) for context
                           - the last refresh-job run summary

  write_metrics_snapshot(db)  Compute the summary and journal it to
                              data/flywheel/metrics/<UTC-stamp>.json (plus a
                              stable latest.json), so a cron can build a time
                              series on disk without a DB migration. Returns the
                              (summary, path) pair.

The heavy lifting (top-N accuracy, confusion counting) reuses the exact helpers
the feedback stats endpoint already uses, so the numbers match what the
FeedbackStatsScreen shows. Everything is read-only and degrades gracefully.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = _BACKEND_DIR / "data" / "flywheel" / "metrics"


def _norm(p: Optional[str]) -> str:
    return (p or "").strip().lower().lstrip("0") or "0"


def _confusion_pairs_and_coverage(rows) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Top confusion pairs + coverage counts over a list of ScanFeedback rows.

    A "confusion" is a row whose corrected part differs from the predicted part.
    Coverage counts the distinct parts and colours that confirmed feedback has
    ground-truth for (i.e. what the flywheel has learned to recognise).
    Pure over the rows — no DB or model access — so it is unit-testable.
    """
    pair_counter: Counter = Counter()
    parts_seen = set()
    colors_seen = set()
    for r in rows:
        corr = (r.correct_part_num or "").strip()
        pred = (r.predicted_part_num or "").strip()
        if corr:
            parts_seen.add(_norm(corr))
        cid = (str(r.correct_color_id).strip() if r.correct_color_id is not None else "")
        if cid:
            colors_seen.add(cid)
        if corr and pred and _norm(corr) != _norm(pred):
            pair_counter[(pred, corr)] += 1
    top_pairs = [
        {"predicted_part_num": pred, "correct_part_num": corr, "count": cnt}
        for (pred, corr), cnt in pair_counter.most_common(10)
    ]
    coverage = {
        "parts_with_feedback": len(parts_seen),
        "colors_with_feedback": len(colors_seen),
    }
    return top_pairs, coverage


def _recent_volume(rows, now: Optional[datetime] = None) -> Dict[str, int]:
    """Count confirmed-feedback rows in the last 24h / 7d / 30d. Pure."""
    now = now or datetime.now(timezone.utc)
    cutoffs = {"last_24h": now - timedelta(days=1),
               "last_7d": now - timedelta(days=7),
               "last_30d": now - timedelta(days=30)}
    out = {k: 0 for k in cutoffs}
    out["total"] = 0
    for r in rows:
        out["total"] += 1
        ts = r.timestamp
        if ts is None:
            continue
        # Tolerate naive timestamps (older SQLite rows) by assuming UTC.
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        for k, cutoff in cutoffs.items():
            if ts >= cutoff:
                out[k] += 1
    return out


def flywheel_metrics(db) -> Dict[str, Any]:
    """Point-in-time operational health summary of the flywheel.

    Read-only. `db` is a synchronous SQLAlchemy Session (get_local_db). Safe to
    call when the galleries / encoder aren't deployed — those fields just read 0.
    """
    from app.local_inventory.models import (
        ScanFeedback as ScanFeedbackModel,
        FeedbackEvalSnapshot as FeedbackEvalSnapshotModel,
    )

    all_rows = db.query(ScanFeedbackModel).all()
    top_pairs, coverage = _confusion_pairs_and_coverage(all_rows)
    volume = _recent_volume(all_rows)

    # Live gallery + encoder snapshot (reuses the ingest module's status probe).
    gallery: Dict[str, Any] = {}
    try:
        from app.ml.flywheel_ingest import gallery_status
        gallery = gallery_status()
    except Exception as e:  # noqa: BLE001
        logger.warning("flywheel_metrics: gallery_status failed: %s", e)
        gallery = {}

    # Latest frozen accuracy snapshot (top1/top3), for trend context.
    latest_snapshot: Optional[Dict[str, Any]] = None
    try:
        s = (
            db.query(FeedbackEvalSnapshotModel)
            .order_by(FeedbackEvalSnapshotModel.snapshot_date.desc())
            .first()
        )
        if s is not None:
            latest_snapshot = {
                "snapshot_date": s.snapshot_date.isoformat() if s.snapshot_date else None,
                "top1_accuracy": s.top1_accuracy,
                "top3_accuracy": s.top3_accuracy,
                "sample_size": s.sample_size,
                "window_days": s.window_days,
            }
    except Exception as e:  # noqa: BLE001
        logger.warning("flywheel_metrics: snapshot read failed: %s", e)

    # Last refresh-job run summary (when the append loop last ran).
    last_refresh: Optional[Dict[str, Any]] = None
    try:
        from app.ml.flywheel_refresh import read_state
        st = read_state()
        if st:
            last_refresh = {
                "updated_at": st.get("updated_at"),
                "cumulative_ingested": st.get("cumulative_ingested"),
                "total_runs": st.get("total_runs"),
                "last_run": st.get("last_run"),
            }
    except Exception as e:  # noqa: BLE001
        logger.warning("flywheel_metrics: refresh state read failed: %s", e)

    pending_ingest = (
        db.query(ScanFeedbackModel)
        .filter(ScanFeedbackModel.image_path.isnot(None))
        .filter(ScanFeedbackModel.used_for_training == False)  # noqa: E712
        .count()
    )

    return {
        "generated_at": time.time(),
        "generated_at_iso": datetime.now(timezone.utc).isoformat(),
        "gallery": {
            "encoder_available": bool(gallery.get("encoder_available", False)),
            "visual_search_loaded": bool(gallery.get("visual_search_loaded", False)),
            "visual_search_size": int(gallery.get("visual_search_size", 0) or 0),
            "embedding_library_size": int(gallery.get("embedding_library_size", 0) or 0),
            "color_exemplars_pending": int(gallery.get("color_exemplars_pending", 0) or 0),
            "part_exemplars_journaled": int(gallery.get("part_exemplars_journaled", 0) or 0),
        },
        "coverage": coverage,
        "correction_volume": volume,
        "pending_ingest": int(pending_ingest),
        "top_confusion_pairs": top_pairs,
        "latest_accuracy_snapshot": latest_snapshot,
        "last_refresh": last_refresh,
    }


def write_metrics_snapshot(db) -> Tuple[Dict[str, Any], str]:
    """Compute the summary and journal it to data/flywheel/metrics/.

    Writes a UTC-stamped file (the time series) AND overwrites latest.json (a
    stable handle for dashboards / quick reads). Returns (summary, path_written).
    Best-effort on the file write — the summary is always returned.
    """
    summary = flywheel_metrics(db)
    path_str = ""
    try:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = METRICS_DIR / f"{stamp}.json"
        out.write_text(json.dumps(summary, indent=2))
        # Stable latest pointer (atomic replace).
        latest = METRICS_DIR / "latest.json"
        tmp = latest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=2))
        tmp.replace(latest)
        path_str = str(out)
        logger.info("flywheel_metrics: snapshot written → %s", out)
    except Exception as e:  # noqa: BLE001
        logger.warning("flywheel_metrics: could not write snapshot: %s", e)
    return summary, path_str
