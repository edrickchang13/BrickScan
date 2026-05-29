"""
Scheduled FLYWHEEL refresh — operationalise the proven confirm→append loop.

The flywheel hot path (app/ml/flywheel_ingest.py + /flywheel/confirm) folds a
single confirmed scan into the galleries the moment a user confirms it. This
module is the *batch* counterpart that turns that one-shot mechanism into a
running loop:

  refresh_galleries(db)   Replay every CONFIRMED ScanFeedback row that has a
                          stored crop but has NOT yet been folded into the
                          galleries (used_for_training == False), re-embedding
                          and appending each via the SAME hot path
                          (ingest_confirmed). Marks each row used_for_training
                          on success so re-runs are idempotent — the flag is the
                          durable cursor, so the job is resumable after a crash.
                          NO retraining: this only grows the gallery the k-NN
                          searches, exactly like a live confirmation.

  heavy_refresh_status()  Report whether the slower-cadence heavy refresh (a
                          full re-embed / hand-off to the weekly distillation in
                          ml/scripts/active_learning_cron.sh, "Track D") is wired
                          up. The fast append loop above never needs it; this is
                          the occasional gallery-compaction / student-retrain
                          path that the cron script triggers on a slower cadence.

Why replay at all, when /flywheel/confirm already appends live?
  - Resilience: if the encoder ONNX wasn't deployed when a confirmation arrived,
    ingest_confirmed journals the crop but skips the gallery append (by design —
    it never breaks a scan). Once the encoder is deployed, this job folds those
    backlogged crops in.
  - Bulk import: corrections submitted via /scan-feedback (not /flywheel/confirm)
    also land labelled crops on disk; this job appends them too.
  - A restored / rebuilt gallery can be re-warmed from the confirmed history.

Everything degrades gracefully and never raises into the caller: a missing
encoder, a missing crop file, or a single bad row is logged and skipped; the
batch keeps going and the run summary records what happened.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# backend/ root (this file is backend/app/ml/flywheel_refresh.py).
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
# Durable run-state journal: last run summary + a monotonically-growing total.
# The authoritative idempotency cursor is ScanFeedback.used_for_training in the
# DB; this file is for monitoring / "when did the loop last run" introspection.
REFRESH_STATE_DIR = _BACKEND_DIR / "data" / "flywheel" / "refresh_state"
REFRESH_STATE_FILE = REFRESH_STATE_DIR / "last_refresh.json"

# The slower-cadence heavy refresh (full re-embed / student distillation) is
# delegated to this script when it exists. Referenced in ml/FLYWHEEL.md as
# "Track D"; the fast append loop here does NOT depend on it.
HEAVY_REFRESH_SCRIPT = _BACKEND_DIR.parent / "ml" / "scripts" / "active_learning_cron.sh"


@dataclass
class RefreshResult:
    """Outcome of one batch refresh pass over the confirmed feedback backlog."""
    started_at: float
    finished_at: float = 0.0
    candidates: int = 0          # rows eligible (un-ingested, with a crop on disk)
    ingested: int = 0            # rows whose crop produced a gallery append
    embedded: int = 0            # rows the frozen encoder actually embedded
    skipped_no_image: int = 0    # rows whose crop file was missing/unreadable
    skipped_no_part: int = 0     # rows with an empty/unknown part_num
    errors: int = 0              # rows that raised (logged, then skipped)
    marked_used: int = 0         # rows flipped used_for_training=True this pass
    visual_search_size_before: int = 0
    visual_search_size_after: int = 0
    embedding_library_size_before: int = 0
    embedding_library_size_after: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.finished_at or time.time()) - self.started_at)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 3),
            "candidates": self.candidates,
            "ingested": self.ingested,
            "embedded": self.embedded,
            "skipped_no_image": self.skipped_no_image,
            "skipped_no_part": self.skipped_no_part,
            "errors": self.errors,
            "marked_used": self.marked_used,
            "visual_search_size_before": self.visual_search_size_before,
            "visual_search_size_after": self.visual_search_size_after,
            "embedding_library_size_before": self.embedding_library_size_before,
            "embedding_library_size_after": self.embedding_library_size_after,
            "notes": self.notes,
        }


def _read_crop_bytes(image_path: Optional[str]) -> Optional[bytes]:
    """Read a journaled crop back off disk. Returns None if absent/unreadable."""
    if not image_path:
        return None
    try:
        p = Path(image_path)
        if not p.is_file():
            return None
        return p.read_bytes()
    except Exception as e:  # noqa: BLE001 — never let one bad file kill the batch
        logger.warning("flywheel_refresh: cannot read crop %s: %s", image_path, e)
        return None


def refresh_galleries(
    db,
    *,
    limit: int = 1000,
    mark_used: bool = True,
    only_corrections: bool = False,
    dry_run: bool = False,
) -> RefreshResult:
    """Fold the confirmed-feedback backlog into the galleries (NO retraining).

    Pulls ScanFeedback rows with a stored crop that have not yet been folded in
    (used_for_training == False), re-embeds each crop with the frozen encoder and
    appends it through ingest_confirmed — the identical hot path a live
    /flywheel/confirm uses. On success the row is flipped used_for_training=True
    so the next run skips it (idempotent + resumable; the DB flag is the cursor).

    Args:
        db: a synchronous SQLAlchemy Session (app.local_inventory get_local_db).
        limit: cap rows processed per pass (keeps a single run bounded).
        mark_used: flip used_for_training=True on appended rows (the cursor).
                   Set False to re-warm a gallery from history without consuming
                   the backlog.
        only_corrections: when True, only replay rows where the model was wrong
                          (correct != predicted) — the most informative exemplars.
        dry_run: count candidates and report sizes but append nothing.

    Returns a RefreshResult. Never raises on a missing model or a bad row.
    """
    from app.local_inventory.models import ScanFeedback as ScanFeedbackModel

    result = RefreshResult(started_at=time.time())

    # Gallery sizes before, for the delta in the run summary + monitoring.
    before = _gallery_sizes()
    result.visual_search_size_before = before["visual_search_size"]
    result.embedding_library_size_before = before["embedding_library_size"]

    # Eligible rows: have a crop on disk, not yet folded in. Oldest first so a
    # capped run drains the backlog in arrival order (FIFO, fully resumable).
    query = (
        db.query(ScanFeedbackModel)
        .filter(ScanFeedbackModel.image_path.isnot(None))
        .filter(ScanFeedbackModel.used_for_training == False)  # noqa: E712
        .order_by(ScanFeedbackModel.timestamp.asc())
        .limit(max(1, min(limit, 10000)))
    )
    rows = query.all()
    result.candidates = len(rows)

    if dry_run:
        result.notes.append("dry_run: counted candidates, appended nothing")
        result.finished_at = time.time()
        after = _gallery_sizes()
        result.visual_search_size_after = after["visual_search_size"]
        result.embedding_library_size_after = after["embedding_library_size"]
        _write_state(result)
        return result

    from app.ml.flywheel_ingest import ingest_confirmed

    for row in rows:
        try:
            # Optionally restrict to corrections (model was wrong).
            if only_corrections:
                pred = (row.predicted_part_num or "").strip().lower()
                corr = (row.correct_part_num or "").strip().lower()
                if pred == corr:
                    continue

            part_num = (row.correct_part_num or "").strip()
            if not part_num or part_num.lower() == "unknown":
                result.skipped_no_part += 1
                continue

            crop = _read_crop_bytes(row.image_path)
            if crop is None:
                result.skipped_no_image += 1
                continue

            ingest = ingest_confirmed(
                crop_bytes=crop,
                part_num=part_num,
                color_id=row.correct_color_id,
                journal_crops=False,  # the crop is already on disk — don't dup it
            )
            if ingest.embedded:
                result.embedded += 1
            if ingest.visual_search_added or ingest.embedding_library_added:
                result.ingested += 1
                if mark_used:
                    row.used_for_training = True
                    result.marked_used += 1
            else:
                # Nothing was appended (e.g. encoder not deployed). Leave
                # used_for_training=False so a later run retries this row.
                if ingest.notes:
                    for n in ingest.notes:
                        if n not in result.notes:
                            result.notes.append(n)
        except Exception as e:  # noqa: BLE001
            logger.warning("flywheel_refresh: row %s failed: %s",
                           getattr(row, "id", "?"), e)
            result.errors += 1

    # One commit for the whole batch (the flag flips). A crash before commit
    # just means those rows get retried next run — still idempotent.
    if result.marked_used:
        try:
            db.commit()
        except Exception as e:  # noqa: BLE001
            logger.error("flywheel_refresh: commit failed, rolling back: %s", e)
            db.rollback()
            result.notes.append(f"commit_failed: {e}")
            result.marked_used = 0

    result.finished_at = time.time()
    after = _gallery_sizes()
    result.visual_search_size_after = after["visual_search_size"]
    result.embedding_library_size_after = after["embedding_library_size"]

    logger.info(
        "flywheel_refresh: candidates=%d ingested=%d embedded=%d "
        "skip_img=%d skip_part=%d errors=%d  vs=%d→%d el=%d→%d (%.2fs)",
        result.candidates, result.ingested, result.embedded,
        result.skipped_no_image, result.skipped_no_part, result.errors,
        result.visual_search_size_before, result.visual_search_size_after,
        result.embedding_library_size_before, result.embedding_library_size_after,
        result.duration_s,
    )
    _write_state(result)
    return result


def _gallery_sizes() -> Dict[str, int]:
    """Cheap (visual_search, embedding_library) sizes — reuses gallery_status."""
    try:
        from app.ml.flywheel_ingest import gallery_status
        s = gallery_status()
        return {
            "visual_search_size": int(s.get("visual_search_size", 0) or 0),
            "embedding_library_size": int(s.get("embedding_library_size", 0) or 0),
        }
    except Exception:  # noqa: BLE001
        return {"visual_search_size": 0, "embedding_library_size": 0}


def _write_state(result: RefreshResult) -> None:
    """Persist the last-run summary + a cumulative counter for monitoring.

    Idempotent and best-effort: a failure to journal never affects the refresh.
    """
    try:
        REFRESH_STATE_DIR.mkdir(parents=True, exist_ok=True)
        prev_total = 0
        prev_runs = 0
        if REFRESH_STATE_FILE.exists():
            try:
                prev = json.loads(REFRESH_STATE_FILE.read_text())
                prev_total = int(prev.get("cumulative_ingested", 0) or 0)
                prev_runs = int(prev.get("total_runs", 0) or 0)
            except Exception:  # noqa: BLE001
                pass
        blob = {
            "last_run": result.as_dict(),
            "cumulative_ingested": prev_total + result.ingested,
            "total_runs": prev_runs + 1,
            "updated_at": time.time(),
        }
        tmp = REFRESH_STATE_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(blob, indent=2))
        tmp.replace(REFRESH_STATE_FILE)
    except Exception as e:  # noqa: BLE001
        logger.warning("flywheel_refresh: could not write state file: %s", e)


def read_state() -> Optional[Dict[str, Any]]:
    """Return the persisted last-refresh summary, or None if never run."""
    try:
        if REFRESH_STATE_FILE.exists():
            return json.loads(REFRESH_STATE_FILE.read_text())
    except Exception:  # noqa: BLE001
        pass
    return None


def heavy_refresh_status() -> Dict[str, Any]:
    """Is the slower-cadence heavy refresh (re-embed / distillation) wired up?

    The fast append loop (refresh_galleries) is self-sufficient and needs NO
    retraining. The heavy refresh is the occasional path that compresses the
    grown gallery back into the backbone (the weekly student distillation in
    ml/scripts/active_learning_cron.sh, "Track D"). This reports whether that
    script exists so the cron wrapper can decide whether to invoke it; we never
    launch a heavy GPU job from inside the request process.
    """
    exists = HEAVY_REFRESH_SCRIPT.exists()
    return {
        "heavy_refresh_script": str(HEAVY_REFRESH_SCRIPT),
        "available": exists,
        "note": (
            "delegated to active_learning_cron.sh on a slower cadence"
            if exists else
            "active_learning_cron.sh not present — heavy refresh is a no-op; the "
            "fast append loop (refresh_galleries) raises accuracy without it"
        ),
    }
