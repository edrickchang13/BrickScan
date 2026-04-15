"""
Feedback & updated scan-multi routes — Workstream B + Workstream A step 5.

Endpoints added:
  POST /api/local-inventory/scan-feedback      — save user correction
  GET  /api/local-inventory/feedback/stats     — confusion-pair analytics
  POST /api/local-inventory/scan-multi         — YOLO-aware version (overrides
                                                  the quadrant-split in routes.py
                                                  when registered first)

HOW TO REGISTER (add to main.py before the existing local_inventory router):
---------------------------------------------------------------------------
    from app.local_inventory.feedback_routes import feedback_router
    app.include_router(feedback_router)       # add BEFORE existing router
---------------------------------------------------------------------------
"""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.local_inventory.database import get_local_db
from app.local_inventory.models import (
    ScanFeedback as ScanFeedbackModel,
    FeedbackEvalSnapshot as FeedbackEvalSnapshotModel,
)
from app.local_inventory.schemas_feedback import (
    AccuracyTrendPoint,
    FeedbackConfusionPair,
    FeedbackSnapshotResponse,
    FeedbackStats,
    ScanFeedback,
    ScanFeedbackResponse,
    SourceStats,
)

# Runtime imports from schemas.py (readable by the server process)
from app.local_inventory.schemas import (   # noqa: E402
    DetectedPiece,
    MultiPieceScanRequest,
    MultiPieceScanResponse,
    ScanPrediction,
)
from app.services.hybrid_recognition import hybrid_predict

logger = logging.getLogger(__name__)

# Directory where labelled images are stored
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
FEEDBACK_IMAGES_DIR = _BACKEND_DIR / "data" / "feedback_images"

feedback_router = APIRouter(
    prefix="/api/local-inventory",
    tags=["feedback"],
)


# ── POST /scan-feedback ───────────────────────────────────────────────────────

@feedback_router.post("/scan-feedback", response_model=ScanFeedbackResponse)
async def submit_scan_feedback(
    request: ScanFeedback,
    db: Session = Depends(get_local_db),
) -> ScanFeedbackResponse:
    """
    Accept a user correction and persist it for active learning.

    - If correct_part_num != predicted_part_num AND image_base64 is provided,
      saves a labelled JPEG to data/feedback_images/<correct_part_num>/<scan_id>.jpg.
    - v2 clients send `feedback_type`, `correct_rank`, `predictions_shown`, and
      `time_to_confirm_ms`. When absent (legacy client), feedback_type is
      derived heuristically from part_num comparison and correct_rank is None.
    - Sets `will_improve_model=true` when a new labelled image was saved.
    """
    is_correction = (
        request.correct_part_num.strip().lower()
        != request.predicted_part_num.strip().lower()
    )

    # Derive feedback_type if the client didn't send one (legacy path).
    feedback_type = request.feedback_type or _derive_feedback_type(
        predicted=request.predicted_part_num,
        correct=request.correct_part_num,
        correct_color_id=request.correct_color_id,
        shown=request.predictions_shown,
    )

    image_path: str | None = None

    if is_correction and request.image_base64:
        try:
            img_dir = FEEDBACK_IMAGES_DIR / request.correct_part_num
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = img_dir / f"{request.scan_id}.jpg"

            # Decode base64 → JPEG bytes
            raw = request.image_base64
            if "," in raw:
                raw = raw.split(",", 1)[1]
            img_bytes = base64.b64decode(raw)

            # Re-encode as JPEG for storage efficiency
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            img.save(str(img_path), format="JPEG", quality=85)
            image_path = str(img_path)
            logger.info("Saved feedback image: %s", img_path)
        except Exception as e:
            logger.warning("Failed to save feedback image: %s", e)
            # Don't fail the whole request — continue without image

    predictions_shown_json: Optional[str] = None
    if request.predictions_shown:
        try:
            predictions_shown_json = json.dumps(
                [p.model_dump() for p in request.predictions_shown],
                separators=(",", ":"),
            )
        except Exception as e:
            logger.warning("Failed to serialise predictions_shown: %s", e)

    # Persist to DB
    record = ScanFeedbackModel(
        scan_id=request.scan_id,
        predicted_part_num=request.predicted_part_num,
        correct_part_num=request.correct_part_num,
        correct_color_id=request.correct_color_id,
        image_path=image_path,
        confidence=request.confidence,
        source=request.source,
        used_for_training=False,
        feedback_type=feedback_type,
        correct_rank=request.correct_rank,
        predictions_shown_json=predictions_shown_json,
        time_to_confirm_ms=request.time_to_confirm_ms,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    logger.info(
        "Feedback saved: scan_id=%s  %s→%s  type=%s  rank=%s",
        request.scan_id,
        request.predicted_part_num,
        request.correct_part_num,
        feedback_type,
        request.correct_rank,
    )

    return ScanFeedbackResponse(
        saved=True,
        will_improve_model=is_correction and image_path is not None,
        feedback_id=record.id,
    )


def _derive_feedback_type(
    predicted: str,
    correct: str,
    correct_color_id: Optional[str],
    shown: Optional[list],
) -> str:
    """
    Fallback classifier for legacy clients that don't send feedback_type.

    Order of checks:
      1. Same part_num + color given → partially_correct (only colour changed)
      2. Same part_num (no color diff signal) → top_correct
      3. Different part_num, and the correct one appears in the shown top-5 → alternative_correct
      4. Different part_num, not in shown list → none_correct
    """
    pred_norm = predicted.strip().lower()
    corr_norm = correct.strip().lower()

    if pred_norm == corr_norm:
        return "partially_correct" if correct_color_id else "top_correct"

    if shown:
        for entry in shown:
            part_num = getattr(entry, "part_num", None) if not isinstance(entry, dict) else entry.get("part_num")
            if part_num and str(part_num).strip().lower() == corr_norm:
                return "alternative_correct"
    return "none_correct"


# ── GET /feedback/stats ───────────────────────────────────────────────────────

@feedback_router.get("/feedback/stats", response_model=FeedbackStats)
async def get_feedback_stats(db: Session = Depends(get_local_db)) -> FeedbackStats:
    """
    Return aggregate statistics for the active-learning feedback collection,
    including v2 accuracy metrics (top-1, top-3, by_source, accuracy_trend).

    Legacy fields (total_corrections, agreement_count, …) are preserved so
    any older client consuming this endpoint keeps working.
    """
    all_feedback = db.query(ScanFeedbackModel).all()

    # ── Legacy confusion / correction counts ──────────────────────────────────
    corrections = [
        f for f in all_feedback
        if f.correct_part_num.strip().lower() != f.predicted_part_num.strip().lower()
    ]
    agreements = len(all_feedback) - len(corrections)

    pair_counter: Counter = Counter()
    for f in corrections:
        pair_counter[(f.predicted_part_num, f.correct_part_num)] += 1
    top_pairs = [
        FeedbackConfusionPair(
            predicted_part_num=pred,
            correct_part_num=correct,
            count=cnt,
        )
        for (pred, correct), cnt in pair_counter.most_common(10)
    ]

    images_saved = sum(1 for f in corrections if f.image_path is not None)
    parts_with_feedback = len({f.correct_part_num for f in corrections if f.image_path})

    pending_training = (
        db.query(func.count(ScanFeedbackModel.id))
        .filter(ScanFeedbackModel.used_for_training == False)  # noqa: E712
        .scalar()
        or 0
    )

    # ── v2: rolling 30-day accuracy + per-source + trend ──────────────────────
    window_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    recent = [f for f in all_feedback if f.timestamp and f.timestamp >= window_cutoff]
    top1_acc, top3_acc = _compute_topn_accuracy(recent)
    by_source = _compute_by_source(recent)

    # Trend = last 8 snapshots, most-recent last (for chart sorting)
    snapshots = (
        db.query(FeedbackEvalSnapshotModel)
        .order_by(FeedbackEvalSnapshotModel.snapshot_date.desc())
        .limit(8)
        .all()
    )
    accuracy_trend = [
        AccuracyTrendPoint(
            week_ending=s.snapshot_date.date().isoformat(),
            top1_accuracy=s.top1_accuracy,
            top3_accuracy=s.top3_accuracy,
            sample_size=s.sample_size,
        )
        for s in reversed(snapshots)
    ]

    return FeedbackStats(
        total_corrections=len(corrections),
        agreement_count=agreements,
        top_confused_pairs=top_pairs,
        parts_with_feedback=parts_with_feedback,
        images_saved=images_saved,
        pending_training=int(pending_training),
        top1_accuracy=top1_acc,
        top3_accuracy=top3_acc,
        by_source=by_source,
        accuracy_trend=accuracy_trend,
    )


def _compute_topn_accuracy(rows: List[ScanFeedbackModel]) -> tuple[float, float]:
    """
    Compute top-1 and top-3 accuracy over the given feedback rows.

    Rules:
      top-1 correct: feedback_type == 'top_correct' OR 'partially_correct'
                     (brick identity was right, even if colour was off)
      top-3 correct: top-1 OR (feedback_type == 'alternative_correct' AND correct_rank <= 2)

    Rows without feedback_type (legacy) are included but only contribute to
    top-1 when predicted == correct.
    """
    if not rows:
        return (0.0, 0.0)
    top1 = 0
    top3 = 0
    for r in rows:
        ft = r.feedback_type
        if ft in ("top_correct", "partially_correct"):
            top1 += 1
            top3 += 1
        elif ft == "alternative_correct":
            if r.correct_rank is not None and r.correct_rank <= 2:
                top3 += 1
        elif ft is None:
            # Legacy row
            if r.predicted_part_num.strip().lower() == r.correct_part_num.strip().lower():
                top1 += 1
                top3 += 1
    return (top1 / len(rows), top3 / len(rows))


def _compute_by_source(rows: List[ScanFeedbackModel]) -> List[SourceStats]:
    """Count correct vs. wrong by `source` field (which model made the top prediction)."""
    buckets: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0, "correct": 0})
    for r in rows:
        src = (r.source or "unknown").strip().lower() or "unknown"
        buckets[src]["count"] += 1
        ft = r.feedback_type
        if ft in ("top_correct", "partially_correct"):
            buckets[src]["correct"] += 1
        elif ft is None and r.predicted_part_num.strip().lower() == r.correct_part_num.strip().lower():
            buckets[src]["correct"] += 1
    return [
        SourceStats(
            source=src,
            count=data["count"],
            correct=data["correct"],
            accuracy=(data["correct"] / data["count"]) if data["count"] else 0.0,
        )
        for src, data in sorted(buckets.items(), key=lambda kv: kv[1]["count"], reverse=True)
    ]


# ── GET /feedback/export.csv ──────────────────────────────────────────────────

@feedback_router.get("/feedback/export.csv")
async def export_feedback_csv(
    include_used: bool = False,
    db: Session = Depends(get_local_db),
) -> Response:
    """
    Stream all corrections not yet used for training as a CSV ready for
    `ml/retrain_from_feedback.py`.

    Columns: image_path, correct_part_num, correct_color_id, original_prediction,
             source, confidence, timestamp, scan_id, feedback_type, correct_rank

    Query params:
      include_used=true  — also include rows where used_for_training=True
                           (useful for full-history audits; default false)
    """
    query = db.query(ScanFeedbackModel).filter(
        ScanFeedbackModel.image_path.isnot(None)  # can't train without an image
    )
    if not include_used:
        query = query.filter(ScanFeedbackModel.used_for_training == False)  # noqa: E712

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=[
        "image_path", "correct_part_num", "correct_color_id",
        "original_prediction", "source", "confidence",
        "timestamp", "scan_id", "feedback_type", "correct_rank",
    ])
    writer.writeheader()
    count = 0
    for r in query.yield_per(500):
        writer.writerow({
            "image_path":          r.image_path,
            "correct_part_num":    r.correct_part_num,
            "correct_color_id":    r.correct_color_id or "",
            "original_prediction": r.predicted_part_num,
            "source":              r.source,
            "confidence":          f"{r.confidence:.4f}",
            "timestamp":           r.timestamp.isoformat() if r.timestamp else "",
            "scan_id":             r.scan_id,
            "feedback_type":       r.feedback_type or "",
            "correct_rank":        r.correct_rank if r.correct_rank is not None else "",
        })
        count += 1
    logger.info("Feedback CSV export: %d rows", count)

    headers = {
        "Content-Disposition": f'attachment; filename="feedback_export_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.csv"',
        "X-Row-Count": str(count),
    }
    return Response(content=buf.getvalue(), media_type="text/csv", headers=headers)


# ── GET /feedback/eval-set.json ───────────────────────────────────────────────

@feedback_router.get("/feedback/eval-set.json")
async def export_eval_set(
    limit: int = 500,
    include_used: bool = False,
    db: Session = Depends(get_local_db),
) -> Response:
    """
    Return the most recent N ScanFeedback rows with ground-truth labels as JSON,
    for programmatic evaluation via `ml/scripts/eval_against_feedback.py`.

    Only rows where:
      - feedback_type IN ('top_correct', 'alternative_correct', 'none_correct')
      - image_path IS NOT NULL

    `partially_correct` is excluded because the user confirmed the part
    identity was right (only the colour was wrong) — that row isn't useful
    signal for a part-classification eval.

    Shape of each element:
      {
        "image_path":           "...",
        "correct_part_num":     "3001",
        "correct_color_id":     "4",
        "original_prediction":  "3002",
        "source":               "brickognize",
        "confidence":           0.73,
        "timestamp":            "2026-04-15T00:32:01Z",
        "scan_id":              "scan_...",
        "feedback_type":        "alternative_correct",
        "correct_rank":         1,
        "predictions_shown":    [{...}, ...] | null
      }
    """
    valid_types = ("top_correct", "alternative_correct", "none_correct")
    query = (
        db.query(ScanFeedbackModel)
        .filter(ScanFeedbackModel.image_path.isnot(None))
        .filter(ScanFeedbackModel.feedback_type.in_(valid_types))
    )
    if not include_used:
        query = query.filter(ScanFeedbackModel.used_for_training == False)  # noqa: E712
    # Apply ordering + limit AFTER all filters (SQLAlchemy rejects filter
    # after limit/offset is materialised).
    query = query.order_by(ScanFeedbackModel.timestamp.desc()).limit(max(1, min(limit, 5000)))

    rows = []
    for r in query.all():
        predictions_shown = None
        if r.predictions_shown_json:
            try:
                import json as _json
                predictions_shown = _json.loads(r.predictions_shown_json)
            except Exception:
                predictions_shown = None
        rows.append({
            "image_path":          r.image_path,
            "correct_part_num":    r.correct_part_num,
            "correct_color_id":    r.correct_color_id,
            "original_prediction": r.predicted_part_num,
            "source":              r.source,
            "confidence":          round(float(r.confidence), 4) if r.confidence is not None else None,
            "timestamp":           r.timestamp.isoformat() if r.timestamp else None,
            "scan_id":             r.scan_id,
            "feedback_type":       r.feedback_type,
            "correct_rank":        r.correct_rank,
            "predictions_shown":   predictions_shown,
        })
    logger.info("Eval-set export: %d rows (limit=%d)", len(rows), limit)

    import json as _json
    headers = {"X-Row-Count": str(len(rows))}
    return Response(
        content=_json.dumps(rows, separators=(",", ":")),
        media_type="application/json",
        headers=headers,
    )


# ── POST /feedback/snapshot ───────────────────────────────────────────────────

@feedback_router.post("/feedback/snapshot", response_model=FeedbackSnapshotResponse)
async def create_feedback_snapshot(
    window_days: int = 30,
    db: Session = Depends(get_local_db),
) -> FeedbackSnapshotResponse:
    """
    Freeze the current accuracy window as a weekly datapoint. Idempotent per
    date — running twice on the same day replaces the existing row.

    The FeedbackStatsScreen trend chart plots these.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    rows = (
        db.query(ScanFeedbackModel)
        .filter(ScanFeedbackModel.timestamp >= cutoff)
        .all()
    )
    top1, top3 = _compute_topn_accuracy(rows)
    by_src = _compute_by_source(rows)
    by_src_dict = {s.source: {"count": s.count, "correct": s.correct, "accuracy": s.accuracy} for s in by_src}

    # Upsert by date (one snapshot per day max)
    today = datetime.now(timezone.utc).date()
    existing = (
        db.query(FeedbackEvalSnapshotModel)
        .filter(func.date(FeedbackEvalSnapshotModel.snapshot_date) == today)
        .first()
    )
    if existing:
        existing.top1_accuracy = top1
        existing.top3_accuracy = top3
        existing.by_source_json = json.dumps(by_src_dict, separators=(",", ":"))
        existing.sample_size = len(rows)
        existing.window_days = window_days
        existing.snapshot_date = datetime.now(timezone.utc)
        record = existing
    else:
        record = FeedbackEvalSnapshotModel(
            top1_accuracy=top1,
            top3_accuracy=top3,
            by_source_json=json.dumps(by_src_dict, separators=(",", ":")),
            sample_size=len(rows),
            window_days=window_days,
        )
        db.add(record)
    db.commit()
    db.refresh(record)

    logger.info(
        "Snapshot created: date=%s  top1=%.2f  top3=%.2f  n=%d",
        record.snapshot_date.date(), top1, top3, len(rows),
    )
    return FeedbackSnapshotResponse(
        snapshot_date=record.snapshot_date.isoformat(),
        top1_accuracy=top1,
        top3_accuracy=top3,
        sample_size=len(rows),
        by_source=by_src_dict,
    )


# ── POST /scan-multi (YOLO-aware override) ────────────────────────────────────

@feedback_router.post("/scan-multi", response_model=MultiPieceScanResponse)
async def scan_multi_piece_yolo(
    request: MultiPieceScanRequest,
    db: Session = Depends(get_local_db),
) -> MultiPieceScanResponse:
    """
    Multi-piece scan with YOLO detection when available, quadrant-split fallback.

    This endpoint REPLACES the quadrant-only version in routes.py when this
    router is registered first in main.py.

    Detection strategy:
      1. Try YOLOv8 detector (brickscan/backend/models/yolo_lego.onnx)
         → precise bounding boxes per piece
      2. Fall back to 2×2 quadrant grid when YOLO model is absent
    """
    from app.local_inventory.image_processor import validate_and_decode_image

    logger.info("Multi-piece scan (YOLO-aware)")

    try:
        image_bytes = validate_and_decode_image(request.image_base64)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    from app.ml.model_manager import ModelManager
    mm = ModelManager.get()

    detected_pieces: List[DetectedPiece] = []
    detection_method = "quadrant"

    # ── Path 1: YOLO detector ─────────────────────────────────────────────────
    if mm.yolo_available:
        boxes = mm.detect_pieces(image_bytes)
        if boxes:
            detection_method = "yolo"
            logger.info("YOLO detected %d boxes", len(boxes))

            async def predict_box(box, idx: int):
                try:
                    preds = await hybrid_predict(box.crop_bytes)
                    return {"preds": preds, "box": box, "index": idx}
                except Exception as e:
                    logger.warning("Box %d prediction failed: %s", idx, e)
                    return {"preds": [], "box": box, "index": idx}

            results = await asyncio.gather(*[predict_box(b, i) for i, b in enumerate(boxes)])
            THRESHOLD = 0.30

            for result in results:
                preds = result["preds"]
                box   = result["box"]
                if not preds or preds[0].get("confidence", 0) < THRESHOLD:
                    continue
                piece_preds = [
                    ScanPrediction(
                        part_num=p.get("part_num", "unknown"),
                        part_name=p.get("part_name", ""),
                        confidence=p.get("confidence", 0.0),
                        color_id=p.get("color_id"),
                        color_name=p.get("color_name"),
                        color_hex=p.get("color_hex"),
                        source=p.get("source"),
                        image_url=p.get("image_url"),
                    )
                    for p in preds[:3]
                ]
                detected_pieces.append(DetectedPiece(
                    piece_index=len(detected_pieces),
                    predictions=piece_preds,
                    primary_prediction=piece_preds[0],
                    bbox=[box.x1, box.y1, box.x2, box.y2],
                ))

    # ── Path 2: 2×2 quadrant fallback ────────────────────────────────────────
    if not detected_pieces:
        detection_method = "quadrant"
        logger.info("Falling back to quadrant split")
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(image_bytes))
        w, h = img.size
        hw, hh = w // 2, h // 2
        regions = [
            (0, 0, hw, hh), (hw, 0, w, hh),
            (0, hh, hw, h), (hw, hh, w, h),
        ]
        quads = []
        for i, (x1, y1, x2, y2) in enumerate(regions):
            crop = img.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=85)
            quads.append({
                "bytes": buf.getvalue(),
                "bbox":  [x1 / w, y1 / h, x2 / w, y2 / h],
                "index": i,
            })

        async def predict_quad(q):
            try:
                preds = await hybrid_predict(q["bytes"])
                return {"preds": preds, "bbox": q["bbox"]}
            except Exception:
                return {"preds": [], "bbox": q["bbox"]}

        quad_results = await asyncio.gather(*[predict_quad(q) for q in quads])
        THRESHOLD = 0.30
        for result in quad_results:
            preds = result["preds"]
            if not preds or preds[0].get("confidence", 0) < THRESHOLD:
                continue
            piece_preds = [
                ScanPrediction(
                    part_num=p.get("part_num", "unknown"),
                    part_name=p.get("part_name", ""),
                    confidence=p.get("confidence", 0.0),
                    color_id=p.get("color_id"),
                    color_name=p.get("color_name"),
                    color_hex=p.get("color_hex"),
                    source=p.get("source"),
                    image_url=p.get("image_url"),
                )
                for p in preds[:3]
            ]
            detected_pieces.append(DetectedPiece(
                piece_index=len(detected_pieces),
                predictions=piece_preds,
                primary_prediction=piece_preds[0],
                bbox=result["bbox"],
            ))

    pieces_detected = len(detected_pieces)
    scan_status = "success" if pieces_detected > 0 else "partial"
    logger.info(
        "Multi-piece scan: %d pieces via %s", pieces_detected, detection_method
    )

    return MultiPieceScanResponse(
        pieces_detected=pieces_detected,
        pieces=detected_pieces,
        status=scan_status,
    )
