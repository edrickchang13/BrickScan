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
import io
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.local_inventory.database import get_local_db
from app.local_inventory.models import ScanFeedback as ScanFeedbackModel
from app.local_inventory.schemas_feedback import (
    FeedbackConfusionPair,
    FeedbackStats,
    ScanFeedback,
    ScanFeedbackResponse,
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
      saves a labelled JPEG to data/feedback_images/<correct_part_num>/<scan_id>.jpg
    - Sets will_improve_model=True when a new labelled image was saved
    """
    is_correction = (
        request.correct_part_num.strip().lower()
        != request.predicted_part_num.strip().lower()
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
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    logger.info(
        "Feedback saved: scan_id=%s  %s→%s  correction=%s",
        request.scan_id,
        request.predicted_part_num,
        request.correct_part_num,
        is_correction,
    )

    return ScanFeedbackResponse(
        saved=True,
        will_improve_model=is_correction and image_path is not None,
        feedback_id=record.id,
    )


# ── GET /feedback/stats ───────────────────────────────────────────────────────

@feedback_router.get("/feedback/stats", response_model=FeedbackStats)
async def get_feedback_stats(db: Session = Depends(get_local_db)) -> FeedbackStats:
    """
    Return aggregate statistics for the active-learning feedback collection.
    """
    all_feedback = db.query(ScanFeedbackModel).all()

    corrections = [
        f for f in all_feedback
        if f.correct_part_num.strip().lower() != f.predicted_part_num.strip().lower()
    ]
    agreements = len(all_feedback) - len(corrections)

    # Top 10 confused pairs
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

    # Count saved images and parts covered
    images_saved = sum(1 for f in corrections if f.image_path is not None)
    parts_with_feedback = len({f.correct_part_num for f in corrections if f.image_path})

    pending_training = (
        db.query(func.count(ScanFeedbackModel.id))
        .filter(ScanFeedbackModel.used_for_training == False)  # noqa: E712
        .scalar()
        or 0
    )

    return FeedbackStats(
        total_corrections=len(corrections),
        agreement_count=agreements,
        top_confused_pairs=top_pairs,
        parts_with_feedback=parts_with_feedback,
        images_saved=images_saved,
        pending_training=int(pending_training),
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
