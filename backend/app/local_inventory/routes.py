"""
FastAPI routes for local inventory scanning and management.

Endpoints:
- POST /api/scan: Scan a brick image, return prediction + confidence
- POST /api/inventory/add: Add confirmed part to inventory
- GET /api/inventory: List all local inventory parts
- PUT /api/inventory/{id}: Update part quantity or correct prediction
- DELETE /api/inventory/{id}: Remove part from inventory
- GET /api/inventory/export: Export as CSV
- POST /api/scan-session/start: Start a named scanning session
- POST /api/scan-session/{id}/complete: Mark session complete

Design principles:
- Confidence threshold 75%: >75% = "known" (auto-add), <75% = "uncertain" (user picks)
- All predictions include top-3 candidates for uncertain cases
- Images saved locally for retraining on low-confidence predictions
- No authentication required (local device, offline-first)
"""

import asyncio
import logging
import csv
import io
from collections import Counter
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse, Response
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.local_inventory.database import get_local_db, init_db
from app.local_inventory.models import LocalInventoryPart, ScanSession
from app.local_inventory.schemas import (
    ScanRequest,
    ScanResponse,
    ScanPrediction,
    LocalInventoryPartSchema,
    UpdateInventoryQuantityRequest,
    CorrectPredictionRequest,
    ScanSessionRequest,
    ScanSessionSchema,
    ConfirmPredictionRequest,
    VideoScanRequest,
    VideoScanResponse,
    MultiPieceScanRequest,
    MultiPieceScanResponse,
    DetectedPiece,
)
from app.local_inventory.image_processor import (
    validate_and_decode_image,
    save_scan_image,
)
from app.services.hybrid_recognition import hybrid_predict
from app.services.ml_inference import predict as ml_predict
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/local-inventory",
    tags=["local-inventory"],
)

# Initialize database on startup
try:
    init_db()
except Exception as e:
    logger.warning(f"Local inventory database initialization: {e}")


@router.post("/scan", response_model=ScanResponse)
async def scan_brick(
    request: ScanRequest,
    db: Session = Depends(get_local_db),
) -> ScanResponse:
    """
    Scan a LEGO brick image and predict part + color.

    Process:
    1. Validate and decode base64 image
    2. Preprocess: resize to 224x224, normalize
    3. Run ONNX model inference
    4. Return top-3 predictions with confidence
    5. Save image if confidence < 80% (uncertain)

    Response:
    - status: "known" (>80%) or "uncertain" (<80%)
    - predictions: Top-3 candidate parts
    - primary_prediction: Best match
    - save_image: Whether image was saved

    Returns:
        ScanResponse with predictions or HTTPException if image invalid
    """
    logger.info("Processing scan request")

    # Validate and decode image
    try:
        image_bytes = validate_and_decode_image(request.image_base64)
    except ValueError as e:
        logger.error(f"Image validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image: {str(e)}",
        )

    # Run hybrid recognition pipeline (Brickognize → Gemini → ONNX)
    try:
        predictions = await hybrid_predict(image_bytes)
    except Exception as e:
        logger.error(f"Hybrid recognition failed: {e}")
        # Fallback to ONNX-only if hybrid pipeline crashes
        try:
            logger.info("Falling back to ONNX-only inference")
            predictions = await ml_predict(image_bytes)
        except Exception as e2:
            logger.error(f"ONNX fallback also failed: {e2}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="All recognition methods failed",
            )

    if not predictions:
        logger.warning("No predictions from any recognition method")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No recognition methods returned predictions",
        )

    # Extract top prediction
    top = predictions[0]
    confidence = top.get("confidence", 0.0)
    part_num = top.get("part_num", "unknown")

    # Determine status: "known" if > 80%, else "uncertain"
    confidence_threshold = getattr(settings, "CONFIDENCE_THRESHOLD", 0.80)
    status_result = "known" if confidence >= confidence_threshold else "uncertain"

    logger.info(
        f"Prediction: {part_num} ({confidence:.1%}) - {status_result}"
    )

    # Save image if uncertain (for retraining)
    save_image = status_result == "uncertain"
    image_path = None
    if save_image:
        try:
            image_path = save_scan_image(image_bytes, part_num, confidence)
            logger.info(f"Saved uncertain scan image: {image_path}")
        except IOError as e:
            logger.warning(f"Failed to save image: {e}")
            # Don't fail the scan, just skip image save

    # Format predictions for response
    response_predictions = [
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
        for p in predictions[:5]  # Up to 5 from hybrid pipeline
    ]

    primary = response_predictions[0]

    return ScanResponse(
        status=status_result,
        predictions=response_predictions,
        primary_prediction=primary,
        save_image=save_image and image_path is not None,
    )


@router.post("/scan-video", response_model=VideoScanResponse)
async def scan_video(
    request: VideoScanRequest,
    db: Session = Depends(get_local_db),
) -> VideoScanResponse:
    """
    Scan a LEGO brick from multiple angles (video mode).

    Accepts 2-10 frames captured at different angles. Runs the hybrid
    recognition pipeline on each frame, then aggregates predictions using
    weighted voting. This solves ambiguity cases like brick-vs-plate where
    a single top-down photo can't distinguish height.

    Process:
    1. Decode and validate all frames
    2. Run hybrid_predict on each frame concurrently
    3. Aggregate: weighted vote by confidence across all frames
    4. Return consensus predictions with an agreement score

    Returns:
        VideoScanResponse with aggregated predictions
    """
    logger.info(f"Video scan: {len(request.frames)} frames received")

    # Decode all frames
    decoded_frames = []
    for i, frame_b64 in enumerate(request.frames):
        try:
            image_bytes = validate_and_decode_image(frame_b64)
            decoded_frames.append(image_bytes)
        except ValueError as e:
            logger.warning(f"Frame {i} invalid: {e}")
            continue

    if len(decoded_frames) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 2 valid frames for video scan",
        )

    # Run hybrid predict on all frames concurrently
    async def predict_frame(frame_bytes: bytes, idx: int):
        try:
            preds = await hybrid_predict(frame_bytes)
            return preds
        except Exception as e:
            logger.warning(f"Frame {idx} prediction failed: {e}")
            return []

    tasks = [predict_frame(fb, i) for i, fb in enumerate(decoded_frames)]
    all_frame_predictions = await asyncio.gather(*tasks)

    # Filter out empty results
    valid_frames = [fp for fp in all_frame_predictions if fp]
    frames_analyzed = len(valid_frames)

    if frames_analyzed == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="All frames failed recognition",
        )

    # Aggregate predictions using weighted voting
    # Key: part_num -> accumulated confidence, count, best prediction data
    vote_map: Dict[str, Dict[str, Any]] = {}

    for frame_preds in valid_frames:
        for pred in frame_preds[:3]:  # Top 3 from each frame
            pnum = pred.get("part_num", "unknown")
            conf = pred.get("confidence", 0.0)

            if pnum not in vote_map:
                vote_map[pnum] = {
                    "total_confidence": 0.0,
                    "frame_count": 0,
                    "best_confidence": 0.0,
                    "best_pred": pred,
                }

            vote_map[pnum]["total_confidence"] += conf
            vote_map[pnum]["frame_count"] += 1
            if conf > vote_map[pnum]["best_confidence"]:
                vote_map[pnum]["best_confidence"] = conf
                vote_map[pnum]["best_pred"] = pred

    # Score = (avg_confidence * 0.6) + (frame_fraction * 0.4)
    # This rewards parts that appear consistently AND with high confidence
    scored = []
    for pnum, data in vote_map.items():
        avg_conf = data["total_confidence"] / data["frame_count"]
        frame_fraction = data["frame_count"] / frames_analyzed
        score = (avg_conf * 0.6) + (frame_fraction * 0.4)

        pred = data["best_pred"].copy()
        pred["confidence"] = min(score, 1.0)
        pred["source"] = (pred.get("source", "unknown") or "unknown") + "+video"
        scored.append(pred)

    # Sort by consensus score
    scored.sort(key=lambda x: x["confidence"], reverse=True)

    # Calculate agreement: what fraction of frames agreed on the top prediction
    top_part = scored[0]["part_num"] if scored else "unknown"
    agreeing_frames = sum(
        1 for fp in valid_frames
        if fp and fp[0].get("part_num") == top_part
    )
    agreement_score = agreeing_frames / frames_analyzed if frames_analyzed > 0 else 0.0

    # Determine status
    top_conf = scored[0]["confidence"] if scored else 0.0
    confidence_threshold = getattr(settings, "CONFIDENCE_THRESHOLD", 0.80)
    status_result = "known" if (top_conf >= confidence_threshold and agreement_score >= 0.5) else "uncertain"

    logger.info(
        f"Video scan result: {top_part} ({top_conf:.1%}), "
        f"agreement={agreement_score:.1%}, frames={frames_analyzed}"
    )

    # Format response
    response_predictions = [
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
        for p in scored[:5]
    ]

    primary = response_predictions[0]

    return VideoScanResponse(
        status=status_result,
        predictions=response_predictions,
        primary_prediction=primary,
        frames_analyzed=frames_analyzed,
        agreement_score=round(agreement_score, 3),
        save_image=status_result == "uncertain",
    )


@router.post("/scan-multi", response_model=MultiPieceScanResponse)
async def scan_multi_piece(
    request: MultiPieceScanRequest,
    db: Session = Depends(get_local_db),
) -> MultiPieceScanResponse:
    """
    Scan multiple LEGO pieces in a single image.

    Currently uses a grid-based approach: splits the image into quadrants
    and runs recognition on each. When YOLOv8 detector is trained, this
    endpoint will use proper object detection to locate individual pieces.

    Process:
    1. Decode and validate image
    2. Split into grid regions (2x2 = 4 quadrants)
    3. Run hybrid recognition on each region
    4. Filter out low-confidence results (likely empty quadrants)
    5. Return per-piece predictions

    Returns:
        MultiPieceScanResponse with detected pieces
    """
    logger.info("Multi-piece scan request received")

    try:
        image_bytes = validate_and_decode_image(request.image_base64)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image: {str(e)}",
        )

    from PIL import Image as PILImage
    import io as _io

    MULTI_THRESHOLD = 0.30

    # ── Try YOLO detector first (available once Spark training completes) ─────
    regions_to_scan: list[dict] = []
    used_yolo = False

    try:
        from app.ml.model_manager import ModelManager
        mm = ModelManager.get()
        if mm.yolo_available:
            boxes = mm.detect_pieces(image_bytes)
            if boxes:
                logger.info("YOLO detected %d pieces", len(boxes))
                for i, box in enumerate(boxes):
                    regions_to_scan.append({
                        "bytes": box.crop_bytes,
                        "bbox": [box.x1, box.y1, box.x2, box.y2],
                        "index": i,
                    })
                used_yolo = True
    except Exception as e:
        logger.warning("YOLO detection failed, falling back to grid: %s", e)

    # ── Fallback: 2×2 grid splitting when YOLO is not yet trained ─────────────
    if not used_yolo:
        img = PILImage.open(_io.BytesIO(image_bytes))
        w, h = img.size
        half_w, half_h = w // 2, h // 2
        grid_regions = [
            (0, 0, half_w, half_h),
            (half_w, 0, w, half_h),
            (0, half_h, half_w, h),
            (half_w, half_h, w, h),
        ]
        for i, (x1, y1, x2, y2) in enumerate(grid_regions):
            crop = img.crop((x1, y1, x2, y2))
            buf = _io.BytesIO()
            crop.save(buf, format="JPEG", quality=85)
            regions_to_scan.append({
                "bytes": buf.getvalue(),
                "bbox": [x1 / w, y1 / h, x2 / w, y2 / h],
                "index": i,
            })
        logger.info("Grid fallback: %d regions", len(regions_to_scan))

    # ── Run hybrid recognition on each region concurrently ────────────────────
    async def predict_region(region: dict):
        try:
            preds = await hybrid_predict(region["bytes"])
            return {"preds": preds, "bbox": region["bbox"], "index": region["index"]}
        except Exception as e:
            logger.warning(f"Region {region['index']} failed: {e}")
            return {"preds": [], "bbox": region["bbox"], "index": region["index"]}

    results = await asyncio.gather(*[predict_region(r) for r in regions_to_scan])

    detected_pieces = []
    for result in results:
        preds = result["preds"]
        if not preds:
            continue
        # For YOLO crops we trust the detector confidence; for grid, filter empties
        top_conf = preds[0].get("confidence", 0.0)
        if not used_yolo and top_conf < MULTI_THRESHOLD:
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
    detection_method = "yolo" if used_yolo else "grid"
    status_result = "success" if pieces_detected > 0 else "partial"

    logger.info(
        "Multi-piece scan (%s): %d pieces from %d regions",
        detection_method, pieces_detected, len(regions_to_scan),
    )

    return MultiPieceScanResponse(
        pieces_detected=pieces_detected,
        pieces=detected_pieces,
        status=status_result,
    )


@router.post("/inventory/add", response_model=LocalInventoryPartSchema)
async def add_to_inventory(
    request: ConfirmPredictionRequest,
    db: Session = Depends(get_local_db),
) -> LocalInventoryPartSchema:
    """
    Add a confirmed prediction to local inventory.

    If part+color already exists, increment quantity. Otherwise, create new entry.

    Args:
        part_num: LEGO part number to add
        color_id: Rebrickable color ID
        color_name: Human-readable color name
        quantity: Number of pieces

    Returns:
        LocalInventoryPartSchema with updated part details
    """
    logger.info(f"Adding to inventory: {request.part_num} qty={request.quantity}")

    # Check if part already in inventory
    existing = db.query(LocalInventoryPart).filter(
        (LocalInventoryPart.part_num == request.part_num)
        & (LocalInventoryPart.color_id == request.color_id)
    ).first()

    if existing:
        # Increment quantity
        existing.quantity += request.quantity
        existing.user_confirmed = True
        db.commit()
        db.refresh(existing)
        logger.info(f"Updated inventory: {request.part_num} -> qty={existing.quantity}")
        item = existing
    else:
        # Create new entry
        item = LocalInventoryPart(
            part_num=request.part_num,
            color_id=request.color_id,
            color_name=request.color_name,
            color_hex=request.color_hex,
            quantity=request.quantity,
            confidence=1.0,  # User-confirmed = 100% confidence
            user_confirmed=True,
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        logger.info(f"Created inventory item: {request.part_num} qty={request.quantity}")

    return LocalInventoryPartSchema.model_validate(item)


@router.get("/inventory", response_model=List[LocalInventoryPartSchema])
async def get_inventory(db: Session = Depends(get_local_db)) -> List[LocalInventoryPartSchema]:
    """
    List all parts in local inventory.

    Returns:
        List of LocalInventoryPartSchema, ordered by part_num
    """
    logger.info("Fetching inventory")
    items = (
        db.query(LocalInventoryPart)
        .order_by(LocalInventoryPart.part_num)
        .all()
    )

    return [LocalInventoryPartSchema.model_validate(item) for item in items]


@router.get("/inventory/stats")
async def get_inventory_stats(db: Session = Depends(get_local_db)):
    """
    Get aggregate statistics about local inventory.

    Returns:
        - total_parts: Number of unique parts
        - total_quantity: Sum of all quantities
        - user_confirmed: Count of confirmed parts
        - uncertain_parts: Count of unconfirmed predictions
    """
    total_parts = db.query(func.count(LocalInventoryPart.id)).scalar() or 0
    total_qty = db.query(func.sum(LocalInventoryPart.quantity)).scalar() or 0
    confirmed = (
        db.query(func.count(LocalInventoryPart.id))
        .filter(LocalInventoryPart.user_confirmed == True)
        .scalar()
        or 0
    )
    uncertain = total_parts - confirmed

    return {
        "total_parts": total_parts,
        "total_quantity": total_qty,
        "user_confirmed": confirmed,
        "uncertain_parts": uncertain,
    }


@router.put("/inventory/{item_id}", response_model=LocalInventoryPartSchema)
async def update_inventory(
    item_id: str,
    request: UpdateInventoryQuantityRequest,
    db: Session = Depends(get_local_db),
) -> LocalInventoryPartSchema:
    """
    Update quantity of an inventory part.

    If quantity is 0, the part is deleted.

    Args:
        item_id: LocalInventoryPart UUID
        quantity: New quantity (0 to remove)

    Returns:
        Updated LocalInventoryPartSchema or 404 if not found
    """
    item = db.query(LocalInventoryPart).filter(LocalInventoryPart.id == item_id).first()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Inventory part not found",
        )

    if request.quantity == 0:
        db.delete(item)
        db.commit()
        logger.info(f"Deleted inventory item: {item.part_num}")
        return Response(status_code=204)

    item.quantity = request.quantity
    db.commit()
    db.refresh(item)
    logger.info(f"Updated quantity: {item.part_num} -> {request.quantity}")

    return LocalInventoryPartSchema.model_validate(item)


@router.post("/inventory/{item_id}/correct", response_model=LocalInventoryPartSchema)
async def correct_part_prediction(
    item_id: str,
    request: CorrectPredictionRequest,
    db: Session = Depends(get_local_db),
) -> LocalInventoryPartSchema:
    """
    Correct a mispredicted part in inventory.

    Used when the model or user realization reveals a wrong part number.
    Updates the part details and marks it as user-confirmed.

    Args:
        item_id: LocalInventoryPart ID to correct
        correct_part_num: The actual correct LEGO part number
        correct_color_id: Correct color ID
        correct_color_name: Correct color name

    Returns:
        Updated LocalInventoryPartSchema or 404
    """
    item = db.query(LocalInventoryPart).filter(LocalInventoryPart.id == item_id).first()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Inventory part not found",
        )

    old_part = item.part_num
    item.part_num = request.correct_part_num
    if request.correct_color_id is not None:
        item.color_id = request.correct_color_id
    if request.correct_color_name is not None:
        item.color_name = request.correct_color_name

    item.user_confirmed = True
    item.confidence = 1.0  # Manually corrected = perfect confidence

    db.commit()
    db.refresh(item)

    logger.info(f"Corrected part: {old_part} -> {request.correct_part_num}")

    return LocalInventoryPartSchema.model_validate(item)


@router.delete("/inventory/{item_id}")
async def delete_inventory_item(
    item_id: str,
    db: Session = Depends(get_local_db),
):
    """
    Delete a part from inventory.

    Args:
        item_id: LocalInventoryPart UUID

    Returns:
        {"message": "Item deleted"} or 404
    """
    item = db.query(LocalInventoryPart).filter(LocalInventoryPart.id == item_id).first()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Inventory part not found",
        )

    part_num = item.part_num
    db.delete(item)
    db.commit()

    logger.info(f"Deleted inventory item: {part_num}")

    return {"message": "Item deleted"}


@router.get("/inventory/export", response_class=PlainTextResponse)
async def export_inventory_csv(db: Session = Depends(get_local_db)) -> str:
    """
    Export local inventory as CSV.

    Columns: part_num, color_name, color_hex, quantity, confidence, user_confirmed, created_at

    Returns:
        CSV string (can be saved to file)
    """
    logger.info("Exporting inventory as CSV")

    items = db.query(LocalInventoryPart).order_by(LocalInventoryPart.part_num).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Part Number",
        "Color",
        "Hex",
        "Quantity",
        "Confidence",
        "User Confirmed",
        "Created",
    ])

    for item in items:
        writer.writerow([
            item.part_num,
            item.color_name or "",
            item.color_hex or "",
            item.quantity,
            f"{item.confidence:.3f}",
            "Yes" if item.user_confirmed else "No",
            item.created_at.isoformat(),
        ])

    return output.getvalue()


@router.post("/scan-session/start", response_model=ScanSessionSchema)
async def start_scan_session(
    request: ScanSessionRequest,
    db: Session = Depends(get_local_db),
) -> ScanSessionSchema:
    """
    Start a named scanning session.

    Groups multiple scans (e.g., "Technic 42145", "Bulk Sort Apr 2024").

    Args:
        set_name: Human-friendly name for this session

    Returns:
        ScanSessionSchema with generated UUID
    """
    session = ScanSession(set_name=request.set_name, completed=False)
    db.add(session)
    db.commit()
    db.refresh(session)

    logger.info(f"Started scan session: {request.set_name}")

    return ScanSessionSchema.model_validate(session)


@router.get("/scan-session", response_model=List[ScanSessionSchema])
async def list_scan_sessions(
    completed_only: bool = False,
    db: Session = Depends(get_local_db),
) -> List[ScanSessionSchema]:
    """
    List all scan sessions, optionally filtered to completed ones only.

    Args:
        completed_only: If True, return only completed sessions

    Returns:
        List of ScanSessionSchema
    """
    query = db.query(ScanSession)
    if completed_only:
        query = query.filter(ScanSession.completed == True)

    sessions = query.order_by(ScanSession.created_at.desc()).all()
    return [ScanSessionSchema.model_validate(s) for s in sessions]


@router.post("/scan-session/{session_id}/complete", response_model=ScanSessionSchema)
async def complete_scan_session(
    session_id: str,
    db: Session = Depends(get_local_db),
) -> ScanSessionSchema:
    """
    Mark a scan session as complete.

    Args:
        session_id: ScanSession UUID

    Returns:
        Updated ScanSessionSchema or 404
    """
    session = (
        db.query(ScanSession)
        .filter(ScanSession.id == session_id)
        .first()
    )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan session not found",
        )

    session.completed = True
    db.commit()
    db.refresh(session)

    logger.info(f"Completed scan session: {session.set_name}")

    return ScanSessionSchema.model_validate(session)


@router.get("/parts/info/{part_num}")
async def get_part_info(part_num: str):
    """
    Look up a LEGO part's name and category from the local Rebrickable CSV catalog.

    Falls back gracefully if the CSV hasn't been downloaded yet. Used by the
    mobile app to enrich scan results with proper part names.

    Args:
        part_num: LEGO / Rebrickable part number (e.g. "3001", "3039a")

    Returns:
        JSON with part_num, part_name, category_id, category_name
    """
    import csv as _csv
    import os as _os

    # Paths where the Rebrickable catalog may have been downloaded
    _search_paths = [
        _os.path.expanduser("~/brickscan/ml/training_data/rebrickable_csv/parts.csv"),
        "/app/data/parts.csv",
        "/data/parts.csv",
        _os.path.join(_os.path.dirname(__file__), "..", "..", "data", "parts.csv"),
    ]

    parts_csv = next((_p for _p in _search_paths if _os.path.exists(_p)), None)

    if not parts_csv:
        # No catalog available — return minimal info so the app still works
        return {
            "part_num": part_num,
            "part_name": f"Part {part_num}",
            "category_id": None,
            "category_name": None,
            "source": "fallback",
        }

    # Search the CSV (it's small enough to scan linearly at ~2MB)
    try:
        part_num_lower = part_num.lower().strip()
        with open(parts_csv, encoding="utf-8", errors="ignore") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                if row.get("part_num", "").lower().strip() == part_num_lower:
                    return {
                        "part_num": row["part_num"],
                        "part_name": row.get("name", f"Part {part_num}"),
                        "category_id": row.get("part_cat_id"),
                        "category_name": None,
                        "source": "rebrickable_csv",
                    }
    except Exception as e:
        logger.warning(f"Part lookup CSV error: {e}")

    return {
        "part_num": part_num,
        "part_name": f"Part {part_num}",
        "category_id": None,
        "category_name": None,
        "source": "not_found",
    }


@router.get("/parts/search")
async def search_parts(q: str, limit: int = 20):
    """
    Search parts by name or number from the local Rebrickable CSV catalog.

    Args:
        q: Search query (part number or name substring)
        limit: Max results to return (default 20)

    Returns:
        List of matching parts with part_num, part_name, category_id
    """
    import csv as _csv
    import os as _os

    _search_paths = [
        _os.path.expanduser("~/brickscan/ml/training_data/rebrickable_csv/parts.csv"),
        "/app/data/parts.csv",
        "/data/parts.csv",
        _os.path.join(_os.path.dirname(__file__), "..", "..", "data", "parts.csv"),
    ]

    parts_csv = next((_p for _p in _search_paths if _os.path.exists(_p)), None)

    if not parts_csv:
        return []

    q_lower = q.lower().strip()
    results = []

    try:
        with open(parts_csv, encoding="utf-8", errors="ignore") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                pnum = row.get("part_num", "").lower()
                pname = row.get("name", "").lower()
                if q_lower in pnum or q_lower in pname:
                    results.append({
                        "part_num": row["part_num"],
                        "part_name": row.get("name", row["part_num"]),
                        "category_id": row.get("part_cat_id"),
                    })
                    if len(results) >= limit:
                        break
    except Exception as e:
        logger.warning(f"Part search CSV error: {e}")

    return results


@router.delete("/scan-session/{session_id}")
async def delete_scan_session(
    session_id: str,
    db: Session = Depends(get_local_db),
):
    """
    Delete a scan session (marks all associated scans orphaned).

    Args:
        session_id: ScanSession UUID

    Returns:
        {"message": "Session deleted"} or 404
    """
    session = (
        db.query(ScanSession)
        .filter(ScanSession.id == session_id)
        .first()
    )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan session not found",
        )

    set_name = session.set_name
    db.delete(session)
    db.commit()

    logger.info(f"Deleted scan session: {set_name}")

    return {"message": "Session deleted"}
