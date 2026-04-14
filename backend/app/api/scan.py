from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import base64
from typing import Optional, List
import logging
from io import BytesIO
from pathlib import Path
import csv
import subprocess
import json
from datetime import datetime, timezone

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.models.inventory import ScanLog
from app.models.part import Part, Color
from app.schemas.scan import ScanRequest, ScanResponse, ScanPrediction, ConfirmScanRequest, StudGrid
from app.services.ml_inference import predict as ml_predict
from app.services.gemini_service import identify_piece

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scan", tags=["scan"])

# Directory for storing scan thumbnails
SCAN_UPLOADS_DIR = Path("/tmp/brickscan_uploads/scans")


@router.post("", response_model=ScanResponse)
async def scan(
    request: ScanRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = current_user.get("sub")

    try:
        image_bytes = base64.b64decode(request.image_base64)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 image",
        )

    predictions = []
    confidence_score = 0.0
    predicted_part_num = None

    ml_results = await ml_predict(image_bytes)

    if ml_results and len(ml_results) > 0:
        top_result = ml_results[0]
        confidence_score = top_result.get("confidence", 0.0)

        if confidence_score >= settings.CONFIDENCE_THRESHOLD:
            predictions = ml_results[:3]
            predicted_part_num = ml_results[0].get("part_num")
        else:
            gemini_results = await identify_piece(image_bytes)
            predictions = gemini_results[:3]
            if gemini_results:
                predicted_part_num = gemini_results[0].get("part_num")
                confidence_score = gemini_results[0].get("confidence", 0.0)
    else:
        gemini_results = await identify_piece(image_bytes)
        predictions = gemini_results[:3]
        if gemini_results:
            predicted_part_num = gemini_results[0].get("part_num")
            confidence_score = gemini_results[0].get("confidence", 0.0)

    scan_log = ScanLog(
        user_id=user_id,
        predicted_part_num=predicted_part_num,
        confidence=confidence_score,
    )
    db.add(scan_log)
    await db.commit()
    await db.refresh(scan_log)

    # Save thumbnail image
    try:
        if Image and image_bytes:
            # Ensure directory exists
            scan_user_dir = SCAN_UPLOADS_DIR / str(user_id)
            scan_user_dir.mkdir(parents=True, exist_ok=True)

            # Load image and resize to max 400px
            image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
            image_pil.thumbnail((400, 400), Image.Resampling.LANCZOS)

            # Save as JPEG
            thumbnail_path = scan_user_dir / f"{str(scan_log.id)}.jpg"
            image_pil.save(thumbnail_path, format="JPEG", quality=85)
            logger.info(f"Saved thumbnail for scan {scan_log.id}")
    except Exception as e:
        logger.warning(f"Failed to save thumbnail for scan {scan_log.id}: {e}")

    response_predictions = []
    for pred in predictions:
        part_num = pred.get("part_num")
        part_name = pred.get("part_name", "Unknown")
        color_name = pred.get("color_name")
        color_hex = pred.get("color_hex")
        confidence = pred.get("confidence", 0.0)

        part_result = await db.execute(
            select(Part).where(Part.part_num == part_num)
        )
        part = part_result.scalars().first()

        if part:
            if not part_name or part_name == "Unknown":
                part_name = part.name

            if not color_name and color_hex:
                color_result = await db.execute(
                    select(Color).where(Color.hex_code == color_hex)
                )
                color = color_result.scalars().first()
                if color:
                    color_name = color.name

        response_predictions.append(
            ScanPrediction(
                part_num=part_num,
                part_name=part_name,
                color_name=color_name,
                color_hex=color_hex,
                confidence=confidence,
            )
        )

    # Attempt stud grid detection to constrain predictions
    stud_grid_response = None
    try:
        from ml.preprocessing.stud_detector import detect_stud_grid, constrain_predictions as constrain_by_grid
        from PIL import Image
        import io

        # Decode image for stud detection
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run stud detection
        grid_result = detect_stud_grid(image_pil)

        if grid_result and grid_result.confidence > 0.6:
            # Build a part dimensions lookup from the database
            part_dimensions = {}
            for pred in predictions:
                part_num = pred.get("part_num")
                part_result = await db.execute(
                    select(Part).where(Part.part_num == part_num)
                )
                part = part_result.scalars().first()
                if part:
                    # Assume part has width, length, height attributes (or similar)
                    # Fallback: try common attribute names
                    width = getattr(part, "width", None) or getattr(part, "stud_width", None)
                    length = getattr(part, "length", None) or getattr(part, "stud_length", None)
                    height = getattr(part, "height", None)

                    if width is not None and length is not None:
                        part_dimensions[part_num] = (width, length, height)

            # Re-score predictions based on grid
            if part_dimensions:
                predictions_dict = [
                    {
                        "part_num": p.part_num,
                        "confidence": p.confidence,
                    }
                    for p in response_predictions
                ]
                constrained = constrain_by_grid(
                    predictions_dict, grid_result, part_dimensions, tolerance=1
                )

                # Update response predictions with new confidences
                for i, constrained_pred in enumerate(constrained):
                    if i < len(response_predictions):
                        response_predictions[i].confidence = constrained_pred.get(
                            "confidence", response_predictions[i].confidence
                        )

            # Add grid info to response
            stud_grid_response = StudGrid(
                cols=grid_result.cols,
                rows=grid_result.rows,
                confidence=grid_result.confidence,
            )

            logger.info(
                "Stud grid detected: %dx%d (confidence=%.2f)",
                grid_result.cols, grid_result.rows, grid_result.confidence,
            )

    except ImportError:
        logger.debug("Stud detection module not available (opencv or dependencies missing)")
    except Exception as e:
        logger.warning("Stud detection failed (non-fatal): %s", e)

    # Generate thumbnail URL
    thumbnail_url = f"/api/scans/{str(scan_log.id)}/thumbnail"

    return ScanResponse(
        predictions=response_predictions,
        stud_grid=stud_grid_response,
        scan_id=str(scan_log.id),
        thumbnail_url=thumbnail_url,
    )


@router.post("/depth", response_model=ScanResponse)
async def scan_with_depth(
    file: UploadFile = File(...),
    depth_file: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ScanResponse:
    """
    Scan a LEGO piece from a multipart upload that may include LiDAR depth data.

    Unlike POST /api/scan (base64 JSON), this endpoint accepts multipart file
    uploads so the iOS DepthCaptureModule can send the raw depth map alongside
    the RGB frame without base64 bloat.

    Response shape matches POST /api/scan so the mobile client needs no branching.
    Depth bytes are currently read-but-not-used — reserved for future 3D matching
    via the DGX Spark vision server.
    """
    if Image is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image processing not available",
        )

    # Read + validate RGB image
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file",
            )
        Image.open(BytesIO(image_bytes)).convert("RGB")  # Validate decodability
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Failed to load image: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file",
        )

    # Depth is optional — read if present but don't fail if it's malformed.
    if depth_file is not None:
        try:
            _ = await depth_file.read()
        except Exception as e:
            logger.debug("Failed to read depth file (non-fatal): %s", e)

    # Run the same ML pipeline as the base64 endpoint.
    try:
        predictions = await ml_predict(image_bytes) or []
    except Exception as e:
        logger.error("ML inference failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detection service unavailable",
        )

    return ScanResponse(
        predictions=[
            ScanPrediction(
                part_num=p.get("part_num", "unknown"),
                part_name=p.get("part_name", ""),
                color_name=p.get("color_name"),
                color_hex=p.get("color_hex"),
                confidence=p.get("confidence", 0.0),
            )
            for p in predictions
        ],
    )


@router.get("/{scan_id}/thumbnail")
async def get_scan_thumbnail(
    scan_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get thumbnail image for a scan.
    Returns the stored JPEG thumbnail or 404 if not found.
    """
    user_id = current_user.get("sub")

    # Verify scan belongs to current user
    result = await db.execute(
        select(ScanLog).where(
            (ScanLog.id == scan_id) & (ScanLog.user_id == user_id)
        )
    )
    scan_log = result.scalars().first()

    if not scan_log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan not found",
        )

    # Check if thumbnail file exists
    thumbnail_path = SCAN_UPLOADS_DIR / str(user_id) / f"{scan_id}.jpg"

    if not thumbnail_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thumbnail not found",
        )

    return FileResponse(
        path=thumbnail_path,
        media_type="image/jpeg",
        filename=f"scan_{scan_id}.jpg",
    )


@router.post("/confirm")
async def confirm_scan(
    request: ConfirmScanRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = current_user.get("sub")

    result = await db.execute(
        select(ScanLog).where(
            (ScanLog.id == request.scan_log_id) & (ScanLog.user_id == user_id)
        )
    )
    scan_log = result.scalars().first()

    if not scan_log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan log not found",
        )

    scan_log.confirmed_part_num = request.confirmed_part_num
    await db.commit()
    await db.refresh(scan_log)

    return {
        "id": str(scan_log.id),
        "confirmed_part_num": scan_log.confirmed_part_num,
    }


@router.post("/pile")
async def scan_pile(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Scan a pile of loose LEGO bricks in a single image.
    Detects all bricks, classifies them, and returns aggregated results.
    """
    if not Image:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image processing not available",
        )

    user_id = current_user.get("sub")

    try:
        # Read uploaded image file
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file",
            )

        # Load image into PIL
        image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.warning("Failed to load image: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file",
        )

    # Run YOLO detection on full image
    try:
        detections = await ml_predict(image_bytes)
    except Exception as e:
        logger.error("YOLO detection failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detection service unavailable",
        )

    if not detections:
        # No bricks detected — return empty array (not error)
        return []

    # Aggregate results by part_num
    # Key: part_num, Value: {count, confidences, best_crop}
    aggregated: dict = {}

    for det in detections:
        part_num = det.get("part_num")
        confidence = det.get("confidence", 0.0)

        # Filter low-confidence detections
        if not part_num or confidence < 0.4:
            continue

        if part_num not in aggregated:
            aggregated[part_num] = {
                "part_name": det.get("part_name", part_num),
                "color_id": det.get("color_id"),
                "color_name": det.get("color_name"),
                "count": 0,
                "confidences": [],
                "best_crop": None,
                "best_confidence": 0.0,
            }

        aggregated[part_num]["count"] += 1
        aggregated[part_num]["confidences"].append(confidence)

        # Track the crop with highest confidence for thumbnail
        if confidence > aggregated[part_num]["best_confidence"]:
            aggregated[part_num]["best_confidence"] = confidence
            # If detection has bbox, crop it; otherwise use full image as fallback
            bbox = det.get("bbox")
            if bbox:
                try:
                    x, y, w, h = bbox.get("x"), bbox.get("y"), bbox.get("w"), bbox.get("h")
                    if x is not None and y is not None and w is not None and h is not None:
                        # Add 10px padding
                        x1 = max(0, x - 10)
                        y1 = max(0, y - 10)
                        x2 = min(image_pil.width, x + w + 10)
                        y2 = min(image_pil.height, y + h + 10)

                        crop = image_pil.crop((x1, y1, x2, y2))
                        # Resize to thumbnail size (64x64)
                        crop.thumbnail((64, 64), Image.Resampling.LANCZOS)

                        # Encode as base64
                        from io import BytesIO
                        buf = BytesIO()
                        crop.save(buf, format="JPEG", quality=70)
                        aggregated[part_num]["best_crop"] = base64.b64encode(buf.getvalue()).decode("utf-8")
                except Exception as e:
                    logger.debug("Failed to crop thumbnail: %s", e)

    # Build response list, sorted by count descending
    result_list = []
    for part_num, data in aggregated.items():
        # Average confidence across detections
        avg_confidence = sum(data["confidences"]) / len(data["confidences"]) if data["confidences"] else 0.0

        result_list.append({
            "part_num": part_num,
            "part_name": data["part_name"],
            "count": data["count"],
            "confidence": avg_confidence,
            "color_id": data["color_id"],
            "color_name": data["color_name"],
            "crop_image_base64": data["best_crop"],
        })

    # Sort by count descending
    result_list.sort(key=lambda x: x["count"], reverse=True)

    logger.info("Pile scan: detected %d unique parts, %d total pieces", len(result_list), sum(r["count"] for r in result_list))

    return result_list


@router.post("/multiview", response_model=ScanResponse)
async def scan_multiview(
    files: List[UploadFile] = [],
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Multi-view scan using attention pooling.

    Accepts up to 16 frames and uses the MultiViewPooling model to extract
    features from each frame and pool them via self-attention. Returns a single
    classification result with higher confidence than single-frame.

    Args:
        files: List of image files (up to 16)

    Returns:
        ScanResponse with predictions from multi-view pooling
    """
    user_id = current_user.get("sub")

    if not files or len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image files provided. Send 2-16 frames.",
        )

    if len(files) > 16:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many frames. Maximum 16 frames allowed.",
        )

    if len(files) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 2 frames for multi-view pooling.",
        )

    try:
        # Load and decode frames
        frames_bytes = []
        for file in files:
            content = await file.read()
            frames_bytes.append(content)

        # Convert bytes to PIL Images
        frames = []
        for frame_bytes in frames_bytes:
            try:
                frame_img = Image.open(BytesIO(frame_bytes)).convert('RGB')
                frames.append(frame_img)
            except Exception as e:
                logger.warning(f"Failed to decode frame: {e}")
                continue

        if len(frames) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not decode frames. Ensure images are valid JPEG/PNG.",
            )

        # Run multi-view pooling inference
        predictions = []
        confidence_score = 0.0
        predicted_part_num = None

        try:
            from ml.multi_view_pooling import MultiViewPooling, MultiViewPoolingInference
            import torch

            # Load model checkpoint (placeholder path — configure as needed)
            model_path = Path(settings.MODEL_DIR) / "multiview_final.pt"
            if not model_path.exists():
                logger.warning(f"MultiView model not found at {model_path}, falling back to single-frame")
                # Fallback to first frame only
                ml_results = await ml_predict(frames_bytes[0])
                predictions = ml_results[:3]
                if ml_results:
                    predicted_part_num = ml_results[0].get("part_num")
                    confidence_score = ml_results[0].get("confidence", 0.0)
            else:
                # Load and run multi-view model
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = MultiViewPooling(
                    num_parts=6500,  # Adjust based on your catalog size
                    num_colors=150,
                )
                model.load_state_dict(torch.load(model_path, map_location=device))
                inference = MultiViewPoolingInference(model, device=device)

                result = inference.predict(frames, top_k=3)

                # Convert model outputs to prediction format
                part_indices = result['top_part_indices']
                part_scores = result['top_part_scores']

                # Map indices back to part numbers (requires part index mapping)
                # Placeholder: would need to load part_num -> index mapping from DB
                for i, (idx, score) in enumerate(zip(part_indices, part_scores)):
                    # This requires a reverse mapping from part index to part_num
                    # Stub implementation: in production, load this mapping
                    part_num = str(idx)  # Placeholder
                    predictions.append({
                        "part_num": part_num,
                        "part_name": f"Part {part_num}",
                        "confidence": float(score),
                        "color_name": None,
                        "color_hex": None,
                    })

                if predictions:
                    predicted_part_num = predictions[0].get("part_num")
                    confidence_score = predictions[0].get("confidence", 0.0)

        except ImportError:
            logger.warning("MultiViewPooling not available, falling back to single-frame inference")
            ml_results = await ml_predict(frames_bytes[0])
            predictions = ml_results[:3]
            if ml_results:
                predicted_part_num = ml_results[0].get("part_num")
                confidence_score = ml_results[0].get("confidence", 0.0)

        # Log scan
        scan_log = ScanLog(
            user_id=user_id,
            predicted_part_num=predicted_part_num,
            confidence=confidence_score,
        )
        db.add(scan_log)
        await db.commit()
        await db.refresh(scan_log)

        # Enrich predictions with database info
        response_predictions = []
        for pred in predictions:
            part_num = pred.get("part_num")
            part_name = pred.get("part_name", "Unknown")
            color_name = pred.get("color_name")
            color_hex = pred.get("color_hex")
            confidence = pred.get("confidence", 0.0)

            part_result = await db.execute(
                select(Part).where(Part.part_num == part_num)
            )
            part = part_result.scalars().first()

            if part:
                if not part_name or part_name == "Unknown":
                    part_name = part.name

            response_predictions.append(
                ScanPrediction(
                    part_num=part_num,
                    part_name=part_name,
                    color_name=color_name,
                    color_hex=color_hex,
                    confidence=confidence,
                )
            )

        return ScanResponse(predictions=response_predictions)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MultiView scan failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MultiView scan failed: {str(e)}",
        )


# ==============================================================================
# ADMIN ENDPOINTS
# ==============================================================================

@router.post("/admin/trigger-retrain")
async def trigger_retrain(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Admin-only endpoint to trigger model retraining on user feedback corrections.

    Exports all user corrections to CSV and optionally triggers a fine-tuning run
    on the ML server. Admin role required.

    Returns:
        {
            "status": "triggered" | "error",
            "feedback_count": N,
            "csv_path": "...",
            "retrain_job_id": "..." (if SSH triggered)
        }
    """
    user_id = current_user.get("sub")
    user_role = current_user.get("role", "user")

    # Check admin role
    if user_role != "admin":
        logger.warning(f"Non-admin user {user_id} attempted to trigger retrain")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )

    logger.info(f"Admin {user_id} triggered retrain")

    # Query all corrections from feedback_corrections table (if it exists)
    # For now, assume table structure based on scan.py
    # In production, modify to query your actual feedback model

    try:
        # Export feedback corrections to CSV
        feedback_csv = await _export_feedback_to_csv(db)

        response = {
            "status": "triggered",
            "feedback_count": feedback_csv.get("count", 0),
            "csv_path": str(feedback_csv.get("path", "")),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Optionally trigger SSH to ML server
        retrain_result = await _trigger_ssh_retrain(feedback_csv.get("path"))
        if retrain_result:
            response["retrain_job_id"] = retrain_result.get("job_id")
            response["ml_server"] = retrain_result.get("server")

        logger.info(f"Retrain triggered: {response}")
        return response

    except Exception as e:
        logger.error(f"Failed to trigger retrain: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


async def _export_feedback_to_csv(db: AsyncSession) -> dict:
    """
    Export user feedback corrections to CSV file.

    Assumes a FeedbackCorrection model exists with:
    - image_path
    - correct_part_num
    - correct_color_id
    - original_prediction
    - timestamp
    - user_id

    Returns:
        Dict with 'path' (str) and 'count' (int)
    """
    try:
        # Try to import feedback model
        try:
            from app.models.feedback import FeedbackCorrection
        except ImportError:
            logger.warning("FeedbackCorrection model not found, creating dummy export")
            return {"path": None, "count": 0}

        # Query all corrections
        result = await db.execute(select(FeedbackCorrection))
        corrections = result.scalars().all()

        if not corrections:
            logger.info("No feedback corrections to export")
            return {"path": None, "count": 0}

        # Write to CSV
        export_dir = Path(settings.DATA_DIR or "/tmp") / "feedback_exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        csv_path = export_dir / f"feedback_corrections_{timestamp}.csv"

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'image_path', 'correct_part_num', 'correct_color_id',
                'original_prediction', 'timestamp', 'user_id'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for correction in corrections:
                writer.writerow({
                    'image_path': getattr(correction, 'image_path', ''),
                    'correct_part_num': getattr(correction, 'correct_part_num', ''),
                    'correct_color_id': getattr(correction, 'correct_color_id', ''),
                    'original_prediction': getattr(correction, 'original_prediction', ''),
                    'timestamp': str(getattr(correction, 'timestamp', '')),
                    'user_id': getattr(correction, 'user_id', ''),
                })

        logger.info(f"Exported {len(corrections)} corrections to {csv_path}")
        return {
            "path": str(csv_path),
            "count": len(corrections),
        }

    except Exception as e:
        logger.error(f"Failed to export feedback CSV: {e}")
        raise


async def _trigger_ssh_retrain(csv_path: Optional[str]) -> Optional[dict]:
    """
    Optionally SSH to ML server and trigger fine-tuning job.

    Requires configuration in settings:
    - ML_SERVER_HOST: hostname or IP
    - ML_SERVER_USER: SSH username
    - ML_SERVER_KEY: path to SSH private key
    - ML_RETRAIN_SCRIPT: path to retrain_from_feedback.py on server

    Args:
        csv_path: Path to exported feedback CSV

    Returns:
        Dict with job info if successful, None otherwise
    """
    # Check if SSH is configured
    if not all([
        getattr(settings, 'ML_SERVER_HOST', None),
        getattr(settings, 'ML_SERVER_USER', None),
        getattr(settings, 'ML_RETRAIN_SCRIPT', None),
    ]):
        logger.info("ML server SSH not configured, skipping remote trigger")
        return None

    try:
        ml_host = settings.ML_SERVER_HOST
        ml_user = settings.ML_SERVER_USER
        ml_key = getattr(settings, 'ML_SERVER_KEY', None)
        retrain_script = settings.ML_RETRAIN_SCRIPT

        # Build SSH command
        ssh_cmd = [
            'ssh',
            f'{ml_user}@{ml_host}',
        ]

        if ml_key:
            ssh_cmd.extend(['-i', ml_key])

        # Remote Python command
        remote_cmd = (
            f"python3 {retrain_script} "
            f"--feedback-csv {csv_path} "
            f"--checkpoint ./models/best.pt "
            f"--output-dir ./models"
        )

        ssh_cmd.append(remote_cmd)

        # Trigger async (don't wait for completion)
        job_id = f"retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Triggering SSH retrain on {ml_host}: {remote_cmd}")

        # Use subprocess.Popen to run async
        subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return {
            "job_id": job_id,
            "server": ml_host,
            "status": "queued",
        }

    except Exception as e:
        logger.warning(f"Failed to trigger SSH retrain: {e}")
        return None
