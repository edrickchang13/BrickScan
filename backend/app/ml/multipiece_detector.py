"""
OpenCV-based multi-piece brick detector — bootstrap until LEGO-trained YOLO weights arrive.

Strategy: brick-on-surface scans usually have high colour contrast against
the background (table, paper, mat). Convert to HSV, threshold out background
tones (white/grey/wood-brown), find external contours, filter by area, return
bounding boxes with crops.

When `backend/models/yolo_lego.onnx` exists, ModelManager.detect_pieces() will
return YOLO results and this fallback is skipped (see scan pipeline).

Returns the same BoundingBox shape as model_manager so call sites are interchangeable.
"""

from __future__ import annotations

import io
import logging
from typing import List, Tuple

from app.ml.model_manager import BoundingBox

logger = logging.getLogger(__name__)

# Tuned for typical phone scans of bricks on a flat surface.
MIN_AREA_FRACTION = 0.005   # ignore blobs smaller than 0.5% of the frame
MAX_AREA_FRACTION = 0.50    # ignore blobs larger than 50% (probably background/hand)
MIN_BOX_DIM_PX    = 40      # ignore boxes with either side < 40px
NMS_IOU           = 0.30
MAX_DETECTIONS    = 20


def detect_pieces_cv(image_bytes: bytes) -> List[BoundingBox]:
    """Detect brick-like blobs via classical CV. Empty list if opencv missing or no detections."""
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError:
        logger.debug("OpenCV/numpy/PIL not available — multi-piece CV detector disabled")
        return []

    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.array(pil)
        h, w = rgb.shape[:2]
        frame_area = h * w

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Foreground mask: anything with reasonable saturation OR strong value
        # (catches both bright-coloured bricks and white/black bricks against neutral bg)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask_colour = sat > 60                          # any saturated colour
        mask_dark   = val < 60                          # very dark (black bricks)
        mask_bright = (val > 230) & (sat < 30)          # near-white bricks against non-white bg
        fg = (mask_colour | mask_dark | mask_bright).astype(np.uint8) * 255

        # Clean up: open then close to remove specks and fill brick interiors
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Tuple[float, int, int, int, int]] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < MIN_AREA_FRACTION * frame_area or area > MAX_AREA_FRACTION * frame_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < MIN_BOX_DIM_PX or bh < MIN_BOX_DIM_PX:
                continue
            # Pseudo-confidence from area density: larger + more solid = higher
            density = area / max(1.0, bw * bh)
            conf = min(0.95, 0.40 + 0.4 * density + 0.15 * (area / frame_area))
            candidates.append((conf, x, y, bw, bh))

        candidates.sort(key=lambda t: t[0], reverse=True)
        kept = _nms(candidates, iou_threshold=NMS_IOU)[:MAX_DETECTIONS]

        if not kept:
            return []

        boxes: List[BoundingBox] = []
        for conf, x, y, bw, bh in kept:
            x2, y2 = x + bw, y + bh
            crop_pil = pil.crop((x, y, x2, y2))
            buf = io.BytesIO()
            crop_pil.save(buf, format="JPEG", quality=85)
            boxes.append(BoundingBox(
                x1=x / w, y1=y / h, x2=x2 / w, y2=y2 / h,
                confidence=conf,
                crop_bytes=buf.getvalue(),
            ))

        logger.info("CV multi-piece: %d detection(s)", len(boxes))
        return boxes

    except Exception as e:
        logger.warning("CV multi-piece detector failed: %s", e)
        return []


def _nms(cands: List[Tuple[float, int, int, int, int]], iou_threshold: float
         ) -> List[Tuple[float, int, int, int, int]]:
    kept: List[Tuple[float, int, int, int, int]] = []
    for c in cands:
        if all(_iou(c, k) < iou_threshold for k in kept):
            kept.append(c)
    return kept


def _iou(a: Tuple[float, int, int, int, int], b: Tuple[float, int, int, int, int]) -> float:
    _, ax, ay, aw, ah = a
    _, bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    xi1, yi1 = max(ax, bx), max(ay, by)
    xi2, yi2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0
