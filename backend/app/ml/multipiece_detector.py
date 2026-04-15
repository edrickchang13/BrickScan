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
import threading
from typing import List, Optional, Tuple

from app.ml.model_manager import BoundingBox

logger = logging.getLogger(__name__)

# Module-level MOG2 background subtractor — learns the scene gradually so that
# bricks (foreground) pop out against a stable table/paper (background).
# Reset every MOG2_RESET_INTERVAL frames so a long-running worker doesn't drift
# or accumulate stale "background" from a scene that changed hours ago.
_mog2_lock = threading.Lock()
_mog2_subtractor: Optional[object] = None  # cv2.BackgroundSubtractorMOG2 when built
_mog2_frames_seen = 0
MOG2_RESET_INTERVAL = 500
# MOG2 kwargs — history=200 means the subtractor needs ~200 frames to stabilize.
# varThreshold=16 is OpenCV's default (tighter = fewer false positives).
MOG2_KWARGS = dict(history=200, varThreshold=16, detectShadows=False)

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


# ─────────────────────────────────────────────────────────────────────────────
# MOG2 background-subtraction detector
#
# When the user is scanning multiple bricks on a stable table/paper, MOG2
# learns the background across frames and highlights brick pixels as
# foreground. More accurate than HSV thresholding when the scene has varied
# colours — HSV can't separate two bricks of similar hue, but MOG2 can because
# the bricks are the only thing that MOVED (or wasn't there before).
#
# Based on Daniel West's Universal LEGO Sorter pipeline, adapted for mobile
# scanning where the background isn't a conveyor belt but usually a table.
# ─────────────────────────────────────────────────────────────────────────────

def detect_pieces_mog2(image_bytes: bytes) -> List[BoundingBox]:
    """
    MOG2 background-subtraction detector. Learns the background across
    successive frames; returns foreground blobs as BoundingBoxes.

    Returns [] when:
      - OpenCV / PIL are not available
      - The subtractor is still cold (< 50 frames seen)
      - No foreground blobs pass the area threshold
    In every "no result" case the caller should fall back to the HSV detector.
    """
    global _mog2_subtractor, _mog2_frames_seen

    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError:
        return []

    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.array(pil)
        h, w = rgb.shape[:2]
        frame_area = h * w
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        with _mog2_lock:
            # Lazy-init + periodic reset so old scenes don't pollute the model
            if _mog2_subtractor is None or _mog2_frames_seen >= MOG2_RESET_INTERVAL:
                _mog2_subtractor = cv2.createBackgroundSubtractorMOG2(**MOG2_KWARGS)
                _mog2_frames_seen = 0
            fg_mask = _mog2_subtractor.apply(bgr)
            _mog2_frames_seen += 1
            frames_seen = _mog2_frames_seen

        # During the first ~50 frames MOG2 returns near-total foreground while
        # it's still learning. Skip detection and let HSV handle it.
        if frames_seen < 50:
            logger.debug("MOG2 still warming up (%d frames) — yielding to fallback", frames_seen)
            return []

        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Tuple[float, int, int, int, int]] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < MIN_AREA_FRACTION * frame_area or area > MAX_AREA_FRACTION * frame_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < MIN_BOX_DIM_PX or bh < MIN_BOX_DIM_PX:
                continue
            density = area / max(1.0, bw * bh)
            conf = min(0.95, 0.50 + 0.4 * density + 0.05 * (area / frame_area))
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

        logger.info("MOG2 multi-piece: %d detection(s) after %d frames warm-up", len(boxes), frames_seen)
        return boxes

    except Exception as e:
        logger.warning("MOG2 detector failed: %s", e)
        return []


def detect_pieces(image_bytes: bytes, prefer_mog2: bool = True) -> List[BoundingBox]:
    """
    Dispatcher: try MOG2 first (better on stable backgrounds), fall through
    to the HSV detector when MOG2 is cold / returns nothing / fails.

    This is the function scan.py's `_maybe_detect_multipiece` should call
    going forward. The module keeps `detect_pieces_cv` and `detect_pieces_mog2`
    exposed so tests can target either one directly.
    """
    if prefer_mog2:
        boxes = detect_pieces_mog2(image_bytes)
        if boxes:
            return boxes
    return detect_pieces_cv(image_bytes)
