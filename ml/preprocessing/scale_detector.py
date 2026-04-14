"""
Scale detector for LEGO brick scanning.

Detects the physical scale of a brick image by:
1. Detecting skin-colored blobs (finger) or credit card
2. Estimating width in pixels
3. Falling back to LEGO stud diameter detection for scale inference
4. Filtering predictions by physical size consistency

Provides scale calibration for downstream inference tasks.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from pathlib import Path
import logging

import numpy as np
import cv2
from PIL import Image

from .stud_detector import detect_stud_grid

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

# Standard LEGO stud diameter: 8 mm
LEGO_STUD_DIAMETER_MM = 8.0

# Credit card dimensions (ISO/IEC 7810 ID-1)
CREDIT_CARD_WIDTH_MM = 85.6
CREDIT_CARD_HEIGHT_MM = 53.98

# Skin color range in HSV (finger detection)
SKIN_HUE_MIN = 0
SKIN_HUE_MAX = 20
SKIN_SAT_MIN = 10
SKIN_SAT_MAX = 255
SKIN_VAL_MIN = 60
SKIN_VAL_MAX = 255


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class ScaleResult:
    """Result of scale detection."""
    pixels_per_mm: float
    reference_type: str  # 'finger', 'credit_card', 'stud_grid', or 'unknown'
    confidence: float    # 0-1
    estimated_width_mm: Optional[float] = None


@dataclass
class SizeFilterResult:
    """Result of size-based filtering."""
    part_num: str
    confidence: float
    is_physically_plausible: bool
    measured_size_mm: Optional[float]
    expected_size_mm: Optional[float]


# ─── Finger detection ────────────────────────────────────────────────────────

def detect_finger_scale(image: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Detect a finger in the image and estimate its width.

    Args:
        image: BGR or RGB numpy array (H, W, 3)

    Returns:
        Tuple of (pixels_per_mm, confidence) if finger detected, else None.
        Assumes average finger width is 20mm.
    """
    try:
        # Convert to HSV for skin detection
        if image.shape[2] == 3:
            # Assume BGR
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            logger.warning("Invalid image shape for finger detection: %s", image.shape)
            return None

        # Create skin color mask
        lower_skin = np.array([SKIN_HUE_MIN, SKIN_SAT_MIN, SKIN_VAL_MIN])
        upper_skin = np.array([SKIN_HUE_MAX, SKIN_SAT_MAX, SKIN_VAL_MAX])
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Get largest contour (should be the finger)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < 500:  # Too small to be a finger
            return None

        # Fit ellipse to contour
        if len(largest_contour) < 5:
            return None

        ellipse = cv2.fitEllipse(largest_contour)
        (cx, cy), (width, height), angle = ellipse

        # Finger width is the minor axis of the ellipse
        finger_width_px = min(width, height)

        if finger_width_px < 20:  # Unreasonably small
            return None

        # Estimate: average finger width is ~20mm
        pixels_per_mm = finger_width_px / 20.0
        confidence = min(area / 5000, 1.0)  # Higher area = higher confidence

        return (pixels_per_mm, confidence)

    except Exception as e:
        logger.warning("Finger detection failed: %s", e)
        return None


# ─── Credit card detection ──────────────────────────────────────────────────

def detect_credit_card_scale(image: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Detect a credit card in the image and estimate scale from its dimensions.

    Uses edge detection and contour fitting.

    Args:
        image: BGR or RGB numpy array (H, W, 3)

    Returns:
        Tuple of (pixels_per_mm, confidence) if card detected, else None.
    """
    try:
        if len(image.shape) != 3 or image.shape[2] not in (3, 4):
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Look for rectangular contours (credit card shape is 85.6mm × 53.98mm ≈ 1.585 aspect ratio)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5000:  # Too small
                continue

            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) != 4:  # Not a quadrilateral
                continue

            # Fit rectangle
            rect = cv2.minAreaRect(contour)
            (cx, cy), (w, h), angle = rect

            # Check aspect ratio (should be ~1.585 for credit card)
            aspect_ratio = max(w, h) / min(w, h)
            if not (1.4 < aspect_ratio < 1.8):
                continue

            # Estimate pixels per mm from card width
            card_width_px = max(w, h)
            pixels_per_mm = card_width_px / CREDIT_CARD_WIDTH_MM
            confidence = 0.85  # High confidence for structured shape

            return (pixels_per_mm, confidence)

        return None

    except Exception as e:
        logger.warning("Credit card detection failed: %s", e)
        return None


# ─── Stud grid fallback ──────────────────────────────────────────────────────

def detect_stud_based_scale(image: Union[str, Path, Image.Image, np.ndarray]) -> Optional[Tuple[float, float]]:
    """
    Detect scale from LEGO stud grid.

    Args:
        image: Image input (same as stud_detector.detect_stud_grid)

    Returns:
        Tuple of (pixels_per_mm, confidence) if studs detected, else None.
    """
    try:
        result = detect_stud_grid(image)
        if not result:
            return None

        # Stud pitch: 8mm per stud
        # Detected pitch in pixels per stud
        pixels_per_stud = result.pixel_pitch
        pixels_per_mm = pixels_per_stud / LEGO_STUD_DIAMETER_MM
        confidence = result.confidence * 0.9  # Slightly lower than direct measurement

        return (pixels_per_mm, confidence)

    except Exception as e:
        logger.warning("Stud-based scale detection failed: %s", e)
        return None


# ─── Main scale detection ────────────────────────────────────────────────────

def detect_scale(image: Union[str, Path, Image.Image, np.ndarray]) -> ScaleResult:
    """
    Detect the physical scale of a brick image.

    Tries multiple methods in order:
    1. Finger detection (highest confidence)
    2. Credit card detection
    3. LEGO stud grid detection (fallback)

    Args:
        image: Image input — file path, PIL Image, or numpy array

    Returns:
        ScaleResult with pixels_per_mm, reference_type, and confidence.
    """
    # Convert to numpy array if needed
    if isinstance(image, (str, Path)):
        img_array = cv2.imread(str(image))
        if img_array is None:
            logger.warning("Failed to load image")
            return ScaleResult(pixels_per_mm=0, reference_type='unknown', confidence=0.0)
    elif isinstance(image, Image.Image):
        img_array = np.array(image.convert('RGB'))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_array = image.copy() if isinstance(image, np.ndarray) else None

    if img_array is None or not isinstance(img_array, np.ndarray):
        return ScaleResult(pixels_per_mm=0, reference_type='unknown', confidence=0.0)

    # Try finger detection
    finger_result = detect_finger_scale(img_array)
    if finger_result:
        pixels_per_mm, confidence = finger_result
        if confidence > 0.3:
            return ScaleResult(
                pixels_per_mm=pixels_per_mm,
                reference_type='finger',
                confidence=confidence,
            )

    # Try credit card detection
    card_result = detect_credit_card_scale(img_array)
    if card_result:
        pixels_per_mm, confidence = card_result
        if confidence > 0.3:
            return ScaleResult(
                pixels_per_mm=pixels_per_mm,
                reference_type='credit_card',
                confidence=confidence,
            )

    # Try stud grid detection
    stud_result = detect_stud_based_scale(img_array)
    if stud_result:
        pixels_per_mm, confidence = stud_result
        if confidence > 0.3:
            return ScaleResult(
                pixels_per_mm=pixels_per_mm,
                reference_type='stud_grid',
                confidence=confidence,
            )

    # No scale detected
    return ScaleResult(pixels_per_mm=0, reference_type='unknown', confidence=0.0)


# ─── Size filtering ──────────────────────────────────────────────────────────

def filter_by_size(
    predictions: List[dict],
    scale_result: ScaleResult,
    part_dimensions_mm: dict,
) -> List[SizeFilterResult]:
    """
    Filter predictions based on physical size consistency.

    Given detected predictions and a measured scale, check if each prediction's
    physical dimensions are plausible (±25% tolerance).

    Args:
        predictions: List of prediction dicts with 'part_num', 'confidence', 'bbox' (optional)
        scale_result: ScaleResult from detect_scale()
        part_dimensions_mm: Dict mapping part_num → {'width': float, 'length': float, 'height': float}

    Returns:
        List of SizeFilterResult objects indicating physical plausibility.
    """
    results = []

    if scale_result.pixels_per_mm <= 0:
        # No scale detected; pass all predictions through
        for pred in predictions:
            results.append(SizeFilterResult(
                part_num=pred.get('part_num', ''),
                confidence=pred.get('confidence', 0.0),
                is_physically_plausible=True,
                measured_size_mm=None,
                expected_size_mm=None,
            ))
        return results

    for pred in predictions:
        part_num = pred.get('part_num', '')
        confidence = pred.get('confidence', 0.0)
        bbox = pred.get('bbox')  # [x, y, width, height] in pixels

        if part_num not in part_dimensions_mm or not bbox:
            # No dimension data or bbox; assume plausible
            results.append(SizeFilterResult(
                part_num=part_num,
                confidence=confidence,
                is_physically_plausible=True,
                measured_size_mm=None,
                expected_size_mm=None,
            ))
            continue

        # Get expected dimensions
        dims = part_dimensions_mm[part_num]
        expected_width_mm = dims.get('width')
        expected_length_mm = dims.get('length')

        # Measure from bbox
        bbox_width_px, bbox_height_px = bbox[2], bbox[3]
        measured_width_mm = bbox_width_px / scale_result.pixels_per_mm
        measured_length_mm = bbox_height_px / scale_result.pixels_per_mm

        # Check consistency (±25% tolerance)
        tolerance = 1.25
        is_plausible = True
        if expected_width_mm:
            if not (expected_width_mm / tolerance <= measured_width_mm <= expected_width_mm * tolerance):
                is_plausible = False
        if expected_length_mm and is_plausible:
            if not (expected_length_mm / tolerance <= measured_length_mm <= expected_length_mm * tolerance):
                is_plausible = False

        results.append(SizeFilterResult(
            part_num=part_num,
            confidence=confidence,
            is_physically_plausible=is_plausible,
            measured_size_mm=measured_width_mm,
            expected_size_mm=expected_width_mm,
        ))

    return results
