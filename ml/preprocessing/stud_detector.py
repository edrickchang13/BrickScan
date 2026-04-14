"""
Stud-grid geometric detector for LEGO bricks.

Analyzes a brick photo to extract stud grid dimensions (cols x rows),
which provides near-certain knowledge of brick width and length.
Uses classical computer vision: circle detection, grid fitting, and clustering.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from pathlib import Path
import io
import logging

import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class StudGridResult:
    """Result of stud grid detection."""
    cols: int                           # stud columns detected (width)
    rows: int                           # stud rows detected (length)
    confidence: float                   # 0-1, how confident the detection is
    stud_count: int                     # raw number of studs found
    pixel_pitch: float                  # detected stud-to-stud pixel distance
    centers: List[Tuple[int, int]]     # pixel centers of each detected stud


def detect_stud_grid(
    image: Union[str, Path, Image.Image, np.ndarray]
) -> Optional[StudGridResult]:
    """
    Detect the stud grid on a LEGO brick from a photo.

    Args:
        image: Image input — file path, PIL Image, or numpy array (BGR or RGB).

    Returns:
        StudGridResult if grid detected with confidence > 0.5 and >= 4 studs,
        else None.
    """
    try:
        # Convert to numpy BGR for OpenCV
        if isinstance(image, (str, Path)):
            img_array = cv2.imread(str(image))
            if img_array is None:
                logger.warning("Failed to load image from %s", image)
                return None
        elif isinstance(image, Image.Image):
            img_array = np.array(image.convert("RGB"))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            # Assume BGR if 3 channels, else grayscale
            img_array = image.copy()
            if len(img_array.shape) == 2:
                # Grayscale to BGR
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif img_array.shape[2] == 3:
                # Assume BGR
                pass
            else:
                logger.warning("Invalid image shape: %s", img_array.shape)
                return None
        else:
            logger.warning("Unsupported image type: %s", type(image))
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 1.5)

        # Try HoughCircles at multiple scales
        best_circles = None
        best_scale = 1.0
        best_count = 0

        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            scaled_gray = cv2.resize(gray, None, fx=scale, fy=scale)

            circles = cv2.HoughCircles(
                scaled_gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=15,  # Studs are ~8mm grid pitch, so min 15px separation
                param1=50,   # Canny edge threshold
                param2=30,   # Accumulator threshold
                minRadius=int(6 * scale),
                maxRadius=int(40 * scale),
            )

            if circles is not None:
                circles = circles[0]
                circle_count = len(circles)

                if circle_count > best_count:
                    best_count = circle_count
                    # Scale back to original resolution
                    if scale != 1.0:
                        circles = circles / scale
                    best_circles = circles
                    best_scale = scale

        if best_circles is None or len(best_circles) < 4:
            logger.debug("No studs detected (found %d circles)", best_count)
            return None

        # Extract centers and radii
        centers = [(int(x), int(y)) for x, y, r in best_circles]
        radii = [r for x, y, r in best_circles]

        # Filter circles: keep only those whose centers form a regular grid
        filtered_centers, pitch = _filter_grid_circles(centers, radii)

        if len(filtered_centers) < 4:
            logger.debug("Filtered circles dropped below 4 (had %d)", len(centers))
            return None

        # Fit a 2D grid to the filtered centers
        grid_rows, grid_cols = _fit_grid_to_centers(filtered_centers)

        if grid_rows < 2 or grid_cols < 2:
            logger.debug("Fitted grid too small: %dx%d", grid_cols, grid_rows)
            return None

        # Calculate confidence: how many of the detected circles align to the grid
        confidence = len(filtered_centers) / len(centers) if centers else 0.0

        if confidence < 0.5:
            logger.debug("Confidence too low: %.2f", confidence)
            return None

        result = StudGridResult(
            cols=grid_cols,
            rows=grid_rows,
            confidence=confidence,
            stud_count=len(filtered_centers),
            pixel_pitch=pitch,
            centers=filtered_centers,
        )

        logger.info(
            "Stud grid detected: %dx%d (confidence=%.2f, pitch=%.1fpx, %d studs)",
            grid_cols, grid_rows, confidence, pitch, len(filtered_centers),
        )
        return result

    except Exception as e:
        logger.error("Stud detection error: %s", e)
        return None


def _filter_grid_circles(
    centers: List[Tuple[int, int]],
    radii: List[float],
    tolerance: float = 0.2,
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Filter detected circles to keep only those forming a regular grid.

    Checks that inter-center distances cluster around a consistent pitch.
    Rejects outliers beyond tolerance (default 20%) of median pitch.

    Args:
        centers: List of (x, y) stud centers.
        radii: List of radii for weighting.
        tolerance: Allowed deviation from median pitch (default 0.2 = 20%).

    Returns:
        (filtered_centers, pitch) where filtered_centers form a regular grid
        and pitch is the detected stud-to-stud distance in pixels.
    """
    if len(centers) < 4:
        return centers, 1.0

    centers_array = np.array(centers, dtype=np.float32)

    # Compute pairwise distances (excluding 0)
    distances = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.linalg.norm(centers_array[i] - centers_array[j])
            if dist > 1.0:  # Avoid duplicates / numerical errors
                distances.append(dist)

    if not distances:
        return centers, 1.0

    distances = np.array(distances)
    median_dist = np.median(distances)

    # The pitch is roughly the smallest cluster of distances
    # (nearest neighbors in a regular grid)
    sorted_dists = np.sort(distances)

    # Find the pitch: look for a cluster of small distances
    # Heuristic: the pitch should be in the lower quartile
    pitch_candidates = sorted_dists[: max(4, len(sorted_dists) // 4)]
    pitch = np.median(pitch_candidates)

    if pitch < 2.0:
        pitch = np.median(distances)

    # Filter centers: keep only those whose nearest neighbor distance
    # is close to the pitch (within tolerance)
    filtered_centers = []
    for center in centers:
        center = np.array(center, dtype=np.float32)
        # Find nearest neighbor distance
        dists_to_others = np.linalg.norm(centers_array - center, axis=1)
        dists_to_others = dists_to_others[dists_to_others > 0.5]

        if len(dists_to_others) == 0:
            continue

        nearest_dist = np.min(dists_to_others)

        # Keep if nearest neighbor is within pitch * (1 ± tolerance)
        if pitch * (1 - tolerance) <= nearest_dist <= pitch * (1 + tolerance):
            filtered_centers.append(tuple(map(int, center)))

    return filtered_centers, float(pitch)


def _fit_grid_to_centers(
    centers: List[Tuple[int, int]]
) -> Tuple[int, int]:
    """
    Fit a 2D grid to stud centers using 1D clustering on X and Y coordinates.

    Args:
        centers: List of (x, y) stud centers.

    Returns:
        (rows, cols) — number of distinct rows and columns.
    """
    if not centers:
        return 0, 0

    centers_array = np.array(centers)
    x_coords = centers_array[:, 0]
    y_coords = centers_array[:, 1]

    # Cluster X coordinates to find columns
    cols = _cluster_1d_coords(x_coords)

    # Cluster Y coordinates to find rows
    rows = _cluster_1d_coords(y_coords)

    return rows, cols


def _cluster_1d_coords(coords: np.ndarray, tolerance: float = 15.0) -> int:
    """
    Cluster 1D coordinates to find distinct lines.

    Uses histogram peak finding: counts coordinates in bins and identifies
    local maxima as cluster centers.

    Args:
        coords: 1D array of coordinates.
        tolerance: Bin width for clustering.

    Returns:
        Number of distinct clusters.
    """
    if len(coords) < 2:
        return 1

    # Use histogram-based clustering
    hist, bin_edges = np.histogram(coords, bins=max(2, len(coords) // 3))

    # Find local maxima in the histogram
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            peaks.append(i)

    if not peaks:
        # Fallback: use density-based clustering
        if len(np.unique(coords)) < len(coords) / 2:
            return len(np.unique(coords))
        else:
            # Use K-means-like approach with sorted coords
            sorted_coords = np.sort(coords)
            clusters = 1
            for i in range(1, len(sorted_coords)):
                if sorted_coords[i] - sorted_coords[i - 1] > tolerance:
                    clusters += 1
            return clusters

    return len(peaks)


def constrain_predictions(
    predictions: List[dict],
    grid: StudGridResult,
    part_dimensions: dict,
    tolerance: int = 1,
) -> List[dict]:
    """
    Re-score predictions based on stud grid detection.

    Multiplies each prediction's confidence by 1.5 if its declared dimensions
    match the detected grid (within tolerance), by 0.1 if they contradict it.
    Re-normalizes the list so confidences sum to 1.0 (or less if many fail).

    Args:
        predictions: List of dicts with keys: part_num, confidence, ...
        grid: Detected StudGridResult.
        part_dimensions: Dict mapping part_num -> (width, length, height).
                         Width and length are in studs.
        tolerance: Allow ±tolerance studs of deviation (default 1).

    Returns:
        List of predictions with adjusted confidence scores.
    """
    if not predictions or grid.confidence < 0.5:
        return predictions

    detected_width = grid.cols
    detected_length = grid.rows

    adjusted = []
    for pred in predictions:
        part_num = pred.get("part_num")
        confidence = pred.get("confidence", 0.0)

        # Look up part dimensions
        dims = part_dimensions.get(part_num)

        if dims is None:
            # No dimension data: keep as-is
            adjusted.append({**pred, "confidence": confidence})
            continue

        width, length, height = dims

        # Check if dimensions match detected grid (within tolerance)
        # Note: we don't know brick orientation, so check both orientations
        matches_orientation_1 = (
            abs(width - detected_width) <= tolerance
            and abs(length - detected_length) <= tolerance
        )
        matches_orientation_2 = (
            abs(width - detected_length) <= tolerance
            and abs(length - detected_width) <= tolerance
        )

        if matches_orientation_1 or matches_orientation_2:
            # Boost confidence
            new_conf = confidence * 1.5
        else:
            # Penalize confidence
            new_conf = confidence * 0.1

        adjusted.append({**pred, "confidence": new_conf})

    # Re-normalize so sum ≈ 1.0 (or less if all are heavily penalized)
    total_conf = sum(p["confidence"] for p in adjusted)
    if total_conf > 0.001:
        adjusted = [
            {**p, "confidence": p["confidence"] / total_conf}
            for p in adjusted
        ]

    return adjusted
