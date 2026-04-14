#!/usr/bin/env python3
"""
Extract and auto-label LEGO brick crops from YouTube/local videos.

Extracts frames from videos, runs YOLO detection, crops bricks, and attempts
to auto-label each crop using the BrickScan scan API.

Usage:
  python3 extract_video_frames.py --video-dir ./videos --output-dir ./video_crops \\
    --scan-api http://localhost:8000 --api-key YOUR_KEY

  python3 extract_video_frames.py --video ./single_video.mp4 --output-dir ./crops
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Iterator, Dict
from io import BytesIO
import base64
import json

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: opencv-python and numpy required. Run: pip3 install opencv-python numpy")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests required. Run: pip3 install requests")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    Image = None
    print("WARNING: Pillow not installed. Image handling may fail. Run: pip3 install Pillow")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("extract_video_frames")


def extract_frames_from_video(
    video_path: Path,
    output_fps: float = 1.0,
) -> Iterator[np.ndarray]:
    """
    Extract frames from a video file at specified FPS.

    Args:
        video_path: Path to video file (.mp4, .avi, .mov, etc.)
        output_fps: Frame rate to extract (default: 1.0 = 1 frame/sec)

    Yields:
        BGR frames as numpy arrays
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        log.error(f"Failed to open video: {video_path}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30  # Fallback

    frame_interval = int(source_fps / output_fps)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            yield frame

        frame_count += 1

    cap.release()


def detect_and_crop_bricks(
    frame: np.ndarray,
    detector_model: Optional[object],
    min_confidence: float = 0.5,
) -> List[np.ndarray]:
    """
    Detect bricks in a frame and crop them.

    Currently a placeholder using Canny edge detection (no YOLO model required).
    In production, substitute with real YOLO detection.

    Args:
        frame: Input frame (BGR)
        detector_model: YOLO model (unused in placeholder)
        min_confidence: Confidence threshold

    Returns:
        List of cropped brick images
    """
    crops = []

    # Simple edge-based detection as placeholder
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = frame.shape[:2]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filter by size (reasonable brick size)
        if w < 20 or h < 20 or w > W * 0.8 or h > H * 0.8:
            continue

        # Add padding
        x1 = max(0, x - 5)
        y1 = max(0, y - 5)
        x2 = min(W, x + w + 5)
        y2 = min(H, y + h + 5)

        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)

    return crops


def auto_label_crop(
    crop: np.ndarray,
    scan_api_url: str,
    min_confidence: float = 0.85,
) -> Optional[Dict[str, str]]:
    """
    Attempt to auto-label a crop by querying the BrickScan scan API.

    Args:
        crop: Cropped brick image (BGR numpy array)
        scan_api_url: Base URL of BrickScan API (e.g., http://localhost:8000)
        min_confidence: Minimum confidence to accept prediction

    Returns:
        Dict with 'part_num', 'color_id', 'confidence' if successful, else None
    """
    if scan_api_url.endswith('/'):
        scan_api_url = scan_api_url[:-1]

    try:
        # Encode image as JPEG and base64
        _, buffer = cv2.imencode('.jpg', crop)
        b64 = base64.b64encode(buffer).decode('utf-8')

        # Call scan API
        response = requests.post(
            f"{scan_api_url}/api/scan",
            json={"image_base64": b64},
            timeout=10,
        )
        response.raise_for_status()

        data = response.json()
        predictions = data.get('predictions', [])

        if predictions:
            top = predictions[0]
            confidence = float(top.get('confidence', 0.0))

            if confidence >= min_confidence:
                return {
                    'part_num': top.get('part_num'),
                    'color_id': str(top.get('color_id', 'unknown')),
                    'confidence': str(confidence),
                }

    except Exception as e:
        log.debug(f"Auto-label failed: {e}")

    return None


def process_video_directory(
    video_dir: Path,
    output_dir: Path,
    scan_api_url: Optional[str] = None,
    min_label_confidence: float = 0.85,
):
    """
    Process all video files in a directory.

    Args:
        video_dir: Directory containing .mp4, .avi, .mov files
        output_dir: Output directory for crops
        scan_api_url: Optional BrickScan API URL for auto-labeling
        min_label_confidence: Minimum confidence to accept auto-label
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unlabeled_dir = output_dir / "_unlabeled"
    unlabeled_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_files.extend(Path(video_dir).glob(ext))

    log.info(f"Found {len(video_files)} video files in {video_dir}")

    total_crops = 0
    labeled_crops = 0
    unlabeled_crops = 0

    for video_file in video_files:
        log.info(f"Processing: {video_file.name}")

        frame_idx = 0
        for frame in extract_frames_from_video(video_file):
            crops = detect_and_crop_bricks(frame)

            for crop_idx, crop in enumerate(crops):
                # Try to auto-label
                label = None
                if scan_api_url:
                    label = auto_label_crop(crop, scan_api_url, min_label_confidence)

                if label:
                    # Save to labeled directory
                    part_num = label['part_num']
                    part_dir = output_dir / part_num
                    part_dir.mkdir(parents=True, exist_ok=True)

                    crop_name = f"{video_file.stem}_frame{frame_idx:04d}_crop{crop_idx:02d}.jpg"
                    crop_path = part_dir / crop_name

                    cv2.imwrite(str(crop_path), crop)

                    log.debug(f"  Labeled: {crop_name} → {part_num}")
                    labeled_crops += 1

                else:
                    # Save to unlabeled directory
                    crop_name = f"{video_file.stem}_frame{frame_idx:04d}_crop{crop_idx:02d}.jpg"
                    crop_path = unlabeled_dir / crop_name

                    cv2.imwrite(str(crop_path), crop)

                    log.debug(f"  Unlabeled: {crop_name}")
                    unlabeled_crops += 1

                total_crops += 1

            frame_idx += 1

    log.info("=" * 60)
    log.info(f"Video extraction complete!")
    log.info(f"Total crops: {total_crops}")
    log.info(f"  Labeled: {labeled_crops}")
    log.info(f"  Unlabeled: {unlabeled_crops}")
    log.info(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and auto-label LEGO bricks from videos"
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Single video file to process"
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Directory containing video files"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for extracted crops"
    )
    parser.add_argument(
        "--scan-api",
        default=None,
        help="BrickScan API URL for auto-labeling (e.g., http://localhost:8000)"
    )
    parser.add_argument(
        "--min-label-confidence",
        type=float,
        default=0.85,
        help="Minimum confidence to accept auto-label (default: 0.85)"
    )
    parser.add_argument(
        "--extract-fps",
        type=float,
        default=1.0,
        help="Frames per second to extract (default: 1.0 = 1 frame/sec)"
    )

    args = parser.parse_args()

    if not args.video and not args.video_dir:
        print("ERROR: Must provide either --video or --video-dir")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    if args.video:
        video_file = Path(args.video)
        if not video_file.exists():
            log.error(f"Video file not found: {video_file}")
            sys.exit(1)

        log.info(f"Processing single video: {video_file}")
        output_dir.mkdir(parents=True, exist_ok=True)
        unlabeled_dir = output_dir / "_unlabeled"
        unlabeled_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        labeled_count = 0
        unlabeled_count = 0

        for frame in extract_frames_from_video(video_file):
            crops = detect_and_crop_bricks(frame)

            for crop_idx, crop in enumerate(crops):
                label = None
                if args.scan_api:
                    label = auto_label_crop(crop, args.scan_api, args.min_label_confidence)

                if label:
                    part_dir = output_dir / label['part_num']
                    part_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = part_dir / f"{video_file.stem}_f{frame_idx:04d}_c{crop_idx:02d}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    labeled_count += 1
                else:
                    crop_path = unlabeled_dir / f"{video_file.stem}_f{frame_idx:04d}_c{crop_idx:02d}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    unlabeled_count += 1

            frame_idx += 1

        log.info(f"Complete: {labeled_count} labeled, {unlabeled_count} unlabeled")

    else:
        video_dir = Path(args.video_dir)
        if not video_dir.exists():
            log.error(f"Video directory not found: {video_dir}")
            sys.exit(1)

        process_video_directory(
            video_dir,
            output_dir,
            scan_api_url=args.scan_api,
            min_label_confidence=args.min_label_confidence,
        )


if __name__ == "__main__":
    main()
