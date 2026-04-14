"""
Image preprocessing for local inventory scanning.

Handles:
- Decoding base64 image data from mobile camera
- Resizing and normalizing to 224x224 for ONNX model
- Saving original images for later review/retraining
- Validation of image format and size

Used by the /api/scan endpoint to prepare images for ML inference.
"""

import io
import os
import logging
from pathlib import Path
from typing import Tuple
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Image storage directory
_IMAGES_DIR = os.path.expanduser("~/brickscan_images")

# Model input size (must match training)
MODEL_INPUT_SIZE = 224

# Max image size before validation rejection (10 MB)
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024


def _ensure_images_dir() -> None:
    """Create images directory if it doesn't exist."""
    Path(_IMAGES_DIR).mkdir(parents=True, exist_ok=True)


def validate_and_decode_image(image_base64: str) -> bytes:
    """
    Decode and validate base64 image data.

    Args:
        image_base64: Base64-encoded image string

    Returns:
        Raw image bytes (PNG or JPEG)

    Raises:
        ValueError: If image is invalid, too large, or unsupported format
    """
    import base64

    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception as e:
        raise ValueError(f"Invalid base64 encoding: {e}")

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise ValueError(
            f"Image too large: {len(image_bytes)} bytes "
            f"(max {MAX_IMAGE_SIZE_BYTES} bytes)"
        )

    # Validate it's a real image
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Lightweight validation
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image: {e}")

    return image_bytes


def preprocess_for_inference(image_bytes: bytes) -> Tuple[np.ndarray, Image.Image]:
    """
    Prepare image for ONNX model inference.

    Converts to RGB, resizes to 224x224, normalizes using ImageNet statistics.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Tuple of:
        - numpy array [1, 3, 224, 224] float32 normalized
        - Original PIL Image (for display/saving)

    Raises:
        ValueError: If image cannot be processed
    """
    try:
        # Load and convert to RGB
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize with high-quality resampling
        img_resized = img.resize(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            Image.Resampling.LANCZOS,
        )

        # Convert to numpy array and normalize to [0, 1]
        arr = np.array(img_resized, dtype=np.float32) / 255.0

        # Convert HWC to CHW
        arr = arr.transpose(2, 0, 1)

        # ImageNet normalization (match training pipeline)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        arr = (arr - mean) / std

        # Add batch dimension: [1, 3, 224, 224]
        arr = np.expand_dims(arr, 0)

        return arr, img

    except Exception as e:
        raise ValueError(f"Failed to preprocess image: {e}")


def save_scan_image(image_bytes: bytes, part_num: str, confidence: float) -> str:
    """
    Save the original scanned image for later review and retraining.

    Images organized as:
        ~/brickscan_images/
            3001_white_0.92_20240412_143022.png
            3001_white_0.92_20240412_143022_orig.jpg
            ...

    Args:
        image_bytes: Raw image bytes
        part_num: Predicted LEGO part number
        confidence: Model confidence (0.0-1.0)

    Returns:
        Relative path suitable for storing in database

    Raises:
        IOError: If save fails
    """
    _ensure_images_dir()

    try:
        from datetime import datetime

        # Sanitize part_num for filename
        part_safe = part_num.replace("/", "_").replace(" ", "_")

        # Create filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S%f")
        conf_str = f"{confidence:.2f}".replace(".", "")
        filename = f"{part_safe}_{conf_str}_{timestamp}.png"
        filepath = os.path.join(_IMAGES_DIR, filename)

        # Save image (PIL auto-detects format)
        img = Image.open(io.BytesIO(image_bytes))
        img.save(filepath, "PNG")

        logger.info(f"Saved scan image: {filename}")
        return filename

    except Exception as e:
        logger.error(f"Failed to save scan image: {e}")
        raise IOError(f"Could not save image: {e}")


def get_images_dir() -> str:
    """Return path to images storage directory."""
    return _IMAGES_DIR
