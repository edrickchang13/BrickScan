"""
Image preprocessing and storage service.
Handles: validation, resizing, JPEG conversion, optional S3 storage.
"""

import base64
import io
import logging
import uuid
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
TARGET_SIZE = (512, 512)  # Resize before ML inference
JPEG_QUALITY = 85


def decode_base64_image(image_base64: str) -> bytes:
    """
    Decode base64 string to raw image bytes.

    Handles both raw base64 and data URL formats:
    - data:image/jpeg;base64,{data}
    - data:image/png;base64,{data}
    - {raw_base64}

    Args:
    - image_base64: Base64-encoded image string

    Returns:
    - Decoded image bytes

    Raises:
    - ValueError: If base64 is invalid or malformed
    """
    try:
        # Handle data URL format: data:image/jpeg;base64,{data}
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        return base64.b64decode(image_base64)

    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise ValueError(f"Invalid base64 image: {e}")


def validate_and_preprocess_image(image_bytes: bytes) -> bytes:
    """
    Validate image and preprocess for ML inference.

    - Validates file size (max 10MB)
    - Verifies it's a valid image format
    - Converts to RGB
    - Resizes to target size (512x512) with padding
    - Converts to JPEG format

    Args:
    - image_bytes: Raw image file bytes

    Returns:
    - Preprocessed image bytes (JPEG format)

    Raises:
    - ValueError: If image is invalid, too large, or corrupted
    """
    # Check file size
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise ValueError(
            f"Image too large: {len(image_bytes)} bytes "
            f"(max {MAX_IMAGE_SIZE_BYTES})"
        )

    try:
        # Open and validate image
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB (handles RGBA, grayscale, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize maintaining aspect ratio using thumbnail
        # This doesn't crop, just fits within the size
        img.thumbnail(TARGET_SIZE, Image.LANCZOS)

        # Pad to square if needed (center the image)
        if img.size != TARGET_SIZE:
            padded = Image.new("RGB", TARGET_SIZE, (128, 128, 128))
            offset = (
                (TARGET_SIZE[0] - img.size[0]) // 2,
                (TARGET_SIZE[1] - img.size[1]) // 2,
            )
            padded.paste(img, offset)
            img = padded

        # Convert to JPEG bytes
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=JPEG_QUALITY)
        output.seek(0)
        return output.getvalue()

    except Image.UnidentifiedImageError as e:
        logger.error(f"Invalid image data: {e}")
        raise ValueError(f"Invalid image data: {e}")
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise ValueError(f"Image processing failed: {e}")


async def save_scan_image_to_s3(
    image_bytes: bytes, user_id: str, s3_client=None, bucket_name: str = None
) -> Optional[str]:
    """
    Upload preprocessed scan image to S3 for training data collection.

    Args:
    - image_bytes: Preprocessed JPEG image bytes
    - user_id: User who submitted the scan
    - s3_client: Async boto3 S3 client (optional)
    - bucket_name: S3 bucket name (optional)

    Returns:
    - S3 object key (path), or None if S3 not configured

    Raises:
    - Exception: If S3 upload fails (logs and returns None)
    """
    if not s3_client or not bucket_name:
        logger.debug("S3 not configured, skipping image upload")
        return None

    try:
        # Generate unique key: scans/{user_id}/{uuid}.jpg
        image_key = f"scans/{user_id}/{uuid.uuid4()}.jpg"

        # Upload to S3
        await s3_client.put_object(
            Bucket=bucket_name,
            Key=image_key,
            Body=image_bytes,
            ContentType="image/jpeg",
            ServerSideEncryption="AES256",
        )

        logger.info(f"Uploaded scan image to S3: {image_key}")
        return image_key

    except Exception as e:
        logger.error(f"Failed to upload scan image to S3: {e}")
        # Return None but don't raise - S3 upload failure shouldn't block scans
        return None


def get_image_metadata(image_bytes: bytes) -> dict:
    """
    Extract metadata from image bytes.

    Args:
    - image_bytes: Raw image bytes

    Returns:
    - Dictionary with: width, height, format, mode
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
        }
    except Exception as e:
        logger.error(f"Error extracting image metadata: {e}")
        return {}


def resize_image_for_thumbnail(image_bytes: bytes, size: tuple = (200, 200)) -> bytes:
    """
    Create a thumbnail from image bytes.

    Args:
    - image_bytes: Raw image bytes
    - size: Tuple (width, height) for thumbnail

    Returns:
    - JPEG thumbnail bytes

    Raises:
    - ValueError: If image is invalid
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.convert("RGB")
        img.thumbnail(size, Image.LANCZOS)

        output = io.BytesIO()
        img.save(output, format="JPEG", quality=75)
        output.seek(0)
        return output.getvalue()

    except Exception as e:
        logger.error(f"Thumbnail generation failed: {e}")
        raise ValueError(f"Failed to create thumbnail: {e}")
