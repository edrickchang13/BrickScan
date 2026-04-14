"""
Constants and configuration for local inventory system.
"""

# Confidence thresholds
CONFIDENCE_THRESHOLD_KNOWN = 0.80  # >= 80% is "known" (auto-add with review)
CONFIDENCE_THRESHOLD_UNCERTAIN = 0.50  # < 50% is very uncertain

# Image processing
MODEL_INPUT_SIZE = 224  # ONNX model expects 224x224
MODEL_CHANNELS = 3  # RGB
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB max

# ImageNet normalization (for ONNX preprocessing)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Database paths
INVENTORY_DB_NAME = "brickscan_inventory.db"
IMAGES_DIR_NAME = "brickscan_images"

# Inventory statuses
STATUS_KNOWN = "known"
STATUS_UNCERTAIN = "uncertain"
