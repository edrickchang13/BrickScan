#!/bin/bash
# Quick-start script for YOLOv8 LEGO piece detection training

set -e

# Configuration
PARTS_DIR="${PARTS_DIR:-~/brickscan/ml/data/test_renders}"
OUTPUT_DIR="${OUTPUT_DIR:-./yolo_detector}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PATIENCE="${PATIENCE:-15}"
IMAGE_SIZE="${IMAGE_SIZE:-640}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parts-dir)
            PARTS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --no-generate-data)
            NO_GENERATE_DATA="--no-generate-data"
            shift
            ;;
        --no-train)
            NO_TRAIN="--no-train"
            shift
            ;;
        --no-export)
            NO_EXPORT="--no-export"
            shift
            ;;
        --help)
            echo "YOLOv8 LEGO Piece Detection Training Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --parts-dir DIR         Directory with rendered LEGO parts (default: ~/brickscan/ml/data/test_renders)"
            echo "  --output-dir DIR        Output directory (default: ./yolo_detector)"
            echo "  --epochs N              Training epochs (default: 100)"
            echo "  --batch-size N          Batch size (default: 32)"
            echo "  --patience N            Early stopping patience (default: 15)"
            echo "  --image-size N          Image size in pixels (default: 640)"
            echo "  --no-generate-data      Skip synthetic data generation"
            echo "  --no-train              Skip model training"
            echo "  --no-export             Skip ONNX export"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Full pipeline with defaults"
            echo "  $0"
            echo ""
            echo "  # Train with custom hyperparameters"
            echo "  $0 --epochs 50 --batch-size 64 --patience 10"
            echo ""
            echo "  # Skip data generation (use existing dataset)"
            echo "  $0 --no-generate-data"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Expand paths
PARTS_DIR=$(eval echo "$PARTS_DIR")
OUTPUT_DIR=$(eval echo "$OUTPUT_DIR")

# Print configuration
echo "========================================================================"
echo "YOLOv8 LEGO Piece Detection Training"
echo "========================================================================"
echo "Parts directory:      $PARTS_DIR"
echo "Output directory:     $OUTPUT_DIR"
echo "Image size:           $IMAGE_SIZE x $IMAGE_SIZE"
echo "Epochs:               $EPOCHS"
echo "Batch size:           $BATCH_SIZE"
echo "Early stopping:       patience=$PATIENCE"
echo ""

# Verify parts directory exists
if [ ! -d "$PARTS_DIR" ]; then
    echo "ERROR: Parts directory not found: $PARTS_DIR"
    exit 1
fi

echo "Found $(ls -1 "$PARTS_DIR"/*.png 2>/dev/null | wc -l) part images"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/train_yolo_detector.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Training script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Run training
echo "Starting training pipeline..."
echo ""

python "$PYTHON_SCRIPT" \
    --parts-dir "$PARTS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patience "$PATIENCE" \
    --image-size "$IMAGE_SIZE" \
    $NO_GENERATE_DATA \
    $NO_TRAIN \
    $NO_EXPORT

echo ""
echo "========================================================================"
echo "Training complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================================================"
echo ""
echo "Key files:"
echo "  - Trained model: $OUTPUT_DIR/training/yolo_detector/weights/best.pt"
echo "  - ONNX model:    $OUTPUT_DIR/models/best.onnx"
echo "  - Metrics:       $OUTPUT_DIR/models/training_metrics.json"
echo "  - Plots:         $OUTPUT_DIR/training/yolo_detector/plots/"
echo ""
