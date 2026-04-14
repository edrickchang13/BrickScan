# YOLOv8 LEGO Piece Detection Training

Complete training pipeline for detecting individual LEGO pieces in cluttered scenes using YOLOv8m on NVIDIA DGX Spark GB10.

## Quick Start

```bash
cd ~/brickscan/ml

# Run complete pipeline (generate data + train + export)
python train_yolo_detector.py \
    --parts-dir ~/brickscan/ml/data/test_renders \
    --output-dir ./yolo_detector

# Or use the shell wrapper
./train_yolo.sh
```

**Total time: ~4-5 hours**

Results saved to: `./yolo_detector/`

## Files Included

| File | Purpose |
|------|---------|
| `train_yolo_detector.py` | Main training script (691 lines) |
| `train_yolo.sh` | Shell wrapper for easy execution |
| `YOLO_TRAINING_GUIDE.md` | Complete user guide with examples |
| `YOLO_SCRIPT_DOCUMENTATION.md` | Technical documentation of script components |
| `YOLO_EXAMPLES.md` | 11 detailed usage examples with expected outputs |
| `YOLO_README.md` | This file |

## What the Script Does

### 1. Synthetic Data Generation
- Takes 4 rendered LEGO part images from `~/brickscan/ml/data/test_renders/`
- Composites 3-15 parts randomly onto varied backgrounds
- Creates 5000 training + 1000 validation images
- Outputs in YOLO format (image + normalized bbox labels)

### 2. YOLOv8m Training
- Trains on generated synthetic data
- 100 epochs with early stopping (patience=15)
- Strong augmentation: mosaic, mixup, HSV shifts
- Uses GPU automatically

### 3. Model Export
- Exports best model to ONNX format
- Framework-agnostic, deployable on CPU/GPU/mobile
- Saves training metrics and visualization plots

## Output Structure

```
yolo_detector/
├── dataset/              # Synthetic training data
│   ├── images/
│   │   ├── train/       (5000 JPG files)
│   │   └── val/         (1000 JPG files)
│   ├── labels/
│   │   ├── train/       (YOLO format annotations)
│   │   └── val/
│   └── data.yaml        (YOLO config)
├── training/            # Training results
│   └── yolo_detector/
│       ├── weights/
│       │   └── best.pt  (Trained model)
│       ├── results.csv  (Metrics per epoch)
│       └── plots/       (Visualizations)
└── models/              # Final outputs
    ├── best.onnx        (Deployed model)
    └── training_metrics.json
```

## Hardware & Environment

- **GPU**: NVIDIA DGX Spark GB10 Blackwell (130.6GB VRAM)
- **Python Environment**: `~/brickscan/ml/venv`
- **Pre-installed**: torch, torchvision, PIL, numpy, pandas, tqdm
- **Auto-installed**: ultralytics (YOLOv8)

## Key Features

✓ **Synthetic data generation** - No manual annotation needed
✓ **Strong augmentation** - Mosaic, mixup, HSV, geometric
✓ **Single-class detection** - "lego_piece" class
✓ **Early stopping** - Prevents overfitting
✓ **ONNX export** - Ready for deployment
✓ **Comprehensive logging** - Track progress at each stage
✓ **Flexible pipeline** - Can skip data gen or training
✓ **Customizable hyperparameters** - All exposed via CLI

## Common Commands

### Full Pipeline (Recommended)
```bash
python train_yolo_detector.py \
    --parts-dir ~/brickscan/ml/data/test_renders \
    --output-dir ./yolo_detector
```

### Data Generation Only
```bash
python train_yolo_detector.py \
    --parts-dir ~/brickscan/ml/data/test_renders \
    --output-dir ./yolo_detector \
    --no-train --no-export
```

### Training Only
```bash
python train_yolo_detector.py \
    --output-dir ./yolo_detector \
    --no-generate-data
```

### Custom Parameters
```bash
python train_yolo_detector.py \
    --parts-dir ~/brickscan/ml/data/test_renders \
    --output-dir ./yolo_detector \
    --epochs 50 \
    --batch-size 64 \
    --patience 10 \
    --image-size 512
```

## Inference

After training, use the model:

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolo_detector/training/yolo_detector/weights/best.pt')

# Detect
results = model.predict('image.jpg', conf=0.5)

# Process results
for r in results:
    for box in r.boxes:
        print(f"Confidence: {box.conf:.3f}")
        print(f"Bbox: {box.xyxy}")
```

## Performance

Expected on DGX Spark GB10:

| Stage | Time | Notes |
|-------|------|-------|
| Data generation | ~1 hour | 6000 images |
| Training | ~3-4 hours | 100 epochs |
| ONNX export | ~2 minutes | One-time |
| Inference | 50-100 FPS | Batch of 32 images |

## Documentation

- **YOLO_TRAINING_GUIDE.md** - Comprehensive user guide
  - Dataset generation details
  - Training parameters
  - Model evaluation
  - Troubleshooting
  
- **YOLO_SCRIPT_DOCUMENTATION.md** - Technical deep-dive
  - Component breakdown
  - Data flow
  - Implementation details
  - Debugging tips

- **YOLO_EXAMPLES.md** - Real-world usage examples
  - 11 complete examples with expected outputs
  - Different training scenarios
  - Inference patterns
  - Performance benchmarks

## Troubleshooting

### Out of Memory (OOM)
```bash
python train_yolo_detector.py \
    --output-dir ./yolo_detector \
    --batch-size 16 \
    --image-size 512
```

### Slow Training
```bash
python train_yolo_detector.py \
    --output-dir ./yolo_detector \
    --batch-size 128  # DGX can handle it
```

### Poor Results
- Increase training data: `--num-train 10000`
- Extend training: `--epochs 200`
- Check data quality in `dataset/images/train/`

## Requirements Met

✓ Synthetic data generation from rendered parts
✓ 5000 training + 1000 validation images
✓ YOLO format output (image + bbox labels)
✓ YOLOv8m model training
✓ Strong augmentation (mosaic, mixup, HSV)
✓ 100 epochs with early stopping (patience=15)
✓ ONNX export
✓ Proper logging and argument parsing
✓ Complete documentation
✓ Ready for GB10 GPU (130.6GB VRAM available)

## Next Steps

1. Run training: `python train_yolo_detector.py --help` for all options
2. Monitor progress: Check `yolo_detector/training/yolo_detector/results.csv`
3. Evaluate model: Run inference on test images
4. Deploy: Use exported ONNX model
5. Fine-tune: Adjust hyperparameters if needed

## Support

For detailed information, see:
- Training parameters → YOLO_TRAINING_GUIDE.md
- Script internals → YOLO_SCRIPT_DOCUMENTATION.md
- Usage examples → YOLO_EXAMPLES.md
