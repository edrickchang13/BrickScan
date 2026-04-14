# YOLOv8 LEGO Piece Detection Training Guide

Complete training script for detecting individual LEGO pieces in cluttered scenes using YOLOv8m on NVIDIA DGX Spark.

## Overview

This training pipeline:
1. **Generates synthetic detection data** by compositing rendered LEGO parts onto randomized backgrounds
2. **Trains YOLOv8m** for single-class object detection ("lego_piece")
3. **Applies strong augmentation** (mosaic, mixup, HSV) to improve robustness
4. **Trains for 100 epochs** with early stopping (patience=15)
5. **Exports to ONNX** for deployment flexibility
6. **Saves metrics and plots** for analysis

## Hardware & Environment

- **GPU**: NVIDIA DGX Spark with GB10 Blackwell GPU (130.6GB VRAM)
- **Python Environment**: `~/brickscan/ml/venv`
- **Pre-installed packages**: torch, torchvision, PIL, numpy, pandas, tqdm
- **Auto-installed**: ultralytics (YOLOv8)

## Dataset Generation

### Source Data

The script uses rendered LEGO part images from:
```
~/brickscan/ml/data/test_renders/
```

Each part image:
- Is a transparent PNG with the LEGO piece on a transparent background
- Can be in RGBA or RGB format
- Dimensions vary based on part size

### Synthetic Data Strategy

For each generated image:
1. **Create background** from 4 types:
   - Solid colors (white, gray, etc.)
   - Gradients
   - Noise patterns
   - Wood-like texture

2. **Composite 3-15 parts** randomly:
   - Random position (no constraint on avoiding edges)
   - Random scale (5-25% of image width)
   - Random rotation (-15 to +15 degrees)
   - Allow overlapping pieces (realistic)

3. **Generate YOLO annotations**:
   - Bounding box extracted from alpha channel
   - Normalized coordinates (center_x, center_y, width, height)
   - Single class: "lego_piece"

4. **Output structure**:
   ```
   yolo_detector/
   ├── dataset/
   │   ├── images/
   │   │   ├── train/  (5000 images)
   │   │   └── val/    (1000 images)
   │   ├── labels/
   │   │   ├── train/  (YOLO format .txt files)
   │   │   └── val/
   │   └── data.yaml   (YOLO configuration)
   ├── training/
   │   └── yolo_detector/
   │       ├── weights/  (best.pt, last.pt)
   │       ├── results.csv
   │       └── plots/
   └── models/
       ├── best.onnx
       └── training_metrics.json
   ```

## Quick Start

### 1. Full Pipeline (Recommended)

Generate data, train, and export in one command:

```bash
cd ~/brickscan/ml
python train_yolo_detector.py \
    --parts-dir ~/brickscan/ml/data/test_renders \
    --output-dir ./yolo_detector
```

**Expected time**: ~4-6 hours (data gen ~1h, training ~3-5h)

### 2. Training Only (Skip Data Generation)

If you already have a dataset:

```bash
python train_yolo_detector.py \
    --parts-dir ~/brickscan/ml/data/test_renders \
    --output-dir ./yolo_detector \
    --no-generate-data
```

### 3. Data Generation Only

To just create the synthetic dataset without training:

```bash
python train_yolo_detector.py \
    --parts-dir ~/brickscan/ml/data/test_renders \
    --output-dir ./yolo_detector \
    --no-train \
    --no-export
```

### 4. Custom Hyperparameters

Adjust training parameters:

```bash
python train_yolo_detector.py \
    --parts-dir ~/brickscan/ml/data/test_renders \
    --output-dir ./yolo_detector \
    --epochs 50 \
    --batch-size 64 \
    --patience 10 \
    --image-size 512
```

## Command Line Arguments

### Data Generation
- `--parts-dir` (default: `~/brickscan/ml/data/test_renders`)
  - Directory containing rendered LEGO part images
  
- `--output-dir` (default: `./yolo_detector`)
  - Base output directory for all results
  
- `--image-size` (default: 640)
  - Size of generated images (square, pixels)
  
- `--num-train` (default: 5000)
  - Number of training images to generate
  
- `--num-val` (default: 1000)
  - Number of validation images to generate

### Training
- `--epochs` (default: 100)
  - Total training epochs
  
- `--batch-size` (default: 32)
  - Batch size (adjust based on GPU memory)
  - For DGX Spark GB10, can use 64+ safely
  
- `--patience` (default: 15)
  - Early stopping patience (epochs without fitness improvement)
  
- `--device` (default: '0')
  - CUDA device ID (0 for first GPU, '0,1' for multiple)

### Pipeline Control
- `--no-generate-data`
  - Skip synthetic data generation
  
- `--no-train`
  - Skip model training
  
- `--no-export`
  - Skip ONNX export

## Training Details

### Model Architecture
- **Base Model**: YOLOv8m (medium)
  - ~25.9M parameters
  - Good balance between speed and accuracy
  - Suitable for GB10 GPU resources

### Augmentation Strategy
Strong augmentation applied during training:
- **Mosaic augmentation**: Combines 4 images in mosaics
- **Mixup**: Blends images at pixel level
- **HSV augmentation**: 
  - H: ±1.5%
  - S: ±70%
  - V: ±40%
- **Geometric**: 
  - Rotation: ±10°
  - Translation: ±10%
  - Scale: 0.5x to 2.0x
  - Flip: 50% horizontal and vertical

### Training Parameters
- **Learning rate**: 0.01 (starts), 0.01 (final)
- **Momentum**: 0.937
- **Weight decay**: 0.0005
- **Warmup**: 3 epochs at 0.1 initial LR
- **Optimizer**: SGD

### Stopping Criteria
- Runs for 100 epochs OR
- Stops early if validation fitness doesn't improve for 15 consecutive epochs
- Always saves best model based on validation metrics

## Understanding Training Output

### During Training
```
Epoch 1/100:
  train/box_loss:    0.234
  train/cls_loss:    0.145
  train/dfl_loss:    0.089
  val/box_loss:      0.256
  val/cls_loss:      0.167
  val/dfl_loss:      0.098
```

### Final Results
Located in: `yolo_detector/training/yolo_detector/`

Key files:
- `weights/best.pt` - Best model (for inference)
- `weights/last.pt` - Last epoch model
- `results.csv` - Epoch-by-epoch metrics
- `plots/results.png` - Training curves visualization
- `plots/confusion_matrix.png` - Confusion matrix
- `plots/val_batch0_pred.jpg` - Sample predictions

## Model Inference

### Using Trained Model

```python
from ultralytics import YOLO

# Load best model
model = YOLO('yolo_detector/training/yolo_detector/weights/best.pt')

# Predict on image
results = model.predict('test_image.jpg', conf=0.5)

# Get detections
for r in results:
    for box in r.boxes:
        print(f"Confidence: {box.conf}")
        print(f"Bbox: {box.xyxy}")  # x1, y1, x2, y2
```

### Using ONNX Model

```python
import onnx
import onnxruntime as rt

# Load ONNX model
sess = rt.InferenceSession('yolo_detector/models/best.onnx')

# Prepare input (640x640 RGB image, normalized 0-1)
# Run inference
output = sess.run(None, {'images': input_data})
```

## Troubleshooting

### Out of Memory (OOM)

If you encounter CUDA out of memory errors:

1. **Reduce batch size**:
   ```bash
   python train_yolo_detector.py ... --batch-size 16
   ```

2. **Reduce image size**:
   ```bash
   python train_yolo_detector.py ... --image-size 512
   ```

3. **Reduce number of parts per image**:
   - Modify `SyntheticDataGenerator.generate_image()` line with:
   ```python
   num_parts = random.randint(2, 10)  # Was 3-15
   ```

### Poor Detection Results

If validation metrics are poor:

1. **Increase training data**:
   ```bash
   python train_yolo_detector.py ... --num-train 10000 --num-val 2000
   ```

2. **Extend training**:
   ```bash
   python train_yolo_detector.py ... --epochs 200
   ```

3. **Check data generation**:
   - Verify parts have good transparency/contrast
   - Ensure bounding boxes are being extracted correctly
   - Look at `yolo_detector/dataset/images/train/` visually

### Slow Training

If training is slower than expected:

1. **Use larger batch size** (GB10 can handle it):
   ```bash
   python train_yolo_detector.py ... --batch-size 128
   ```

2. **Use multiple GPUs** (if available):
   ```bash
   python train_yolo_detector.py ... --device 0,1
   ```

## Advanced Customization

### Modifying Part Composition Strategy

Edit `SyntheticDataGenerator.generate_image()`:

```python
# Change number of parts range
num_parts = random.randint(5, 20)  # Was 3-15

# Increase overlap probability (currently allows all)
# Add collision detection if desired

# Adjust part scale range
scale = random.uniform(0.1, max_scale)  # Was 0.05
```

### Changing Background Types

In `BackgroundGenerator.generate()`, modify or add methods:

```python
@staticmethod
def checkerboard(size: Tuple[int, int]) -> Image.Image:
    """Generate a checkerboard pattern."""
    img = Image.new('RGB', size, (255, 255, 255))
    # ... implementation
    return img
```

### Adjusting Training Augmentation

Modify `train_yolo_model()` parameters:

```python
model.train(
    # ... existing params
    
    # Reduce augmentation
    mosaic=0.5,
    mixup=0.0,
    
    # Increase augmentation
    degrees=20,
    translate=0.2,
)
```

## Performance Benchmarks

Expected performance on DGX Spark GB10:

| Metric | Value |
|--------|-------|
| Data generation | ~1 hour (6000 images) |
| Training (100 epochs) | ~3-4 hours |
| Batch processing (inference) | ~50-100 FPS |
| Model size | ~25.9M params (98 MB) |
| ONNX export | ~2-3 minutes |

## Environment Variables

Optional environment variables:

```bash
# Use specific CUDA devices
export CUDA_VISIBLE_DEVICES=0

# Control number of threads
export OMP_NUM_THREADS=8

# Disable mixed precision (for debugging)
export YOLO_AMP=0
```

## Next Steps

1. **Evaluate Model**: Run on real LEGO images
2. **Fine-tune**: If results are poor, adjust data or training
3. **Deploy**: Use exported ONNX model in production
4. **Monitor**: Track inference latency and accuracy over time

## References

- YOLOv8 Docs: https://docs.ultralytics.com/
- YOLO Format: https://docs.ultralytics.com/datasets/detect/
- ONNX Runtime: https://onnxruntime.ai/
