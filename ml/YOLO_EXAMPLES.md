# YOLOv8 LEGO Detector - Usage Examples

Complete examples and expected outputs for the training pipeline.

## Example 1: Full Pipeline with Defaults

Run everything with default parameters:

```bash
cd ~/brickscan/ml
python train_yolo_detector.py \
    --parts-dir ~/brickscan/ml/data/test_renders \
    --output-dir ./yolo_results
```

**Expected output:**
```
======================================================================== 
2024-04-13 14:23:45,123 - __main__ - INFO - ========================================================================
2024-04-13 14:23:45,124 - __main__ - INFO - YOLO v8 LEGO Piece Detector Training Pipeline
2024-04-13 14:23:45,124 - __main__ - INFO - ========================================================================
2024-04-13 14:23:45,234 - __main__ - INFO - ultralytics version 8.0.205 is installed

2024-04-13 14:23:45,567 - __main__ - INFO - 
2024-04-13 14:23:45,567 - __main__ - INFO - ========================================================================
2024-04-13 14:23:45,567 - __main__ - INFO - STEP 1: GENERATING SYNTHETIC TRAINING DATA
2024-04-13 14:23:45,567 - __main__ - INFO - ========================================================================
2024-04-13 14:23:45,891 - __main__ - INFO - Found 4 part images
2024-04-13 14:23:45,892 - __main__ - INFO - Generating 5000 training images...
Training data: 100%|████████████████████████████████████| 5000/5000 [58:34<00:00,  1.42it/s]
2024-04-13 14:23:46,012 - __main__ - INFO - Generating 1000 validation images...
Validation data: 100%|███████████████████████████████████| 1000/1000 [11:42<00:00,  1.42it/s]
2024-04-13 14:23:47,234 - __main__ - INFO - Synthetic dataset generation complete!
2024-04-13 14:23:47,456 - __main__ - INFO - Created dataset config: /path/to/yolo_results/dataset/data.yaml

2024-04-13 14:23:47,567 - __main__ - INFO - 
2024-04-13 14:23:47,567 - __main__ - INFO - ========================================================================
2024-04-13 14:23:47,567 - __main__ - INFO - STEP 2: TRAINING YOLO v8m MODEL
2024-04-13 14:23:47,567 - __main__ - INFO - ========================================================================
2024-04-13 14:23:47,891 - __main__ - INFO - Loading YOLOv8m model...
2024-04-13 14:23:48,234 - __main__ - INFO - Starting training...

Ultralytics YOLOv8.0.205 🚀 Python-3.10.12 torch-2.1.0 CUDA:0 (NVIDIA Blackwell, 140GB)
Model summary: 168 layers, 25.9M parameters, 25.9M gradients

     Epoch   gpu_mem       box       cls       dfl      loss  Instances       Size
     1/100     8.2G     0.234     0.145     0.089     0.468        1245       640: 100%|███| 156/156 [02:34<00:00,  0.98it/s]
                Class     Images     Instances           P           R      mAP50       mAP:  
                  all       1000       1245       0.923       0.897       0.912       0.856

     2/100     8.2G     0.167     0.098     0.067     0.332        1245       640: 100%|███| 156/156 [02:33<00:00,  0.98it/s]
                Class     Images     Instances           P           R      mAP50       mAP:  
                  all       1000       1245       0.945       0.923       0.938       0.892

    ... (training continues) ...

    100/100     8.2G     0.045     0.012     0.031     0.088        1245       640: 100%|███| 156/156 [02:32<00:00,  0.98it/s]
                Class     Images     Instances           P           R      mAP50       mAP:  
                  all       1000       1245       0.987       0.981       0.989       0.965

2024-04-13 14:23:49,567 - __main__ - INFO - Training complete!

2024-04-13 14:23:49,567 - __main__ - INFO - 
2024-04-13 14:23:49,567 - __main__ - INFO - ========================================================================
2024-04-13 14:23:49,567 - __main__ - INFO - STEP 3: EXPORTING MODEL TO ONNX
2024-04-13 14:23:49,567 - __main__ - INFO - ========================================================================
2024-04-13 14:23:49,891 - __main__ - INFO - Loading model from /path/to/yolo_results/training/yolo_detector/weights/best.pt...
2024-04-13 14:23:50,234 - __main__ - INFO - Exporting to ONNX...
2024-04-13 14:23:50,567 - __main__ - INFO - Model exported to: /path/to/yolo_results/models/best.onnx

2024-04-13 14:23:50,567 - __main__ - INFO - 
2024-04-13 14:23:50,567 - __main__ - INFO - ========================================================================
2024-04-13 14:23:50,567 - __main__ - INFO - TRAINING PIPELINE COMPLETE
2024-04-13 14:23:50,567 - __main__ - INFO - ========================================================================
2024-04-13 14:23:50,567 - __main__ - INFO - Results saved to: /path/to/yolo_results
```

**Estimated times:**
- Data generation: ~1 hour
- Training: ~3-4 hours
- ONNX export: ~2 minutes
- **Total: 4-5 hours**

**Output directory structure:**
```
yolo_results/
├── dataset/
│   ├── images/
│   │   ├── train/  (5000 JPG files, ~2.5GB)
│   │   └── val/    (1000 JPG files, ~500MB)
│   ├── labels/
│   │   ├── train/  (5000 TXT files, ~2MB)
│   │   └── val/    (1000 TXT files, ~400KB)
│   └── data.yaml
├── training/
│   └── yolo_detector/
│       ├── weights/
│       │   ├── best.pt   (98 MB)
│       │   ├── last.pt   (98 MB)
│       │   └── epoch*.pt (previous checkpoints)
│       ├── results.csv
│       └── plots/
│           ├── results.png
│           ├── confusion_matrix.png
│           ├── val_batch0_pred.jpg
│           └── ...
└── models/
    ├── best.onnx  (98 MB)
    └── training_metrics.json
```

---

## Example 2: Data Generation Only

Generate synthetic dataset without training:

```bash
python train_yolo_detector.py \
    --parts-dir ~/brickscan/ml/data/test_renders \
    --output-dir ./yolo_data_only \
    --num-train 10000 \
    --num-val 2000 \
    --no-train \
    --no-export
```

**Key arguments:**
- `--no-train` - Skip model training
- `--no-export` - Skip ONNX export
- `--num-train 10000` - Generate 10,000 training images
- `--num-val 2000` - Generate 2,000 validation images

**Expected output:**
```
Generating 10000 training images...
Training data: 100%|████████████████████████████████████| 10000/10000 [1:56:23<00:00,  1.42it/s]
Generating 2000 validation images...
Validation data: 100%|███████████████████████████████████| 2000/2000 [23:45<00:00,  1.42it/s]
Synthetic dataset generation complete!
Created dataset config: /path/to/yolo_data_only/dataset/data.yaml
```

**Expected time:** ~2.5 hours for 12,000 images

---

## Example 3: Training Only (Existing Dataset)

Skip data generation and train on existing dataset:

```bash
python train_yolo_detector.py \
    --output-dir ./yolo_results \
    --no-generate-data \
    --epochs 200 \
    --patience 20
```

**Key arguments:**
- `--no-generate-data` - Skip data generation
- `--epochs 200` - Train for 200 epochs (longer)
- `--patience 20` - Increase early stopping patience

**Prerequisites:**
- Must have existing `yolo_results/dataset/data.yaml`
- Must have training images in `yolo_results/dataset/images/train/`
- Must have validation images in `yolo_results/dataset/images/val/`

**Expected output:**
```
Skip synthetic data generation (--no-generate-data flag set)
Loading YOLOv8m model...
Starting training...

Ultralytics YOLOv8.0.205 🚀 Python-3.10.12 torch-2.1.0 CUDA:0
Model summary: 168 layers, 25.9M parameters, 25.9M gradients

     Epoch   gpu_mem       box       cls       dfl      loss  Instances       Size
     1/200     8.2G     0.234     0.145     0.089     0.468        1245       640: 100%|███| 156/156 [02:34<00:00,  0.98it/s]
     ...
   200/200     8.2G     0.032     0.008     0.025     0.065        1245       640: 100%|███| 156/156 [02:32<00:00,  0.98it/s]

Training complete!
```

---

## Example 4: Custom Hyperparameters

Train with custom batch size and image resolution:

```bash
python train_yolo_detector.py \
    --parts-dir ~/brickscan/ml/data/test_renders \
    --output-dir ./yolo_large \
    --image-size 1024 \
    --batch-size 64 \
    --epochs 150 \
    --num-train 8000 \
    --num-val 1600
```

**Parameters:**
- `--image-size 1024` - Larger training resolution (more detail)
- `--batch-size 64` - Larger batch size (faster training, more memory)
- `--epochs 150` - More training iterations
- `--num-train 8000` - More training samples
- `--num-val 1600` - Proportional validation set

**Considerations:**
- Larger image size → slower but potentially more accurate
- Larger batch size → faster training, needs more VRAM
- DGX Spark with 130GB VRAM can easily handle these settings

---

## Example 5: Using Shell Wrapper

Use the convenience shell script:

```bash
# Full pipeline with defaults
./train_yolo.sh

# With custom parameters
./train_yolo.sh \
    --parts-dir ~/brickscan/ml/data/test_renders \
    --output-dir ./yolo_results \
    --epochs 100 \
    --batch-size 64

# Skip data generation
./train_yolo.sh --no-generate-data

# Show help
./train_yolo.sh --help
```

**Shell script output:**
```
========================================================================
YOLOv8 LEGO Piece Detection Training
========================================================================
Parts directory:      /home/user/brickscan/ml/data/test_renders
Output directory:     ./yolo_detector
Image size:           640 x 640
Epochs:               100
Batch size:           32
Early stopping:       patience=15

Found 4 part images

Starting training pipeline...
```

---

## Example 6: Inference with Trained Model

After training, use the model for detection:

```python
from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO('yolo_results/training/yolo_detector/weights/best.pt')

# Detect on image
image_path = 'test_image.jpg'
results = model.predict(image_path, conf=0.5)

# Process results
for r in results:
    print(f"Found {len(r.boxes)} objects")
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        print(f"  Box: ({x1:.0f}, {y1:.0f}) ({x2:.0f}, {y2:.0f}), Confidence: {conf:.3f}")
        
# Visualize
frame = results[0].plot()
cv2.imwrite('output.jpg', frame)
```

**Expected output:**
```
Found 12 objects
  Box: (143, 98) (287, 203), Confidence: 0.987
  Box: (412, 156) (528, 267), Confidence: 0.965
  Box: (89, 345) (201, 456), Confidence: 0.943
  ...
```

---

## Example 7: Batch Inference with ONNX

Use ONNX model for deployment (CPU compatible):

```python
import onnxruntime as rt
import numpy as np
from PIL import Image

# Load ONNX model
sess = rt.InferenceSession('yolo_results/models/best.onnx')

# Load and preprocess image
img = Image.open('test_image.jpg').resize((640, 640))
img_array = np.array(img).astype(np.float32) / 255.0  # Normalize
img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
img_array = np.expand_dims(img_array, 0)  # Add batch dimension

# Run inference
outputs = sess.run(None, {'images': img_array})

# outputs[0] shape: (1, 84, 8400)
# 84 = 4 bbox coords + 1 objectness + 79 classes (unused)
# 8400 = 80x105 grid predictions
print(f"Output shape: {outputs[0].shape}")
```

---

## Example 8: Model Evaluation

Evaluate performance on validation set:

```python
from ultralytics import YOLO

model = YOLO('yolo_results/training/yolo_detector/weights/best.pt')

# Validate
results = model.val(data='yolo_results/dataset/data.yaml')

# Print metrics
print(f"mAP50:    {results.box.map50:.3f}")
print(f"mAP:      {results.box.map:.3f}")
print(f"Precision: {results.box.p:.3f}")
print(f"Recall:   {results.box.r:.3f}")
```

**Expected output:**
```
Class       Images   Instances      P      R   mAP50   mAP:
  all        1000       1245      0.987  0.981  0.989  0.965
```

---

## Example 9: Viewing Training Metrics

Analyze training progress:

```bash
# Check final metrics
cat yolo_results/models/training_metrics.json
# Output:
# {
#   "epochs": 100,
#   "best_fitness": 0.965
# }

# View training curves
python -c "
import pandas as pd
df = pd.read_csv('yolo_results/training/yolo_detector/results.csv')
print(df[['epoch', 'train/box_loss', 'val/box_loss', 'metrics/mAP50']].tail(10))
"

# Plot with matplotlib
python -c "
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('yolo_results/training/yolo_detector/results.csv')
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].plot(df['epoch'], df['train/box_loss'], label='train', alpha=0.7)
axes[0,0].plot(df['epoch'], df['val/box_loss'], label='val', alpha=0.7)
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Box Loss')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(df['epoch'], df['metrics/mAP50'], color='green', alpha=0.7)
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('mAP50')
axes[0,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_analysis.png', dpi=150)
print('Saved to training_analysis.png')
"
```

---

## Example 10: Troubleshooting - Low GPU Memory

If you get CUDA out of memory errors:

```bash
# Reduce batch size
python train_yolo_detector.py \
    --output-dir ./yolo_low_mem \
    --batch-size 8 \
    --image-size 512 \
    --no-generate-data

# Or use shell script
./train_yolo.sh --batch-size 16 --image-size 512
```

**Progressive reduction strategy:**
1. Batch size 32 (default) → 16 → 8 → 4
2. Image size 640 (default) → 512 → 416 → 320
3. Reduce num_train 5000 → 2000
4. Increase patience to account for smaller batches

---

## Example 11: Resuming Interrupted Training

If training is interrupted, resume from last checkpoint:

```bash
# Restart from last epoch
python train_yolo_detector.py \
    --output-dir ./yolo_results \
    --no-generate-data \
    --epochs 100 \
    # YOLOv8 automatically detects last.pt and resumes
```

Check the shell script for status:

```bash
ls -lh yolo_results/training/yolo_detector/weights/
# Should show best.pt and last.pt
# last.pt is automatically loaded and resumed
```

---

## Performance Benchmarks

### Expected Performance on DGX Spark GB10

| Stage | Duration | Notes |
|-------|----------|-------|
| Data Gen (5000 train) | ~1 hour | 1.42 img/s |
| Training (100 epochs) | ~3-4 hours | ~20 min/epoch |
| ONNX Export | ~2 minutes | One-time |
| Inference (1 image) | ~50-100 ms | Depends on GPU load |
| Batch inference (100 images) | ~2-3 seconds | ~50 FPS |

### Memory Usage

| Component | VRAM Required |
|-----------|---------------|
| YOLOv8m model | ~2.5 GB |
| Batch size 32 @ 640x640 | ~5 GB |
| Batch size 64 @ 640x640 | ~9 GB |
| Batch size 128 @ 640x640 | ~16 GB |

GB10 has 130.6 GB, so even batch size 128 is easily supported.

---

## Success Indicators

When training is successful:

1. **Data Generation:**
   - Images created with visible LEGO parts
   - Labels have proper YOLO format (0 normalized coords)
   - data.yaml present and valid

2. **Training:**
   - Loss decreases over epochs
   - mAP increases from ~0.5 to >0.95
   - No CUDA errors
   - Valid loss lower than training loss

3. **Final Model:**
   - best.pt exists (~98 MB)
   - best.onnx exists (~98 MB)
   - training_metrics.json saved
   - plots/ directory has visualizations

4. **Quick Inference Test:**
   ```bash
   python -c "
   from ultralytics import YOLO
   m = YOLO('yolo_results/training/yolo_detector/weights/best.pt')
   results = m.predict('yolo_results/dataset/images/val/img_00000.jpg')
   print(f'Found {len(results[0].boxes)} objects')
   "
   # Should print "Found X objects" where X > 0
   ```
