# LEGO Brick Classifier Training Pipeline

Complete PyTorch ML training pipeline for LEGO brick part + color classification.

## Overview

This pipeline trains a dual-head EfficientNet-B3 classifier that simultaneously predicts:
- **LEGO part types** (classification task)
- **LEGO colors** (classification task)

The trained model is exported to ONNX format for inference on a FastAPI backend.

## Architecture

```
Input: 224x224 RGB image (ImageNet normalized)
         |
    Backbone: EfficientNet-B3 (pretrained on ImageNet)
         |
    GlobalAvgPool → 1536 features
         |
    ├─ Part Head:  Linear(1536 → 512 → num_parts)   + ReLU + Dropout(0.4)
    └─ Color Head: Linear(1536 → 256 → num_colors)  + ReLU + Dropout(0.3)
         |
    Training output: (part_logits, color_logits)
    ONNX output: Concatenated [part_logits | color_logits]
```

## Files

### Core Training Files

- **`dataset.py`**: PyTorch Dataset class and label encoder utilities
  - `LegoPartsDataset`: Loads images and applies augmentations
  - `build_label_encoders()`: Creates part→idx and color→idx mappings
  - `save_label_encoders()`: Saves encoders to JSON
  - Augmentations: random crop, horizontal flip, color jitter, rotation, normalization

- **`model.py`**: Dual-head EfficientNet-B3 classifier
  - `LegoBrickClassifier`: Main model class
  - `forward()`: Returns (part_logits, color_logits) for training
  - `forward_onnx()`: Returns concatenated logits for ONNX export

- **`train.py`**: Full training script with mixed precision and checkpointing
  - `Trainer` class handles training/validation loops
  - Features: mixed precision (AMP), gradient clipping, OneCycleLR scheduler
  - Early stopping with patience=10
  - TensorBoard logging (loss, accuracies, learning rate)
  - Checkpoint saving every 5 epochs + best model tracking
  - Combined loss: `0.6 * part_loss + 0.4 * color_loss`

- **`export_onnx.py`**: Export trained checkpoint to ONNX format
  - Validates ONNX model integrity with `onnx.checker`
  - Tests inference with onnxruntime
  - Prints output shape and class counts

- **`evaluate.py`**: Full evaluation on test set
  - Part top-1 and top-5 accuracy
  - Color top-1 accuracy
  - Confusion matrix for top-20 most common parts (PNG)
  - Results saved to JSON

### Configuration Files

- **`requirements.txt`**: Python dependencies
- **`train.sh`**: Bash launcher script with GPU setup
- **`__init__.py`**: Package initialization

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r training/requirements.txt
```

### 2. Prepare Data

Data should be organized as:
```
data/
  renders/
    index.csv              # CSV with: image_path, part_num, color_id, color_name, color_r, color_g, color_b
    image1.png
    image2.png
    ...
models/                    # Output directory for models
```

## Training

### Quick Start

```bash
# Using launcher script
cd training
./train.sh
```

### Manual Training

```bash
python train.py \
  --data-dir /path/to/data \
  --output-dir /path/to/models \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-4 \
  --workers 4 \
  --val-split 0.15
```

### Resume Training

```bash
python train.py \
  --data-dir /path/to/data \
  --output-dir /path/to/models \
  --epochs 50 \
  --resume /path/to/models/checkpoint_epoch_25.pt
```

### Training Options

- `--data-dir`: Path to data directory containing `renders/`
- `--output-dir`: Where to save models, logs, and label encoders
- `--epochs`: Total epochs (default: 50)
- `--batch-size`: Training batch size (default: 64)
- `--lr`: Learning rate (default: 1e-4)
- `--workers`: DataLoader workers (default: 4)
- `--val-split`: Validation split ratio (default: 0.15)
- `--resume`: Optional checkpoint to resume from

### Outputs

After training, you'll find:

```
models/
  logs/                          # TensorBoard logs
    events.out.tfevents.xxx
  best_model.pt                  # Best checkpoint by validation loss
  checkpoint_epoch_5.pt          # Checkpoint every 5 epochs
  checkpoint_epoch_10.pt
  ...
  part_labels.json               # Part encoder (part_num → index)
  color_labels.json              # Color encoder (color_name → index)
```

### Monitor Training

```bash
tensorboard --logdir models/logs
# Open http://localhost:6006 in browser
```

## Exporting to ONNX

### Export Command

```bash
python export_onnx.py \
  --checkpoint models/best_model.pt \
  --output models/lego_classifier.onnx \
  --labels-dir models
```

### Verify Export

The script automatically:
- Validates ONNX model integrity
- Tests inference with onnxruntime
- Prints output shape and class counts

Example output:
```
Output shape: (1, 152)          # num_parts + num_colors
Number of part classes: 95
Number of color classes: 57
```

## Evaluation

### Run Evaluation

```bash
python evaluate.py \
  --checkpoint models/best_model.pt \
  --data-dir /path/to/data \
  --output-dir models \
  --batch-size 128 \
  --workers 4
```

### Outputs

```
models/
  evaluation_results.json                    # Top-1/Top-5 accuracies
  confusion_matrix_top20_parts.png           # Confusion matrix visualization
```

## Integration with FastAPI Backend

The ONNX model outputs a single tensor of shape `[batch, num_parts + num_colors]` containing concatenated logits:

```python
# In your FastAPI inference service
import onnxruntime as ort
import json

# Load model and labels
sess = ort.InferenceSession('lego_classifier.onnx')
with open('part_labels.json') as f:
    part_labels = json.load(f)
with open('color_labels.json') as f:
    color_labels = json.load(f)

num_parts = len(part_labels)
part_idx_to_label = {v: k for k, v in part_labels.items()}
color_idx_to_label = {v: k for k, v in color_labels.items()}

# Inference
outputs = sess.run(None, {'input': image_tensor})  # shape: (1, num_parts + num_colors)
logits = outputs[0][0]

# Split logits
part_logits = logits[:num_parts]
color_logits = logits[num_parts:]

# Get predictions
part_probs = softmax(part_logits)
color_probs = softmax(color_logits)

# Top-3 parts
top_3_part_indices = np.argsort(-part_probs)[:3]
top_3_parts = [(part_idx_to_label[i], float(part_probs[i])) for i in top_3_part_indices]

# Best color
best_color_idx = np.argmax(color_probs)
best_color = (color_idx_to_label[best_color_idx], float(color_probs[best_color_idx]))
```

## Training Details

### Data Augmentation

**Training:**
- RandomResizedCrop(224, scale=(0.8, 1.0))
- RandomHorizontalFlip(p=0.5)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
- RandomRotation(degrees=15)
- Normalize (ImageNet mean/std)

**Validation/Test:**
- Resize(256)
- CenterCrop(224)
- Normalize (ImageNet mean/std)

### Loss Function

Combined loss with weighted heads:
```
loss = 0.6 * CrossEntropyLoss(part_logits, part_labels) +
       0.4 * CrossEntropyLoss(color_logits, color_labels)
```

The 0.6/0.4 weighting prioritizes part classification (harder task with more classes).

### Optimization

- **Optimizer**: AdamW with weight_decay=1e-5
- **Scheduler**: OneCycleLR with pct_start=0.1
- **Precision**: Mixed precision (torch.cuda.amp) for faster training
- **Gradient clipping**: max_norm=1.0

### Early Stopping

- Patience: 10 epochs without improvement in validation loss
- Metric: validation loss (equally weighted parts and colors)

## Performance Expectations

Typical metrics on a well-balanced dataset with 50 epochs:

- **Part Top-1 Accuracy**: 85-92% (depending on number of classes)
- **Part Top-5 Accuracy**: 94-98%
- **Color Top-1 Accuracy**: 90-96%
- **Training time**: 2-4 hours on NVIDIA GPU (batch size 64)

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size` (try 32 or 16)
- Use gradient accumulation (requires code modification)

### Poor Validation Accuracy
- Check data augmentation (disable if too aggressive)
- Increase `--epochs`
- Reduce `--lr` if training is unstable
- Ensure `--val-split` is large enough (15-20% recommended)

### Label Encoder Issues
- Verify `renders/index.csv` has unique `part_num` and `color_name` values
- Check that all image paths in CSV actually exist
- Run `build_label_encoders()` standalone to debug

## File Size Reference

- Checkpoint file: ~150-200 MB
- ONNX model: ~100-150 MB
- TensorBoard logs: 1-10 MB per epoch
- Evaluation results: <1 MB

## License & Notes

- Uses pretrained EfficientNet-B3 from ImageNet1K
- Designed for LEGO BrickScan iOS app inference
- All code is production-ready and fully typed
