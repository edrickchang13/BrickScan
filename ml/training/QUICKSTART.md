# Quick Start Guide

## One-liner Training (with defaults)

```bash
cd /mnt/Lego/brickscan/ml/training
./train.sh
```

## Step-by-Step Manual

### 1. Install dependencies
```bash
pip install -r training/requirements.txt
```

### 2. Train model
```bash
python training/train.py \
  --data-dir /mnt/Lego/brickscan/ml/data \
  --output-dir /mnt/Lego/brickscan/ml/models \
  --epochs 50 \
  --batch-size 64
```

### 3. Export to ONNX
```bash
python training/export_onnx.py \
  --checkpoint /mnt/Lego/brickscan/ml/models/best_model.pt \
  --output /mnt/Lego/brickscan/ml/models/lego_classifier.onnx \
  --labels-dir /mnt/Lego/brickscan/ml/models
```

### 4. Evaluate
```bash
python training/evaluate.py \
  --checkpoint /mnt/Lego/brickscan/ml/models/best_model.pt \
  --data-dir /mnt/Lego/brickscan/ml/data \
  --output-dir /mnt/Lego/brickscan/ml/models
```

## Output Files

After training and export, you'll have:

- `models/best_model.pt` - Best checkpoint
- `models/lego_classifier.onnx` - ONNX model for FastAPI
- `models/part_labels.json` - Part number to index mapping
- `models/color_labels.json` - Color name to index mapping
- `models/logs/` - TensorBoard logs
- `models/evaluation_results.json` - Evaluation metrics
- `models/confusion_matrix_top20_parts.png` - Confusion matrix

## Monitor Training

```bash
tensorboard --logdir /mnt/Lego/brickscan/ml/models/logs
# Open http://localhost:6006
```

## Resume Training

```bash
python training/train.py \
  --data-dir /mnt/Lego/brickscan/ml/data \
  --output-dir /mnt/Lego/brickscan/ml/models \
  --epochs 100 \
  --resume /mnt/Lego/brickscan/ml/models/checkpoint_epoch_50.pt
```

## FastAPI Integration

Load ONNX model:
```python
import onnxruntime as ort
import json
import numpy as np
from scipy.special import softmax

# Load
sess = ort.InferenceSession('/path/to/lego_classifier.onnx')
part_labels = json.load(open('/path/to/part_labels.json'))
color_labels = json.load(open('/path/to/color_labels.json'))

# Infer (image_tensor shape: 1, 3, 224, 224)
outputs = sess.run(None, {'input': image_tensor})
logits = outputs[0][0]

num_parts = len(part_labels)
part_logits = logits[:num_parts]
color_logits = logits[num_parts:]

# Get top-3 parts
part_probs = softmax(part_logits)
top3_idx = np.argsort(-part_probs)[:3]
part_labels_inv = {v: k for k, v in part_labels.items()}
predictions = [(part_labels_inv[i], float(part_probs[i])) for i in top3_idx]

# Get color
color_probs = softmax(color_logits)
color_idx = np.argmax(color_probs)
color_labels_inv = {v: k for k, v in color_labels.items()}
color = (color_labels_inv[color_idx], float(color_probs[color_idx]))
```

## Hyperparameter Tuning

Start with defaults, then adjust:
- `--lr`: Learning rate. Try 1e-5, 5e-5, 1e-4 (default), 5e-4
- `--batch-size`: Batch size. Try 32, 64 (default), 128 (if GPU memory allows)
- `--epochs`: Training epochs. Default 50, try up to 100
- `--val-split`: Validation split. Default 0.15, try 0.1-0.2

## Troubleshooting

**GPU out of memory:**
```bash
python training/train.py --batch-size 32 ...
```

**Poor accuracy:**
- Ensure data in `renders/index.csv` is correct
- Try longer training: `--epochs 100`
- Check TensorBoard logs for training curves

**Export fails:**
- Verify checkpoint path is correct
- Check that label JSON files exist
- Ensure onnxruntime is installed: `pip install onnxruntime-gpu`
