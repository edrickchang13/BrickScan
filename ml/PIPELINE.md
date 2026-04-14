# BrickScan ML Training Pipeline

Complete, production-ready ML training pipeline for LEGO piece recognition. Trains a vision model to identify 3,000 LEGO parts with export to Core ML (iOS) and ONNX (backend inference).

## Architecture

```
brickscan/ml/
├── requirements.txt                    # Python dependencies
├── training/                           # Training code
│   ├── config.yaml                    # Training configuration
│   ├── dataset.py                     # PyTorch Dataset class
│   ├── model.py                       # EfficientNet model
│   ├── train.py                       # Training script
│   └── evaluate.py                    # Evaluation script
├── export/                            # Model export
│   ├── to_coreml.py                  # Export to Core ML (iOS)
│   └── to_onnx.py                    # Export to ONNX (backend)
├── inference/                         # Inference code
│   └── predictor.py                  # ONNX inference class
└── data_pipeline/                     # Data processing
    ├── ldraw_renderer.py             # Blender synthetic rendering
    ├── augmentation.py               # Data augmentation
    └── rebrickable_images.py         # Download Rebrickable images
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Training Data

Dataset structure:
```
data/
├── train/{part_num}/{image.jpg}
├── val/{part_num}/{image.jpg}
└── test/{part_num}/{image.jpg}
```

**Option A: Download from Rebrickable** (3,000 most common parts)
```bash
export REBRICKABLE_API_KEY="your-api-key"
python data_pipeline/rebrickable_images.py \
    --output_dir ./data/catalog_images \
    --max_parts 3000
```

**Option B: Render synthetic data with Blender**
```bash
blender --background --python data_pipeline/ldraw_renderer.py -- \
    --parts_dir /path/to/ldraw/parts \
    --output_dir ./data/renders \
    --num_renders 50
```

**Option C: Augment existing dataset** (synthetic-to-real domain adaptation)
```bash
python data_pipeline/augmentation.py \
    --input_dir ./data/renders \
    --output_dir ./data/augmented \
    --multiplier 3 \
    --synthetic_to_real
```

### 3. Train Model

```bash
cd training
python train.py --config config.yaml
```

Monitor with Weights & Biases (optional):
```bash
export WANDB_ENTITY="your-username"
# Then run train.py
```

**Training details:**
- Model: EfficientNet-B4 (pretrained on ImageNet)
- Classes: 3,000 LEGO parts
- Batch size: 64
- Epochs: 100 (with early stopping after 15 epochs no improvement)
- Optimizer: AdamW with cosine annealing + warmup
- Input size: 224x224
- Label smoothing: 0.1
- Data augmentation: horizontal flip, rotation, brightness/contrast, GaussNoise, perspective

### 4. Evaluate Model

```bash
cd training
python evaluate.py \
    --checkpoint ./checkpoints/best_model.pt \
    --config config.yaml \
    --output_dir ./eval_results
```

Generates:
- Overall accuracy metrics (top-1, top-5)
- Per-class accuracy breakdown (easiest/hardest parts)
- Confusion matrix for most confused pairs
- Confidence calibration analysis
- Visualizations (histograms, calibration curves)

### 5. Export for Deployment

**To Core ML (iOS app):**
```bash
python export/to_coreml.py \
    --checkpoint ./training/checkpoints/best_model.pt \
    --output ./models/BrickScan.mlpackage
```

Generates:
- `BrickScan.mlpackage/` - Core ML model for iOS 16+
- `class_mapping.json` - Part number to class index mapping

**To ONNX (backend service):**
```bash
python export/to_onnx.py \
    --checkpoint ./training/checkpoints/best_model.pt \
    --output ./models/brickscan.onnx
```

Generates:
- `brickscan.onnx` - ONNX inference model
- `class_mapping.json` - Class mapping
- `model_info.json` - Model metadata

### 6. Run Inference

```python
from inference.predictor import LegoPredictor

# Initialize
predictor = LegoPredictor(
    model_path="./models/brickscan.onnx",
    class_map_path="./models/class_mapping.json",
    top_k=5,
)

# Load image
with open("path/to/lego_piece.jpg", "rb") as f:
    image_bytes = f.read()

# Predict
predictions = predictor.predict(image_bytes)

# Print results
for pred in predictions:
    print(f"{pred['part_num']}: {pred['confidence']:.2%}")

# Or with more details
detailed = predictor.predict_with_details(image_bytes)
for pred in detailed:
    print(f"Rank {pred['rank']}: {pred['part_num']} "
          f"({pred['confidence']:.2%}, logit: {pred['logit']:.2f})")

# Batch inference
batch_predictions = predictor.batch_predict([image1_bytes, image2_bytes, ...])
```

## Configuration

Edit `training/config.yaml` to customize:

```yaml
model:
  architecture: efficientnet_b4        # timm architecture
  num_classes: 3000                   # LEGO parts to classify
  input_size: 224                     # Input image size
  dropout: 0.3                        # Dropout rate

training:
  epochs: 100                         # Max epochs
  batch_size: 64                      # Batch size
  learning_rate: 0.001                # Initial learning rate
  weight_decay: 0.01                  # L2 regularization
  warmup_epochs: 5                    # Warmup phase
  label_smoothing: 0.1                # Label smoothing

augmentation:
  train:
    random_horizontal_flip: true
    random_rotation: 45
    color_jitter: {brightness: 0.3, contrast: 0.3}
    gaussian_blur: 0.1
```

## Data Augmentation

### Training Transform (augmentation.py)
- Resize + random crop
- Horizontal flip
- Rotation (up to 45 degrees)
- Brightness/contrast jitter
- Hue/saturation shift
- Gaussian noise
- Gaussian blur
- Perspective distortion
- ImageNet normalization

### Synthetic-to-Real Transform
Makes synthetic renders look like real photos:
- Random brightness/contrast
- Hue/saturation variation
- Gaussian noise (camera noise simulation)
- Image compression (JPEG artifacts)
- Shadows and occlusions
- Perspective distortion

## Model Details

### Backbone: EfficientNet-B4
- Pretrained on ImageNet
- 388M parameters
- Efficient scaling (compound model scaling)

### Classifier Head
```
Dropout(0.3)
└─ Linear(1280 -> 512)
   └─ ReLU
      └─ Dropout(0.2)
         └─ Linear(512 -> 3000)
```

### Embeddings
Use `.get_embeddings(x)` to get 512-dimensional features for:
- Nearest-neighbor search
- Similarity learning
- Clustering analysis

## Outputs

### Training Checkpoint
`checkpoints/best_model.pt` contains:
```python
{
    'model_state_dict': {...},      # Model weights
    'optimizer_state_dict': {...},  # Optimizer state
    'scheduler_state_dict': {...},  # LR scheduler state
    'config': {...},                # Training config
    'class_to_idx': {...},          # Part mapping
    'metrics': {...},               # Validation metrics
    'epoch': 42,                    # Training epoch
}
```

### Evaluation Report
`eval_results/evaluation_metrics.json` contains:
```python
{
    'overall_accuracy': 0.92,
    'top5_accuracy': 0.98,
    'per_class_accuracy': {
        '3001': 0.95,
        '3002': 0.88,
        ...
    },
    'easiest_classes': [('3001', 0.99), ...],
    'hardest_classes': [('3045', 0.45), ...],
    'top_confused_pairs': [
        {'from_part': '3001', 'to_part': '3002', 'count': 15},
        ...
    ],
    'calibration': {...}
}
```

Plus plots:
- `per_class_accuracy_hist.png` - Distribution of per-class accuracy
- `calibration.png` - Confidence calibration curve
- `easiest_hardest.png` - Top-10 easiest/hardest classes

## Performance

Expected performance (on large diverse LEGO dataset):
- **Top-1 Accuracy**: 88-92%
- **Top-5 Accuracy**: 96-99%
- **Inference Time** (ONNX, batch=1): 50-100ms (CPU)
- **Model Size**: ~160 MB (PyTorch), ~85 MB (ONNX), ~45 MB (Core ML)

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config.yaml
- Use mixed precision training (add `amp: true` to config)
- Use smaller architecture (e.g., efficientnet_b2)

### Low Accuracy
- Check data quality and split
- Increase training epochs
- Reduce learning rate
- Add more data (especially for hardest classes)
- Check class imbalance (use weighted sampling)

### Slow Training
- Increase `num_workers` in config.yaml
- Use larger batch size (if GPU memory allows)
- Use smaller input size temporarily

## Dependencies

```
torch==2.1.2
torchvision==0.16.2
timm==0.9.12                         # Model zoo
coremltools==7.0                     # iOS export
onnx==1.15.0
onnxruntime==1.17.0
albumentations==1.3.1                # Data augmentation
opencv-python-headless==4.8.1.78
numpy==1.24.3
pandas==2.1.4
scikit-learn==1.3.2                  # Metrics
matplotlib==3.8.2
seaborn==0.13.0
wandb==0.16.1                        # Experiment tracking
PyYAML==6.0.1
httpx==0.25.2                        # Async HTTP for Rebrickable
```

## Production Checklist

- [ ] Train on full 3,000 part dataset (or top N parts by frequency)
- [ ] Validate on held-out test set
- [ ] Analyze failure modes (confusion matrix)
- [ ] Check confidence calibration
- [ ] Export to ONNX and Core ML
- [ ] Test inference on iOS app
- [ ] Benchmark inference latency
- [ ] Set up monitoring for predictions in production
- [ ] Create fallback strategy (what to do on low confidence?)
- [ ] Document model limitations and failure modes

## License

Internal use only. LEGO is a trademark of the LEGO Group.
