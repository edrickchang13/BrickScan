================================================================================
BRICKSCAN ML TRAINING SCRIPTS
================================================================================

Two production-ready training scripts for LEGO brick recognition on DGX Spark.

REQUIREMENTS:
- NVIDIA DGX Spark: GB10 Blackwell, 130GB VRAM, CUDA 13.0, PyTorch 2.12.0
- Architecture: aarch64
- Training data: 400K images, 1000 LEGO part classes
  Location: ~/brickscan/ml/training_data/huggingface_legobricks/images/{class_id}/*.jpg

PACKAGE DEPENDENCIES:
pip install timm pillow coremltools scikit-learn torch torchvision tqdm

================================================================================
SCRIPT 1: train_contrastive.py
================================================================================

Purpose:
  Multi-view contrastive learning (SimCLR-style) to solve the brick vs plate
  confusion problem. Learns embeddings where the same part from different
  angles clusters together.

Architecture:
  - Backbone: DINOv2 ViT-B/14 (pretrained from timm)
  - Projection head: Linear(768, 512) -> BN -> ReLU -> Linear(512, 128)
  - Loss: NT-Xent (normalized temperature-scaled cross entropy)

Key Features:
  - Heavy augmentation pipeline (crop, color jitter, rotation 0-360°, blur)
  - Hard negative mining for brick/plate pairs
  - k-NN evaluation on learned embeddings
  - ONNX export for inference

Training Hyperparameters:
  - Batch size: 256 (adjustable with --batch-size)
  - Epochs: 100 (adjustable with --epochs)
  - Learning rate: 1e-4 (adjustable with --lr)
  - Temperature: 0.07 (adjustable with --temperature)
  - Optimizer: AdamW with cosine annealing
  - Checkpoint interval: every 10 epochs

Usage:
  python3 train_contrastive.py \
    --data-dir ~/brickscan/ml/training_data/huggingface_legobricks/images \
    --output-dir ~/brickscan/ml/output/contrastive \
    --batch-size 256 \
    --epochs 100 \
    --lr 1e-4

Output:
  - Checkpoints: ~/brickscan/ml/output/contrastive/YYYYMMDD/checkpoints/
  - ONNX export: ~/brickscan/ml/output/contrastive/YYYYMMDD/exports/
  - Logs: ~/brickscan/ml/logs/train_contrastive_TIMESTAMP.log

================================================================================
SCRIPT 2: train_distillation.py
================================================================================

Purpose:
  Knowledge distillation of large DINOv2 ViT-B/14 into lightweight
  MobileNetV3-Small for on-device CoreML inference (iPhone).

Architecture:
  - Teacher: DINOv2 ViT-B/14 (from train_contrastive.py checkpoint)
  - Student: MobileNetV3-Small (1000-class head)
  - Feature adapter: Learned projection to align student->teacher features

Three-Component Loss:
  1. CE loss (weight 0.3): student predictions vs ground truth
  2. KL divergence (weight 0.5): soft labels (T=4)
  3. Feature MSE (weight 0.2): intermediate layer alignment

Progressive Training:
  - Epochs 1-20: Feature alignment only
  - Epochs 21-50: All three losses combined

Data:
  - Student input: 224x224 (MobileNet native)
  - Teacher input: 518x518 (DINOv2 native)
  - Resized independently for each model

Hyperparameters:
  - Batch size: 128 (adjustable with --batch-size)
  - Epochs: 50 (adjustable with --epochs)
  - Learning rate: 5e-4 (adjustable with --lr)
  - Temperature: 4.0 (adjustable with --temperature)
  - Early stopping patience: 10 epochs
  - Optimizer: AdamW with cosine annealing

Usage:
  python3 train_distillation.py \
    --data-dir ~/brickscan/ml/training_data/huggingface_legobricks/images \
    --teacher-checkpoint ~/brickscan/ml/output/contrastive/YYYYMMDD/checkpoints/checkpoint_epoch_099.pt \
    --output-dir ~/brickscan/ml/output/distillation \
    --batch-size 128 \
    --epochs 50

Output:
  - Checkpoints: ~/brickscan/ml/output/distillation/YYYYMMDD/checkpoints/
  - ONNX export: ~/brickscan/ml/output/distillation/YYYYMMDD/exports/
  - CoreML export: ~/brickscan/ml/output/distillation/YYYYMMDD/exports/LEGOBrickClassifier.mlmodel
  - Logs: ~/brickscan/ml/logs/train_distillation_TIMESTAMP.log

================================================================================
TYPICAL WORKFLOW
================================================================================

1. Train contrastive model (100 epochs):
   python3 train_contrastive.py --epochs 100 --batch-size 256

2. Evaluate k-NN accuracy and export ONNX embedding model

3. Distill to MobileNetV3 for on-device inference:
   python3 train_distillation.py \
     --teacher-checkpoint ~/brickscan/ml/output/contrastive/YYYYMMDD/checkpoints/checkpoint_epoch_099.pt \
     --epochs 50

4. Deploy CoreML model to iOS app

================================================================================
ENVIRONMENT NOTES
================================================================================

- Both scripts use mixed precision training (AMP) for efficiency
- Gradient clipping (max_norm=1.0) for stability
- Data loading optimized for aarch64 with CUDA 13.0
- Automatic package dependency checking at startup
- Comprehensive logging to file + console
- Early stopping available in distillation script

Configuration:
  - CUDA device selection: --device cuda (default if available)
  - Data loader workers: --num-workers 8 (adjustable)
  - Device: aarch64 Linux with CUDA 13.0

================================================================================
