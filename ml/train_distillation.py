#!/usr/bin/env python3
"""
Knowledge Distillation: DINOv2 ViT-B/14 -> MobileNetV3-Small

This script distills a large trained DINOv2 model into a lightweight MobileNetV3-Small
that can run on-device in CoreML without network calls.

Three-component loss:
1. CE loss: student predictions vs ground truth (weight 0.3)
2. KL divergence: soft labels (teacher softmax T=4 vs student softmax T=4) (weight 0.5)
3. Feature MSE: intermediate layer alignment with learned adapter (weight 0.2)

Progressive training:
- Epochs 1-20: feature alignment only
- Epochs 21-50: all three losses combined
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small

# Check and report required packages
REQUIRED_PACKAGES = ['timm', 'pillow', 'coremltools', 'scikit-learn']
MISSING_PACKAGES = []

for pkg in REQUIRED_PACKAGES:
    try:
        __import__(pkg)
    except ImportError:
        MISSING_PACKAGES.append(pkg)

if MISSING_PACKAGES:
    print(f"\nMissing required packages: {', '.join(MISSING_PACKAGES)}")
    print("\nInstall with:")
    print(f"  pip install {' '.join(MISSING_PACKAGES)}")
    sys.exit(1)

import timm
from PIL import Image
from tqdm import tqdm
import coremltools as ct


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_distillation_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


# ============================================================================
# Dataset & Data Loading
# ============================================================================

class LegoBricksDataset(Dataset):
    """Loads LEGO brick images organized by class in subdirectories."""

    def __init__(self, data_dir: str, image_size: int = 224, split: str = 'train',
                 train_ratio: float = 0.8):
        """
        Args:
            data_dir: Path to directory containing class subdirectories
            image_size: Input image size
            split: 'train' or 'val'
            train_ratio: Fraction of data for training
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.split = split

        # Build image list
        self.images = []
        self.class_to_idx = {}

        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_id = class_dir.name
            if class_id not in self.class_to_idx:
                self.class_to_idx[class_id] = len(self.class_to_idx)

            for img_path in class_dir.glob('*.jpg'):
                self.images.append((str(img_path), class_id))

        # Train/val split
        np.random.seed(42)
        indices = np.arange(len(self.images))
        np.random.shuffle(indices)

        split_idx = int(len(indices) * train_ratio)

        if split == 'train':
            indices = indices[:split_idx]
        else:
            indices = indices[split_idx:]

        self.images = [self.images[i] for i in indices]
        self.num_classes = len(self.class_to_idx)

    def _get_transform(self):
        """Data augmentation for training, validation uses only normalization."""
        if self.split == 'train':
            return T.Compose([
                T.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomRotation(degrees=15),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, class_id = self.images[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.warning(f"Failed to load {img_path}: {e}")
            image = Image.new('RGB', (self.image_size, self.image_size))

        transform = self._get_transform()
        image = transform(image)

        class_idx = self.class_to_idx[class_id]

        return {
            'image': image,
            'label': class_idx,
            'class_id': class_id
        }


# ============================================================================
# Teacher & Student Models
# ============================================================================

class TeacherModel(nn.Module):
    """
    Teacher: Trained DINOv2 ViT-B/14 with 1000-class classification head.
    Loads from checkpoint.
    """

    def __init__(self, checkpoint_path: str, num_classes: int, device):
        super().__init__()

        # Load backbone
        self.backbone = timm.create_model('vit_base_patch14_dinov2', pretrained=True)
        backbone_dim = self.backbone.embed_dim

        # Classification head
        self.classifier = nn.Linear(backbone_dim, num_classes)

        # Load checkpoint
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.load_state_dict(checkpoint, strict=False)
            logging.info(f"Loaded teacher checkpoint: {checkpoint_path}")
        else:
            logging.warning(f"Teacher checkpoint not found: {checkpoint_path}")

        # Register hook to capture intermediate features
        self.intermediate_features = None

        def hook_fn(module, input, output):
            if isinstance(output, dict):
                self.intermediate_features = output.get('x', output)
            else:
                self.intermediate_features = output

        self.backbone.register_forward_hook(hook_fn)

    def forward(self, x, return_features=False):
        """
        Args:
            x: Input image [B, 3, 518, 518]
            return_features: If True, also return intermediate features

        Returns:
            logits [B, num_classes] or (logits, features)
        """
        features = self.backbone.forward_features(x)

        if isinstance(features, dict):
            features = features.get('x', features.get('cls_token', features))

        if features.dim() == 3:
            features = features[:, 0, :]

        logits = self.classifier(features)

        if return_features:
            return logits, features

        return logits

    def freeze(self):
        """Freeze teacher parameters."""
        for param in self.parameters():
            param.requires_grad = False


class StudentModel(nn.Module):
    """
    Student: MobileNetV3-Small with 1000-class head.
    Lightweight model for on-device deployment.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # Load pretrained MobileNetV3-Small
        self.backbone = mobilenet_v3_small(pretrained=True)
        backbone_dim = self.backbone.classifier[0].in_features

        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x, return_features=False):
        """
        Args:
            x: Input image [B, 3, 224, 224]
            return_features: If True, also return intermediate features

        Returns:
            logits [B, num_classes] or (logits, features)
        """
        # Extract features before classifier
        features = self.backbone.features(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)

        logits = self.backbone.classifier(features)

        if return_features:
            return logits, features

        return logits


# ============================================================================
# Feature Alignment Adapter
# ============================================================================

class FeatureAdapter(nn.Module):
    """Learned adapter to map student features to teacher feature space."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(student_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, teacher_dim)
        )

    def forward(self, x):
        return self.adapter(x)


# ============================================================================
# Loss Functions
# ============================================================================

class DistillationLoss(nn.Module):
    """Three-component distillation loss."""

    def __init__(self, temperature: float = 4.0, alpha_ce: float = 0.3,
                 alpha_kd: float = 0.5, alpha_feat: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_kd = alpha_kd
        self.alpha_feat = alpha_feat

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, student_logits, teacher_logits, student_features,
                teacher_features, labels, use_feature_loss: bool = True):
        """
        Args:
            student_logits: [B, num_classes]
            teacher_logits: [B, num_classes]
            student_features: [B, student_dim]
            teacher_features: [B, teacher_dim]
            labels: [B]
            use_feature_loss: If False, skip feature alignment loss

        Returns:
            Total loss
        """
        # Component 1: CE loss on student predictions
        ce_loss = self.ce_loss(student_logits, labels)

        # Component 2: KL divergence on soft labels
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')

        # Component 3: Feature alignment MSE (optional)
        if use_feature_loss:
            feat_loss = self.mse_loss(student_features, teacher_features)
        else:
            feat_loss = torch.tensor(0.0, device=student_logits.device)

        # Weighted sum
        total_loss = (self.alpha_ce * ce_loss +
                     self.alpha_kd * kd_loss +
                     self.alpha_feat * feat_loss)

        return {
            'total': total_loss,
            'ce': ce_loss,
            'kd': kd_loss,
            'feat': feat_loss
        }


# ============================================================================
# Training & Validation
# ============================================================================

def train_epoch(student, teacher, adapter, train_loader, optimizer, scaler,
                loss_fn, device, logger, epoch, num_epochs, use_feature_loss=True):
    """Single training epoch."""
    student.train()
    teacher.eval()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

    with torch.no_grad():
        teacher.eval()

    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        with autocast():
            # Student forward (224x224)
            student_logits, student_features = student(images, return_features=True)

            # Teacher forward (need to resize to 518x518)
            teacher_images = F.interpolate(images, size=(518, 518), mode='bilinear')
            with torch.no_grad():
                teacher_logits, teacher_features = teacher(teacher_images, return_features=True)

            # Adapt student features to teacher space
            adapted_features = adapter(student_features)

            # Compute loss
            losses = loss_fn(student_logits, teacher_logits.detach(),
                           adapted_features, teacher_features.detach(),
                           labels, use_feature_loss=use_feature_loss)

            loss = losses['total']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'ce': f"{losses['ce'].item():.4f}",
            'kd': f"{losses['kd'].item():.4f}"
        })

    avg_loss = total_loss / num_batches
    logger.info(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
    return avg_loss


def validate(student, val_loader, device, logger):
    """Validation step."""
    student.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = student(images)
            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    logger.info(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy


def save_checkpoint(student, optimizer, epoch, loss, checkpoint_dir, logger):
    """Save training checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"

    torch.save({
        'epoch': epoch,
        'model_state_dict': student.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

    logger.info(f"Saved checkpoint: {checkpoint_path}")


def export_onnx(student, output_dir: Path, logger):
    """Export student to ONNX format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "mobilenetv3_legobricks.onnx"

    device = next(student.parameters()).device
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    try:
        torch.onnx.export(
            student,
            dummy_input,
            str(onnx_path),
            input_names=['image'],
            output_names=['logits'],
            dynamic_axes={'image': {0: 'batch_size'}},
            opset_version=14,
            do_constant_folding=True,
            verbose=False
        )
        logger.info(f"Exported ONNX: {onnx_path}")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")


def export_coreml(student, output_dir: Path, num_classes: int, logger):
    """Export student to CoreML format for on-device iOS inference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    coreml_path = output_dir / "LEGOBrickClassifier.mlmodel"

    device = next(student.parameters()).device

    try:
        # Create example input
        example_input = torch.randn(1, 3, 224, 224)

        # Convert to CoreML
        traced_model = torch.jit.trace(student, example_input.to(device))

        ml_model = ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=[ct.ImageType(name="image", shape=(1, 3, 224, 224))],
            outputs=[ct.TensorType(name="logits")]
        )

        ml_model.save(str(coreml_path))
        logger.info(f"Exported CoreML: {coreml_path}")

    except Exception as e:
        logger.error(f"CoreML export failed: {e}")


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation: DINOv2 ViT-B/14 -> MobileNetV3"
    )
    parser.add_argument('--data-dir', type=str,
                       default=os.path.expanduser('~/brickscan/ml/training_data/huggingface_legobricks/images'),
                       help='Path to training data directory')
    parser.add_argument('--teacher-checkpoint', type=str, default=None,
                       help='Path to trained teacher model checkpoint')
    parser.add_argument('--output-dir', type=str,
                       default=os.path.expanduser('~/brickscan/ml/output/distillation'),
                       help='Output directory for checkpoints')
    parser.add_argument('--log-dir', type=str,
                       default=os.path.expanduser('~/brickscan/ml/logs'),
                       help='Log directory')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--temperature', type=float, default=4.0,
                       help='Temperature for KL divergence')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience in epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')

    args = parser.parse_args()

    # Setup logging
    log_dir = Path(args.log_dir)
    logger = setup_logging(log_dir)

    logger.info("=" * 80)
    logger.info("KNOWLEDGE DISTILLATION: DINOv2 ViT-B/14 -> MobileNetV3-Small")
    logger.info("=" * 80)
    logger.info(f"Args: {json.dumps(vars(args), indent=2)}")

    # Device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"CUDA version: {torch.version.cuda}")

    # Datasets
    logger.info(f"Loading dataset from {args.data_dir}")
    train_dataset = LegoBricksDataset(args.data_dir, image_size=224, split='train')
    val_dataset = LegoBricksDataset(args.data_dir, image_size=224, split='val')

    num_classes = train_dataset.num_classes
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Models
    logger.info("Initializing teacher model (DINOv2 ViT-B/14)")
    teacher = TeacherModel(args.teacher_checkpoint, num_classes, device)
    teacher.to(device)
    teacher.freeze()

    logger.info("Initializing student model (MobileNetV3-Small)")
    student = StudentModel(num_classes=num_classes)
    student.to(device)

    total_params = sum(p.numel() for p in student.parameters())
    logger.info(f"Student total parameters: {total_params:,}")

    # Feature adapter
    adapter = nn.Linear(1024, 768)  # MobileNetV3 features -> DINOv2 features
    adapter.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(
        list(student.parameters()) + list(adapter.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss function
    loss_fn = DistillationLoss(temperature=args.temperature)
    loss_fn.to(device)

    # Mixed precision training
    scaler = GradScaler()

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Training loop with progressive training
    logger.info("Starting training...")
    best_val_acc = 0.0
    no_improve_count = 0

    for epoch in range(args.epochs):
        # Progressive training: feature alignment only for first 20 epochs
        use_feature_loss = (epoch >= 20)

        if epoch == 20:
            logger.info("Switching to full three-component loss")

        train_loss = train_epoch(
            student, teacher, adapter, train_loader, optimizer, scaler, loss_fn,
            device, logger, epoch, args.epochs, use_feature_loss=use_feature_loss
        )

        val_acc = validate(student, val_loader, device, logger)

        scheduler.step()

        # Save checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(student, optimizer, epoch, train_loss,
                          output_dir / "checkpoints", logger)
            no_improve_count = 0
        else:
            no_improve_count += 1

        logger.info(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if no_improve_count >= args.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Export models
    logger.info("Exporting student model...")
    export_onnx(student, output_dir / "exports", logger)
    export_coreml(student, output_dir / "exports", num_classes, logger)

    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Checkpoints: {output_dir / 'checkpoints'}")
    logger.info(f"Exports: {output_dir / 'exports'}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
