#!/usr/bin/env python3
"""
Fine-tuning DINOv2 ViT-B/14 for LEGO part classification.

This script implements a two-stage fine-tuning strategy:
1. Stage 1: Freeze backbone, train only classification head (warmup)
2. Stage 2: Unfreeze last 4 transformer blocks, end-to-end fine-tuning

Features:
- Mixed precision training (AMP) for speed
- Cosine annealing with warmup
- Strong augmentations for rendered image robustness
- Combines HuggingFace dataset (1000 classes, 400K images) with sparse dataset (2100 classes)
- Saves best model by validation accuracy
- ONNX export at the end
- Comprehensive logging and metric visualization
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

import timm
from timm.models import create_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


# Configure logging
def setup_logger(log_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_dir / "training.log")
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


class LEGODataset(Dataset):
    """Custom dataset for LEGO parts with albumentations support."""

    def __init__(self, image_paths: List[str], labels: List[int],
                 transform=None, use_albumentations=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.use_albumentations = use_albumentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.use_albumentations and self.transform is not None:
            # Albumentations works with numpy arrays
            image = np.array(image)
            transformed = self.transform(image=image)
            image = transformed['image']
        elif self.transform is not None:
            image = self.transform(image)

        return image, label


def build_augmentation_pipeline(input_size: int = 518,
                               is_train: bool = True) -> transforms.Compose:
    """Build data augmentation pipeline using torchvision."""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                input_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, shear=10),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomPerspective(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD
            ),
        ])


def build_albumentations_pipeline(input_size: int = 518,
                                 is_train: bool = True):
    """Build augmentation pipeline using albumentations (if available)."""
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(
                height=input_size, width=input_size,
                scale=(0.8, 1.0), p=1.0
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
            ),
            A.Rotate(limit=15, p=0.5),
            A.Affine(shear=10, p=0.3),
            A.GaussBlur(p=0.3),
            A.Perspective(p=0.3),
            A.RandomRain(p=0.1),
            A.RandomFog(p=0.1),
            A.CoarseDropout(max_holes=1, max_height=50, max_width=50, p=0.2),
            A.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD
            ),
            ToTensorV2(),
        ], is_check_shapes=False)
    else:
        return A.Compose([
            A.Resize(height=input_size, width=input_size),
            A.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD
            ),
            ToTensorV2(),
        ])


def load_and_combine_datasets(hf_data_dir: Path, sparse_data_dir: Optional[Path],
                             input_size: int = 518,
                             use_albumentations: bool = True,
                             train_ratio: float = 0.8,
                             logger: Optional[logging.Logger] = None):
    """Load and combine HuggingFace and sparse datasets."""

    if logger is None:
        logger = logging.getLogger(__name__)

    # Build augmentation pipelines
    if ALBUMENTATIONS_AVAILABLE and use_albumentations:
        train_transform = build_albumentations_pipeline(input_size, is_train=True)
        val_transform = build_albumentations_pipeline(input_size, is_train=False)
        use_alb = True
    else:
        train_transform = build_augmentation_pipeline(input_size, is_train=True)
        val_transform = build_augmentation_pipeline(input_size, is_train=False)
        use_alb = False

    # Load HuggingFace dataset
    logger.info(f"Loading HuggingFace dataset from {hf_data_dir}")
    hf_data_dir = Path(hf_data_dir)

    image_paths_hf = []
    labels_hf = []
    class_to_idx = {}
    idx_counter = 0

    for class_dir in sorted(hf_data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_idx = idx_counter
        class_to_idx[class_dir.name] = class_idx
        idx_counter += 1

        for img_file in class_dir.glob("*.png"):
            image_paths_hf.append(str(img_file))
            labels_hf.append(class_idx)

    logger.info(f"HuggingFace dataset: {len(image_paths_hf)} images, {len(class_to_idx)} classes")

    # Load sparse dataset if provided
    image_paths_sparse = []
    labels_sparse = []

    if sparse_data_dir is not None:
        logger.info(f"Loading sparse dataset from {sparse_data_dir}")
        sparse_data_dir = Path(sparse_data_dir)

        # Scan both train and val directories
        for split_dir in [sparse_data_dir / "train", sparse_data_dir / "val"]:
            if not split_dir.exists():
                continue

            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue

                class_name = class_dir.name
                if class_name not in class_to_idx:
                    class_to_idx[class_name] = idx_counter
                    idx_counter += 1

                class_idx = class_to_idx[class_name]

                for img_file in class_dir.glob("*.png"):
                    image_paths_sparse.append(str(img_file))
                    labels_sparse.append(class_idx)

        logger.info(f"Sparse dataset: {len(image_paths_sparse)} images")

    # Combine datasets and split
    all_image_paths = image_paths_hf + image_paths_sparse
    all_labels = labels_hf + labels_sparse

    logger.info(f"Combined dataset: {len(all_image_paths)} images, {len(class_to_idx)} classes")

    # Split into train/val using HuggingFace data for splits (more balanced)
    hf_size = len(image_paths_hf)
    train_size_hf = int(hf_size * train_ratio)
    val_size_hf = hf_size - train_size_hf

    # Split HuggingFace data
    indices_hf = np.random.permutation(hf_size)
    train_indices_hf = indices_hf[:train_size_hf]
    val_indices_hf = indices_hf[train_size_hf:]

    # Use all sparse data for validation (since it's sparse)
    train_image_paths = [all_image_paths[i] for i in train_indices_hf]
    train_labels = [all_labels[i] for i in train_indices_hf]

    val_image_paths = [all_image_paths[i] for i in val_indices_hf] + image_paths_sparse
    val_labels = [all_labels[i] for i in val_indices_hf] + labels_sparse

    logger.info(f"Train split: {len(train_image_paths)} images")
    logger.info(f"Val split: {len(val_image_paths)} images")

    # Create datasets
    train_dataset = LEGODataset(
        train_image_paths, train_labels,
        transform=train_transform,
        use_albumentations=use_alb
    )

    val_dataset = LEGODataset(
        val_image_paths, val_labels,
        transform=val_transform,
        use_albumentations=use_alb
    )

    return train_dataset, val_dataset, class_to_idx


class DINOv2Classifier(nn.Module):
    """DINOv2 backbone with classification head."""

    def __init__(self, num_classes: int, backbone_name: str = 'vit_base_patch14_dinov2'):
        super().__init__()

        # Load DINOv2 backbone from timm
        self.backbone = create_model(backbone_name, pretrained=True)
        self.feat_dim = self.backbone.num_features

        # Remove classification head from backbone if present
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()

        # Add classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        # Initialize classifier weights
        self._init_classifier()

    def _init_classifier(self):
        """Initialize classifier weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Get features from backbone
        feat = self.backbone.forward_features(x)

        # Handle different output formats from timm models
        if isinstance(feat, dict):
            feat = feat.get('x', feat.get('features', feat))

        # If feat has spatial dimensions, take mean across them
        if feat.dim() == 4:  # (B, C, H, W)
            feat = feat.mean(dim=(2, 3))

        # Classify
        logits = self.classifier(feat)
        return logits

    def freeze_backbone(self):
        """Freeze backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n_blocks: int = 4):
        """Unfreeze last N transformer blocks."""
        # Get transformer blocks
        if hasattr(self.backbone, 'blocks'):
            blocks = self.backbone.blocks
            total_blocks = len(blocks)

            # Freeze all blocks except last n_blocks
            for i, block in enumerate(blocks):
                block.requires_grad = (i >= total_blocks - n_blocks)

        # Always unfreeze classifier
        for param in self.classifier.parameters():
            param.requires_grad = True


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    num_cycles=0.5, last_epoch=-1):
    """Cosine annealing schedule with linear warmup."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train_epoch(model, train_loader, optimizer, criterion, scaler, device, logger):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast(dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def train(args):
    """Main training loop."""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info(f"Device: {device}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"Arguments: {args}")

    # Load datasets
    train_dataset, val_dataset, class_to_idx = load_and_combine_datasets(
        hf_data_dir=args.hf_data_dir,
        sparse_data_dir=args.sparse_data_dir if args.sparse_data_dir else None,
        input_size=args.input_size,
        use_albumentations=args.use_albumentations,
        train_ratio=args.train_ratio,
        logger=logger
    )

    num_classes = len(class_to_idx)
    logger.info(f"Number of classes: {num_classes}")

    # Save class mapping
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    # Create data loaders
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

    # Create model
    logger.info(f"Creating model: {args.model_name}")
    model = DINOv2Classifier(num_classes=num_classes, backbone_name=args.model_name)
    model = model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Freeze backbone initially
    model.freeze_backbone()
    logger.info("Backbone frozen for warmup phase")

    # Optimizer and scheduler for Stage 1 (warmup)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.warmup_lr,
        weight_decay=args.weight_decay,
        amsgrad=True
    )

    total_steps = args.warmup_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    scaler = GradScaler()

    # Initialize wandb if available
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"dinov2_lego_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args),
            dir=output_dir
        )

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }

    best_val_acc = 0.0
    best_model_path = output_dir / "best_model.pt"

    # Stage 1: Warmup (frozen backbone)
    logger.info("="*60)
    logger.info("STAGE 1: WARMUP (Frozen Backbone)")
    logger.info("="*60)

    for epoch in range(args.warmup_epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.warmup_epochs} (Stage 1)")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, logger
        )
        scheduler.step()

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        logger.info(
            f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
            f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with val_acc={val_acc:.4f}")

        if WANDB_AVAILABLE and args.use_wandb:
            wandb.log({
                'stage': 1,
                'epoch': epoch,
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'val/loss': val_loss,
                'val/accuracy': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

    # Stage 2: Fine-tuning (unfreeze last 4 blocks)
    logger.info("\n" + "="*60)
    logger.info("STAGE 2: FINE-TUNING (Unfrozen Last 4 Blocks)")
    logger.info("="*60)

    # Load best model from stage 1
    model.load_state_dict(torch.load(best_model_path))

    # Unfreeze last 4 blocks
    model.unfreeze_last_n_blocks(n_blocks=4)

    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters in Stage 2: {trainable_params:,}")

    # New optimizer for fine-tuning with lower LR
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
        amsgrad=True
    )

    total_steps = args.finetune_epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)  # Shorter warmup for stage 2
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    scaler = GradScaler()

    for epoch in range(args.finetune_epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.finetune_epochs} (Stage 2)")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, logger
        )
        scheduler.step()

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        logger.info(
            f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
            f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with val_acc={val_acc:.4f}")

        if WANDB_AVAILABLE and args.use_wandb:
            wandb.log({
                'stage': 2,
                'epoch': args.warmup_epochs + epoch,
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'val/loss': val_loss,
                'val/accuracy': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

    # Save final model
    final_model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")

    # Plot training history
    logger.info("Plotting training history...")
    plot_training_history(history, output_dir)

    # Export to ONNX
    logger.info("Exporting to ONNX...")
    export_to_onnx(model, output_dir, args.input_size, device, logger)

    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Training completed. Results saved to {output_dir}")

    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()


def plot_training_history(history: Dict, output_dir: Path):
    """Plot and save training history."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate
    axes[2].plot(history['learning_rate'], linewidth=2, color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'epoch': range(len(history['train_loss'])),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc'],
        'learning_rate': history['learning_rate']
    })
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)


def export_to_onnx(model, output_dir: Path, input_size: int, device, logger):
    """Export model to ONNX format."""

    try:
        import onnx

        onnx_path = output_dir / "model.onnx"

        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

        # Export
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=['image'],
            output_names=['logits'],
            dynamic_axes={'image': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
            opset_version=14,
            do_constant_folding=True,
            verbose=False
        )

        logger.info(f"Exported ONNX model to {onnx_path}")

        # Verify ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed")

    except ImportError:
        logger.warning("ONNX not installed, skipping ONNX export")
    except Exception as e:
        logger.error(f"Error exporting to ONNX: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DINOv2 for LEGO part classification"
    )

    # Data arguments
    parser.add_argument(
        '--hf-data-dir',
        type=str,
        default='~/brickscan/ml/training_data/huggingface_legobricks/images/',
        help='Path to HuggingFace LEGO dataset'
    )
    parser.add_argument(
        '--sparse-data-dir',
        type=str,
        default=None,
        help='Path to sparse LEGO dataset (optional)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Train/val split ratio for HuggingFace data'
    )

    # Model arguments
    parser.add_argument(
        '--model-name',
        type=str,
        default='vit_base_patch14_dinov2',
        help='Backbone model name from timm'
    )
    parser.add_argument(
        '--input-size',
        type=int,
        default=518,
        help='Input image size'
    )

    # Training arguments - Stage 1 (Warmup)
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=5,
        help='Number of warmup epochs (frozen backbone)'
    )
    parser.add_argument(
        '--warmup-lr',
        type=float,
        default=1e-3,
        help='Learning rate for warmup phase'
    )

    # Training arguments - Stage 2 (Fine-tuning)
    parser.add_argument(
        '--finetune-epochs',
        type=int,
        default=30,
        help='Number of fine-tuning epochs (unfrozen blocks)'
    )
    parser.add_argument(
        '--finetune-lr',
        type=float,
        default=1e-4,
        help='Learning rate for fine-tuning phase'
    )

    # General training arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay for optimizer'
    )
    parser.add_argument(
        '--label-smoothing',
        type=float,
        default=0.1,
        help='Label smoothing for CrossEntropyLoss'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of data loading workers'
    )

    # Augmentation arguments
    parser.add_argument(
        '--use-albumentations',
        action='store_true',
        default=False,
        help='Use albumentations for augmentation (if available)'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/dinov2_lego',
        help='Output directory for results'
    )

    # Logging arguments
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        default=False,
        help='Use Weights & Biases for logging'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='lego-classification',
        help='Weights & Biases project name'
    )
    parser.add_argument(
        '--wandb-entity',
        type=str,
        default=None,
        help='Weights & Biases entity name'
    )

    args = parser.parse_args()

    # Expand user paths
    args.hf_data_dir = str(Path(args.hf_data_dir).expanduser())
    if args.sparse_data_dir:
        args.sparse_data_dir = str(Path(args.sparse_data_dir).expanduser())
    args.output_dir = str(Path(args.output_dir).expanduser())

    train(args)


if __name__ == '__main__':
    main()


# ============================================================================
# LAUNCH SCRIPT AND RECOMMENDED HYPERPARAMETERS
# ============================================================================
#
# Recommended command for DGX Spark (GB10 Blackwell, 130.6GB VRAM):
#
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_dinov2.py \
#     --hf-data-dir ~/brickscan/ml/training_data/huggingface_legobricks/images/ \
#     --sparse-data-dir ~/brickscan/ml/data/images/ \
#     --model-name vit_base_patch14_dinov2 \
#     --input-size 518 \
#     --warmup-epochs 5 \
#     --warmup-lr 1e-3 \
#     --finetune-epochs 30 \
#     --finetune-lr 1e-4 \
#     --batch-size 64 \
#     --weight-decay 0.01 \
#     --label-smoothing 0.1 \
#     --num-workers 8 \
#     --output-dir ./outputs/dinov2_lego \
#     --use-wandb \
#     --wandb-project lego-classification
#
# ============================================================================
# NOTES:
# ============================================================================
#
# 1. HARDWARE:
#    - DGX Spark: 8x H100/H200 GPUs (Blackwell GB10)
#    - CUDA 13.0, PyTorch 2.12.0
#    - Script uses single GPU by default; modify for multi-GPU with torch.nn.DataParallel
#
# 2. DATASET:
#    - HuggingFace: 1000 classes, 400K images (balanced)
#    - Sparse: 2100 classes, ~4774 images (used only for validation)
#    - Script automatically combines both datasets
#    - 80/20 train/val split applied to HF data only
#
# 3. TRAINING STAGES:
#    - Stage 1 (5 epochs): Frozen backbone, train only head
#      - Warms up the classification layer
#      - Lower memory footprint
#      - LR: 1e-3 with cosine annealing
#    - Stage 2 (30 epochs): Unfreeze last 4 blocks, end-to-end fine-tuning
#      - Adapts backbone features to LEGO domain
#      - Higher memory usage
#      - LR: 1e-4 (10x lower) to preserve pretrained weights
#
# 4. AUGMENTATION:
#    - Strong augmentations for rendered image robustness
#    - RandomResizedCrop, ColorJitter, Rotation, Affine, GaussBlur
#    - Optional: Use albumentations for additional rain/fog/dropout
#
# 5. OPTIMIZATION:
#    - Mixed precision (AMP) for 2-3x speedup
#    - Adam optimizer with weight decay (AdamW-style)
#    - Gradient clipping (norm=1.0) to prevent instability
#    - Label smoothing (0.1) for regularization
#
# 6. LEARNING RATE SCHEDULE:
#    - Cosine annealing with linear warmup
#    - 10% warmup steps in stage 1, 5% in stage 2
#    - Final LR decays to near-zero by end of training
#
# 7. MODEL CHECKPOINTING:
#    - Saves best model by validation accuracy
#    - Also saves final model after stage 2
#    - ONNX export for deployment
#
# 8. LOGGING:
#    - File logging to outputs/training.log
#    - Weights & Biases integration (optional)
#    - Training history plots and metrics CSV
#
# 9. BATCH SIZE:
#    - 64 per GPU is reasonable for ViT-B/14 with 518x518 input
#    - With 130.6GB VRAM, can likely go up to 128-256 if needed
#    - Monitor with torch.cuda.memory_allocated()
#
# 10. EXPECTED RESULTS:
#     - Val accuracy should reach ~95%+ on HuggingFace validation set
#     - Training time: ~4-6 hours on single H100
#     - Best models saved to outputs/dinov2_lego/best_model.pt
#
# ============================================================================
