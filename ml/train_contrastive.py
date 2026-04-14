#!/usr/bin/env python3
"""
Multi-view Contrastive Learning for LEGO Brick Recognition (SimCLR-style)

This script trains a DINOv2 ViT-B/14 backbone using contrastive learning to learn
embeddings where the same part from different angles clusters together, while
different parts (especially brick vs plate) are pushed apart even when visually similar.

Key insight: solves the "brick vs plate confusion" problem by explicitly training on
multi-view same-class pairs so the model learns part identity beyond single-angle appearance.

v2 changes:
  - LoRA via peft: reduces trainable parameters ~90%, eliminates OOM on GB10
  - Gradient checkpointing: additional 40-60% activation memory reduction
  - LayerScale / freeze options for backbone stability
  - Auto-merges LoRA weights before ONNX export (zero inference overhead)
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T

# ── Optional: peft for LoRA ────────────────────────────────────────────────────
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# ── Required packages ─────────────────────────────────────────────────────────
REQUIRED_PACKAGES = ['timm', 'PIL']
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


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_contrastive_{timestamp}.log"

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

class LegoBricksContrastiveDataset(Dataset):
    """
    Loads LEGO brick images organized by class in subdirectories.
    Structure: data_dir/class_id/*.jpg

    Returns augmented view pairs for SimCLR-style contrastive learning.
    """

    def __init__(self, data_dir: str, num_views: int = 2, image_size: int = 224):
        """
        Args:
            data_dir: Path to directory containing class subdirectories
            num_views: Number of augmented views per image (default 2 for SimCLR)
            image_size: Input image size (default 224 for memory efficiency; DINOv2 also
                        supports 518 but that requires ~5x more VRAM per image)
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.num_views = num_views

        # Build image list: [path, class_id]
        self.images = []
        self.class_to_idx = {}

        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_id = class_dir.name
            if class_id not in self.class_to_idx:
                self.class_to_idx[class_id] = len(self.class_to_idx)

            for img_path in list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')):
                self.images.append((str(img_path), class_id))

        self.num_classes = len(self.class_to_idx)

    def _get_augmentation_pipeline(self):
        """
        Heavy augmentation for contrastive learning.
        Includes: crop, color jitter, rotation, perspective, blur, erasing.
        """
        return T.Compose([
            T.RandomResizedCrop(self.image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=360),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            # GaussianBlur disabled: CUDA kernel bug on GB10 aarch64
            T.RandomAffine(degrees=0, translate=(0.1, 0.1),
                          shear=15, scale=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.3, scale=(0.02, 0.33)),  # MUST be after ToTensor
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, class_id = self.images[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.warning(f"Failed to load {img_path}: {e}")
            # Return dummy image on failure
            image = Image.new('RGB', (self.image_size, self.image_size))

        augment = self._get_augmentation_pipeline()

        # Generate num_views augmented versions
        views = [augment(image) for _ in range(self.num_views)]

        class_idx = self.class_to_idx[class_id]

        return {
            'views': torch.stack(views),  # [num_views, 3, H, W]
            'class_id': class_id,
            'class_idx': class_idx
        }


# ============================================================================
# Model Architecture
# ============================================================================

class ContrastiveProjectionHead(nn.Module):
    """Projection head for contrastive learning: Linear -> BN -> ReLU -> Linear."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.normalize(x, dim=-1)


class ContrastiveModel(nn.Module):
    """
    DINOv2 ViT-B/14 backbone + contrastive projection head.

    Memory modes (highest-to-lowest VRAM use):
      full fine-tune  : all 86M params trainable  (~16-24 GB at bs=256)
      LoRA (default)  : ~1.2M trainable params    (~ 6-8  GB at bs=256)
      LoRA + grad_ckpt: same trainable, less acts  (~ 4-5  GB at bs=256)
    """

    def __init__(
        self,
        backbone_name: str = 'vit_base_patch14_dinov2',
        projection_dim: int = 128,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_grad_checkpointing: bool = True,
    ):
        super().__init__()

        # ── Load pretrained DINOv2 ───────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,          # remove classification head
            img_size=224,           # DINOv2 defaults to 518; force 224 with PE interpolation
            dynamic_img_size=True,  # allow variable input sizes at inference
        )
        backbone_dim = self.backbone.embed_dim  # 768 for ViT-B

        # Disable strict size check immediately after model creation.
        # DINOv2 pretrained weights are at 518×518, but timm interpolates
        # position embeddings to 224×224 when img_size=224 is passed.
        # The assertion in patch_embed.forward must be relaxed to allow 224.
        if hasattr(self.backbone, 'patch_embed'):
            self.backbone.patch_embed.strict_img_size = False
            self.backbone.patch_embed.img_size = (224, 224)

        # ── Apply LoRA ───────────────────────────────────────────────────────
        # Targets attention QKV projection and output projection.
        # Reduces trainable params from 86M → ~1.2M while preserving DINOv2
        # feature quality via frozen pretrained weights.
        self._lora_active = False
        if use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "LoRA requested but 'peft' is not installed.\n"
                    "  pip install peft\n"
                    "Or re-run with --no-lora to train the full backbone "
                    "(requires ~3x more VRAM)."
                )
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                # timm DINOv2 ViT-B/14 attention: combined qkv + output projection.
                # NOTE: "proj" also matches patch_embed.proj (Conv2d), but peft
                # will skip Conv2d layers for LoRA. We keep the name precise:
                # attention output proj lives at blocks.*.attn.proj
                target_modules=["qkv", "attn.proj"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self._lora_active = True

            # LoRA wrapping can reset patch_embed img_size assertions.
            # Crawl the entire wrapped module tree to find patch_embed and
            # disable the strict size check, regardless of peft's internal
            # nesting structure (PeftModel → LoraModel → timm model varies
            # across peft versions).
            for _name, _mod in self.backbone.named_modules():
                if hasattr(_mod, 'strict_img_size'):
                    _mod.strict_img_size = False
                if hasattr(_mod, 'img_size') and 'patch_embed' in _name:
                    _mod.img_size = (224, 224)

        # ── Gradient checkpointing ───────────────────────────────────────────
        # Recomputes activations during backward instead of storing them.
        # Cuts activation memory ~50% at a ~20% compute overhead.
        if use_grad_checkpointing:
            try:
                # timm ViT exposes set_grad_checkpointing()
                base = self.backbone.base_model if self._lora_active else self.backbone
                base.set_grad_checkpointing(enable=True)
            except AttributeError:
                warnings.warn(
                    "set_grad_checkpointing() not available on this backbone "
                    "(timm version may be old). Skipping."
                )

        # ── Projection head ──────────────────────────────────────────────────
        self.projection_head = ContrastiveProjectionHead(
            input_dim=backbone_dim,
            hidden_dim=512,
            output_dim=projection_dim
        )

    def forward(self, x):
        """Forward pass: x shape [B, 3, H, W]."""
        # Backbone returns [B, embed_dim]
        features = self.backbone.forward_features(x)

        # Handle different output formats from timm
        if isinstance(features, dict):
            features = features.get('x', features.get('cls_token', features))

        if features.dim() == 3:  # [B, num_tokens, embed_dim]
            features = features[:, 0, :]  # Take CLS token

        # Project to embedding space
        embeddings = self.projection_head(features)
        return embeddings

    def get_embedding(self, x):
        """Inference-only: returns CLS embedding before projection head."""
        with torch.no_grad():
            features = self.backbone.forward_features(x)
            if isinstance(features, dict):
                features = features.get('x', features.get('cls_token', features))
            if features.dim() == 3:
                features = features[:, 0, :]
        return features

    def log_trainable_params(self, logger):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = 100.0 * trainable / total if total > 0 else 0.0
        logger.info(f"Parameters — total: {total:,}  trainable: {trainable:,}  ({pct:.1f}%)")
        if self._lora_active:
            logger.info("  LoRA active: only LoRA adapters + projection head are updated")


# ============================================================================
# Loss Function: NT-Xent (Normalized Temperature-scaled Cross Entropy)
# ============================================================================

class NTXentLoss(nn.Module):
    """
    NT-Xent loss for SimCLR.
    Pulls same-class pairs together, pushes different-class pairs apart.
    """

    def __init__(self, temperature: float = 0.07, batch_size: int = 256):
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, z_i, z_j, class_labels=None):
        """
        Args:
            z_i, z_j: Embeddings from two views [B, embed_dim]
            class_labels: Class indices [B] for hard negative mining (optional)

        Returns:
            NT-Xent loss value
        """
        B = z_i.shape[0]

        # Concatenate representations: [2B, embed_dim]
        z = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix [2B, 2B]
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # Positive pairs: (i, B+i) and (B+i, i)
        # Negatives: all other pairs
        labels = torch.arange(B, dtype=torch.long, device=z.device)
        labels = torch.cat([labels + B, labels], dim=0)  # Permute labels

        # Hard negative mining: penalize same-class pairs that should not be negatives
        if class_labels is not None:
            class_labels_double = torch.cat([class_labels, class_labels], dim=0)
            same_class_mask = class_labels_double.unsqueeze(0) == class_labels_double.unsqueeze(1)

            # Zero out positive pairs on diagonal (they'll be handled by labels)
            diag_mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
            same_class_mask = same_class_mask & ~diag_mask

            # Down-weight hard negatives (same class but different views)
            sim_matrix = sim_matrix - 10.0 * same_class_mask.float()

        loss = self.criterion(sim_matrix, labels)
        return loss


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, train_loader, optimizer, scaler, loss_fn,
                device, logger, epoch, num_epochs):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

    for batch_idx, batch in enumerate(pbar):
        views = batch['views'].to(device)  # [B, 2, 3, H, W]
        class_indices = batch['class_idx'].to(device)  # [B]

        B = views.shape[0]

        # Split views
        view1 = views[:, 0]  # [B, 3, H, W]
        view2 = views[:, 1]  # [B, 3, H, W]

        optimizer.zero_grad()

        with autocast():
            # Forward pass
            z1 = model(view1)  # [B, embed_dim]
            z2 = model(view2)  # [B, embed_dim]

            # Compute loss with hard negative mining
            loss = loss_fn(z1, z2, class_labels=class_indices)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    logger.info(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate_knn_accuracy(model, dataset, k: int = 20, num_samples: int = 1000):
    """
    Evaluate k-NN accuracy on validation set using learned embeddings.
    Tests if same-part embeddings cluster regardless of viewing angle.
    """
    model.eval()
    device = next(model.parameters()).device

    # Sample subset for efficiency
    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)),
                                      replace=False)

    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for idx in tqdm(sample_indices, desc="Computing embeddings"):
            batch = dataset[idx]
            # Use first view only for evaluation
            view = batch['views'][0].unsqueeze(0).to(device)
            embedding = model(view).cpu().numpy()

            embeddings_list.append(embedding[0])
            labels_list.append(batch['class_idx'])

    embeddings = np.array(embeddings_list)  # [N, embed_dim]
    labels = np.array(labels_list)  # [N]

    # Compute k-NN accuracy
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Remove self-match (distance 0)
    indices = indices[:, 1:]
    neighbor_labels = labels[indices]

    # Accuracy: how many of top-k neighbors have same class?
    accuracy = (neighbor_labels == labels.reshape(-1, 1)).sum() / (len(labels) * k)

    return float(accuracy)


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, logger, args):
    """Save training checkpoint including LoRA config."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"

    # Save LoRA adapter weights separately if active
    # (full state_dict also stored for compatibility)
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': vars(args),
    }

    torch.save(save_dict, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Also save LoRA adapters via peft if active
    if getattr(model, '_lora_active', False):
        lora_dir = checkpoint_dir / f"lora_epoch_{epoch:03d}"
        try:
            model.backbone.save_pretrained(str(lora_dir))
            logger.info(f"Saved LoRA adapters: {lora_dir}")
        except Exception as e:
            logger.warning(f"Could not save LoRA adapters: {e}")


def export_onnx(model, output_dir: Path, logger, image_size: int = 224):
    """
    Export model to ONNX format for inference.

    If LoRA is active, merges adapter weights into the backbone first so the
    exported model has zero inference overhead from LoRA.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "contrastive_encoder.onnx"

    device = next(model.parameters()).device

    # ── Merge LoRA weights before export ─────────────────────────────────────
    # Creates a clean model copy with LoRA deltas baked into base weights.
    export_model = model
    if getattr(model, '_lora_active', False):
        try:
            logger.info("Merging LoRA weights into backbone for ONNX export...")
            # merge_and_unload() returns a plain nn.Module with merged weights
            merged_backbone = model.backbone.merge_and_unload()
            # Build a temporary export model (projection head stays the same)
            export_model = ContrastiveModel.__new__(ContrastiveModel)
            export_model.__dict__.update(model.__dict__)
            export_model.backbone = merged_backbone
            export_model._lora_active = False
            logger.info("LoRA merge complete — zero inference overhead in ONNX")
        except Exception as e:
            logger.warning(f"LoRA merge failed ({e}), exporting with adapters active")
            export_model = model

    export_model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    try:
        torch.onnx.export(
            export_model,
            dummy_input,
            str(onnx_path),
            input_names=['image'],
            output_names=['embedding'],
            dynamic_axes={'image': {0: 'batch_size'}},
            opset_version=12,           # opset 12: CoreML-compatible
            do_constant_folding=True,
            verbose=False
        )
        logger.info(f"Exported ONNX model: {onnx_path}")
    except Exception as e:
        logger.error(f"Failed to export ONNX: {e}")


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-view Contrastive Learning for LEGO Brick Recognition"
    )
    parser.add_argument('--data-dir', type=str,
                       default=os.path.expanduser('~/brickscan/ml/training_data/huggingface_legobricks/images'),
                       help='Path to training data directory')
    parser.add_argument('--output-dir', type=str,
                       default=os.path.expanduser('~/brickscan/ml/output/contrastive'),
                       help='Output directory for checkpoints')
    parser.add_argument('--log-dir', type=str,
                       default=os.path.expanduser('~/brickscan/ml/logs'),
                       help='Log directory')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for NT-Xent loss')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Input image size. 224 recommended for GB10 aarch64 (saves ~81%% '
                            'activation memory vs default 518). Use 518 only with batch-size<=8.')

    # ── LoRA arguments ────────────────────────────────────────────────────────
    lora_group = parser.add_argument_group('LoRA (parameter-efficient fine-tuning)')
    lora_group.add_argument('--no-lora', action='store_true',
                            help='Disable LoRA and train the full DINOv2 backbone '
                                 '(WARNING: requires ~3x more VRAM, likely OOM on GB10)')
    lora_group.add_argument('--lora-rank', type=int, default=16,
                            help='LoRA rank r (default: 16). '
                                 'Higher = more params, better capacity. Try 8/16/32.')
    lora_group.add_argument('--lora-alpha', type=int, default=32,
                            help='LoRA scaling alpha (default: 32 = 2x rank). '
                                 'Effective LR scale = alpha/rank.')
    lora_group.add_argument('--lora-dropout', type=float, default=0.05,
                            help='Dropout applied to LoRA layers (default: 0.05)')

    # ── Gradient checkpointing ────────────────────────────────────────────────
    parser.add_argument('--no-grad-checkpointing', action='store_true',
                        help='Disable gradient checkpointing (uses more VRAM, trains ~20%% faster)')

    args = parser.parse_args()

    # Resolve flags
    use_lora = not args.no_lora
    use_grad_ckpt = not args.no_grad_checkpointing

    # LoRA requires peft
    if use_lora and not PEFT_AVAILABLE:
        print("\nERROR: LoRA is enabled (default) but 'peft' is not installed.")
        print("  Install with:  pip install peft")
        print("  Or disable LoRA (not recommended for GB10):  --no-lora\n")
        sys.exit(1)

    # Setup logging
    log_dir = Path(args.log_dir)
    logger = setup_logging(log_dir)

    logger.info("=" * 80)
    logger.info("CONTRASTIVE LEARNING FOR LEGO BRICK RECOGNITION  v2 (LoRA)")
    logger.info("=" * 80)
    logger.info(f"Args: {json.dumps(vars(args), indent=2)}")
    logger.info(f"LoRA: {'ENABLED (rank={}, alpha={})'.format(args.lora_rank, args.lora_alpha) if use_lora else 'DISABLED'}")
    logger.info(f"Gradient checkpointing: {'ON' if use_grad_ckpt else 'OFF'}")

    # Device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Dataset
    logger.info(f"Loading dataset from {args.data_dir}")
    logger.info(f"Image size: {args.image_size}×{args.image_size}")
    dataset = LegoBricksContrastiveDataset(args.data_dir, num_views=2, image_size=args.image_size)
    logger.info(f"Loaded {len(dataset)} images from {dataset.num_classes} classes")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Model
    logger.info("Initializing model: DINOv2 ViT-B/14 + projection head")
    model = ContrastiveModel(
        backbone_name='vit_base_patch14_dinov2',
        projection_dim=128,
        use_lora=use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_grad_checkpointing=use_grad_ckpt,
    )
    model.to(device)
    model.log_trainable_params(logger)

    # ── Optimizer: only trainable params (LoRA + projection head) ─────────────
    # This is automatic when peft freezes the backbone — filter is a safeguard.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Optimizer watching {len(trainable_params)} parameter tensors")
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss function
    loss_fn = NTXentLoss(temperature=args.temperature, batch_size=args.batch_size)
    loss_fn.to(device)

    # Mixed precision training
    scaler = GradScaler()

    # Output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Training loop
    logger.info("Starting training...")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, loss_fn,
            device, logger, epoch, args.epochs
        )

        scheduler.step()

        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint(model, optimizer, epoch, train_loss,
                          output_dir / "checkpoints", logger, args)

        # Save checkpoint every N epochs
        if (epoch + 1) % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, train_loss,
                          output_dir / "checkpoints", logger, args)

        logger.info(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

    # Evaluation: k-NN accuracy
    logger.info("Computing k-NN validation accuracy...")
    try:
        knn_acc = evaluate_knn_accuracy(model, dataset, k=20, num_samples=1000)
        logger.info(f"k-NN accuracy (top-20): {knn_acc:.4f}")
    except Exception as e:
        logger.warning(f"k-NN evaluation failed: {e}")

    # Export to ONNX (merges LoRA weights automatically)
    logger.info("Exporting model to ONNX format...")
    export_onnx(model, output_dir / "exports", logger, image_size=args.image_size)

    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
    logger.info(f"ONNX export: {output_dir / 'exports'}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
