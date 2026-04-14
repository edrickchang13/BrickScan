"""
Production-quality training script for BrickScan stage 2: standalone brick classifier.

Trains a dual-head EfficientNet-B3 on cropped single-brick images to predict both
part number and color ID. Implements focal loss for class imbalance, advanced
augmentation (Mixup, CutMix), TTA at inference, and ONNX export.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor in range (0, 1) to balance classes. Default 0.25.
        gamma: Exponent of the modulating factor (1 - p_t)^gamma. Default 2.0.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model outputs [B, C]
            targets: Ground truth class indices [B]

        Returns:
            Scalar focal loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - p_t) ** self.gamma) * ce_loss
        return focal_loss.mean()


class BrickClassifierDataset(Dataset):
    """
    Dataset for LEGO brick classification.

    Expected directory structure:
        <root>/<part_num>/<color_id>/<image.jpg>  (preferred)
        OR
        <root>/<part_num>/<image.jpg>  (color_id defaults to 0)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: int = 300,
        use_mixup: bool = True,
        use_cutmix: bool = True,
        input_channels: int = 3,
        part_to_idx: Optional[Dict[str, int]] = None,
        color_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize dataset.

        Args:
            root_dir: Root directory containing part/color/image structure
            split: 'train', 'val', or 'test'
            image_size: Size to resize images to (default 300x300)
            use_mixup: Whether to apply Mixup augmentation
            use_cutmix: Whether to apply CutMix augmentation
            input_channels: Number of input channels (3=RGB, 4=RGBD, 6=RGBD+normals)
            part_to_idx: Pre-built part->index mapping (for consistency across splits)
            color_to_idx: Pre-built color->index mapping (for consistency across splits)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.input_channels = input_channels
        self.use_mixup = use_mixup and (split == 'train')
        self.use_cutmix = use_cutmix and (split == 'train')

        # Build samples and mappings
        self.samples = []
        self._build_part_to_idx = part_to_idx is None

        if self._build_part_to_idx:
            # First pass: collect all unique part numbers and color IDs
            all_parts = set()
            all_colors = set()

            for part_dir in sorted(self.root_dir.iterdir()):
                if not part_dir.is_dir():
                    continue

                part_num = part_dir.name
                all_parts.add(part_num)

                # Check if color_id subdirectories or images directly
                for item in part_dir.iterdir():
                    if item.is_file() and item.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                        all_colors.add('0')  # Default color
                    elif item.is_dir():
                        all_colors.add(item.name)

            self.part_to_idx = {p: i for i, p in enumerate(sorted(all_parts))}
            self.color_to_idx = {c: i for i, c in enumerate(sorted(all_colors))}
        else:
            self.part_to_idx = part_to_idx
            self.color_to_idx = color_to_idx

        # Second pass: collect all samples
        self.all_samples = []
        for part_dir in sorted(self.root_dir.iterdir()):
            if not part_dir.is_dir():
                continue

            part_num = part_dir.name

            for item in part_dir.iterdir():
                if item.is_file() and item.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    # Image directly in part directory
                    color_id = '0'
                    self.all_samples.append((item, part_num, color_id))
                elif item.is_dir():
                    # Color subdirectory
                    color_id = item.name
                    for img_file in item.glob('*'):
                        if img_file.is_file() and img_file.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                            self.all_samples.append((img_file, part_num, color_id))

        # Split samples by part (stratified by part)
        part_to_samples = {}
        for img_path, part_num, color_id in self.all_samples:
            if part_num not in part_to_samples:
                part_to_samples[part_num] = []
            part_to_samples[part_num].append((img_path, part_num, color_id))

        # 80/10/10 split
        train_samples, val_test_samples = [], []
        for part_num, samples in part_to_samples.items():
            n = len(samples)
            indices = list(range(n))
            random.shuffle(indices)

            train_count = int(0.8 * n)
            for i in indices[:train_count]:
                train_samples.append(samples[i])
            for i in indices[train_count:]:
                val_test_samples.append(samples[i])

        # Further split val/test 50/50
        val_count = len(val_test_samples) // 2
        test_samples = val_test_samples[val_count:]
        val_samples = val_test_samples[:val_count]

        if split == 'train':
            self.samples = train_samples
        elif split == 'val':
            self.samples = val_samples
        else:  # test
            self.samples = test_samples

        logger.info(f"Loaded {len(self.samples)} samples for split '{split}'")

        # Define augmentation pipelines
        if split == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5, border_mode=0),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.5),
                A.Affine(translate_percent=(0.1, 0.1), p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ], bbox_params=None)
            self.to_tensor = transforms.ToTensor()
        else:
            # Validation/test: center crop + normalize
            self.transform = A.Compose([
                A.CenterCrop(height=image_size, width=image_size, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get a sample.

        Returns:
            (image_tensor, part_idx, color_idx)
            image_tensor shape: (C, H, W) where C = input_channels (3, 4, or 6)
        """
        img_path, part_num, color_id = self.samples[idx]

        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)

        # Resize to target size
        image_np = np.array(Image.fromarray(image_np).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        ))

        # Apply augmentation
        augmented = self.transform(image=image_np)
        image_np = augmented['image']

        # Handle multi-channel inputs (depth, normals)
        if self.input_channels > 3:
            # Look for paired depth file in depth/ subdirectory
            depth_path = Path(img_path).parent.parent / 'depth' / (Path(img_path).stem + '.png')

            if depth_path.exists():
                # Load depth file
                try:
                    from ml.preprocessing.depth_processor import load_depth_png, depth_to_4channel, depth_and_normals_to_6channel

                    depth_mm = load_depth_png(str(depth_path))

                    if self.input_channels == 4:
                        # RGBD: combine RGB + normalized depth
                        image_tensor = torch.from_numpy(depth_to_4channel(image_np, depth_mm, target_size=(self.image_size, self.image_size)))
                        image_tensor = image_tensor.permute(2, 0, 1).float()
                    elif self.input_channels == 6:
                        # RGBD+Normals: combine RGB + depth + surface normals
                        image_tensor = torch.from_numpy(depth_and_normals_to_6channel(image_np, depth_mm, target_size=(self.image_size, self.image_size)))
                        image_tensor = image_tensor.permute(2, 0, 1).float()
                except Exception as e:
                    logger.warning(f"Failed to load depth for {img_path}: {e}. Using zeros.")
                    # Graceful degradation: pad with zeros if depth loading fails
                    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
                    padding = torch.zeros(self.input_channels - 3, self.image_size, self.image_size)
                    image_tensor = torch.cat([image_tensor, padding], dim=0)
            else:
                # No depth file found; gracefully pad with zeros
                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
                padding = torch.zeros(self.input_channels - 3, self.image_size, self.image_size)
                image_tensor = torch.cat([image_tensor, padding], dim=0)
        else:
            # RGB only
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

        part_idx = self.part_to_idx[part_num]
        color_idx = self.color_to_idx[color_id]

        return image_tensor, part_idx, color_idx


class LegoBrickClassifier(nn.Module):
    """
    Dual-head EfficientNet-B3 classifier for LEGO parts and colors.
    Supports RGB (3-channel), RGBD (4-channel), and RGBD+normals (6-channel) inputs.
    """

    def __init__(self, num_parts: int, num_colors: int, input_channels: int = 3):
        super().__init__()
        self.num_parts = num_parts
        self.num_colors = num_colors
        self.input_channels = input_channels

        # Load pretrained EfficientNet-B3
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

        # Adapt first conv layer for variable input channels
        if input_channels != 3:
            # Get the first conv layer
            first_conv = self.backbone.features[0][0]
            # Create new conv layer with modified input channels
            new_first_conv = nn.Conv2d(
                input_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            # Initialize new input channels with random weights
            # Keep pretrained RGB channels if input_channels > 3
            if input_channels > 3:
                new_first_conv.weight.data[:, :3] = first_conv.weight.data
                # Random init for extra channels
                nn.init.kaiming_normal_(new_first_conv.weight.data[:, 3:], mode='fan_out', nonlinearity='relu')
            else:
                nn.init.kaiming_normal_(new_first_conv.weight.data, mode='fan_out', nonlinearity='relu')
            self.backbone.features[0][0] = new_first_conv

        # Remove default classifier
        self.backbone.classifier = nn.Identity()

        # Feature dimension from EfficientNet-B3
        self.feature_dim = 1536

        # Dual output heads
        self.part_head = nn.Linear(self.feature_dim, num_parts)
        self.color_head = nn.Linear(self.feature_dim, num_colors)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits for both heads."""
        features = self.backbone(x)
        part_logits = self.part_head(features)
        color_logits = self.color_head(features)
        return part_logits, color_logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone."""
        return self.backbone(x)


def mixup_batch(
    images: torch.Tensor,
    part_targets: torch.Tensor,
    color_targets: torch.Tensor,
    alpha: float = 0.4,
    p: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply Mixup augmentation to a batch.

    Args:
        images: Batch of images [B, C, H, W]
        part_targets: Part class indices [B]
        color_targets: Color class indices [B]
        alpha: Beta distribution parameter
        p: Probability of applying mixup

    Returns:
        (mixed_images, mixed_part_targets, mixed_color_targets, lam)
        where targets are soft labels scaled by lam
    """
    if random.random() > p:
        batch_size = images.size(0)
        return images, part_targets.float(), color_targets.float(), 1.0

    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha)

    index = torch.randperm(batch_size).to(images.device)

    mixed_images = lam * images + (1 - lam) * images[index]

    # Soft labels for both heads
    part_targets_a = F.one_hot(part_targets, num_classes=-1).float()
    part_targets_b = F.one_hot(part_targets[index], num_classes=-1).float()
    mixed_part_targets = lam * part_targets_a + (1 - lam) * part_targets_b

    color_targets_a = F.one_hot(color_targets, num_classes=-1).float()
    color_targets_b = F.one_hot(color_targets[index], num_classes=-1).float()
    mixed_color_targets = lam * color_targets_a + (1 - lam) * color_targets_b

    return mixed_images, mixed_part_targets, mixed_color_targets, lam


def cutmix_batch(
    images: torch.Tensor,
    part_targets: torch.Tensor,
    color_targets: torch.Tensor,
    alpha: float = 1.0,
    p: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply CutMix augmentation to a batch.

    Args:
        images: Batch of images [B, C, H, W]
        part_targets: Part class indices [B]
        color_targets: Color class indices [B]
        alpha: Beta distribution parameter
        p: Probability of applying cutmix

    Returns:
        (cut_images, mixed_part_targets, mixed_color_targets, lam)
    """
    if random.random() > p:
        batch_size = images.size(0)
        return images, part_targets.float(), color_targets.float(), 1.0

    batch_size, _, H, W = images.shape
    lam = np.random.beta(alpha, alpha)

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = np.random.randint(0, W)
    cy = np.random.randint(0, H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    index = torch.randperm(batch_size).to(images.device)
    cut_images = images.clone()
    cut_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

    # Recompute lam based on actual box area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))

    # Soft labels
    part_targets_a = F.one_hot(part_targets, num_classes=-1).float()
    part_targets_b = F.one_hot(part_targets[index], num_classes=-1).float()
    mixed_part_targets = lam * part_targets_a + (1 - lam) * part_targets_b

    color_targets_a = F.one_hot(color_targets, num_classes=-1).float()
    color_targets_b = F.one_hot(color_targets[index], num_classes=-1).float()
    mixed_color_targets = lam * color_targets_a + (1 - lam) * color_targets_b

    return cut_images, mixed_part_targets, mixed_color_targets, lam


def softmax_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for soft targets."""
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


class Trainer:
    """Training orchestrator."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        output_dir: Path,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        num_parts: int = None,
        num_colors: int = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.lr = lr

        # Loss functions
        self.focal_part_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.focal_color_loss = FocalLoss(alpha=0.25, gamma=2.0)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # LR scheduler
        total_steps = len(train_loader) * epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='linear'
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=True)

        # Tracking
        self.best_part_acc = 0.0
        self.best_checkpoint_path = None
        self.training_log = []

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_part_loss = 0.0
        total_color_loss = 0.0
        part_correct = 0
        color_correct = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for images, part_targets, color_targets in pbar:
            images = images.to(self.device)
            part_targets = part_targets.to(self.device)
            color_targets = color_targets.to(self.device)

            # Apply Mixup and CutMix
            if random.random() < 0.5:
                images, part_targets_mixed, color_targets_mixed, _ = mixup_batch(
                    images, part_targets, color_targets, alpha=0.4, p=0.3
                )
            else:
                images, part_targets_mixed, color_targets_mixed, _ = cutmix_batch(
                    images, part_targets, color_targets, alpha=1.0, p=0.2
                )

            self.optimizer.zero_grad()

            with autocast(dtype=torch.float16):
                part_logits, color_logits = self.model(images)

                # Compute losses (handle both hard and soft targets)
                if part_targets_mixed.dim() > 1:
                    part_loss = softmax_cross_entropy(part_logits, part_targets_mixed)
                    color_loss = softmax_cross_entropy(color_logits, color_targets_mixed)
                else:
                    part_loss = self.focal_part_loss(part_logits, part_targets)
                    color_loss = self.focal_color_loss(color_logits, color_targets)

                loss = 0.7 * part_loss + 0.3 * color_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_part_loss += part_loss.item()
            total_color_loss += color_loss.item()

            part_preds = part_logits.argmax(dim=1)
            color_preds = color_logits.argmax(dim=1)
            part_correct += (part_preds == part_targets).sum().item()
            color_correct += (color_preds == color_targets).sum().item()
            total_samples += images.size(0)

            pbar.set_postfix({
                'loss': loss.item(),
                'part_acc': part_correct / total_samples,
                'color_acc': color_correct / total_samples
            })

        return {
            'part_loss': total_part_loss / len(self.train_loader),
            'color_loss': total_color_loss / len(self.train_loader),
            'part_acc': part_correct / total_samples,
            'color_acc': color_correct / total_samples,
        }

    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()

        total_part_loss = 0.0
        total_color_loss = 0.0
        part_correct = 0
        color_correct = 0
        part_top5_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, part_targets, color_targets in tqdm(
                self.val_loader, desc='Validating', leave=False
            ):
                images = images.to(self.device)
                part_targets = part_targets.to(self.device)
                color_targets = color_targets.to(self.device)

                with autocast(dtype=torch.float16):
                    part_logits, color_logits = self.model(images)
                    part_loss = self.focal_part_loss(part_logits, part_targets)
                    color_loss = self.focal_color_loss(color_logits, color_targets)

                total_part_loss += part_loss.item()
                total_color_loss += color_loss.item()

                part_preds = part_logits.argmax(dim=1)
                color_preds = color_logits.argmax(dim=1)
                part_correct += (part_preds == part_targets).sum().item()
                color_correct += (color_preds == color_targets).sum().item()

                # Top-5 accuracy for parts
                _, top5_preds = part_logits.topk(5, 1, True, True)
                top5_correct = top5_preds.eq(part_targets.view(-1, 1).expand_as(top5_preds))
                part_top5_correct += top5_correct.any(dim=1).sum().item()

                total_samples += images.size(0)

        return {
            'part_loss': total_part_loss / len(self.val_loader),
            'color_loss': total_color_loss / len(self.val_loader),
            'part_acc': part_correct / total_samples,
            'color_acc': color_correct / total_samples,
            'part_top5_acc': part_top5_correct / total_samples,
        }

    def test(self) -> Dict[str, float]:
        """Evaluate on test set."""
        self.model.eval()

        total_part_loss = 0.0
        total_color_loss = 0.0
        part_correct = 0
        color_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, part_targets, color_targets in tqdm(
                self.test_loader, desc='Testing', leave=False
            ):
                images = images.to(self.device)
                part_targets = part_targets.to(self.device)
                color_targets = color_targets.to(self.device)

                with autocast(dtype=torch.float16):
                    part_logits, color_logits = self.model(images)
                    part_loss = self.focal_part_loss(part_logits, part_targets)
                    color_loss = self.focal_color_loss(color_logits, color_targets)

                total_part_loss += part_loss.item()
                total_color_loss += color_loss.item()

                part_preds = part_logits.argmax(dim=1)
                color_preds = color_logits.argmax(dim=1)
                part_correct += (part_preds == part_targets).sum().item()
                color_correct += (color_preds == color_targets).sum().item()
                total_samples += images.size(0)

        return {
            'part_loss': total_part_loss / len(self.test_loader),
            'color_loss': total_color_loss / len(self.test_loader),
            'part_acc': part_correct / total_samples,
            'color_acc': color_correct / total_samples,
        }

    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        start_epoch = 0

        if resume_from:
            logger.info(f"Resuming from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            self.best_part_acc = checkpoint.get('best_part_acc', 0.0)
            self.training_log = checkpoint.get('training_log', [])

        for epoch in range(start_epoch, self.epochs):
            logger.info(f"\n=== Epoch {epoch + 1}/{self.epochs} ===")

            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train - Part Loss: {train_metrics['part_loss']:.4f}, "
                       f"Part Acc: {train_metrics['part_acc']:.4f}, "
                       f"Color Acc: {train_metrics['color_acc']:.4f}")

            # Validate
            val_metrics = self.validate()
            logger.info(f"Val - Part Loss: {val_metrics['part_loss']:.4f}, "
                       f"Part Acc: {val_metrics['part_acc']:.4f}, "
                       f"Color Acc: {val_metrics['color_acc']:.4f}, "
                       f"Part Top5: {val_metrics['part_top5_acc']:.4f}")

            # Log to JSONL
            log_entry = {
                'epoch': epoch,
                'train_loss': train_metrics['part_loss'] + train_metrics['color_loss'],
                'val_part_acc': val_metrics['part_acc'],
                'val_color_acc': val_metrics['color_acc'],
                'val_top5_acc': val_metrics['part_top5_acc'],
                'lr': self.scheduler.get_last_lr()[0],
            }
            self.training_log.append(log_entry)

            # Save best checkpoint
            if val_metrics['part_acc'] > self.best_part_acc:
                self.best_part_acc = val_metrics['part_acc']
                self.best_checkpoint_path = self.output_dir / f'best_model.pt'

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_part_acc': self.best_part_acc,
                    'training_log': self.training_log,
                }
                torch.save(checkpoint, self.best_checkpoint_path)
                logger.info(f"Best checkpoint saved to {self.best_checkpoint_path}")

        # Save training log
        log_file = self.output_dir / 'training_log.jsonl'
        with open(log_file, 'w') as f:
            for entry in self.training_log:
                f.write(json.dumps(entry) + '\n')
        logger.info(f"Training log saved to {log_file}")

        # Test on test set
        logger.info("\n=== Final Test ===")
        test_metrics = self.test()
        logger.info(f"Test - Part Loss: {test_metrics['part_loss']:.4f}, "
                   f"Part Acc: {test_metrics['part_acc']:.4f}, "
                   f"Color Acc: {test_metrics['color_acc']:.4f}")


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    num_parts: int,
    num_colors: int,
    device: torch.device = torch.device('cpu'),
):
    """
    Export model to ONNX format.

    Args:
        model: Trained model
        output_path: Path to save ONNX file
        num_parts: Number of part classes
        num_colors: Number of color classes
        device: Device to use for export
    """
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 300, 300, device=device)

    # Export with custom outputs
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['images'],
        output_names=['part_logits', 'color_logits'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'part_logits': {0: 'batch_size'},
            'color_logits': {0: 'batch_size'},
        },
        opset_version=14,
        do_constant_folding=True,
        verbose=False,
    )
    logger.info(f"Model exported to ONNX: {output_path}")


def tta_predict(
    image_path: str,
    model: nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Test-Time Augmentation inference.

    Runs inference on 7 versions of the image (original, h-flip, v-flip, +15°, -15°,
    brightness+10%, brightness-10%) and averages softmax outputs.

    Args:
        image_path: Path to input image
        model: Trained model
        device: Device to use

    Returns:
        (averaged_part_logits, averaged_color_logits)
    """
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    image_np = np.array(Image.fromarray(image_np).resize((300, 300), Image.BILINEAR))

    # Define TTA transforms
    tta_transforms = [
        A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        A.Compose([A.HorizontalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        A.Compose([A.VerticalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        A.Compose([A.Rotate(limit=15, p=1.0, border_mode=0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        A.Compose([A.Rotate(limit=-15, p=1.0, border_mode=0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        A.Compose([A.RandomBrightnessContrast(brightness_limit=-0.1, contrast_limit=0, p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    ]

    part_probs_list = []
    color_probs_list = []

    with torch.no_grad():
        for transform in tta_transforms:
            augmented = transform(image=image_np)
            aug_image = augmented['image']

            tensor = torch.from_numpy(aug_image).permute(2, 0, 1).float().unsqueeze(0).to(device)

            part_logits, color_logits = model(tensor)
            part_probs = F.softmax(part_logits, dim=1)
            color_probs = F.softmax(color_logits, dim=1)

            part_probs_list.append(part_probs)
            color_probs_list.append(color_probs)

    # Average probabilities
    avg_part_probs = torch.stack(part_probs_list).mean(dim=0)
    avg_color_probs = torch.stack(color_probs_list).mean(dim=0)

    return avg_part_probs, avg_color_probs


def main():
    parser = argparse.ArgumentParser(
        description='Train BrickScan stage 2: dual-head brick classifier'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to data root directory')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Where to save checkpoints and exports')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--export-only', type=str, default=None,
                        help='Skip training and just export a checkpoint to ONNX')
    parser.add_argument('--channels', type=int, default=3, choices=[3, 4, 6],
                        help='Input channels: 3=RGB, 4=RGBD, 6=RGBD+normals')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f"Using device: {device}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    logger.info(f"Loading datasets with {args.channels} input channels...")
    train_dataset = BrickClassifierDataset(
        args.data_dir,
        split='train',
        image_size=300,
        use_mixup=True,
        use_cutmix=True,
        input_channels=args.channels,
    )

    val_dataset = BrickClassifierDataset(
        args.data_dir,
        split='val',
        image_size=300,
        use_mixup=False,
        use_cutmix=False,
        input_channels=args.channels,
        part_to_idx=train_dataset.part_to_idx,
        color_to_idx=train_dataset.color_to_idx,
    )

    test_dataset = BrickClassifierDataset(
        args.data_dir,
        split='test',
        image_size=300,
        use_mixup=False,
        use_cutmix=False,
        input_channels=args.channels,
        part_to_idx=train_dataset.part_to_idx,
        color_to_idx=train_dataset.color_to_idx,
    )

    num_parts = len(train_dataset.part_to_idx)
    num_colors = len(train_dataset.color_to_idx)
    logger.info(f"Num parts: {num_parts}, Num colors: {num_colors}")

    # Save class maps
    class_map = {
        'part_to_idx': train_dataset.part_to_idx,
        'color_to_idx': train_dataset.color_to_idx,
        'idx_to_part': {str(v): k for k, v in train_dataset.part_to_idx.items()},
        'idx_to_color': {str(v): k for k, v in train_dataset.color_to_idx.items()},
    }
    with open(output_dir / 'class_map.json', 'w') as f:
        json.dump(class_map, f, indent=2)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Model
    model = LegoBrickClassifier(num_parts, num_colors, input_channels=args.channels)
    logger.info(f"Model created with {args.channels} input channels")

    if args.export_only:
        logger.info(f"Loading checkpoint from {args.export_only}")
        checkpoint = torch.load(args.export_only, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        export_path = output_dir / 'brickscan_classifier.onnx'
        export_to_onnx(model, str(export_path), num_parts, num_colors, device)
        logger.info("Export complete!")
        return

    # Trainer
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        output_dir,
        epochs=args.epochs,
        lr=args.lr,
        num_parts=num_parts,
        num_colors=num_colors,
    )

    # Train
    trainer.train(resume_from=args.resume)

    # Export best model to ONNX
    logger.info("Exporting best model to ONNX...")
    best_checkpoint = torch.load(trainer.best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    export_path = output_dir / 'brickscan_classifier.onnx'
    export_to_onnx(model, str(export_path), num_parts, num_colors, device)

    logger.info("\nTraining complete!")
    logger.info(f"Best checkpoint: {trainer.best_checkpoint_path}")
    logger.info(f"ONNX model: {export_path}")
    logger.info(f"Class map: {output_dir / 'class_map.json'}")


if __name__ == '__main__':
    main()
