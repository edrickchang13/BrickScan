"""
Training script for MultiViewPooling model.

Fine-tunes the attention pooling layer on sequences of augmented views
of the same part. Data augmentation creates multi-view sequences from
single-part images (rotation, lighting, slight translation).

Usage:
    python train_multi_view.py \
        --data-dir ./data/parts \
        --output-dir ./checkpoints \
        --epochs 20 \
        --batch-size 16 \
        --num-views 8
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple
import logging
from multi_view_pooling import MultiViewPooling

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiViewPartDataset(Dataset):
    """
    Dataset of multi-view LEGO part sequences.

    Expects directory structure:
        data/parts/
            3002/  (part number)
                image_1.jpg
                image_2.jpg
                ...
            3003/
                ...

    Data augmentation creates multi-view sequences by applying random
    rotations, lighting adjustments, and slight translations.
    """

    def __init__(
        self,
        data_dir: Path,
        num_views: int = 8,
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.num_views = num_views
        self.image_size = image_size
        self.augment = augment

        # Index parts by directory
        self.part_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        self.part_to_idx = {d.name: i for i, d in enumerate(self.part_dirs)}

        logger.info(f"Found {len(self.part_dirs)} parts")

    def __len__(self) -> int:
        return len(self.part_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Return a sequence of augmented views of the same part.

        Returns:
            (view_sequence, part_idx) where view_sequence is [num_views, 3, H, W]
        """
        part_dir = self.part_dirs[idx]
        part_idx = self.part_to_idx[part_dir.name]

        # Load images from this part
        image_paths = sorted(list(part_dir.glob('*.jpg')) + list(part_dir.glob('*.png')))
        if not image_paths:
            logger.warning(f"No images found in {part_dir}")
            return torch.zeros(self.num_views, 3, *self.image_size), part_idx

        # Load and augment images
        views = []
        for i in range(self.num_views):
            # Cycle through available images
            img_path = image_paths[i % len(image_paths)]

            try:
                from PIL import Image
                import torchvision.transforms as transforms

                img = Image.open(img_path).convert('RGB')

                # Augmentation: random rotation, brightness, contrast
                if self.augment:
                    augment_transform = transforms.Compose([
                        transforms.RandomRotation(degrees=15),
                        transforms.ColorJitter(
                            brightness=0.2,
                            contrast=0.2,
                            saturation=0.1,
                            hue=0.05
                        ),
                        transforms.RandomAffine(
                            degrees=0,
                            translate=(0.05, 0.05),
                        ),
                    ])
                    img = augment_transform(img)

                # Resize and normalize
                resize_transform = transforms.Compose([
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
                view = resize_transform(img)
                views.append(view)

            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                views.append(torch.zeros(3, *self.image_size))

        view_sequence = torch.stack(views)  # [num_views, 3, H, W]
        return view_sequence, part_idx


def train_epoch(
    model: MultiViewPooling,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    """Train one epoch and return average part and color loss."""
    model.train()
    total_part_loss = 0.0
    total_color_loss = 0.0
    total_samples = 0

    part_criterion = nn.CrossEntropyLoss()
    color_criterion = nn.CrossEntropyLoss()

    for batch_idx, (view_sequences, part_indices) in enumerate(dataloader):
        batch_size = view_sequences.shape[0]
        view_sequences = view_sequences.to(device)  # [B, num_views, 3, H, W]
        part_indices = part_indices.to(device)  # [B]

        # Flatten batch and views for feature extraction
        B, V, C, H, W = view_sequences.shape
        flattened = view_sequences.view(B * V, C, H, W)

        # Extract features (in practice, use backbone)
        # For now, this is a placeholder — actual training would extract via backbone
        with torch.no_grad():
            # In real training, extract features from EfficientNet backbone
            # For stub, create dummy features
            features = torch.randn(B * V, model.feature_dim, device=device)

        features = features.view(B, V, model.feature_dim)

        # Forward pass through pooling
        # Process each sequence in the batch
        part_logits_list = []
        color_logits_list = []

        for i in range(B):
            part_logits, color_logits = model(features[i])
            part_logits_list.append(part_logits)
            color_logits_list.append(color_logits)

        part_logits = torch.cat(part_logits_list, dim=0)  # [B, num_parts]
        color_logits = torch.cat(color_logits_list, dim=0)  # [B, num_colors]

        # Compute loss (color loss is stubbed — no color labels in this example)
        part_loss = part_criterion(part_logits, part_indices)
        # color_loss would require color labels
        color_loss = torch.tensor(0.0, device=device)

        loss = part_loss + color_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_part_loss += part_loss.item() * batch_size
        total_color_loss += color_loss.item() * batch_size
        total_samples += batch_size

        if batch_idx % 10 == 0:
            logger.info(
                f"Batch {batch_idx}: part_loss={part_loss.item():.4f}, "
                f"color_loss={color_loss.item():.4f}"
            )

    avg_part_loss = total_part_loss / total_samples if total_samples > 0 else 0.0
    avg_color_loss = total_color_loss / total_samples if total_samples > 0 else 0.0

    return avg_part_loss, avg_color_loss


def main():
    parser = argparse.ArgumentParser(description='Train MultiViewPooling model')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to parts dataset')
    parser.add_argument('--output-dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-views', type=int, default=8, help='Number of views per part')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--num-parts', type=int, default=6500, help='Number of part classes')
    parser.add_argument('--num-colors', type=int, default=150, help='Number of color classes')

    args = parser.parse_args()

    # Setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using device: {device}")
    logger.info(f"Output directory: {output_dir}")

    # Create model
    model = MultiViewPooling(
        num_parts=args.num_parts,
        num_colors=args.num_colors,
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataset
    dataset = MultiViewPartDataset(
        data_dir=args.data_dir,
        num_views=args.num_views,
        augment=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    logger.info(f"Dataset: {len(dataset)} parts")

    # Training loop
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        avg_part_loss, avg_color_loss = train_epoch(model, dataloader, optimizer, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"part_loss={avg_part_loss:.4f}, color_loss={avg_color_loss:.4f}, "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = output_dir / f"multiview_epoch{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Final checkpoint
    final_path = output_dir / "multiview_final.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training complete. Final model saved to {final_path}")


if __name__ == '__main__':
    main()
