"""Training script for LEGO brick classifier."""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tensorboard.compat.tensorflow_stub import io as tf_io
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from dataset import LegoPartsDataset, build_label_encoders, save_label_encoders
from model import LegoBrickClassifier


class Trainer:
    """Trainer class for LEGO brick classifier."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: Path,
        epochs: int = 50,
        lr: float = 1e-4,
        use_amp: bool = True,
        patience: int = 15
    ):
        """
        Initialize trainer.

        Args:
            model: Neural network model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: torch.device (CPU or CUDA)
            output_dir: Directory to save checkpoints and logs
            epochs: Number of training epochs
            lr: Learning rate
            use_amp: Whether to use mixed precision training
            patience: Early stopping patience
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.lr = lr
        self.use_amp = use_amp

        # Loss functions
        self.part_loss_fn = nn.CrossEntropyLoss()
        self.color_loss_fn = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=use_amp)

        # LR scheduler: OneCycleLR
        total_steps = len(train_loader) * epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='linear'
        )

        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'logs')

        # Early stopping
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_checkpoint = None

    def train_epoch(self) -> Tuple[float, float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (total_loss, part_accuracy, color_accuracy)
        """
        self.model.train()
        total_loss = 0.0
        part_correct = 0
        color_correct = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch_idx, (images, part_labels, color_labels) in enumerate(pbar):
            images = images.to(self.device)
            part_labels = part_labels.to(self.device)
            color_labels = color_labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                part_logits, color_logits = self.model(images)

                # Combined loss
                part_loss = self.part_loss_fn(part_logits, part_labels)
                color_loss = self.color_loss_fn(color_logits, color_labels)
                loss = 0.6 * part_loss + 0.4 * color_loss

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # Metrics
            total_loss += loss.item()
            part_preds = part_logits.argmax(dim=1)
            color_preds = color_logits.argmax(dim=1)
            part_correct += (part_preds == part_labels).sum().item()
            color_correct += (color_preds == color_labels).sum().item()
            total_samples += images.size(0)

            pbar.set_postfix({
                'loss': loss.item(),
                'part_acc': part_correct / total_samples,
                'color_acc': color_correct / total_samples
            })

        avg_loss = total_loss / len(self.train_loader)
        part_accuracy = part_correct / total_samples
        color_accuracy = color_correct / total_samples

        return avg_loss, part_accuracy, color_accuracy

    def validate(self) -> Tuple[float, float, float]:
        """
        Validate on the validation set.

        Returns:
            Tuple of (total_loss, part_accuracy, color_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        part_correct = 0
        color_correct = 0
        total_samples = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating', leave=False)
            for images, part_labels, color_labels in pbar:
                images = images.to(self.device)
                part_labels = part_labels.to(self.device)
                color_labels = color_labels.to(self.device)

                with autocast(enabled=self.use_amp):
                    part_logits, color_logits = self.model(images)
                    part_loss = self.part_loss_fn(part_logits, part_labels)
                    color_loss = self.color_loss_fn(color_logits, color_labels)
                    loss = 0.6 * part_loss + 0.4 * color_loss

                total_loss += loss.item()
                part_preds = part_logits.argmax(dim=1)
                color_preds = color_logits.argmax(dim=1)
                part_correct += (part_preds == part_labels).sum().item()
                color_correct += (color_preds == color_labels).sum().item()
                total_samples += images.size(0)

                pbar.set_postfix({
                    'loss': loss.item(),
                    'part_acc': part_correct / total_samples,
                    'color_acc': color_correct / total_samples
                })

        avg_loss = total_loss / len(self.val_loader)
        part_accuracy = part_correct / total_samples
        color_accuracy = color_correct / total_samples

        return avg_loss, part_accuracy, color_accuracy

    def train(self, resume_from: str = None):
        """
        Train the model for the specified number of epochs.

        Args:
            resume_from: Path to checkpoint to resume from
        """
        start_epoch = 0

        if resume_from:
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            # Do NOT restore scheduler state — rebuild it for the remaining epochs
            # so OneCycleLR total_steps matches the actual steps left to run.
            start_epoch = checkpoint['epoch'] + 1
            remaining_epochs = self.epochs - start_epoch
            remaining_steps = len(self.train_loader) * remaining_epochs
            if remaining_steps > 0:
                self.scheduler = OneCycleLR(
                    self.optimizer,
                    max_lr=self.lr * 0.5,   # lower peak LR for fine-tuning
                    total_steps=remaining_steps,
                    pct_start=0.05,
                    anneal_strategy='cos'
                )
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed from epoch {start_epoch}, {remaining_epochs} epochs remaining")

        for epoch in range(start_epoch, self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            # Training
            train_loss, train_part_acc, train_color_acc = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f} | Part Acc: {train_part_acc:.4f} | Color Acc: {train_color_acc:.4f}")

            # Validation
            val_loss, val_part_acc, val_color_acc = self.validate()
            print(f"Val Loss: {val_loss:.4f} | Part Acc: {val_part_acc:.4f} | Color Acc: {val_color_acc:.4f}")

            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/part/train', train_part_acc, epoch)
            self.writer.add_scalar('Accuracy/part/val', val_part_acc, epoch)
            self.writer.add_scalar('Accuracy/color/train', train_color_acc, epoch)
            self.writer.add_scalar('Accuracy/color/val', val_color_acc, epoch)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch + 1}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                    'best_val_loss': self.best_val_loss
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

            # Early stopping and best model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_checkpoint = self.output_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                    'best_val_loss': self.best_val_loss
                }, self.best_checkpoint)
                print(f"Saved best model: {self.best_checkpoint}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {self.patience} epochs without improvement")
                    break

        self.writer.close()
        print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description='Train LEGO brick classifier')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory containing index.csv and images/')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for models and logs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--val-split', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (default: 15)')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build label encoders
    csv_path = data_dir / 'index.csv'
    print(f"Building label encoders from {csv_path}")
    part_encoder, color_encoder = build_label_encoders(str(csv_path))
    save_label_encoders(part_encoder, color_encoder, str(output_dir))

    print(f"Number of parts: {len(part_encoder)}")
    print(f"Number of colors: {len(color_encoder)}")

    # Create full dataset for train/val split
    full_dataset = LegoPartsDataset(
        csv_path=str(csv_path),
        images_dir=str(data_dir / 'images'),
        part_encoder=part_encoder,
        color_encoder=color_encoder,
        is_train=True
    )

    # Train/val split by indices
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=args.val_split,
        random_state=42
    )

    # Create subsets
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    # Create separate datasets with proper transforms
    train_dataset = LegoPartsDataset(
        csv_path=str(csv_path),
        images_dir=str(data_dir / 'images'),
        part_encoder=part_encoder,
        color_encoder=color_encoder,
        is_train=True
    )
    train_dataset = Subset(train_dataset, train_indices)

    val_dataset = LegoPartsDataset(
        csv_path=str(csv_path),
        images_dir=str(data_dir / 'images'),
        part_encoder=part_encoder,
        color_encoder=color_encoder,
        is_train=False
    )
    val_dataset = Subset(val_dataset, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # Create model
    model = LegoBrickClassifier(
        num_parts=len(part_encoder),
        num_colors=len(color_encoder)
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        epochs=args.epochs,
        lr=args.lr,
        use_amp=True,
        patience=args.patience
    )

    # Train
    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()
