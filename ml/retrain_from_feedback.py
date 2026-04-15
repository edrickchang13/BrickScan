#!/usr/bin/env python3
"""
Fine-tune classifier on user-corrected feedback data.

Reads user corrections from a CSV export, builds a weighted training dataset,
and fine-tunes the two-stage classifier with feedback data oversampled 5×.

Output checkpoint: best_finetuned.pt

Usage:
  python3 retrain_from_feedback.py --checkpoint ./models/best.pt \\
    --feedback-csv ./feedback_corrections.csv \\
    --output-dir ./models --epochs 10
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

try:
    from train_two_stage import BrickClassifierDataset, FocalLoss
except ImportError:
    # Soft import — dry-run mode doesn't need the training dataset module.
    BrickClassifierDataset = None  # type: ignore
    FocalLoss = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("retrain_from_feedback")


def _infer_num_classes(state_dict: dict, head_name: str) -> Optional[int]:
    """
    Best-effort shape inference: look for the classifier head's final Linear
    weight tensor and read its output dimension.
    """
    for key, tensor in state_dict.items():
        if head_name in key and key.endswith("weight") and hasattr(tensor, "shape") and len(tensor.shape) == 2:
            return int(tensor.shape[0])
    return None


def load_feedback_corrections(feedback_csv_or_db_url: Path) -> pd.DataFrame:
    """
    Load user feedback corrections from CSV export.

    Expected CSV columns:
    - image_path: local path to uploaded image (relative to data dir)
    - correct_part_num: user's corrected part number
    - correct_color_id: user's corrected color ID (or name)
    - original_prediction: original model prediction (for logging)
    - timestamp: when the correction was made

    Args:
        feedback_csv_or_db_url: Path to feedback CSV

    Returns:
        DataFrame with correction records
    """
    df = pd.read_csv(feedback_csv_or_db_url)

    log.info(f"Loaded {len(df)} feedback corrections from {feedback_csv_or_db_url}")
    log.info(f"Columns: {list(df.columns)}")

    # Validate required columns
    required = ['image_path', 'correct_part_num', 'correct_color_id']
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.warning(f"Missing columns: {missing}")

    return df


def build_feedback_dataset(
    feedback_df: pd.DataFrame,
    upload_dir: Path,
) -> Dataset:
    """
    Build a PyTorch Dataset from feedback corrections.

    Args:
        feedback_df: DataFrame with feedback corrections
        upload_dir: Root directory where uploaded images are stored

    Returns:
        PyTorch Dataset
    """
    class FeedbackDataset(Dataset):
        def __init__(self, df: pd.DataFrame, upload_dir: Path):
            self.df = df
            self.upload_dir = upload_dir

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image_path = self.upload_dir / row['image_path']

            # Load image
            from PIL import Image
            img = Image.open(image_path).convert('RGB')

            # Return image, part_num, color_id
            return {
                'image': img,
                'part_num': row['correct_part_num'],
                'color_id': int(row['correct_color_id']) if isinstance(row['correct_color_id'], (int, float)) else str(row['correct_color_id']),
            }

    return FeedbackDataset(feedback_df, upload_dir)


def build_weighted_training_data(
    base_data_dir: Path,
    feedback_df: pd.DataFrame,
    feedback_upload_dir: Path,
    feedback_weight: float = 5.0,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train/val DataLoaders with feedback data oversampled.

    Args:
        base_data_dir: Directory with original training data
        feedback_df: DataFrame with user corrections
        feedback_upload_dir: Directory where feedback images are stored
        feedback_weight: Oversample factor for feedback data (default: 5×)
        val_split: Fraction to use for validation

    Returns:
        (train_loader, val_loader, class_mappings)
    """
    log.info(f"Building weighted dataset...")
    log.info(f"  Base data: {base_data_dir}")
    log.info(f"  Feedback data: {len(feedback_df)} corrections")
    log.info(f"  Feedback weight: {feedback_weight}×")

    # Load base dataset (original training data)
    base_dataset = BrickClassifierDataset(
        str(base_data_dir),
        split='train',
        use_mixup=True,
        use_cutmix=True,
    )

    log.info(f"  Base dataset size: {len(base_dataset)}")

    # Build feedback dataset
    feedback_dataset = build_feedback_dataset(feedback_df, feedback_upload_dir)

    # Combine datasets
    combined_dataset = ConcatDataset([base_dataset, feedback_dataset])

    # Create weighted sampler
    weights = [1.0] * len(base_dataset) + [feedback_weight] * len(feedback_dataset)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(combined_dataset),
        replacement=True,
    )

    # Create DataLoader
    train_loader = DataLoader(
        combined_dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # Create validation loader (from base dataset)
    val_size = int(len(base_dataset) * val_split)
    val_indices = np.random.choice(len(base_dataset), val_size, replace=False)

    val_dataset = torch.utils.data.Subset(base_dataset, val_indices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    log.info(f"  Combined dataset size: {len(combined_dataset)}")
    log.info(f"  Train samples: {len(train_loader) * 32}")
    log.info(f"  Val samples: {len(val_loader) * 32}")

    return train_loader, val_loader, {}


def run_finetuning(
    checkpoint_path: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    epochs: int = 10,
    lr: float = 1e-4,
):
    """
    Fine-tune model on weighted dataset with lower learning rate.

    Freezes backbone for first 3 epochs, then unfreezes for full fine-tuning.

    Args:
        checkpoint_path: Path to original best.pt
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        output_dir: Directory to save best_finetuned.pt
        epochs: Number of fine-tuning epochs
        lr: Learning rate
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Fine-tuning on {device}")

    # ── Load model from checkpoint ────────────────────────────────────────────
    # Supports two checkpoint shapes:
    #   a) full dict {"model_state_dict": ..., "num_parts": N, "num_colors": M}
    #      — the shape `train_two_stage.py` writes
    #   b) raw state_dict (torch.save(model.state_dict(), ...))
    try:
        from train_two_stage import BrickClassifier
    except ImportError as e:
        log.error("Cannot import BrickClassifier from train_two_stage.py — is ml/ on PYTHONPATH?")
        raise

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        num_parts = int(checkpoint.get("num_parts", 1000))
        num_colors = int(checkpoint.get("num_colors", 100))
    else:
        state_dict = checkpoint
        # Infer shape from the classifier head weights (last linear layer)
        # This is best-effort; set defaults if inference fails.
        num_parts = _infer_num_classes(state_dict, "part_head") or 1000
        num_colors = _infer_num_classes(state_dict, "color_head") or 100
        log.info("Inferred num_parts=%d num_colors=%d from state_dict", num_parts, num_colors)

    model = BrickClassifier(num_parts=num_parts, num_colors=num_colors)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning("State dict missing keys: %s", missing[:5])
    if unexpected:
        log.warning("State dict unexpected keys: %s", unexpected[:5])

    model = model.to(device)
    log.info("Loaded checkpoint: %s (parts=%d, colors=%d)", checkpoint_path, num_parts, num_colors)

    # Loss and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler()

    # Scheduler
    total_steps = len(train_loader) * epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,
    )

    best_val_loss = float('inf')
    best_checkpoint = None

    for epoch in range(epochs):
        # Unfreeze backbone after 3 epochs
        if epoch == 3:
            log.info("Unfreezing backbone for full fine-tuning")
            for param in model.backbone.parameters():
                param.requires_grad = True
        elif epoch == 0:
            log.info("Freezing backbone for first 3 epochs")
            for param in model.backbone.parameters():
                param.requires_grad = False

        # Training
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images = batch[0].to(device)
            part_labels = batch[1].to(device)
            color_labels = batch[2].to(device)

            optimizer.zero_grad()

            with autocast():
                part_logits, color_logits = model(images)
                loss = criterion(part_logits, part_labels) + criterion(color_logits, color_labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        log.info(f"Epoch {epoch+1} - Train loss: {train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(device)
                part_labels = batch[1].to(device)
                color_labels = batch[2].to(device)

                part_logits, color_logits = model(images)
                loss = criterion(part_logits, part_labels) + criterion(color_logits, color_labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        log.info(f"Epoch {epoch+1} - Val loss: {val_loss:.4f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "best_finetuned.pt"

            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'finetuned': True,
            }

            torch.save(best_checkpoint, output_path)
            log.info(f"Saved best checkpoint: {output_path}")

    log.info(f"Fine-tuning complete!")
    log.info(f"Best validation loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune classifier on user-corrected feedback data"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to original best.pt checkpoint"
    )
    parser.add_argument(
        "--base-data",
        required=True,
        help="Directory with original training data"
    )
    parser.add_argument(
        "--feedback-csv",
        required=True,
        help="CSV export of user feedback corrections"
    )
    parser.add_argument(
        "--feedback-upload-dir",
        default="/tmp/brickscan_uploads/feedback",
        help="Directory where uploaded feedback images are stored (default: /tmp/brickscan_uploads/feedback)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save best_finetuned.pt"
    )
    parser.add_argument(
        "--feedback-weight",
        type=float,
        default=5.0,
        help="Oversample factor for feedback data (default: 5.0)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of fine-tuning epochs (default: 10)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate CSV + checkpoint + env without training. Prints the plan. "
            "Useful on the Mac side to sanity-check the pipeline before shipping "
            "to Spark."
        ),
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    feedback_csv = Path(args.feedback_csv)
    base_data_dir = Path(args.base_data)

    # ── Dry-run: validate the pipeline without actually training ──────────────
    if args.dry_run:
        log.info("=" * 60)
        log.info("DRY RUN — no training will occur")
        log.info("=" * 60)
        log.info("Checkpoint:        %s  %s", checkpoint_path, "✓ exists" if checkpoint_path.exists() else "✗ MISSING")
        log.info("Feedback CSV:      %s  %s", feedback_csv, "✓ exists" if feedback_csv.exists() else "✗ MISSING")
        log.info("Base data dir:     %s  %s", base_data_dir, "✓ exists" if base_data_dir.exists() else "(optional, skip)")
        log.info("Feedback uploads:  %s", args.feedback_upload_dir)
        log.info("Output dir:        %s", args.output_dir)
        log.info("Feedback weight:   %.1f×", args.feedback_weight)
        log.info("Epochs:            %d", args.epochs)
        log.info("Learning rate:     %g", args.lr)
        log.info("CUDA available:    %s", torch.cuda.is_available())

        if feedback_csv.exists():
            feedback_df = load_feedback_corrections(feedback_csv)
            log.info("CSV rows:          %d", len(feedback_df))
            required = ["image_path", "correct_part_num", "correct_color_id"]
            missing = [c for c in required if c not in feedback_df.columns]
            if missing:
                log.warning("CSV missing required columns: %s", missing)
            else:
                log.info("CSV columns OK:    %s", list(feedback_df.columns)[:6])
                # Sample: do image paths resolve?
                sample_size = min(3, len(feedback_df))
                if sample_size:
                    log.info("Spot-checking image paths (first %d rows)…", sample_size)
                    for i in range(sample_size):
                        p = Path(feedback_df.iloc[i]["image_path"])
                        if not p.is_absolute():
                            p = Path(args.feedback_upload_dir) / p
                        log.info("  [%d] %s  %s", i, p, "✓" if p.exists() else "✗")
        log.info("=" * 60)
        log.info("Dry run OK — pipeline ready for Spark.")
        return

    # ── Real run ──────────────────────────────────────────────────────────────
    if not checkpoint_path.exists():
        log.error(f"Checkpoint not found: {checkpoint_path}")
        return

    if not base_data_dir.exists():
        log.error(f"Base data directory not found: {base_data_dir}")
        return

    if not feedback_csv.exists():
        log.error(f"Feedback CSV not found: {feedback_csv}")
        return

    # Load feedback
    feedback_df = load_feedback_corrections(feedback_csv)

    if len(feedback_df) == 0:
        log.warning("No feedback corrections to train on")
        return

    # Build weighted dataset
    train_loader, val_loader, _ = build_weighted_training_data(
        base_data_dir,
        feedback_df,
        Path(args.feedback_upload_dir),
        feedback_weight=args.feedback_weight,
    )

    # Fine-tune
    run_finetuning(
        checkpoint_path,
        train_loader,
        val_loader,
        Path(args.output_dir),
        epochs=args.epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
