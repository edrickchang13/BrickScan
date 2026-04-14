"""Evaluation script for LEGO brick classifier."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LegoPartsDataset, build_label_encoders
from model import LegoBrickClassifier


def get_inverse_encoder(encoder: Dict[str, int]) -> Dict[int, str]:
    """Inverse mapping from index to label."""
    return {idx: label for label, idx in encoder.items()}


def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    part_encoder: Dict[str, int],
    color_encoder: Dict[str, int]
) -> Tuple[float, float, float, float, np.ndarray, List[str]]:
    """
    Evaluate model on validation set.

    Returns:
        Tuple of (part_top1, part_top5, color_top1, color_accuracy, confusion_matrix, top_parts)
    """
    model.eval()
    part_top1_correct = 0
    part_top5_correct = 0
    color_correct = 0
    total_samples = 0

    part_predictions = []
    part_ground_truth = []
    color_predictions = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Evaluating', leave=False)
        for images, part_labels, color_labels in pbar:
            images = images.to(device)
            part_labels = part_labels.to(device)
            color_labels = color_labels.to(device)

            part_logits, color_logits = model(images)

            # Part classification metrics
            part_preds_top1 = part_logits.argmax(dim=1)
            part_preds_top5 = torch.topk(part_logits, k=5, dim=1)[1]

            part_top1_correct += (part_preds_top1 == part_labels).sum().item()
            part_top5_correct += (part_preds_top5 == part_labels.unsqueeze(1)).any(dim=1).sum().item()

            # Color classification metrics
            color_preds = color_logits.argmax(dim=1)
            color_correct += (color_preds == color_labels).sum().item()

            # Store predictions for confusion matrix
            part_predictions.extend(part_preds_top1.cpu().numpy())
            part_ground_truth.extend(part_labels.cpu().numpy())
            color_predictions.extend(color_preds.cpu().numpy())

            total_samples += images.size(0)

            pbar.set_postfix({
                'part_top1': part_top1_correct / total_samples,
                'part_top5': part_top5_correct / total_samples,
                'color': color_correct / total_samples
            })

    part_top1 = part_top1_correct / total_samples
    part_top5 = part_top5_correct / total_samples
    color_top1 = color_correct / total_samples

    # Confusion matrix for top-20 most common parts
    part_inverse = get_inverse_encoder(part_encoder)
    part_names = [part_inverse[i] for i in range(len(part_encoder))]

    # Get top-20 most common parts in ground truth
    unique_parts, counts = np.unique(part_ground_truth, return_counts=True)
    top_20_parts_idx = unique_parts[np.argsort(-counts)[:20]]
    top_20_parts = [part_names[i] for i in top_20_parts_idx]

    # Build confusion matrix for top-20
    mask = np.isin(part_ground_truth, top_20_parts_idx)
    filtered_gt = [p for p, m in zip(part_ground_truth, mask) if m]
    filtered_pred = [p for p, m in zip(part_predictions, mask) if m]

    cm = confusion_matrix(filtered_gt, top_20_parts_idx)

    return part_top1, part_top5, color_top1, cm, top_20_parts


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path):
    """Plot and save confusion matrix as PNG."""
    fig, ax = plt.subplots(figsize=(16, 14))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='Ground Truth',
        xlabel='Predictions'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black',
                fontsize=8
            )

    fig.tight_layout()
    plt.savefig(str(output_path), dpi=100, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate LEGO brick classifier')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build label encoders
    csv_path = data_dir / 'renders' / 'index.csv'
    part_encoder, color_encoder = build_label_encoders(str(csv_path))

    # Create dataset
    dataset = LegoPartsDataset(
        csv_path=str(csv_path),
        images_dir=str(data_dir / 'renders'),
        part_encoder=part_encoder,
        color_encoder=color_encoder,
        is_train=False
    )

    val_loader = DataLoader(
        dataset,
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
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Evaluate
    part_top1, part_top5, color_top1, cm, top_20_parts = evaluate(
        model=model,
        val_loader=val_loader,
        device=device,
        part_encoder=part_encoder,
        color_encoder=color_encoder
    )

    # Print results
    print(f"\n=== Evaluation Results ===")
    print(f"Part Top-1 Accuracy: {part_top1:.4f}")
    print(f"Part Top-5 Accuracy: {part_top5:.4f}")
    print(f"Color Top-1 Accuracy: {color_top1:.4f}")

    # Save results
    results = {
        'part_top1_accuracy': float(part_top1),
        'part_top5_accuracy': float(part_top5),
        'color_top1_accuracy': float(color_top1),
        'num_parts': len(part_encoder),
        'num_colors': len(color_encoder)
    }

    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Plot confusion matrix
    cm_path = output_dir / 'confusion_matrix_top20_parts.png'
    plot_confusion_matrix(cm, top_20_parts, cm_path)


if __name__ == '__main__':
    main()
