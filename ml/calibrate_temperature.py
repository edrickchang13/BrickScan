#!/usr/bin/env python3
"""
Temperature scaling calibration for model confidence scores.

After training, this script finds the optimal temperature T such that
predicted confidence scores match actual accuracy (calibration).

For example: if the model says 80% confident, it should actually be correct
80% of the time.

Expected Calibration Error (ECE) metric is used to evaluate calibration.

Usage:
  python3 calibrate_temperature.py --checkpoint ./models/best.pt \\
    --val-data ./data/val --output-dir ./models
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from scipy.optimize import minimize_scalar
except ImportError:
    minimize_scalar = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("calibrate_temperature")


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the gap between predicted confidence and actual accuracy.
    Lower is better (0.0 = perfectly calibrated).

    Args:
        probs: Predicted probabilities [N, C] or confidences [N]
        labels: Ground truth labels [N]
        n_bins: Number of bins to use for ECE calculation

    Returns:
        ECE value (0.0 to 1.0)
    """
    # If 2D, take max probability (argmax confidence)
    if len(probs.shape) == 2:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
    else:
        confidences = probs
        predictions = (probs > 0.5).astype(int)

    correct = (predictions == labels).astype(float)

    # Bin predictions by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue

        bin_correct = correct[mask].mean()
        bin_confidence = confidences[mask].mean()
        bin_weight = mask.sum() / len(labels)

        ece += bin_weight * abs(bin_correct - bin_confidence)

    return ece


def apply_temperature_scaling(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Apply temperature scaling to logits.

    Args:
        logits: Raw model outputs [B, C]
        temperature: Temperature parameter (0.5 to 5.0)

    Returns:
        Scaled probabilities
    """
    return torch.softmax(logits / temperature, dim=-1)


def find_optimal_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cuda",
    max_temp: float = 5.0,
) -> float:
    """
    Find optimal temperature T using binary search on validation set.

    Minimizes Negative Log-Likelihood (NLL) over temperature range [0.5, max_temp].

    Args:
        model: Trained classifier
        val_loader: Validation data loader
        device: Device to run on
        max_temp: Maximum temperature to search

    Returns:
        Optimal temperature value
    """
    model = model.to(device)
    model.eval()

    # Collect all logits and labels
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting logits"):
            images = batch[0].to(device)
            labels = batch[1].to(device)

            # Get logits (before softmax)
            if hasattr(model, 'classifier'):
                # Multi-head model
                logits = model.classifier(model.backbone(images))
            else:
                logits = model(images)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    # Define NLL loss as function of temperature
    def nll_loss(T: float) -> float:
        """Compute NLL for a given temperature."""
        scaled_logits = logits_tensor / T
        nll = F.cross_entropy(scaled_logits, labels_tensor)
        return nll.item()

    # Binary search for optimal temperature
    log.info(f"Searching for optimal temperature in range [0.5, {max_temp}]...")

    if minimize_scalar:
        # Use scipy's minimize_scalar for efficiency
        result = minimize_scalar(
            nll_loss,
            bounds=(0.5, max_temp),
            method='bounded',
        )
        optimal_T = result.x
        optimal_nll = result.fun
    else:
        # Fallback: grid search
        log.warning("scipy not available, using grid search")
        temps = np.linspace(0.5, max_temp, 50)
        nlls = [nll_loss(T) for T in temps]
        optimal_idx = np.argmin(nlls)
        optimal_T = temps[optimal_idx]
        optimal_nll = nlls[optimal_idx]

    log.info(f"Optimal temperature: {optimal_T:.4f} (NLL: {optimal_nll:.4f})")

    # Compute ECE before and after
    with torch.no_grad():
        uncalibrated_probs = F.softmax(logits_tensor, dim=-1).numpy()
        calibrated_probs = apply_temperature_scaling(logits_tensor, optimal_T).numpy()

    uncalibrated_ece = compute_ece(uncalibrated_probs, labels_tensor.numpy())
    calibrated_ece = compute_ece(calibrated_probs, labels_tensor.numpy())

    log.info(f"ECE before calibration: {uncalibrated_ece:.4f}")
    log.info(f"ECE after calibration: {calibrated_ece:.4f}")

    return optimal_T


def save_calibration(output_dir: Path, temperature: float):
    """
    Save calibration data to JSON alongside model checkpoint.

    Args:
        output_dir: Directory to save calibration.json
        temperature: Optimal temperature value
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cal_file = output_dir / "calibration.json"

    calibration_data = {
        "temperature": float(temperature),
        "method": "temperature_scaling",
        "description": "Divide logits by this temperature before softmax for calibrated confidence scores",
    }

    with open(cal_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)

    log.info(f"Saved calibration to {cal_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate model confidence scores via temperature scaling"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--val-data",
        required=True,
        help="Directory with validation dataset"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save calibration.json"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--max-temp",
        type=float,
        default=5.0,
        help="Maximum temperature to search (default: 5.0)"
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        log.error(f"Checkpoint not found: {checkpoint_path}")
        return

    val_data_dir = Path(args.val_data)
    if not val_data_dir.exists():
        log.error(f"Validation data not found: {val_data_dir}")
        return

    # Load model (assumes standard PyTorch checkpoint format)
    log.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)

    # Create model and load state
    # NOTE: User must provide model architecture here or modify this section
    log.error("ERROR: Model architecture not specified. Modify main() to load your model.")
    log.error("Example:")
    log.error("  from my_model import MyClassifier")
    log.error("  model = MyClassifier(num_classes=1000)")
    log.error("  model.load_state_dict(checkpoint['model_state_dict'])")
    return

    # Create validation loader
    # NOTE: Assumes standard image dataset structure
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    val_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_dataset = ImageFolder(val_data_dir, transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Find optimal temperature
    optimal_T = find_optimal_temperature(
        model,
        val_loader,
        device=args.device,
        max_temp=args.max_temp,
    )

    # Save calibration
    save_calibration(Path(args.output_dir), optimal_T)


if __name__ == "__main__":
    main()
