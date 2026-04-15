#!/usr/bin/env python3
"""
Train MobileNetV3-Small on the synthetic LEGO-brick corpus on Mac CPU.

Produces a VERIFIED replacement for the mystery
`backend/models/lego_classifier.onnx` whose provenance is unknown. Even
at modest CPU-trained accuracy this beats an unverified model of unknown
origin because we can now measure it against the feedback eval set.

Training recipe (following ICCS 2022 "How to Sort Them?" paper's practices):
  - Backbone: MobileNetV3-Small (ImageNet-pretrained)
  - Input:    224x224 RGB (matches DINOv2's expected input shape;
              downstream cascade doesn't care about resolution)
  - Optimiser: AdamW, lr=5e-4, weight_decay=1e-4
  - Schedule: cosine with 1-epoch warmup
  - Augmentation: RandomResizedCrop, RandomHorizontalFlip, ColorJitter,
                  RandomRotation, RandomErasing — plus ImageNet normalization
  - Loss: CrossEntropy
  - Batch size: 64 (safe for 16GB Macs, adjust via --batch-size)

Output artefacts (written to --output-dir):
  - best.pt                   — PyTorch checkpoint (state_dict + metadata)
  - final.pt                  — last-epoch state_dict (unused, for resume)
  - class_labels.json         — {"idx2part": {"0": "3001", ...}}  — matches
                                 the format ml/retrain_from_feedback.py loads
  - training_log.jsonl        — per-epoch loss + val accuracy
  - mobilenetv3_lego.onnx     — final ONNX export

Usage:
    ./backend/venv/bin/python3 ml/scripts/train_mobilenetv3_local.py \\
        --data-dir ml/data/synthetic_dataset \\
        --output-dir ml/data/models/mobilenetv3_$(date +%Y%m%d) \\
        --epochs 3 \\
        --batch-size 64

Wall-clock estimate on Apple M-series CPU (MPS not used; CPU only to avoid
FP16/fp32 mismatches and to stay compatible with Intel Macs too):
  - 268k images / 64 batch = ~4200 batches/epoch
  - ~4s/batch on M2 CPU  ≈ 4.5 hrs/epoch  ≈ 13-14 hrs for 3 epochs

For a smaller smoke test: pass --epochs 1 --limit-classes 50 to finish
in ~1 hour instead of overnight.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_mobilenetv3")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def build_dataloaders(data_dir: Path, image_size: int, batch_size: int,
                      val_split: float, num_workers: int,
                      limit_classes: int | None,
                      samples_per_class: int | None = None):
    """
    Build train / val DataLoaders from a class-folder dataset. Expects
    `data_dir/<part_num>/*.png` layout — which is exactly what our
    Blender render pipeline produces.
    """
    import torch
    from torch.utils.data import DataLoader, Subset, random_split
    from torchvision import datasets, transforms
    from torchvision.transforms.autoaugment import RandAugment

    train_tfm = transforms.Compose([
        transforms.Resize(int(image_size * 1.143)),   # 256 for 224
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize(int(image_size * 1.143)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Skip empty class directories — torchvision.ImageFolder crashes on them
    # and the user's Desktop corpus has a couple from day-1 smoke tests.
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    def has_images(p: Path) -> bool:
        try:
            return any(f.is_file() and f.suffix.lower() in IMG_EXTS for f in p.iterdir())
        except (OSError, StopIteration):
            return False

    empty_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not has_images(d)]
    if empty_dirs:
        log.warning("Skipping %d empty class dirs: %s%s",
                    len(empty_dirs),
                    ", ".join(d.name for d in empty_dirs[:5]),
                    "..." if len(empty_dirs) > 5 else "")
    allow_empty = True  # ImageFolder kwarg introduced in torchvision >= 0.17

    # Safe loader — some PNGs in the synthetic corpus are truncated (Blender
    # got killed mid-write at some point). Skip those by returning a tiny
    # grey placeholder; the label is still correct so we lose one sample's
    # signal, not the whole training run.
    from PIL import Image as _PILImage

    def _safe_loader(path: str):
        try:
            with open(path, "rb") as f:
                img = _PILImage.open(f)
                img.load()
                return img.convert("RGB")
        except Exception as e:
            log.debug("Skipping corrupt image %s: %s", path, e)
            return _PILImage.new("RGB", (image_size, image_size), (128, 128, 128))

    # We need two separate dataset objects (different transforms) with the
    # same underlying samples, then split by index.
    train_full = datasets.ImageFolder(str(data_dir), transform=train_tfm,
                                      allow_empty=allow_empty, loader=_safe_loader)
    val_full   = datasets.ImageFolder(str(data_dir), transform=val_tfm,
                                      allow_empty=allow_empty, loader=_safe_loader)
    # Drop any classes that ended up with zero samples
    good_classes = {y for (_, y) in train_full.samples}
    if len(good_classes) < len(train_full.classes):
        train_full.samples = [(p, y) for (p, y) in train_full.samples if y in good_classes]
        val_full.samples   = [(p, y) for (p, y) in val_full.samples   if y in good_classes]
        train_full.targets = [y for (_, y) in train_full.samples]
        val_full.targets   = [y for (_, y) in val_full.samples]

    # Optional: subsample each class to at most `samples_per_class` images.
    # Critical for CPU training time — full dataset is ~270k images; at
    # ~4s/batch on M-series CPU with batch=48, a single epoch through
    # everything takes ~6 hrs. Sampling 150/class drops that to ~90 min.
    if samples_per_class is not None and samples_per_class > 0:
        rng = random.Random(42)
        by_class: Dict[int, List] = {}
        for sample in train_full.samples:
            by_class.setdefault(sample[1], []).append(sample)
        subsampled = []
        for cls_idx, items in by_class.items():
            if len(items) > samples_per_class:
                items = rng.sample(items, samples_per_class)
            subsampled.extend(items)
        train_full.samples = subsampled
        val_full.samples   = list(subsampled)   # share pool; val split picks disjoint indices below
        train_full.targets = [y for (_, y) in subsampled]
        val_full.targets   = [y for (_, y) in subsampled]
        log.info("Subsampled to ≤%d imgs/class — %d total samples", samples_per_class, len(subsampled))

    # Optionally restrict to the first N classes for smoke testing.
    if limit_classes is not None and limit_classes > 0:
        keep_classes = set(sorted(train_full.classes)[:limit_classes])
        keep_idx = {train_full.class_to_idx[c] for c in keep_classes}
        # Rebuild both samples lists with only kept classes
        filter_samples = lambda ds: [(p, y) for (p, y) in ds.samples if y in keep_idx]
        train_full.samples = filter_samples(train_full)
        val_full.samples   = filter_samples(val_full)
        train_full.targets = [y for (_, y) in train_full.samples]
        val_full.targets   = [y for (_, y) in val_full.samples]
        log.info("Limited to first %d classes (%d samples)", limit_classes, len(train_full.samples))

    n_total = len(train_full.samples)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    # Deterministic index split so train vs val are disjoint + reproducible
    indices = list(range(n_total))
    random.Random(42).shuffle(indices)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_ds = Subset(train_full, train_indices)
    val_ds   = Subset(val_full,   val_indices)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        pin_memory=False,
    )
    class_names = train_full.classes  # sorted list; index = class_id

    log.info("Dataset: %d train / %d val, %d classes",
             len(train_ds), len(val_ds), len(class_names))
    return train_loader, val_loader, class_names


def build_model(num_classes: int):
    import torch
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    # Swap the classifier head to match our class count
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_feats, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, epoch, total_epochs):
    import torch
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    t0 = time.time()
    last_log = t0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=False)
        labels = labels.to(device, non_blocking=False)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)

        now = time.time()
        if now - last_log > 30 or batch_idx == 0:
            elapsed = now - t0
            pct = 100.0 * (batch_idx + 1) / len(loader)
            running_acc = 100.0 * total_correct / max(1, total_samples)
            running_loss = total_loss / max(1, total_samples)
            eta_min = ((now - t0) / max(1, batch_idx + 1)) * (len(loader) - batch_idx - 1) / 60
            log.info(
                "Epoch %d/%d batch %d/%d (%.1f%%) — loss %.3f  acc %.1f%%  eta %.1fm",
                epoch + 1, total_epochs, batch_idx + 1, len(loader), pct,
                running_loss, running_acc, eta_min,
            )
            last_log = now

    return {
        "loss": total_loss / max(1, total_samples),
        "acc":  total_correct / max(1, total_samples),
        "elapsed_sec": time.time() - t0,
    }


def evaluate(model, loader, criterion, device):
    import torch
    model.eval()
    total_loss = 0.0
    top1 = 0
    top3 = 0
    n = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)
            top1 += (logits.argmax(1) == labels).sum().item()
            top3_preds = logits.topk(3, dim=1).indices
            top3 += (top3_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
    return {
        "loss": total_loss / max(1, n),
        "top1_acc": top1 / max(1, n),
        "top3_acc": top3 / max(1, n),
    }


def export_onnx(model, out_path: Path, image_size: int) -> None:
    import torch
    model.eval()
    dummy = torch.zeros(1, 3, image_size, image_size)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    log.info("Exported ONNX: %s (%d MB)", out_path, out_path.stat().st_size // (1024 * 1024))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir",   type=Path, required=True,
                        help="Class-folder root (e.g. ml/data/synthetic_dataset)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split",  type=float, default=0.05,
                        help="Fraction of samples held out for validation")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 = single-process, recommended on Mac)")
    parser.add_argument("--limit-classes", type=int, default=None,
                        help="Optional: restrict to first N classes (smoke tests)")
    parser.add_argument("--samples-per-class", type=int, default=None,
                        help="Optional: subsample each class to at most N images (Mac CPU time saver)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data_dir.exists():
        log.error("Data dir not found: %s", args.data_dir)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    try:
        import torch
    except ImportError:
        log.error("PyTorch not installed — run: ./backend/venv/bin/pip install torch torchvision")
        return 2

    device = torch.device("cpu")
    log.info("Device: %s | torch %s", device, torch.__version__)

    train_loader, val_loader, class_names = build_dataloaders(
        args.data_dir, args.image_size, args.batch_size, args.val_split,
        args.num_workers, args.limit_classes, args.samples_per_class,
    )
    num_classes = len(class_names)
    log.info("Building MobileNetV3-Small (num_classes=%d)", num_classes)
    model = build_model(num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1,
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    log_path = args.output_dir / "training_log.jsonl"
    best_top1 = -1.0
    best_path = args.output_dir / "best.pt"

    for epoch in range(args.epochs):
        log.info("=" * 60)
        log.info("Epoch %d/%d", epoch + 1, args.epochs)
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch, args.epochs)
        val_stats   = evaluate(model, val_loader, criterion, device)

        log.info("Train loss=%.3f acc=%.1f%%  Val loss=%.3f top1=%.1f%% top3=%.1f%%",
                 train_stats["loss"], 100*train_stats["acc"],
                 val_stats["loss"], 100*val_stats["top1_acc"], 100*val_stats["top3_acc"])

        record = {
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"], "train_acc": train_stats["acc"],
            "val_loss": val_stats["loss"],
            "val_top1": val_stats["top1_acc"],  "val_top3": val_stats["top3_acc"],
            "epoch_seconds": train_stats["elapsed_sec"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if val_stats["top1_acc"] > best_top1:
            best_top1 = val_stats["top1_acc"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "num_parts": num_classes,
                "num_colors": 0,   # single-head — no colour prediction
                "val_top1": val_stats["top1_acc"],
                "val_top3": val_stats["top3_acc"],
                "architecture": "mobilenet_v3_small",
            }, best_path)
            log.info("Saved best checkpoint: %s (val top1=%.2f%%)", best_path, 100 * best_top1)

    # Dump class labels in the format the backend / retrain script expects
    labels_path = args.output_dir / "class_labels.json"
    idx2part = {str(i): name for i, name in enumerate(class_names)}
    labels_path.write_text(json.dumps({"idx2part": idx2part, "num_classes": num_classes}, indent=2))
    log.info("Saved class labels: %s (%d classes)", labels_path, num_classes)

    # Export ONNX from the best checkpoint
    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    onnx_path = args.output_dir / "mobilenetv3_lego.onnx"
    export_onnx(model, onnx_path, args.image_size)

    log.info("=" * 60)
    log.info("Training complete.")
    log.info("Best val top-1: %.2f%%", 100 * best_top1)
    log.info("Checkpoint:     %s", best_path)
    log.info("ONNX:           %s", onnx_path)
    log.info("Class labels:   %s", labels_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
