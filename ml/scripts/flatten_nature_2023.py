#!/usr/bin/env python3
"""
Flatten Nature 2023 zip extracts into the directory layouts our training
scripts expect. Zero copies — only symlinks — so it's fast and uses no
extra disk space.

Input (what the zips extract to):
  ml/data/nature_2023/
    detection/
      photos/<class_N>/*.jpg   — real photos (nested by internal class)
      renders/.../*.jpg         — synthetic renders
      labels/*.txt              — YOLO format (already produced by downloader)
    classification_real/
      <Category>/<part_num>/*.jpg  — 2-level: category folder containing part dirs

Output (canonical layout):
  ml/data/nature_2023/
    detection/
      images/            ← NEW: symlink tree flattening photos/* + renders/*
        *.jpg
      labels/            ← unchanged: YOLO txt files
    classification_real_flat/  ← NEW: 1-level part-folder layout
      <part_num>/*.jpg

After running, our training scripts can consume them directly:
  ml/data/nature_lego.yaml → path: nature_2023/detection, train: images
  train_contrastive.py / train_mobilenetv3 --data-dir ml/data/nature_2023/classification_real_flat

Idempotent: skips symlinks that already exist.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("flatten")


def flatten_detection_images(detection_dir: Path) -> int:
    """
    Walk detection/photos and detection/renders, symlink every image
    into detection/images/<filename>. Returns count created.
    """
    images_dir = detection_dir / "images"
    images_dir.mkdir(exist_ok=True)

    sources = []
    for sub in ("photos", "renders"):
        src = detection_dir / sub
        if src.is_dir():
            sources.append(src)
    if not sources:
        log.warning("No photos/ or renders/ subdir under %s", detection_dir)
        return 0

    created = 0
    collisions = 0
    for src_root in sources:
        for jpg in src_root.rglob("*.jpg"):
            link = images_dir / jpg.name
            if link.exists() or link.is_symlink():
                # Name collision — use a prefix to disambiguate
                if link.resolve() != jpg.resolve():
                    collisions += 1
                    prefixed = images_dir / f"{src_root.name}__{jpg.name}"
                    if not (prefixed.exists() or prefixed.is_symlink()):
                        prefixed.symlink_to(jpg.resolve())
                        created += 1
                continue
            link.symlink_to(jpg.resolve())
            created += 1

    log.info("detection/images: %d symlinks (+%d disambiguated collisions)", created, collisions)
    return created


def flatten_classification(class_dir: Path) -> int:
    """
    Collapse `<Category>/<part_num>/*.jpg` → `classification_real_flat/<part_num>/*.jpg`.
    Returns count of part folders produced.
    """
    flat = class_dir.parent / "classification_real_flat"
    flat.mkdir(exist_ok=True)

    part_counter: Counter = Counter()
    for category in sorted(class_dir.iterdir()):
        if not category.is_dir():
            continue
        for part_dir in sorted(category.iterdir()):
            if not part_dir.is_dir():
                continue
            pn = part_dir.name
            target = flat / pn
            target.mkdir(exist_ok=True)
            for jpg in part_dir.glob("*.jpg"):
                link = target / jpg.name
                if link.exists() or link.is_symlink():
                    continue
                link.symlink_to(jpg.resolve())
                part_counter[pn] += 1

    total_parts = len(part_counter)
    total_imgs = sum(part_counter.values())
    log.info("classification_real_flat: %d part folders, %d images",
             total_parts, total_imgs)
    return total_parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path,
                        default=Path("ml/data/nature_2023"),
                        help="Nature 2023 extract root (default ml/data/nature_2023)")
    args = parser.parse_args()

    if not args.root.exists():
        log.error("Not found: %s", args.root)
        return 2

    log.info("Flattening %s …", args.root)

    det = args.root / "detection"
    cls = args.root / "classification_real"

    if det.exists():
        flatten_detection_images(det)
    else:
        log.warning("No detection/ subdir")

    if cls.exists():
        flatten_classification(cls)
    else:
        log.warning("No classification_real/ subdir")

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
