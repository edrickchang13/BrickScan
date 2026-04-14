#!/usr/bin/env python3
"""
download_datasets.py — Download and normalize LEGO training datasets.

Downloads from multiple sources and normalises them into a single flat
directory structure compatible with BrickScan's training pipeline:

  output_dir/
    <part_num>/
      img_0001.jpg
      img_0002.jpg
      ...

Supported sources
-----------------
1. HuggingFace: pvrancx/legobricks        (already on Spark, skipped by default)
2. HuggingFace: joosthazelzet/lego-bricks  — 50 common parts, synthetic renders
3. Kaggle:      brendan45774/test-lego     — B200C real photos, 200 classes
4. Roboflow:    LEGO detection dataset     — bounding-box scenes (extracts crops)

Usage
-----
  # Download everything except the large Kaggle datasets (requires kaggle CLI):
  python3 download_datasets.py --output-dir ~/Desktop/extra_training_data

  # Include Kaggle (needs `pip install kaggle` + ~/.kaggle/kaggle.json):
  python3 download_datasets.py --output-dir ~/Desktop/extra_training_data --kaggle

  # Single source only:
  python3 download_datasets.py --output-dir ~/Desktop/extra_training_data --source hf-joost

After downloading, run merge_datasets.py to combine with your main dataset.
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download_datasets")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def ensure_deps():
    missing = []
    for pkg in ["requests", "tqdm", "PIL"]:
        try:
            __import__(pkg if pkg != "PIL" else "PIL.Image")
        except ImportError:
            missing.append(pkg if pkg != "PIL" else "Pillow")
    if missing:
        log.error(f"Missing packages: {missing}. Run: pip3 install {' '.join(missing)} --break-system-packages")
        sys.exit(1)


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    import requests
    from tqdm import tqdm
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc or dest.name) as bar:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def safe_part_num(s: str) -> str:
    """Normalise a string to a valid part number (alphanumeric + common suffixes)."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "", s).strip("_-") or "unknown"


# ─── Source 1: HuggingFace joosthazelzet/lego-bricks ─────────────────────────

def download_hf_joost(output_dir: Path):
    """
    Download joosthazelzet/lego-bricks from HuggingFace.
    ~6K images, 50 parts, clean synthetic renders from multiple angles.
    """
    log.info("=== HuggingFace: joosthazelzet/lego-bricks ===")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.error("huggingface_hub not installed. Run: pip3 install huggingface-hub --break-system-packages")
        return

    cache = output_dir / "_hf_cache" / "joost"
    cache.mkdir(parents=True, exist_ok=True)

    log.info("Downloading snapshot (may take a few minutes)…")
    try:
        local = snapshot_download(
            repo_id="joosthazelzet/lego-bricks",
            repo_type="dataset",
            local_dir=str(cache),
            ignore_patterns=["*.parquet"],
        )
    except Exception as e:
        log.error(f"HuggingFace download failed: {e}")
        return

    # Walk downloaded images, group by class label
    dest_root = output_dir / "hf_joost"
    count = 0
    for img_path in Path(local).rglob("*.png"):
        # Directory name is the label (e.g. "3001 Brick 2 x 4")
        label = img_path.parent.name
        part_num = safe_part_num(label.split()[0]) if label else "unknown"
        dest_dir = dest_root / part_num
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest_dir / img_path.name)
        count += 1

    for img_path in Path(local).rglob("*.jpg"):
        label = img_path.parent.name
        part_num = safe_part_num(label.split()[0]) if label else "unknown"
        dest_dir = dest_root / part_num
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest_dir / img_path.name)
        count += 1

    log.info(f"joosthazelzet: {count} images → {dest_root}")


# ─── Source 2: Kaggle B200C (brendan45774/test-lego) ─────────────────────────

def download_kaggle_b200c(output_dir: Path):
    """
    Download B200C dataset from Kaggle: brendan45774/test-lego
    ~40K real photos, 200 LEGO classes. Requires kaggle CLI + credentials.
    """
    log.info("=== Kaggle: brendan45774/test-lego (B200C) ===")
    try:
        import kaggle  # noqa: F401
    except ImportError:
        log.error("kaggle not installed. Run: pip3 install kaggle --break-system-packages")
        log.error("Also ensure ~/.kaggle/kaggle.json exists with your API credentials.")
        return

    cache = output_dir / "_kaggle_cache" / "b200c"
    cache.mkdir(parents=True, exist_ok=True)

    log.info("Downloading via kaggle API…")
    try:
        import subprocess
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "brendan45774/test-lego", "-p", str(cache), "--unzip"],
            check=True,
        )
    except Exception as e:
        log.error(f"Kaggle download failed: {e}")
        return

    # Walk the unzipped structure — classes are subdirectories
    dest_root = output_dir / "kaggle_b200c"
    count = 0
    for img_path in cache.rglob("*.jpg"):
        part_num = safe_part_num(img_path.parent.name)
        dest_dir = dest_root / part_num
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest_dir / img_path.name)
        count += 1
    for img_path in cache.rglob("*.png"):
        part_num = safe_part_num(img_path.parent.name)
        dest_dir = dest_root / part_num
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest_dir / img_path.name)
        count += 1

    log.info(f"B200C: {count} images → {dest_root}")


# ─── Source 3: Kaggle LEGO brick images (joosthazelzet, kaggle mirror) ────────

def download_kaggle_joost(output_dir: Path):
    """
    Kaggle mirror of joosthazelzet's dataset: joosthazelzet/lego-brick-images
    Only use this if HuggingFace version fails.
    """
    log.info("=== Kaggle: joosthazelzet/lego-brick-images ===")
    try:
        import subprocess
        cache = output_dir / "_kaggle_cache" / "joost"
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "joosthazelzet/lego-brick-images",
             "-p", str(cache), "--unzip"],
            check=True,
        )
    except Exception as e:
        log.error(f"Kaggle joost download failed: {e}")
        return

    dest_root = output_dir / "kaggle_joost"
    count = 0
    for img_path in (output_dir / "_kaggle_cache" / "joost").rglob("*.png"):
        label = img_path.parent.name
        part_num = safe_part_num(label.split()[0]) if " " in label else safe_part_num(label)
        dest_dir = dest_root / part_num
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest_dir / img_path.name)
        count += 1

    log.info(f"joosthazelzet kaggle: {count} images → {dest_root}")


# ─── Source 4: Roboflow LEGO detection (crops from bounding-box dataset) ──────

def download_roboflow_lego(output_dir: Path, api_key: Optional[str] = None):
    """
    Download the Roboflow LEGO detection dataset.

    If api_key is provided, uses the Roboflow API.
    Otherwise, downloads the public export URL (640px YOLOv8 format).

    Then crops each detected region from the scene images and saves them
    as individual part images for classification training.
    """
    log.info("=== Roboflow: LEGO detection dataset ===")

    # Public YOLOv8 export URL (no API key needed — open dataset)
    RF_DOWNLOAD_URL = (
        "https://universe.roboflow.com/ds/lego-bricks-detection"
        "?key=PdZHp7tWTz"  # public export key from the open dataset page
    )

    # More reliable: direct download from the open dataset
    # Dataset: "LEGO Bricks" by user "LEGO-bnimj" on Roboflow Universe
    # We use the direct link to their free export
    RF_FALLBACK_URLS = [
        "https://app.roboflow.com/ds/lego-bricks-detection/1/YOLOv8",
    ]

    cache = output_dir / "_rf_cache"
    cache.mkdir(parents=True, exist_ok=True)

    downloaded = False

    # Try roboflow pip package first if api_key provided
    if api_key:
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=api_key)
            project = rf.workspace().project("lego-bricks-detection")
            version = project.version(1)
            version.download("yolov8", location=str(cache / "rf_lego"))
            downloaded = True
            log.info("Downloaded via Roboflow API")
        except Exception as e:
            log.warning(f"Roboflow API download failed ({e}), trying direct URL…")

    if not downloaded:
        log.info("Roboflow public dataset requires an API key or manual download.")
        log.info("Visit: https://universe.roboflow.com/search?q=lego+bricks")
        log.info("Download YOLOv8 format and place the zip in:")
        log.info(f"  {cache / 'rf_lego.zip'}")

        rf_zip = cache / "rf_lego.zip"
        if not rf_zip.exists():
            log.warning("Roboflow zip not found — skipping. Download manually and re-run.")
            return

        with zipfile.ZipFile(rf_zip, "r") as z:
            z.extractall(cache / "rf_lego")
        downloaded = True

    if not downloaded:
        return

    # Crop individual bricks from scene images using YOLO labels
    _crop_yolo_dataset(cache / "rf_lego", output_dir / "roboflow_crops")


def _crop_yolo_dataset(yolo_dir: Path, output_dir: Path):
    """
    Given a YOLO dataset directory (images/ + labels/ subdirs),
    crop each bounding box out of the scene image and save as individual images.
    Class 0 = "lego_piece" — saved to output_dir/lego_piece/
    """
    from PIL import Image as PILImage
    import numpy as np

    images_dir = None
    labels_dir = None

    # Find images and labels dirs
    for candidate in [yolo_dir, yolo_dir / "train", yolo_dir / "valid"]:
        if (candidate / "images").exists():
            images_dir = candidate / "images"
            labels_dir = candidate / "labels"
            break

    if not images_dir:
        log.warning(f"Could not find images/ dir under {yolo_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for img_path in images_dir.rglob("*.jpg"):
        label_path = labels_dir / img_path.relative_to(images_dir).with_suffix(".txt")
        if not label_path.exists():
            continue

        try:
            img = PILImage.open(img_path).convert("RGB")
            W, H = img.size
        except Exception:
            continue

        with open(label_path) as f:
            for i, line in enumerate(f):
                parts_line = line.strip().split()
                if len(parts_line) < 5:
                    continue
                cls, cx, cy, w, h = int(parts_line[0]), *map(float, parts_line[1:5])
                x1 = int((cx - w / 2) * W)
                y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W)
                y2 = int((cy + h / 2) * H)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                if x2 - x1 < 16 or y2 - y1 < 16:
                    continue

                crop = img.crop((x1, y1, x2, y2))
                class_dir = output_dir / f"class_{cls:03d}"
                class_dir.mkdir(exist_ok=True)
                crop_name = f"{img_path.stem}_crop{i:03d}.jpg"
                crop.save(class_dir / crop_name, quality=92)
                count += 1

    log.info(f"Roboflow crops: {count} crops → {output_dir}")


# ─── Source 5: COCO val images (for realistic backgrounds) ───────────────────

def download_coco_backgrounds(output_dir: Path, max_images: int = 5000):
    """
    Download a subset of COCO val2017 images to use as real-world backgrounds
    in generate_multipiece_scenes.py (--backgrounds-dir flag).
    These are NOT training labels — just background textures.
    """
    log.info(f"=== COCO val2017 backgrounds (up to {max_images} images) ===")
    import requests

    COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    cache = output_dir / "_coco_cache"
    cache.mkdir(parents=True, exist_ok=True)

    ann_zip = cache / "annotations_trainval2017.zip"
    if not ann_zip.exists():
        log.info("Downloading COCO annotations (~241MB)…")
        download_file(COCO_ANNOTATIONS_URL, ann_zip, "COCO annotations")

    ann_dir = cache / "annotations"
    if not ann_dir.exists():
        log.info("Extracting annotations…")
        with zipfile.ZipFile(ann_zip) as z:
            z.extractall(cache)

    # Load image URLs from val2017
    import json as _json
    ann_file = cache / "annotations" / "instances_val2017.json"
    with open(ann_file) as f:
        coco_data = _json.load(f)

    images = coco_data["images"][:max_images]
    bg_dir = output_dir / "coco_backgrounds"
    bg_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Downloading {len(images)} COCO images…")
    from tqdm import tqdm
    success = 0
    for img_info in tqdm(images, desc="COCO images"):
        url = img_info["coco_url"]
        fname = bg_dir / img_info["file_name"]
        if fname.exists():
            success += 1
            continue
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                fname.write_bytes(r.content)
                success += 1
        except Exception:
            pass

    log.info(f"COCO backgrounds: {success}/{len(images)} images → {bg_dir}")
    log.info(f"Use with: python3 generate_multipiece_scenes.py --backgrounds-dir {bg_dir}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True,
                        help="Root directory to save downloaded datasets")
    parser.add_argument("--source", default="all",
                        choices=["all", "hf-joost", "kaggle-b200c", "kaggle-joost",
                                 "roboflow", "coco-bg"],
                        help="Which source to download (default: all)")
    parser.add_argument("--kaggle", action="store_true",
                        help="Include Kaggle downloads (requires kaggle CLI + credentials)")
    parser.add_argument("--roboflow-key", default=None,
                        help="Roboflow API key (optional — allows direct API download)")
    parser.add_argument("--coco-max", type=int, default=5000,
                        help="Max COCO background images to download (default 5000)")
    args = parser.parse_args()

    ensure_deps()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Output directory: {output_dir}")

    src = args.source

    if src in ("all", "hf-joost"):
        try:
            download_hf_joost(output_dir)
        except Exception as e:
            log.error(f"hf-joost failed: {e}")

    if args.kaggle or src == "kaggle-b200c":
        try:
            download_kaggle_b200c(output_dir)
        except Exception as e:
            log.error(f"kaggle-b200c failed: {e}")

    if args.kaggle or src == "kaggle-joost":
        try:
            download_kaggle_joost(output_dir)
        except Exception as e:
            log.error(f"kaggle-joost failed: {e}")

    if src in ("all", "roboflow"):
        try:
            download_roboflow_lego(output_dir, api_key=args.roboflow_key)
        except Exception as e:
            log.error(f"roboflow failed: {e}")

    if src in ("all", "coco-bg"):
        try:
            download_coco_backgrounds(output_dir, max_images=args.coco_max)
        except Exception as e:
            log.error(f"coco-bg failed: {e}")

    log.info("=" * 60)
    log.info(f"Downloads complete. Next step:")
    log.info(f"  python3 merge_datasets.py \\")
    log.info(f"    --sources {output_dir} ~/Desktop/synthetic_dataset \\")
    log.info(f"    --output ~/Desktop/merged_training_data")


if __name__ == "__main__":
    main()
