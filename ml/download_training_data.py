#!/usr/bin/env python3
"""
Download large-scale LEGO training datasets for DINOv2 fine-tuning.

Sources:
1. Hugging Face pvrancx/legobricks — 1,000 classes × 400 rendered images
2. Rebrickable LDraw renders — catalog images for all known parts
3. Rebrickable CSV catalog — part metadata (names, categories)

Usage:
    python3 download_training_data.py --all
    python3 download_training_data.py --hf-only
    python3 download_training_data.py --rebrickable-only
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path.home() / "brickscan" / "ml" / "training_data"
HF_DIR = BASE_DIR / "huggingface_legobricks"
REBRICKABLE_DIR = BASE_DIR / "rebrickable"
REBRICKABLE_CSV_DIR = BASE_DIR / "rebrickable_csv"


def download_huggingface_dataset():
    """
    Download the pvrancx/legobricks dataset from Hugging Face.
    Contains ~400,000 rendered images across 1,000 LEGO part classes.
    """
    logger.info("=" * 60)
    logger.info("Downloading Hugging Face pvrancx/legobricks dataset")
    logger.info("=" * 60)

    HF_DIR.mkdir(parents=True, exist_ok=True)

    # Check if huggingface_hub is available
    try:
        import huggingface_hub
        logger.info(f"huggingface_hub version: {huggingface_hub.__version__}")
    except ImportError:
        logger.info("Installing huggingface_hub...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "huggingface_hub[cli]", "datasets", "--quiet",
        ])
        import huggingface_hub

    from datasets import load_dataset

    logger.info("Loading dataset (this will download ~20GB)...")
    start = time.time()

    try:
        ds = load_dataset("pvrancx/legobricks", split="train", cache_dir=str(HF_DIR / "cache"))
        logger.info(f"Dataset loaded: {len(ds)} images, took {time.time()-start:.0f}s")

        # Get class distribution
        if "label" in ds.column_names:
            labels = set(ds["label"])
            logger.info(f"Unique classes: {len(labels)}")

        # Save images organized by class
        img_dir = HF_DIR / "images"
        img_dir.mkdir(exist_ok=True)

        saved = 0
        for i, example in enumerate(ds):
            label = str(example.get("label", "unknown"))
            class_dir = img_dir / label
            class_dir.mkdir(exist_ok=True)

            img = example.get("image")
            if img is not None:
                img_path = class_dir / f"{i:06d}.png"
                if not img_path.exists():
                    img.save(str(img_path))
                    saved += 1

            if (i + 1) % 10000 == 0:
                logger.info(f"  Saved {saved}/{i+1} images...")

        logger.info(f"Done! Saved {saved} images to {img_dir}")

    except Exception as e:
        logger.error(f"Failed to download HF dataset: {e}")
        logger.info("Trying alternative: direct download via git clone...")
        try:
            subprocess.run([
                "git", "clone", "--depth", "1",
                "https://huggingface.co/datasets/pvrancx/legobricks",
                str(HF_DIR / "repo"),
            ], check=True, timeout=3600)
            logger.info("Git clone completed")
        except Exception as e2:
            logger.error(f"Git clone also failed: {e2}")
            return False

    return True


def download_rebrickable_csv():
    """Download Rebrickable's CSV catalog files (parts, colors, categories)."""
    logger.info("=" * 60)
    logger.info("Downloading Rebrickable CSV catalog")
    logger.info("=" * 60)

    REBRICKABLE_CSV_DIR.mkdir(parents=True, exist_ok=True)

    import httpx

    files = [
        "parts.csv.gz",
        "part_categories.csv.gz",
        "colors.csv.gz",
        "elements.csv.gz",
        "part_relationships.csv.gz",
    ]

    for fname in files:
        url = f"https://cdn.rebrickable.com/media/downloads/{fname}"
        out_path = REBRICKABLE_CSV_DIR / fname
        if out_path.exists():
            logger.info(f"  {fname} already exists, skipping")
            continue

        logger.info(f"  Downloading {fname}...")
        try:
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
                out_path.write_bytes(resp.content)
                logger.info(f"  Saved {fname} ({len(resp.content)/1024:.0f} KB)")
        except Exception as e:
            logger.warning(f"  Failed to download {fname}: {e}")

    # Decompress
    import gzip
    for gz_file in REBRICKABLE_CSV_DIR.glob("*.csv.gz"):
        csv_file = gz_file.with_suffix("")  # Remove .gz
        if csv_file.exists():
            continue
        logger.info(f"  Decompressing {gz_file.name}...")
        with gzip.open(gz_file, "rb") as f_in:
            csv_file.write_bytes(f_in.read())

    return True


def download_rebrickable_renders():
    """
    Download Rebrickable's LDraw-rendered part images.
    These are the rendered images used on the Rebrickable website.
    """
    logger.info("=" * 60)
    logger.info("Downloading Rebrickable LDraw renders")
    logger.info("=" * 60)

    REBRICKABLE_DIR.mkdir(parents=True, exist_ok=True)

    import httpx

    # First download the parts list to know what to fetch
    parts_csv = REBRICKABLE_CSV_DIR / "parts.csv"
    if not parts_csv.exists():
        download_rebrickable_csv()

    if not parts_csv.exists():
        logger.error("Cannot download renders without parts.csv")
        return False

    import csv

    part_nums = []
    with open(parts_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            part_nums.append(row["part_num"])

    logger.info(f"Found {len(part_nums)} parts in Rebrickable catalog")

    # Download renders (Rebrickable provides LDraw renders at a known URL pattern)
    renders_dir = REBRICKABLE_DIR / "renders"
    renders_dir.mkdir(exist_ok=True)

    already_have = len(list(renders_dir.glob("*.png")))
    logger.info(f"Already have {already_have} renders")

    downloaded = 0
    failed = 0

    def fetch_render(part_num):
        img_path = renders_dir / f"{part_num}.png"
        if img_path.exists():
            return "skip"

        # Rebrickable render URL pattern (color 11 = black, most complete)
        # Try multiple URL patterns since Rebrickable uses different paths
        url = f"https://cdn.rebrickable.com/media/parts/elements/{part_num}.jpg"
        try:
            with httpx.Client(timeout=15.0, follow_redirects=True) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    img_path.write_bytes(resp.content)
                    return "ok"
                else:
                    return "fail"
        except Exception:
            return "fail"

    # Download in parallel with 8 threads
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_render, pn): pn for pn in part_nums}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result == "ok":
                downloaded += 1
            elif result == "fail":
                failed += 1

            if (i + 1) % 500 == 0:
                logger.info(
                    f"  Progress: {i+1}/{len(part_nums)} "
                    f"(downloaded: {downloaded}, failed: {failed})"
                )

    logger.info(
        f"Render download complete: {downloaded} new, "
        f"{already_have} existing, {failed} failed"
    )
    return True


def print_summary():
    """Print summary of all downloaded data."""
    logger.info("=" * 60)
    logger.info("TRAINING DATA SUMMARY")
    logger.info("=" * 60)

    for name, path in [
        ("Hugging Face images", HF_DIR / "images"),
        ("Rebrickable renders", REBRICKABLE_DIR / "renders"),
        ("Rebrickable CSV", REBRICKABLE_CSV_DIR),
        ("Existing train data", Path.home() / "brickscan" / "ml" / "data" / "images" / "train"),
        ("Existing val data", Path.home() / "brickscan" / "ml" / "data" / "images" / "val"),
    ]:
        if path.exists():
            if path.is_dir():
                files = list(path.rglob("*"))
                file_count = sum(1 for f in files if f.is_file())
                dir_count = sum(1 for f in files if f.is_dir())
                size = sum(f.stat().st_size for f in files if f.is_file())
                logger.info(
                    f"  {name}: {file_count} files, "
                    f"{dir_count} dirs, {size/1e9:.2f} GB"
                )
            else:
                logger.info(f"  {name}: {path.stat().st_size/1e6:.1f} MB")
        else:
            logger.info(f"  {name}: NOT FOUND")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LEGO training data")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--hf-only", action="store_true", help="Only Hugging Face dataset")
    parser.add_argument("--rebrickable-only", action="store_true", help="Only Rebrickable data")
    parser.add_argument("--csv-only", action="store_true", help="Only Rebrickable CSV catalog")
    parser.add_argument("--summary", action="store_true", help="Print data summary only")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        sys.exit(0)

    if args.all or args.csv_only or args.rebrickable_only:
        download_rebrickable_csv()

    if args.all or args.rebrickable_only:
        download_rebrickable_renders()

    if args.all or args.hf_only:
        download_huggingface_dataset()

    print_summary()
    logger.info("Done!")
