#!/usr/bin/env python3
"""
Download additional LEGO part images to improve training data balance.

The current model has ~3.6 images/part on average and is overfitting (97% train, 69% val).
This script fetches additional images from Rebrickable and BrickLink for parts that have
fewer than the target number of images.

Data sources:
  - Rebrickable API: part images and alt images
  - BrickLink CDN: https://img.bricklink.com/ItemImage/PN/{bl_color_id}/{part_num}.png

Usage:
    python download_extra_images.py --data-dir . --api-key YOUR_KEY
    python download_extra_images.py --data-dir . --api-key YOUR_KEY --min-images 15 --max-per-part 20

The script:
  1. Reads the existing index.csv to find which parts already have images
  2. For parts with fewer than --min-images images, fetches additional images
  3. Saves downloaded images into the images directory
  4. Updates index.csv with new entries
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = None  # Set by argument
IMAGES_DIR = None
INDEX_CSV = None
CACHE_DIR = None

# ── BrickLink color ID map (Rebrickable ID → BrickLink ID) ─────────────────────
BL_COLOR_MAP: Dict[int, int] = {
    0: 0, 1: 1, 2: 5, 3: 6, 4: 2, 5: 4, 6: 15, 7: 8,
    9: 11, 10: 7, 11: 9, 12: 3, 13: 62, 14: 16, 15: 10,
    17: 12, 18: 13, 19: 19, 25: 120, 26: 47, 28: 80,
    29: 30, 36: 59, 38: 57, 40: 28, 41: 32, 42: 34,
    43: 35, 44: 36, 46: 27, 47: 52, 48: 44, 49: 43,
    50: 50, 51: 51, 52: 53, 53: 45, 54: 42, 55: 41,
    57: 179, 272: 63, 288: 288, 320: 47,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_extra")

HEADERS = {
    "User-Agent": "BrickScan-ML/1.0 (training data downloader)",
    "Accept": "image/webp,image/png,image/jpeg,*/*",
}


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _fetch(url: str, timeout: int = 20) -> Optional[bytes]:
    """Fetch URL and return raw bytes."""
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def _fetch_image(url: str) -> Optional[Image.Image]:
    """Fetch URL and return PIL Image, or None if invalid."""
    data = _fetch(url)
    if data and len(data) > 1000:
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            if img.width >= 32 and img.height >= 32:
                return img
        except Exception:
            pass
    return None


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyze_current_images(index_csv: Path) -> Dict[str, int]:
    """
    Count images per part from existing index.csv.

    Returns:
        Dict mapping part_num -> count of images
    """
    if not index_csv.exists():
        return {}

    df = pd.read_csv(index_csv)
    counts = df["part_num"].value_counts().to_dict()
    return counts


def find_parts_needing_images(
    image_counts: Dict[str, int],
    min_images: int,
) -> List[str]:
    """Find parts with fewer than min_images."""
    return [part for part, count in image_counts.items() if count < min_images]


# ── Download ───────────────────────────────────────────────────────────────────

def _try_download_image(part_num: str, color_id: int, api_key: Optional[str]) -> Optional[Image.Image]:
    """Try to download image from BrickLink or Rebrickable."""
    # 1. BrickLink (best per-color coverage)
    bl_color = BL_COLOR_MAP.get(color_id, color_id)
    img = _fetch_image(f"https://img.bricklink.com/ItemImage/PN/{bl_color}/{part_num}.png")
    if img:
        return img

    # 2. Rebrickable CDN default photo
    img = _fetch_image(f"https://cdn.rebrickable.com/media/parts/photos/{part_num}.jpg")
    if img:
        return img

    # 3. Rebrickable API (only if key provided) - try main image
    if api_key:
        try:
            data = _fetch(f"https://rebrickable.com/api/v3/lego/parts/{part_num}/?key={api_key}")
            if data:
                part_data = json.loads(data)
                img_url = part_data.get("part_img_url", "")
                if img_url:
                    img = _fetch_image(img_url)
                    if img:
                        return img
        except Exception:
            pass

    return None


def _fetch_part_colors(part_num: str, api_key: Optional[str]) -> List[int]:
    """Fetch available colors for a part from Rebrickable API."""
    if not api_key:
        return []

    colors = []
    try:
        url = f"https://rebrickable.com/api/v3/lego/parts/{part_num}/colors/"
        headers = {"Authorization": f"key {api_key}"}
        req = Request(url, headers=headers)
        with urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
            results = data.get("results", [])
            colors = [c.get("color_id") for c in results if c.get("color_id")]
    except Exception:
        pass

    return colors


def _process_row(
    part_num: str,
    color_id: int,
    color_name: str,
    color_r: float,
    color_g: float,
    color_b: float,
    api_key: Optional[str],
    image_size: int,
) -> Optional[dict]:
    """Download and save a single image."""

    out_dir = IMAGES_DIR / part_num
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use unique counter if file already exists
    counter = 0
    out_path = out_dir / f"{color_id}_{counter:04d}.jpg"
    while out_path.exists():
        counter += 1
        out_path = out_dir / f"{color_id}_{counter:04d}.jpg"

    # Skip if we already have this image
    if out_path.exists() and out_path.stat().st_size > 1000:
        return {
            "image_path": f"{part_num}/{out_path.name}",
            "part_num": part_num,
            "color_id": color_id,
            "color_name": color_name,
            "color_r": color_r,
            "color_g": color_g,
            "color_b": color_b,
        }

    img = _try_download_image(part_num, color_id, api_key)
    if img is None:
        return None

    # Resize to square canvas
    img.thumbnail((image_size, image_size), Image.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    canvas.paste(img, ((image_size - img.width) // 2, (image_size - img.height) // 2))
    canvas.save(out_path, "JPEG", quality=90)

    return {
        "image_path": f"{part_num}/{out_path.name}",
        "part_num": part_num,
        "color_id": color_id,
        "color_name": color_name,
        "color_r": color_r,
        "color_g": color_g,
        "color_b": color_b,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Main download routine."""
    global DATA_DIR, IMAGES_DIR, INDEX_CSV, CACHE_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Data directory (where index.csv lives)")
    parser.add_argument("--api-key", required=True, help="Rebrickable API key")
    parser.add_argument("--min-images", type=int, default=10, help="Min images per part (default: 10)")
    parser.add_argument("--max-per-part", type=int, default=15, help="Max images per part (default: 15)")
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers (default: 16)")
    parser.add_argument("--image-size", type=int, default=256, help="Image size (default: 256)")
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    IMAGES_DIR = DATA_DIR / "images"
    INDEX_CSV = DATA_DIR / "index.csv"
    CACHE_DIR = DATA_DIR / ".cache"

    if not INDEX_CSV.exists():
        log.error("index.csv not found at %s", INDEX_CSV)
        sys.exit(1)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Analyze current images
    log.info("Analyzing current images...")
    image_counts = analyze_current_images(INDEX_CSV)
    log.info("  Found %d unique parts with %d total images", len(image_counts), sum(image_counts.values()))

    # Find parts that need more images
    parts_needing = find_parts_needing_images(image_counts, args.min_images)
    log.info("  %d parts have < %d images", len(parts_needing), args.min_images)

    if not parts_needing:
        log.info("All parts have sufficient images. Nothing to do.")
        return

    # Build color candidates for each part
    log.info("Fetching available colors for %d parts…", len(parts_needing))
    part_colors: Dict[str, List[int]] = {}

    for part in parts_needing:
        colors = _fetch_part_colors(part, args.api_key)
        part_colors[part] = colors
        time.sleep(0.1)  # Rate limiting

    # Read existing index to get color metadata
    df_existing = pd.read_csv(INDEX_CSV)
    color_meta = {}
    for _, row in df_existing.iterrows():
        cid = int(row["color_id"])
        if cid not in color_meta:
            color_meta[cid] = {
                "name": row["color_name"],
                "r": float(row["color_r"]),
                "g": float(row["color_g"]),
                "b": float(row["color_b"]),
            }

    # Build list of (part, color) combos to download
    rows_to_download = []
    current_counts = image_counts.copy()

    for part in parts_needing:
        current_count = current_counts.get(part, 0)
        colors = part_colors.get(part, [])

        for color_id in colors:
            if current_count >= args.max_per_part:
                break

            # Only add if we don't have this combo yet
            existing = df_existing[
                (df_existing["part_num"] == part) & (df_existing["color_id"] == color_id)
            ]
            if len(existing) == 0:
                meta = color_meta.get(color_id, {"name": str(color_id), "r": 0.5, "g": 0.5, "b": 0.5})
                rows_to_download.append({
                    "part_num": part,
                    "color_id": color_id,
                    "color_name": meta["name"],
                    "color_r": meta["r"],
                    "color_g": meta["g"],
                    "color_b": meta["b"],
                })
                current_count += 1

    if not rows_to_download:
        log.info("No new combos to download.")
        return

    log.info("Downloading %d images with %d workers…", len(rows_to_download), args.workers)
    results: List[dict] = []
    failed = 0
    t0 = time.time()

    def process_row(row):
        return _process_row(
            row["part_num"], int(row["color_id"]),
            row["color_name"],
            float(row["color_r"]), float(row["color_g"]), float(row["color_b"]),
            args.api_key, args.image_size
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_row, r): r for r in rows_to_download}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                results.append(result)
            else:
                failed += 1

            if i % 100 == 0 or i == len(rows_to_download):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1)
                eta = (len(rows_to_download) - i) / max(rate, 0.001)
                log.info(
                    "  %d/%d | ok=%d fail=%d | %.0f/s | ETA %.1fm",
                    i, len(rows_to_download), len(results), failed, rate, eta / 60
                )

    # Append to index.csv
    if results:
        new_df = pd.DataFrame(results, columns=[
            "image_path", "part_num", "color_id", "color_name", "color_r", "color_g", "color_b"
        ])
        df_combined = pd.concat([df_existing, new_df], ignore_index=True)
        df_combined.to_csv(INDEX_CSV, index=False)
        log.info("Updated index.csv")

    log.info("\n" + "=" * 55)
    log.info("  New images downloaded: %d (%d failed)", len(results), failed)
    log.info("  Index CSV: %s", INDEX_CSV)
    log.info("=" * 55)


if __name__ == "__main__":
    main()
