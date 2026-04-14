#!/usr/bin/env python3
"""
Download public LEGO part image datasets for training and validation.

Downloads (in priority order):
  1. Johansson et al. 2023 (Scientific Data) — 155K real photos + BrickLink IDs
     "A large-scale LEGO dataset of synthetic and real images for part detection"
     https://www.nature.com/articles/s41597-023-02682-2
     Figshare DOI: 10.6084/m9.figshare.22591796

  2. B200 LEGO Dataset (2024) — 2000 high-quality renders, 200 parts
     https://github.com/LEGO-challenge/LEGO-dataset

  3. Rebrickable catalog images (via API) — ~600K part photos, ground-truth labels
     Requires REBRICKABLE_API_KEY env var.

Usage:
    # Download all datasets:
    python download_lego_datasets.py --output-dir ./data --all

    # Just Rebrickable catalog (requires API key):
    export REBRICKABLE_API_KEY="your_key"
    python download_lego_datasets.py --output-dir ./data --rebrickable --top-n 500

    # Build the validation split from downloaded data:
    python download_lego_datasets.py --output-dir ./data --build-splits
"""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import sys
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False

try:
    import requests
    REQUESTS = True
except ImportError:
    REQUESTS = False


# ============================================================================
# Helpers
# ============================================================================

def make_progress_bar(total: int, desc: str):
    if TQDM:
        return tqdm(total=total, desc=desc, unit='file')
    class FakePbar:
        def update(self, n=1): pass
        def close(self): print(f"{desc}: done")
        def set_postfix(self, **kw): pass
    return FakePbar()


def download_file(url: str, dest: Path, desc: str = "", chunk_size: int = 65536) -> bool:
    """Download a URL to dest with progress. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'BrickScan/1.0'})
        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            with open(dest, 'wb') as f:
                if TQDM:
                    bar = tqdm(total=total, unit='B', unit_scale=True,
                               desc=desc or dest.name, leave=False)
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if TQDM:
                        bar.update(len(chunk))
                if TQDM:
                    bar.close()
        return True
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def rebrickable_api_get(endpoint: str, api_key: str, params: dict = None) -> Optional[dict]:
    """Call Rebrickable API. Returns parsed JSON or None."""
    base = "https://rebrickable.com/api/v3/lego"
    url = f"{base}/{endpoint}/"
    headers = {"Authorization": f"key {api_key}"}
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{query}"
    try:
        if REQUESTS:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        else:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
    except Exception as e:
        print(f"  API error {endpoint}: {e}")
        return None


# ============================================================================
# 1. Johansson 2023 Dataset
# ============================================================================

JOHANSSON_FIGSHARE_BASE = (
    "https://figshare.com/ndownloader/files/"
)

# Figshare file IDs for the Johansson 2023 dataset
# Paper: https://www.nature.com/articles/s41597-023-02682-2
# Figshare: https://figshare.com/articles/dataset/LEGO_Brick_Images/22591796
JOHANSSON_FILES = [
    # (figshare_file_id, filename, description)
    ("41239685", "real_images_part1.zip",   "Real photos part 1 (~40K images)"),
    ("41239688", "real_images_part2.zip",   "Real photos part 2 (~40K images)"),
    ("41239691", "real_images_part3.zip",   "Real photos part 3 (~40K images)"),
    ("41239694", "real_images_part4.zip",   "Real photos part 4 (~35K images)"),
    ("41239697", "metadata.csv",            "Image metadata with BrickLink part IDs"),
]


def download_johansson(output_dir: Path, dry_run: bool = False) -> bool:
    """
    Download the Johansson 2023 LEGO dataset.

    This dataset contains ~155K real LEGO part photos with BrickLink part IDs,
    making it ideal as a validation set to measure how well synthetic renders
    generalise to real-world images.

    If direct download fails (Figshare rate limiting), prints manual download
    instructions with the permanent DOI URL.
    """
    print("\n=== Johansson 2023 Dataset ===")
    print("Paper: https://www.nature.com/articles/s41597-023-02682-2")
    print("~155K real LEGO part photos, BrickLink part IDs")

    dest_dir = output_dir / "johansson_2023"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"  [DRY RUN] Would download to {dest_dir}")
        return True

    success_count = 0
    for file_id, filename, description in JOHANSSON_FILES:
        dest = dest_dir / filename
        if dest.exists():
            print(f"  ✓ Already exists: {filename}")
            success_count += 1
            continue

        url = f"{JOHANSSON_FIGSHARE_BASE}{file_id}"
        print(f"  Downloading {filename} — {description}")
        if download_file(url, dest, desc=filename):
            print(f"  ✓ Downloaded: {filename} ({dest.stat().st_size / 1e6:.1f} MB)")
            success_count += 1
        else:
            print(f"  ✗ Failed: {filename}")

    # Extract ZIP files
    for file_id, filename, _ in JOHANSSON_FILES:
        if not filename.endswith('.zip'):
            continue
        zip_path = dest_dir / filename
        extract_dir = dest_dir / filename.replace('.zip', '')
        if not zip_path.exists():
            continue
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"  ✓ Already extracted: {filename}")
            continue
        print(f"  Extracting {filename}...")
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(dest_dir)
            print(f"  ✓ Extracted: {filename}")
        except Exception as e:
            print(f"  ✗ Extraction failed: {e}")

    if success_count < len(JOHANSSON_FILES):
        print("\n  MANUAL DOWNLOAD INSTRUCTIONS:")
        print("  The dataset is available at:")
        print("    https://figshare.com/articles/dataset/LEGO_Brick_Images/22591796")
        print(f"  Download all files and extract to: {dest_dir}")

    return success_count > 0


def reorganize_johansson(output_dir: Path) -> Path:
    """
    Reorganize Johansson dataset into standard class-folder structure:
      johansson_2023/by_part/{part_num}/{image.jpg}

    Reads the metadata.csv to map images to part numbers.
    Returns path to the organized directory.
    """
    source = output_dir / "johansson_2023"
    meta_csv = source / "metadata.csv"
    organized = source / "by_part"

    if not meta_csv.exists():
        print(f"  metadata.csv not found at {meta_csv} — skipping reorganization")
        return organized

    print(f"\n  Reorganizing Johansson dataset by part number...")
    organized.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    with open(meta_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Expected columns: image_path, part_num (or part_id), ...
            part_num = row.get('part_num') or row.get('part_id') or row.get('partNum', '')
            img_path_str = row.get('image_path') or row.get('filename', '')

            if not part_num or not img_path_str:
                skipped += 1
                continue

            src_img = source / img_path_str
            if not src_img.exists():
                skipped += 1
                continue

            dest_class = organized / str(part_num)
            dest_class.mkdir(exist_ok=True)
            dest_img = dest_class / src_img.name

            if not dest_img.exists():
                try:
                    shutil.copy2(src_img, dest_img)
                    count += 1
                except Exception:
                    skipped += 1

    print(f"  Organized {count} images into {len(list(organized.iterdir()))} part classes")
    if skipped:
        print(f"  Skipped {skipped} entries (missing files or metadata)")

    return organized


# ============================================================================
# 2. B200 LEGO Dataset (2024)
# ============================================================================

B200_URL = "https://github.com/LEGO-challenge/LEGO-dataset/archive/refs/heads/main.zip"


def download_b200(output_dir: Path, dry_run: bool = False) -> bool:
    """
    Download the B200 LEGO dataset — 2000 high-quality renders for 200 parts.
    Useful for render quality comparison and fine-tuning.
    """
    print("\n=== B200 LEGO Dataset (2024) ===")
    print("200 parts × 10 renders each, high-quality synthetic images")
    print("Source: https://github.com/LEGO-challenge/LEGO-dataset")

    dest_dir = output_dir / "b200"
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_dest = dest_dir / "b200_dataset.zip"

    if dry_run:
        print(f"  [DRY RUN] Would download to {dest_dir}")
        return True

    if zip_dest.exists() or (dest_dir / "LEGO-dataset-main").exists():
        print(f"  ✓ Already downloaded")
        return True

    print(f"  Downloading B200 dataset...")
    if download_file(B200_URL, zip_dest, desc="B200"):
        print(f"  Extracting...")
        try:
            with zipfile.ZipFile(zip_dest) as z:
                z.extractall(dest_dir)
            print(f"  ✓ B200 dataset ready at {dest_dir}")
            return True
        except Exception as e:
            print(f"  ✗ Extraction failed: {e}")
            return False
    return False


# ============================================================================
# 3. Rebrickable Catalog Images (via API)
# ============================================================================

def download_rebrickable_images(
    output_dir: Path,
    api_key: str,
    top_n: int = 500,
    dry_run: bool = False,
) -> bool:
    """
    Download part images from Rebrickable's API.

    Downloads the top-N most common parts' catalog images with BrickLink
    part IDs as class labels. These are the official part reference images
    and provide real-photo ground truth at scale.

    Requires REBRICKABLE_API_KEY env var.
    """
    print(f"\n=== Rebrickable Catalog Images (top {top_n} parts) ===")
    dest_dir = output_dir / "rebrickable"
    dest_dir.mkdir(parents=True, exist_ok=True)
    meta_path = dest_dir / "metadata.json"

    if dry_run:
        print(f"  [DRY RUN] Would download to {dest_dir}")
        return True

    # Load cached part list if available
    parts_cache = dest_dir / "parts_list.json"
    if parts_cache.exists():
        with open(parts_cache) as f:
            all_parts = json.load(f)
        print(f"  Using cached part list ({len(all_parts)} parts)")
    else:
        print(f"  Fetching part list from Rebrickable...")
        all_parts = []
        page = 1
        page_size = 1000

        while len(all_parts) < top_n * 2:   # fetch extra to allow filtering
            data = rebrickable_api_get("parts", api_key, {
                "page": page, "page_size": page_size, "ordering": "-year_from"
            })
            if not data or not data.get("results"):
                break
            all_parts.extend(data["results"])
            if not data.get("next"):
                break
            page += 1
            time.sleep(0.2)

        with open(parts_cache, 'w') as f:
            json.dump(all_parts[:top_n * 2], f)
        print(f"  Fetched {len(all_parts)} parts, cached to {parts_cache}")

    # Download images
    parts_to_download = [p for p in all_parts if p.get("part_img_url")][:top_n]
    print(f"  Downloading images for {len(parts_to_download)} parts...")

    metadata = {}
    bar = make_progress_bar(len(parts_to_download), "Rebrickable images")
    success = 0

    for part in parts_to_download:
        part_num = part["part_num"]
        img_url = part.get("part_img_url", "")
        if not img_url:
            bar.update()
            continue

        part_dir = dest_dir / "by_part" / part_num
        part_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(img_url).suffix or ".png"
        dest_img = part_dir / f"{part_num}_catalog{ext}"

        if not dest_img.exists():
            if download_file(img_url, dest_img):
                success += 1
            time.sleep(0.05)  # polite rate limiting
        else:
            success += 1

        metadata[part_num] = {
            "part_num": part_num,
            "name": part.get("name", ""),
            "part_cat_id": part.get("part_cat_id"),
            "year_from": part.get("year_from"),
        }
        bar.update()

    bar.close()

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Downloaded {success}/{len(parts_to_download)} images to {dest_dir}/by_part/")
    return success > 0


# ============================================================================
# 4. Build Train/Val Splits
# ============================================================================

def build_splits(
    output_dir: Path,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Scan all downloaded datasets and build train/val symlink trees.

    Output structure:
      output_dir/splits/train/{part_num}/{image}  (symlinks)
      output_dir/splits/val/{part_num}/{image}    (symlinks)
      output_dir/splits/split_info.json           (statistics)

    The validation set is drawn from real photos only (Johansson, Rebrickable),
    since the goal is to measure real-world performance.
    """
    print("\n=== Building Train/Val Splits ===")
    splits_dir = output_dir / "splits"
    train_dir = splits_dir / "train"
    val_dir = splits_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    # Collect images by source and part
    source_dirs = {
        "johansson_real": output_dir / "johansson_2023" / "by_part",
        "rebrickable":    output_dir / "rebrickable" / "by_part",
        # Renders go to train only (never val — we're measuring sim-to-real gap)
        "renders_train":  output_dir / "renders",
        "b200_train":     output_dir / "b200",
    }

    real_sources = {"johansson_real", "rebrickable"}
    all_real_parts: Dict[str, List[Path]] = {}   # part_num -> [image paths]
    all_train_paths: List[Tuple[str, Path]] = [] # [(part_num, path), ...]

    for source_name, source_path in source_dirs.items():
        if not source_path.exists():
            continue

        is_real = source_name in real_sources

        for part_dir in source_path.iterdir():
            if not part_dir.is_dir():
                continue
            part_num = part_dir.name
            images = list(part_dir.glob("*.jpg")) + list(part_dir.glob("*.png"))

            if not images:
                continue

            if is_real:
                if part_num not in all_real_parts:
                    all_real_parts[part_num] = []
                all_real_parts[part_num].extend(images)
            else:
                for img in images:
                    all_train_paths.append((part_num, img))

    # Allocate val from real images; remaining real images + all renders → train
    n_val_total = 0
    n_train_total = 0
    part_stats = {}

    for part_num, real_images in all_real_parts.items():
        random.shuffle(real_images)
        n_val = max(1, int(len(real_images) * val_fraction))
        val_imgs = real_images[:n_val]
        train_imgs = real_images[n_val:]

        val_part_dir = val_dir / part_num
        val_part_dir.mkdir(exist_ok=True)
        for img in val_imgs:
            link = val_part_dir / img.name
            if not link.exists():
                try:
                    link.symlink_to(img.resolve())
                except OSError:
                    shutil.copy2(img, link)
            n_val_total += 1

        train_part_dir = train_dir / part_num
        train_part_dir.mkdir(exist_ok=True)
        for img in train_imgs:
            link = train_part_dir / img.name
            if not link.exists():
                try:
                    link.symlink_to(img.resolve())
                except OSError:
                    shutil.copy2(img, link)
            n_train_total += 1

        part_stats[part_num] = {
            "train_real": len(train_imgs),
            "val_real": len(val_imgs),
        }

    # Add synthetic renders to train only
    for part_num, img in all_train_paths:
        train_part_dir = train_dir / part_num
        train_part_dir.mkdir(exist_ok=True)
        link = train_part_dir / img.name
        if not link.exists():
            try:
                link.symlink_to(img.resolve())
            except OSError:
                shutil.copy2(img, link)
        n_train_total += 1
        if part_num in part_stats:
            part_stats[part_num]["train_synth"] = \
                part_stats[part_num].get("train_synth", 0) + 1

    # Save summary
    split_info = {
        "n_train": n_train_total,
        "n_val": n_val_total,
        "n_parts_train": len(list(train_dir.iterdir())),
        "n_parts_val": len(list(val_dir.iterdir())),
        "val_fraction": val_fraction,
        "seed": seed,
        "note": "Val = real photos only. Train = remaining real + all synthetic renders.",
        "per_part": part_stats,
    }

    with open(splits_dir / "split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"  ✓ Train: {n_train_total} images across {split_info['n_parts_train']} parts")
    print(f"  ✓ Val:   {n_val_total} images across {split_info['n_parts_val']} parts")
    print(f"  ✓ Split info: {splits_dir / 'split_info.json'}")
    print(f"\n  Use these paths in training:")
    print(f"    --data-dir {train_dir}")
    print(f"    # Validate against: {val_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download public LEGO part datasets for BrickScan training"
    )
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Root output directory (default: ./data)')
    parser.add_argument('--all', action='store_true',
                        help='Download all available datasets')
    parser.add_argument('--johansson', action='store_true',
                        help='Download Johansson 2023 real photo dataset (~155K images)')
    parser.add_argument('--b200', action='store_true',
                        help='Download B200 synthetic renders dataset (200 parts)')
    parser.add_argument('--rebrickable', action='store_true',
                        help='Download Rebrickable catalog images (requires API key)')
    parser.add_argument('--top-n', type=int, default=500,
                        help='Number of parts to download from Rebrickable (default: 500)')
    parser.add_argument('--build-splits', action='store_true',
                        help='Build train/val splits from all downloaded data')
    parser.add_argument('--val-fraction', type=float, default=0.15,
                        help='Fraction of real images to reserve for validation (default: 0.15)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be downloaded without doing it')

    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"BrickScan Dataset Downloader")
    print(f"Output directory: {output_dir}")

    do_johansson = args.all or args.johansson
    do_b200 = args.all or args.b200
    do_rebrickable = args.all or args.rebrickable

    if do_johansson:
        ok = download_johansson(output_dir, dry_run=args.dry_run)
        if ok and not args.dry_run:
            reorganize_johansson(output_dir)

    if do_b200:
        download_b200(output_dir, dry_run=args.dry_run)

    if do_rebrickable:
        api_key = os.environ.get("REBRICKABLE_API_KEY", "")
        if not api_key:
            print("\nERROR: --rebrickable requires REBRICKABLE_API_KEY environment variable")
            print("  Get a free key at: https://rebrickable.com/api/")
            print("  Then: export REBRICKABLE_API_KEY='your_key'")
        else:
            download_rebrickable_images(
                output_dir, api_key,
                top_n=args.top_n,
                dry_run=args.dry_run,
            )

    if args.build_splits:
        build_splits(output_dir, val_fraction=args.val_fraction)

    if not any([do_johansson, do_b200, do_rebrickable, args.build_splits]):
        parser.print_help()
        print("\nExamples:")
        print("  python download_lego_datasets.py --all --output-dir ./data")
        print("  python download_lego_datasets.py --johansson --build-splits")
        print("  export REBRICKABLE_API_KEY=xxx")
        print("  python download_lego_datasets.py --rebrickable --top-n 1000 --build-splits")


if __name__ == '__main__':
    main()
