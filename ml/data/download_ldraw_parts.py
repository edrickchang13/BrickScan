#!/usr/bin/env python3
"""
download_ldraw_parts.py — Batch download LDraw .dat files for training parts.

Downloads official LDraw part files (.dat) from the LDraw Parts Library for
every part number in your training set. These are used by batch_render.py /
render_parts.py to generate synthetic training images in Blender.

Output:
  ldraw_dir/
    LDConfig.ldr          (colour definitions)
    parts/
      3001.dat
      3004.dat
      ...
    p/                    (primitives — auto-resolved when parts reference them)

Usage:
  # Download .dat files for every part in your training set:
  python3 download_ldraw_parts.py \\
    --parts-source ~/brickscan/ml/training_data/huggingface_legobricks/images/ \\
    --ldraw-dir    ~/brickscan/ml/data/ldraw/

  # Or pass a flat text file with one part number per line:
  python3 download_ldraw_parts.py \\
    --parts-file   ./top_1000_parts.txt \\
    --ldraw-dir    ~/brickscan/ml/data/ldraw/

  # Download the complete official library (large — ~100MB unzipped):
  python3 download_ldraw_parts.py --full-library --ldraw-dir ~/brickscan/ml/data/ldraw/
"""

import argparse
import logging
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download_ldraw_parts")

# LDraw official parts library
LDRAW_COMPLETE_ZIP = "https://library.ldraw.org/library/updates/complete.zip"
LDRAW_PARTS_BASE   = "https://library.ldraw.org/library/official/parts"
LDRAW_P_BASE       = "https://library.ldraw.org/library/official/p"
LDRAW_CONFIG_URL   = "https://www.ldraw.org/library/official/LDConfig.ldr"

# How long to wait between requests (be polite to ldraw.org)
REQUEST_DELAY_S = 0.25


# ─── Part number discovery ────────────────────────────────────────────────────

def discover_parts_from_dir(images_dir: Path) -> List[str]:
    """
    Walk a training images directory (e.g. HuggingFace legobricks layout)
    and extract all part numbers from subdirectory names.

    Handles both:
      images_dir/<part_num>/img.jpg        (flat)
      images_dir/<int_index>/              (integer-indexed, needs dataset_info.json)
    """
    parts = set()
    dataset_info = images_dir.parent / "dataset_infos.json"

    # Try integer-index layout with mapping file
    if dataset_info.exists():
        import json
        try:
            with open(dataset_info) as f:
                info = json.load(f)
            # Walk the features/label/names list
            for entry in info.values():
                names = (entry.get("features", {})
                              .get("label", {})
                              .get("names", []))
                for name in names:
                    pn = _extract_part_num(name)
                    if pn:
                        parts.add(pn)
            if parts:
                log.info(f"Discovered {len(parts)} parts from dataset_infos.json")
                return sorted(parts)
        except Exception as e:
            log.warning(f"Could not parse dataset_infos.json: {e}")

    # Fall back to directory names
    for d in images_dir.iterdir():
        if d.is_dir():
            pn = _extract_part_num(d.name)
            if pn:
                parts.add(pn)

    log.info(f"Discovered {len(parts)} parts from directory names")
    return sorted(parts)


def discover_parts_from_file(parts_file: Path) -> List[str]:
    parts = []
    with open(parts_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts.append(line)
    return parts


def _extract_part_num(name: str) -> Optional[str]:
    """Extract a plausible LDraw part number from a string."""
    # Common patterns: "3001", "3001b", "32140", "u9061"
    m = re.match(r"^([a-zA-Z]?\d+[a-zA-Z0-9]*)$", name.strip())
    if m:
        return m.group(1)
    # Also handle "3001 Brick 2x4" style
    m = re.match(r"^([a-zA-Z]?\d+[a-zA-Z0-9]*)\s", name.strip())
    if m:
        return m.group(1)
    return None


# ─── Full library download ────────────────────────────────────────────────────

def download_full_library(ldraw_dir: Path):
    """Download and extract the complete official LDraw parts library (~100MB)."""
    import urllib.request

    ldraw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = ldraw_dir / "complete.zip"

    if not zip_path.exists():
        log.info(f"Downloading complete LDraw library from {LDRAW_COMPLETE_ZIP}")
        log.info("This is ~100MB and may take a few minutes…")
        urllib.request.urlretrieve(LDRAW_COMPLETE_ZIP, zip_path,
                                   reporthook=_progress_hook())
        print()

    log.info(f"Extracting to {ldraw_dir}…")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ldraw_dir)

    # The zip extracts into an "ldraw/" subfolder — flatten it
    extracted = ldraw_dir / "ldraw"
    if extracted.exists():
        for item in extracted.iterdir():
            target = ldraw_dir / item.name
            if not target.exists():
                item.rename(target)
        extracted.rmdir()

    log.info(f"LDraw library ready at {ldraw_dir}")


def _progress_hook():
    last_pct = [-1]
    def hook(count, block_size, total_size):
        if total_size <= 0:
            return
        pct = int(count * block_size * 100 / total_size)
        if pct != last_pct[0] and pct % 5 == 0:
            print(f"\r  {pct}%", end="", flush=True)
            last_pct[0] = pct
    return hook


# ─── Individual part download ─────────────────────────────────────────────────

def download_single_part(part_num: str, ldraw_dir: Path,
                          session=None) -> tuple[str, bool, str]:
    """
    Download a single .dat file for part_num.
    Tries parts/ first, then p/ (primitives).
    Returns (part_num, success, message).
    """
    import requests

    parts_dir = ldraw_dir / "parts"
    p_dir     = ldraw_dir / "p"
    parts_dir.mkdir(parents=True, exist_ok=True)
    p_dir.mkdir(parents=True, exist_ok=True)

    dest = parts_dir / f"{part_num}.dat"
    if dest.exists():
        return part_num, True, "already exists"

    # Try lowercase and uppercase variants
    for variant in [part_num, part_num.lower(), part_num.upper()]:
        for base_url, dest_dir in [
            (LDRAW_PARTS_BASE, parts_dir),
            (LDRAW_P_BASE,     p_dir),
        ]:
            url = f"{base_url}/{variant}.dat"
            try:
                sess = session or requests
                r = sess.get(url, timeout=15)
                if r.status_code == 200 and len(r.content) > 50:
                    out = dest_dir / f"{part_num}.dat"
                    out.write_bytes(r.content)
                    time.sleep(REQUEST_DELAY_S)
                    return part_num, True, f"downloaded from {base_url}"
            except Exception as e:
                continue

    return part_num, False, "not found in LDraw library"


def download_ldconfig(ldraw_dir: Path):
    """Download LDConfig.ldr if not present."""
    import urllib.request
    dest = ldraw_dir / "LDConfig.ldr"
    if dest.exists():
        log.info("LDConfig.ldr already present.")
        return
    log.info("Downloading LDConfig.ldr…")
    ldraw_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(LDRAW_CONFIG_URL, dest)
    log.info(f"LDConfig.ldr → {dest}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument("--parts-source",
                           help="Path to training images dir (discovers parts from subdirs)")
    src_group.add_argument("--parts-file",
                           help="Text file with one part number per line")
    src_group.add_argument("--full-library", action="store_true",
                           help="Download the entire LDraw library (~100MB)")

    parser.add_argument("--ldraw-dir", required=True,
                        help="Destination directory for LDraw files")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download workers (default 8, be polite: max 16)")
    parser.add_argument("--skip-ldconfig", action="store_true",
                        help="Skip downloading LDConfig.ldr")
    args = parser.parse_args()

    ldraw_dir = Path(args.ldraw_dir).expanduser().resolve()

    # Always get LDConfig.ldr unless skipped
    if not args.skip_ldconfig:
        download_ldconfig(ldraw_dir)

    # Full library mode
    if args.full_library:
        download_full_library(ldraw_dir)
        return

    # Discover parts
    if args.parts_source:
        parts = discover_parts_from_dir(Path(args.parts_source).expanduser())
    elif args.parts_file:
        parts = discover_parts_from_file(Path(args.parts_file).expanduser())
    else:
        parser.error("Specify --parts-source, --parts-file, or --full-library")
        return

    if not parts:
        log.error("No part numbers found.")
        sys.exit(1)

    log.info(f"Parts to download: {len(parts)}")
    log.info(f"Destination: {ldraw_dir}")

    # Check which already exist
    already = sum(1 for p in parts if (ldraw_dir / "parts" / f"{p}.dat").exists())
    log.info(f"Already downloaded: {already}/{len(parts)}")
    to_download = [p for p in parts if not (ldraw_dir / "parts" / f"{p}.dat").exists()]
    log.info(f"Downloading: {len(to_download)}")

    if not to_download:
        log.info("All parts already downloaded!")
        return

    # Download with thread pool
    try:
        import requests
        session = requests.Session()
        session.headers["User-Agent"] = "BrickScan-DataPipeline/1.0 (training data)"
    except ImportError:
        session = None

    success_count = 0
    fail_count = 0
    not_found = []

    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        _tqdm = None

    workers = min(args.workers, 16)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_single_part, p, ldraw_dir, session): p
            for p in to_download
        }

        iterator = _tqdm(as_completed(futures), total=len(futures),
                         desc="Downloading") if _tqdm else as_completed(futures)

        for future in iterator:
            part_num, ok, msg = future.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
                not_found.append(part_num)

    log.info("=" * 60)
    log.info(f"Done. Success: {success_count}, Not found: {fail_count}")

    if not_found:
        not_found_file = ldraw_dir / "not_found_parts.txt"
        not_found_file.write_text("\n".join(not_found))
        log.info(f"Parts not found in LDraw library saved to: {not_found_file}")
        log.info("These may have alternate IDs — check https://www.ldraw.org/cgi-bin/ptdetail.cgi")

    log.info(f"LDraw .dat files ready at: {ldraw_dir / 'parts'}")
    log.info(f"Next step: python3 batch_render.py --ldraw-dir {ldraw_dir} ...")


if __name__ == "__main__":
    main()
