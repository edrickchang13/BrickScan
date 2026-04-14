"""
Download LEGO part images from Rebrickable + BrickLink for training data.

Data sources (no API key needed):
  - Rebrickable bulk CSVs: colors, parts, elements (part+color combos)
    https://rebrickable.com/downloads/
  - BrickLink part images:
    https://img.bricklink.com/ItemImage/PN/{bl_color_id}/{part_num}.png
  - Rebrickable CDN images (fallback):
    https://cdn.rebrickable.com/media/parts/photos/{part_num}.jpg

Output layout (compatible with dataset.py):
  data/
    images/
      {part_num}/
        {color_id}.jpg      ← per-part-color image
    index.csv               ← image_path, part_num, color_id, color_name, color_r, color_g, color_b

Usage:
  python download_rebrickable.py                       # download everything
  python download_rebrickable.py --max-parts 500       # quick test with 500 most common parts
  python download_rebrickable.py --api-key YOUR_KEY    # better image coverage via Rebrickable API
  python download_rebrickable.py --workers 32          # parallel downloads (default: 16)
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR
IMAGES_DIR = DATA_DIR / "images"
INDEX_CSV = DATA_DIR / "index.csv"
CACHE_DIR = DATA_DIR / ".cache"

# ── Rebrickable bulk download URLs ─────────────────────────────────────────────
REBRICKABLE_BASE = "https://rebrickable.com/media/downloads"
BULK_URLS = {
    "colors":    f"{REBRICKABLE_BASE}/colors.csv.gz",
    "parts":     f"{REBRICKABLE_BASE}/parts.csv.gz",
    "elements":  f"{REBRICKABLE_BASE}/elements.csv.gz",
    "inv_parts": f"{REBRICKABLE_BASE}/inventory_parts.csv.gz",
}

# ── BrickLink color ID map (Rebrickable ID → BrickLink ID) ────────────────────
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
log = logging.getLogger("download")

HEADERS = {
    "User-Agent": "BrickScan-ML/1.0 (training data downloader)",
    "Accept": "image/webp,image/png,image/jpeg,*/*",
}


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _fetch(url: str, timeout: int = 20) -> Optional[bytes]:
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def _fetch_image(url: str) -> Optional[Image.Image]:
    data = _fetch(url)
    if data and len(data) > 1000:
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            if img.width >= 32 and img.height >= 32:
                return img
        except Exception:
            pass
    return None


# ── Bulk CSV download ──────────────────────────────────────────────────────────

def _load_bulk_csv(name: str, url: str) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{name}.csv"

    if cache_path.exists() and cache_path.stat().st_size > 1000:
        log.info(f"  cached {name}.csv")
        return pd.read_csv(cache_path, low_memory=False)

    log.info(f"  downloading {name}…")
    data = _fetch(url, timeout=120)
    if data is None:
        log.error(f"  failed to download {name}")
        return pd.DataFrame()

    try:
        import gzip
        csv_bytes = gzip.decompress(data)
        df = pd.read_csv(io.StringIO(csv_bytes.decode("utf-8")), low_memory=False)
        df.to_csv(cache_path, index=False)
        log.info(f"  {name}: {len(df):,} rows")
        return df
    except Exception as e:
        log.error(f"  parse error for {name}: {e}")
        return pd.DataFrame()


def load_rebrickable_data(max_parts: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("Loading Rebrickable bulk data…")
    colors_df = _load_bulk_csv("colors", BULK_URLS["colors"])
    parts_df  = _load_bulk_csv("parts",  BULK_URLS["parts"])
    elements_df = _load_bulk_csv("elements", BULK_URLS["elements"])

    if elements_df.empty or "part_num" not in elements_df.columns:
        log.warning("elements.csv empty, falling back to inventory_parts.csv")
        inv_df = _load_bulk_csv("inv_parts", BULK_URLS["inv_parts"])
        combos_df = inv_df[["part_num", "color_id"]].drop_duplicates() if not inv_df.empty else pd.DataFrame()
    else:
        combos_df = elements_df[["part_num", "color_id"]].drop_duplicates()

    # Remove "any color" rows
    combos_df = combos_df[combos_df["color_id"].astype(int) >= 0]

    if max_parts:
        top_parts = combos_df["part_num"].value_counts().head(max_parts).index
        combos_df = combos_df[combos_df["part_num"].isin(top_parts)]

    log.info(f"  {len(combos_df):,} combos | {combos_df['part_num'].nunique():,} parts | {combos_df['color_id'].nunique():,} colors")
    return parts_df, colors_df, combos_df


def build_color_meta(colors_df: pd.DataFrame) -> Dict[int, dict]:
    meta: Dict[int, dict] = {}
    for _, row in colors_df.iterrows():
        try:
            rgb = str(row.get("rgb", "808080") or "808080").zfill(6)
            r, g, b = int(rgb[0:2], 16), int(rgb[2:4], 16), int(rgb[4:6], 16)
        except Exception:
            r = g = b = 128
        meta[int(row["id"])] = {"name": str(row.get("name", row["id"])), "r": r, "g": g, "b": b}
    return meta


# ── Image download ─────────────────────────────────────────────────────────────

def _try_download_image(part_num: str, color_id: int, api_key: Optional[str]) -> Optional[Image.Image]:
    # 1. BrickLink (best per-color coverage)
    bl_color = BL_COLOR_MAP.get(color_id, color_id)
    img = _fetch_image(f"https://img.bricklink.com/ItemImage/PN/{bl_color}/{part_num}.png")
    if img:
        return img

    # 2. Rebrickable CDN default photo
    img = _fetch_image(f"https://cdn.rebrickable.com/media/parts/photos/{part_num}.jpg")
    if img:
        return img

    # 3. Rebrickable API (only if key provided)
    if api_key:
        data = _fetch(f"https://rebrickable.com/api/v3/lego/parts/{part_num}/?key={api_key}")
        if data:
            try:
                img_url = json.loads(data).get("part_img_url", "")
                if img_url:
                    img = _fetch_image(img_url)
                    if img:
                        return img
            except Exception:
                pass

    return None


def _process_row(row: dict, color_meta: Dict[int, dict], api_key: Optional[str], image_size: int) -> Optional[dict]:
    part_num = str(row["part_num"])
    color_id = int(row["color_id"])

    out_dir = IMAGES_DIR / part_num
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{color_id}.jpg"

    meta = color_meta.get(color_id, {"name": str(color_id), "r": 128, "g": 128, "b": 128})

    # Skip if already downloaded
    if out_path.exists() and out_path.stat().st_size > 1000:
        return {
            "image_path": f"{part_num}/{color_id}.jpg",
            "part_num": part_num,
            "color_id": color_id,
            "color_name": meta["name"],
            "color_r": meta["r"], "color_g": meta["g"], "color_b": meta["b"],
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
        "image_path": f"{part_num}/{color_id}.jpg",
        "part_num": part_num,
        "color_id": color_id,
        "color_name": meta["name"],
        "color_r": meta["r"], "color_g": meta["g"], "color_b": meta["b"],
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",    default=os.getenv("REBRICKABLE_KEY", ""))
    parser.add_argument("--max-parts",  type=int, default=None)
    parser.add_argument("--workers",    type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    _, colors_df, combos_df = load_rebrickable_data(max_parts=args.max_parts)
    if combos_df.empty:
        log.error("No combos found. Check network connection.")
        sys.exit(1)

    color_meta = build_color_meta(colors_df)
    rows = combos_df.to_dict("records")

    log.info(f"Downloading {len(rows):,} images with {args.workers} workers…")
    results: List[dict] = []
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_row, r, color_meta, args.api_key or None, args.image_size): r for r in rows}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                results.append(result)
            else:
                failed += 1
            if i % 1000 == 0 or i == len(rows):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1)
                eta = (len(rows) - i) / max(rate, 0.001)
                log.info(f"  {i:,}/{len(rows):,} | ok={len(results):,} fail={failed:,} | {rate:.0f}/s | ETA {eta/60:.1f}m")

    if not results:
        log.error("No images downloaded!")
        sys.exit(1)

    index_df = pd.DataFrame(results, columns=[
        "image_path", "part_num", "color_id", "color_name", "color_r", "color_g", "color_b"
    ])
    index_df.to_csv(INDEX_CSV, index=False)

    log.info(f"\n{'='*55}")
    log.info(f"  Images downloaded: {len(results):,}  ({failed:,} failed)")
    log.info(f"  Unique parts:      {index_df['part_num'].nunique():,}")
    log.info(f"  Unique colors:     {index_df['color_name'].nunique():,}")
    log.info(f"  Index CSV:         {INDEX_CSV}")
    log.info(f"{'='*55}")
    log.info("Next step: cd training && bash train.sh")


if __name__ == "__main__":
    main()
