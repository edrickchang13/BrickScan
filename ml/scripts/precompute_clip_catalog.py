#!/usr/bin/env python3
"""
Precompute OpenAI CLIP image embeddings for the Rebrickable part catalog.

Produces a pickle in the exact format `backend/app/ml/embedding_library.py`
already loads: `{"embeddings": {part_num: np.float32_vec}}`. Dropping the
file at the path EmbeddingLibrary expects makes the backend's
`_safe_local_predict()` cascade pick it up as a k-NN voter on the next
scan. **Zero new backend code needed.**

CLIP ViT-B/32 is a 512-dim image/text model trained on 400M web pairs;
it transfers surprisingly well to LEGO parts because Rebrickable's
catalog photos look similar to what users actually scan (bricks on a
clean background). It's a solid zero-shot baseline until DGX Spark
trains the contrastive DINOv2 encoder.

Catalog image source priority per part:
  1. BrickLink CDN: https://img.bricklink.com/ItemImage/PN/11/{part_num}.png
  2. Rebrickable elements: https://cdn.rebrickable.com/media/parts/elements/{part_num}.jpg

Both have gaps — 2-tier fallback maximises coverage.

Output path (default): backend/data/clip_catalog_embeddings.pkl
EmbeddingLibrary reads from `backend/data/embeddings_cache.pkl` by default
— pass `--output` if you want to drop-in replace that.

Usage:
  ./backend/venv/bin/python3 ml/scripts/precompute_clip_catalog.py \\
      --limit 500 \\
      --parts-csv ml/data/parts.csv
"""

from __future__ import annotations

import argparse
import io
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clip_precompute")


def load_top_parts(parts_csv: Path, limit: int,
                   synthetic_dir: Optional[Path] = None) -> List[str]:
    """
    Build the target part list for embedding.

    Strategy:
      1. Start with a curated 'most common LEGO parts' list (prefix).
         parts.csv on its own sorts alphabetically, which surfaces
         obscure stickers first — useless for us. The curated list covers
         the parts users actually scan.
      2. Extend with every class directory in the local synthetic_dataset
         (if provided) — that's what we trained the classifier on, so
         matching CLIP embeddings are directly useful.
      3. Backfill with remaining rows from parts.csv up to `limit`.
    """
    COMMON = [
        # Basic bricks
        "3001", "3002", "3003", "3004", "3005", "3009", "3010",
        # Plates
        "3020", "3021", "3022", "3023", "3024", "3025", "3030", "3031", "3032", "3033", "3034", "3035",
        "3710", "3795", "87079",
        # Tiles
        "3068b", "3069b", "3070b", "2412b", "6636",
        # Slopes
        "3039", "3040", "3665", "3660",
        # Round plates / cylinders
        "4073", "4274", "4286", "4477", "6091",
        # Technic
        "3700", "3701", "3702", "3703", "6536", "6558", "32013", "32014",
        # Minifigure staples
        "3626c", "3815", "3816", "3817", "3818", "3819",
        # Modern common pieces
        "15068", "87087", "98138", "11477", "85984", "87580", "99207", "3024",
    ]

    picked: List[str] = []
    seen: set = set()
    for pn in COMMON:
        if pn not in seen:
            picked.append(pn)
            seen.add(pn)

    # Prefer parts we actually trained on (local synthetic corpus classes)
    if synthetic_dir and synthetic_dir.exists():
        for d in sorted(synthetic_dir.iterdir()):
            if not d.is_dir():
                continue
            pn = d.name
            if pn not in seen:
                picked.append(pn)
                seen.add(pn)
            if len(picked) >= limit:
                break

    # Backfill from parts.csv to hit the requested limit
    if parts_csv.exists() and len(picked) < limit:
        import csv as _csv
        try:
            with open(parts_csv, encoding="utf-8", errors="ignore") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    pn = (row.get("part_num") or "").strip()
                    if pn and pn not in seen:
                        picked.append(pn)
                        seen.add(pn)
                        if len(picked) >= limit:
                            break
        except Exception as e:
            log.warning("Couldn't parse parts.csv (%s) — skipping backfill", e)

    log.info("Target list: %d parts (common=%d, from synthetic=%d, csv-backfill=%d)",
             len(picked), len(COMMON),
             max(0, len(picked) - len(COMMON)),
             max(0, len(picked) - len(COMMON)))
    return picked[:limit]


def fetch_image(part_num: str, cache_dir: Path,
                timeout: int = 15) -> Optional[bytes]:
    """Try BrickLink first, then Rebrickable. Cache on disk; return bytes or None."""
    import requests

    # Cache key — safe filename from the part_num
    safe_pn = "".join(c if c.isalnum() or c in "-_" else "_" for c in part_num)
    for ext in (".png", ".jpg"):
        p = cache_dir / f"{safe_pn}{ext}"
        if p.exists() and p.stat().st_size > 500:
            try:
                return p.read_bytes()
            except Exception:
                continue

    urls = [
        (f"https://img.bricklink.com/ItemImage/PN/11/{part_num}.png", ".png"),
        (f"https://cdn.rebrickable.com/media/parts/elements/{part_num}.jpg", ".jpg"),
    ]
    for url, ext in urls:
        try:
            r = requests.get(url, timeout=timeout,
                             headers={"User-Agent": "BrickScan-CLIP-Precompute/1.0"})
            if r.status_code == 200 and len(r.content) > 500:
                out = cache_dir / f"{safe_pn}{ext}"
                out.write_bytes(r.content)
                return r.content
        except requests.RequestException as e:
            log.debug("Fetch failed %s: %s", url, e)
    return None


def embed_batch(model, preprocess, images: List[bytes], device):
    """Run CLIP over a batch of image bytes. Returns (N, 512) float32 tensor."""
    import torch
    from PIL import Image

    tensors = []
    for img_bytes in images:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensors.append(preprocess(img))
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)   # L2 normalise — matches EmbeddingLibrary
    return feats.cpu().numpy().astype("float32")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts-csv", type=Path,
                        default=Path("data_pipeline/rebrickable_data/parts.csv"),
                        help="CSV with part_num column (default: data_pipeline/rebrickable_data/parts.csv)")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max number of parts to embed (default 500)")
    parser.add_argument("--output", type=Path,
                        default=Path("backend/data/embeddings_cache.pkl"),
                        help="Output pickle path (default matches EmbeddingLibrary's expected location)")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("ml/data/catalog_image_cache"),
                        help="Where to cache downloaded catalog images")
    parser.add_argument("--model", default="ViT-B-32",
                        help="open_clip model name (default ViT-B-32)")
    parser.add_argument("--pretrained", default="openai",
                        help="open_clip pretrained weights tag (default openai)")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import open_clip
        import torch
        import numpy as np
    except ImportError as e:
        log.error("Missing dep: %s — run: ./backend/venv/bin/pip install open_clip_torch", e)
        return 2

    log.info("Loading CLIP %s (%s)…", args.model, args.pretrained)
    device = "cpu"  # CPU is fine for embedding 500 images
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device,
    )
    model.eval()

    part_nums = load_top_parts(
        args.parts_csv, args.limit,
        synthetic_dir=Path("ml/data/synthetic_dataset"),
    )
    log.info("Target parts: %d", len(part_nums))

    embeddings: Dict[str, "np.ndarray"] = {}
    misses: List[str] = []
    buffer_imgs: List[bytes] = []
    buffer_keys: List[str] = []

    def flush():
        if not buffer_imgs:
            return
        try:
            feats = embed_batch(model, preprocess, buffer_imgs, device)
            for k, f in zip(buffer_keys, feats):
                embeddings[k] = f
        except Exception as e:
            log.warning("Batch embed failed: %s — skipping %d images", e, len(buffer_imgs))
        buffer_imgs.clear()
        buffer_keys.clear()

    for i, pn in enumerate(part_nums):
        img = fetch_image(pn, args.cache_dir)
        if img is None:
            misses.append(pn)
            continue
        buffer_imgs.append(img)
        buffer_keys.append(pn)
        if len(buffer_imgs) >= args.batch_size:
            flush()
        if (i + 1) % 25 == 0:
            log.info("Progress: %d / %d (embedded=%d  misses=%d)",
                     i + 1, len(part_nums), len(embeddings), len(misses))
    flush()

    log.info("=" * 60)
    log.info("Embedded: %d / %d parts", len(embeddings), len(part_nums))
    log.info("Missing catalog image for: %d parts (sample: %s)", len(misses), misses[:8])
    log.info("Embedding dim: %d", len(next(iter(embeddings.values()))) if embeddings else 0)

    # Write the pickle in the exact format EmbeddingLibrary loads
    with open(args.output, "wb") as f:
        pickle.dump({"embeddings": embeddings}, f, protocol=4)
    log.info("Saved: %s (%.1f MB)", args.output, args.output.stat().st_size / 1e6)

    # Companion miss-list for debugging
    miss_path = args.output.with_suffix(".misses.txt")
    miss_path.write_text("\n".join(misses))
    log.info("Miss list: %s (%d parts)", miss_path, len(misses))

    return 0


if __name__ == "__main__":
    sys.exit(main())
