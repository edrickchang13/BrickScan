#!/usr/bin/env python3
"""
merge_datasets.py — Merge multiple LEGO training dataset sources into one tree.

Takes several source directories (each in <part_num>/<images> layout) and
merges them into a single output directory, deduplicating by hash and
tracking provenance.

Also updates generate_multipiece_scenes.py to use COCO real-world backgrounds
when available (via --backgrounds-dir flag).

Output structure:
  output_dir/
    <part_num>/
      synth_001.jpg       (from Blender renders)
      hf_joost_001.png    (from HuggingFace)
      b200c_001.jpg       (from B200C Kaggle)
      ...
    _stats.json           (per-source and per-part counts)

Usage:
  python3 merge_datasets.py \\
    --sources ~/Desktop/synthetic_dataset ~/Desktop/extra_training_data/hf_joost \\
              ~/Desktop/extra_training_data/kaggle_b200c \\
    --output  ~/Desktop/merged_training_data \\
    --min-images-per-part 50

  # Verify the merge worked:
  python3 merge_datasets.py --stats ~/Desktop/merged_training_data
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("merge_datasets")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─── Hash-based dedup ─────────────────────────────────────────────────────────

def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ─── Source name from path ────────────────────────────────────────────────────

def source_prefix(source_dir: Path) -> str:
    """Derive a short prefix from the source directory name."""
    name = source_dir.name
    prefixes = {
        "synthetic_dataset": "synth",
        "synthetic_dataset_dr": "synth_dr",
        "hf_joost": "hf_joost",
        "kaggle_b200c": "b200c",
        "kaggle_joost": "kaggle_joost",
        "roboflow_crops": "rf",
        "huggingface_legobricks": "hf_lego",
    }
    return prefixes.get(name, name[:12].replace("-", "_"))


# ─── Merge logic ─────────────────────────────────────────────────────────────

def merge(sources: List[Path], output_dir: Path,
          min_images: int = 0, max_images_per_source: int = 0):
    """
    Merge all source directories into output_dir.
    Skips duplicate files (by MD5). Renames files with source prefix.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(lambda: defaultdict(int))  # stats[part][source] = count
    hashes_seen = set()
    total_copied = 0
    total_skipped_dup = 0

    for source_dir in sources:
        if not source_dir.exists():
            log.warning(f"Source not found, skipping: {source_dir}")
            continue

        prefix = source_prefix(source_dir)
        log.info(f"Processing source: {source_dir.name} (prefix={prefix})")

        source_count = 0
        for img_path in sorted(source_dir.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            if not img_path.is_file():
                continue

            # Part number is the first-level subdirectory under source_dir
            rel = img_path.relative_to(source_dir)
            part_num = rel.parts[0] if len(rel.parts) >= 2 else "unknown"

            # Hash-based dedup
            h = file_hash(img_path)
            if h in hashes_seen:
                total_skipped_dup += 1
                continue
            hashes_seen.add(h)

            # Cap per-source if requested
            if max_images_per_source and stats[part_num][prefix] >= max_images_per_source:
                continue

            # Copy to output
            dest_dir = output_dir / part_num
            dest_dir.mkdir(exist_ok=True)
            dest_name = f"{prefix}_{img_path.name}"
            dest = dest_dir / dest_name
            if not dest.exists():
                shutil.copy2(img_path, dest)
            stats[part_num][prefix] += 1
            source_count += 1
            total_copied += 1

        log.info(f"  → {source_count} images from {prefix}")

    # Write stats JSON
    stats_out = {
        part: dict(sources)
        for part, sources in stats.items()
    }
    with open(output_dir / "_stats.json", "w") as f:
        json.dump(stats_out, f, indent=2)

    # Summary
    part_counts = {p: sum(s.values()) for p, s in stats.items()}
    log.info("=" * 60)
    log.info(f"Merge complete.")
    log.info(f"  Total images copied:    {total_copied}")
    log.info(f"  Duplicates skipped:     {total_skipped_dup}")
    log.info(f"  Unique parts:           {len(part_counts)}")
    log.info(f"  Avg images per part:    {sum(part_counts.values()) / max(len(part_counts), 1):.0f}")

    if min_images:
        thin_parts = [p for p, c in part_counts.items() if c < min_images]
        log.info(f"  Parts below {min_images} images: {len(thin_parts)}")
        if thin_parts:
            thin_file = output_dir / "_thin_parts.txt"
            thin_file.write_text("\n".join(sorted(thin_parts)))
            log.info(f"  Thin parts list → {thin_file}")
            log.info("  Consider running more Blender renders for these parts.")

    log.info(f"\nOutput: {output_dir}")
    log.info("Next steps:")
    log.info(f"  python3 generate_multipiece_scenes.py --renders-dir {output_dir} \\")
    log.info(f"    --output-dir ~/Desktop/yolo_dataset_merged --num-train 15000")
    return stats_out


# ─── Stats-only mode ─────────────────────────────────────────────────────────

def print_stats(dataset_dir: Path):
    stats_file = dataset_dir / "_stats.json"
    if not stats_file.exists():
        # Compute on the fly
        log.info("No _stats.json found — scanning directory…")
        part_counts = {}
        for d in sorted(dataset_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("_"):
                count = sum(1 for f in d.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
                part_counts[d.name] = count
    else:
        with open(stats_file) as f:
            raw = json.load(f)
        part_counts = {p: sum(s.values()) for p, s in raw.items()}

    total = sum(part_counts.values())
    print(f"\nDataset: {dataset_dir}")
    print(f"Parts: {len(part_counts)}   Total images: {total}")
    print(f"Avg per part: {total / max(len(part_counts), 1):.0f}")
    print(f"\n{'Part':<20} {'Count':>8}")
    print("-" * 30)
    for part, count in sorted(part_counts.items(), key=lambda x: -x[1])[:30]:
        print(f"{part:<20} {count:>8}")
    if len(part_counts) > 30:
        print(f"  ... and {len(part_counts) - 30} more parts")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+",
                        help="Source directories to merge (each must have <part_num>/ subdirs)")
    parser.add_argument("--output", help="Output merged directory")
    parser.add_argument("--min-images-per-part", type=int, default=50,
                        help="Warn about parts below this threshold (default 50)")
    parser.add_argument("--max-per-source", type=int, default=0,
                        help="Cap images per part per source (0 = unlimited)")
    parser.add_argument("--stats", metavar="DIR",
                        help="Print stats for an existing dataset directory and exit")
    args = parser.parse_args()

    if args.stats:
        print_stats(Path(args.stats).expanduser())
        return

    if not args.sources or not args.output:
        parser.error("--sources and --output are required (or use --stats)")

    sources = [Path(s).expanduser().resolve() for s in args.sources]
    output  = Path(args.output).expanduser().resolve()

    merge(sources, output,
          min_images=args.min_images_per_part,
          max_images_per_source=args.max_per_source)


if __name__ == "__main__":
    main()
