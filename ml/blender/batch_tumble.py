#!/usr/bin/env python3
"""
Batch orchestrator for tumble rendering.
Runs in system Python (not inside Blender).
Spawns Blender processes to render parts from all orientations.
"""

import argparse
import csv
import logging
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional
import json

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# LDraw color utilities
try:
    from ldraw_colors import load_colors as _load_ldraw_colors
    _LDRAW_COLORS_AVAILABLE = True
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from ldraw_colors import load_colors as _load_ldraw_colors
        _LDRAW_COLORS_AVAILABLE = True
    except ImportError:
        _LDRAW_COLORS_AVAILABLE = False


def setup_logging(log_file: Path):
    """Configure logging to file and stdout"""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger('BrickScan-Tumble')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def hex_to_rgb_linear(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color to linear float RGB (0-1)."""
    def _srgb_to_linear(v: float) -> float:
        if v <= 0.04045:
            return v / 12.92
        return ((v + 0.055) / 1.055) ** 2.4

    h = hex_color.lstrip('#')
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = _srgb_to_linear(int(h[0:2], 16) / 255.0)
    g = _srgb_to_linear(int(h[2:4], 16) / 255.0)
    b = _srgb_to_linear(int(h[4:6], 16) / 255.0)
    return (r, g, b)


def load_colors(
    ldconfig_path: Optional[Path] = None,
    colors_csv: Optional[Path] = None,
) -> dict:
    """Load LEGO colors as linear RGB."""
    if _LDRAW_COLORS_AVAILABLE:
        entries = _load_ldraw_colors(
            ldconfig_path=ldconfig_path,
            rebrickable_csv=colors_csv,
        )
        return {
            code: {
                'name': e.name,
                'hex': e.hex_srgb,
                'rgb': e.rgb_linear,
                'alpha': e.alpha,
                'finish': e.finish,
            }
            for code, e in entries.items()
        }

    # Fallback: parse CSV
    if colors_csv and Path(colors_csv).exists():
        colors = {}
        with open(colors_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                color_id = int(row['id'])
                colors[color_id] = {
                    'name': row['name'],
                    'hex': row['rgb'],
                    'rgb': hex_to_rgb_linear(row['rgb']),
                }
        return colors

    raise RuntimeError("No color source available. Provide --ldconfig or --colors-csv.")


def find_ldraw_part_file(part_num: str, ldraw_parts_dir: Path) -> Optional[Path]:
    """Find a .dat file for the given part number in LDraw library"""
    # Try exact match first
    dat_file = ldraw_parts_dir / "parts" / f"{part_num}.dat"
    if dat_file.exists():
        return dat_file

    # Try with p/ subfolder (primitives)
    dat_file = ldraw_parts_dir / "p" / f"{part_num}.dat"
    if dat_file.exists():
        return dat_file

    # Try lowercase
    dat_file = ldraw_parts_dir / "parts" / f"{part_num.lower()}.dat"
    if dat_file.exists():
        return dat_file

    return None


def render_part_tumble(
    blender_exe: str,
    render_script: Path,
    part_num: str,
    part_file: Path,
    color_id: int,
    color_name: str,
    color_r: float,
    color_g: float,
    color_b: float,
    output_dir: Path,
    resolution: int = 224,
    index_csv: Optional[Path] = None,
) -> Tuple[bool, str]:
    """
    Spawn a Blender process to render a single part+color with tumble orientations.
    Returns (success, message).
    """
    try:
        cmd = [
            blender_exe,
            "--background",
            "--python", str(render_script),
            "--",
            "--part-file", str(part_file),
            "--output-dir", str(output_dir),
            "--color-r", str(color_r),
            "--color-g", str(color_g),
            "--color-b", str(color_b),
            "--part-num", str(part_num),
            "--color-id", str(color_id),
            "--color-name", str(color_name),
            "--resolution", str(resolution),
        ]

        if index_csv:
            cmd.extend(["--index-csv", str(index_csv)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout (48 orientations)
        )

        if result.returncode == 0:
            return True, f"Rendered {part_num} color {color_name} (48 orientations)"
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            return False, f"Failed: {part_num} {color_name} - {error_msg[:200]}"

    except subprocess.TimeoutExpired:
        return False, f"Timeout: {part_num} {color_name}"
    except Exception as e:
        return False, f"Error: {part_num} {color_name} - {str(e)[:200]}"


def main():
    parser = argparse.ArgumentParser(
        description="Batch render LEGO parts with 360° tumble orientations"
    )
    parser.add_argument(
        "--blender",
        default="blender",
        help="Path to Blender executable (default: 'blender')"
    )
    parser.add_argument(
        "--ldraw-dir",
        default="./ml/data/ldraw",
        help="Path to LDraw library root (default: ./ml/data/ldraw)"
    )
    parser.add_argument(
        "--ldconfig",
        default=None,
        help="Path to LDConfig.ldr for accurate LEGO color definitions"
    )
    parser.add_argument(
        "--colors-csv",
        default="./ml/data/colors.csv",
        help="Path to Rebrickable colors.csv (fallback)"
    )
    parser.add_argument(
        "--parts-dir",
        required=True,
        help="Directory containing .dat part files"
    )
    parser.add_argument(
        "--renders-dir",
        required=True,
        help="Output directory for tumble renders"
    )
    parser.add_argument(
        "--colors",
        type=int,
        nargs='+',
        default=None,
        help="Specific color IDs to render (if not set, uses all colors)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="Output image resolution (default: 224)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel Blender processes (default: 2, tumble is heavy)"
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=None,
        help="Limit number of parts to render (for testing)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip parts that already have tumble renders"
    )
    parser.add_argument(
        "--index-csv",
        default=None,
        help="CSV to append image metadata to (default: renders-dir/../tumble_index.csv)"
    )

    args = parser.parse_args()

    # Setup paths
    blender_exe = args.blender
    ldraw_dir = Path(args.ldraw_dir).resolve()
    colors_csv = Path(args.colors_csv).resolve()
    parts_dir = Path(args.parts_dir).resolve()
    renders_dir = Path(args.renders_dir).resolve()
    log_file = renders_dir.parent / "tumble_renders.log"

    ldconfig_path = Path(args.ldconfig).resolve() if args.ldconfig else (ldraw_dir / "LDConfig.ldr")
    render_script = Path(__file__).parent / "render_tumble.py"

    if not render_script.exists():
        print(f"ERROR: render_tumble.py not found at {render_script}")
        sys.exit(1)

    if args.index_csv:
        index_csv = Path(args.index_csv).resolve()
    else:
        index_csv = renders_dir.parent / "tumble_index.csv"

    logger = setup_logging(log_file)
    logger.info(f"Starting BrickScan tumble batch render")
    logger.info(f"Blender: {blender_exe}")
    logger.info(f"LDraw dir: {ldraw_dir}")
    logger.info(f"Parts dir: {parts_dir}")
    logger.info(f"Renders dir: {renders_dir}")
    logger.info(f"Workers: {args.workers} (tumble rendering is GPU-intensive)")

    # Load colors
    logger.info("Loading colors...")
    try:
        colors = load_colors(
            ldconfig_path=ldconfig_path if ldconfig_path.exists() or True else None,
            colors_csv=colors_csv if colors_csv.exists() else None,
        )
        logger.info(f"Loaded {len(colors)} colors")
    except Exception as e:
        logger.error(f"Failed to load colors: {e}")
        sys.exit(1)

    # Filter to requested colors
    if args.colors:
        colors = {cid: colors[cid] for cid in args.colors if cid in colors}
        logger.info(f"Filtered to {len(colors)} requested colors")

    # Find part files in parts_dir
    logger.info(f"Scanning {parts_dir} for .dat files...")
    part_files = sorted(parts_dir.glob("*.dat"))

    if not part_files:
        logger.error(f"No .dat files found in {parts_dir}")
        sys.exit(1)

    logger.info(f"Found {len(part_files)} part files")

    if args.max_parts:
        part_files = part_files[:args.max_parts]
        logger.info(f"Limited to {len(part_files)} parts")

    # Build render tasks
    tasks = []
    for part_file in part_files:
        part_num = part_file.stem

        part_output_dir = renders_dir / part_num
        if args.skip_existing and part_output_dir.exists():
            # Check if we have significant renders already (e.g., >40 of 48)
            existing = list(part_output_dir.glob("*.png"))
            if len(existing) >= 40:
                logger.info(f"Skipping {part_num} (already has {len(existing)} renders)")
                continue

        for color_id, color_info in colors.items():
            r, g, b = color_info['rgb']
            tasks.append((
                part_num,
                part_file,
                color_id,
                color_info['name'],
                r, g, b,
                part_output_dir,
            ))

    logger.info(f"Total tasks: {len(tasks)} ({len(part_files)} parts × {len(colors)} colors)")

    if not tasks:
        logger.warning("No tasks to render")
        sys.exit(0)

    # Render with progress bar
    success_count = 0
    failure_count = 0

    render_fn = lambda task: render_part_tumble(
        blender_exe,
        render_script,
        task[0], task[1], task[2], task[3], task[4], task[5], task[6], task[7],
        resolution=args.resolution,
        index_csv=index_csv,
    )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(render_fn, task): task for task in tasks}

        # Use tqdm if available
        if tqdm:
            iterator = tqdm(as_completed(futures), total=len(futures), desc="Rendering")
        else:
            iterator = as_completed(futures)

        for future in iterator:
            task = futures[future]
            try:
                success, msg = future.result()
                if success:
                    success_count += 1
                    logger.info(msg)
                else:
                    failure_count += 1
                    logger.error(msg)
            except Exception as e:
                failure_count += 1
                logger.error(f"Task error: {task[0]} - {e}")

    # Summary
    logger.info("=" * 60)
    logger.info(f"Tumble render complete!")
    logger.info(f"Success: {success_count}, Failures: {failure_count}")
    logger.info(f"Log: {log_file}")
    logger.info(f"Output: {renders_dir}")
    logger.info(f"Index CSV: {index_csv}")

    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
