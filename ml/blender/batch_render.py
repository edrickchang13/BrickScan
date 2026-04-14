#!/usr/bin/env python3
"""
Batch renderer orchestrator for LEGO synthetic training data.
Runs in system Python (not inside Blender).
Spawns multiple Blender processes to render parts in parallel.
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

# LDraw color utilities — sRGB→Linear conversion + LDConfig.ldr parser
try:
    from ldraw_colors import load_colors as _load_ldraw_colors, ColorEntry
    _LDRAW_COLORS_AVAILABLE = True
except ImportError:
    # Fallback if run from a different working directory
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from ldraw_colors import load_colors as _load_ldraw_colors, ColorEntry
        _LDRAW_COLORS_AVAILABLE = True
    except ImportError:
        _LDRAW_COLORS_AVAILABLE = False


# ==============================================================================
# LOGGING SETUP
# ==============================================================================

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

    logger = logging.getLogger('BrickScan')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ==============================================================================
# COLOR LOADING
# ==============================================================================

def _srgb_to_linear(v: float) -> float:
    """IEC 61966-2-1 sRGB → linear (used as fallback if ldraw_colors not available)."""
    if v <= 0.04045:
        return v / 12.92
    return ((v + 0.055) / 1.055) ** 2.4


def hex_to_rgb_linear(hex_color: str) -> Tuple[float, float, float]:
    """
    Convert hex color (e.g., '#05131D' or '05131D') → linear float RGB (0-1).

    sRGB hex values from LDraw/Rebrickable must be gamma-corrected to linear
    before passing to Blender's Principled BSDF — otherwise colours appear
    washed-out (brights) or too dark (darks).
    """
    h = hex_color.lstrip('#')
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = _srgb_to_linear(int(h[0:2], 16) / 255.0)
    g = _srgb_to_linear(int(h[2:4], 16) / 255.0)
    b = _srgb_to_linear(int(h[4:6], 16) / 255.0)
    return (r, g, b)


# Keep old name as alias for callers that import it directly
hex_to_rgb_float = hex_to_rgb_linear


# ==============================================================================
# LOADING PARTS AND COLORS
# ==============================================================================

def load_colors(
    ldconfig_path: Optional[Path] = None,
    colors_csv: Optional[Path] = None,
) -> dict:
    """
    Load LEGO colors as linear RGB, ready for Blender.

    Priority:
      1. LDConfig.ldr  — official LDraw color standard, most accurate
      2. Rebrickable colors.csv  — fallback with same hex values, less metadata
      3. Hardcoded 15-color minimal set  — last resort

    Returns: {color_id: {'name': str, 'hex': str, 'rgb': (r_lin, g_lin, b_lin)}}
    """
    if _LDRAW_COLORS_AVAILABLE:
        entries = _load_ldraw_colors(
            ldconfig_path=ldconfig_path,
            rebrickable_csv=colors_csv,
        )
        return {
            code: {
                'name': e.name,
                'hex':  e.hex_srgb,
                'rgb':  e.rgb_linear,   # ← proper linear values
                'alpha': e.alpha,
                'finish': e.finish,
            }
            for code, e in entries.items()
        }

    # Fallback: parse CSV with manual sRGB→linear conversion
    if colors_csv and Path(colors_csv).exists():
        colors = {}
        with open(colors_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                color_id = int(row['id'])
                colors[color_id] = {
                    'name': row['name'],
                    'hex':  row['rgb'],
                    'rgb':  hex_to_rgb_linear(row['rgb']),  # ← corrected
                }
        return colors

    raise RuntimeError(
        "No color source available. Provide --ldconfig or --colors-csv."
    )


def load_colors_from_csv(colors_csv: Path) -> dict:
    """Legacy wrapper — loads Rebrickable CSV with proper sRGB→linear conversion."""
    return load_colors(colors_csv=colors_csv)


def load_parts_subset(parts_file: Path, ldraw_parts_dir: Path) -> List[str]:
    """Load part numbers from file, verify existence in LDraw library"""
    parts = []
    with open(parts_file, 'r') as f:
        for line in f:
            part_num = line.strip()
            if part_num and not part_num.startswith('#'):
                parts.append(part_num)
    return parts


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


# ==============================================================================
# RENDERING TASK
# ==============================================================================

def render_part(
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
    num_angles: int = 36,
    resolution: int = 224,
    index_csv: Optional[Path] = None,
    domain_randomize: bool = False,
    hdri_dir: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Spawn a Blender process to render a single part+color combination.
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
            "--num-angles", str(num_angles),
            "--resolution", str(resolution),
        ]

        if index_csv:
            cmd.extend(["--index-csv", str(index_csv)])

        if domain_randomize:
            cmd.append("--domain-randomize")

        if hdri_dir:
            cmd.extend(["--hdri-dir", str(hdri_dir)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per render
        )

        if result.returncode == 0:
            return True, f"Rendered {part_num} color {color_name}"
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            return False, f"Failed: {part_num} {color_name} - {error_msg[:200]}"

    except subprocess.TimeoutExpired:
        return False, f"Timeout: {part_num} {color_name}"
    except Exception as e:
        return False, f"Error: {part_num} {color_name} - {str(e)[:200]}"


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch render LEGO parts for synthetic training data"
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
        help=(
            "Path to LDConfig.ldr for accurate LEGO color definitions "
            "(default: <ldraw-dir>/LDConfig.ldr). "
            "Downloaded automatically if missing. "
            "Provides correct sRGB→linear values for Blender materials."
        )
    )
    parser.add_argument(
        "--colors-csv",
        default="./ml/data/colors.csv",
        help="Path to Rebrickable colors.csv (fallback if LDConfig.ldr not available)"
    )
    parser.add_argument(
        "--parts-file",
        default=None,
        help="Text file with part numbers (one per line). If not set, uses top 500 parts."
    )
    parser.add_argument(
        "--output-dir",
        default="./ml/data/renders",
        help="Base output directory for renders"
    )
    parser.add_argument(
        "--index-csv",
        default=None,
        help="CSV to append image metadata to (default: output-dir/../index.csv)"
    )
    parser.add_argument(
        "--num-angles",
        type=int,
        default=36,
        help="Number of camera angles per elevation (default: 36)"
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
        default=4,
        help="Number of parallel Blender processes (default: 4)"
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=None,
        help="Limit number of parts to render (for testing)"
    )
    parser.add_argument(
        "--colors",
        type=int,
        nargs='+',
        default=None,
        help="Specific color IDs to render (if not set, uses all colors)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip parts that already have rendered output"
    )
    parser.add_argument(
        "--domain-randomize",
        action="store_true",
        help="Enable domain randomization: varied lighting, HDRI backgrounds, material jitter. "
             "Produces ~2x more images per part (6 elevations instead of 3). "
             "Strongly recommended for real-world generalisation."
    )
    parser.add_argument(
        "--hdri-dir",
        type=str,
        default=None,
        help="Path to directory of .hdr/.exr environment maps for random backgrounds. "
             "If not set, uses solid random colors. Free HDRIs: https://polyhaven.com/hdris"
    )

    args = parser.parse_args()

    # Setup paths
    blender_exe = args.blender
    ldraw_dir = Path(args.ldraw_dir).resolve()
    colors_csv = Path(args.colors_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    log_file = output_dir.parent / "failed_renders.log"

    # LDConfig.ldr path: explicit arg > <ldraw_dir>/LDConfig.ldr > auto-download
    if args.ldconfig:
        ldconfig_path = Path(args.ldconfig).resolve()
    else:
        ldconfig_path = ldraw_dir / "LDConfig.ldr"

    render_script = Path(__file__).parent / "render_parts.py"
    if not render_script.exists():
        print(f"ERROR: render_parts.py not found at {render_script}")
        sys.exit(1)

    if args.index_csv:
        index_csv = Path(args.index_csv).resolve()
    else:
        index_csv = output_dir.parent / "index.csv"

    logger = setup_logging(log_file)
    logger.info(f"Starting BrickScan batch render")
    logger.info(f"Blender: {blender_exe}")
    logger.info(f"LDraw dir: {ldraw_dir}")
    logger.info(f"LDConfig: {ldconfig_path}")
    logger.info(f"Colors CSV: {colors_csv}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Workers: {args.workers}")
    if args.domain_randomize:
        logger.info(f"Domain randomization: ENABLED")
        logger.info(f"HDRI dir: {args.hdri_dir or '(none — using solid colors)'}")

    # Load colors — prefer LDConfig.ldr for accurate LinearRGB values
    logger.info("Loading colors (LDConfig.ldr preferred for sRGB→linear accuracy)...")
    try:
        colors = load_colors(
            ldconfig_path=ldconfig_path if ldconfig_path.exists() or True else None,
            colors_csv=colors_csv if colors_csv.exists() else None,
        )
        logger.info(f"Loaded {len(colors)} colors with linear RGB values")
    except Exception as e:
        logger.error(f"Failed to load colors: {e}")
        sys.exit(1)

    # Filter to requested colors
    if args.colors:
        colors = {cid: colors[cid] for cid in args.colors if cid in colors}
        logger.info(f"Filtered to {len(colors)} requested colors")

    # Load part numbers
    logger.info("Loading part numbers...")
    if args.parts_file and Path(args.parts_file).exists():
        parts = load_parts_subset(Path(args.parts_file), ldraw_dir)
        logger.info(f"Loaded {len(parts)} parts from {args.parts_file}")
    else:
        # Default: scan LDraw library for all .dat files in parts/ subfolder
        parts_dir = ldraw_dir / "parts"
        if not parts_dir.exists():
            logger.error(f"LDraw parts directory not found: {parts_dir}")
            logger.error("Run setup_ldraw.sh first to download the library")
            sys.exit(1)

        parts = []
        for dat_file in sorted(parts_dir.glob("*.dat"))[:500]:  # Top 500
            parts.append(dat_file.stem)
        logger.info(f"Found {len(parts)} parts in LDraw library")

    if args.max_parts:
        parts = parts[:args.max_parts]
        logger.info(f"Limited to {len(parts)} parts")

    # Build render tasks
    tasks = []
    for part_num in parts:
        part_file = find_ldraw_part_file(part_num, ldraw_dir)
        if not part_file:
            logger.warning(f"Part file not found: {part_num}")
            continue

        part_output_dir = output_dir / part_num
        if args.skip_existing and part_output_dir.exists() and list(part_output_dir.glob("*.png")):
            logger.info(f"Skipping {part_num} (already rendered)")
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

    logger.info(f"Total tasks: {len(tasks)}")

    if not tasks:
        logger.warning("No tasks to render")
        sys.exit(0)

    # Render with progress bar
    success_count = 0
    failure_count = 0

    render_fn = lambda task: render_part(
        blender_exe,
        render_script,
        task[0], task[1], task[2], task[3], task[4], task[5], task[6], task[7],
        num_angles=args.num_angles,
        resolution=args.resolution,
        index_csv=index_csv,
        domain_randomize=args.domain_randomize,
        hdri_dir=args.hdri_dir,
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
    logger.info(f"Render complete!")
    logger.info(f"Success: {success_count}, Failures: {failure_count}")
    logger.info(f"Log: {log_file}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Index CSV: {index_csv}")

    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
