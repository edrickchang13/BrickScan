#!/usr/bin/env python3
"""
LDraw / LEGO color utilities for BrickScan render pipeline.

Provides:
  - parse_ldconfig(path)   → dict mapping LDraw color code → ColorEntry
  - srgb_to_linear(v)      → proper IEC 61966-2-1 sRGB → linear conversion
  - hex_to_linear(hex_str) → parse #RRGGBB hex and return (r_lin, g_lin, b_lin)
  - load_colors(...)       → unified loader: LDConfig.ldr first, fallback to CSV

LDConfig.ldr is the authoritative LEGO color standard maintained by the LDraw
organization.  Each colour line looks like:

  0 !COLOUR Black  CODE   0  VALUE #05131D  EDGE #595959

Optional fields: ALPHA <0-255>, LUMINANCE <0-255>, CHROME, METAL, RUBBER,
MATTE_METALLIC, PEARLESCENT, GLITTER <val>, SPECKLE <val>,
MATERIAL GLITTER VALUE #hex FRACTION <f> VFRACTION <f> SIZE <s>

We extract CODE, NAME, VALUE (sRGB hex) and any ALPHA / LUMINANCE modifiers.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# sRGB ↔ Linear conversion
# ---------------------------------------------------------------------------

def srgb_to_linear(v: float) -> float:
    """
    Convert a single channel from sRGB (0-1) to linear (0-1).
    Uses the exact IEC 61966-2-1 piecewise formula — not the 2.2 power
    approximation, which gives ~3% error in midtones.
    """
    if v <= 0.04045:
        return v / 12.92
    return ((v + 0.055) / 1.055) ** 2.4


def linear_to_srgb(v: float) -> float:
    """Inverse: linear → sRGB (for display / debug only)."""
    if v <= 0.0031308:
        return v * 12.92
    return 1.055 * (v ** (1.0 / 2.4)) - 0.055


def hex_to_srgb(hex_color: str) -> Tuple[float, float, float]:
    """Parse '#RRGGBB' or 'RRGGBB' → (r, g, b) in sRGB 0-1 space."""
    h = hex_color.lstrip('#')
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b)


def hex_to_linear(hex_color: str) -> Tuple[float, float, float]:
    """Parse '#RRGGBB' or 'RRGGBB' → (r, g, b) in **linear** 0-1 space."""
    r, g, b = hex_to_srgb(hex_color)
    return (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b))


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ColorEntry:
    code: int               # LDraw color code (0-based int)
    name: str               # Human-readable name, e.g. "Bright Red"
    hex_srgb: str           # '#RRGGBB' sRGB value from LDConfig
    rgb_linear: Tuple[float, float, float]  # Linear float tuple for Blender
    rgb_srgb:   Tuple[float, float, float]  # sRGB float tuple (0-1)
    alpha: float = 1.0      # 0-1; <1 for transparent parts
    luminance: float = 0.0  # 0-1; >0 for glow/emission parts
    finish: str = "standard"  # standard | chrome | metal | rubber |
                               # matte_metallic | pearlescent


# ---------------------------------------------------------------------------
# LDConfig.ldr parser
# ---------------------------------------------------------------------------

# Regex for the core !COLOUR declaration
# Example:
#   0 !COLOUR Black                              CODE   0   VALUE #05131D   EDGE #595959
_COLOUR_RE = re.compile(
    r'^0\s+!COLOUR\s+(?P<name>[^\t]+?)\s+CODE\s+(?P<code>\d+)\s+'
    r'VALUE\s+(?P<value>#[0-9A-Fa-f]{6})\s+EDGE\s+(?P<edge>#[0-9A-Fa-f]{6})',
    re.IGNORECASE,
)
_ALPHA_RE      = re.compile(r'ALPHA\s+(\d+)',      re.IGNORECASE)
_LUMINANCE_RE  = re.compile(r'LUMINANCE\s+(\d+)',  re.IGNORECASE)
_CHROME_RE     = re.compile(r'\bCHROME\b',         re.IGNORECASE)
_METAL_RE      = re.compile(r'\bMETAL\b',          re.IGNORECASE)
_RUBBER_RE     = re.compile(r'\bRUBBER\b',         re.IGNORECASE)
_MATTE_RE      = re.compile(r'\bMATTE_METALLIC\b', re.IGNORECASE)
_PEARL_RE      = re.compile(r'\bPEARLESCENT\b',    re.IGNORECASE)


def parse_ldconfig(ldconfig_path: Path) -> Dict[int, ColorEntry]:
    """
    Parse an LDConfig.ldr file and return a dict of {code: ColorEntry}.

    Downloads from ldraw.org if the file doesn't exist (requires requests).
    """
    ldconfig_path = Path(ldconfig_path)
    if not ldconfig_path.exists():
        _download_ldconfig(ldconfig_path)

    colors: Dict[int, ColorEntry] = {}

    with open(ldconfig_path, 'r', encoding='utf-8', errors='replace') as fh:
        for raw_line in fh:
            line = raw_line.strip()
            m = _COLOUR_RE.match(line)
            if not m:
                continue

            name  = m.group('name').strip()
            code  = int(m.group('code'))
            value = m.group('value').upper()

            # Optional modifiers
            alpha_m = _ALPHA_RE.search(line)
            alpha = int(alpha_m.group(1)) / 255.0 if alpha_m else 1.0

            lum_m = _LUMINANCE_RE.search(line)
            luminance = int(lum_m.group(1)) / 255.0 if lum_m else 0.0

            if _CHROME_RE.search(line):
                finish = 'chrome'
            elif _METAL_RE.search(line):
                finish = 'metal'
            elif _MATTE_RE.search(line):
                finish = 'matte_metallic'
            elif _PEARL_RE.search(line):
                finish = 'pearlescent'
            elif _RUBBER_RE.search(line):
                finish = 'rubber'
            else:
                finish = 'standard'

            srgb   = hex_to_srgb(value)
            linear = hex_to_linear(value)

            colors[code] = ColorEntry(
                code=code,
                name=name,
                hex_srgb=value,
                rgb_linear=linear,
                rgb_srgb=srgb,
                alpha=alpha,
                luminance=luminance,
                finish=finish,
            )

    return colors


def _download_ldconfig(dest: Path) -> None:
    """Download LDConfig.ldr from the official LDraw server."""
    url = 'https://www.ldraw.org/library/official/LDConfig.ldr'
    try:
        import urllib.request
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[ldraw_colors] Downloading LDConfig.ldr from {url} …")
        urllib.request.urlretrieve(url, dest)
        print(f"[ldraw_colors] Saved to {dest}")
    except Exception as exc:
        raise RuntimeError(
            f"Could not download LDConfig.ldr: {exc}\n"
            f"Download manually from {url} and place at {dest}"
        ) from exc


# ---------------------------------------------------------------------------
# Rebrickable colors.csv loader (fallback)
# ---------------------------------------------------------------------------

def load_rebrickable_colors(csv_path: Path) -> Dict[int, ColorEntry]:
    """
    Load Rebrickable colors.csv.  Column layout:
      id, name, rgb, is_trans

    'rgb' is a 6-char sRGB hex string WITHOUT the '#' prefix.
    Returns {rebrickable_id: ColorEntry}.
    """
    colors: Dict[int, ColorEntry] = {}
    with open(csv_path, 'r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                code = int(row['id'])
                name = row['name']
                hex_val = '#' + row['rgb'].strip().upper()
                srgb   = hex_to_srgb(hex_val)
                linear = hex_to_linear(hex_val)
                is_trans = row.get('is_trans', 'f').lower() in ('t', 'true', '1', 'yes')
                colors[code] = ColorEntry(
                    code=code,
                    name=name,
                    hex_srgb=hex_val,
                    rgb_linear=linear,
                    rgb_srgb=srgb,
                    alpha=0.5 if is_trans else 1.0,
                )
            except (KeyError, ValueError):
                continue
    return colors


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load_colors(
    ldconfig_path: Optional[Path] = None,
    rebrickable_csv: Optional[Path] = None,
) -> Dict[int, ColorEntry]:
    """
    Load colors, preferring LDConfig.ldr for accuracy.

    Priority:
      1. LDConfig.ldr  (most accurate, official LDraw/LEGO standard)
      2. Rebrickable colors.csv  (fallback if LDConfig not available)
      3. Hardcoded minimal set (last resort, 15 most common colors)

    Returns {color_id: ColorEntry} with linear RGB values ready for Blender.
    """
    if ldconfig_path:
        p = Path(ldconfig_path)
        if p.exists() or True:  # allow auto-download
            try:
                colors = parse_ldconfig(p)
                print(f"[ldraw_colors] Loaded {len(colors)} colors from LDConfig.ldr")
                return colors
            except Exception as e:
                print(f"[ldraw_colors] Warning: LDConfig.ldr failed ({e}), trying fallback")

    if rebrickable_csv and Path(rebrickable_csv).exists():
        colors = load_rebrickable_colors(Path(rebrickable_csv))
        print(f"[ldraw_colors] Loaded {len(colors)} colors from Rebrickable CSV")
        return colors

    # Last resort: hardcoded set with proper linear values
    print("[ldraw_colors] Warning: using hardcoded color fallback (38 official LDraw colors)")
    return _minimal_fallback_colors()


def _minimal_fallback_colors() -> Dict[int, ColorEntry]:
    """
    Official LDraw solid-color palette (38 entries), pre-converted to linear.

    Expanded from 15 → 38 to cover the full set of common LEGO colors seen
    in real scans. Hex values come from LDConfig.ldr in the LDraw spec —
    same source `parse_ldconfig()` uses when the file is reachable.
    When it's NOT reachable (offline / sandbox / first-boot before download)
    this fallback provides enough coverage for Blender to pick a realistic
    random color per render.
    """
    # (code, name, hex_srgb) — ordered by LDraw code
    _TABLE = [
        (0,   "Black",                  "#05131D"),
        (1,   "Blue",                   "#0055BF"),
        (2,   "Green",                  "#257A3E"),
        (3,   "Dark Turquoise",         "#00838F"),
        (4,   "Red",                    "#C91A09"),
        (5,   "Dark Pink",              "#C870A0"),
        (6,   "Brown",                  "#583927"),
        (7,   "Light Gray",             "#9BA19D"),
        (8,   "Dark Gray",              "#6D6E5C"),
        (9,   "Light Blue",             "#B4D2E3"),
        (10,  "Bright Green",           "#4B9F4A"),
        (11,  "Light Turquoise",        "#55A5AF"),
        (12,  "Salmon",                 "#F2705E"),
        (13,  "Pink",                   "#FC97AC"),
        (14,  "Yellow",                 "#F2CD37"),
        (15,  "White",                  "#FFFFFF"),
        (17,  "Light Green",            "#C2DAB8"),
        (18,  "Light Yellow",           "#FBE696"),
        (19,  "Tan",                    "#E4CD9E"),
        (20,  "Light Violet",           "#C9CAE2"),
        (22,  "Purple",                 "#81007B"),
        (23,  "Dark Blue-Violet",       "#2032B0"),
        (25,  "Orange",                 "#FE8A18"),
        (26,  "Magenta",                "#923978"),
        (27,  "Lime",                   "#BBE90B"),
        (28,  "Dark Tan",               "#958A73"),
        (29,  "Bright Pink",            "#E4ADC8"),
        (30,  "Medium Lavender",        "#AC78BA"),
        (31,  "Lavender",               "#E1D5ED"),
        (69,  "Dark Purple",            "#3F3691"),
        (70,  "Reddish Brown",          "#582A12"),
        (71,  "Light Bluish Gray",      "#A0A5A9"),
        (72,  "Dark Bluish Gray",       "#6C6E68"),
        (73,  "Medium Blue",            "#5A93DB"),
        (74,  "Medium Green",           "#73DCA1"),
        (77,  "Light Pink",             "#FECCCF"),
        (78,  "Light Flesh",            "#F6D7B3"),
        (84,  "Medium Dark Flesh",      "#CC702A"),
    ]
    colors = {}
    for code, name, hex_val in _TABLE:
        srgb   = hex_to_srgb(hex_val)
        linear = hex_to_linear(hex_val)
        colors[code] = ColorEntry(
            code=code, name=name, hex_srgb=hex_val,
            rgb_linear=linear, rgb_srgb=srgb,
        )
    return colors


# ---------------------------------------------------------------------------
# CLI helper — dump the parsed color table
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Dump LDraw color table")
    ap.add_argument('--ldconfig', default='./ml/data/ldraw/LDConfig.ldr')
    ap.add_argument('--csv',      default=None)
    ap.add_argument('--code',     type=int, default=None, help="Show one color by code")
    args = ap.parse_args()

    colors = load_colors(
        ldconfig_path=args.ldconfig if args.ldconfig else None,
        rebrickable_csv=args.csv if args.csv else None,
    )

    if args.code is not None:
        c = colors.get(args.code)
        if c:
            print(f"Code {c.code:4d}  {c.name:<30s}  sRGB {c.hex_srgb}  "
                  f"linear ({c.rgb_linear[0]:.4f}, {c.rgb_linear[1]:.4f}, {c.rgb_linear[2]:.4f})  "
                  f"finish={c.finish}  alpha={c.alpha:.2f}")
        else:
            print(f"Color code {args.code} not found")
            sys.exit(1)
    else:
        print(f"{'Code':>6}  {'Name':<32}  {'sRGB':>8}  {'R_lin':>7}  {'G_lin':>7}  {'B_lin':>7}  Finish")
        print("-" * 90)
        for code in sorted(colors):
            c = colors[code]
            print(f"{c.code:6d}  {c.name:<32s}  {c.hex_srgb}  "
                  f"{c.rgb_linear[0]:7.4f}  {c.rgb_linear[1]:7.4f}  {c.rgb_linear[2]:7.4f}  {c.finish}")
