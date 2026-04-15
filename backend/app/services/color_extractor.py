"""
Extract the dominant colour of a LEGO-part scan for cascade re-ranking.

Used by hybrid_recognition when SCAN_COLOR_RERANK=true: after merging
predictions from all sources, we compare the scan's dominant colour
against each candidate's `color_hex` and downweight (not drop)
candidates whose colours disagree. This catches Gemini / Brickognize
disagreements on e.g. "red brick 2x4" vs "dark-red brick 2x4".

Uses PIL + a simple histogram-binning approach (no sklearn / kmeans
dependency — a 10x10x10 RGB histogram + argmax is plenty for this
coarse-grained re-rank).
"""

from __future__ import annotations

import io
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# 10 bins per channel = 1000 total bins — coarse enough to be robust to
# shadow / anti-aliasing, fine enough to distinguish LEGO colour families.
_BINS_PER_CHANNEL = 10


def extract_dominant_color(image_bytes: bytes) -> Optional[Tuple[int, int, int]]:
    """
    Return the dominant (R, G, B) colour in an image, or None on failure.

    Only the center 60% of the image is sampled — the background of a
    scan is usually uniform, and the brick is almost always near the
    middle of the frame. This drops most of the background contribution
    from the histogram.

    No-op-safe: returns None if PIL isn't importable, image can't be
    decoded, or the image is empty.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        if w < 20 or h < 20:
            return None

        # Center crop — 60% of width/height — typical brick location
        cx0, cy0 = int(w * 0.2), int(h * 0.2)
        cx1, cy1 = int(w * 0.8), int(h * 0.8)
        crop = img.crop((cx0, cy0, cx1, cy1))

        # Downsample before histogramming so we're not iterating 200k pixels
        crop.thumbnail((128, 128))

        # Build the histogram
        counts: dict = {}
        bin_w = 256 // _BINS_PER_CHANNEL
        for pixel in crop.getdata():
            r, g, b = pixel[:3]
            key = (r // bin_w, g // bin_w, b // bin_w)
            counts[key] = counts.get(key, 0) + 1

        if not counts:
            return None

        # Pick the mode bin; reconstruct an RGB from the bin center
        best_bin = max(counts.items(), key=lambda kv: kv[1])[0]
        br, bg, bb = best_bin
        center = bin_w // 2
        return (br * bin_w + center, bg * bin_w + center, bb * bin_w + center)

    except Exception as e:
        logger.debug("dominant color extraction failed: %s", e)
        return None


def _hex_to_rgb(hex_str: str) -> Optional[Tuple[int, int, int]]:
    """Parse '#RRGGBB' or 'RRGGBB' → (r, g, b). None on bad input."""
    if not hex_str:
        return None
    s = hex_str.lstrip("#")
    if len(s) != 6:
        return None
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return None


def color_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    """Euclidean distance in RGB space. 0 = identical, ~441 = opposite corners of cube."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def rerank_predictions_by_color(
    predictions: list,
    scan_rgb: Tuple[int, int, int],
    mismatch_threshold: float = 100.0,
    mismatch_penalty: float = 0.7,
) -> list:
    """
    Downweight candidates whose `color_hex` disagrees with the scan's
    dominant colour. Candidates without a colour_hex are left alone.

    Args:
        predictions:          list of prediction dicts (order preserved)
        scan_rgb:             (r, g, b) from extract_dominant_color()
        mismatch_threshold:   RGB Euclidean distance above which a colour
                              counts as "mismatched". 100 is empirical —
                              distinguishes red vs dark-red but not red vs orange.
        mismatch_penalty:     multiplier applied to confidence when
                              mismatched (0.7 = 30% penalty, never drop)

    Returns a new list; doesn't mutate the input.
    """
    out = []
    for p in predictions:
        hex_str = p.get("color_hex")
        cand_rgb = _hex_to_rgb(hex_str) if hex_str else None
        if cand_rgb is None:
            out.append(p)
            continue
        dist = color_distance(scan_rgb, cand_rgb)
        if dist > mismatch_threshold:
            new = {**p}
            new["confidence"] = float(new.get("confidence", 0.0)) * mismatch_penalty
            new["_color_mismatch_dist"] = round(dist, 1)  # observability
            out.append(new)
        else:
            out.append(p)
    # Re-sort by new confidence (stable)
    out.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    return out
