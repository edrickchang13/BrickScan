#!/usr/bin/env python3
"""
generate_multipiece_scenes.py — Synthetic multi-piece scene generator for YOLO training.

Takes individual part renders (from batch_render.py) and composites 2–8 random
parts onto randomised backgrounds, outputting YOLO-format bounding box labels
WITH per-part class IDs so the model can both detect and classify simultaneously.

Key improvements over v1:
  - Per-class YOLO labels (class = part index in parts.txt, not always 0)
  - Extreme lighting augmentations: very dark, harsh spotlight, coloured cast
  - Shadow casting between parts (simulates piled/scattered bricks)
  - Realistic surface backgrounds: wood grain, carpet texture, concrete, hand-held
  - Motion blur on individual parts (simulate pick-up motion)
  - JPEG compression artifact simulation (real phone camera output)
  - Partial occlusion: bricks can hang off the frame edge

Output structure:
  output_dir/
    images/train/       scene_00001.jpg ...
    images/val/         scene_00001.jpg ...
    labels/train/       scene_00001.txt ...  (YOLO: class cx cy w h, normalised)
    labels/val/         ...
    parts.txt           class_id → part_num mapping
    lego.yaml           YOLO dataset config (nc = number of unique parts)

Usage:
  python generate_multipiece_scenes.py \\
    --renders-dir  ./ml/data/renders \\
    --output-dir   ./ml/data/yolo_dataset \\
    --num-train    10000 \\
    --num-val      2000
"""

import os
import sys
import io
import math
import json
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import (Image, ImageFilter, ImageEnhance, ImageDraw,
                 ImageChops, ImageOps)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ─── Argument parsing ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--renders-dir', required=True,
                        help='Root dir of part renders (from batch_render.py)')
    parser.add_argument('--output-dir', required=True,
                        help='Output dir for YOLO dataset')
    parser.add_argument('--num-train', type=int, default=10000)
    parser.add_argument('--num-val', type=int, default=2000)
    parser.add_argument('--image-size', type=int, default=640,
                        help='Output scene size in pixels (default 640)')
    parser.add_argument('--min-parts', type=int, default=1,
                        help='Min parts per scene (default 1 — single-part also trained)')
    parser.add_argument('--max-parts', type=int, default=8)
    parser.add_argument('--min-scale', type=float, default=0.10,
                        help='Min part size as fraction of scene width')
    parser.add_argument('--max-scale', type=float, default=0.40,
                        help='Max part size as fraction of scene width')
    parser.add_argument('--allow-partial', action='store_true', default=True,
                        help='Allow parts to partially hang off the frame edge')
    parser.add_argument('--jpeg-quality-min', type=int, default=60,
                        help='Min JPEG quality (simulates phone compression)')
    parser.add_argument('--jpeg-quality-max', type=int, default=95)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel workers (>1 requires joblib)')
    return parser.parse_args()


# ─── Part index: discover renders, assign class IDs ─────────────────────────

def build_part_index(renders_dir: Path):
    """
    Walk renders_dir and return:
      part_to_class: {part_num: class_id}
      class_to_part: {class_id: part_num}
      images_by_part: {part_num: [Path, ...]}

    Expected directory layout (from batch_render.py):
      renders_dir/<part_num>/<color_name>/<angle>.png
    OR flat:
      renders_dir/<part_num>_<color>_<angle>.png
    """
    images_by_part = defaultdict(list)

    for path in renders_dir.rglob('*.png'):
        # Try to derive part_num from directory structure
        rel = path.relative_to(renders_dir)
        parts_of_path = rel.parts
        if len(parts_of_path) >= 2:
            part_num = parts_of_path[0]   # first subdir is part number
        else:
            # flat layout: try filename prefix before first underscore
            stem = path.stem
            part_num = stem.split('_')[0] if '_' in stem else stem
        images_by_part[part_num].append(path)

    # Also accept .jpg
    for path in renders_dir.rglob('*.jpg'):
        rel = path.relative_to(renders_dir)
        parts_of_path = rel.parts
        if len(parts_of_path) >= 2:
            part_num = parts_of_path[0]
        else:
            stem = path.stem
            part_num = stem.split('_')[0] if '_' in stem else stem
        images_by_part[part_num].append(path)

    if not images_by_part:
        logger.error(f"No renders found in {renders_dir}")
        sys.exit(1)

    sorted_parts = sorted(images_by_part.keys())
    part_to_class = {p: i for i, p in enumerate(sorted_parts)}
    class_to_part = {i: p for i, p in enumerate(sorted_parts)}

    logger.info(f"Found {len(sorted_parts)} unique parts, "
                f"{sum(len(v) for v in images_by_part.values())} total renders")

    return part_to_class, class_to_part, images_by_part


# ─── Background generation ───────────────────────────────────────────────────

def make_plain_bg(size, rng):
    """Uniform colour, slightly noisy — common tabletop scan."""
    import colorsys
    hue = rng.uniform(0, 1)
    sat = rng.uniform(0.0, 0.25)
    val = rng.uniform(0.25, 0.92)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    base = (int(r * 255), int(g * 255), int(b * 255))
    img = Image.new('RGB', (size, size), base)
    noise = np.random.randint(-20, 20, (size, size, 3), dtype=np.int16)
    arr = np.clip(np.array(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_wood_bg(size, rng):
    """Procedural wood grain — most common real-world scan surface."""
    arr = np.zeros((size, size, 3), dtype=np.float32)
    base_r = rng.uniform(0.35, 0.65)
    base_g = rng.uniform(0.20, 0.45)
    base_b = rng.uniform(0.05, 0.20)

    # Grain direction angle
    angle = rng.uniform(-15, 15) * math.pi / 180
    freq = rng.uniform(8, 25)
    for y in range(size):
        for_x = np.arange(size)
        grain = y * math.cos(angle) + for_x * math.sin(angle)
        wave = np.sin(grain * freq / size * 2 * math.pi + rng.uniform(0, 6.28))
        wave2 = np.sin(grain * freq * 2.3 / size * 2 * math.pi + rng.uniform(0, 6.28)) * 0.4
        intensity = (wave + wave2) * 0.5 + 0.5  # 0-1
        t = intensity * 0.35 + rng.uniform(-0.02, 0.02)
        arr[y, :, 0] = np.clip(base_r + t * 0.3, 0, 1)
        arr[y, :, 1] = np.clip(base_g + t * 0.2, 0, 1)
        arr[y, :, 2] = np.clip(base_b + t * 0.05, 0, 1)

    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_carpet_bg(size, rng):
    """High-frequency noise carpet texture."""
    base = np.array([
        rng.randint(80, 180),
        rng.randint(80, 180),
        rng.randint(80, 180),
    ])
    noise = np.random.randint(-40, 40, (size, size, 3), dtype=np.int16)
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    # Slight directional blur to simulate fibres
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return img


def make_concrete_bg(size, rng):
    """Grey concrete / floor texture."""
    base_v = rng.randint(120, 200)
    arr = np.full((size, size, 3), base_v, dtype=np.int16)
    # Large-scale variation
    coarse = np.random.randint(-30, 30, (size // 8, size // 8, 3), dtype=np.int16)
    coarse_up = np.array(Image.fromarray(
        np.clip(coarse + base_v, 0, 255).astype(np.uint8)
    ).resize((size, size), Image.BILINEAR), dtype=np.int16) - base_v
    arr = np.clip(arr + coarse_up + np.random.randint(-10, 10, (size, size, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_gradient_bg(size, rng):
    """Simple gradient — common in controlled photo setups."""
    import colorsys
    hue = rng.uniform(0, 1)
    c1 = np.array(colorsys.hsv_to_rgb(hue, rng.uniform(0, 0.3), rng.uniform(0.5, 0.95))) * 255
    c2 = np.array(colorsys.hsv_to_rgb(hue, rng.uniform(0, 0.3), rng.uniform(0.4, 0.85))) * 255
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        t = i / size
        arr[i, :] = ((1 - t) * c1 + t * c2).astype(np.uint8)
    return Image.fromarray(arr)


def make_background(size, rng):
    weights = [0.30, 0.25, 0.15, 0.15, 0.15]  # plain, wood, carpet, concrete, gradient
    choice = rng.choices(['plain', 'wood', 'carpet', 'concrete', 'gradient'],
                         weights=weights)[0]
    if choice == 'plain':
        return make_plain_bg(size, rng)
    elif choice == 'wood':
        return make_wood_bg(size, rng)
    elif choice == 'carpet':
        return make_carpet_bg(size, rng)
    elif choice == 'concrete':
        return make_concrete_bg(size, rng)
    else:
        return make_gradient_bg(size, rng)


# ─── Lighting overlays ───────────────────────────────────────────────────────

def apply_lighting(img: Image.Image, rng) -> Image.Image:
    """
    Apply a random lighting condition to the composited scene.
    Covers: normal, dim room, harsh shadow, spotlight, coloured tint,
    overexposed (near-window), and mixed (two light sources).
    """
    mode = rng.choices(
        ['normal', 'dim', 'harsh_shadow', 'spotlight', 'colour_cast',
         'overexposed', 'underexposed', 'mixed'],
        weights=[0.25, 0.12, 0.12, 0.10, 0.10, 0.08, 0.08, 0.15]
    )[0]

    w, h = img.size

    if mode == 'normal':
        brightness = rng.uniform(0.85, 1.15)
        img = ImageEnhance.Brightness(img).enhance(brightness)

    elif mode == 'dim':
        # Low ambient — like scanning in a dim room
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.3, 0.55))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.1))

    elif mode == 'harsh_shadow':
        # One half of the image is much darker (directional shadow)
        arr = np.array(img, dtype=np.float32)
        mask = np.zeros((h, w, 3), dtype=np.float32)
        split = rng.randint(int(w * 0.2), int(w * 0.8))
        strength = rng.uniform(0.3, 0.6)
        if rng.random() < 0.5:
            mask[:, split:] = strength
        else:
            mask[:split, :] = strength
        arr = np.clip(arr * (1.0 - mask), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    elif mode == 'spotlight':
        # Bright circular area, dark surrounds — like a single desk lamp
        arr = np.array(img, dtype=np.float32)
        cx = rng.randint(int(w * 0.2), int(w * 0.8))
        cy = rng.randint(int(h * 0.2), int(h * 0.8))
        radius = rng.randint(int(min(w, h) * 0.2), int(min(w, h) * 0.6))
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        falloff = np.clip(1.0 - (dist / radius) ** 1.5, 0.15, 1.0)
        arr = np.clip(arr * falloff[:, :, None], 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    elif mode == 'colour_cast':
        # Warm (tungsten) or cool (fluorescent / window) colour cast
        arr = np.array(img, dtype=np.float32)
        cast_type = rng.choice(['warm', 'cool', 'green'])
        strength = rng.uniform(0.05, 0.18)
        if cast_type == 'warm':
            arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 + strength), 0, 255)  # +R
            arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 - strength), 0, 255)  # -B
        elif cast_type == 'cool':
            arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 + strength), 0, 255)  # +B
            arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 - strength * 0.5), 0, 255)
        else:
            arr[:, :, 1] = np.clip(arr[:, :, 1] * (1 + strength), 0, 255)  # +G
        img = Image.fromarray(arr.astype(np.uint8))

    elif mode == 'overexposed':
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(1.4, 2.0))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.6, 0.9))

    elif mode == 'underexposed':
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.15, 0.35))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))

    elif mode == 'mixed':
        # Two passes with different adjustments per half
        arr = np.array(img, dtype=np.float32)
        split = rng.randint(int(w * 0.3), int(w * 0.7))
        arr[:, :split] = np.clip(arr[:, :split] * rng.uniform(0.5, 1.0), 0, 255)
        arr[:, split:] = np.clip(arr[:, split:] * rng.uniform(0.9, 1.5), 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    return img


# ─── Part augmentation ───────────────────────────────────────────────────────

def augment_part(img: Image.Image, rng) -> Image.Image:
    """
    Apply per-part augmentations before compositing.
    Simulates: lighting variation, camera focus, motion, viewing angle.
    """
    # Brightness / contrast jitter
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.5, 1.5))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.4))

    # Saturation jitter (faded vs vivid)
    if img.mode == 'RGBA':
        rgb = img.convert('RGB')
        rgb = ImageEnhance.Color(rgb).enhance(rng.uniform(0.5, 1.4))
        img = Image.merge('RGBA', [*rgb.split(), img.split()[3]])
    else:
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.5, 1.4))

    # Defocus blur (1 in 4 parts slightly out of focus)
    if rng.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8, 2.5)))

    # Motion blur (1 in 5 — simulate mid-placement movement)
    if rng.random() < 0.20:
        angle = rng.uniform(0, 180)
        length = rng.randint(3, 8)
        img = _motion_blur(img, angle, length)

    return img


def _motion_blur(img: Image.Image, angle_deg: float, length: int) -> Image.Image:
    """Apply directional motion blur using numpy convolution (avoids PIL kernel size limits)."""
    # Build motion kernel in numpy, then convolve each channel manually
    k = max(3, length | 1)  # force odd, minimum 3
    kernel = np.zeros((k, k), dtype=np.float32)
    mid = k // 2
    angle_rad = math.radians(angle_deg)
    for i in range(k):
        offset = i - mid
        x = mid + int(round(offset * math.cos(angle_rad)))
        y = mid + int(round(offset * math.sin(angle_rad)))
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1.0
    s = kernel.sum()
    if s > 0:
        kernel /= s

    from scipy.ndimage import convolve
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 3:
        out = np.stack([
            np.clip(convolve(arr[:, :, c], kernel), 0, 255)
            for c in range(arr.shape[2])
        ], axis=2).astype(np.uint8)
    else:
        out = np.clip(convolve(arr, kernel), 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode=img.mode)


# ─── Shadow casting ──────────────────────────────────────────────────────────

def cast_shadow(scene: Image.Image, part_img: Image.Image,
                x1: int, y1: int, rng) -> Image.Image:
    """
    Composite a soft drop-shadow under the part before pasting the part itself.
    Makes stacked/adjacent bricks look grounded and realistic.
    """
    if part_img.mode != 'RGBA':
        return scene

    alpha = part_img.split()[3]
    shadow_offset_x = rng.randint(2, 8)
    shadow_offset_y = rng.randint(3, 10)
    shadow_blur = rng.uniform(3, 8)
    shadow_opacity = rng.uniform(0.25, 0.55)

    # Create shadow layer same size as scene
    shadow_layer = Image.new('RGBA', scene.size, (0, 0, 0, 0))
    # Paste the alpha mask offset into the shadow layer
    dark = Image.new('RGBA', part_img.size, (0, 0, 0, 180))
    dark.putalpha(alpha)
    shadow_layer.paste(dark, (x1 + shadow_offset_x, y1 + shadow_offset_y),
                       mask=dark.split()[3])
    # Blur the shadow
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
    # Reduce opacity
    r, g, b, a = shadow_layer.split()
    a = a.point(lambda p: int(p * shadow_opacity))
    shadow_layer.putalpha(a)

    scene_rgba = scene.convert('RGBA')
    scene_rgba = Image.alpha_composite(scene_rgba, shadow_layer)
    return scene_rgba.convert('RGB')


# ─── JPEG compression simulation ─────────────────────────────────────────────

def simulate_jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Round-trip through JPEG to simulate phone camera compression artifacts."""
    buf = io.BytesIO()
    img.convert('RGB').save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


# ─── Part placement ──────────────────────────────────────────────────────────

def place_parts(scene_size, part_paths_with_classes, num_parts,
                min_scale, max_scale, allow_partial, rng):
    """
    Returns list of (pil_rgba, class_id, part_num, x1, y1, x2, y2).
    x1/y1 may be negative if allow_partial=True (brick overhangs frame edge).
    The YOLO bounding box is clipped to the scene bounds.
    """
    placed = []
    max_attempts = num_parts * 30

    for _ in range(max_attempts):
        if len(placed) >= num_parts:
            break

        path, class_id, part_num = rng.choice(part_paths_with_classes)
        try:
            img = Image.open(path).convert('RGBA')
        except Exception:
            continue

        scale = rng.uniform(min_scale, max_scale)
        target_w = max(12, int(scene_size * scale))
        aspect = img.height / max(img.width, 1)
        target_h = max(12, int(target_w * aspect))
        img = img.resize((target_w, target_h), Image.LANCZOS)

        # Random rotation (full 360° — bricks can be any orientation)
        angle = rng.uniform(0, 360)
        img = img.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

        # Position: allow partial overhang if enabled
        if allow_partial:
            overhang = int(min(img.width, img.height) * 0.4)
            x1 = rng.randint(-overhang, scene_size - img.width + overhang)
            y1 = rng.randint(-overhang, scene_size - img.height + overhang)
        else:
            if img.width >= scene_size or img.height >= scene_size:
                continue
            x1 = rng.randint(0, scene_size - img.width)
            y1 = rng.randint(0, scene_size - img.height)

        x2 = x1 + img.width
        y2 = y1 + img.height

        # Clip visible area
        vis_x1 = max(0, x1)
        vis_y1 = max(0, y1)
        vis_x2 = min(scene_size, x2)
        vis_y2 = min(scene_size, y2)
        vis_w = vis_x2 - vis_x1
        vis_h = vis_y2 - vis_y1

        # Skip if basically invisible
        if vis_w < 8 or vis_h < 8:
            continue
        visible_fraction = (vis_w * vis_h) / max((x2 - x1) * (y2 - y1), 1)
        if visible_fraction < 0.15:
            continue

        # Reject only if VERY heavily overlapping with existing (allow moderate overlap —
        # stacked bricks in real life DO overlap significantly)
        overlap_ok = True
        for (_, _, _, ox1, oy1, ox2, oy2) in placed:
            inter_x = max(0, min(x2, ox2) - max(x1, ox1))
            inter_y = max(0, min(y2, oy2) - max(y1, oy1))
            inter = inter_x * inter_y
            area = (x2 - x1) * (y2 - y1)
            if area > 0 and inter / area > 0.70:  # >70% covered = skip
                overlap_ok = False
                break

        if overlap_ok:
            placed.append((img, class_id, part_num, x1, y1, x2, y2))

    return placed


# ─── Scene compositing ───────────────────────────────────────────────────────

def composite_scene(scene_size, part_paths_with_classes, min_parts, max_parts,
                    min_scale, max_scale, allow_partial, jpeg_quality, rng):
    """
    Build one synthetic multi-brick scene.
    Returns (scene_rgb, yolo_labels).
    yolo_labels = [[class_id, cx, cy, w, h], ...] all normalised 0-1, clipped to scene.
    """
    bg = make_background(scene_size, rng)
    scene = bg.convert('RGB')

    num_parts = rng.randint(min_parts, max_parts)
    placed = place_parts(scene_size, part_paths_with_classes, num_parts,
                         min_scale, max_scale, allow_partial, rng)

    yolo_labels = []

    for (part_img, class_id, part_num, x1, y1, x2, y2) in placed:
        part_img = augment_part(part_img, rng)

        # Cast drop shadow first (before pasting the part)
        scene = cast_shadow(scene, part_img, x1, y1, rng)

        # Paste part (with alpha)
        if part_img.mode == 'RGBA':
            scene_rgba = scene.convert('RGBA')
            scene_rgba.paste(part_img, (x1, y1), mask=part_img.split()[3])
            scene = scene_rgba.convert('RGB')
        else:
            scene.paste(part_img, (x1, y1))

        # YOLO bounding box: clip to scene bounds, normalise
        vis_x1 = max(0, x1)
        vis_y1 = max(0, y1)
        vis_x2 = min(scene_size, x2)
        vis_y2 = min(scene_size, y2)

        cx = (vis_x1 + vis_x2) / 2 / scene_size
        cy = (vis_y1 + vis_y2) / 2 / scene_size
        w  = (vis_x2 - vis_x1) / scene_size
        h  = (vis_y2 - vis_y1) / scene_size
        yolo_labels.append([class_id, cx, cy, w, h])

    # Scene-level lighting
    scene = apply_lighting(scene, rng)

    # Optional scene-level blur (camera shake / defocus)
    if rng.random() < 0.15:
        scene = scene.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))

    # JPEG compression simulation (always — phones always compress)
    quality = rng.randint(jpeg_quality[0], jpeg_quality[1])
    scene = simulate_jpeg(scene, quality)

    return scene, yolo_labels


# ─── Dataset generation ─────────────────────────────────────────────────────

def generate_split(part_paths_with_classes, output_dir, split, count,
                   scene_size, min_parts, max_parts, min_scale, max_scale,
                   allow_partial, jpeg_quality, rng):
    img_dir = output_dir / 'images' / split
    lbl_dir = output_dir / 'labels' / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {count} {split} scenes…")
    for i in range(count):
        scene, labels = composite_scene(
            scene_size, part_paths_with_classes,
            min_parts, max_parts, min_scale, max_scale,
            allow_partial, jpeg_quality, rng,
        )
        idx = f'{i:06d}'
        scene.save(img_dir / f'scene_{idx}.jpg', quality=92)

        with open(lbl_dir / f'scene_{idx}.txt', 'w') as f:
            for lbl in labels:
                f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

        if (i + 1) % 500 == 0:
            logger.info(f"  {split}: {i + 1}/{count}")

    logger.info(f"  {split} done: {count} scenes → {img_dir}")


def write_yaml(output_dir: Path, class_to_part: dict):
    """Write YOLO dataset config with per-class part numbers."""
    nc = len(class_to_part)
    names_lines = '\n'.join(
        f'  {i}: "{part}"' for i, part in sorted(class_to_part.items())
    )
    yaml_content = f"""# BrickScan LEGO multi-piece detection dataset
# Auto-generated by generate_multipiece_scenes.py
# {nc} classes — one per unique part number

path: {output_dir.resolve()}
train: images/train
val:   images/val

nc: {nc}
names:
{names_lines}
"""
    (output_dir / 'lego.yaml').write_text(yaml_content)
    logger.info(f"Wrote {output_dir / 'lego.yaml'} ({nc} classes)")


def write_parts_txt(output_dir: Path, class_to_part: dict):
    """Write a class_id → part_num mapping for easy reference."""
    with open(output_dir / 'parts.txt', 'w') as f:
        f.write('# class_id\tpart_num\n')
        for class_id in sorted(class_to_part.keys()):
            f.write(f"{class_id}\t{class_to_part[class_id]}\n")
    logger.info(f"Wrote {output_dir / 'parts.txt'}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    renders_dir = Path(args.renders_dir).expanduser().resolve()
    output_dir  = Path(args.output_dir).expanduser().resolve()

    if not renders_dir.exists():
        logger.error(f"Renders dir not found: {renders_dir}")
        sys.exit(1)

    # Build part index
    part_to_class, class_to_part, images_by_part = build_part_index(renders_dir)

    # Flat list of (path, class_id, part_num) for sampling
    part_paths_with_classes = []
    for part_num, paths in images_by_part.items():
        class_id = part_to_class[part_num]
        for p in paths:
            part_paths_with_classes.append((p, class_id, part_num))

    logger.info(f"Total render images available: {len(part_paths_with_classes)}")
    logger.info(f"Scene size: {args.image_size}×{args.image_size}")
    logger.info(f"Parts per scene: {args.min_parts}–{args.max_parts}")
    logger.info(f"Allow partial overhang: {args.allow_partial}")
    logger.info(f"JPEG quality range: {args.jpeg_quality_min}–{args.jpeg_quality_max}")

    jpeg_quality = (args.jpeg_quality_min, args.jpeg_quality_max)

    generate_split(part_paths_with_classes, output_dir, 'train', args.num_train,
                   args.image_size, args.min_parts, args.max_parts,
                   args.min_scale, args.max_scale,
                   args.allow_partial, jpeg_quality, rng)

    generate_split(part_paths_with_classes, output_dir, 'val', args.num_val,
                   args.image_size, args.min_parts, args.max_parts,
                   args.min_scale, args.max_scale,
                   args.allow_partial, jpeg_quality, rng)

    write_yaml(output_dir, class_to_part)
    write_parts_txt(output_dir, class_to_part)

    total = args.num_train + args.num_val
    logger.info("=" * 60)
    logger.info(f"Dataset ready: {total} scenes → {output_dir}")
    logger.info(f"Classes: {len(class_to_part)} unique LEGO parts")
    logger.info(f"Next step:")
    logger.info(f"  python train_yolo.py --data {output_dir}/lego.yaml \\")
    logger.info(f"    --model yolov8m.pt --epochs 100 --imgsz {args.image_size}")


if __name__ == '__main__':
    main()
