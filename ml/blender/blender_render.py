#!/usr/bin/env python3
"""
blender_render.py — Headless Blender renderer for LEGO part images.

Renders a single LDraw .dat part at:
  36 rotation angles × 5 lighting presets × 3 zoom levels = 540 images/part

Usage (headless):
  /Applications/Blender.app/Contents/MacOS/Blender --background \
    --python blender_render.py -- \
    --part-id 3001 \
    --ldraw-dir ~/ldraw \
    --output-dir ./synthetic_dataset
"""

import bpy
import bmesh
import math
import os
import sys
import random
import json
import argparse
from pathlib import Path
from mathutils import Matrix, Vector

# ─── Parse args passed after '--' ──────────────────────────────────────────
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument('--part-id',   required=True,  help='LDraw part number, e.g. 3001')
parser.add_argument('--ldraw-dir', required=True,  help='Path to LDraw library root')
parser.add_argument('--output-dir',required=True,  help='Base output directory')
parser.add_argument('--num-angles',type=int, default=36,  help='Rotation steps (default 36 = every 10°)')
parser.add_argument('--num-lights',type=int, default=5,   help='Lighting presets (default 5)')
parser.add_argument('--num-zooms', type=int, default=3,   help='Zoom levels (default 3)')
parser.add_argument('--resolution',type=int, default=224, help='Output image size in px (default 224)')
parser.add_argument('--color',     type=str, default='4', help='LDraw color ID (default 4=red)')
args = parser.parse_args(argv)


# ─── LDraw parser ──────────────────────────────────────────────────────────

# Minimal LDraw→RGB color table (expand as needed)
LDRAW_COLORS = {
    0:  (0.04, 0.04, 0.04, 1.0),   # Black
    1:  (0.00, 0.33, 0.69, 1.0),   # Blue
    2:  (0.00, 0.55, 0.08, 1.0),   # Green
    4:  (0.78, 0.00, 0.00, 1.0),   # Red
    5:  (0.78, 0.39, 0.71, 1.0),   # Dark Pink
    7:  (0.49, 0.49, 0.49, 1.0),   # Light Gray
    14: (0.98, 0.81, 0.00, 1.0),   # Yellow
    15: (1.00, 1.00, 1.00, 1.0),   # White
    16: (0.70, 0.70, 0.70, 1.0),   # Main color (placeholder)
    19: (0.90, 0.80, 0.60, 1.0),   # Tan
    25: (0.98, 0.51, 0.00, 1.0),   # Orange
    70: (0.24, 0.13, 0.08, 1.0),   # Dark Brown
    71: (0.64, 0.65, 0.64, 1.0),   # Light Bluish Gray
    72: (0.36, 0.37, 0.41, 1.0),   # Dark Bluish Gray
}

def ldraw_color_to_rgba(code_str, parent_code=16):
    """Convert LDraw color code to RGBA tuple."""
    try:
        code = int(code_str)
    except ValueError:
        return (0.7, 0.7, 0.7, 1.0)
    if code == 16:  # inherit from parent
        code = parent_code
    if code == 24:  # edge color
        return (0.1, 0.1, 0.1, 1.0)
    return LDRAW_COLORS.get(code, (0.7, 0.7, 0.7, 1.0))


class LDrawParser:
    """
    Recursive LDraw .dat parser.
    Supports line types 1 (sub-ref), 3 (triangle), 4 (quad).
    Coordinate system: LDraw Y-up inverted → Blender Z-up via swap.
    Scale: 1 LDraw unit = 0.4mm, dividing by 25 gives nice ~cm scale.
    """

    LDU_SCALE = 0.04   # 1 LDU → 0.04 Blender units (≈ 1.6mm per Blender unit)

    def __init__(self, ldraw_dir: str):
        self.ldraw_dir = Path(ldraw_dir)
        self.search_paths = [
            self.ldraw_dir / 'parts',
            self.ldraw_dir / 'p' / '48',
            self.ldraw_dir / 'p',
            self.ldraw_dir,
        ]
        self._cache = {}

    def find_file(self, name: str) -> Path | None:
        name = name.replace('\\', os.sep).replace('/', os.sep)
        for sp in self.search_paths:
            for candidate in [name, name.lower(), name.upper()]:
                p = sp / candidate
                if p.exists():
                    return p
        return None

    def parse(self, filename: str, transform=None, parent_color=16, depth=0):
        """Return list of (v0, v1, v2, rgba) triangles in world space."""
        if depth > 25:
            return []
        if transform is None:
            transform = Matrix.Identity(4)

        key = (filename, depth)
        filepath = self.find_file(filename)
        if filepath is None:
            return []

        tris = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith('0'):
                        continue
                    tokens = line.split()
                    ltype = tokens[0]

                    if ltype == '1' and len(tokens) >= 15:
                        color    = int(tokens[1])
                        tx, ty, tz = float(tokens[2]), float(tokens[3]), float(tokens[4])
                        a, b, c  = float(tokens[5]),  float(tokens[6]),  float(tokens[7])
                        d, e, f  = float(tokens[8]),  float(tokens[9]),  float(tokens[10])
                        g, h, i  = float(tokens[11]), float(tokens[12]), float(tokens[13])
                        sub_name = ' '.join(tokens[14:])
                        local = Matrix([
                            [a,  b,  c,  tx],
                            [d,  e,  f,  ty],
                            [g,  h,  i,  tz],
                            [0,  0,  0,  1 ],
                        ])
                        sub_color = color if color != 16 else parent_color
                        tris += self.parse(sub_name, transform @ local, sub_color, depth+1)

                    elif ltype == '3' and len(tokens) >= 11:
                        rgba  = ldraw_color_to_rgba(tokens[1], parent_color)
                        verts = []
                        for k in range(3):
                            lx = float(tokens[2 + k*3])
                            ly = float(tokens[3 + k*3])
                            lz = float(tokens[4 + k*3])
                            w  = transform @ Vector((lx, ly, lz, 1.0))
                            # LDraw→Blender: swap Y/Z, invert original Y
                            verts.append(Vector((w.x, -w.z, w.y)) * self.LDU_SCALE)
                        tris.append((*verts, rgba))

                    elif ltype == '4' and len(tokens) >= 14:
                        rgba  = ldraw_color_to_rgba(tokens[1], parent_color)
                        verts = []
                        for k in range(4):
                            lx = float(tokens[2 + k*3])
                            ly = float(tokens[3 + k*3])
                            lz = float(tokens[4 + k*3])
                            w  = transform @ Vector((lx, ly, lz, 1.0))
                            verts.append(Vector((w.x, -w.z, w.y)) * self.LDU_SCALE)
                        tris.append((verts[0], verts[1], verts[2], rgba))
                        tris.append((verts[0], verts[2], verts[3], rgba))
        except Exception:
            pass
        return tris


# ─── Scene helpers ─────────────────────────────────────────────────────────

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for col in bpy.data.collections:
        bpy.data.collections.remove(col)


def build_mesh(tris, part_id, base_color_rgba):
    """Create a single Blender mesh object from triangle soup."""
    mesh = bpy.data.meshes.new(f'part_{part_id}')
    obj  = bpy.data.objects.new(f'part_{part_id}', mesh)
    bpy.context.scene.collection.objects.link(obj)

    bm = bmesh.new()
    vert_map = {}

    def get_or_add(v):
        key = (round(v.x, 5), round(v.y, 5), round(v.z, 5))
        if key not in vert_map:
            vert_map[key] = bm.verts.new(v)
        return vert_map[key]

    for tri in tris:
        v0, v1, v2, rgba = tri
        bv0, bv1, bv2 = get_or_add(v0), get_or_add(v1), get_or_add(v2)
        try:
            bm.faces.new([bv0, bv1, bv2])
        except Exception:
            pass  # duplicate face

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    # Apply a single principled BSDF material using the base_color_rgba
    mat = bpy.data.materials.new(name=f'mat_{part_id}')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = base_color_rgba
        bsdf.inputs['Roughness'].default_value  = 0.3
        bsdf.inputs['Specular IOR Level'].default_value  = 0.5
    mesh.materials.append(mat)

    return obj


def center_and_normalize(obj):
    """Move object so its bounding-box center is at origin; return diameter."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)
    bpy.context.view_layer.update()
    dims = obj.dimensions
    diameter = max(dims.x, dims.y, dims.z)
    if diameter < 1e-6:
        diameter = 1.0
    return diameter


def setup_camera(diameter, zoom_level):
    """
    Add a camera pointing at origin.
    zoom_level: 0=far(full part), 1=medium, 2=close-up
    """
    distances = [diameter * 3.5, diameter * 2.5, diameter * 1.8]
    dist = distances[zoom_level % len(distances)]

    cam_data = bpy.data.cameras.new('Camera')
    cam_obj  = bpy.data.objects.new('Camera', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    # 30° elevation, looking down at the part
    elevation = math.radians(30)
    cam_obj.location = (dist * math.cos(elevation),
                        -dist * math.cos(elevation),
                         dist * math.sin(elevation))
    # Point at origin
    direction = Vector((0, 0, 0)) - cam_obj.location
    rot_quat  = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    cam_data.lens = 50
    bpy.context.scene.camera = cam_obj
    return cam_obj


def setup_three_point_lighting(diameter, preset_idx, rng):
    """
    3-point lighting rig: key + fill + rim.
    preset_idx 0-4 varies angles/intensities for diversity.
    Returns list of light objects.
    """
    presets = [
        dict(key_energy=800,  fill_energy=200, rim_energy=400),
        dict(key_energy=1000, fill_energy=150, rim_energy=600),
        dict(key_energy=600,  fill_energy=300, rim_energy=200),
        dict(key_energy=1200, fill_energy=100, rim_energy=800),
        dict(key_energy=700,  fill_energy=400, rim_energy=300),
    ]
    p  = presets[preset_idx % len(presets)]
    d  = diameter * 4
    jitter = 0.2  # ±20% intensity randomization

    def add_light(name, loc, energy, light_type='AREA'):
        light_data = bpy.data.lights.new(name=name, type=light_type)
        light_data.energy = energy * (1 + rng.uniform(-jitter, jitter))
        light_data.shadow_soft_size = diameter
        light_obj  = bpy.data.objects.new(name, light_data)
        light_obj.location = loc
        # Point toward origin
        direction = Vector((0, 0, 0)) - Vector(loc)
        light_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        bpy.context.scene.collection.objects.link(light_obj)
        return light_obj

    key_angle  = math.radians(45 + preset_idx * 15)
    lights = [
        add_light('Key',  ( d*math.cos(key_angle),          d*math.sin(key_angle),           d*0.8), p['key_energy']),
        add_light('Fill', (-d*math.cos(key_angle)*0.7,      -d*math.sin(key_angle)*0.5,       d*0.5), p['fill_energy']),
        add_light('Rim',  ( 0,                               d,                              -d*0.3), p['rim_energy']),
    ]
    return lights


def remove_lights_and_camera():
    for obj in list(bpy.context.scene.objects):
        if obj.type in ('LIGHT', 'CAMERA'):
            bpy.data.objects.remove(obj, do_unlink=True)


def configure_renderer(resolution):
    scene = bpy.context.scene
    scene.render.engine               = 'CYCLES'
    scene.cycles.samples              = 32          # fast but decent quality
    scene.cycles.use_denoising        = True
    scene.cycles.device               = 'GPU' if bpy.context.preferences.addons.get('cycles') else 'CPU'

    # Try to use Metal on Mac
    prefs = bpy.context.preferences.addons.get('cycles')
    if prefs:
        cprefs = prefs.preferences
        try:
            cprefs.compute_device_type = 'METAL'
        except Exception:
            try:
                cprefs.compute_device_type = 'NONE'
            except Exception:
                pass

    scene.render.film_transparent      = True       # transparent background
    scene.render.image_settings.file_format        = 'PNG'
    scene.render.image_settings.color_mode         = 'RGBA'
    scene.render.image_settings.color_depth        = '8'
    scene.render.image_settings.compression        = 15
    scene.render.resolution_x         = resolution
    scene.render.resolution_y         = resolution
    scene.render.resolution_percentage = 100
    scene.world.use_nodes              = True
    # Set world background to transparent
    bg_node = scene.world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Strength'].default_value = 0.0


# ─── Main render loop ──────────────────────────────────────────────────────

def main():
    rng = random.Random(42)

    ldraw_dir  = Path(args.ldraw_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    part_id    = args.part_id
    resolution = args.resolution
    color_code = int(args.color)
    base_color = ldraw_color_to_rgba(str(color_code))

    out_part_dir = output_dir / str(part_id)
    out_part_dir.mkdir(parents=True, exist_ok=True)

    # ── Parse LDraw geometry ──────────────────────────────────────────────
    print(f"[blender_render] Parsing {part_id}.dat from {ldraw_dir}")
    parser_ldraw = LDrawParser(str(ldraw_dir))
    tris = parser_ldraw.parse(f'{part_id}.dat', parent_color=color_code)

    if not tris:
        print(f"[blender_render] ERROR: No geometry found for {part_id}. Check LDraw path.")
        sys.exit(1)

    print(f"[blender_render] Parsed {len(tris)} triangles")

    # ── Build scene once ──────────────────────────────────────────────────
    clear_scene()
    configure_renderer(resolution)

    part_obj = build_mesh(tris, part_id, base_color)
    diameter  = center_and_normalize(part_obj)
    print(f"[blender_render] Part diameter: {diameter:.3f} Blender units")

    # ── Render loop ───────────────────────────────────────────────────────
    total      = args.num_angles * args.num_lights * args.num_zooms
    rendered   = 0
    skipped    = 0

    for angle_idx in range(args.num_angles):
        angle_deg = (360.0 / args.num_angles) * angle_idx
        angle_rad = math.radians(angle_deg)
        part_obj.rotation_euler = (0, 0, angle_rad)

        for light_idx in range(args.num_lights):
            for zoom_idx in range(args.num_zooms):
                filename = f'{angle_idx:03d}_{light_idx}_{zoom_idx}.png'
                filepath = out_part_dir / filename

                if filepath.exists():
                    skipped += 1
                    continue

                # Fresh lighting + camera each frame to avoid stale transforms
                remove_lights_and_camera()
                setup_three_point_lighting(diameter, light_idx, rng)
                setup_camera(diameter, zoom_idx)

                bpy.context.scene.render.filepath = str(filepath)
                bpy.ops.render.render(write_still=True)
                rendered += 1

                if rendered % 20 == 0:
                    print(f"[blender_render] {rendered}/{total - skipped} rendered, {skipped} skipped")

    print(f"[blender_render] Done: {rendered} rendered, {skipped} skipped → {out_part_dir}")


if __name__ == '__main__':
    main()
