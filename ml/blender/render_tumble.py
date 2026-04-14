#!/usr/bin/env python3
"""
Blender 360° tumble renderer for LEGO parts — renders parts from all realistic orientations.
Runs INSIDE Blender: blender --background --python render_tumble.py -- <args>

Orientations covered:
1. Upright (stud-side up): 8 azimuths × 2 elevations = 16 shots
2. Inverted (stud-side down): 8 azimuths × 2 elevations = 16 shots
3. On side (longest axis horizontal): 4 rotations × 2 faces = 8 shots
4. Diagonal/tumbling (45° tilts): 8 shots
= 48 orientations total per part per color
"""

import bpy
import sys
import argparse
import math
import csv
from pathlib import Path
from mathutils import Vector, Euler, Matrix

# Parse command-line arguments
argv = sys.argv[sys.argv.index("--") + 1:]
parser = argparse.ArgumentParser(description="Render LEGO part from all tumble orientations")
parser.add_argument("--part-file", required=True, help="Path to LDraw .dat file")
parser.add_argument("--output-dir", required=True, help="Output directory for renders")
parser.add_argument("--color-r", type=float, required=True, help="Red channel (0-1)")
parser.add_argument("--color-g", type=float, required=True, help="Green channel (0-1)")
parser.add_argument("--color-b", type=float, required=True, help="Blue channel (0-1)")
parser.add_argument("--part-num", required=True, help="Part number string")
parser.add_argument("--color-id", type=int, required=True, help="Color ID from Rebrickable")
parser.add_argument("--color-name", required=True, help="Color name for logging")
parser.add_argument("--resolution", type=int, default=224, help="Output resolution (default: 224)")
parser.add_argument("--index-csv", default=None, help="Path to index.csv for appending rows")
args = parser.parse_args(argv)

print(f"[BrickScan Tumble Renderer] Starting render for part {args.part_num}, color {args.color_name}")

# ==============================================================================
# SCENE SETUP (shared with render_parts.py)
# ==============================================================================

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

world = bpy.data.worlds["World"]
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.05, 1.0)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.cycles.use_denoising = True

# GPU setup (macOS Metal, Linux CUDA/OptiX)
import platform as _platform
prefs = bpy.context.preferences
cycles_prefs = prefs.addons["cycles"].preferences

_is_mac = _platform.system() == "Darwin"
_is_linux = _platform.system() == "Linux"

_device_priority = ("METAL",) if _is_mac else (("OPTIX", "CUDA", "HIP") if _is_linux else ("CUDA", "OPTIX"))
_gpu_configured = False

for _device_type in _device_priority:
    try:
        cycles_prefs.compute_device_type = _device_type
        cycles_prefs.refresh_devices()
        _enabled = [d for d in cycles_prefs.devices if d.type != "CPU"]
        if _enabled:
            for d in cycles_prefs.devices:
                d.use = True
            _gpu_configured = True
            print(f"[BrickScan Tumble Renderer] ✓ GPU: {_device_type}")
            break
    except Exception as _e:
        print(f"[BrickScan Tumble Renderer] {_device_type} unavailable: {_e}")

if not _gpu_configured:
    print("[BrickScan Tumble Renderer] ⚠ Using CPU rendering")
    scene.cycles.device = "CPU"

scene.render.resolution_x = args.resolution
scene.render.resolution_y = args.resolution
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = True

# ==============================================================================
# LEGO PART IMPORT (copied from render_parts.py)
# ==============================================================================

LDRAW_SCALE = 0.04

def _mat_mul(a, b):
    r = [[0.0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                r[i][j] += a[i][k] * b[k][j]
    return r

def _transform_vertex(m, x, y, z):
    nx = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3]
    ny = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3]
    nz = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3]
    return (nx * LDRAW_SCALE, -nz * LDRAW_SCALE, -ny * LDRAW_SCALE)

def _find_ldraw_file(filename, ldraw_root, parent_path):
    """Find LDraw file in library."""
    if filename.startswith("s/"):
        filename = filename[2:]

    candidates = [
        Path(ldraw_root) / "parts" / filename,
        Path(ldraw_root) / "p" / filename,
        Path(ldraw_root) / "models" / filename,
        parent_path.parent / filename if parent_path else None,
    ]

    for cand in candidates:
        if cand and cand.exists():
            return cand

    return None

def _parse_ldraw_matrix(tokens):
    """Parse LDraw 1 command transformation matrix."""
    if len(tokens) < 15:
        return None
    try:
        x, y, z = float(tokens[2]), float(tokens[3]), float(tokens[4])
        a11, a12, a13 = float(tokens[5]), float(tokens[6]), float(tokens[7])
        a21, a22, a23 = float(tokens[8]), float(tokens[9]), float(tokens[10])
        a31, a32, a33 = float(tokens[11]), float(tokens[12]), float(tokens[13])
        return [
            [a11, a12, a13, x],
            [a21, a22, a23, y],
            [a31, a32, a33, z],
            [0, 0, 0, 1]
        ]
    except (ValueError, IndexError):
        return None

def _load_ldraw_part(part_file, ldraw_root, color_rgb, depth=0, parent_matrix=None):
    """Recursively load LDraw part and build Blender mesh."""
    if depth > 20:
        print(f"[BrickScan] Recursion limit reached")
        return None

    if not Path(part_file).exists():
        print(f"[BrickScan] Part file not found: {part_file}")
        return None

    mesh_data = bpy.data.meshes.new("ldraw_mesh")
    vertices = []
    faces = []

    try:
        with open(part_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("0"):
                    continue

                tokens = line.split()
                if len(tokens) < 1:
                    continue

                # Type 1: Part reference
                if tokens[0] == "1" and len(tokens) >= 15:
                    sub_file = tokens[14]
                    sub_path = _find_ldraw_file(sub_file, ldraw_root, Path(part_file))

                    if sub_path:
                        sub_matrix = _parse_ldraw_matrix(tokens)
                        if sub_matrix:
                            if parent_matrix:
                                sub_matrix = _mat_mul(parent_matrix, sub_matrix)
                            _load_ldraw_part(str(sub_path), ldraw_root, color_rgb, depth + 1, sub_matrix)

                # Type 3: Triangle
                elif tokens[0] == "3" and len(tokens) >= 11:
                    try:
                        x1, y1, z1 = float(tokens[2]), float(tokens[3]), float(tokens[4])
                        x2, y2, z2 = float(tokens[5]), float(tokens[6]), float(tokens[7])
                        x3, y3, z3 = float(tokens[8]), float(tokens[9]), float(tokens[10])

                        identity = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
                        matrix = parent_matrix or identity

                        v1 = _transform_vertex(matrix, x1, y1, z1)
                        v2 = _transform_vertex(matrix, x2, y2, z2)
                        v3 = _transform_vertex(matrix, x3, y3, z3)

                        idx = len(vertices)
                        vertices.extend([v1, v2, v3])
                        faces.append((idx, idx+1, idx+2))
                    except (ValueError, IndexError):
                        pass

                # Type 4: Quad
                elif tokens[0] == "4" and len(tokens) >= 14:
                    try:
                        x1, y1, z1 = float(tokens[2]), float(tokens[3]), float(tokens[4])
                        x2, y2, z2 = float(tokens[5]), float(tokens[6]), float(tokens[7])
                        x3, y3, z3 = float(tokens[8]), float(tokens[9]), float(tokens[10])
                        x4, y4, z4 = float(tokens[11]), float(tokens[12]), float(tokens[13])

                        identity = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
                        matrix = parent_matrix or identity

                        v1 = _transform_vertex(matrix, x1, y1, z1)
                        v2 = _transform_vertex(matrix, x2, y2, z2)
                        v3 = _transform_vertex(matrix, x3, y3, z3)
                        v4 = _transform_vertex(matrix, x4, y4, z4)

                        idx = len(vertices)
                        vertices.extend([v1, v2, v3, v4])
                        faces.append((idx, idx+1, idx+2))
                        faces.append((idx, idx+2, idx+3))
                    except (ValueError, IndexError):
                        pass

    except Exception as e:
        print(f"[BrickScan] Error parsing {part_file}: {e}")
        return None

    if not vertices:
        return None

    mesh_data.from_pydata(vertices, [], faces)
    mesh_data.update()

    obj = bpy.data.objects.new("lego_part", mesh_data)
    bpy.context.collection.objects.link(obj)

    # Apply material
    mat = bpy.data.materials.new(name="lego_material")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (args.color_r, args.color_g, args.color_b, 1.0)
    mat.node_tree.nodes["Principled BSDF"].inputs[9].default_value = 0.3  # roughness
    obj.data.materials.append(mat)

    return obj

# Load the part
part_obj = _load_ldraw_part(args.part_file, Path(args.part_file).parent.parent, (args.color_r, args.color_g, args.color_b))

if not part_obj:
    print("[BrickScan Tumble Renderer] ERROR: Failed to load part")
    sys.exit(1)

# Center and compute bounds
bpy.context.view_layer.update()
bpy.context.view_layer.objects.active = part_obj
bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')

bbox_min = Vector(part_obj.bound_box[0])
bbox_max = Vector(part_obj.bound_box[6])
bbox_center = (bbox_min + bbox_max) / 2
bbox_size = (bbox_max - bbox_min).length

part_obj.location = -bbox_center

print(f"[BrickScan Tumble Renderer] Part bounds: {bbox_size:.3f}m")

# ==============================================================================
# CAMERA & LIGHTING SETUP
# ==============================================================================

camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
bpy.context.collection.objects.link(camera)
scene.camera = camera

light = bpy.data.objects.new("Light", bpy.data.lights.new("Light", type="SUN"))
light.data.energy = 1500
bpy.context.collection.objects.link(light)

# Position camera at distance proportional to bbox
camera_dist = bbox_size * 2.5

# ==============================================================================
# TUMBLE ORIENTATIONS
# ==============================================================================

def generate_tumble_orientations():
    """
    Generate all 48 orientations for tumble rendering.
    Returns list of (name, euler_xyz_radians)
    """
    orientations = []

    # 1. UPRIGHT (stud-side up)
    # 8 azimuths × 2 elevations = 16
    for azim_idx in range(8):
        azim_deg = azim_idx * 45
        for elev_deg in [30, 60]:
            name = f"upright_{azim_deg:03d}_{elev_deg:02d}"
            # Rotation: azimuth around Z, elevation around local X
            rot_x = math.radians(elev_deg)
            rot_z = math.radians(azim_deg)
            orientations.append((name, (rot_x, 0, rot_z)))

    # 2. INVERTED (stud-side down)
    # 8 azimuths × 2 elevations = 16
    for azim_idx in range(8):
        azim_deg = azim_idx * 45
        for elev_deg in [30, 60]:
            name = f"inverted_{azim_deg:03d}_{elev_deg:02d}"
            # 180° flip around X first, then elevate and rotate
            rot_x = math.radians(180 + elev_deg)
            rot_z = math.radians(azim_deg)
            orientations.append((name, (rot_x, 0, rot_z)))

    # 3. ON SIDE (longest axis horizontal)
    # 4 rotations × 2 faces = 8
    for side_idx in range(4):
        side_rot = side_idx * 90
        for face_idx in range(2):
            face_flip = face_idx * 180
            name = f"side_{side_rot:03d}_{face_flip:03d}"
            rot_x = math.radians(90)  # Tip over 90° to lay on side
            rot_z = math.radians(side_rot)
            rot_y = math.radians(face_flip)
            # Apply all rotations
            orientations.append((name, (rot_x, rot_y, rot_z)))

    # 4. DIAGONAL/TUMBLING (45° tilts on multiple axes)
    # 8 random 45° tilts
    for diag_idx in range(8):
        azim_deg = diag_idx * 45
        name = f"diagonal_{azim_deg:03d}"
        rot_x = math.radians(45)
        rot_y = math.radians(45)
        rot_z = math.radians(azim_deg)
        orientations.append((name, (rot_x, rot_y, rot_z)))

    return orientations

orientations = generate_tumble_orientations()
print(f"[BrickScan Tumble Renderer] Generated {len(orientations)} orientations")

# ==============================================================================
# RENDER LOOP
# ==============================================================================

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

csv_rows = []

for orient_name, (rot_x, rot_y, rot_z) in orientations:
    # Apply rotation to part
    part_obj.rotation_euler = (rot_x, rot_y, rot_z)

    # Position camera looking down at origin
    camera.location = (0, -camera_dist, camera_dist * 0.7)
    camera.rotation_euler = (math.radians(45), 0, 0)

    # Update light position
    light.location = camera.location + Vector((bbox_size, -bbox_size, bbox_size))

    # Render
    output_path = output_dir / f"{orient_name}.png"
    scene.render.filepath = str(output_path)

    try:
        bpy.ops.render.render(write_still=True)
        print(f"[BrickScan Tumble Renderer] ✓ {orient_name}")

        # Extract azimuth and elevation from name
        parts = orient_name.split('_')
        if len(parts) >= 3:
            azim = parts[-2]
            elev = parts[-1]
            csv_rows.append({
                'part_num': args.part_num,
                'color_id': args.color_id,
                'color_name': args.color_name,
                'orientation': orient_name,
                'azimuth': azim,
                'elevation': elev,
                'image_path': str(output_path.relative_to(output_dir.parent)),
            })
    except Exception as e:
        print(f"[BrickScan Tumble Renderer] ✗ {orient_name}: {e}")

# ==============================================================================
# WRITE INDEX CSV
# ==============================================================================

if args.index_csv and csv_rows:
    try:
        index_csv_path = Path(args.index_csv)
        index_csv_path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = index_csv_path.exists()
        with open(index_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(csv_rows)

        print(f"[BrickScan Tumble Renderer] ✓ Index CSV: {index_csv_path}")
    except Exception as e:
        print(f"[BrickScan Tumble Renderer] ✗ Failed to write index CSV: {e}")

print(f"[BrickScan Tumble Renderer] Complete: {len(csv_rows)} images")
