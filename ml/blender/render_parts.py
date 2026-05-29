#!/usr/bin/env python3
"""
Blender GPU rendering script for LEGO part synthetic data generation.
Runs INSIDE Blender: blender --background --python render_parts.py -- <args>
"""

import bpy
import os
import sys
import argparse
import random
import math
import csv
from pathlib import Path
from mathutils import Vector, Euler

# Parse command-line arguments
argv = sys.argv[sys.argv.index("--") + 1:]
parser = argparse.ArgumentParser(description="Render LEGO part with specified color")
parser.add_argument("--part-file", required=True, help="Path to LDraw .dat file")
parser.add_argument("--output-dir", required=True, help="Output directory for renders")
parser.add_argument("--color-r", type=float, required=True, help="Red channel (0-1)")
parser.add_argument("--color-g", type=float, required=True, help="Green channel (0-1)")
parser.add_argument("--color-b", type=float, required=True, help="Blue channel (0-1)")
parser.add_argument("--part-num", required=True, help="Part number string")
parser.add_argument("--color-id", type=int, required=True, help="Color ID from Rebrickable")
parser.add_argument("--color-name", required=True, help="Color name for logging")
parser.add_argument("--num-angles", type=int, default=36, help="Number of azimuth angles per elevation (default: 36)")
parser.add_argument("--resolution", type=int, default=224, help="Output resolution (default: 224)")
parser.add_argument("--dr-strength", type=float, default=1.0, help="Domain-randomization strength 0..1 (default: 1.0)")
parser.add_argument("--index-csv", default=None, help="Path to index.csv for appending rows")
args = parser.parse_args(argv)

print(f"[BrickScan Renderer] Starting render for part {args.part_num}, color {args.color_name}")

# ==============================================================================
# 1. SETUP SCENE
# ==============================================================================

# Clear default scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# Create world/environment
world = bpy.data.worlds["World"]
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.05, 1.0)

# ==============================================================================
# 2. CONFIGURE GPU RENDERING
# ==============================================================================

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 128

# ── Cross-platform GPU setup ──────────────────────────────────────────────────
# macOS Apple Silicon → Metal
# Linux NVIDIA (GB10 Blackwell) → OptiX → CUDA fallback
# Set BRICKSCAN_FORCE_CPU=1 to skip the GPU probe entirely (e.g. smoke tests, or
# when the GPU is occupied by training). This avoids contending a busy GPU.
import platform as _platform
import sys as _sys

prefs = bpy.context.preferences
cycles_prefs = prefs.addons["cycles"].preferences

_is_mac = _platform.system() == "Darwin"
_is_linux = _platform.system() == "Linux"
_force_cpu = os.environ.get("BRICKSCAN_FORCE_CPU", "").strip() not in ("", "0", "false", "False")

# Pick device type priority based on platform
if _is_mac:
    _device_priority = ("METAL",)
elif _is_linux:
    _device_priority = ("OPTIX", "CUDA", "HIP")
else:
    _device_priority = ("CUDA", "OPTIX")

_gpu_configured = False
if _force_cpu:
    print("[BrickScan Renderer] BRICKSCAN_FORCE_CPU set — using CPU rendering")
else:
    for _device_type in _device_priority:
        try:
            cycles_prefs.compute_device_type = _device_type
            cycles_prefs.refresh_devices()
            _enabled = [d for d in cycles_prefs.devices if d.type != "CPU"]
            if _enabled:
                for d in cycles_prefs.devices:
                    d.use = True   # enable all devices (CPU + GPU unified memory)
                _gpu_configured = True
                print(f"[BrickScan Renderer] ✓ GPU backend: {_device_type} "
                      f"({len(_enabled)} device(s), platform: {_platform.system()})")
                break
        except Exception as _e:
            print(f"[BrickScan Renderer] {_device_type} unavailable: {_e}")

if _gpu_configured:
    scene.cycles.device = "GPU"
else:
    if not _force_cpu:
        print("[BrickScan Renderer] ⚠ No GPU found — falling back to CPU rendering")
    scene.cycles.device = "CPU"

# Denoiser: set after device config so the enum is populated. OpenImageDenoise
# works on both CPU and GPU; fall back gracefully if unavailable.
scene.cycles.use_denoising = True
try:
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
except (TypeError, AttributeError):
    try:
        scene.cycles.denoiser = "OPTIX" if scene.cycles.device == "GPU" else "OPENIMAGEDENOISE"
    except (TypeError, AttributeError):
        scene.cycles.use_denoising = False
        print("[BrickScan Renderer] ⚠ Denoiser unavailable — rendering without denoise")

# Output settings
scene.render.resolution_x = args.resolution
scene.render.resolution_y = args.resolution
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = True

print(f"[BrickScan Renderer] Render configured: device={scene.cycles.device}, "
      f"samples={scene.cycles.samples}, denoise={scene.cycles.use_denoising}")

# ==============================================================================
# 3. IMPORT LEGO PART FROM LDRAW (built-in parser — no add-on required)
# ==============================================================================

# ── LDraw coordinate system ──
# LDraw: X right, Y down, Z back
# Blender: X right, Y into screen, Z up
# Transform: (ldx, ldy, ldz) → (ldx, -ldz, -ldy), scaled by 0.04 (LDU → m-ish)

LDRAW_SCALE = 0.04  # LDraw units → Blender units

def _mat_mul(a, b):
    """4×4 row-major matrix multiply."""
    r = [[0.0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                r[i][j] += a[i][k] * b[k][j]
    return r

def _transform_vertex(m, x, y, z):
    """Apply 4×4 LDraw matrix to a vertex and convert coordinate system."""
    nx = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3]
    ny = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3]
    nz = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3]
    # LDraw (X, Y, Z) → Blender (X, -Z, -Y), scaled
    return (nx * LDRAW_SCALE, -nz * LDRAW_SCALE, -ny * LDRAW_SCALE)

def _find_ldraw_file(filename, ldraw_root, parent_path):
    """Search LDraw library directories for a sub-file."""
    fn = filename.lower().replace("\\", "/")
    base = fn.replace("s/", "")
    candidates = [
        os.path.join(ldraw_root, "parts", fn),
        os.path.join(ldraw_root, "p", fn),
        os.path.join(ldraw_root, "models", fn),
        os.path.join(ldraw_root, "parts", "s", base),
        os.path.join(ldraw_root, "p", "48", base),
        os.path.join(ldraw_root, "p", "8", base),
        os.path.join(os.path.dirname(parent_path), fn),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def _parse_ldraw(filepath, ldraw_root, matrix=None, verts=None, faces=None, depth=0):
    """Recursively parse an LDraw .dat/.ldr file into flat vertex/face lists."""
    if depth > 12:
        return
    if verts is None:
        verts = []
    if faces is None:
        faces = []
    if matrix is None:
        matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
    except OSError:
        return

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        ltype = parts[0]

        if ltype == "1" and len(parts) >= 15:
            # Sub-file reference
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            a, b, c = float(parts[5]), float(parts[6]), float(parts[7])
            d, e, f_ = float(parts[8]), float(parts[9]), float(parts[10])
            g, h, i  = float(parts[11]), float(parts[12]), float(parts[13])
            sub_name = " ".join(parts[14:])
            sub_mat  = [[a,b,c,x],[d,e,f_,y],[g,h,i,z],[0,0,0,1]]
            combined = _mat_mul(matrix, sub_mat)
            sub_path = _find_ldraw_file(sub_name, ldraw_root, filepath)
            if sub_path:
                _parse_ldraw(sub_path, ldraw_root, combined, verts, faces, depth+1)

        elif ltype == "3" and len(parts) >= 11:
            # Triangle
            tri = []
            for j in range(3):
                vx, vy, vz = _transform_vertex(
                    matrix,
                    float(parts[2 + j*3]),
                    float(parts[3 + j*3]),
                    float(parts[4 + j*3]),
                )
                tri.append(len(verts))
                verts.append((vx, vy, vz))
            faces.append(tuple(tri))

        elif ltype == "4" and len(parts) >= 14:
            # Quad
            quad = []
            for j in range(4):
                vx, vy, vz = _transform_vertex(
                    matrix,
                    float(parts[2 + j*3]),
                    float(parts[3 + j*3]),
                    float(parts[4 + j*3]),
                )
                quad.append(len(verts))
                verts.append((vx, vy, vz))
            faces.append(tuple(quad))

    return verts, faces

def import_ldraw_to_blender(dat_file, ldraw_root):
    """Parse an LDraw file and create a Blender mesh object."""
    verts = []
    faces = []
    _parse_ldraw(dat_file, ldraw_root, verts=verts, faces=faces)

    if not verts:
        raise ValueError("No geometry found in LDraw file")

    mesh = bpy.data.meshes.new("LegoPart")
    obj  = bpy.data.objects.new("LegoPart", mesh)
    bpy.context.collection.objects.link(obj)

    mesh.from_pydata(verts, [], faces)
    mesh.update(calc_edges=True)

    # Merge duplicate verts & fix normals
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=0.0005)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    print(f"[BrickScan Renderer] Built mesh: {len(mesh.vertices)} verts, "
          f"{len(mesh.polygons)} faces")
    return obj

# ── Locate LDraw library root ──────────────────────────────────────────────────
part_file = args.part_file
if not os.path.exists(part_file):
    print(f"[ERROR] Part file not found: {part_file}")
    sys.exit(1)

# Derive ldraw_root from part_file location (…/ldraw/parts/3001.dat → …/ldraw)
_p = Path(part_file).resolve()
ldraw_root = str(_p.parent.parent)   # go up from parts/ to ldraw/
if not os.path.isdir(os.path.join(ldraw_root, "parts")):
    print(f"[ERROR] Cannot locate LDraw library root near {part_file}")
    sys.exit(1)
print(f"[BrickScan Renderer] LDraw root: {ldraw_root}")

try:
    imported_obj = import_ldraw_to_blender(part_file, ldraw_root)
except Exception as e:
    print(f"[ERROR] Failed to import LDraw part: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print(f"[BrickScan Renderer] Loaded part mesh: {imported_obj.name}")

# ==============================================================================
# 4. APPLY MATERIAL
# ==============================================================================

# Create or get material
mat_name = f"LEGO_Color_{args.color_id}"
if mat_name in bpy.data.materials:
    mat = bpy.data.materials[mat_name]
else:
    mat = bpy.data.materials.new(name=mat_name)

mat.use_nodes = True
mat.node_tree.nodes.clear()

# Create Principled BSDF with LEGO color
bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
bsdf.inputs["Base Color"].default_value = (args.color_r, args.color_g, args.color_b, 1.0)
bsdf.inputs["Roughness"].default_value = 0.3
bsdf.inputs["Metallic"].default_value = 0.0

output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

# Assign material to object
if imported_obj.data.materials:
    imported_obj.data.materials[0] = mat
else:
    imported_obj.data.materials.append(mat)

print(f"[BrickScan Renderer] Applied color: RGB({args.color_r:.2f}, {args.color_g:.2f}, {args.color_b:.2f})")

# ==============================================================================
# 5. GET BOUNDING BOX AND CALCULATE CAMERA DISTANCE
# ==============================================================================

def get_object_dimensions(obj):
    """Calculate bounding box of object"""
    if not obj.data.vertices:
        return Vector((1, 1, 1))

    vertices = [obj.matrix_world @ v.co for v in obj.data.vertices]

    min_coord = Vector(vertices[0])
    max_coord = Vector(vertices[0])

    for v in vertices:
        for i in range(3):
            min_coord[i] = min(min_coord[i], v[i])
            max_coord[i] = max(max_coord[i], v[i])

    return max_coord - min_coord

# Center object at origin
imported_obj.location = (0, 0, 0)
bpy.context.view_layer.update()

dimensions = get_object_dimensions(imported_obj)
max_dim = max(dimensions)
camera_distance = max_dim * 2.0  # Fit with some margin

print(f"[BrickScan Renderer] Object dimensions: {dimensions}, camera distance: {camera_distance:.2f}")

# ==============================================================================
# 6. SETUP CAMERA
# ==============================================================================

camera = bpy.data.cameras.new("Camera")
camera.lens = 50  # mm focal length
camera_obj = bpy.data.objects.new("Camera", camera)
bpy.context.collection.objects.link(camera_obj)
scene.camera = camera_obj

# ==============================================================================
# 7. SETUP LIGHTING (3-POINT)
# ==============================================================================

def create_light(name, light_type, location, energy, color=(1, 1, 1)):
    """Helper to create a light"""
    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_data.energy = energy
    light_obj = bpy.data.objects.new(name, light_data)
    light_obj.location = location
    bpy.context.collection.objects.link(light_obj)

    # Set light color
    if hasattr(light_data, 'color'):
        light_data.color = color

    return light_obj

# Key light (warm, strong)
key_light = create_light("KeyLight", "SUN", (3, 4, 5), 2.5, (1.0, 0.95, 0.8))

# Fill light (cool, soft)
fill_light = create_light("FillLight", "SUN", (-2, 1, 3), 1.0, (0.8, 0.9, 1.0))

# Rim light (white, back)
rim_light = create_light("RimLight", "SUN", (0, -2, 4), 1.5, (1.0, 1.0, 1.0))

print("[BrickScan Renderer] 3-point lighting setup complete")

# ==============================================================================
# 8. RENDER LOOP WITH MULTIPLE CAMERA ANGLES
# ==============================================================================

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

elevation_angles = [-20, 10, 30]  # degrees
num_azimuths = args.num_angles

# Domain-randomization strength (0 = none, 1 = default). Scales per-shot jitter
# of lighting, camera distance, and background brightness. Deterministic per
# (part, color) via a seeded RNG so reruns reproduce the same gallery views.
dr = max(0.0, float(args.dr_strength))
dr_rng = random.Random(hash((args.part_num, args.color_id)) & 0xFFFFFFFF)

# Capture the base (key/fill/rim) light energies so we can jitter around them.
_base_energies = {lo.name: lo.data.energy for lo in (key_light, fill_light, rim_light)}

csv_path = args.index_csv
if csv_path is None:
    csv_path = output_dir.parent / "index.csv"

rendered_count = 0

for elev_idx, elevation_deg in enumerate(elevation_angles):
    elevation_rad = math.radians(elevation_deg)

    # Divide azimuths evenly across 360 degrees
    azimuths = [360 * i / num_azimuths for i in range(num_azimuths)]

    for az_idx, azimuth_deg in enumerate(azimuths):
        azimuth_rad = math.radians(azimuth_deg)

        # Per-shot camera-distance jitter (domain randomization)
        dist = camera_distance * (1.0 + dr * dr_rng.uniform(-0.12, 0.12))

        # Calculate camera position (spherical coordinates)
        cam_x = dist * math.cos(elevation_rad) * math.cos(azimuth_rad)
        cam_y = dist * math.cos(elevation_rad) * math.sin(azimuth_rad)
        cam_z = dist * math.sin(elevation_rad)

        camera_obj.location = (cam_x, cam_y, cam_z)

        # Point camera at object origin
        direction = -camera_obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera_obj.rotation_euler = rot_quat.to_euler()

        bpy.context.view_layer.update()

        # ── Domain randomization: background brightness + per-light jitter ──
        bg_node = world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs["Strength"].default_value = 1.0 + dr * dr_rng.uniform(-0.2, 0.4)
        for _lo in (key_light, fill_light, rim_light):
            base = _base_energies[_lo.name]
            _lo.data.energy = base * (1.0 + dr * dr_rng.uniform(-0.3, 0.3))

        # Filename: include elevation so the 3 elevations don't overwrite each
        # other. angle_idx encodes elevation*num_azimuths + azimuth → unique.
        angle_idx = elev_idx * num_azimuths + az_idx
        filename = f"{args.part_num}_{args.color_id}_{angle_idx:04d}.png"
        filepath = output_dir / filename

        scene.render.filepath = str(filepath)

        # Render
        bpy.ops.render.render(write_still=True)
        print(f"[BrickScan Renderer] Rendered: {filename}")

        # Log to CSV
        if csv_path and not os.path.exists(csv_path):
            # Create CSV with headers if it doesn't exist
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["image_path", "part_num", "color_id", "color_name", "color_r", "color_g", "color_b"])

        if csv_path:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                rel_path = str(filepath.relative_to(output_dir.parent.parent)) if output_dir.parent.parent in filepath.parents else str(filepath)
                writer.writerow([
                    rel_path,
                    args.part_num,
                    args.color_id,
                    args.color_name,
                    f"{args.color_r:.4f}",
                    f"{args.color_g:.4f}",
                    f"{args.color_b:.4f}"
                ])

        rendered_count += 1

print(f"[BrickScan Renderer] Completed! Rendered {rendered_count} images.")
