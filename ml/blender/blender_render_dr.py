#!/usr/bin/env python3
"""
Blender GPU rendering script for LEGO part synthetic data generation.
Runs INSIDE Blender: blender --background --python render_parts.py -- <args>

Includes domain randomization for improved generalization:
- Random HDRI backgrounds or solid colors
- Per-angle lighting jitter (color temp, energy, azimuth)
- Per-part material variation (roughness, specular, color noise)
- Per-angle scale jitter
- Extended elevation angles when domain randomization is enabled
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
parser.add_argument("--num-angles", type=int, default=36, help="Number of camera angles (default: 36)")
parser.add_argument("--resolution", type=int, default=224, help="Output resolution (default: 224)")
parser.add_argument("--index-csv", default=None, help="Path to index.csv for appending rows")
parser.add_argument("--domain-randomize", action="store_true", default=False, help="Enable domain randomization")
parser.add_argument("--hdri-dir", type=str, default=None, help="Path to directory of .hdr/.exr files for backgrounds")
args = parser.parse_args(argv)

print(f"[BrickScan Renderer] Starting render for part {args.part_num}, color {args.color_name}")
if args.domain_randomize:
    print(f"[BrickScan Renderer] Domain randomization ENABLED")
    if args.hdri_dir:
        print(f"[BrickScan Renderer] HDRI dir: {args.hdri_dir}")

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
scene.cycles.device = "GPU"          # GPU rendering
scene.cycles.samples = 128
scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.cycles.use_denoising = True

# ── Cross-platform GPU setup ──────────────────────────────────────────────────
# macOS Apple Silicon → Metal
# Linux NVIDIA (GB10 Blackwell) → OptiX → CUDA fallback
import platform as _platform
import sys as _sys

prefs = bpy.context.preferences
cycles_prefs = prefs.addons["cycles"].preferences

_is_mac = _platform.system() == "Darwin"
_is_linux = _platform.system() == "Linux"

# Pick device type priority based on platform
if _is_mac:
    _device_priority = ("METAL",)
elif _is_linux:
    _device_priority = ("OPTIX", "CUDA", "HIP")
else:
    _device_priority = ("CUDA", "OPTIX")

_gpu_configured = False
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

if not _gpu_configured:
    print("[BrickScan Renderer] ⚠ No GPU found — falling back to CPU rendering")
    scene.cycles.device = "CPU"

# Output settings
scene.render.resolution_x = args.resolution
scene.render.resolution_y = args.resolution
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = True

print(f"[BrickScan Renderer] GPU rendering configured (samples: {scene.cycles.samples})")

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
# 4. APPLY MATERIAL WITH DOMAIN RANDOMIZATION
# ==============================================================================

# Create or get material
mat_name = f"LEGO_Color_{args.color_id}"
if mat_name in bpy.data.materials:
    mat = bpy.data.materials[mat_name]
else:
    mat = bpy.data.materials.new(name=mat_name)

mat.use_nodes = True
mat.node_tree.nodes.clear()

# Apply material variation if domain randomization is enabled
base_color_r = args.color_r
base_color_g = args.color_g
base_color_b = args.color_b

if args.domain_randomize:
    # Add slight noise to base color (±0.03 per channel, clamped 0-1)
    color_noise_r = random.uniform(-0.03, 0.03)
    color_noise_g = random.uniform(-0.03, 0.03)
    color_noise_b = random.uniform(-0.03, 0.03)
    base_color_r = max(0.0, min(1.0, base_color_r + color_noise_r))
    base_color_g = max(0.0, min(1.0, base_color_g + color_noise_g))
    base_color_b = max(0.0, min(1.0, base_color_b + color_noise_b))

    material_roughness = random.uniform(0.15, 0.50)
    material_specular = random.uniform(0.3, 0.7)
else:
    material_roughness = 0.3
    material_specular = 0.5

# Create Principled BSDF with LEGO color
bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
bsdf.inputs["Base Color"].default_value = (base_color_r, base_color_g, base_color_b, 1.0)
bsdf.inputs["Roughness"].default_value = material_roughness
bsdf.inputs["Specular"].default_value = material_specular
bsdf.inputs["Metallic"].default_value = 0.0

output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

# Assign material to object
if imported_obj.data.materials:
    imported_obj.data.materials[0] = mat
else:
    imported_obj.data.materials.append(mat)

print(f"[BrickScan Renderer] Applied color: RGB({base_color_r:.2f}, {base_color_g:.2f}, {base_color_b:.2f})")
if args.domain_randomize:
    print(f"[BrickScan Renderer] Material variation: roughness={material_roughness:.2f}, specular={material_specular:.2f}")

# ==============================================================================
# 5. SETUP BACKGROUND WITH DOMAIN RANDOMIZATION
# ==============================================================================

def load_hdri_files(hdri_dir):
    """Load list of .hdr and .exr files from directory."""
    if not hdri_dir or not os.path.isdir(hdri_dir):
        return []
    hdri_files = []
    for ext in ['*.hdr', '*.exr', '*.HDR', '*.EXR']:
        hdri_files.extend(Path(hdri_dir).glob(ext))
    return [str(f) for f in hdri_files]

def hex_to_rgb(hex_color):
    """Convert hex color string to RGB tuple (0-1)."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)

background_hdri_file = None
background_color = (0.05, 0.05, 0.05)
background_strength = 1.0

if args.domain_randomize:
    # Try to load HDRI files if directory is provided
    hdri_files = load_hdri_files(args.hdri_dir)

    if hdri_files:
        background_hdri_file = random.choice(hdri_files)
        background_strength = random.uniform(0.3, 1.5)
        print(f"[BrickScan Renderer] Using HDRI: {Path(background_hdri_file).name}, strength={background_strength:.2f}")
    else:
        # Use random solid background color
        color_choices = ['#FFFFFF', '#E8E8E8', '#2A2A2A', '#1A3A5A', '#4A2020', '#F5F0E8']
        background_color = hex_to_rgb(random.choice(color_choices))
        background_strength = 1.0
        print(f"[BrickScan Renderer] Using random solid background: RGB{background_color}")

# Set up world environment
bg_node = world.node_tree.nodes.get("Background")
if bg_node is None:
    bg_node = world.node_tree.nodes.new("ShaderNodeBackground")
    world_output = world.node_tree.nodes.get("World Output")
    if world_output:
        world.node_tree.links.new(bg_node.outputs["Background"], world_output.inputs["Surface"])

if background_hdri_file:
    # Load HDRI texture
    env_tex = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
    try:
        env_tex.image = bpy.data.images.load(background_hdri_file)
        world.node_tree.links.clear()
        world.node_tree.links.new(env_tex.outputs["Color"], bg_node.inputs["Color"])
        bg_node.inputs["Strength"].default_value = background_strength

        # Apply random Z-rotation to HDRI
        if len(world.node_tree.nodes) > 0:
            hdri_rotation = random.uniform(0, 360)
            # Note: rotation applied via mapping node (optional, for now just loaded)
    except Exception as e:
        print(f"[BrickScan Renderer] Failed to load HDRI {background_hdri_file}: {e}")
else:
    # Use solid background color
    bg_node.inputs[0].default_value = (background_color[0], background_color[1], background_color[2], 1.0)
    bg_node.inputs["Strength"].default_value = background_strength

print("[BrickScan Renderer] Background setup complete")

# ==============================================================================
# 6. GET BOUNDING BOX AND CALCULATE CAMERA DISTANCE
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
# 7. SETUP CAMERA
# ==============================================================================

camera = bpy.data.cameras.new("Camera")
camera.lens = 50  # mm focal length
camera_obj = bpy.data.objects.new("Camera", camera)
bpy.context.collection.objects.link(camera_obj)
scene.camera = camera_obj

# ==============================================================================
# 8. SETUP LIGHTING (3-POINT)
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

# Store base light configuration for later jitter
key_light = create_light("KeyLight", "SUN", (3, 4, 5), 2.5, (1.0, 0.95, 0.8))
fill_light = create_light("FillLight", "SUN", (-2, 1, 3), 1.0, (0.8, 0.9, 1.0))
rim_light = create_light("RimLight", "SUN", (0, -2, 4), 1.5, (1.0, 1.0, 1.0))

# Store base key light azimuth (radians)
key_light_base_azimuth = math.atan2(4, 3)  # atan2(y, x) from location (3, 4, z)

print("[BrickScan Renderer] 3-point lighting setup complete")

# ==============================================================================
# 9. RENDER LOOP WITH MULTIPLE CAMERA ANGLES AND DOMAIN RANDOMIZATION
# ==============================================================================

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Use extended elevation angles if domain randomization is enabled
if args.domain_randomize:
    elevation_angles = [-30, -15, 0, 15, 30, 45]  # 6 elevations
else:
    elevation_angles = [-20, 10, 30]  # 3 elevations

num_azimuths = args.num_angles

csv_path = args.index_csv
if csv_path is None:
    csv_path = output_dir.parent / "index.csv"

rendered_count = 0

for elevation_deg in elevation_angles:
    elevation_rad = math.radians(elevation_deg)

    # Divide azimuths evenly across 360 degrees
    azimuths = [360 * i / num_azimuths for i in range(num_azimuths)]

    for azimuth_deg in azimuths:
        # Per-angle lighting jitter (if domain randomization enabled)
        if args.domain_randomize:
            # Jitter key light azimuth
            jittered_key_azimuth = key_light_base_azimuth + math.radians(random.uniform(-40, 40))

            # Jitter key light energy
            key_energy = random.uniform(1.5, 4.0)

            # Jitter key light color temperature (warm to cool)
            color_temp_t = random.uniform(0, 1)
            warm = (1.0, 0.90, 0.75)
            cool = (0.80, 0.88, 1.0)
            key_color = (
                warm[0] + (cool[0] - warm[0]) * color_temp_t,
                warm[1] + (cool[1] - warm[1]) * color_temp_t,
                warm[2] + (cool[2] - warm[2]) * color_temp_t,
            )

            # Jitter fill light energy
            fill_energy = random.uniform(0.3, 1.5)

            # Jitter rim light energy
            rim_energy = random.uniform(0.5, 2.5)

            # Apply jittered lighting
            key_light.data.energy = key_energy
            key_light.data.color = key_color

            # Update key light position based on jittered azimuth
            key_distance = math.sqrt(3**2 + 4**2)  # preserve distance from base location
            key_light.location = (
                key_distance * math.cos(jittered_key_azimuth),
                key_distance * math.sin(jittered_key_azimuth),
                5
            )

            fill_light.data.energy = fill_energy
            rim_light.data.energy = rim_energy
        else:
            # Fixed lighting (existing behavior)
            key_light.data.energy = 2.5
            key_light.data.color = (1.0, 0.95, 0.8)
            key_light.location = (3, 4, 5)

            fill_light.data.energy = 1.0
            rim_light.data.energy = 1.5

        # Optionally add accent light (10% probability per angle if domain randomization enabled)
        accent_light = None
        if args.domain_randomize and random.random() < 0.1:
            accent_pos = (
                random.uniform(-3, 3),
                random.uniform(-3, 3),
                random.uniform(2, 6)
            )
            accent_energy = random.uniform(0.5, 2.0)
            accent_light = create_light("AccentLight", "AREA", accent_pos, accent_energy, (1.0, 1.0, 1.0))

        # Per-angle scale jitter
        if args.domain_randomize:
            scale_jitter = random.uniform(0.92, 1.08)
            imported_obj.scale = (scale_jitter, scale_jitter, scale_jitter)

        azimuth_rad = math.radians(azimuth_deg)

        # Calculate camera position (spherical coordinates)
        cam_x = camera_distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        cam_y = camera_distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        cam_z = camera_distance * math.sin(elevation_rad)

        camera_obj.location = (cam_x, cam_y, cam_z)

        # Point camera at object origin
        direction = -camera_obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera_obj.rotation_euler = rot_quat.to_euler()

        bpy.context.view_layer.update()

        # Vary background brightness slightly between shots (if not using HDRI)
        if not background_hdri_file and not args.domain_randomize:
            bg_node = world.node_tree.nodes.get("Background")
            if bg_node:
                bg_node.inputs["Strength"].default_value = random.uniform(0.8, 1.2)

        # Filename with angle index
        angle_idx = int(azimuth_deg / (360 / num_azimuths)) if num_azimuths > 0 else 0
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
                    f"{base_color_r:.4f}",
                    f"{base_color_g:.4f}",
                    f"{base_color_b:.4f}"
                ])

        # Clean up accent light if created
        if accent_light:
            bpy.data.objects.remove(accent_light, do_unlink=True)

        # Restore original scale if it was jittered
        if args.domain_randomize:
            imported_obj.scale = (1.0, 1.0, 1.0)

        rendered_count += 1

print(f"[BrickScan Renderer] Completed! Rendered {rendered_count} images.")
