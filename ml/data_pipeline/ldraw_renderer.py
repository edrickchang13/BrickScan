"""
BrickScan LDraw Renderer
Blender script to generate synthetic LEGO training images from LDraw 3D models

This script runs inside Blender (headless) and renders LDraw .dat files from multiple
angles with randomized lighting, backgrounds, and LEGO colors to create synthetic
training data for piece recognition.

Usage (run inside Blender headless):
    blender --background --python ldraw_renderer.py -- \
        --part_file /path/to/ldraw/parts/3001.dat \
        --output_dir /path/to/output \
        --num_renders 80 \
        --colors all \
        --gpu

Or to render multiple parts:
    blender --background --python ldraw_renderer.py -- \
        --parts_dir /path/to/ldraw/parts \
        --output_dir /path/to/output \
        --num_renders 50 \
        --top_parts_file /path/to/top_parts.txt \
        --gpu

Command line arguments (passed after '--'):
    --part_file PATH          Single .dat file to render
    --parts_dir PATH          Directory of .dat files (renders all)
    --top_parts_file PATH     Text file with part numbers to render (one per line)
    --output_dir PATH         Output directory (required)
    --num_renders N           Renders per part (default: 60)
    --colors CHOICE           Color variation: all|common|none (default: common)
    --resolution N            Image resolution in pixels (default: 512)
    --engine CHOICE           EEVEE or CYCLES (default: EEVEE)
    --gpu                     Enable GPU rendering (only with EEVEE/CYCLES)
"""

import bpy
import sys
import os
import math
import random
import argparse
import json
from pathlib import Path
from mathutils import Vector, Euler
from datetime import datetime

# LEGO Official ABS Plastic Colors — **Linear RGB** values for Blender.
#
# Source: LDraw LDConfig.ldr (https://www.ldraw.org/library/official/LDConfig.ldr)
# Each VALUE hex is the official sRGB colour; the tuples below are the result of
# applying the IEC 61966-2-1 sRGB→linear formula:
#   linear = ((srgb + 0.055) / 1.055) ** 2.4   (for srgb > 0.04045)
#   linear = srgb / 12.92                        (otherwise)
#
# Blender's Principled BSDF Base Color input expects **linear** values.
# Passing raw sRGB hex-divided-by-255 floats causes incorrect colour rendering.
#
# Format: "Name": (r_linear, g_linear, b_linear)
# Corresponding LDraw CODE shown in comment for reference.

def _s2l(v):
    """sRGB channel → linear (IEC 61966-2-1)."""
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

def _hex_lin(h):
    """'#RRGGBB' → (r_lin, g_lin, b_lin)."""
    h = h.lstrip('#')
    return (_s2l(int(h[0:2],16)/255), _s2l(int(h[2:4],16)/255), _s2l(int(h[4:6],16)/255))

LEGO_COLORS = {
    # Code  Name               LDraw VALUE hex
    "White":              _hex_lin("#FFFFFF"),   #  15
    "Black":              _hex_lin("#05131D"),   #   0
    "Blue":               _hex_lin("#0055BF"),   #   1
    "Green":              _hex_lin("#257A3E"),   #   2
    "Dark_Turquoise":     _hex_lin("#00838F"),   #   3
    "Red":                _hex_lin("#C91A09"),   #   4
    "Dark_Pink":          _hex_lin("#C870A0"),   #   5
    "Brown":              _hex_lin("#583927"),   #   6
    "Light_Gray":         _hex_lin("#9BA19D"),   #   7
    "Dark_Gray":          _hex_lin("#6D6E5C"),   #   8
    "Light_Blue":         _hex_lin("#B4D2E3"),   #   9
    "Bright_Green":       _hex_lin("#4B9F4A"),   #  10
    "Medium_Turquoise":   _hex_lin("#55A5AF"),   #  11
    "Salmon":             _hex_lin("#F2705E"),   #  12
    "Pink":               _hex_lin("#FC97AC"),   #  13
    "Yellow":             _hex_lin("#F2CD37"),   #  14
    "Tan":                _hex_lin("#E4CD9E"),   #  19
    "Purple":             _hex_lin("#81007B"),   #  22
    "Dark_Red":           _hex_lin("#720E0F"),   #  59
    "Dark_Orange":        _hex_lin("#A95500"),   #  68 (approx Dark_Nougat)
    "Orange":             _hex_lin("#FE8A18"),   #  25
    "Dark_Blue":          _hex_lin("#003852"),   #  63
    "Dark_Green":         _hex_lin("#184632"),   #  80 (approx)
    "Lime":               _hex_lin("#BBE90B"),   #  27 (Lime = Yellowish Green)
    "Sand_Green":         _hex_lin("#789082"),   #  48
    "Sand_Blue":          _hex_lin("#6074A1"),   #  55
    "Dark_Tan":           _hex_lin("#958A73"),   #  69
    "Reddish_Brown":      _hex_lin("#582A12"),   #  70
    "Light_Bluish_Gray":  _hex_lin("#A0A5A9"),   #  71
    "Dark_Bluish_Gray":   _hex_lin("#6C6E68"),   #  72
    "Medium_Blue":        _hex_lin("#5A93DB"),   #  73
    "Medium_Azure":       _hex_lin("#36AEBF"),   #  322 (approx)
    "Magenta":            _hex_lin("#923978"),   #  26
    "Dark_Nougat":        _hex_lin("#AD6140"),   # 128
    "Nougat":             _hex_lin("#D09168"),   #  28
    "Medium_Nougat":      _hex_lin("#AA7D55"),   # 150
    "Bright_Pink":        _hex_lin("#E4ADC8"),   # 104
    "Medium_Lavender":    _hex_lin("#AC78BA"),   # 324
    "Lavender":           _hex_lin("#E1D5ED"),   # 325
    "Transparent_Clear":  _hex_lin("#EEEEEE"),   #  47 (alpha set separately)
    "Transparent_Red":    _hex_lin("#C91A09"),   #  36
    "Transparent_Yellow": _hex_lin("#F2CD37"),   #  46
    "Transparent_Blue":   _hex_lin("#0055BF"),   #  43
    "Transparent_Green":  _hex_lin("#84B68D"),   #  34
    "Chrome_Silver":      _hex_lin("#E0E0E0"),   #  383
    "Pearl_Dark_Gray":    _hex_lin("#575857"),   # 148
    "Flat_Silver":        _hex_lin("#898788"),   # 179
    "Pearl_Gold":         _hex_lin("#AA7F2E"),   # 297
}

# Background types with descriptive names
BACKGROUNDS = ["white", "light_gray", "beige", "dark_gray", "blue_gradient"]

# Metadata tracking for dataset
RENDER_LOG = []


def parse_args():
    """Parse command line arguments passed after '--' to Blender"""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="BrickScan LDraw Renderer for Blender")
    parser.add_argument("--part_file", type=str,
                       help="Single .dat LDraw part file to render")
    parser.add_argument("--parts_dir", type=str,
                       help="Directory containing .dat files (renders all)")
    parser.add_argument("--top_parts_file", type=str,
                       help="Text file with list of part numbers to render (one per line)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for rendered images")
    parser.add_argument("--num_renders", type=int, default=60,
                       help="Number of render variations per part")
    parser.add_argument("--colors", type=str, default="common",
                       choices=["all", "common", "none"],
                       help="Color variation: all=all LEGO colors, common=top 8, none=gray only")
    parser.add_argument("--resolution", type=int, default=512,
                       help="Output image resolution (pixels)")
    parser.add_argument("--engine", type=str, default="EEVEE",
                       choices=["EEVEE", "CYCLES"],
                       help="Render engine: EEVEE (fast) or CYCLES (photorealistic)")
    parser.add_argument("--gpu", action="store_true",
                       help="Enable GPU rendering")
    parser.add_argument("--samples", type=int, default=None,
                       help="Override render samples (EEVEE: 64, CYCLES: 128)")

    return parser.parse_args(argv)


def log_info(msg: str):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def clear_scene():
    """Remove all objects, lights, and cameras from the scene"""
    log_info("Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

    # Clean up orphaned data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def setup_render_engine(args):
    """Configure render engine, quality, and output settings"""
    scene = bpy.context.scene
    log_info(f"Setting up {args.engine} render engine...")

    # Configure engine
    if args.engine == "EEVEE":
        scene.render.engine = 'BLENDER_EEVEE_NEXT' if hasattr(scene, 'eevee') else 'BLENDER_EEVEE'
        eevee = scene.eevee

        # Quality settings for EEVEE
        samples = args.samples or 64
        eevee.taa_render_samples = samples
        eevee.use_soft_shadows = True
        eevee.use_ssr = True
        eevee.shadow_cube_size = '1024'
        eevee.use_gtao = True

    else:  # CYCLES
        scene.render.engine = 'CYCLES'
        cycles = scene.cycles

        samples = args.samples or 128
        cycles.samples = samples
        cycles.use_denoising = True
        cycles.denoiser = 'OPTIX'

        if args.gpu:
            cycles.device = 'GPU'
            prefs = bpy.context.preferences.addons['cycles'].preferences
            prefs.compute_device_type = 'CUDA'
            log_info("GPU acceleration enabled for CYCLES")

    # Output settings
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.compression = 95

    # Use square pixels
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0


def create_camera():
    """Create and configure perspective camera"""
    bpy.ops.object.camera_add(location=(0, 0, 5))
    camera = bpy.context.active_object
    camera.name = "MainCamera"

    bpy.context.scene.camera = camera

    camera.data.type = 'PERSP'
    camera.data.lens = 50
    camera.data.sensor_width = 36
    camera.data.clip_start = 0.1
    camera.data.clip_end = 1000

    return camera


def position_camera_orbit(camera, target_pos=(0, 0, 0), min_dist=3.0, max_dist=6.0):
    """
    Position camera at random orbit point looking at target position.
    Uses spherical coordinates to ensure good coverage of the part.
    """
    distance = random.uniform(min_dist, max_dist)

    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(math.pi / 6, math.pi / 2.2)

    x = distance * math.sin(phi) * math.cos(theta)
    y = distance * math.sin(phi) * math.sin(theta)
    z = distance * math.cos(phi)

    target = Vector(target_pos)
    cam_pos = Vector((x, y, z)) + target
    camera.location = cam_pos

    # Point camera at target
    direction = target - cam_pos
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def setup_3point_lighting():
    """
    Create randomized 3-point lighting setup:
    - Key light: main bright directional light
    - Fill light: secondary softer light to reduce shadows
    - Rim light: back light to separate object from background
    """
    lights = []

    # Key Light (main)
    bpy.ops.object.light_add(type='AREA')
    key = bpy.context.active_object
    key.name = "KeyLight"
    key.data.energy = random.uniform(200, 500)
    key.data.size = random.uniform(2, 5)

    key.location = Vector((
        random.uniform(2, 6),
        random.uniform(-3, 3),
        random.uniform(3, 8)
    ))

    direction_to_origin = -key.location.normalized()
    rot_quat = direction_to_origin.to_track_quat('-Z', 'Y')
    key.rotation_euler = rot_quat.to_euler()
    lights.append(key)

    # Fill Light (secondary, softer)
    bpy.ops.object.light_add(type='AREA')
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = random.uniform(80, 200)
    fill.data.size = random.uniform(4, 10)

    fill.location = Vector((
        random.uniform(-6, -2),
        random.uniform(-3, 3),
        random.uniform(1, 4)
    ))

    direction_to_origin = -fill.location.normalized()
    rot_quat = direction_to_origin.to_track_quat('-Z', 'Y')
    fill.rotation_euler = rot_quat.to_euler()
    lights.append(fill)

    # Rim Light (back light for separation)
    bpy.ops.object.light_add(type='SPOT')
    rim = bpy.context.active_object
    rim.name = "RimLight"
    rim.data.energy = random.uniform(100, 250)
    rim.data.spot_size = math.radians(45)
    rim.data.spot_blend = 0.5

    rim.location = Vector((
        random.uniform(-2, 2),
        random.uniform(-8, -4),
        random.uniform(0, 3)
    ))
    lights.append(rim)

    # Ambient world lighting
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs[1].default_value = random.uniform(0.3, 0.7)

    return lights


def setup_background(bg_type: str):
    """
    Set up background: world color + ground plane with material
    Creates a simple but effective background for piece photography simulation.
    """

    bg_colors = {
        "white": (0.98, 0.98, 0.98, 1.0),
        "light_gray": (0.75, 0.75, 0.75, 1.0),
        "beige": (0.9, 0.85, 0.75, 1.0),
        "dark_gray": (0.25, 0.25, 0.25, 1.0),
        "blue_gradient": (0.8, 0.85, 0.95, 1.0),
    }

    color = bg_colors.get(bg_type, bg_colors["white"])

    # Set world background
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    world.node_tree.nodes.clear()

    # Build world shader: simple background color
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    bg_shader = nodes.new('ShaderNodeBackground')
    bg_shader.inputs[0].default_value = color
    bg_shader.inputs[1].default_value = 1.0

    output = nodes.new('ShaderNodeOutputWorld')
    links.new(bg_shader.outputs[0], output.inputs[0])

    # Create ground plane
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, -0.01))
    plane = bpy.context.active_object
    plane.name = "GroundPlane"

    # Material: slightly textured surface
    mat = bpy.data.materials.new(name="GroundMaterial")
    mat.use_nodes = True
    mat.node_tree.nodes.clear()

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Principled BSDF for realistic ground
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.85
    principled.inputs["Specular"].default_value = 0.1

    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(principled.outputs[0], output.inputs[0])

    plane.data.materials.append(mat)


def import_ldraw_part(part_file: str) -> object:
    """
    Import an LDraw .dat file into Blender.

    Strategy:
    1. Try using LDraw addon if available (best quality)
    2. Fall back to manual DAT file parsing
    3. Fall back to simple cube if parsing fails
    """

    # Try LDraw addon import first
    try:
        addon_prefs = bpy.context.preferences.addons.get('io_scene_importldraw')
        if addon_prefs:
            ldraw_path = str(Path(part_file).parent.parent)
            bpy.ops.import_scene.importldraw(
                filepath=str(part_file),
                ldrawPath=ldraw_path
            )
            imported = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if imported:
                obj = imported[0]
                log_info(f"Imported {Path(part_file).name} using LDraw addon")
                return obj
    except Exception as e:
        log_info(f"LDraw addon not available: {e}")

    # Manual DAT file parsing
    vertices = []
    faces = []

    try:
        with open(part_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 1:
                    continue

                try:
                    line_type = parts[0]

                    # Triangle: type color x1 y1 z1 x2 y2 z2 x3 y3 z3
                    if line_type == '3' and len(parts) >= 11:
                        scale = 0.01
                        v_start = len(vertices)

                        for i in range(3):
                            x = float(parts[2 + i*3]) * scale
                            y = float(parts[3 + i*3]) * scale
                            z = float(parts[4 + i*3]) * scale
                            # LDraw Y-up to Blender Z-up conversion
                            vertices.append((x, -z, y))

                        faces.append((v_start, v_start+1, v_start+2))

                    # Quad: type color x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4
                    elif line_type == '4' and len(parts) >= 14:
                        scale = 0.01
                        v_start = len(vertices)

                        for i in range(4):
                            x = float(parts[2 + i*3]) * scale
                            y = float(parts[3 + i*3]) * scale
                            z = float(parts[4 + i*3]) * scale
                            vertices.append((x, -z, y))

                        faces.append((v_start, v_start+1, v_start+2, v_start+3))

                except (ValueError, IndexError) as e:
                    # Skip malformed lines
                    continue

        if not vertices:
            log_info(f"No geometry found in {Path(part_file).name}, using fallback cube")
            bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
            return bpy.context.active_object

        # Create mesh from parsed data
        mesh = bpy.data.meshes.new(name="LegoPart")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()

        # Validate and smooth normals
        mesh.validate(clean_customdata=False)

        obj = bpy.data.objects.new("LegoPart", mesh)
        bpy.context.collection.objects.link(obj)

        log_info(f"Parsed {Path(part_file).name}: {len(vertices)} vertices, {len(faces)} faces")
        return obj

    except Exception as e:
        log_info(f"Error parsing {part_file}: {e}. Using fallback geometry.")
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
        return bpy.context.active_object


def apply_lego_material(obj, color_name: str, color_rgb: tuple):
    """
    Apply a physically-accurate LEGO ABS plastic material.

    LEGO bricks are injection-molded ABS plastic with these properties:
    - Slightly glossy but not highly reflective
    - Some specularity from surface
    - Transparent variants have appropriate IOR
    - Matte finish on most parts
    """

    mat = bpy.data.materials.new(name=f"LEGO_{color_name}")
    mat.use_nodes = True
    mat.shadow_method = 'HASHED'

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Principled BSDF shader (PBR material)
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)

    # Base color
    principled.inputs["Base Color"].default_value = (*color_rgb, 1.0)

    # Material properties for ABS plastic
    principled.inputs["Specular"].default_value = 0.4
    principled.inputs["Roughness"].default_value = 0.2
    principled.inputs["Metallic"].default_value = 0.0

    # Transparent parts
    is_transparent = "trans" in color_name.lower() or "transparent" in color_name.lower()
    if is_transparent:
        principled.inputs["Transmission"].default_value = 0.9
        principled.inputs["Roughness"].default_value = 0.1
        principled.inputs["IOR"].default_value = 1.49
        principled.inputs["Alpha"].default_value = 0.7

    # Output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def center_object(obj):
    """
    Center object at world origin.
    Place bottom of bounding box at z=0 (on ground plane).
    """
    # Set origin to geometry bounds center
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    # Get bounding box after origin set
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_z = min(v.z for v in bbox)

    # Center in X/Y, ground in Z
    obj.location.x = 0
    obj.location.y = 0
    obj.location.z = -min_z + 0.01


def randomize_rotation(obj):
    """Apply random rotation to piece for viewing angle variation"""
    obj.rotation_euler = Euler((
        random.uniform(0, 2 * math.pi),
        random.uniform(0, 2 * math.pi),
        random.uniform(0, 2 * math.pi),
    ))


def render_to_file(output_path: str) -> bool:
    """
    Render current scene to PNG file.
    Returns True if successful, False otherwise.
    """
    try:
        scene = bpy.context.scene
        scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        log_info(f"Render error: {e}")
        return False


def render_part(
    part_file: str,
    output_dir: str,
    num_renders: int,
    color_mode: str,
    resolution: int,
    args
) -> int:
    """
    Render a single LDraw part with multiple variations.

    Variations:
    - Colors: LEGO official colors (subset based on color_mode)
    - Camera angles: Random orbit around part
    - Backgrounds: Random background type
    - Lighting: Random 3-point setup

    Output files: {part_num}/{index:05d}.png
    """

    part_num = Path(part_file).stem
    part_output_dir = Path(output_dir) / part_num
    part_output_dir.mkdir(parents=True, exist_ok=True)

    log_info(f"Rendering part {part_num}: {num_renders} images")

    # Determine color palette
    if color_mode == "all":
        colors = list(LEGO_COLORS.items())
    elif color_mode == "common":
        # Top 8 most common LEGO colors
        common = ["White", "Black", "Red", "Yellow", "Blue", "Green",
                  "Dark_Bluish_Gray", "Light_Bluish_Gray"]
        colors = [(c, LEGO_COLORS[c]) for c in common if c in LEGO_COLORS]
    else:  # none
        colors = [("Light_Bluish_Gray", LEGO_COLORS["Light_Bluish_Gray"])]

    renders_per_color = max(1, num_renders // len(colors))
    render_index = 0
    successful_renders = 0

    for color_idx, (color_name, color_rgb) in enumerate(colors):
        for angle_idx in range(renders_per_color):
            output_filename = f"{render_index:05d}.png"
            output_path = str(part_output_dir / output_filename)

            # Skip if already rendered
            if os.path.exists(output_path):
                render_index += 1
                successful_renders += 1
                continue

            try:
                # Fresh scene for each render
                clear_scene()
                setup_render_engine(args)

                # Scene setup
                bg_type = random.choice(BACKGROUNDS)
                setup_background(bg_type)
                setup_3point_lighting()

                # Import part
                obj = import_ldraw_part(part_file)
                if obj is None:
                    log_info(f"Failed to import {part_file}")
                    render_index += 1
                    continue

                center_object(obj)
                randomize_rotation(obj)
                apply_lego_material(obj, color_name, color_rgb)

                # Camera
                camera = create_camera()
                position_camera_orbit(camera)

                # Render
                success = render_to_file(output_path)

                if success:
                    successful_renders += 1
                    # Log metadata
                    metadata = {
                        "part_num": part_num,
                        "color": color_name,
                        "background": bg_type,
                        "angle_index": angle_idx,
                        "filename": output_filename,
                    }
                    RENDER_LOG.append(metadata)

                    if render_index % 10 == 0:
                        log_info(f"  {part_num}: {successful_renders}/{render_index} complete")

                render_index += 1

            except Exception as e:
                log_info(f"Error rendering {output_path}: {e}")
                render_index += 1
                continue

    log_info(f"Part {part_num}: {successful_renders}/{render_index} renders completed")
    return successful_renders


def save_metadata(output_dir: str, part_count: int, total_renders: int):
    """Save render metadata and statistics to JSON"""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "parts_rendered": part_count,
        "total_images": total_renders,
        "renders": RENDER_LOG,
    }

    metadata_file = Path(output_dir) / "render_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    log_info(f"Metadata saved to {metadata_file}")


def main():
    """Main entry point for render pipeline"""
    args = parse_args()

    log_info("=" * 70)
    log_info("BrickScan LDraw Renderer Starting")
    log_info(f"Output directory: {args.output_dir}")
    log_info(f"Render engine: {args.engine}")
    log_info(f"Colors: {args.colors}")
    log_info(f"Renders per part: {args.num_renders}")
    log_info("=" * 70)

    # Determine which parts to render
    parts_to_render = []

    if args.part_file:
        parts_to_render = [args.part_file]

    elif args.top_parts_file:
        with open(args.top_parts_file) as f:
            part_nums = [line.strip().split()[0] for line in f if line.strip()]
        parts_dir = args.parts_dir or Path(args.top_parts_file).parent / "parts"

        for part_num in part_nums:
            part_path = Path(parts_dir) / f"{part_num}.dat"
            if part_path.exists():
                parts_to_render.append(str(part_path))
            else:
                log_info(f"Part not found: {part_num}.dat in {parts_dir}")

    elif args.parts_dir:
        parts_to_render = sorted(list(Path(args.parts_dir).glob("*.dat")))
        parts_to_render = [str(p) for p in parts_to_render]

    if not parts_to_render:
        log_info("ERROR: No parts to render!")
        return

    log_info(f"Rendering {len(parts_to_render)} parts")

    # Render each part
    total_rendered = 0
    for i, part_file in enumerate(parts_to_render):
        log_info(f"\n[{i+1}/{len(parts_to_render)}] Processing {Path(part_file).stem}")
        try:
            count = render_part(
                part_file,
                args.output_dir,
                args.num_renders,
                args.colors,
                args.resolution,
                args
            )
            total_rendered += count
        except Exception as e:
            log_info(f"ERROR rendering {part_file}: {e}")

    # Save metadata
    save_metadata(args.output_dir, len(parts_to_render), total_rendered)

    log_info("\n" + "=" * 70)
    log_info(f"RENDER COMPLETE: {total_rendered} images from {len(parts_to_render)} parts")
    log_info("=" * 70)


if __name__ == "__main__":
    main()
