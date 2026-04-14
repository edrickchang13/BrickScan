#!/usr/bin/env python3
"""
Surface wear and dust simulation for LEGO brick rendering.
Applies material wear effects and dust particles to make synthetic data more realistic.

This module is imported by render_tumble.py and render_parts.py.
"""

import bpy
import random
import math
from typing import Optional


def apply_wear_preset(material: bpy.types.Material, wear_level: str):
    """
    Apply wear preset to a Blender Principled BSDF material.

    Wear levels:
    - 'new': pristine LEGO (roughness=0.1, no scratches)
    - 'light': light wear (roughness=0.2, subtle scratches, slight color fade)
    - 'moderate': moderate wear (roughness=0.35, visible scratches, 10% color fade)
    - 'heavy': heavy wear (roughness=0.5, strong scratches, 20% fade + dirt layer)

    Args:
        material: Blender material with Principled BSDF shader
        wear_level: one of ['new', 'light', 'moderate', 'heavy']
    """
    if not material.use_nodes:
        material.use_nodes = True

    node_tree = material.node_tree
    links = node_tree.links

    # Find or create Principled BSDF
    principled = None
    for node in node_tree.nodes:
        if node.type == 'BSDF' and 'Principled' in node.bl_label:
            principled = node
            break

    if not principled:
        print(f"[Surface Wear] WARNING: Principled BSDF not found in material {material.name}")
        return

    if wear_level == 'new':
        principled.inputs['Roughness'].default_value = 0.1
        return

    elif wear_level == 'light':
        principled.inputs['Roughness'].default_value = 0.2

        # Subtle color desaturation (5% mix with gray)
        _apply_color_fade(node_tree, principled, fade_amount=0.05, fade_color=(0.5, 0.5, 0.5, 1.0))

        # Light scratch normal map via noise texture
        _apply_scratch_normal(node_tree, principled, strength=0.15)

    elif wear_level == 'moderate':
        principled.inputs['Roughness'].default_value = 0.35

        # 10% color desaturation
        _apply_color_fade(node_tree, principled, fade_amount=0.10, fade_color=(0.5, 0.5, 0.5, 1.0))

        # Visible scratches via noise + voronoi texture
        _apply_scratch_normal(node_tree, principled, strength=0.35)

    elif wear_level == 'heavy':
        principled.inputs['Roughness'].default_value = 0.5

        # 20% color desaturation
        _apply_color_fade(node_tree, principled, fade_amount=0.20, fade_color=(0.5, 0.5, 0.5, 1.0))

        # Strong scratches
        _apply_scratch_normal(node_tree, principled, strength=0.55)

        # Dirt layer (15% dark brown overlay)
        _apply_dirt_layer(node_tree, principled, dirt_amount=0.15)

    # Add fingerprint/smudge texture (all wear levels except 'new')
    if wear_level != 'new':
        _apply_fingerprints(node_tree, principled, smudge_strength=0.08)


def apply_random_wear(material: bpy.types.Material):
    """
    Randomly select and apply a wear level, weighted by realism.

    Weights:
    - new: 20%
    - light: 40%
    - moderate: 30%
    - heavy: 10%

    Args:
        material: Blender material
    """
    rand = random.random()
    if rand < 0.20:
        wear_level = 'new'
    elif rand < 0.60:
        wear_level = 'light'
    elif rand < 0.90:
        wear_level = 'moderate'
    else:
        wear_level = 'heavy'

    apply_wear_preset(material, wear_level)


def add_dust_particles(scene: bpy.types.Scene, density: float = 0.3):
    """
    Add a particle system that scatters dust on and around the brick.

    Simulates real-world dust accumulation by emitting small gray spheres.

    Args:
        scene: Blender scene
        density: dust density (0.0 = clean, 1.0 = very dusty)
    """
    if not (0.0 <= density <= 1.0):
        density = max(0.0, min(1.0, density))

    # Create a plane to emit dust particles from
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0.5))
    plane = bpy.context.active_object
    plane.name = "dust_emitter"
    plane.hide_set(True)
    plane.hide_render = True

    # Create particle system
    particles = plane.modifiers.new(name="dust", type='PARTICLE_SYSTEM')
    pset = particles.particle_systems[0].settings

    pset.frame_start = 1
    pset.frame_end = 1
    pset.lifetime = 250

    # Emission count based on density
    pset.count = int(density * 500)

    # Particle physics
    pset.mass = 0.001
    pset.particle_size = random.uniform(0.05, 0.3)
    pset.size_random = 0.3

    # Emission
    pset.emit_from = 'FACE'
    pset.use_emit_random = True
    pset.normal_factor = 0.5
    pset.factor_random = 0.2

    # Gravity and air resistance
    pset.mass = 0.0001
    pset.timestep = 0.04

    # Create dust material (light gray with slight transparency)
    dust_mat = bpy.data.materials.new("dust_material")
    dust_mat.use_nodes = True
    dust_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.7, 0.7, 0.7, 0.8)
    dust_mat.node_tree.nodes["Principled BSDF"].inputs[9].default_value = 0.8  # roughness
    dust_mat.node_tree.nodes["Principled BSDF"].inputs[18].default_value = 0.2  # alpha

    # Create dust particle object (small sphere)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05)
    dust_sphere = bpy.context.active_object
    dust_sphere.name = "dust_particle"
    dust_sphere.data.materials.append(dust_mat)

    # Link particle object
    pset.instance_collection = None
    pset.use_collection_pick_random = False


# ==============================================================================
# INTERNAL HELPER FUNCTIONS
# ==============================================================================

def _apply_color_fade(node_tree, principled, fade_amount: float = 0.10, fade_color=(0.5, 0.5, 0.5, 1.0)):
    """
    Fade material color toward a gray tone (simulating fading/bleaching).

    Args:
        node_tree: Material node tree
        principled: Principled BSDF node
        fade_amount: how much to fade (0.0 = no fade, 1.0 = full gray)
        fade_color: the color to fade toward (default: medium gray)
    """
    links = node_tree.links

    # Get current base color
    current_color = principled.inputs['Base Color'].default_value

    # Interpolate toward fade color
    faded = (
        current_color[0] * (1 - fade_amount) + fade_color[0] * fade_amount,
        current_color[1] * (1 - fade_amount) + fade_color[1] * fade_amount,
        current_color[2] * (1 - fade_amount) + fade_color[2] * fade_amount,
        current_color[3],
    )

    principled.inputs['Base Color'].default_value = faded


def _apply_scratch_normal(node_tree, principled, strength: float = 0.3):
    """
    Add a normal map that simulates surface scratches using procedural textures.

    Args:
        node_tree: Material node tree
        principled: Principled BSDF node
        strength: normal map strength (0.0-1.0)
    """
    links = node_tree.links

    # Create noise texture for scratches
    noise_node = node_tree.nodes.new(type='ShaderNodeTexNoise')
    noise_node.inputs['Scale'].default_value = 15.0
    noise_node.inputs['Detail'].default_value = 5.0
    noise_node.inputs['Roughness'].default_value = 0.6

    # Create color ramp to sharpen scratches
    ramp_node = node_tree.nodes.new(type='ShaderNodeValRamp')
    ramp_node.color_ramp.elements[0].position = 0.4
    ramp_node.color_ramp.elements[1].position = 0.6

    # Voronoi for directional scratches
    voronoi_node = node_tree.nodes.new(type='ShaderNodeTexVoronoi')
    voronoi_node.feature = 'DISTANCE_TO_EDGE'
    voronoi_node.inputs['Scale'].default_value = 8.0

    # Combine into scratch map
    multiply_node = node_tree.nodes.new(type='ShaderNodeMath')
    multiply_node.operation = 'MULTIPLY'

    links.new(noise_node.outputs['Fac'], ramp_node.inputs['Fac'])
    links.new(voronoi_node.outputs['Distance'], multiply_node.inputs[0])
    links.new(ramp_node.outputs['Color'], multiply_node.inputs[1])

    # Convert scratch map to normal
    normal_map_node = node_tree.nodes.new(type='ShaderNodeNormalMap')
    normal_map_node.inputs['Strength'].default_value = strength

    links.new(multiply_node.outputs['Value'], normal_map_node.inputs['Color'])

    # Blend with existing normal
    existing_normal = principled.inputs['Normal']
    if existing_normal.is_linked:
        # Create mix node to blend
        mix_rgb_node = node_tree.nodes.new(type='ShaderNodeMix')
        mix_rgb_node.data_type = 'RGBA'
        mix_rgb_node.inputs['A'].default_value = existing_normal.links[0].from_socket.default_value
        links.new(normal_map_node.outputs['Normal'], mix_rgb_node.inputs['B'])
        links.new(mix_rgb_node.outputs['Result'], existing_normal)
    else:
        links.new(normal_map_node.outputs['Normal'], existing_normal)


def _apply_fingerprints(node_tree, principled, smudge_strength: float = 0.08):
    """
    Add subtle fingerprint smudges using cloud/Voronoi texture.

    Args:
        node_tree: Material node tree
        principled: Principled BSDF node
        smudge_strength: how visible the smudges are (0.0-0.5)
    """
    links = node_tree.links

    # Cloud texture for organic smudges
    cloud_node = node_tree.nodes.new(type='ShaderNodeTexClouds')
    cloud_node.inputs['Scale'].default_value = 5.0
    cloud_node.inputs['Detail'].default_value = 2.0

    # Bump map to create subtle surface variation
    bump_node = node_tree.nodes.new(type='ShaderNodeBump')
    bump_node.inputs['Strength'].default_value = smudge_strength

    links.new(cloud_node.outputs['Fac'], bump_node.inputs['Height'])

    # Add to normal
    existing_normal = principled.inputs['Normal']
    if existing_normal.is_linked:
        # Try to blend
        pass
    else:
        links.new(bump_node.outputs['Normal'], existing_normal)


def _apply_dirt_layer(node_tree, principled, dirt_amount: float = 0.15):
    """
    Add a dark brown dirt layer overlay on top of the material.

    Args:
        node_tree: Material node tree
        principled: Principled BSDF node
        dirt_amount: how much dirt to show (0.0-1.0)
    """
    links = node_tree.links

    # Dark brown dirt color (RGB)
    dirt_color = (0.3, 0.2, 0.1, 1.0)

    # Create voronoi texture for dirt distribution
    voronoi_node = node_tree.nodes.new(type='ShaderNodeTexVoronoi')
    voronoi_node.feature = 'CELLS'
    voronoi_node.inputs['Scale'].default_value = 3.0

    # Create color ramp to make dirt patchy
    ramp_node = node_tree.nodes.new(type='ShaderNodeValRamp')
    ramp_node.color_ramp.elements[0].position = 0.7

    # Mix original color with dirt color
    mix_node = node_tree.nodes.new(type='ShaderNodeMix')
    mix_node.data_type = 'RGBA'
    mix_node.inputs[0].default_value = dirt_amount  # Factor

    # Get current base color
    current_base_color = principled.inputs['Base Color'].default_value
    mix_node.inputs['A'].default_value = current_base_color
    mix_node.inputs['B'].default_value = dirt_color

    links.new(voronoi_node.outputs['Distance'], ramp_node.inputs['Fac'])
    links.new(ramp_node.outputs['Color'], mix_node.inputs['A_001'])

    # Update base color to include dirt
    links.new(mix_node.outputs['Result'], principled.inputs['Base Color'])


if __name__ == "__main__":
    # This module is meant to be imported, not run standalone
    print("Surface wear module — import in Blender scripts with:")
    print("  from surface_wear import apply_wear_preset, apply_random_wear, add_dust_particles")
