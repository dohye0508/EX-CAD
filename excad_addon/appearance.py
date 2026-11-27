# SPDX-License-Identifier: GPL-2.0-or-later
"""Semantic finishes and a three-point light rig framed from the subject's evaluated bounds."""

import bpy
from mathutils import Vector

from .scene_layout import bounds

FINISHES = {
    "BRUSHED_METAL": ((0.55, 0.59, 0.64), 0.28, 0.92),
    "CERAMIC": ((0.66, 0.76, 0.72), 0.22, 0.0),
    "PLASTIC": ((0.13, 0.23, 0.31), 0.38, 0.0),
    "RUBBER": ((0.025, 0.029, 0.032), 0.74, 0.0),
    "WOOD": ((0.36, 0.20, 0.09), 0.55, 0.0),
    "FABRIC": ((0.22, 0.28, 0.32), 0.86, 0.0),
}


def set_material(obj, preset, color=None, roughness=None, metallic=None):
    obj = bpy.context.scene.objects[obj]
    renderable = {"MESH", "CURVE", "SURFACE", "FONT"}
    targets = (
        [obj] if obj.type in renderable else [child for child in obj.children_recursive if child.type in renderable]
    )
    if not targets:
        raise ValueError("The material target needs renderable geometry")
    base, r, m = FINISHES[preset]
    color = list(color if color is not None else base)
    if len(color) != 3 or any(not 0 <= v <= 1 for v in color):
        raise ValueError("Use three linear RGB channels between 0 and 1")
    roughness = r if roughness is None else roughness
    metallic = m if metallic is None else metallic
    if not 0 <= roughness <= 1 or not 0 <= metallic <= 1:
        raise ValueError("Roughness and metallic must be between 0 and 1")
    assigned = [_finish(target, preset, color, roughness, metallic) for target in targets]
    return {"material": assigned[0]} if len(assigned) == 1 else {"materials": assigned}


def _finish(obj, preset, color, roughness, metallic):
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    slots = obj.data.materials
    mat = slots[0] if slots and slots[0] and slots[0].get("excad_finish") and slots[0].users == 1 else None
    if mat is None:
        mat = bpy.data.materials.new(obj.name + " finish")
    mat["excad_finish"] = preset
    mat.use_nodes = True
    mat.diffuse_color = (*color, 1)
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    slots.clear()
    slots.append(mat)
    return mat.name


def light_studio(target, style="PRODUCT", power=1.0):
    if style not in {"PRODUCT", "PORTRAIT", "SOFT"} or power <= 0:
        raise ValueError("Use PRODUCT, PORTRAIT or SOFT with positive power")
    obj = bpy.context.scene.objects[target]
    bb = bounds(obj)
    center = Vector(bb.mean(0))
    extent = max(float(max(bb[1] - bb[0])), 0.01)
    lights = []
    for role, direction, energy, diameter in (
        ("Key", (-1.5, -2, 2.4), 420, 1.25),
        ("Fill", (2, -1, 1.3), 220, 1.6),
        ("Rim", (0.8, 1.5, 1.8), 330, 0.9),
    ):
        lamp = next(
            (
                o
                for o in bpy.context.scene.objects
                if o.type == "LIGHT" and o.get("excad_subject") == target and o.get("excad_light") == role
            ),
            None,
        )
        if lamp is None:
            data = bpy.data.lights.new(target + " " + role, "AREA")
            lamp = bpy.data.objects.new(data.name, data)
            bpy.context.collection.objects.link(lamp)
        lamp["excad_subject"], lamp["excad_light"] = target, role
        lamp.location = center + Vector(direction) * extent
        lamp.rotation_euler = (center - lamp.location).to_track_quat("-Z", "Y").to_euler()
        lamp.data.shape = "DISK"
        lamp.data.size = extent * diameter * (1.8 if style == "SOFT" else 1)
        lamp.data.energy = energy * extent**2 * power
        lamp.data.color = (1, 0.86, 0.74) if style == "PORTRAIT" and role == "Key" else (1, 1, 1)
        lights.append(lamp.name)
    return {"lights": lights}
