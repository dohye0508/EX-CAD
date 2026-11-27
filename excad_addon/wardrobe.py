# SPDX-License-Identifier: GPL-2.0-or-later
"""Evaluated, rig-aware hair, clothing and portrait details for MPFB basemeshes."""

from contextlib import contextmanager
import gzip
import math
from pathlib import Path

import bmesh
import bpy
from mathutils import Quaternion, Vector

from .character_stylizer import _pbr_mat, _vgroup_center
from .face_fit import set_targets
from .human_assets import equip_asset
from .human_engine import _import_mpfb

SKINS = {"FAIR_IVORY": (0.80, 0.54, 0.42), "WARM_BEIGE": (0.60, 0.34, 0.22),
         "TAN": (0.36, 0.18, 0.095), "DEEP": (0.15, 0.067, 0.038)}
HAIR = {"SHORT": "short02", "BOB": "bob01", "PONYTAIL": "ponytail01", "LONG_STRAIGHT": "long01",
        "LONG_WAVY": "long01", "CURTAIN_BANGS": "long01"}
CLOTHES = {"TSHIRT": ("shirts", "toigo_basic_tucked_t-shirt"), "PANTS": ("pants", "toigo_wool_pants"),
           "SKIRT": ("skirts", "frankyaye_mini_skirt_01"), "SHOES": ("system", "shoes01")}
EXPRESSIONS = {"SMILE": {"mouth-corner-puller": 0.62, "mouth-upward-retraction": 0.24},
               "SERIOUS": {"mouth-compression": 0.38, "eyebrows-left-down": 0.25, "eyebrows-right-down": 0.25},
               "SURPRISED": {"mouth-open": 0.70, "eyebrows-left-up": 0.58, "eyebrows-right-up": 0.58,
                              "eye-left-opened-up": 0.30, "eye-right-opened-up": 0.30},
               "WINK": {"eye-left-closure": 0.92, "eyebrows-right-up": 0.16}}


@contextmanager
def rest_mesh(human):
    rig = human.find_armature()
    pose = rig.data.pose_position if rig else None
    disabled = [m for m in human.modifiers if m.show_viewport and m.type in {"SUBSURF", "MASK"}]
    try:
        if rig:
            rig.data.pose_position = "REST"
        for mod in disabled:
            mod.show_viewport = False
        bpy.context.view_layer.update()
        evaluated = human.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        try:
            if len(mesh.vertices) != len(human.data.vertices):
                raise ValueError("Character styling requires indexed MPFB topology")
            yield mesh
        finally:
            evaluated.to_mesh_clear()
    finally:
        for mod in disabled:
            mod.show_viewport = True
        if rig:
            rig.data.pose_position = pose
        bpy.context.view_layer.update()


def _attach(obj, human):
    obj.parent = human
    obj["excad_wearer"] = human.name
    obj.matrix_parent_inverse.identity()
    rig = human.find_armature()
    if rig and not any(m.type == "ARMATURE" for m in obj.modifiers):
        mod = obj.modifiers.new("Character pose", "ARMATURE")
        mod.object = rig
        mod.use_deform_preserve_volume = True
    return obj


def head_weight(obj, human):
    rig = human.find_armature()
    head = next((b.name for b in rig.data.bones if b.name.lower().split(":")[-1] in {"head","mixamorighead"}),
                None) if rig else "head"
    if not head:
        raise ValueError("Selected armature has no head bone")
    if rig:
        for group in list(obj.vertex_groups):
            if group.name in rig.data.bones and group.name != head:
                obj.vertex_groups.remove(group)
    group = obj.vertex_groups.get(head) or obj.vertex_groups.new(name=head)
    group.add(list(range(len(obj.data.vertices))), 1, "REPLACE")


def _object(human, name, points, faces, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(mat)
    for p in mesh.polygons:
        p.use_smooth = True
    return _attach(obj, human)


def _eye_centers(human, mesh):
    left = _vgroup_center(human, mesh, ("helper-l-eye", "l-eye"))
    right = _vgroup_center(human, mesh, ("helper-r-eye", "r-eye"))
    if left is None or right is None:
        raise ValueError("Character has no MPFB eye landmarks")
    return left, right


def _curtains(human, mesh):
    left, right = _eye_centers(human, mesh)
    center = (left+right)/2
    span = (left-right).length
    verts, faces = [], []
    for side in (-1, 1):
        for lock in range(8):
            start = len(verts)
            layer = lock/7
            for j in range(13):
                t = j/12
                x = side*span*(0.06 + 1.26*math.sin(t*math.pi/2) + 0.12*layer)
                y = span*(-0.17 - 0.35*math.sin(t*math.pi) + 0.52*t*t + 0.06*layer)
                z = span*(1.32-1.95*t + 0.12*layer)
                w = span*0.09*(0.05 + math.sin(math.pi*t)**0.7)
                p = center + Vector((x, y, z))
                verts.extend((p+Vector((-w,0,0)), p+Vector((0,-0.22*w,0)), p+Vector((w,0,0))))
                if j:
                    k = start+(j-1)*3
                    faces.extend(((k,k+1,k+4,k+3), (k+1,k+2,k+5,k+4)))
    obj = _object(human, "Curtain fringe", verts, faces, _pbr_mat("EXCAD Chestnut", (0.064,0.026,0.014), 0.40))
    head_weight(obj, human)
    return obj


def hair_style(human, style):
    obj = equip_asset(human, HAIR[style])
    head_weight(obj, human)
    result = [obj]
    if style == "LONG_WAVY":
        key = obj.shape_key_add(name="EXCAD waves", from_mix=True)
        heights = [p.co.z for p in key.data]
        high, low = max(heights), min(heights)
        for p in key.data:
            t = (high-p.co.z)/max(high-low, 1e-6)
            side = -1 if p.co.x < 0 else 1
            p.co.x += side*0.022*t*math.sin(12*t + 0.3*side)
            p.co.y += 0.015*t*math.sin(14*t + 2*p.co.x)
        key.value = 1
    if style == "CURTAIN_BANGS":
        with rest_mesh(human) as mesh:
            result.append(_curtains(human, mesh))
    return result


def transfer_weights(human, obj):
    rig = human.find_armature()
    if rig is None:
        return
    # Evaluated helper geometry can be closer than the skin. Transfer from the body's
    # visible skin with clothes masks temporarily disabled, in the same rest pose.
    masked = [m for m in human.modifiers if m.type == "MASK" and "helper" not in m.name.lower() and m.show_viewport]
    pose = rig.data.pose_position
    existing_groups = {group.name for group in obj.vertex_groups}
    try:
        rig.data.pose_position = "REST"
        for m in masked:
            m.show_viewport = False
        bpy.context.view_layer.update()
        with bpy.context.temp_override(object=human, active_object=human, selected_objects=[human, obj],
                                       selected_editable_objects=[human, obj]):
            result = bpy.ops.object.data_transfer(data_type="VGROUP_WEIGHTS", use_create=True,
                                                  vert_mapping="POLYINTERP_NEAREST", layers_select_src="ALL",
                                                  layers_select_dst="NAME", mix_mode="REPLACE")
        if "FINISHED" not in result:
            raise ValueError("Could not transfer garment weights")
        for group in list(obj.vertex_groups):
            if group.name not in existing_groups and group.name not in rig.data.bones:
                obj.vertex_groups.remove(group)
    finally:
        for m in masked:
            m.show_viewport = True
        rig.data.pose_position = pose
        bpy.context.view_layer.update()


def _tailored(human, style):
    with rest_mesh(human) as mesh:
        body = human.vertex_groups["body"].index
        indices = {v.index for v in human.data.vertices if any(g.group == body for g in v.groups)}
        lo = min(mesh.vertices[i].co.z for i in indices)
        h = max(mesh.vertices[i].co.z for i in indices)-lo
        center_y = min(mesh.vertices[i].co.y for i in indices if (mesh.vertices[i].co.z-lo)/h > 0.82) + 0.053*h
        faces, kept = [], {}
        for poly in mesh.polygons:
            if not all(i in indices for i in poly.vertices):
                continue
            x,y,z = poly.center
            height = (z-lo)/h
            width = abs(x)/h
            if height > 0.850 or height < 0.49 or width > 0.43:
                continue
            if width < 0.033 and height > 0.805:
                continue
            if y < center_y and style in {"VNECK_SWEATER", "JACKET"}:
                neckline = (0.725 if style == "VNECK_SWEATER" else 0.69) + 1.65*width
                if height > neckline or (style == "JACKET" and width < 0.0045):
                    continue
            faces.append(tuple(kept.setdefault(i,len(kept)) for i in poly.vertices))
        ease = {"JACKET":0.010, "HOODIE":0.009, "VNECK_SWEATER":0.005}[style]*h
        verts = [mesh.vertices[i].co + mesh.vertices[i].normal*ease for i in kept]
        if style == "HOODIE":
            # A folded hood follows the back of the neck, opening away from the shoulders.
            start = len(verts)
            for j in range(7):
                t = j/6
                for i in range(33):
                    a = math.pi*i/32
                    verts.append(Vector((h*(0.041+0.040*t)*math.cos(a),
                                         center_y + h*(0.035+0.065*t)*math.sin(a),
                                         lo+h*(0.84-0.08*t*math.sin(a)-0.013*t))))
            for j in range(6):
                for i in range(32):
                    a = start+j*33+i
                    faces.append((a,a+1,a+34,a+33))
        tones = {"VNECK_SWEATER": (0.63,0.52,0.39), "JACKET": (0.075,0.115,0.17), "HOODIE": (0.18,0.31,0.29)}
        obj = _object(human, style.title(), verts, faces, _pbr_mat("EXCAD "+style, tones[style], 0.78, sheen=0.28))
    transfer_weights(human, obj)
    sub = obj.modifiers.new("Finished neckline and cuffs", "SUBSURF")
    sub.levels = sub.render_levels = 1
    sol = obj.modifiers.new("Fabric thickness", "SOLIDIFY")
    sol.thickness, sol.offset = max(0.001, h*0.002), 1
    return obj


def garment(human, style):
    if style in CLOTHES:
        pack, name = CLOTHES[style]
        obj = equip_asset(human, name, pack)
        transfer_weights(human, obj)
        return obj
    if style not in {"VNECK_SWEATER","HOODIE","JACKET"}:
        raise ValueError(f"Unknown garment {style}")
    return _tailored(human, style)


def skin(human, tone, lip_tint):
    color = SKINS[tone]
    mat = _pbr_mat(f"{human.name} Portrait Skin", color, roughness=0.44)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Subsurface Weight"].default_value = 0.16
    bsdf.inputs["Subsurface Radius"].default_value = (1.0,0.43,0.23)
    bsdf.inputs["Subsurface Scale"].default_value = 0.0032
    attr = human.data.color_attributes.get("EXCAD Lips")
    if attr is None:
        attr = human.data.color_attributes.new("EXCAD Lips", "FLOAT_COLOR", "POINT")
        directory = Path(_import_mpfb()[3].get_mpfb_data("targets"))/"mouth"
        for lip in ("upperlip","lowerlip"):
            with gzip.open(directory/f"mouth-{lip}-volume-incr.target.gz","rt") as inp:
                for line in inp:
                    cols = line.split()
                    if len(cols) == 4 and cols[0].isdigit():
                        weight = min(1, math.sqrt(sum(float(v)**2 for v in cols[1:]))*30)
                        item = attr.data[int(cols[0])]
                        w = max(item.color[0], weight)
                        item.color = (w,w,w,1)
    nodes = mat.node_tree.nodes
    mask = nodes.get("Lip mask") or nodes.new("ShaderNodeVertexColor")
    mask.name, mask.layer_name = "Lip mask", attr.name
    mix = nodes.get("Lip tint") or nodes.new("ShaderNodeMixRGB")
    mix.name = "Lip tint"
    mix.inputs[1].default_value = (*color,1)
    mix.inputs[2].default_value = (*lip_tint,1)
    mat.node_tree.links.new(mask.outputs["Color"],mix.inputs[0])
    mat.node_tree.links.new(mix.outputs[0],bsdf.inputs["Base Color"])
    if human.data.materials:
        human.data.materials[0] = mat
    else:
        human.data.materials.append(mat)


def eyes(human, iris_color, iris_size):
    if not 0.2 <= iris_size <= 0.75:
        raise ValueError("iris_size is the fraction of eyeball radius, from 0.2 to 0.75")
    white = _pbr_mat("EXCAD Sclera", (0.82,0.84,0.80), 0.18)
    iris = _pbr_mat(human.name+" Iris", iris_color, 0.18)
    pupil = _pbr_mat("EXCAD Pupil", (0.005,0.006,0.007), 0.1)
    with rest_mesh(human) as mesh:
        centers = _eye_centers(human, mesh)
    radius = (centers[0]-centers[1]).length*0.194
    result = []
    for center in centers:
        # Latitude rings around the gaze axis align the pupil and iris boundaries.
        angles = [0, math.asin(iris_size*0.43), math.asin(iris_size)]
        edge = angles[-1]
        angles.extend(edge+(math.pi-edge)*j/24 for j in range(1,25))
        coords = [center+Vector((0,-radius,0))]
        faces, slots = [], []
        n = 40
        for j, theta in enumerate(angles[1:-1]):
            for i in range(n):
                phi = math.tau*i/n
                coords.append(center+radius*Vector((math.sin(theta)*math.cos(phi),-math.cos(theta),
                                                    math.sin(theta)*math.sin(phi))))
                a,b = 1+j*n+i, 1+j*n+(i+1)%n
                faces.append((0,a,b) if j == 0 else (a-n,a,b,b-n))
                slots.append(2 if j == 0 else 1 if j == 1 else 0)
        end = len(coords)
        coords.append(center+Vector((0,radius,0)))
        for i in range(n):
            faces.append((end-n+i,end,end-n+(i+1)%n))
            slots.append(0)
        obj = _object(human, "Portrait eye", coords, faces, white)
        obj.data.materials.append(iris)
        obj.data.materials.append(pupil)
        for p, slot in zip(obj.data.polygons, slots):
            p.material_index = slot
        head_weight(obj, human)
        result.append(obj)
    return result


def expression(human, preset):
    # The bundled facial units use a shared topology; this is an expression library,
    # independent of portrait appearance and of the subject in a reference image.
    targets = {"expression/units/caucasian/"+name: value for name,value in EXPRESSIONS[preset].items()}
    set_targets(human, targets, "excad_expression")


def hand_pose(human, preset):
    rig = human.find_armature()
    if not rig:
        raise ValueError("Finger poses require a character armature")
    amount = {"OPEN":0.0, "RELAXED":0.22, "FIST":1.15, "POINT":1.02}[preset]
    changes = 0
    for side in ("l","r"):
        for finger_id, finger in enumerate(("thumb","index","middle","ring","pinky"),1):
            for joint in range(1,4):
                candidates = (f"{finger}_{joint:02}_{side}", f"finger{finger_id}-{joint}.{side.upper()}",
                              f"mixamorig:{'Left' if side=='l' else 'Right'}Hand{finger.title()}{joint}")
                bone = next((rig.pose.bones[n] for n in candidates if n in rig.pose.bones), None)
                if not bone:
                    continue
                angle = 0 if preset == "POINT" and finger == "index" else amount*(0.52 if finger=="thumb" else 1)
                bone.rotation_mode = "QUATERNION"
                bone.rotation_quaternion = Quaternion((1,0,0), angle)
                changes += 1
    if not changes:
        raise ValueError("Armature has no recognized finger bones")
    bpy.context.view_layer.update()
    return changes
