# SPDX-FileCopyrightText: 2026 EX-CAD Authors
# SPDX-License-Identifier: GPL-2.0-or-later

import math
import random

import bpy
import bmesh
from mathutils import Matrix, Vector


def _mat(name: str, color, roughness=0.5, metallic=0.0, emission=None, emission_strength=0.0):
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name=name)
    m.use_nodes = True
    m.diffuse_color = (*color, 1.0)
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = float(roughness)
        bsdf.inputs["Metallic"].default_value = float(metallic)
        if emission is not None and "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
            bsdf.inputs["Emission Strength"].default_value = float(emission_strength)
    return m


def _box(bm, center, size):
    ret = bmesh.ops.create_cube(bm, size=1.0)
    mat = Matrix.Translation(Vector(center)) @ Matrix.Diagonal((size[0], size[1], size[2], 1.0))
    bmesh.ops.transform(bm, matrix=mat, verts=ret["verts"])


def _cyl(bm, center, r1, r2, depth, segs=12):
    ret = bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=segs, radius1=r1, radius2=r2, depth=depth
    )
    bmesh.ops.transform(bm, matrix=Matrix.Translation(Vector(center)), verts=ret["verts"])


def _sphere(bm, center, radius, u_seg=10, v_seg=8):
    ret = bmesh.ops.create_uvsphere(bm, u_segments=u_seg, v_segments=v_seg, radius=radius)
    bmesh.ops.transform(bm, matrix=Matrix.Translation(Vector(center)), verts=ret["verts"])


def _bm_to_obj(bm, name: str, mat, smooth=False):
    if len(bm.verts) == 0:
        bm.free()
        return None
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    if smooth:
        for p in obj.data.polygons:
            p.use_smooth = True
    return obj


def generate_city(
    grid_blocks: int = 5,
    block_size: float = 14.0,
    road_width: float = 5.0,
    seed: int = 42,
    night_mode: bool = False,
):
    rng = random.Random(int(seed))
    n = max(3, min(11, int(grid_blocks)))
    b_sz = max(8.0, float(block_size))
    r_w = max(3.0, float(road_width))
    pitch = b_sz + r_w
    total_span = n * b_sz + (n + 1) * r_w
    half_span = 0.5 * total_span

    # Materials
    m_asphalt = _mat("City_Asphalt", (0.07, 0.075, 0.085), roughness=0.88)
    m_marking = _mat("City_RoadMarking", (0.88, 0.86, 0.78), roughness=0.65)
    m_sidewalk = _mat("City_Sidewalk", (0.36, 0.37, 0.39), roughness=0.82)
    m_grass = _mat("City_ParkGrass", (0.14, 0.36, 0.14), roughness=0.90)
    m_water = _mat("City_Water", (0.10, 0.38, 0.52), roughness=0.06, metallic=0.1)
    m_trunk = _mat("City_TreeTrunk", (0.24, 0.15, 0.09), roughness=0.85)
    m_leaves = _mat("City_TreeFoliage", (0.12, 0.34, 0.12), roughness=0.75)

    m_concrete = _mat("City_Concrete", (0.52, 0.53, 0.55), roughness=0.72)
    m_brick = _mat("City_Brick", (0.48, 0.25, 0.19), roughness=0.80)
    m_glass = _mat("City_Glass", (0.11, 0.22, 0.32), roughness=0.08, metallic=0.65)
    m_roof = _mat("City_RoofMetal", (0.28, 0.30, 0.33), roughness=0.35, metallic=0.80)
    m_lit = _mat(
        "City_LitWindow",
        (0.95, 0.82, 0.50),
        roughness=0.20,
        emission=(1.0, 0.82, 0.45),
        emission_strength=3.2 if night_mode else 0.7,
    )
    m_lamp = _mat(
        "City_LampGlow",
        (1.0, 0.90, 0.65),
        roughness=0.1,
        emission=(1.0, 0.88, 0.60),
        emission_strength=5.0 if night_mode else 1.5,
    )
    m_car = _mat("City_CarBody", (0.68, 0.16, 0.14), roughness=0.22, metallic=0.65)

    bm_road = bmesh.new()
    bm_mark = bmesh.new()
    bm_walk = bmesh.new()
    bm_grass = bmesh.new()
    bm_water = bmesh.new()
    bm_trunk = bmesh.new()
    bm_leaf = bmesh.new()
    bm_conc = bmesh.new()
    bm_brick = bmesh.new()
    bm_glass = bmesh.new()
    bm_roof = bmesh.new()
    bm_lit = bmesh.new()
    bm_lamp = bmesh.new()
    bm_car = bmesh.new()

    # 1. Asphalt base
    _box(bm_road, (0.0, 0.0, -0.10), (total_span + 6.0, total_span + 6.0, 0.20))

    # 2. Dashed center lane lines & crosswalks
    road_centers = [-half_span + 0.5 * r_w + i * pitch for i in range(n + 1)]
    for rc in road_centers:
        # Dashed lines along X and Y inside each block segment
        for bi in range(n):
            seg_c = -half_span + r_w + 0.5 * b_sz + bi * pitch
            for d in range(-2, 3):
                _box(bm_mark, (seg_c + d * 2.4, rc, 0.008), (1.2, 0.14, 0.015))
                _box(bm_mark, (rc, seg_c + d * 2.4, 0.008), (0.14, 1.2, 0.015))

    # Streetlamps & cars along avenues
    for rc in road_centers[1:-1]:
        for bi in range(n):
            seg_c = -half_span + r_w + 0.5 * b_sz + bi * pitch
            # 2 streetlamps per block edge
            for off in (-0.35 * b_sz, 0.35 * b_sz):
                lx = seg_c + off
                ly = rc + 0.5 * r_w + 0.35
                _cyl(bm_roof, (lx, ly, 2.0), 0.08, 0.06, 4.0, 8)
                _sphere(bm_lamp, (lx, ly, 4.1), 0.20, 8, 6)
            # Stylized car on lane
            if rng.random() < 0.65:
                cx = seg_c + rng.uniform(-0.3 * b_sz, 0.3 * b_sz)
                cy = rc + (1.15 if rng.random() < 0.5 else -1.15)
                _box(bm_car, (cx, cy, 0.38), (2.4, 1.15, 0.55))
                _box(bm_glass, (cx - 0.1, cy, 0.78), (1.3, 0.98, 0.42))

    # 3. Blocks: central park + skyline buildings
    mid = n // 2
    max_dist = math.hypot(mid, mid) or 1.0

    for gx in range(n):
        for gy in range(n):
            bx = -half_span + r_w + 0.5 * b_sz + gx * pitch
            by = -half_span + r_w + 0.5 * b_sz + gy * pitch

            # Sidewalk block
            _box(bm_walk, (bx, by, 0.07), (b_sz, b_sz, 0.14))

            if gx == mid and gy == mid:
                # Central Park & Fountain
                _box(bm_grass, (bx, by, 0.16), (b_sz - 1.4, b_sz - 1.4, 0.06))
                _box(bm_walk, (bx, by, 0.195), (b_sz - 1.2, 1.8, 0.03))
                _box(bm_walk, (bx, by, 0.195), (1.8, b_sz - 1.2, 0.03))
                # Central tiered fountain
                _cyl(bm_conc, (bx, by, 0.32), 2.3, 2.5, 0.30, 20)
                _cyl(bm_water, (bx, by, 0.46), 2.1, 2.1, 0.08, 20)
                _cyl(bm_conc, (bx, by, 0.85), 0.45, 0.35, 0.90, 14)
                _cyl(bm_conc, (bx, by, 1.25), 1.0, 1.1, 0.18, 16)
                # Park trees
                for sx in (-1.0, 1.0):
                    for sy in (-1.0, 1.0):
                        for k in range(2):
                            tx = bx + sx * (2.6 + k * 2.1)
                            ty = by + sy * (4.4 - k * 1.8)
                            _cyl(bm_trunk, (tx, ty, 0.95), 0.18, 0.12, 1.6, 8)
                            _sphere(bm_leaf, (tx, ty, 2.15), 1.05, 10, 8)
                            _sphere(bm_leaf, (tx + 0.4, ty - 0.3, 1.85), 0.78, 8, 6)
                continue

            # Building cluster (1 or 2 buildings per block)
            dist_norm = math.hypot(gx - mid, gy - mid) / max_dist
            is_downtown = dist_norm < 0.55

            sub_plots = [(0.0, 0.0, b_sz - 2.4, b_sz - 2.4)]
            if not is_downtown and rng.random() < 0.65:
                sub_plots = [
                    (-0.24 * b_sz, 0.0, 0.40 * b_sz, b_sz - 2.6),
                    (0.24 * b_sz, 0.0, 0.40 * b_sz, b_sz - 2.6),
                ]

            for ox, oy, pw, pd in sub_plots:
                px = bx + ox
                py = by + oy
                w = pw * rng.uniform(0.76, 0.92)
                d = pd * rng.uniform(0.76, 0.92)

                if dist_norm < 0.45:
                    h = rng.uniform(22.0, 38.0)
                    facade_bm = bm_conc
                elif dist_norm < 0.78:
                    h = rng.uniform(11.0, 21.0)
                    facade_bm = bm_conc if rng.random() < 0.6 else bm_brick
                else:
                    h = rng.uniform(5.5, 11.5)
                    facade_bm = bm_brick

                # Podium / base
                podium_h = min(3.2, h * 0.22)
                _box(facade_bm, (px, py, 0.14 + 0.5 * podium_h), (w, d, podium_h))

                # Main tower or stepped tower
                tower_w = w * (0.84 if is_downtown else 0.94)
                tower_d = d * (0.84 if is_downtown else 0.94)
                tower_h = h - podium_h
                _box(bm_glass if is_downtown else facade_bm, (px, py, 0.14 + podium_h + 0.5 * tower_h), (tower_w, tower_d, tower_h))

                # Horizontal floor ledges & lit window bands
                n_floors = max(2, int(tower_h / 2.6))
                for fl in range(n_floors):
                    fz = 0.14 + podium_h + (fl + 0.5) * (tower_h / n_floors)
                    _box(facade_bm, (px, py, fz + 0.45 * (tower_h / n_floors)), (tower_w + 0.25, tower_d + 0.25, 0.28))
                    if is_downtown or (fl % 2 == 0):
                        if rng.random() < (0.45 if night_mode else 0.25):
                            _box(bm_lit, (px, py, fz), (tower_w + 0.08, tower_d + 0.08, 0.95))
                        elif not is_downtown:
                            _box(bm_glass, (px, py, fz), (tower_w + 0.08, tower_d + 0.08, 0.95))

                # Upper setback tier + spire on tall downtown skyscrapers
                if is_downtown and h > 24.0:
                    tier_h = rng.uniform(5.0, 9.0)
                    tz = 0.14 + h + 0.5 * tier_h
                    _box(bm_glass, (px, py, tz), (tower_w * 0.65, tower_d * 0.65, tier_h))
                    _box(bm_roof, (px, py, 0.14 + h + tier_h + 0.2), (tower_w * 0.68, tower_d * 0.68, 0.4))
                    _cyl(bm_roof, (px, py, 0.14 + h + tier_h + 2.8), 0.22, 0.03, 5.2, 8)
                else:
                    # Rooftop parapet & HVAC unit
                    _box(bm_roof, (px, py, 0.14 + h + 0.25), (tower_w + 0.2, tower_d + 0.2, 0.5))
                    _box(bm_roof, (px + 0.15 * tower_w, py - 0.15 * tower_d, 0.14 + h + 0.9), (1.4, 1.1, 0.8))

    created = []
    for bm, name, mat, smooth in (
        (bm_road, "EXCAD_City_Roads", m_asphalt, False),
        (bm_mark, "EXCAD_City_Markings", m_marking, False),
        (bm_walk, "EXCAD_City_Sidewalks", m_sidewalk, False),
        (bm_grass, "EXCAD_City_ParkGrass", m_grass, False),
        (bm_water, "EXCAD_City_Water", m_water, True),
        (bm_trunk, "EXCAD_City_TreeTrunks", m_trunk, True),
        (bm_leaf, "EXCAD_City_TreeFoliage", m_leaves, True),
        (bm_conc, "EXCAD_City_Concrete", m_concrete, False),
        (bm_brick, "EXCAD_City_Brick", m_brick, False),
        (bm_glass, "EXCAD_City_Glass", m_glass, False),
        (bm_roof, "EXCAD_City_Roofs", m_roof, False),
        (bm_lit, "EXCAD_City_LitWindows", m_lit, False),
        (bm_lamp, "EXCAD_City_Lamps", m_lamp, True),
        (bm_car, "EXCAD_City_Cars", m_car, False),
    ):
        obj = _bm_to_obj(bm, name, mat, smooth=smooth)
        if obj is not None:
            created.append(obj)

    # 4. Sun + World Sky + Aerial Camera
    sun_d = bpy.data.lights.new("EXCAD_City_Sun", type='SUN')
    sun_d.energy = 1.2 if night_mode else 4.2
    sun_d.color = (0.55, 0.65, 0.95) if night_mode else (1.0, 0.88, 0.72)
    sun_o = bpy.data.objects.new("EXCAD_City_Sun", sun_d)
    bpy.context.collection.objects.link(sun_o)
    sun_o.rotation_euler = (math.radians(58.0), math.radians(18.0), math.radians(38.0))
    created.append(sun_o)

    world = bpy.context.scene.world or bpy.data.worlds.new("EXCAD_CityWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.04, 0.06, 0.12, 1.0) if night_mode else (0.52, 0.66, 0.84, 1.0)
        bg.inputs["Strength"].default_value = 0.5 if night_mode else 1.0

    cam_d = bpy.data.cameras.new("EXCAD_City_Camera")
    cam_d.lens = 35.0
    cam_d.clip_end = 1000.0
    cam_o = bpy.data.objects.new("EXCAD_City_Camera", cam_d)
    bpy.context.collection.objects.link(cam_o)
    cam_dist = total_span * 0.92
    cam_o.location = Vector((cam_dist * 0.72, -cam_dist * 0.82, cam_dist * 0.56))
    cam_o.rotation_euler = (Vector((0.0, 0.0, 7.0)) - cam_o.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = cam_o
    created.append(cam_o)

    return created
