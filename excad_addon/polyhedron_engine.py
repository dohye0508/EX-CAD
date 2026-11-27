# SPDX-FileCopyrightText: 2026 EX-CAD Authors
# SPDX-License-Identifier: GPL-2.0-or-later

import json
import math
import os

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector


def _normalize(v):
    l = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / l, v[1] / l, v[2] / l) if l > 1e-12 else (0.0, 0.0, 1.0)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _raw_face_normals(poly_type: str) -> list[tuple[float, float, float]]:
    if poly_type == 'CUBE_6':
        return [
            (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
        ]
    if poly_type == 'OCTAHEDRON_8':
        return [
            _normalize((sx, sy, sz))
            for sz in (1.0, -1.0)
            for sy in (-1.0, 1.0)
            for sx in (-1.0, 1.0)
        ]
    phi = (1.0 + math.sqrt(5.0)) * 0.5
    if poly_type == 'DODECAHEDRON_12':
        pts = []
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                pts.append(_normalize((0.0, s1, s2 * phi)))
                pts.append(_normalize((s1, s2 * phi, 0.0)))
                pts.append(_normalize((s2 * phi, 0.0, s1)))
        return pts
    if poly_type == 'ICOSAHEDRON_20':
        inv_phi = 1.0 / phi
        pts = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    pts.append(_normalize((sx, sy, sz)))
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                pts.append(_normalize((0.0, s1 * phi, s2 * inv_phi)))
                pts.append(_normalize((s2 * inv_phi, 0.0, s1 * phi)))
                pts.append(_normalize((s1 * phi, s2 * inv_phi, 0.0)))
        return pts
    raise ValueError(f"Unknown polyhedron type: {poly_type}")


def get_polyhedron_faces(poly_type: str = 'DODECAHEDRON_12') -> list[dict]:
    normals = _raw_face_normals(poly_type)
    faces = []
    for i, n in enumerate(normals):
        if abs(n[2]) < 0.99:
            r = _normalize(_cross((0.0, 0.0, 1.0), n))
        else:
            r = (1.0, 0.0, 0.0)
        u = _normalize(_cross(n, r))
        az = math.degrees(math.atan2(n[1], n[0]))
        el = math.degrees(math.asin(max(-1.0, min(1.0, n[2]))))
        faces.append({
            "index": i,
            "filename": f"face_{i:02d}.png",
            "normal": n,
            "right": r,
            "up": u,
            "azimuth_deg": round(az, 2),
            "elevation_deg": round(el, 2),
        })
    return faces


def _get_or_create_collection(name: str):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    return col


def create_polyhedron_rig(
    poly_type: str = 'DODECAHEDRON_12',
    size: float = 2.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.9),
    output_dir: str = "",
):
    faces = get_polyhedron_faces(poly_type)
    col = _get_or_create_collection("EXCAD_Polyhedron_Rig")
    cx, cy, cz = center
    cam_dist = float(size) * 1.15

    # Wireframe bounding cage
    bm = bmesh.new()
    if poly_type == 'CUBE_6':
        bmesh.ops.create_cube(bm, size=float(size))
    else:
        subdiv = 1 if poly_type == 'OCTAHEDRON_8' else 2
        bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=float(size) * 0.55)
    cage_mesh = bpy.data.meshes.new("EXCAD_Poly_Cage")
    bm.to_mesh(cage_mesh)
    bm.free()
    cage_obj = bpy.data.objects.new("EXCAD_Poly_Cage", cage_mesh)
    cage_obj.location = center
    cage_obj.display_type = 'WIRE'
    col.objects.link(cage_obj)

    cam_objs = []
    for f in faces:
        n = f["normal"]
        r = f["right"]
        u = f["up"]
        cam_data = bpy.data.cameras.new(f"EXCAD_Cam_{f['index']:02d}")
        cam_data.type = 'ORTHO'
        cam_data.ortho_scale = float(size)
        cam_data.clip_start = 0.01
        cam_data.clip_end = float(size) * 10.0

        cam_obj = bpy.data.objects.new(f"EXCAD_Cam_{f['index']:02d}", cam_data)
        mat = Matrix((
            (r[0], u[0], n[0], cx + n[0] * cam_dist),
            (r[1], u[1], n[1], cy + n[1] * cam_dist),
            (r[2], u[2], n[2], cz + n[2] * cam_dist),
            (0.0, 0.0, 0.0, 1.0),
        ))
        cam_obj.matrix_world = mat
        col.objects.link(cam_obj)
        cam_objs.append(cam_obj)

    manifest_path = ""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, "views_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as fp:
            json.dump({
                "poly_type": poly_type,
                "size": float(size),
                "center": list(center),
                "views": faces,
            }, fp, indent=2)

    return faces, cam_objs, manifest_path


def render_polyhedron_views(
    poly_type: str,
    output_dir: str,
    size: float = 2.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.9),
    img_res: int = 256,
):
    faces, cam_objs, manifest_path = create_polyhedron_rig(poly_type, size, center, output_dir)
    scene = bpy.context.scene
    prev_cam = scene.camera
    prev_res_x = scene.render.resolution_x
    prev_res_y = scene.render.resolution_y
    prev_pct = scene.render.resolution_percentage
    prev_trans = scene.render.film_transparent
    prev_filepath = scene.render.filepath

    try:
        scene.render.resolution_x = int(img_res)
        scene.render.resolution_y = int(img_res)
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'

        rendered = []
        for f, cam_obj in zip(faces, cam_objs):
            out_path = os.path.join(output_dir, f["filename"])
            scene.camera = cam_obj
            scene.render.filepath = out_path
            bpy.ops.render.render(write_still=True)
            rendered.append(out_path)
        return rendered, manifest_path
    finally:
        scene.camera = prev_cam
        scene.render.resolution_x = prev_res_x
        scene.render.resolution_y = prev_res_y
        scene.render.resolution_percentage = prev_pct
        scene.render.film_transparent = prev_trans
        scene.render.filepath = prev_filepath


def _load_rgba_and_mask(image_path: str, threshold: float = 0.15):
    existing = set(bpy.data.images.keys())
    img = bpy.data.images.load(image_path, check_existing=True)
    w, h = img.size[0], img.size[1]
    arr = np.array(img.pixels[:], dtype=np.float32).reshape((h, w, 4))
    if img.name not in existing:
        bpy.data.images.remove(img)

    alpha = arr[:, :, 3]
    border_alpha = np.concatenate([alpha[0, :], alpha[-1, :], alpha[:, 0], alpha[:, -1]])
    if float(alpha.min()) < 0.9 and float(border_alpha.mean()) < 0.5:
        mask = alpha > 0.5
    else:
        rgb = arr[:, :, :3]
        border_rgb = np.concatenate([rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0)
        bg = np.median(border_rgb, axis=0)
        dist = np.linalg.norm(rgb - bg[None, None, :], axis=2)
        mask = dist > float(threshold)
    return arr[:, :, :3], mask


def _extract_voxel_surface_bmesh(inside: np.ndarray, origin: tuple[float, float, float], step: float) -> bmesh.types.BMesh:
    nx, ny, nz = inside.shape
    bm = bmesh.new()
    vgrid = np.full((nx + 1, ny + 1, nz + 1), -1, dtype=np.int32)
    bm_verts = []

    def get_v(ix: int, iy: int, iz: int):
        vid = vgrid[ix, iy, iz]
        if vid < 0:
            vid = len(bm_verts)
            bm_verts.append(bm.verts.new((origin[0] + ix * step, origin[1] + iy * step, origin[2] + iz * step)))
            vgrid[ix, iy, iz] = vid
        return bm_verts[vid]

    def emit_quads(cond, corners_fn):
        xs, ys, zs = np.where(cond)
        for x, y, z in zip(xs.tolist(), ys.tolist(), zs.tolist()):
            c0, c1, c2, c3 = corners_fn(x, y, z)
            try:
                bm.faces.new((get_v(*c0), get_v(*c1), get_v(*c2), get_v(*c3)))
            except ValueError:
                pass

    p = np.pad(inside, 1, mode='constant', constant_values=False)
    # +X, -X
    emit_quads(inside & ~p[2:, 1:-1, 1:-1], lambda x, y, z: ((x + 1, y, z), (x + 1, y + 1, z), (x + 1, y + 1, z + 1), (x + 1, y, z + 1)))
    emit_quads(inside & ~p[:-2, 1:-1, 1:-1], lambda x, y, z: ((x, y, z), (x, y, z + 1), (x, y + 1, z + 1), (x, y + 1, z)))
    # +Y, -Y
    emit_quads(inside & ~p[1:-1, 2:, 1:-1], lambda x, y, z: ((x, y + 1, z), (x, y + 1, z + 1), (x + 1, y + 1, z + 1), (x + 1, y + 1, z)))
    emit_quads(inside & ~p[1:-1, :-2, 1:-1], lambda x, y, z: ((x, y, z), (x + 1, y, z), (x + 1, y, z + 1), (x, y, z + 1)))
    # +Z, -Z
    emit_quads(inside & ~p[1:-1, 1:-1, 2:], lambda x, y, z: ((x, y, z + 1), (x + 1, y, z + 1), (x + 1, y + 1, z + 1), (x, y + 1, z + 1)))
    emit_quads(inside & ~p[1:-1, 1:-1, :-2], lambda x, y, z: ((x, y, z), (x, y + 1, z), (x + 1, y + 1, z), (x + 1, y, z)))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def project_colors_to_mesh(
    obj,
    poly_type: str = 'DODECAHEDRON_12',
    image_dir: str = "",
    size: float = 2.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.9),
    threshold: float = 0.15,
) -> int:
    faces = get_polyhedron_faces(poly_type)
    views = []
    for f in faces:
        fpath = os.path.join(image_dir, f["filename"])
        if os.path.isfile(fpath):
            rgb, mask = _load_rgba_and_mask(fpath, threshold=threshold)
            views.append((f, rgb, mask))
    if not views:
        raise FileNotFoundError(f"No face_XX.png files found in {image_dir}")

    mesh = obj.data
    n_verts = len(mesh.vertices)
    if n_verts == 0:
        return 0

    world_mat = np.array(obj.matrix_world, dtype=np.float32)
    rot_mat = world_mat[:3, :3]

    local_co = np.empty(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", local_co)
    local_co = local_co.reshape((n_verts, 3))
    world_co = local_co @ rot_mat.T + world_mat[:3, 3]

    local_no = np.empty(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("normal", local_no)
    local_no = local_no.reshape((n_verts, 3))
    world_no = local_no @ rot_mat.T
    norms = np.linalg.norm(world_no, axis=1, keepdims=True)
    world_no = world_no / np.maximum(norms, 1e-8)

    rel = world_co - np.array(center, dtype=np.float32)
    accum_rgb = np.zeros((n_verts, 3), dtype=np.float32)
    accum_w = np.zeros(n_verts, dtype=np.float32)

    for f, rgb, mask in views:
        h, w, _ = rgb.shape
        n_vec = np.array(f["normal"], dtype=np.float32)
        r_vec = np.array(f["right"], dtype=np.float32)
        u_vec = np.array(f["up"], dtype=np.float32)

        facing = world_no @ n_vec
        du = rel @ r_vec
        dv = rel @ u_vec
        px = np.rint((du / float(size) + 0.5) * (w - 1)).astype(np.int32)
        py = np.rint((dv / float(size) + 0.5) * (h - 1)).astype(np.int32)

        valid = (facing > 0.05) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        idx = np.where(valid)[0]
        if len(idx) == 0:
            continue
        m_hits = mask[py[idx], px[idx]]
        idx = idx[m_hits]
        if len(idx) == 0:
            continue
        weights = (facing[idx] ** 2)
        accum_rgb[idx] += rgb[py[idx], px[idx]] * weights[:, None]
        accum_w[idx] += weights

    final_rgb = np.where(accum_w[:, None] > 1e-6, accum_rgb / np.maximum(accum_w[:, None], 1e-6), 0.72)
    final_rgba = np.ones((n_verts, 4), dtype=np.float32)
    final_rgba[:, :3] = np.clip(final_rgb, 0.0, 1.0)

    attr = mesh.color_attributes.get("EXCAD_ProjectedColor")
    if attr is None:
        attr = mesh.color_attributes.new(name="EXCAD_ProjectedColor", type='BYTE_COLOR', domain='POINT')
    attr.data.foreach_set("color", final_rgba.ravel())

    mat = bpy.data.materials.get(f"{obj.name}_ProjectedMat")
    if mat is None:
        mat = bpy.data.materials.new(name=f"{obj.name}_ProjectedMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    vcol = nt.nodes.new("ShaderNodeVertexColor")
    vcol.layer_name = "EXCAD_ProjectedColor"
    bsdf.inputs["Roughness"].default_value = 0.55
    nt.links.new(vcol.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    mesh.materials.clear()
    mesh.materials.append(mat)
    return len(views)


def reconstruct_from_polyhedron_views(
    poly_type: str = 'DODECAHEDRON_12',
    image_dir: str = "",
    size: float = 2.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.9),
    resolution: int = 56,
    threshold: float = 0.15,
    smooth_iters: int = 4,
    project_colors: bool = True,
    name: str = "EXCAD_PolyMesh",
):
    faces = get_polyhedron_faces(poly_type)
    views = []
    for f in faces:
        fpath = os.path.join(image_dir, f["filename"])
        if os.path.isfile(fpath):
            rgb, mask = _load_rgba_and_mask(fpath, threshold=threshold)
            views.append((f, rgb, mask))
    if not views:
        raise FileNotFoundError(f"No face_00.png .. face_{len(faces)-1:02d}.png found in {image_dir}")

    res = max(16, min(112, int(resolution)))
    half = 0.5 * float(size)
    step = float(size) / res
    coords = np.linspace(-half + 0.5 * step, half - 0.5 * step, res, dtype=np.float32)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing='ij')
    rel = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    inside = np.ones(rel.shape[0], dtype=bool)
    for f, _, mask in views:
        h, w = mask.shape
        r_vec = np.array(f["right"], dtype=np.float32)
        u_vec = np.array(f["up"], dtype=np.float32)
        du = rel @ r_vec
        dv = rel @ u_vec
        px = np.rint((du / float(size) + 0.5) * (w - 1)).astype(np.int32)
        py = np.rint((dv / float(size) + 0.5) * (h - 1)).astype(np.int32)
        in_bounds = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        sampled = np.zeros(rel.shape[0], dtype=bool)
        sampled[in_bounds] = mask[py[in_bounds], px[in_bounds]]
        inside &= sampled

    inside_3d = inside.reshape((res, res, res))
    if not np.any(inside_3d):
        raise RuntimeError("Visual Hull intersection is empty; check background threshold or image alignment")

    origin = (center[0] - half, center[1] - half, center[2] - half)
    bm = _extract_voxel_surface_bmesh(inside_3d, origin, step)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    obj.data.remesh_voxel_size = max(0.008, step * 1.15)
    bpy.ops.object.voxel_remesh()

    if smooth_iters > 0:
        mod = obj.modifiers.new("EXCAD_Smooth", 'SMOOTH')
        mod.factor = 0.5
        mod.iterations = int(smooth_iters)
        bpy.ops.object.modifier_apply(modifier=mod.name)

    for poly in obj.data.polygons:
        poly.use_smooth = True

    if project_colors:
        project_colors_to_mesh(obj, poly_type=poly_type, image_dir=image_dir, size=size, center=center, threshold=threshold)

    return obj, len(views)
