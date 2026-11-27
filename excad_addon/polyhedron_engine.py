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
    from .image_colors import has_srgb_bytes, srgb_to_linear

    img = bpy.data.images.load(image_path, check_existing=False)
    try:
        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError(f"Empty projection image: {image_path}")
        arr = np.empty((h, w, 4), dtype=np.float32)
        img.pixels.foreach_get(arr.ravel())
        encoded = has_srgb_bytes(img)
    finally:
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
    rgb = arr[:, :, :3]
    return (srgb_to_linear(rgb) if encoded else rgb), mask


def project_colors_to_mesh(
    obj,
    poly_type: str = 'DODECAHEDRON_12',
    image_dir: str = "",
    size: float = 2.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.9),
    threshold: float = 0.15,
) -> int:
    from .view_fusion import image_coordinates

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
    world_no = local_no @ np.linalg.pinv(rot_mat)
    norms = np.linalg.norm(world_no, axis=1, keepdims=True)
    world_no = world_no / np.maximum(norms, 1e-8)

    rel = world_co - np.array(center, dtype=np.float32)
    accum_rgb = np.zeros((n_verts, 3), dtype=np.float32)
    accum_w = np.zeros(n_verts, dtype=np.float32)

    for f, rgb, mask in views:
        h, w, _ = rgb.shape
        n_vec = np.array(f["normal"], dtype=np.float32)
        facing = world_no @ n_vec
        px, py = image_coordinates(rel, f, size, w, h)
        px, py = np.rint(px).astype(np.int32), np.rint(py).astype(np.int32)

        valid = (facing > 0.05) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        idx = np.where(valid)[0]
        if len(idx) == 0:
            continue
        m_hits = mask[py[idx], px[idx]]
        idx = idx[m_hits]
        if len(idx) == 0:
            continue
        weights = (facing[idx] ** 3)
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
    use_depth: bool = True,
    texture_mode: str = "UV",
    texture_size: int = 2048,
):
    if bpy.context.mode != "OBJECT" or not np.isfinite(size) or size <= 0:
        raise ValueError("Reconstruction needs a positive size in Object Mode")
    if texture_mode.upper() not in {"UV", "VERTEX"}:
        raise ValueError("Use UV or VERTEX texture mode")
    from .feature_tree import _solid
    from .uv_bake import read_depth
    from .view_fusion import fuse

    if np.shape(center) != (3,) or not np.isfinite(center).all():
        raise ValueError("Provide a finite three-dimensional projection center")
    image_dir = bpy.path.abspath(image_dir)
    views = []
    for face in get_polyhedron_faces(poly_type):
        color_file = os.path.join(image_dir, face["filename"])
        if os.path.isfile(color_file):
            _, mask = _load_rgba_and_mask(color_file, threshold=threshold)
            depth_file = os.path.splitext(color_file)[0] + "_depth.png"
            depth = read_depth(depth_file) if use_depth and os.path.isfile(depth_file) else None
            if depth is not None and depth.shape != mask.shape:
                raise ValueError("Depth and color view dimensions must match")
            views.append((face, mask, depth))
    if not views:
        raise FileNotFoundError(f"No face_XX.png files found in {image_dir}")

    res = max(16, min(112, int(resolution)))
    points, polygons = fuse(views, size, center, res)
    selected = list(bpy.context.selected_objects)
    active = bpy.context.view_layer.objects.active
    mesh = bpy.data.meshes.new(name)
    obj = None
    try:
        mesh.from_pydata(points, [], polygons)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        for other in selected:
            other.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        try:
            _solid(mesh)
        except ValueError:
            # Ambiguous sign changes can connect components at a single vertex.
            # Let Blender resolve their topology without voxelizing every smooth surface.
            mesh.remesh_voxel_size = size / res * 0.85
            if "FINISHED" not in bpy.ops.object.voxel_remesh():
                raise ValueError("Voxel remeshing produced no surface")
            _solid(obj.data)

        if smooth_iters > 0:
            mod = obj.modifiers.new("EXCAD_Smooth", 'SMOOTH')
            mod.factor = 0.5
            mod.iterations = int(smooth_iters)
            bpy.ops.object.modifier_apply(modifier=mod.name)

        for poly in obj.data.polygons:
            poly.use_smooth = True

        if project_colors and texture_mode.upper() == "UV":
            from .uv_bake import bake
            bake(obj, image_dir, poly_type, center, size, texture_size, threshold=threshold, use_depth=use_depth)
        elif project_colors:
            project_colors_to_mesh(obj, poly_type=poly_type, image_dir=image_dir, size=size, center=center, threshold=threshold)
        return obj, len(views)
    except Exception:
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        for other in selected:
            other.select_set(True)
        bpy.context.view_layer.objects.active = active
        raise
