# SPDX-FileCopyrightText: 2026 EX-CAD Authors
# SPDX-License-Identifier: GPL-2.0-or-later

import math
import os

import bpy
import bmesh
import numpy as np

from . import sketch_engine


def _morph_dilate(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask, 1, mode='constant', constant_values=False)
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out |= p[dy:dy + h, dx:dx + w]
    return out


def _morph_erode(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask, 1, mode='constant', constant_values=False)
    h, w = mask.shape
    out = np.ones_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out &= p[dy:dy + h, dx:dx + w]
    return out


def load_foreground_mask(image_path: str, max_dim: int = 256, threshold: float = 0.15) -> np.ndarray:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    existing_names = set(bpy.data.images.keys())
    img = bpy.data.images.load(image_path, check_existing=True)
    w, h = img.size[0], img.size[1]
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image dimensions: {image_path}")

    arr = np.array(img.pixels[:], dtype=np.float32).reshape((h, w, 4))
    if img.name not in existing_names:
        bpy.data.images.remove(img)

    step = max(1, max(w, h) // max_dim)
    if step > 1:
        arr = arr[::step, ::step, :]

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

    # 1 pass morphological closing to fill small pinholes
    mask = _morph_erode(_morph_dilate(mask))
    return mask


def _moore_trace(padded: np.ndarray, sy: int, sx: int) -> list[tuple[int, int]]:
    # 8-connected clockwise neighbors starting from West (-1, 0)
    dirs = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    contour = [(sx, sy)]
    cy, cx = sy, sx
    back_dir = 0  # entered from West
    max_steps = padded.shape[0] * padded.shape[1] * 2

    for _ in range(max_steps):
        found = False
        for k in range(8):
            idx = (back_dir + k) % 8
            dy, dx = dirs[idx]
            ny, nx = cy + dy, cx + dx
            if padded[ny, nx]:
                if (ny, nx) == (sy, sx) and len(contour) > 2:
                    return contour
                contour.append((nx, ny))
                # backtrack direction points back toward previous cell
                back_dir = (idx + 5) % 8
                cy, cx = ny, nx
                found = True
                break
        if not found:
            break
    return contour


def _rdp(pts: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    if len(pts) < 3:
        return pts
    ax, ay = pts[0]
    bx, by = pts[-1]
    dx, dy = bx - ax, by - ay
    denom = math.hypot(dx, dy)
    max_d = -1.0
    max_idx = 0
    for i in range(1, len(pts) - 1):
        px, py = pts[i]
        if denom < 1e-12:
            d = math.hypot(px - ax, py - ay)
        else:
            d = abs(dy * px - dx * py + bx * ay - by * ax) / denom
        if d > max_d:
            max_d = d
            max_idx = i
    if max_d > eps:
        left = _rdp(pts[: max_idx + 1], eps)
        right = _rdp(pts[max_idx:], eps)
        return left[:-1] + right
    return [pts[0], pts[-1]]


def _simplify_closed_loop(pts: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    if len(pts) <= 12:
        return pts
    # Split at farthest point from pts[0] so closed loop simplifies cleanly
    p0 = pts[0]
    split = max(range(1, len(pts)), key=lambda i: (pts[i][0] - p0[0]) ** 2 + (pts[i][1] - p0[1]) ** 2)
    a = _rdp(pts[: split + 1], eps)
    b = _rdp(pts[split:] + [pts[0]], eps)
    merged = a[:-1] + b[:-1]
    if len(merged) < 12 and eps > 1e-4:
        return _simplify_closed_loop(pts, eps * 0.5)
    if len(merged) > 256:
        step = max(1, len(merged) // 200)
        merged = merged[::step]
    return merged


def trace_largest_contour(
    mask: np.ndarray,
    target_height_m: float = 1.75,
    simplify_eps: float = 0.008,
) -> list[tuple[float, float]]:
    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    visited = np.zeros_like(padded, dtype=bool)
    best_contour: list[tuple[int, int]] = []

    H, W = padded.shape
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if padded[y, x] and not padded[y, x - 1] and not visited[y, x]:
                c = _moore_trace(padded, y, x)
                for cx, cy in c:
                    visited[cy, cx] = True
                if len(c) > len(best_contour):
                    best_contour = c

    if len(best_contour) < 3:
        raise ValueError("Could not find a closed foreground contour in image")

    xs = [p[0] for p in best_contour]
    ys = [p[1] for p in best_contour]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pix_h = max(1.0, float(max_y - min_y))
    scale = float(target_height_m) / pix_h
    center_x = 0.5 * (min_x + max_x)

    metric = [((px - center_x) * scale, (py - min_y) * scale) for px, py in best_contour]
    simplified = _simplify_closed_loop(metric, float(simplify_eps) * float(target_height_m))

    # Ensure counter-clockwise winding
    area2 = sum(
        simplified[i][0] * simplified[(i + 1) % len(simplified)][1]
        - simplified[(i + 1) % len(simplified)][0] * simplified[i][1]
        for i in range(len(simplified))
    )
    if area2 < 0.0:
        simplified.reverse()
    return simplified


def _create_curve_from_loop(name: str, plane: str, loop_2d: list[tuple[float, float]]):
    doc = sketch_engine.SketchDocument(plane)
    curve_data = bpy.data.curves.new(name=name, type='CURVE')
    curve_data.dimensions = '3D'
    spline = curve_data.splines.new('POLY')
    spline.points.add(len(loop_2d) - 1)
    for i, (u, v) in enumerate(loop_2d):
        wx, wy, wz = doc.plane_to_world(u, v, 0.0)
        spline.points[i].co = (wx, wy, wz, 1.0)
    spline.use_cyclic_u = True

    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)
    return obj


def trace_image_to_sketch(
    image_path: str,
    plane: str = 'XZ',
    target_height_m: float = 1.75,
    threshold: float = 0.15,
    simplify_eps: float = 0.008,
):
    mask = load_foreground_mask(image_path, max_dim=256, threshold=threshold)
    loop_2d = trace_largest_contour(mask, target_height_m=target_height_m, simplify_eps=simplify_eps)

    doc = sketch_engine.SketchDocument.active()
    doc.reset(plane)
    pids = [doc.add_point(u, v) for u, v in loop_2d]
    for i in range(len(pids)):
        doc.add_line_pts(pids[i], pids[(i + 1) % len(pids)])

    curve_obj = _create_curve_from_loop(f"EXCAD_Sketch_{plane}", plane, loop_2d)
    return loop_2d, curve_obj


def _build_extruded_loop_obj(name: str, plane: str, loop_2d: list[tuple[float, float]], depth: float):
    doc = sketch_engine.SketchDocument(plane)
    pids = [doc.add_point(u, v) for u, v in loop_2d]
    for i in range(len(pids)):
        doc.add_line_pts(pids[i], pids[(i + 1) % len(pids)])
    return sketch_engine.extrude_sketch_doc(doc, depth=depth, symmetric=True, name=name)


def _inflate_mask_to_mesh(mask: np.ndarray, target_height_m: float, depth: float, name: str):
    # Downsample mask to <= 96px for fast, clean relief mesh
    h, w = mask.shape
    step = max(1, max(h, w) // 96)
    m = mask[::step, ::step]
    mh, mw = m.shape

    # Iterative 2-pass Chamfer distance transform
    INF = 1e6
    dist = np.where(m, INF, 0.0).astype(np.float32)
    for y in range(1, mh):
        for x in range(1, mw - 1):
            if m[y, x]:
                dist[y, x] = min(
                    dist[y, x],
                    dist[y - 1, x] + 1.0,
                    dist[y, x - 1] + 1.0,
                    dist[y - 1, x - 1] + 1.414,
                    dist[y - 1, x + 1] + 1.414,
                )
    for y in range(mh - 2, -1, -1):
        for x in range(mw - 2, 0, -1):
            if m[y, x]:
                dist[y, x] = min(
                    dist[y, x],
                    dist[y + 1, x] + 1.0,
                    dist[y, x + 1] + 1.0,
                    dist[y + 1, x + 1] + 1.414,
                    dist[y + 1, x - 1] + 1.414,
                )

    max_d = max(1.0, float(dist.max()))
    ys, xs = np.where(m)
    if len(ys) == 0:
        raise ValueError("Empty foreground mask")
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    scale = float(target_height_m) / max(1.0, float(max_y - min_y))
    cx = 0.5 * (min_x + max_x)
    half_t = 0.5 * float(depth)

    bm = bmesh.new()
    front_v = np.full((mh, mw), -1, dtype=np.int32)
    back_v = np.full((mh, mw), -1, dtype=np.int32)
    verts = []

    for y in range(mh):
        for x in range(mw):
            if not m[y, x]:
                continue
            u = (x - cx) * scale
            v = (y - min_y) * scale
            t = math.sqrt(min(1.0, float(dist[y, x]) / max_d)) * half_t
            if t <= 1e-4:
                vid = len(verts)
                verts.append(bm.verts.new((u, 0.0, v)))
                front_v[y, x] = vid
                back_v[y, x] = vid
            else:
                f_id = len(verts)
                verts.append(bm.verts.new((u, -t, v)))
                b_id = len(verts)
                verts.append(bm.verts.new((u, t, v)))
                front_v[y, x] = f_id
                back_v[y, x] = b_id

    for y in range(mh - 1):
        for x in range(mw - 1):
            if m[y, x] and m[y, x + 1] and m[y + 1, x + 1] and m[y + 1, x]:
                try:
                    bm.faces.new((
                        verts[front_v[y, x]],
                        verts[front_v[y, x + 1]],
                        verts[front_v[y + 1, x + 1]],
                        verts[front_v[y + 1, x]],
                    ))
                except ValueError:
                    pass
                try:
                    bm.faces.new((
                        verts[back_v[y, x]],
                        verts[back_v[y + 1, x]],
                        verts[back_v[y + 1, x + 1]],
                        verts[back_v[y, x + 1]],
                    ))
                except ValueError:
                    pass

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    obj.data.remesh_voxel_size = max(0.01, float(target_height_m) / 72.0)
    bpy.ops.object.voxel_remesh()
    mod = obj.modifiers.new("EXCAD_Smooth", 'SMOOTH')
    mod.factor = 0.5
    mod.iterations = 4
    bpy.ops.object.modifier_apply(modifier=mod.name)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def _boolean_intersect(target_obj, cutter_obj):
    for o in bpy.context.selected_objects:
        o.select_set(False)
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj

    mod = target_obj.modifiers.new("EXCAD_Carve", 'BOOLEAN')
    mod.operation = 'INTERSECT'
    mod.object = cutter_obj
    mod.solver = 'EXACT'
    bpy.ops.object.modifier_apply(modifier=mod.name)

    if len(target_obj.data.vertices) == 0:
        # Fallback to FLOAT solver if EXACT returned empty
        raise RuntimeError("Boolean intersect produced empty mesh")


def synthesize_mesh_from_images(
    front_image: str,
    side_image: str = "",
    top_image: str = "",
    mode: str = 'MULTIVIEW_CARVE',
    target_height_m: float = 1.75,
    depth: float = 0.30,
    threshold: float = 0.15,
    simplify_eps: float = 0.008,
    voxel_size: float = 0.02,
    smooth_iters: int = 4,
    name: str = "EXCAD_ImageMesh",
):
    front_mask = load_foreground_mask(front_image, max_dim=256, threshold=threshold)
    front_loop = trace_largest_contour(front_mask, target_height_m=target_height_m, simplify_eps=simplify_eps)

    # Always register Front sketch in SketchDocument
    doc = sketch_engine.SketchDocument.active()
    doc.reset('XZ')
    pids = [doc.add_point(u, v) for u, v in front_loop]
    for i in range(len(pids)):
        doc.add_line_pts(pids[i], pids[(i + 1) % len(pids)])

    if mode == 'EXTRUDE':
        return _build_extruded_loop_obj(name, 'XZ', front_loop, depth=depth)

    if mode == 'INFLATE' or (mode == 'MULTIVIEW_CARVE' and not side_image):
        return _inflate_mask_to_mesh(front_mask, target_height_m=target_height_m, depth=depth, name=name)

    # MULTIVIEW_CARVE: intersect extruded Front (XZ) + Side (YZ) (+ optional Top XY)
    carve_span = max(2.0, float(target_height_m) * 2.5)
    front_obj = _build_extruded_loop_obj(name, 'XZ', front_loop, depth=carve_span)

    side_mask = load_foreground_mask(side_image, max_dim=256, threshold=threshold)
    side_loop = trace_largest_contour(side_mask, target_height_m=target_height_m, simplify_eps=simplify_eps)
    side_obj = _build_extruded_loop_obj(f"{name}_SideCutter", 'YZ', side_loop, depth=carve_span)

    try:
        _boolean_intersect(front_obj, side_obj)
    finally:
        bpy.data.objects.remove(side_obj, do_unlink=True)

    if top_image and os.path.isfile(top_image):
        top_mask = load_foreground_mask(top_image, max_dim=256, threshold=threshold)
        # Estimate width from front_loop
        us = [p[0] for p in front_loop]
        top_span = max(0.2, max(us) - min(us))
        top_loop = trace_largest_contour(top_mask, target_height_m=top_span, simplify_eps=simplify_eps)
        top_obj = _build_extruded_loop_obj(f"{name}_TopCutter", 'XY', top_loop, depth=carve_span)
        top_obj.location = (0.0, 0.0, 0.5 * float(target_height_m))
        try:
            _boolean_intersect(front_obj, top_obj)
        finally:
            bpy.data.objects.remove(top_obj, do_unlink=True)

    for o in bpy.context.selected_objects:
        o.select_set(False)
    front_obj.select_set(True)
    bpy.context.view_layer.objects.active = front_obj

    front_obj.data.remesh_voxel_size = max(0.005, float(voxel_size))
    bpy.ops.object.voxel_remesh()

    if smooth_iters > 0:
        mod = front_obj.modifiers.new("EXCAD_Smooth", 'SMOOTH')
        mod.factor = 0.5
        mod.iterations = int(smooth_iters)
        bpy.ops.object.modifier_apply(modifier=mod.name)

    for poly in front_obj.data.polygons:
        poly.use_smooth = True

    return front_obj
