# SPDX-License-Identifier: GPL-2.0-or-later
"""Evaluated world-space surfaces used by inspection and scene layout."""

from dataclasses import dataclass
from itertools import combinations

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


@dataclass
class Surface:
    obj: object
    points: np.ndarray
    triangles: np.ndarray
    colors: np.ndarray
    bounds: np.ndarray
    boundary_edges: int
    nonmanifold_edges: int
    shading: object = None
    boundary_vertices: object = None
    _bvh: object = None
    _samples: object = None
    _hem: object = None

    @property
    def bvh(self):
        if self._bvh is None:
            self._bvh = BVHTree.FromPolygons(self.points.tolist(), self.triangles.tolist(), all_triangles=True)
        return self._bvh

    @property
    def closed(self):
        return self.boundary_edges == 0 and self.nonmanifold_edges == 0

    def component_samples(self):
        """Keep one surface point per connected component, including isolated islands."""
        if self._samples is None:
            parent = np.arange(len(self.points), dtype=np.int32)
            while True:
                roots = parent[self.triangles]
                lowest = roots.min(axis=1)
                if np.all(roots == lowest[:, None]):
                    break
                for column in roots.T:
                    np.minimum.at(parent, column, lowest)
                while not np.array_equal(parent, parent[parent]):
                    parent = parent[parent]
            indices = np.unique(parent[self.triangles[:, 0]], return_index=True)[1]
            self._samples = self.points[self.triangles[indices]].mean(axis=1)
        return self._samples

    def hem_faces(self):
        if self._hem is None:
            border = np.zeros(len(self.points), dtype=bool)
            if self.boundary_vertices is not None:
                border[self.boundary_vertices] = True
            for _ in range(2):
                near = border[self.triangles].any(axis=1)
                border[self.triangles[near]] = True
            self._hem = border[self.triangles].any(axis=1)
        return self._hem


def read_surface(obj, graph, shading=False):
    evaluated = obj.evaluated_get(graph)
    mesh = evaluated.to_mesh()
    try:
        if not mesh or not mesh.polygons:
            return None
        mesh.calc_loop_triangles()
        coords = np.empty((len(mesh.vertices), 3), np.float32)
        mesh.vertices.foreach_get("co", coords.ravel())
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float32)
        points = coords @ matrix[:3, :3].T + matrix[:3, 3]
        triangles = np.empty((len(mesh.loop_triangles), 3), np.int32)
        mesh.loop_triangles.foreach_get("vertices", triangles.ravel())
        slots = np.empty(len(triangles), np.int32)
        mesh.loop_triangles.foreach_get("material_index", slots)
        base = (0.42, 0.57, 0.70)
        palette = [tuple(m.diffuse_color[:3]) if m else base for m in mesh.materials] or [base]
        colors = np.asarray(palette, dtype=np.float32)[np.clip(slots, 0, len(palette) - 1)]
        indices = np.empty(len(mesh.loops), np.int32)
        mesh.loops.foreach_get("edge_index", indices)
        count = np.bincount(indices, minlength=len(mesh.edges))
        boundary_vertices = None
        if shading and obj.get("excad_slot") in {"top", "outerwear", "bottom", "footwear"}:
            edges = np.empty((len(mesh.edges), 2), np.int32)
            mesh.edges.foreach_get("vertices", edges.ravel())
            boundary_vertices = np.unique(edges[count == 1])
            seam = obj.vertex_groups.get("EXCAD seams")
            if seam:
                marked = [
                    v.index for v in mesh.vertices if any(g.group == seam.index and g.weight >= 0.15 for g in v.groups)
                ]
                boundary_vertices = np.union1d(boundary_vertices, np.asarray(marked, dtype=np.int32))
        detail = None
        if shading:
            from .inspect_shading import surface_shading

            detail = surface_shading(mesh, matrix, slots)
        return Surface(
            obj,
            points,
            triangles,
            colors,
            np.stack((points.min(0), points.max(0))),
            int(np.count_nonzero(count == 1)),
            int(np.count_nonzero((count == 0) | (count > 2))),
            detail,
            boundary_vertices,
        )
    finally:
        evaluated.to_mesh_clear()


def surfaces(target="ALL", shading=False):
    bpy.context.view_layer.update()
    scene = bpy.context.scene
    if target == "ALL":
        objects = list(scene.objects)
    else:
        obj = scene.objects[target]
        objects = [obj, *obj.children_recursive]
        objects.extend(
            o
            for o in scene.objects
            if (o.get("excad_human") == obj or o.get("excad_wearer") == obj.name) and o not in objects
        )
    custom_shapes = {
        bone.custom_shape
        for rig in scene.objects
        if rig.type == "ARMATURE"
        for bone in rig.pose.bones
        if bone.custom_shape
    }
    graph = bpy.context.evaluated_depsgraph_get()
    result = []
    for obj in objects:
        if obj.type in {"MESH", "CURVE", "SURFACE", "FONT", "META"} and obj.visible_get() and obj not in custom_shapes:
            surface = read_surface(obj, graph, shading)
            if surface:
                result.append(surface)
    return result


def _inside(surface, point, eps):
    if np.any(point <= surface.bounds[0] + eps) or np.any(point >= surface.bounds[1] - eps):
        return False
    tree = surface.bvh
    p = Vector(point)
    _, _, index, distance = tree.find_nearest(p)
    if index is None or distance <= eps:
        return False
    # Odd-even classification also works on reflected meshes and enclosed cavities.
    direction = Vector((0.62371, 0.72193, 0.29977)).normalized()
    hits = 0
    while True:
        location, _, face, _ = tree.ray_cast(p, direction)
        if face is None:
            return hits % 2 == 1
        hits += 1
        p = location + direction * (4 * eps)


def _crosses(a, b, pairs, eps):
    """Distinguish interpenetration from two closed solids sharing a sloped face."""
    if not (a.closed and b.closed):
        return True
    ia, ib = np.asarray(pairs, dtype=np.int32).T
    ta, tb = a.points[a.triangles[ia]], b.points[b.triangles[ib]]

    def normals(triangles):
        v = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-20)

    na, nb = normals(ta), normals(tb)
    da = np.einsum("ijk,ik->ij", ta - tb[:, :1], nb)
    db = np.einsum("ijk,ik->ij", tb - ta[:, :1], na)
    straddle_a = (da.min(axis=1) < -eps) & (da.max(axis=1) > eps)
    straddle_b = (db.min(axis=1) < -eps) & (db.max(axis=1) > eps)
    for i in np.flatnonzero(straddle_a & straddle_b):
        axis = np.cross(na[i], nb[i])
        axis /= max(np.linalg.norm(axis), 1e-12)

        def interval(triangle, distances):
            values = []
            for j in range(3):
                d, e = distances[j - 1], distances[j]
                if (d > 0) != (e > 0):
                    point = triangle[j - 1] + (triangle[j] - triangle[j - 1]) * (d / (d - e))
                    values.append(np.dot(point, axis))
            return min(values), max(values)

        low_a, high_a = interval(ta[i], da[i])
        low_b, high_b = interval(tb[i], db[i])
        if min(high_a, high_b) - max(low_a, low_b) > eps:
            return True
    # Aligned or coincident boundaries may overlap without a transverse triangle
    # crossing. Sample both sides of the overlapping triangle patch inside both solids.
    coplanar = (np.abs(da).max(axis=1) <= eps) & (np.abs(db).max(axis=1) <= eps)
    for first, second, normal in zip(ta[coplanar], tb[coplanar], na[coplanar]):
        point = _patch_center(first, second, normal, eps)
        if point is not None:
            for side in (-1, 1):
                sample = point + normal * (side * 5 * eps)
                if _inside(a, sample, eps) and _inside(b, sample, eps):
                    return True
    # A triangle edge may track a mesh seam all the way through the other solid.
    for subject, container, ids in ((a, b, ia), (b, a, ib)):
        triangles = subject.triangles[np.unique(ids)]
        vertices = subject.points[np.unique(triangles)]
        midpoints = subject.points[triangles].mean(axis=1)
        if any(_inside(container, p, eps) for p in np.concatenate((vertices, midpoints))):
            return True
    return False


def _patch_center(first, second, normal, eps):
    # Clip coplanar triangles in their plane. Their individual centers can both
    # be outside the overlap when only a narrow strip is shared.
    polygon = list(first)
    for i in range(3):
        start, end = second[i - 1], second[i]
        inward = np.cross(end - start, normal)
        if np.dot(second[(i + 1) % 3] - start, inward) < 0:
            inward = -inward
        polygon_out = []
        for j, current in enumerate(polygon):
            previous = polygon[j - 1]
            a, b = np.dot(previous - start, inward), np.dot(current - start, inward)
            if (a >= 0) != (b >= 0):
                polygon_out.append(previous + (current - previous) * (a / (a - b)))
            if b >= 0:
                polygon_out.append(current)
        polygon = polygon_out
        if len(polygon) < 3:
            return None
    center = np.mean(polygon, axis=0)
    area = sum(abs(np.dot(normal, np.cross(p - center, polygon[i - 1] - center))) for i, p in enumerate(polygon))
    return center if area > eps * eps else None


def overlap_kind(a, b, epsilon=1e-5):
    scale = max(float(np.abs(a.bounds).max()), float(np.abs(b.bounds).max()))
    epsilon = max(epsilon, scale * np.finfo(np.float32).eps * 2)
    depth = np.minimum(a.bounds[1], b.bounds[1]) - np.maximum(a.bounds[0], b.bounds[0])
    if np.min(depth) < -epsilon or (a.closed and b.closed and np.min(depth) <= epsilon):
        return None
    pairs = a.bvh.overlap(b.bvh)
    if pairs and a.closed and b.closed and depth[2] < 0.1 * min(np.ptp(a.bounds[:, 2]), np.ptp(b.bounds[:, 2])):
        # At grazing support, BVH can report a short line crossing at the rim
        # although the solids only meet. Check the actual vertical separation.
        upper, lower = (a, b) if a.bounds[0, 2] > b.bounds[0, 2] else (b, a)
        lift = float(lower.bounds[1, 2] - upper.bounds[0, 2])
        clearance = vertical_clearance(upper, lower, (0, 0, lift), best=lift + 2 * epsilon)
        if clearance >= lift - epsilon:
            return None
    if pairs and _crosses(a, b, pairs, epsilon):
        return "crossing"
    for subject, container in ((a, b), (b, a)):
        if container.closed and any(_inside(container, p, epsilon) for p in subject.component_samples()):
            return "contained"
    return None


def assembly_root(obj):
    while obj.parent:
        obj = obj.parent
    return obj


def _project_vertices(subject, other, shift, sign, best):
    """Measure vertical separation at projected vertices, in the other mesh's space."""
    points = subject.points + shift
    inside = np.all(
        (points[:, :2] >= other.bounds[0, :2] - 1e-6) & (points[:, :2] <= other.bounds[1, :2] + 1e-6), axis=1
    )
    points = points[inside]
    cap = other.bounds[1 if sign < 0 else 0, 2]
    origin_z = float(cap - sign * max(1.0, np.ptp(other.bounds[:, 2])))
    ray = Vector((0, 0, sign))
    for i in np.argsort(-sign * points[:, 2]):
        point = points[i]
        if -sign * (point[2] - cap) >= best:
            break
        hit = other.bvh.ray_cast(Vector((point[0], point[1], origin_z)), ray)[0]
        if hit is not None:
            best = min(best, -sign * float(point[2] - hit.z))
            if best <= 1e-6:
                break
    return best


def _project_edges(surface, shift, limit, bottom):
    tris = surface.triangles
    heights = surface.points[tris, 2] + shift[2]
    tri = tris[heights.min(axis=1) <= limit] if bottom else tris[heights.max(axis=1) >= limit]
    edges = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    pairs = np.unique(np.sort(edges, axis=1), axis=0)
    return surface.points[pairs] + shift


def vertical_clearance(upper, lower, shift=(0, 0, 0), best=float("inf")):
    """First contact distance when moving a translated upper mesh straight down.

    Over each pair of projected triangles the difference in height is linear.
    Its minimum is at a projected vertex or at the intersection of two edges.
    """
    shift = np.asarray(shift, dtype=np.float64)
    lo, hi = upper.bounds + shift
    if (
        lo[2] - lower.bounds[1, 2] >= best
        or np.any(hi[:2] < lower.bounds[0, :2])
        or np.any(lo[:2] > lower.bounds[1, :2])
    ):
        return best
    best = _project_vertices(upper, lower, shift, -1, best)
    if best > 1e-6:
        best = _project_vertices(lower, upper, -shift, 1, best)
    if best <= 1e-6:
        return best
    above = _project_edges(upper, shift, lower.bounds[1, 2] + best, True)
    below = _project_edges(lower, np.zeros(3), lo[2] - best, False)
    if not len(above) or not len(below):
        return best
    # Fewer scalar iterations for a dense asset sitting on a simple support.
    sign = 1
    if len(above) > len(below):
        above, below, sign = below, above, -1
    minima, maxima = below[:, :, :2].min(axis=1), below[:, :, :2].max(axis=1)

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    for a, b in above:
        mask = np.all(maxima >= np.minimum(a, b)[:2], axis=1) & np.all(minima <= np.maximum(a, b)[:2], axis=1)
        edges = below[mask]
        if not len(edges):
            continue
        start, end = edges[:, 0], edges[:, 1]
        r, s, q = b - a, end - start, start - a
        denom = cross(r, s)
        valid = np.abs(denom) > 1e-14
        t = np.divide(cross(q, s), denom, out=np.zeros_like(denom), where=valid)
        u = np.divide(cross(q, r), denom, out=np.zeros_like(denom), where=valid)
        valid &= (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        if np.any(valid):
            distance = sign * (a[2] + t * r[2] - start[:, 2] - u * s[:, 2])
            best = min(best, float(distance[valid].min()))
            if best <= 1e-6:
                break
    return best


def intersections(items, epsilon=1e-5):
    result = []
    for a, b in combinations(items, 2):
        kind = overlap_kind(a, b, epsilon)
        if kind:
            result.append({"objects": [a.obj.name, b.obj.name], "kind": kind})
    return result
