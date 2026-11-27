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
    _bvh: object = None

    @property
    def bvh(self):
        if self._bvh is None:
            self._bvh = BVHTree.FromPolygons(self.points.tolist(), self.triangles.tolist(), all_triangles=True)
        return self._bvh


def read_surface(obj, graph):
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
        colors = np.asarray(palette, dtype=np.float32)[np.clip(slots, 0, len(palette)-1)]
        indices = np.empty(len(mesh.loops), np.int32)
        mesh.loops.foreach_get("edge_index", indices)
        count = np.bincount(indices, minlength=len(mesh.edges))
        return Surface(obj, points, triangles, colors, np.stack((points.min(0), points.max(0))),
                       int(np.count_nonzero(count == 1)), int(np.count_nonzero((count == 0) | (count > 2))))
    finally:
        evaluated.to_mesh_clear()


def surfaces(target="ALL"):
    bpy.context.view_layer.update()
    scene = bpy.context.scene
    if target == "ALL":
        objects = list(scene.objects)
    else:
        obj = scene.objects[target]
        objects = [obj, *obj.children_recursive]
    custom_shapes = {bone.custom_shape for rig in scene.objects if rig.type == "ARMATURE"
                     for bone in rig.pose.bones if bone.custom_shape}
    graph = bpy.context.evaluated_depsgraph_get()
    result = []
    for obj in objects:
        if obj.type in {"MESH", "CURVE", "SURFACE", "FONT", "META"} and obj.visible_get() and obj not in custom_shapes:
            surface = read_surface(obj, graph)
            if surface:
                result.append(surface)
    return result


def _inside(tree, point, eps):
    loc, normal, index, distance = tree.find_nearest(Vector(point))
    return index is not None and distance > eps and (Vector(point)-loc).dot(normal) < -eps


def intersections(items, epsilon=1e-5):
    result = []
    for a, b in combinations(items, 2):
        depth = np.minimum(a.bounds[1], b.bounds[1]) - np.maximum(a.bounds[0], b.bounds[0])
        if np.min(depth) <= epsilon:
            continue
        crossing = a.bvh.overlap(b.bvh)
        contained = (not crossing and
                     ((not a.boundary_edges and _inside(a.bvh, b.points[0], epsilon)) or
                      (not b.boundary_edges and _inside(b.bvh, a.points[0], epsilon))))
        if crossing or contained:
            result.append({"objects": [a.obj.name, b.obj.name], "kind": "crossing" if crossing else "contained"})
    return result
