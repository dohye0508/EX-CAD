# SPDX-License-Identifier: GPL-2.0-or-later
"""Semantic placement using evaluated world-space geometry and explicit contact gaps."""

import itertools
import math
from dataclasses import dataclass

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import scene_mesh


def _items(obj):
    result = scene_mesh.surfaces(obj.name)
    if not result:
        raise ValueError(f"{obj.name} has no visible surface")
    return result


def bounds(obj):
    items = _items(obj)
    return _bounds(items)


def _bounds(items):
    return np.stack((np.min([s.bounds[0] for s in items], 0), np.max([s.bounds[1] for s in items], 0)))


def _translate(obj, delta):
    matrix = obj.matrix_world.copy()
    matrix.translation += Vector(delta)
    obj.matrix_world = matrix
    bpy.context.view_layer.update()


def place_relative(obj, target=None, relation="ON_TOP_OF", gap=None):
    obj = bpy.context.scene.objects[obj]
    relation = relation.upper()
    gap = float(gap if gap is not None else 0 if relation == "ALIGN_FLOOR" else 0.05)
    moving = _items(obj)
    a = _bounds(moving)
    delta = np.zeros(3)
    if relation == "ALIGN_FLOOR":
        delta[2] = gap - a[0, 2]
    else:
        other = bpy.context.scene.objects[target]
        if obj == other or other in obj.children_recursive:
            raise ValueError("The support must be independent of the moving object")
        support = _items(other)
        b = _bounds(support)
        delta = b.mean(0) - a.mean(0)
        if relation == "ON_TOP_OF":
            delta[2] = b[1, 2] - a[0, 2]
            clearance = float("inf")
            for part, surface in itertools.product(moving, support):
                clearance = scene_mesh.vertical_clearance(part, surface, delta, clearance)
                if clearance <= 1e-6:
                    break
            if not math.isfinite(clearance):
                raise ValueError("No supporting surface under the centered object")
            delta[2] += gap - clearance
        elif relation == "INSIDE":
            if np.any(np.ptp(a, axis=0) + 2 * gap > np.ptp(b, axis=0) + 1e-6):
                raise ValueError("Object does not fit inside target bounds with the requested clearance")
        else:
            axis, sign = {"LEFT_OF": (0, -1), "RIGHT_OF": (0, 1), "IN_FRONT_OF": (1, -1), "BEHIND": (1, 1)}[relation]
            low, high = (1, 0) if sign > 0 else (0, 1)
            delta[axis] = b[low, axis] + sign * gap - a[high, axis]
            delta[2] = b[0, 2] - a[0, 2]
    root = (obj.find_armature() or obj) if obj.get("excad_character") else obj
    _translate(root, delta)
    return {"object": obj.name, "bbox": (a + delta).round(5).tolist()}


def look_at(obj, target):
    source = bpy.context.scene.objects[obj]
    dest = bpy.context.scene.objects[target]
    position, _, scale = source.matrix_world.decompose()
    try:
        center = Vector(bounds(dest).mean(0))
    except ValueError:
        center = dest.matrix_world.translation
    direction = center - position
    if direction.length < 1e-8:
        raise ValueError("Look-at target coincides with the object origin")
    track = "-Z" if source.type in {"CAMERA", "LIGHT"} else "-Y"
    rotation = direction.to_track_quat(track, "Y" if track == "-Z" else "Z")
    source.matrix_world = Matrix.LocRotScale(position, rotation, scale)
    bpy.context.view_layer.update()
    return {"object": source.name}


def _duplicate(asset):
    source = [asset, *asset.children_recursive]
    copies = {}
    try:
        for obj in source:
            copy = obj.copy()
            bpy.context.collection.objects.link(copy)
            copies[obj] = copy
        for obj, copy in copies.items():
            if obj == asset:
                copy.parent = None
                copy.matrix_world = asset.matrix_world
            else:
                copy.parent = copies[obj.parent]
            for mod in copy.modifiers:
                if mod.type == "ARMATURE" and mod.object in copies:
                    mod.object = copies[mod.object]
            wearer = obj.get("excad_human")
            if wearer in copies:
                copy["excad_human"] = copies[wearer]
                copy["excad_wearer"] = copies[wearer].name
        return copies[asset]
    except Exception:
        for copy in copies.values():
            bpy.data.objects.remove(copy, do_unlink=True)
        raise


def scatter_on_surface(asset, surface_obj, count, min_dist=0.1, align_normal=True, seed=0):
    scene = bpy.context.scene
    source, surface = scene.objects[asset], scene.objects[surface_obj]
    if source.get("excad_character"):
        source = source.find_armature() or source
    if count < 1 or count > 10000 or min_dist < 0:
        raise ValueError("Choose 1..10000 instances and nonnegative spacing")
    items = _items(surface)
    triangles = np.concatenate([s.points[s.triangles] for s in items])
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1)
    if areas.sum() <= 0:
        raise ValueError("Surface is degenerate")
    probability = areas / areas.sum(dtype=np.float64)
    rng = np.random.default_rng(seed)
    cell = max(min_dist, 1e-6)
    lookup, placements = {}, []
    for _ in range(24):
        sample_count = max(32, (count - len(placements)) * 8)
        choices = rng.choice(len(triangles), sample_count, p=probability)
        for fi, (r, s) in zip(choices, rng.random((sample_count, 2))):
            a, b, c = triangles[fi]
            sqrt_r = math.sqrt(r)
            point = a * (1 - sqrt_r) + sqrt_r * ((1 - s) * b + s * c)
            key = tuple(np.floor(point / cell).astype(int))
            nearby = (
                q
                for delta in itertools.product((-1, 0, 1), repeat=3)
                for q in lookup.get(tuple(i + j for i, j in zip(key, delta)), ())
            )
            if any(np.dot(point - q, point - q) < min_dist**2 for q in nearby):
                continue
            lookup.setdefault(key, []).append(point)
            placements.append((point, cross[fi] / areas[fi]))
            if len(placements) == count:
                break
        if len(placements) == count:
            break
    if len(placements) < count:
        raise ValueError("Surface cannot accommodate the requested count and minimum distance")
    cloud = np.concatenate([s.points for s in _items(source)])
    origin, rotation, scale = source.matrix_world.decompose()
    up = rotation @ Vector((0, 0, 1))
    copies = []
    try:
        for point, normal in placements:
            n = Vector(normal)
            rotation_delta = up.rotation_difference(n) if align_normal else Matrix.Identity(3).to_quaternion()
            verts = (cloud - np.array(origin)) @ np.array(rotation_delta.to_matrix()).T
            # Project the center of the footprint onto the sampled surface.
            center = (verts.min(0) + verts.max(0)) / 2
            anchor = center + normal * (float(np.min(verts @ normal)) - float(center @ normal))
            copy = _duplicate(source)
            copies.append(copy)
            copy.matrix_world = Matrix.LocRotScale(Vector(point - anchor), rotation_delta @ rotation, scale)
        bpy.context.view_layer.update()
    except Exception:
        for root in copies:
            for obj in [*root.children_recursive, root]:
                bpy.data.objects.remove(obj, do_unlink=True)
        raise
    return {"objects": [o.name for o in copies], "count": len(copies)}


@dataclass
class Assembly:
    name: str
    root: object
    parts: list
    bounds: np.ndarray

    @classmethod
    def create(cls, name, root, parts):
        bounds = np.stack((np.min([s.bounds[0] for s in parts], axis=0), np.max([s.bounds[1] for s in parts], axis=0)))
        return cls(name, root, parts, bounds)

    def hit(self, other):
        if np.any(np.minimum(self.bounds[1], other.bounds[1]) < np.maximum(self.bounds[0], other.bounds[0]) - 1e-5):
            return None
        for a, b in itertools.product(self.parts, other.parts):
            kind = scene_mesh.overlap_kind(a, b)
            if kind:
                hit = {"objects": [self.name, other.name], "kind": kind}
                if (a.obj.name, b.obj.name) != (self.name, other.name):
                    hit["parts"] = [a.obj.name, b.obj.name]
                return hit
        return None


def _assemblies(names):
    if names is None:
        grouped = {}
        for part in scene_mesh.surfaces():
            root = scene_mesh.assembly_root(part.obj)
            grouped.setdefault(root, []).append(part)
        result = []
        for root, parts in grouped.items():
            bodies = [s.obj for s in parts if s.obj.get("excad_character") and s.obj.find_armature() == root]
            name = bodies[0].name if len(bodies) == 1 else root.name
            result.append(Assembly.create(name, root, parts))
        return result
    if isinstance(names, str):
        raise ValueError("objects must be a list of object names")
    occupied = set()
    result = []
    for name in dict.fromkeys(names):
        obj = bpy.context.scene.objects[name]
        root = (obj.find_armature() or obj) if obj.get("excad_character") else obj
        parts = _items(root)
        members = {s.obj for s in parts}
        if occupied & members:
            raise ValueError("Choose independent assemblies or their individual parts")
        occupied.update(members)
        result.append(Assembly.create(name, root, parts))
    return result


def check_intersections(objects=None):
    groups = _assemblies(objects)
    return {"intersections": [hit for a, b in itertools.combinations(groups, 2) if (hit := a.hit(b))]}


def resolve_overlaps(objects=None, fixed=(), gap=0.01, max_steps=64):
    if gap < 0 or max_steps < 1:
        raise ValueError("Use a nonnegative gap and positive step limit")
    items = _assemblies(objects)
    fixed_names = set(fixed)
    aliases = {
        name: item.name for item in items for name in {item.name, item.root.name, *(s.obj.name for s in item.parts)}
    }
    missing = fixed_names - aliases.keys()
    if missing:
        raise ValueError(f"Unknown fixed collision object: {sorted(missing)[0]}")
    locked = {aliases[name] for name in fixed_names}
    original = {s.root: s.root.matrix_world.copy() for s in items}
    ordered = sorted(items, key=lambda s: (s.name not in locked, s.name))
    moved = set()
    steps = 0
    try:
        for i, item in enumerate(ordered):
            previous = ordered[:i]
            while True:
                touching = [s for s in previous if s.hit(item)]
                if not touching:
                    break
                if item.name in locked or steps >= max_steps:
                    raise ValueError("Unable to separate all collision objects within the step limit")
                # Each candidate clears every currently intersecting obstacle on that axis.
                candidates = []
                for axis in (0, 1):
                    positive = max(s.bounds[1, axis] - item.bounds[0, axis] for s in touching) + gap
                    negative = min(s.bounds[0, axis] - item.bounds[1, axis] for s in touching) - gap
                    for distance in (positive, negative):
                        delta = np.zeros(3)
                        delta[axis] = distance
                        trial = item.bounds + delta
                        hits = sum(
                            np.all(np.minimum(s.bounds[1], trial[1]) - np.maximum(s.bounds[0], trial[0]) > 1e-5)
                            for s in previous
                        )
                        candidates.append((hits, abs(distance), axis, delta))
                _, _, _, delta = min(candidates, key=lambda c: c[:3])
                _translate(item.root, delta)
                item = Assembly.create(item.name, item.root, _items(item.root))
                ordered[i] = item
                moved.add(item.name)
                steps += 1
        remaining = any(a.hit(b) for a, b in itertools.combinations(ordered, 2))
        if remaining:
            raise ValueError("A parent transform introduced another collision")
    except Exception:
        for obj, matrix in original.items():
            obj.matrix_world = matrix
        bpy.context.view_layer.update()
        raise
    return {"moved": sorted(moved), "steps": steps}
