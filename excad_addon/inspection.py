# SPDX-License-Identifier: GPL-2.0-or-later
"""Quick orthographic inspection. Ray cast evaluated triangles without touching the user's scene."""

from pathlib import Path
from itertools import combinations
import struct
import tempfile
import time
import zlib

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import scene_mesh
from .inspect_shading import Shading
from .image_colors import linear_to_srgb

DIRECTIONS = {"FRONT": (0, -1, 0), "SIDE": (1, 0, 0), "ISO": (1.15, -1.65, 1.05), "TOP": (0, 0, 1)}
# Five-pixel labels keep the orientations legible in the standalone PNG.
GLYPHS = {
    "F": (15, 8, 14, 8, 8),
    "R": (14, 9, 14, 10, 9),
    "O": (6, 9, 9, 9, 6),
    "N": (9, 13, 11, 9, 9),
    "T": (15, 6, 6, 6, 6),
    "S": (7, 8, 6, 1, 14),
    "I": (14, 4, 4, 4, 14),
    "D": (14, 9, 9, 9, 14),
    "E": (15, 8, 14, 8, 15),
    "P": (14, 9, 14, 8, 8),
}


def _png(path, rgb):
    def chunk(tag, payload):
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload))

    height, width = rgb.shape[:2]
    data = np.zeros((height, 1 + 3 * width), dtype=np.uint8)
    data[:, 1:] = rgb.reshape(height, -1)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(data.tobytes(), 2))
        + chunk(b"IEND", b"")
    )


def _tile(tree, bounds, shading, view, size):
    eye = Vector(DIRECTIONS[view]).normalized()
    right = Vector((1, 0, 0)) if view == "TOP" else Vector((0, 0, 1)).cross(eye).normalized()
    up = eye.cross(right)
    corners = np.array([[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]])
    step = max(np.ptp(corners @ right), np.ptp(corners @ up), 1e-6) / (size * 0.84)
    distance = float(np.linalg.norm(bounds[1] - bounds[0])) + 1
    center = Vector(bounds.mean(axis=0)) + eye * distance
    start = center + (up - right) * ((size - 1) * step / 2)
    rows = start, -up * step, right * step
    light = (eye * 0.8 + up * 0.9 - right * 0.55).normalized()
    ids = np.full((size, size), -1, dtype=np.int32)
    positions = np.zeros((size, size, 3), dtype=np.float32)
    ray = -eye
    for y in range(size):
        p = rows[0] + y * rows[1]
        for x in range(size):
            position, _, index, _ = tree.ray_cast(p, ray)
            if index is not None:
                ids[y, x] = index
                positions[y, x] = position
            p += rows[2]
    rgb = np.empty((size, size, 3), np.uint8)
    bg = np.linspace((239, 244, 248), (216, 225, 233), size)
    rgb[:] = bg[:, None]
    y, x = np.nonzero(ids >= 0)
    for layer in range(12):
        if not len(x):
            break
        normals, colors, alpha = shading.sample(ids[y, x], positions[y, x])
        visible = alpha >= 0.4
        intensity = 0.33 + 0.67 * np.maximum(0, normals[visible] @ np.asarray(light))
        linear = colors[visible] * intensity[:, None]
        rgb[y[visible], x[visible]] = np.clip(255 * linear_to_srgb(linear), 0, 255).astype(np.uint8)
        y, x = y[~visible], x[~visible]
        # See the face behind transparent hair and eyelash cards.
        for j, i in zip(y, x):
            p = Vector(positions[j, i]) + ray * (step * 0.001)
            hit, _, index, _ = tree.ray_cast(p, ray)
            ids[j, i] = -1 if index is None else index
            if hit is not None:
                positions[j, i] = hit
        valid = ids[y, x] >= 0
        y, x = y[valid], x[valid]
    for i, char in enumerate(view):
        for j, bits in enumerate(GLYPHS[char]):
            for k in range(4):
                if bits & (8 >> k):
                    rgb[6 + j * 2 : 8 + j * 2, 6 + i * 10 + k * 2 : 8 + i * 10 + k * 2] = (55, 71, 84)
    return rgb


def _wearer(obj):
    return obj if obj.get("excad_character") else obj.get("excad_human")


def _fitted_contact(a, b, kind):
    person = _wearer(a.obj)
    if person is None or person != _wearer(b.obj):
        return None

    def slot(obj):
        return obj.get("excad_slot", obj.get("excad_asset_type", ""))

    facial = {"eyes", "lashes", "eyelashes", "eyebrows", "hair", "teeth"}
    if (a.obj == person and slot(b.obj) in facial) or (b.obj == person and slot(a.obj) in facial):
        return "attachment"
    layers = {slot(a.obj), slot(b.obj)}
    if layers in ({"hair"}, {"hair", "eyebrows"}):
        return "attachment"
    if kind == "crossing" and "hair" in layers and layers & {"top", "outerwear"}:
        hair, fabric = (a, b) if slot(a.obj) == "hair" else (b, a)
        if hair.obj.get("excad_draped_hair"):
            pairs = hair.bvh.overlap(fabric.bvh)
            if pairs:
                indices = np.unique(np.asarray(pairs)[:, 0])
                triangles = hair.points[hair.triangles[indices]]
                samples = np.concatenate(
                    (
                        triangles.reshape(-1, 3),
                        triangles.mean(axis=1),
                        ((triangles + np.roll(triangles, 1, axis=1)) / 2).reshape(-1, 3),
                    )
                )
                allowance = max(0.0001, float(np.ptp(fabric.bounds, axis=0).max()) * 0.004)
                winding = -1 if fabric.obj.matrix_world.to_3x3().determinant() < 0 else 1
                for p in samples:
                    point = Vector(p)
                    loc, n, _, _ = fabric.bvh.find_nearest(point)
                    if loc is None or winding * (point - loc).dot(n) < -allowance:
                        break
                else:
                    return "hair_on_fabric"
    if kind == "crossing" and layers in (
        {"top", "bottom"},
        {"outerwear", "bottom"},
        {"bottom", "footwear"},
        {"top", "outerwear"},
    ):
        pairs = a.bvh.overlap(b.bvh)
        if pairs:
            ia, ib = np.asarray(pairs).T
            if np.all(a.hem_faces()[ia] | b.hem_faces()[ib]):
                return "garment_seam"
    return None


def _near_surface(a, b, tolerance):
    # Test both ways so a small foot on a broad platform and a broad table on a
    # narrow post both find their contact. Restrict rays to the shared region.
    for subject, other, sign in ((a, b, -1), (b, a, 1)):
        low, high = other.bounds[0] - tolerance, other.bounds[1] + tolerance
        candidates = subject.points[np.all((subject.points >= low) & (subject.points <= high), axis=1)]
        for point in candidates:
            ray = Vector((0, 0, sign))
            if other.bvh.ray_cast(Vector(point) - ray * tolerance, ray, 2 * tolerance)[2] is not None:
                return True
        centers = subject.points[subject.triangles].mean(axis=1)
        for point in centers[np.all((centers >= low) & (centers <= high), axis=1)]:
            ray = Vector((0, 0, sign))
            if other.bvh.ray_cast(Vector(point) - ray * tolerance, ray, 2 * tolerance)[2] is not None:
                return True
    lift = max(0.0, float(b.bounds[1, 2] - a.bounds[0, 2]))
    clearance = scene_mesh.vertical_clearance(a, b, (0, 0, lift), best=lift + 2 * tolerance)
    return abs(clearance - lift) <= tolerance


def _diagnostics(items):
    warnings, contacts, topology = [], [], []
    for a, b in combinations(items, 2):
        kind = scene_mesh.overlap_kind(a, b)
        if kind:
            detail = {"objects": [a.obj.name, b.obj.name], "detail": kind}
            contact = _fitted_contact(a, b, kind)
            if contact:
                contacts.append({"kind": contact, **detail})
            else:
                warnings.append({"kind": "overlap", **detail})
    groups = {}
    for item in items:
        groups.setdefault(scene_mesh.assembly_root(item.obj), []).append(item)
        if item.boundary_edges or item.nonmanifold_edges:
            record = {
                "kind": "open_mesh",
                "object": item.obj.name,
                "open": item.boundary_edges,
                "nonmanifold": item.nonmanifold_edges,
            }
            # Hair cards, open hems and masked MPFB skin intentionally have boundaries.
            (topology if _wearer(item.obj) is not None else warnings).append(record)
    for root, parts in groups.items():
        lower = np.min([p.bounds[0] for p in parts], axis=0)
        upper = np.max([p.bounds[1] for p in parts], axis=0)
        extent = upper - lower
        label = next((s.obj.name for s in parts if s.obj.get("excad_character")), root.name)
        tolerance = max(1.0, float(extent.max())) * 1e-4
        if lower[2] > tolerance:
            other_parts = (p for key, group in groups.items() if key != root for p in group)
            supported = any(
                o.bounds[0, 2] < lower[2]
                and o.bounds[1, 2] >= lower[2] - tolerance
                and np.all(o.bounds[1, :2] + tolerance >= lower[:2])
                and np.all(o.bounds[0, :2] - tolerance <= upper[:2])
                and any(_near_surface(p, o, tolerance) for p in parts)
                for o in other_parts
            )
            if not supported:
                warnings.append({"kind": "above_floor", "object": label, "gap": round(float(lower[2]), 5)})
        if min(extent) > 1e-8 and max(extent) / min(extent) > 50:
            warnings.append({"kind": "thin", "object": label})
    return warnings, contacts, topology


def render_inspect(target="ALL", views=("FRONT", "SIDE", "ISO", "TOP"), path=None, size=160):
    start = time.perf_counter()
    if len(views) != 4 or any(v not in DIRECTIONS for v in views) or not 64 <= size <= 1024:
        raise ValueError("Use four FRONT/SIDE/ISO/TOP views and a tile size from 64 to 1024")
    items = scene_mesh.surfaces(target, shading=True)
    if not items:
        raise ValueError("No visible geometry to inspect")
    points, faces, offset = [], [], 0
    for item in items:
        points.append(item.points)
        faces.append(item.triangles + offset)
        offset += len(item.points)
    vertices = np.concatenate(points)
    bounds = np.stack((vertices.min(0), vertices.max(0)))
    tree = (
        items[0].bvh
        if len(items) == 1
        else BVHTree.FromPolygons(vertices.tolist(), np.concatenate(faces).tolist(), all_triangles=True)
    )
    shading = Shading(items)
    grid = np.full((size * 2 + 2, size * 2 + 2, 3), 255, np.uint8)
    for i, view in enumerate(views):
        y, x = (i // 2) * (size + 2), (i % 2) * (size + 2)
        grid[y : y + size, x : x + size] = _tile(tree, bounds, shading, view, size)
    warnings, contacts, topology = _diagnostics(items)
    dest = Path(bpy.path.abspath(path)).resolve() if path else None
    directory = dest.parent if dest else Path(tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, prefix="excad-inspect-", suffix=".png", delete=False) as tmp:
        pending = Path(tmp.name)
    try:
        _png(pending, grid)
        if dest is not None:
            pending.replace(dest)
        else:
            dest = pending
    except Exception:
        pending.unlink(missing_ok=True)
        raise
    result = {
        "path": str(dest),
        "views": list(views),
        "bbox": bounds.round(5).tolist(),
        "warnings": warnings,
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
    }
    if contacts:
        result["contacts"] = contacts
    if topology:
        result["surface_topology"] = topology
    return result
