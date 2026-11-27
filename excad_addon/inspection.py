# SPDX-License-Identifier: GPL-2.0-or-later
"""Quick orthographic inspection. Ray cast evaluated triangles without touching the user's scene."""

from pathlib import Path
import struct
import tempfile
import time
import zlib

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import scene_mesh

DIRECTIONS = {"FRONT": (0, -1, 0), "SIDE": (1, 0, 0), "ISO": (1.15, -1.65, 1.05), "TOP": (0, 0, 1)}
# Five-pixel labels keep the orientations legible in the standalone PNG.
GLYPHS = {
    "F": (15,8,14,8,8), "R": (14,9,14,10,9), "O": (6,9,9,9,6), "N": (9,13,11,9,9),
    "T": (15,6,6,6,6), "S": (7,8,6,1,14), "I": (14,4,4,4,14), "D": (14,9,9,9,14),
    "E": (15,8,14,8,15), "P": (14,9,14,8,8),
}


def _png(path, rgb):
    def chunk(tag, payload):
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag+payload))
    height, width = rgb.shape[:2]
    data = np.zeros((height, 1+3*width), dtype=np.uint8)
    data[:, 1:] = rgb.reshape(height, -1)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", zlib.compress(data.tobytes(), 2)) + chunk(b"IEND", b""))


def _tile(tree, bounds, colors, view, size):
    eye = Vector(DIRECTIONS[view]).normalized()
    right = Vector((1, 0, 0)) if view == "TOP" else Vector((0, 0, 1)).cross(eye).normalized()
    up = eye.cross(right)
    corners = np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
    step = max(np.ptp(corners @ right), np.ptp(corners @ up), 1e-6) / (size*0.84)
    distance = float(np.linalg.norm(bounds[1]-bounds[0])) + 1
    center = Vector(bounds.mean(axis=0)) + eye*distance
    start = center + (up-right)*((size-1)*step/2)
    rows = start, -up*step, right*step
    light = (eye*0.8 + up*0.9 - right*0.55).normalized()
    ids = np.full((size, size), -1, dtype=np.int32)
    shade = np.zeros((size, size), dtype=np.float32)
    ray = -eye
    for y in range(size):
        p = rows[0] + y*rows[1]
        for x in range(size):
            _, normal, index, _ = tree.ray_cast(p, ray)
            if index is not None:
                ids[y, x] = index
                shade[y, x] = 0.34 + 0.65*max(0, normal.dot(light))
            p += rows[2]
    hit = ids >= 0
    rgb = np.empty((size, size, 3), np.uint8)
    bg = np.linspace((239, 244, 248), (216, 225, 233), size)
    rgb[:] = bg[:, None]
    linear = colors[ids[hit]] * shade[hit, None]
    rgb[hit] = np.clip(255 * np.power(np.maximum(0, linear), 1/2.2), 0, 255).astype(np.uint8)
    for i, char in enumerate(view):
        for j, bits in enumerate(GLYPHS[char]):
            for k in range(4):
                if bits & (8 >> k):
                    rgb[6+j*2:8+j*2, 6+i*10+k*2:8+i*10+k*2] = (55,71,84)
    return rgb


def render_inspect(target="ALL", views=("FRONT", "SIDE", "ISO", "TOP"), path=None, size=160):
    start = time.perf_counter()
    if len(views) != 4 or any(v not in DIRECTIONS for v in views) or not 64 <= size <= 1024:
        raise ValueError("Use four FRONT/SIDE/ISO/TOP views and a tile size from 64 to 1024")
    items = scene_mesh.surfaces(target)
    if not items:
        raise ValueError("No visible geometry to inspect")
    points, faces, offset = [], [], 0
    for item in items:
        points.append(item.points)
        faces.append(item.triangles + offset)
        offset += len(item.points)
    vertices = np.concatenate(points)
    bounds = np.stack((vertices.min(0), vertices.max(0)))
    tree = (items[0].bvh if len(items) == 1 else
            BVHTree.FromPolygons(vertices.tolist(), np.concatenate(faces).tolist(), all_triangles=True))
    colors = np.concatenate([s.colors for s in items])
    grid = np.full((size*2 + 2, size*2 + 2, 3), 255, np.uint8)
    for i, view in enumerate(views):
        y, x = (i//2)*(size+2), (i%2)*(size+2)
        grid[y:y+size, x:x+size] = _tile(tree, bounds, colors, view, size)
    if path is None:
        with tempfile.NamedTemporaryFile(prefix="excad-inspect-", suffix=".png", delete=False) as tmp:
            path = tmp.name
    dest = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    _png(dest, grid)
    warnings = [{"kind": "overlap", "objects": hit["objects"], "detail": hit["kind"]} for hit in scene_mesh.intersections(items)]
    for item in items:
        a, b = item.bounds
        if item.boundary_edges or item.nonmanifold_edges:
            warnings.append({"kind": "open_mesh", "object": item.obj.name,
                             "open": item.boundary_edges, "nonmanifold": item.nonmanifold_edges})
        if a[2] > 1e-4 and not any(other is not item and other.bounds[0,2] < a[2]
                and other.bounds[1,2] >= a[2]-1e-4 and np.all(other.bounds[1,:2] >= a[:2])
                and np.all(other.bounds[0,:2] <= b[:2]) for other in items):
            warnings.append({"kind": "above_floor", "object": item.obj.name, "gap": round(float(a[2]), 5)})
        d = b-a
        if min(d) > 1e-8 and max(d)/min(d) > 50:
            warnings.append({"kind": "thin", "object": item.obj.name})
    return {"path": str(dest), "views": list(views), "bbox": bounds.round(5).tolist(), "warnings": warnings,
            "elapsed_ms": round((time.perf_counter()-start)*1000, 1)}
