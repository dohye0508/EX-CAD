# SPDX-License-Identifier: GPL-2.0-or-later
"""Compact material/normal samples for diagnostic ray casts in the headless inspector."""

from collections import OrderedDict
from pathlib import Path

import bpy
import numpy as np

_thumbnails = OrderedDict()


def _texture(mat):
    if not mat or not mat.use_nodes:
        return None
    images = []
    for node in mat.node_tree.nodes:
        image = getattr(node, "image", None)
        if node.type != "TEX_IMAGE" or image is None:
            continue
        label = (node.name + " " + node.label + " " + image.name).lower()
        if any(s in label for s in ("normal", "roughness", "bump", "displacement", "specular", "metallic", "opacity")):
            continue
        score = 2 if any(s in label for s in ("diffuse", "albedo", "base color", "basecolor")) else 1
        images.append((score, image))
    return max(images, key=lambda x: x[0])[1] if images else None


def thumbnail(image):
    if image is None or not image.size[0]:
        return None
    revision = None
    if image.source == "FILE" and image.filepath and not image.packed_file:
        try:
            revision = Path(bpy.path.abspath(image.filepath, library=image.library)).stat().st_mtime_ns
        except OSError:
            pass
    key = image.session_uid, tuple(image.size), image.filepath, revision
    if not image.is_dirty and key in _thumbnails:
        _thumbnails.move_to_end(key)
        return _thumbnails[key]
    duplicate = None
    try:
        original = image
        if max(image.size) > 384 and not image.is_dirty:
            duplicate = image.copy()
            w, h = image.size
            duplicate.scale(max(1, round(w * 384 / max(w, h))), max(1, round(h * 384 / max(w, h))))
            image = duplicate
        w, h = image.size
        pixels = np.empty((h, w, 4), np.float32)
        image.pixels.foreach_get(pixels.ravel())
        if max(w, h) > 384:
            stride = (max(w, h) + 383) // 384
            pixels = pixels[::stride, ::stride].copy()
        # Generated and packed images also pass through Blender's colorspace conversion.
        if not original.is_dirty:
            _thumbnails[key] = pixels
            if len(_thumbnails) > 24:
                _thumbnails.popitem(last=False)
        return pixels
    finally:
        if duplicate:
            bpy.data.images.remove(duplicate)


def warm_materials(obj):
    for material in obj.data.materials:
        thumbnail(_texture(material))


def surface_shading(mesh, matrix, slots):
    indices = np.empty((len(mesh.loop_triangles), 3), np.int32)
    mesh.loop_triangles.foreach_get("loops", indices.ravel())
    normals = np.empty((len(mesh.corner_normals), 3), np.float32)
    mesh.corner_normals.foreach_get("vector", normals.ravel())
    normals = normals @ np.linalg.pinv(matrix[:3, :3])
    normals /= np.maximum(1e-9, np.linalg.norm(normals, axis=1, keepdims=True))
    images = [_texture(m) for m in mesh.materials]
    uv = np.zeros((len(mesh.loops), 2), np.float32)
    if mesh.uv_layers.active and any(i is not None for i in images):
        mesh.uv_layers.active.uv.foreach_get("vector", uv.ravel())
    else:
        images = []
    textures = []
    for mat, im in zip(mesh.materials, images):
        pixels = thumbnail(im)
        if pixels is not None and mat.get("excad_albedo_multiplier"):
            pixels = pixels.copy()
            pixels[:, :, :3] *= np.array(mat["excad_albedo_multiplier"])
        textures.append(pixels)
    lip = None
    attribute = mesh.color_attributes.get("EXCAD Lips")
    if attribute and attribute.domain == "POINT":
        tints = np.zeros((max(1, len(mesh.materials)), 3), np.float32)
        for i, material in enumerate(mesh.materials):
            node = material.node_tree.nodes.get("Lip tint") if material and material.use_nodes else None
            if node:
                tints[i] = node.inputs[2].default_value[:3]
        colors = np.empty((len(attribute.data), 4), np.float32)
        attribute.data.foreach_get("color", colors.ravel())
        vertices = np.empty_like(indices)
        mesh.loop_triangles.foreach_get("vertices", vertices.ravel())
        lip = colors[vertices, 0], tints[np.clip(slots, 0, len(tints) - 1)]
    return normals[indices], uv[indices], slots, textures, lip


class Shading:
    def __init__(self, items):
        self.tris = np.concatenate([s.points[s.triangles] for s in items])
        self.normals = np.concatenate([s.shading[0] for s in items])
        self.uv = np.concatenate([s.shading[1] for s in items])
        self.colors = np.concatenate([s.colors for s in items])
        self.textures = []
        ids = []
        for surface in items:
            _, _, slots, textures, _ = surface.shading
            mapping = []
            for t in textures:
                if t is None:
                    mapping.append(-1)
                else:
                    mapping.append(len(self.textures))
                    self.textures.append(t)
            ids.append(
                np.array(mapping, np.int32)[np.clip(slots, 0, len(mapping) - 1)]
                if mapping
                else np.full(len(slots), -1, np.int32)
            )
        self.ids = np.concatenate(ids)
        self.lip = None
        if any(s.shading[4] is not None for s in items):
            masks, colors = [], []
            for s in items:
                mask, tint = s.shading[4] or (None, None)
                masks.append(mask if mask is not None else np.zeros((len(s.triangles), 3), np.float32))
                colors.append(tint if tint is not None else np.zeros((len(s.triangles), 3), np.float32))
            self.lip = np.concatenate(masks), np.concatenate(colors)

    def sample(self, faces, points):
        triangles = self.tris[faces]
        a = triangles[:, 1] - triangles[:, 0]
        b = triangles[:, 2] - triangles[:, 0]
        p = points - triangles[:, 0]

        def dot(u, v):
            return np.einsum("ij,ij->i", u, v)

        aa, ab, bb, pa, pb = dot(a, a), dot(a, b), dot(b, b), dot(p, a), dot(p, b)
        inv = 1 / np.maximum(aa * bb - ab * ab, 1e-20)
        u, v = (bb * pa - ab * pb) * inv, (aa * pb - ab * pa) * inv
        weights = np.stack((1 - u - v, u, v), axis=1)
        normals = np.einsum("ij,ijk->ik", weights, self.normals[faces])
        normals /= np.maximum(1e-9, np.linalg.norm(normals, axis=1, keepdims=True))
        colors = self.colors[faces].copy()
        alpha = np.ones(len(faces), np.float32)
        if self.textures:
            ids = self.ids[faces]
            coords = np.einsum("ij,ijk->ik", weights, self.uv[faces]) % 1
            for i in np.unique(ids[ids >= 0]):
                tex = self.textures[i]
                sel = ids == i
                x, y = (coords[sel] * [tex.shape[1] - 1, tex.shape[0] - 1]).astype(np.int32).T
                samples = tex[y, x]
                colors[sel], alpha[sel] = samples[:, :3], samples[:, 3]
        if self.lip:
            mask, tint = self.lip
            mix = np.sum(weights * mask[faces], axis=1, keepdims=True).clip(0, 1)
            colors = colors * (1 - mix) + tint[faces] * mix
        return normals, colors, alpha
