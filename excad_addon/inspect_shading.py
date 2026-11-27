# SPDX-License-Identifier: GPL-2.0-or-later
"""Compact material/normal samples for diagnostic ray casts in the headless inspector."""

from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import tempfile

import bpy
import numpy as np

from .image_colors import has_srgb_bytes, srgb_to_linear

_thumbnails = OrderedDict()


def _principled(mat):
    if mat and mat.use_nodes:
        for shader in mat.node_tree.nodes:
            if shader.type == "BSDF_PRINCIPLED" and shader.outputs[0].is_linked:
                return shader
    return None


def material_color(mat, fallback=(0.42, 0.57, 0.70)):
    if not mat:
        return fallback
    shader = _principled(mat)
    socket = shader.inputs["Base Color"] if shader else None
    return socket.default_value[:3] if socket and not socket.is_linked else mat.diffuse_color[:3]


def _texture_node(mat):
    if not mat or not mat.use_nodes:
        return None
    shader = _principled(mat)
    socket = shader.inputs["Base Color"] if shader else None
    if socket is not None:
        queue = [link.from_node for link in socket.links]
        seen = set()
        while queue:
            node = queue.pop(0)
            if node in seen:
                continue
            seen.add(node)
            if node.type == "TEX_IMAGE" and node.image:
                return node
            queue.extend(link.from_node for input in node.inputs for link in input.links)
        if not socket.is_linked:
            return None
    # MPFB and other imported shaders can wrap their color inputs in node groups.
    images = []
    for node in mat.node_tree.nodes:
        if node.type != "TEX_IMAGE" or not node.image:
            continue
        label = (node.name + " " + node.label + " " + node.image.name).lower()
        if any(s in label for s in ("normal", "roughness", "bump", "displacement", "specular", "metallic", "opacity")):
            continue
        score = 2 if any(s in label for s in ("diffuse", "albedo", "base color", "basecolor")) else 1
        images.append((score, node))
    return max(images, key=lambda x: x[0])[1] if images else None


def _thumbnail_file(path, revision, colorspace, alpha_mode):
    root = os.getenv("EXCAD_CACHE") or bpy.utils.user_resource("DATAFILES", path="excad")
    identity = (os.path.normcase(str(path.resolve())), revision, colorspace, alpha_mode)
    name = hashlib.sha256(repr(identity).encode("utf8")).hexdigest()
    return Path(root) / "thumbnails-v1" / (name + ".npy")


def _store_thumbnail(path, pixels):
    pending = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
            pending = Path(out.name)
            np.save(out, pixels.astype(np.float16), allow_pickle=False)
        pending.replace(path)
    except OSError:
        pass
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def thumbnail(image, persist=False):
    if image is None:
        return None
    colorspace = image.colorspace_settings.name
    path = None
    revision = None
    can_read_file = image.source == "FILE" and not image.is_dirty
    if can_read_file and image.filepath and not image.packed_file:
        path = Path(bpy.path.abspath(image.filepath, library=image.library))
        try:
            stat = path.stat()
            revision = stat.st_size, stat.st_mtime_ns
        except OSError:
            path = None
    key = image.session_uid, image.filepath, revision, colorspace, image.alpha_mode
    if not image.is_dirty and key in _thumbnails:
        _thumbnails.move_to_end(key)
        return _thumbnails[key]

    pixels = None
    cached = _thumbnail_file(path, revision, colorspace, image.alpha_mode) if path else None
    if cached and cached.is_file():
        try:
            array = np.load(cached, allow_pickle=False, mmap_mode="r")
            if array.ndim == 3 and array.shape[2] == 4 and max(array.shape[:2]) <= 384 and array.dtype == np.float16:
                pixels = array.astype(np.float32)
        except (OSError, ValueError, EOFError):
            pass
    restored = pixels is not None
    if pixels is None and can_read_file and (path or image.packed_file) and colorspace in {"sRGB", "Non-Color"}:
        # Access a thumbnail directly, without decoding a second copy into Blender's
        # image database. Packed and external images use Blender's own file decoder.
        import imbuf

        buffer = None
        try:
            buffer = imbuf.load(str(path)) if path else imbuf.load_from_buffer(image.packed_file.data)
            w, h = buffer.size
            if buffer.buffer_type == "BYTE" and buffer.channels == 4:
                if max(w, h) > 384:
                    buffer.resize(
                        (max(1, round(w * 384 / max(w, h))), max(1, round(h * 384 / max(w, h)))), method="BILINEAR"
                    )
                with buffer.with_buffer() as view:
                    pixels = np.asarray(view, dtype=np.float32) / 255
                if colorspace == "sRGB":
                    pixels[:, :, :3] = srgb_to_linear(pixels[:, :, :3])
        except (OSError, ValueError, RuntimeError):
            pass
        finally:
            if buffer is not None:
                buffer.free()
    if pixels is None:
        w, h = image.size
        if not w or not h:
            return None
        pixels = np.empty((h, w, 4), np.float32)
        image.pixels.foreach_get(pixels.ravel())
        if max(w, h) > 384:
            stride = (max(w, h) + 383) // 384
            pixels = pixels[::stride, ::stride].copy()
        if has_srgb_bytes(image):
            pixels[:, :, :3] = srgb_to_linear(pixels[:, :, :3])
    if not image.is_dirty:
        _thumbnails[key] = pixels
        if len(_thumbnails) > 24:
            _thumbnails.popitem(last=False)
        if persist and cached and not restored:
            _store_thumbnail(cached, pixels)
    return pixels


def warm_materials(obj):
    for slot in obj.material_slots:
        if (node := _texture_node(slot.material)) is not None:
            thumbnail(node.image, persist=True)


def surface_shading(mesh, matrix, slots, materials, indices, triangles):
    normals = np.empty((len(mesh.corner_normals), 3), np.float32)
    mesh.corner_normals.foreach_get("vector", normals.ravel())
    normals = normals @ np.linalg.pinv(matrix[:3, :3])
    normals /= np.maximum(1e-9, np.linalg.norm(normals, axis=1, keepdims=True))
    default = next((layer for layer in mesh.uv_layers if layer.active_render), mesh.uv_layers.active)
    uv = np.zeros((len(slots), 3, 2), np.float32)
    coordinates = {}
    textures = []
    for i, mat in enumerate(materials):
        node = _texture_node(mat)
        layer = default
        if node is not None:
            for link in node.inputs["Vector"].links:
                if link.from_node.type == "UVMAP" and link.from_node.uv_map:
                    layer = mesh.uv_layers.get(link.from_node.uv_map)
        pixels = thumbnail(node.image) if node and layer else None
        if pixels is not None:
            if layer.name not in coordinates:
                values = np.empty((len(mesh.loops), 2), np.float32)
                layer.uv.foreach_get("vector", values.ravel())
                coordinates[layer.name] = values
            select = slots == i
            uv[select] = coordinates[layer.name][indices[select]]
        if pixels is not None and mat.get("excad_albedo_multiplier"):
            pixels = pixels.copy()
            pixels[:, :, :3] *= np.array(mat["excad_albedo_multiplier"])
        textures.append(pixels)
    tint_mask = None
    attribute = mesh.color_attributes.get("EXCAD Lips")
    hem = mesh.color_attributes.get("EXCAD Hem distance")
    layer = attribute or hem
    if layer and layer.domain == "POINT":
        tints = np.zeros((max(1, len(materials)), 3), np.float32)
        enabled = np.zeros(len(tints), np.bool_)
        starts = np.zeros(len(tints), np.float32)
        ends = np.ones(len(tints), np.float32)
        for i, material in enumerate(materials):
            nodes = material.node_tree.nodes if material and material.use_nodes else None
            node = nodes.get("Lip tint" if attribute else "Hem tint") if nodes else None
            if node and not attribute and not nodes.get("Hem width"):
                continue
            if node:
                enabled[i] = True
                tints[i] = node.inputs[2].default_value[:3]
                if not attribute:
                    width = nodes["Hem width"]
                    starts[i], ends[i] = width.inputs["From Max"].default_value, width.inputs["From Min"].default_value
        colors = np.empty((len(layer.data), 4), np.float32)
        layer.data.foreach_get("color", colors.ravel())
        material_ids = np.clip(slots, 0, len(tints) - 1)
        span = (ends - starts)[material_ids, None]
        # Clamp after barycentric interpolation so the trim has a measured width
        # independent of the size or orientation of the garment's polygons.
        mask = np.divide(
            colors[triangles, 0] - starts[material_ids, None],
            span,
            out=np.zeros((len(triangles), 3), np.float32),
            where=np.abs(span) > 1e-8,
        )
        tint_mask = mask * enabled[material_ids, None], tints[material_ids]
    finish = []
    for mat in materials:
        shader = _principled(mat)
        if shader:
            finish.append((shader.inputs["Roughness"].default_value, shader.inputs["Metallic"].default_value))
        else:
            finish.append((mat.roughness, mat.metallic) if mat else (0.6, 0))
    palette = np.asarray(finish or [(0.6, 0)], np.float32)
    return normals[indices], uv, slots, textures, tint_mask, palette[np.clip(slots, 0, len(palette) - 1)]


class Shading:
    def __init__(self, items):
        self.tris = np.concatenate([s.points[s.triangles] for s in items])
        self.normals = np.concatenate([s.shading[0] for s in items])
        self.uv = np.concatenate([s.shading[1] for s in items])
        self.colors = np.concatenate([s.colors for s in items])
        self.finish = np.concatenate([s.shading[5] for s in items])
        self.textures = []
        ids = []
        for surface in items:
            _, _, slots, textures, _, _ = surface.shading
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
        self.tint = None
        if any(s.shading[4] is not None for s in items):
            masks, colors = [], []
            for s in items:
                mask, tint = s.shading[4] or (None, None)
                masks.append(mask if mask is not None else np.zeros((len(s.triangles), 3), np.float32))
                colors.append(tint if tint is not None else np.zeros((len(s.triangles), 3), np.float32))
            self.tint = np.concatenate(masks), np.concatenate(colors)

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
        if self.tint:
            mask, tint = self.tint
            mix = np.sum(weights * mask[faces], axis=1, keepdims=True).clip(0, 1)
            colors = colors * (1 - mix) + tint[faces] * mix
        return normals, colors, alpha
