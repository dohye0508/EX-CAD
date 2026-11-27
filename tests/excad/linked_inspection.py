# SPDX-License-Identifier: GPL-2.0-or-later
"""Linked geometry retains each copy's materials, normals, modifiers and current mesh edits."""

from pathlib import Path
import tempfile

import bpy
import numpy as np

from excad_addon.inspection import render_inspect
from excad_addon.scene_layout import place_relative
from excad_addon.scene_mesh import surfaces


for obj in list(bpy.context.scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24)
source = bpy.context.object
for poly in source.data.polygons:
    poly.use_smooth = True
glazes = []
for name, tint in (("Jade", (0.1, 0.45, 0.23)), ("Clay", (0.58, 0.14, 0.06))):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*tint, 1)
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*tint, 1)
    glazes.append(mat)
source.data.materials.append(glazes[0])
pots = []
for i in range(36):
    pot = source if i == 0 else source.copy()
    if i:
        bpy.context.collection.objects.link(pot)
    pot.name = f"Planter {i:02d}"
    pot.location = ((i % 6) * 1.1, (i // 6) * 1.1, 0.5)
    pot.scale = ((-0.35 if i % 3 else 0.40), 0.28, 0.5)
    pot.rotation_euler.z = i * 0.31
    pot.material_slots[0].link = "OBJECT"
    pot.material_slots[0].material = glazes[i % 2]
    pots.append(pot)
# A modifier gives this linked object a distinct evaluated topology.
subdiv = pots[4].modifiers.new("Subdivide only this pot", "SUBSURF")
subdiv.levels = 1
place_relative(pots[4].name, relation="ALIGN_FLOOR")
pots[-1].location.x -= 0.65
pots[1].location.z += 0.2
shared = surfaces(shading=True)
assert shared[0].triangles is shared[1].triangles
assert len(shared[4].triangles) > len(shared[0].triangles)
assert not np.array_equal(shared[0].colors[0], shared[1].colors[0])


def pixels(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        values = np.empty(len(image.pixels), np.float32)
        image.pixels.foreach_get(values)
        return values.reshape(-1, 4)[:, :3].copy()
    finally:
        bpy.data.images.remove(image)


with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "planters.png"
    before = render_inspect(path=str(path))
    reference = pixels(path)
    assert [(w["kind"], w.get("objects", w.get("object"))) for w in before["warnings"]] == [
        ("overlap", [pots[-2].name, pots[-1].name]), ("above_floor", pots[1].name)
    ], before["warnings"]

    # Edits after a query must invalidate both the preview and its world bounds.
    for vert in source.data.vertices:
        vert.co.x *= 1.3
    source.data.update()
    edited = render_inspect(path=str(path))
    expected = pixels(path)
    assert edited["bbox"] != before["bbox"] and np.count_nonzero(np.abs(reference - expected) > 0.05) > 100

    # Baking each mesh into a private datablock should produce the same picture.
    for pot in pots[1:]:
        pot.data = pot.data.copy()
    independent = render_inspect(path=str(path))
    assert independent["bbox"] == edited["bbox"] and independent["warnings"] == edited["warnings"]
    error = np.max(np.abs(expected - pixels(path)), axis=1)
    assert np.count_nonzero(error > 0.02) <= 8, (np.count_nonzero(error > 0.02), error.max())
print("LINKED INSPECTION PASS", bpy.app.version_string, before["elapsed_ms"], edited["elapsed_ms"])
