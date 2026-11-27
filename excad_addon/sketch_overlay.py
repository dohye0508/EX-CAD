# SPDX-License-Identifier: GPL-2.0-or-later
"""Draw the active parametric sketch in each 3D view; cache solver diagnostics between edits."""

import bpy
import numpy as np
from mathutils import Vector

from .sketch_engine import SketchDocument
from .sketch_profiles import paths_and_loops

_handles = []
_cache_key = None
_cache = None


def constrained_variables(doc):
    n = 2*len(doc.points)
    state = np.array([v for p in doc.points for v in (p.u, p.v)] + [e.radius for e in doc.entities])
    free = [i for i in range(n) if not doc.points[i//2].fixed]
    free += [n+e.eid for e in doc.entities if e.etype in {"CIRCLE", "ARC"} and "through" not in e.params]
    constrained = set(range(len(state))) - set(free)
    if not free or not doc.constraints:
        return constrained
    r = doc._eval_residuals(state)
    if np.max(np.abs(r), initial=0) > 1e-5:
        return constrained
    jac = np.empty((len(r), len(free)))
    for col, index in enumerate(free):
        delta = state.copy()
        delta[index] += 1e-6
        jac[:, col] = (doc._eval_residuals(delta)-r)/1e-6
    _, singular, vh = np.linalg.svd(jac, full_matrices=True)
    nullspace = vh[np.count_nonzero(singular > 1e-5):]
    return constrained | {index for col, index in enumerate(free) if np.linalg.norm(nullspace[:, col]) < 1e-5}


def drawing():
    global _cache_key, _cache
    doc = SketchDocument.active()
    key = SketchDocument._source
    if key == _cache_key:
        return _cache
    fixed = constrained_variables(doc)
    lines = {False: [], True: []}
    labels = []
    previous_states = []
    for e, pts, closed in paths_and_loops(doc)[0]:
        ids = {i for i in (e.p0, e.p1, e.params.get("through", -1), *e.params.get("points", [])) if i >= 0}
        locked = all({2*i, 2*i+1} <= fixed for i in ids)
        if e.etype in {"ARC", "CIRCLE"} and "through" not in e.params:
            locked = locked and 2*len(doc.points)+e.eid in fixed
        if not ids:
            locked = all(previous_states)
        previous_states.append(locked)
        world = [doc.plane_to_world(*p, 0) for p in pts]
        for a, b in zip(world, world[1:] + ([world[0]] if closed else [])):
            lines[locked].extend((a, b))
    for c in doc.constraints:
        if c.ctype not in {"RADIUS", "DISTANCE", "ANGLE"}:
            continue
        a = doc.points[doc.entities[c.a].p0] if c.ctype == "RADIUS" else doc.points[c.a]
        b = a if c.ctype == "RADIUS" else doc.points[c.b]
        pos = doc.plane_to_world((a.u+b.u)/2, (a.v+b.v)/2, 0)
        label = {"RADIUS": "R ", "DISTANCE": "", "ANGLE": "°"}[c.ctype]
        labels.append((Vector(pos), f"{c.value:g}°" if c.ctype == "ANGLE" else f"{label}{c.value:g}"))
    _cache_key, _cache = key, (lines, labels)
    return _cache


def _draw_lines():
    import gpu
    from gpu_extras.batch import batch_for_shader
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    for locked, verts in drawing()[0].items():
        if verts:
            shader.bind()
            shader.uniform_float("color", (0.02, 0.02, 0.02, 1) if locked else (0.12, 0.42, 1, 1))
            batch_for_shader(shader, "LINES", {"pos": verts}).draw(shader)


def _draw_text():
    import blf
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    context = bpy.context
    if context.region_data is None:
        return
    for pos, label in drawing()[1]:
        xy = location_3d_to_region_2d(context.region, context.region_data, pos)
        if xy:
            blf.position(0, xy.x+3, xy.y+4, 0)
            blf.size(0, 12)
            blf.color(0, 0.9, 0.85, 0.4, 1)
            blf.draw(0, label)


def register():
    if not bpy.app.background and not _handles:
        for func, stage in ((_draw_lines, "POST_VIEW"), (_draw_text, "POST_PIXEL")):
            _handles.append(bpy.types.SpaceView3D.draw_handler_add(func, (), "WINDOW", stage))


def unregister():
    while _handles:
        bpy.types.SpaceView3D.draw_handler_remove(_handles.pop(), "WINDOW")
