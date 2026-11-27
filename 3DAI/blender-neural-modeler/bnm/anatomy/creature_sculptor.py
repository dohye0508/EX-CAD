"""Schema-driven organic sculpt (animals, creatures, monsters) from a sphere.

A schema lists parts with a location (in head-normalised coords), a growth action and size;
the plan is generated from the schema + current measured surface, not stored coordinates.
Example schema 'quadruped_head' (dog/bear/wolf-like) below; LLM can author new schemas as JSON.
"""
import numpy as np

from ..sculpt.stroke import line, catmull_rom
from ..sculpt.multiresolution import create_primitive, voxel_remesh
from ..geometry.semantic import SemanticMap, surface_point

QUADRUPED_HEAD = {
    "base": {"scale": (0.85, 1.0, 0.8)},
    "parts": [  # name, action, anchor (x, z, ray), params
        {"name": "snout", "action": "pull", "anchor": (0.0, -0.25, "front"), "dir": (0, -1, -0.15), "length": 0.75, "radius": 0.42},
        {"name": "snout_taper", "action": "pinch_line", "anchor": (0.0, -0.3, "front"), "length": 0.6, "radius": 0.2},
        {"name": "ear", "action": "pull", "anchor": (0.42, 0.62, "front"), "dir": (0.35, 0.15, 1.0), "length": 0.45, "radius": 0.16, "mirror": True},
        {"name": "brow", "action": "clay_band", "anchor": (0.0, 0.25, "front"), "span": 0.55, "radius": 0.12},
        {"name": "eye_socket", "action": "carve", "anchor": (0.32, 0.12, "front"), "radius": 0.1, "mirror": True},
        {"name": "eye", "action": "eyeball", "anchor": (0.32, 0.12, "front"), "radius": 0.08, "mirror": True},
        {"name": "nose_pad", "action": "add", "anchor": (0.0, -0.32, "front_far"), "radius": 0.14},
        {"name": "nostril", "action": "carve", "anchor": (0.06, -0.33, "front_far"), "radius": 0.035, "mirror": True},
        {"name": "mouth_line", "action": "crease_side", "anchor": (0.18, -0.5, "front"), "length": 0.7, "radius": 0.04, "mirror": True},
        {"name": "cheek", "action": "add", "anchor": (0.5, -0.2, "front"), "radius": 0.22, "mirror": True},
        {"name": "jaw", "action": "add", "anchor": (0.0, -0.62, "front"), "radius": 0.2},
    ],
}


def _anchor(S, a):
    x, z, ray = a
    if ray == "front_far":  # frontmost point near (x, z) after the snout has grown
        co = S.st.co
        sel = (np.abs(co[:, 0] - x) < 0.06) & (np.abs(co[:, 2] - z) < 0.12)
        if not sel.any():
            return None, None
        i = np.nonzero(sel)[0][np.argmin(co[sel, 1])]
        return co[i].copy(), S.st.normals[i].copy()
    hit = surface_point(S.st, x, z, ray)
    return (None, None) if hit is None else (hit[0], hit[1])


def sculpt_creature(S, schema=QUADRUPED_HEAD, remesh_voxel=0.02):
    """S: Sculptor on an icosphere. Returns per-part reports."""
    sc = schema.get("base", {}).get("scale")
    reps = []
    if sc:
        # blocking: elastic grabs instead of object scaling (stay a sculpt operation)
        for ax, f in enumerate(sc):
            if abs(f - 1) < 1e-3:
                continue
            p = np.zeros(3)
            p[ax] = 1.0
            hit = S.st.co[np.argmax(S.st.co[:, ax])]
            d = np.zeros(3)
            d[ax] = (f - 1) * abs(hit[ax])
            S.stroke("ELASTIC_DEFORM", [hit, hit + d], 0.7, 1.0, symmetry_x=(ax == 0), region="blocking")
    for part in schema["parts"]:
        mirror = part.get("mirror", False)
        p, n = _anchor(S, part["anchor"])
        if p is None:
            reps.append({"part": part["name"], "skipped": "anchor not on surface"})
            continue
        a = part["action"]
        r = part["radius"]
        if a == "pull":
            d = np.asarray(part["dir"], float)
            d /= np.linalg.norm(d)
            steps = 3
            for k in range(steps):
                p, n = _anchor(S, part["anchor"]) if k == 0 else (p, n)
                tgt = p + d * part["length"] / steps
                rep = S.stroke("SNAKE_HOOK", line(p, tgt, 12), r * (1 - 0.15 * k), 0.95, symmetry_x=mirror, region=part["name"])
                p = tgt
        elif a == "pinch_line":
            path = [p + [0, -t, 0] for t in np.linspace(0, part["length"], 6)]
            rep = S.stroke("PINCH", np.array(path), r, 0.3, normal=(0, 0, 1), region=part["name"])
        elif a == "clay_band":
            path = S.surface_path([(x, part["anchor"][1]) for x in np.linspace(-part["span"], part["span"], 9)])
            rep = S.stroke("CLAY_STRIPS", path, r, 0.5, region=part["name"])
        elif a == "carve":
            rep = S.stroke("DRAW_SUBTRACT", [p], r, 0.8, symmetry_x=mirror, region=part["name"])
        elif a == "add":
            rep = S.stroke("CLAY", [p, p + n * 0.001], r, 0.6, symmetry_x=mirror, region=part["name"])
        elif a == "crease_side":
            pts = []
            for t in np.linspace(0, part["length"], 6):
                co = S.st.co
                sel = (np.abs(co[:, 1] - (p[1] - t)) < 0.03) & (co[:, 0] > 0.05) & (np.abs(co[:, 2] - p[2]) < 0.2)
                if sel.any():
                    i = np.nonzero(sel)[0][np.argmax(co[sel, 0])]
                    pts.append(co[i])
            if len(pts) > 1:
                rep = S.stroke("CREASE", catmull_rom(np.array(pts), 4), r, 0.6, symmetry_x=mirror, region=part["name"])
            else:
                rep = {"skipped": True}
        elif a == "eyeball":
            for sx in ((1, -1) if mirror else (1,)):
                c = p * [sx, 1, 1] + np.array([0, r * 0.9, 0])
                ob = create_primitive("UV_SPHERE", f"BNM_{part['name']}_{'L' if sx > 0 else 'R'}", r, tuple(c), detail=3, log=S.log)
                S.extras.append(ob)
            rep = {"eyeballs": 2 if mirror else 1}
        else:
            rep = {"skipped": f"unknown action {a}"}
        reps.append({"part": part["name"], "action": a, "moved": rep.get("moved_vertices") if isinstance(rep, dict) else None})
        if a == "pull" and remesh_voxel:
            voxel_remesh(S.obj, remesh_voxel, log=S.log)   # restore even density after big pulls
            S.topology_changed()
    S.stroke("SURFACE_SMOOTH", S.surface_path([(x, z) for z in np.linspace(0.5, -0.6, 5) for x in (-0.4, 0, 0.4)]), 0.2, 0.3)
    return reps


def creature_map(obj, schema=QUADRUPED_HEAD):
    """Semantic map for creature parts (same machinery as faces)."""
    lms, regs = {}, {}
    for part in schema["parts"]:
        x, z, ray = part["anchor"]
        ray = "front" if ray == "front_far" else ray
        nm = part["name"] + ("_L" if part.get("mirror") else "")
        lms[nm] = (x, z, ray)
        if part.get("mirror"):
            lms[part["name"] + "_R"] = (-x, z, ray)
        r = part["radius"] * 1.5
        regs[part["name"]] = (nm, (r, r * 2, r), "front")
    return SemanticMap(obj, lms, regs)
