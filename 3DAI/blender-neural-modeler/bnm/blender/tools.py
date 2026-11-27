"""In-Blender implementations of every MCP tool (schemas in tool_schemas.py).

Each tool takes a JSON dict and returns a JSON-safe dict.  Tools that perform several strokes may
return a generator: the live bridge steps it once per viewport tick (so the user sees the sculpt
progress) and the headless runner simply exhausts it.  The generator's return value is the result.
"""
import os
import time
import json
import numpy as np
import bpy
from mathutils import Vector

from .session import Session
from .tool_schemas import TOOLS as SCHEMAS
from ..sculpt.stroke import build_path, catmull_rom, bezier, fill_region
from ..sculpt import multiresolution as MR
from ..geometry.mesh_io import MeshState
from ..geometry import analyzer as AN
from ..geometry.curvature import curvature, curvature_stats, feature_protection
from ..geometry.semantic import ray_hit
from ..sculpt.geodesic import geodesic_ball
from ..validation.geometry_validator import validate_mesh as _validate
from ..perception.camera import CameraModel, VIEWS
from ..perception import render as RN
from ..perception import picker as PK

REGISTRY = {}


def tool(fn):
    REGISTRY[fn.__name__] = fn
    return fn


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items() if not str(k).startswith("_buf")}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 2000 else {"array_shape": list(o.shape)}
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, Vector):
        return list(o)
    return o


def call(name, args):
    if name not in REGISTRY:
        raise KeyError(f"unknown tool {name}")
    return REGISTRY[name](args or {})


def run_to_completion(res):
    """Exhaust a generator tool (headless)."""
    if hasattr(res, "__next__"):
        try:
            while True:
                next(res)
        except StopIteration as e:
            return e.value
    return res


# ======================================================================= helpers
def _S(a, schema=None):
    return Session.get().sculptor(a.get("object"), schema)


def _to_obj_space(S, pts, space):
    P = np.asarray(pts, float).reshape(-1, 3)
    if space == "WORLD":
        mi = np.array(S.obj.matrix_world.inverted())
        P = P @ mi[:3, :3].T + mi[:3, 3]
    return P


def _path(S, a):
    if a.get("path") is not None:
        return _to_obj_space(S, a["path"], a.get("coordinate_space", "OBJECT"))
    if a.get("path_spec"):
        spec = dict(a["path_spec"])
        return build_path(spec.pop("kind"), **spec)
    if a.get("surface_xz"):
        p = S.surface_path(a["surface_xz"])
        if len(p) == 0:
            raise ValueError("surface_xz points did not hit the surface")
        return p
    raise ValueError("give path, path_spec or surface_xz")


def _protect(S, regions):
    if not regions:
        S.eng.protect[:] = 0
        return
    if S.sem is None:
        raise ValueError("protect_regions needs a semantic schema (set_target schema=face|creature)")
    w = np.zeros(len(S.st.co))
    for r in regions:
        w = np.maximum(w, S.sem.weights(r))
    S.eng.protect[:] = w


def _stroke(S, a, brush=None, path=None):
    ses = Session.get()
    b = brush or a.get("brush") or ses.brush["brush"]
    path = _path(S, a) if path is None else path
    if ses.conflicts(path) and not a.get("override_user_edit"):
        return {"paused": True, "reason": "planned stroke overlaps the area the user just edited",
                "user_edit": ses.user_edits[-1], "hint": "re-plan elsewhere or pass override_user_edit=true"}
    _protect(S, a.get("protect_regions"))
    kw = {k: a[k] for k in ("falloff", "spacing", "pressure", "pressure_values", "radius_scale", "direction", "symmetry_x",
                            "geodesic", "frontface", "normal", "plane_offset") if a.get(k) is not None}
    kw.setdefault("falloff", ses.brush.get("falloff"))
    ses.overlay["path"] = [tuple(p) for p in np.asarray(path)]
    ses.busy = True
    try:
        mode = ses.flush_sculpt_mode(S.obj) if a.get("engine", "auto") != "native" else None
        rep = S.stroke(b, path, a.get("radius", ses.brush["radius"]), a.get("strength", ses.brush["strength"]),
                       region=a.get("region"), engine=None if a.get("engine", "auto") == "auto" else a.get("engine"), **kw)
        ses.restore_mode(S.obj, mode)
    finally:
        ses.busy = False
        S.eng.protect[:] = 0
    return rep


def _images(paths):
    return {"images": [p for p in paths if isinstance(p, str) and p.endswith(".png")]}


# ======================================================================= session
@tool
def get_capabilities(a):
    from ..core.capability_detector import detect
    return detect()


@tool
def get_status(a):
    ses = Session.get()
    out = {"mode": ses.guard.mode, "target": ses.target, "brush_defaults": ses.brush, "user_edits": ses.user_edits[-5:],
           "out_dir": ses.out_dir, "busy": ses.busy, "paused": ses.paused}
    if ses.target and ses.target in ses.sculptors:
        S = ses.sculptors[ses.target]
        out["vertices"] = len(S.st.co)
        out["checkpoints"] = S.ckpt.labels()
        out["brush_stats"] = S.eng.stats
        out["semantic"] = S.sem.status if S.sem else None
    return out


@tool
def set_mode(a):
    from ..core.task_router import ModeGuard, MODES
    ses = Session.get()
    ses.guard = ModeGuard(a["mode"], a.get("user_permissions", ()))
    ses.log.mode = a["mode"]
    ses.log.record({"op": "set_mode", "mode": a["mode"], "permissions": a.get("user_permissions", [])})
    return {"mode": a["mode"], "description": MODES[a["mode"]]}


@tool
def set_target(a):
    S = _S(a, a.get("schema"))
    return {"target": S.obj.name, "vertices": len(S.st.co), "schema": a.get("schema")}


@tool
def audit_no_external_mesh(a):
    from ..core.state_manager import audit_scene
    return audit_scene(Session.get().log)


# ======================================================================= base mesh
@tool
def create_initial_sphere(a):
    ses = Session.get()
    ob = MR.create_primitive(a.get("kind", "ICOSPHERE"), a.get("name", "BNM_Head"), a.get("radius", 1.0),
                             detail=a.get("detail", 6), log=ses.log)
    for p in ob.data.polygons:
        p.use_smooth = True
    S = ses.sculptor(ob.name, a.get("schema", "face"))
    return {"object": ob.name, "vertices": len(S.st.co), "schema": a.get("schema", "face")}


@tool
def create_primitive(a):
    ses = Session.get()
    ob = MR.create_primitive(a["kind"], a.get("name", "BNM_" + a["kind"].title()), a.get("size", 1.0),
                             tuple(a.get("location", (0, 0, 0))), a.get("detail", 4), log=ses.log)
    return {"object": ob.name, "vertices": len(ob.data.vertices)}


@tool
def analyze_mesh(a):
    return AN.analyze_mesh(Session.get().obj(a.get("object")), detail=a.get("detail", False))


@tool
def analyze_surface(a):
    S = _S(a)
    p = np.asarray(a["point"], float)
    r = a.get("radius", 0.08)
    d = np.linalg.norm(S.st.co - p, axis=1)
    idx = np.nonzero(d < r)[0]
    if len(idx) == 0:
        return {"error": "no surface within radius"}
    H = curvature(S.st)["mean"][idx]
    rel = AN.relief(S.st)[idx]
    n = S.st.normals[idx].mean(0)
    return {"vertices": int(len(idx)), "center": S.st.co[idx].mean(0).round(4).tolist(),
            "normal": (n / np.linalg.norm(n)).round(4).tolist(), "mean_curvature": float(H.mean()),
            "curvature_range": [float(H.min()), float(H.max())], "relief_mean": float(rel.mean())}


@tool
def get_surface_region(a):
    S = _S(a)
    c = np.asarray(a["center"], float)
    if a.get("geodesic"):
        idx, dd = geodesic_ball(S.st.co, S.st.ptr, S.st.nbr, c, a["radius"])
    else:
        dd = np.linalg.norm(S.st.co - c, axis=1)
        idx = np.nonzero(dd < a["radius"])[0]
    if len(idx) == 0:
        return {"vertices": 0}
    n = S.st.normals[idx].mean(0)
    return {"vertices": int(len(idx)), "center": S.st.co[idx].mean(0).round(4).tolist(),
            "normal": (n / np.linalg.norm(n)).round(4).tolist(),
            "bbox": [S.st.co[idx].min(0).round(4).tolist(), S.st.co[idx].max(0).round(4).tolist()]}


@tool
def extract_cross_section(a):
    S = _S(a)
    info, segs = AN.cross_section(S.st, a["origin"], a["normal"])
    return info


@tool
def estimate_curvature(a):
    S = _S(a)
    if a.get("region"):
        idx = S.sem.indices(a["region"])
        c = curvature(S.st)
        return {k: [float(np.percentile(v[idx], 5)), float(np.median(v[idx])), float(np.percentile(v[idx], 95))] for k, v in c.items()}
    return curvature_stats(S.st)


# ======================================================================= brush state
@tool
def set_sculpt_mode(a):
    from ..sculpt.native_brush import NativeBrushEngine
    S = _S(a)
    ne = NativeBrushEngine(S.obj)
    if not ne.available:
        return {"ok": False, "reason": "no UI viewport (background Blender); virtual engine is used instead"}
    if a.get("enable", True):
        ne.ensure_sculpt()
    else:
        ne.to_object_mode()
    return {"ok": True, "mode": S.obj.mode}


@tool
def select_sculpt_brush(a):
    from ..sculpt.virtual_brush import canonical
    Session.get().brush["brush"] = canonical(a["brush"]) if a["brush"].upper() != "DRAW_SUBTRACT" else "DRAW_SUBTRACT"
    return Session.get().brush


@tool
def set_brush_radius(a):
    Session.get().brush["radius"] = float(a["radius"])
    return Session.get().brush


@tool
def set_brush_strength(a):
    Session.get().brush["strength"] = float(a["strength"])
    return Session.get().brush


@tool
def set_brush_falloff(a):
    from ..sculpt import falloff as FO
    FO.get(a["falloff"])  # validate
    Session.get().brush["falloff"] = a["falloff"]
    return Session.get().brush


# ======================================================================= sculpting
@tool
def execute_brush_stroke(a):
    S = _S(a)
    S.ckpt.snapshot("stroke")          # enables undo_stroke
    return _stroke(S, a)


@tool
def execute_stroke_sequence(a):
    S = _S(a)
    reports = []
    for i, st in enumerate(a["strokes"]):
        sa = {**{k: v for k, v in a.items() if k != "strokes"}, **st}
        S.ckpt.snapshot(f"seq_{i}")
        reports.append(_stroke(S, sa))
        yield {"progress": i + 1, "of": len(a["strokes"])}
    return {"strokes": reports}


@tool
def sculpt_along_curve(a):
    S = _S(a)
    S.ckpt.snapshot("stroke")
    cp = _to_obj_space(S, a["control_points"], a.get("coordinate_space", "OBJECT"))
    path = bezier(*cp[:4], n=40) if a.get("curve") == "bezier" and len(cp) >= 4 else catmull_rom(cp, 10)
    return _stroke(S, a, path=path)


@tool
def sculpt_surface_region(a):
    S = _S(a)
    from ..sculpt.planner import _region_geometry
    g = _region_geometry(S.sem, a["region"], "both")
    R = a.get("radius") or float(np.clip(min(g["ext_major"], g["ext_minor"]) * 0.6, S.st.mean_edge * 3, 0.2))
    path = fill_region(g["center"], g["normal"], g["ext_major"], g["ext_minor"], R * 0.8, g["major"])
    return _stroke(S, {**a, "radius": R, "region": a["region"]}, brush=a["brush"], path=path)


def _region_or_path(S, a):
    if a.get("path"):
        return _to_obj_space(S, a["path"], a.get("coordinate_space", "OBJECT")), a.get("radius", 0.06)
    from ..sculpt.planner import _region_geometry
    g = _region_geometry(S.sem, a["region"], "both")
    c, M = g["center"], g["major"]
    path = catmull_rom(np.array([c - M * g["ext_major"] * 0.7, c, c + M * g["ext_major"] * 0.7]), 6)
    return path, a.get("radius") or float(np.clip(g["ext_minor"], S.st.mean_edge * 3, 0.2))


@tool
def add_surface_volume(a):
    S = _S(a)
    path, R = _region_or_path(S, a)
    amt = a.get("amount", 1.0)
    r1 = _stroke(S, {**a, "radius": R, "strength": float(np.clip(0.35 * amt, 0.02, 1))}, "CLAY_STRIPS", path)
    r2 = _stroke(S, {**a, "radius": R * 1.3, "strength": 0.2}, "SURFACE_SMOOTH", path)
    return {"build": r1, "blend": r2}


@tool
def remove_surface_volume(a):
    S = _S(a)
    path, R = _region_or_path(S, a)
    amt = a.get("amount", 1.0)
    r1 = _stroke(S, {**a, "radius": R, "strength": float(np.clip(0.35 * amt, 0.02, 1))}, "SCRAPE", path)
    r2 = _stroke(S, {**a, "radius": R, "strength": float(np.clip(0.25 * amt, 0.02, 1)), "direction": -1}, "DRAW", path)
    return {"scrape": r1, "carve": r2}


@tool
def deform_surface(a):
    S = _S(a)
    S.ckpt.snapshot("stroke")
    b = a["brush"].upper()
    if b == "POSE":
        idx = S.sem.indices(a["region"]) if a.get("region") else np.nonzero(np.linalg.norm(S.st.co - np.asarray(a["point"]), axis=1) < a.get("radius", 0.2))[0]
        w = S.sem.weights(a["region"])[idx] if a.get("region") else None
        return S.eng.pose_rotate(a["pivot"], a["axis"], a["angle"], idx, w)
    p = np.asarray(a["point"], float)
    j = int(np.argmin(np.linalg.norm(S.st.co - p, axis=1)))
    p = S.st.co[j]
    path = np.array([p, p + np.asarray(a["drag"], float)])
    return _stroke(S, {**a, "strength": a.get("strength", 1.0)}, b, path)


@tool
def smooth_surface(a):
    S = _S(a)
    path, R = _region_or_path(S, a)
    S.ckpt.snapshot("smooth")
    if a.get("protect_features"):
        # curvature-based protection keeps anatomical landmarks (lid margin, lip line) from being erased
        S.eng.protect[:] = feature_protection(S.st)
        try:
            return S.stroke(a.get("kind", "SURFACE_SMOOTH"), path, R, a.get("strength", 0.4), region=a.get("region"))
        finally:
            S.eng.protect[:] = 0
    return _stroke(S, {**a, "radius": R, "strength": a.get("strength", 0.4)}, a.get("kind", "SURFACE_SMOOTH"), path)


@tool
def pinch_surface(a):
    S = _S(a)
    S.ckpt.snapshot("stroke")
    return _stroke(S, {**a, "radius": a.get("radius", 0.03), "strength": a.get("strength", 0.4)}, "PINCH")


@tool
def flatten_surface(a):
    S = _S(a)
    S.ckpt.snapshot("stroke")
    return _stroke(S, {**a, "radius": a.get("radius", 0.08), "strength": a.get("strength", 0.4)}, a.get("kind", "FLATTEN"))


@tool
def sdf_volume_edit(a):
    S = _S(a)
    voxel = a.get("voxel") or max(S.st.mean_edge, 0.008)
    op = a["operation"]

    def edit(g):
        if op == "add":
            g.add_path(_to_obj_space(S, a["path"], "OBJECT"), a.get("radius", 0.05), a.get("blend"))
        elif op == "remove":
            g.remove_path(_to_obj_space(S, a["path"], "OBJECT"), a.get("radius", 0.05), a.get("blend"))
        elif op == "offset":
            g.offset(a.get("amount", 0.01))
        elif op == "repair":
            g.blur(1)
    n0 = len(S.st.co)
    MR.sdf_remesh(S.obj, voxel, edit, log=S.log)
    for p in S.obj.data.polygons:
        p.use_smooth = True
    S.topology_changed()
    return {"operation": op, "voxel": voxel, "vertices_before": n0, "vertices_after": len(S.st.co),
            "note": "vertex indices changed; semantic regions were rebuilt"}


@tool
def undo_stroke(a):
    S = _S(a)
    ok = S.ckpt.undo(a.get("steps", 1))
    S.eng.resync()
    return {"undone": ok}


# ======================================================================= spatial
@tool
def raycast_surface(a):
    S = _S(a)
    j = ray_hit(S.st, a["origin"], a["direction"])
    if j is None:
        return {"hit": False}
    return {"hit": True, "vertex": j, "point": S.st.co[j].round(5).tolist(), "normal": S.st.normals[j].round(4).tolist()}


def _cams(S, views, size):
    return RN.head_cameras(S.obj, size, views=tuple(views))


@tool
def screen_to_surface(a):
    S = _S(a)
    cam = _cams(S, [a["view"]], a.get("size", 256))[a["view"]]
    hit = PK.screen_to_surface([S.obj], cam, a["u"], a["v"])
    return hit or {"hit": False}


@tool
def project_landmark(a):
    S = _S(a)
    cam = _cams(S, [a["view"]], a.get("size", 256))[a["view"]]
    S.sem.rebuild(S.st)
    names = a.get("names") or list(S.sem.landmarks)
    lms = {n: S.sem.landmarks[n]["position"] for n in names if n in S.sem.landmarks}
    return {"view": a["view"], "pixels": PK.project_landmarks(lms, cam, S.obj.matrix_world), "camera": cam.describe()}


@tool
def get_surface_normal(a):
    S = _S(a)
    p = np.asarray(a["point"], float)
    d = np.linalg.norm(S.st.co - p, axis=1)
    idx = np.argsort(d)[:12]
    n = S.st.normals[idx].mean(0)
    return {"normal": (n / np.linalg.norm(n)).round(4).tolist(), "nearest_vertex": int(idx[0]), "distance": float(d[idx[0]])}


@tool
def get_semantic_region(a):
    S = _S(a)
    if S.sem is None:
        raise ValueError("no semantic schema on this object (set_target schema=face|creature)")
    S.sem.rebuild(S.st)
    return S.sem.info(a["region"])


@tool
def get_geodesic_region(a):
    S = _S(a)
    idx, gd = geodesic_ball(S.st.co, S.st.ptr, S.st.nbr, np.asarray(a["center"], float), a["radius"])
    return {"vertices": int(len(idx)), "max_geodesic": float(gd.max()) if len(gd) else 0.0,
            "centroid": S.st.co[idx].mean(0).round(4).tolist() if len(idx) else None}


@tool
def create_region_mask(a):
    S = _S(a)
    w = np.zeros(len(S.st.co))
    if a.get("region"):
        w = S.sem.weights(a["region"])
    elif a.get("view"):
        cam = _cams(S, [a["view"]], a.get("size", 256))[a["view"]]
        idx = PK.region_mask_from_pixels(S.obj, cam, polygon=a.get("polygon"), circle=a.get("circle"))
        w[idx] = 1.0
    elif a.get("center") is not None:
        idx, gd = geodesic_ball(S.st.co, S.st.ptr, S.st.nbr, np.asarray(a["center"], float), a["radius"])
        w[idx] = 1 - gd / a["radius"]
    if a.get("invert"):
        w = 1 - w
    S.eng.mask = w if a.get("clear", True) else np.maximum(S.eng.mask, w)
    S.eng.write_mask()
    S.log.record({"op": "create_region_mask", "region": a.get("region"), "masked_vertices": int((w > 0.05).sum())})
    return {"masked_vertices": int((w > 0.05).sum()), "attribute": ".sculpt_mask"}


@tool
def protect_region(a):
    S = _S(a)
    _protect(S, a["regions"])
    S.persist_protect = list(a["regions"])
    return {"protected_vertices": int((S.eng.protect > 0.05).sum())}


# ======================================================================= resolution
@tool
def voxel_remesh(a):
    S = _S(a)
    vs = a.get("voxel_size") or MR.voxel_size_for(S.obj, a.get("target_vertices", 60000))
    n = MR.voxel_remesh(S.obj, vs, log=S.log)
    for p in S.obj.data.polygons:
        p.use_smooth = True
    S.topology_changed()
    return {"voxel_size": vs, "vertices": n, "note": "indices changed; masks reset, semantic regions rebuilt"}


@tool
def adaptive_subdivide(a):
    S = _S(a)
    MR.adaptive_subdivide(S.obj, S.sem.indices(a["region"]), a.get("cuts", 1), log=S.log)
    S.topology_changed()
    return {"vertices": len(S.st.co)}


@tool
def set_multiresolution(a):
    S = _S(a)
    return {"total_levels": MR.set_multiresolution(S.obj, a.get("levels", 1), log=S.log)}


@tool
def optimize_topology(a):
    S = _S(a)
    return S.eng.smooth_region(np.arange(len(S.st.co)), iterations=a.get("iterations", 5), factor=0.4, tangential=True)


@tool
def validate_mesh(a):
    return _validate(Session.get().obj(a.get("object")))


# ======================================================================= visual feedback
@tool
def capture_viewport(a):
    if bpy.app.background:
        return {"ok": False, "reason": "background session has no viewport; use render_multiview"}
    from ..sculpt.native_brush import find_view3d
    win, area, region, _ = find_view3d()
    path = a.get("path") or os.path.join(Session.get().out_dir, f"viewport_{int(time.time())}.png")
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.screen.screenshot_area(filepath=path)
    return _images([path])


def _render(a, views, modes):
    S = _S(a)
    ses = Session.get()
    objs = [S.obj] + [o for o in S.extras if o.name in bpy.data.objects]
    size = a.get("size", 320)
    cams = _cams(S, views, size)
    name = f"r{int(time.time() * 1000) % 10 ** 8}"
    out = RN.render_views(objs, os.path.join(ses.out_dir, "renders"), name, tuple(views), tuple(modes), size, cameras=cams,
                          curvature_obj=S.obj if "curvature" in modes else None)
    paths = [v for k, v in out.items() if not k.startswith("_buf")]
    if a.get("blender_render"):
        for v, cam in cams.items():
            p = os.path.join(ses.out_dir, "renders", f"{name}_{v}_blender.png")
            RN.workbench_render(objs, cam, p)
            paths.append(p)
    return {**_images(paths), "cameras": {v: c.describe() for v, c in cams.items()}}


@tool
def render_front(a):
    return _render(a, ["front"], a.get("modes", ["clay"]))


@tool
def render_side(a):
    return _render(a, ["left"], a.get("modes", ["clay"]))


@tool
def render_three_quarter(a):
    return _render(a, ["three_quarter_left"], a.get("modes", ["clay"]))


@tool
def render_multiview(a):
    return _render(a, a.get("views", ["front", "left", "three_quarter_left", "top"]), a.get("modes", ["clay"]))


@tool
def render_depth(a):
    return _render(a, a.get("views", ["front", "left"]), ["depth"])


@tool
def render_normals(a):
    return _render(a, a.get("views", ["front", "left"]), ["normal"])


@tool
def analyze_reference_image(a):
    from ..reference.image_analyzer import analyze_image
    from ..reference.structural_specification import build_spec, to_canon_overrides, save
    front, _ = analyze_image(a["path"], a.get("view", "front"), a.get("facing", "left"), a.get("landmarks_px"))
    side = None
    if a.get("side_path"):
        side, _ = analyze_image(a["side_path"], "side", a.get("facing", "left"), a.get("side_landmarks_px"))
    spec = build_spec(front if a.get("view", "front") == "front" else None, side or (front if a.get("view") in ("left", "right", "side") else None))
    p = os.path.join(Session.get().out_dir, "reference_spec.json")
    save(spec, p)
    return {"spec": spec, "canon_overrides": to_canon_overrides(spec), "spec_path": p,
            "front_profile_rows": front["width_profile"]["rows"][::4]}


@tool
def compare_reference(a):
    from ..reference.image_analyzer import foreground_mask
    from ..optimization.geometry_loss import silhouette_loss, row_width_loss
    from ..optimization.iterative_optimizer import fit_mask_to_camera
    S = _S(a)
    view = a.get("view", "front")
    cam = _cams(S, [view], a.get("size", 256))[view]
    buf = RN.raycast_render([S.obj], cam)
    ref = fit_mask_to_camera(foreground_mask(RN.load_image(a["path"])), cam, buf["mask"])
    dw = row_width_loss(buf["mask"], ref)
    rows = np.nonzero(buf["mask"].any(1) | ref.any(1))[0]
    worst = rows[np.argsort(-np.abs(dw[rows]))[:5]]
    return {"silhouette_iou": round(1 - silhouette_loss(buf["mask"], ref), 4),
            "worst_rows": [{"v_px": int(y), "model_minus_ref_halfwidth_px": float(dw[y]),
                            "z_object": float(cam.pixel_to_ray(np.array([cam.W / 2]), np.array([y + 0.5]))[0][0][2])} for y in worst]}


@tool
def identify_shape_errors(a):
    S = _S(a)
    out = {"measured": {}, "anomalies": []}
    if S.sem is not None and "nose_tip" in S.sem.region_defs:
        from ..anatomy import face_canon as FC
        from ..anatomy.head_sculpt_pipeline import face_metrics
        C = getattr(S, "canon", None) or FC.canon()
        m, t = face_metrics(S, C)
        out["measured"] = m
        out["errors_vs_spec"] = {k: round(m[k] - v, 4) for k, v in t.items() if m.get(k) is not None}
    sym = AN.symmetry_error(S.st)
    out["symmetry_x"] = sym
    if sym["relative"] > 0.01:
        out["anomalies"].append("left/right asymmetry above 1% of size")
    cs = curvature_stats(S.st)
    out["curvature"] = cs
    if cs["noise_index"] > 0.002:
        out["anomalies"].append("surface noise: consider SURFACE_SMOOTH with feature protection")
    v = _validate(S.obj)
    out["mesh_problems"] = v["problems"]
    out["note"] = "measured numbers only; visual/qualitative judgement must come from inspecting renders"
    return out


# ======================================================================= iteration
@tool
def plan_sculpt_stroke(a):
    from ..sculpt import planner as PL
    S = _S(a)
    if S.sem is None:
        raise ValueError("planner needs a semantic schema")
    S.sem.rebuild(S.st)
    if a.get("command"):
        plan = PL.plan_from_command(S.sem, a["command"], S.obj.name)
    else:
        plan = {"command": a, "plan": PL.plan_edit(S.sem, a["region"], a.get("intent", "add_volume"), a.get("amount", 1.0),
                                                    a.get("side", "both"), S.obj.name)}
    if a.get("execute"):
        plan["execution"] = PL.execute_plan(S, plan["plan"])
    return plan


@tool
def execute_plan(a):
    from ..sculpt import planner as PL
    S = _S(a)
    return {"reports": PL.execute_plan(S, a["plan"], a.get("max_leak", 0.35))}


@tool
def optimize_stroke_parameters(a):
    from ..optimization.stroke_search import grab_to_target
    S = _S(a)
    best, loss, hist = grab_to_target(S, a["point"], a["target"])
    return {"best_params": best, "final_distance": loss, "evaluations": len(hist)}


@tool
def evaluate_sculpt_result(a):
    S = _S(a)
    label = a.get("checkpoint") or (S.ckpt.labels()[-1] if S.ckpt.labels() else None)
    if label is None:
        return {"error": "no checkpoint to compare against"}
    me_ref = S.ckpt.meshes[label]
    if len(me_ref.vertices) != len(S.st.co):
        return {"checkpoint": label, "note": "topology changed since checkpoint; only global stats",
                "volume_now": AN.volume(S.st.co, S.st.tris), "vertices_now": len(S.st.co)}
    co0 = np.empty(len(me_ref.vertices) * 3, dtype=np.float32)
    me_ref.vertices.foreach_get("co", co0)
    co0 = co0.reshape(-1, 3)
    d = np.linalg.norm(S.st.co - co0, axis=1)
    res = {"checkpoint": label, "moved_vertices": int((d > 1e-6).sum()), "max_displacement": float(d.max()),
           "volume_delta": AN.volume(S.st.co, S.st.tris) - AN.volume(co0.astype(float), S.st.tris),
           "valid": _validate(S.obj)["problems"]}
    if a.get("region") and S.sem is not None:
        w = S.sem.weights(a["region"]) > 0.05
        res["fraction_of_change_inside_region"] = float(d[w].sum() / max(d.sum(), 1e-12))
    return res


@tool
def create_checkpoint(a):
    S = _S(a)
    return {"label": S.ckpt.create(a["label"]), "all": S.ckpt.labels()}


@tool
def restore_checkpoint(a):
    S = _S(a)
    S.ckpt.restore(a["label"])
    S.topology_changed()
    return {"restored": a["label"], "vertices": len(S.st.co)}


@tool
def repeat_sculpt_pass(a):
    S = _S(a)
    times = a.get("times", 3)
    reps = []
    for i in range(times):
        if a.get("plan"):
            from ..sculpt import planner as PL
            r = PL.execute_plan(S, a["plan"])
            reps.append(r)
            if not any(x["accepted"] for x in r):
                break
        else:
            reps.append(_stroke(S, {**a, **a["stroke"]}))
        yield {"pass": i + 1, "of": times}
    return {"passes": len(reps), "reports": reps}


@tool
def run_head_stage(a):
    from ..anatomy import face_canon as FC
    from ..anatomy import head_sculpt_pipeline as HP
    S = _S(a)
    if S.sem is None:
        S = _S(a, "face")
    C = FC.canon(a.get("canon_overrides"))
    S.canon = C
    stage = a["stage"]
    fns = {"stage1_skull": lambda: (HP.stage1_skull(S, C), HP.stage1_neck_and_remesh(S, C)),
           "stage2_planes": lambda: HP.stage2_planes(S, C),
           "stage3_eyes": lambda: (MR.voxel_remesh(S.obj, 0.0105, log=S.log), S.topology_changed(), HP.stage3_eyes(S, C))[-1],
           "stage4_nose": lambda: HP.stage4_nose(S, C), "stage5_mouth": lambda: HP.stage5_mouth(S, C),
           "stage6_ears_neck": lambda: HP.stage6_ears_neck(S, C), "stage7_secondary": lambda: HP.stage7_secondary(S, C),
           "stage8_tertiary": lambda: HP.stage8_tertiary(S, C), "stage9_review": lambda: HP.face_metrics(S, C)}
    order = list(fns)
    todo = order if stage == "all" else [stage]
    results = {}
    for s in todo:
        S.log.set_stage(s)
        results[s] = jsonable(fns[s]())
        S.ckpt.create(s)
        if a.get("render", True):
            results[s + "_renders"] = jsonable(S.observe(s, workbench=False))
        yield {"stage_done": s}
    imgs = [p for k, v in results.items() if k.endswith("_renders") for p in v.values() if isinstance(p, str)]
    return {"results": results, **_images(imgs), "loops": S.loops[-6:]}


@tool
def run_video_workflow_head(a):
    """Run selected stages of the observed video workflow on a head (creates the sphere on first call).
    args: name, canon_overrides, stages=[...] (default: all remaining), hair, glasses, new"""
    from ..anatomy import face_canon as FC
    from ..anatomy import video_workflow as VW
    from ..anatomy import portrait_accessories as PA
    ses = Session.get()
    name = a.get("name", "BNM_Head")
    ob = bpy.data.objects.get(name)
    if a.get("new"):
        for o in [o for o in bpy.data.objects if o.name == name or o.name.startswith(name + ".")]:
            bpy.data.objects.remove(o)
        for S_old in list(ses.sculptors):
            if S_old == name or S_old.startswith(name + "."):
                old_S = ses.sculptors.pop(S_old)
                for e in getattr(old_S, "extras", []):
                    if e.name in bpy.data.objects:
                        bpy.data.objects.remove(e)
        ob = None
    if ob is None:
        ob = MR.create_primitive("ICOSPHERE", name, 1.0, detail=6, log=ses.log)
        for f in ob.data.polygons:
            f.use_smooth = True
        ses.sculptors.pop(ob.name, None)
    S = ses.sculptor(ob.name)
    if a.get("canon_overrides") is not None or not hasattr(S, "canon"):
        S.canon = FC.canon(a.get("canon_overrides"))
    C = S.canon
    P = VW.TS.head_params(C)
    done_before = getattr(S, "vw_done", [])
    stages = a.get("stages") or [s for s in VW.STAGES if s not in done_before]
    for name_ in stages:
        S.log.set_stage(name_)
        ses.overlay["stage"] = name_
        if name_ == "v7_eyes":
            MR.voxel_remesh(S.obj, a.get("detail_voxel", 0.0105), log=S.log)
            for f in S.obj.data.polygons:
                f.use_smooth = True
            S.topology_changed()
            r = VW.v7_eyes(S, C, P)
        elif name_ in ("v9_ears", "v10_neck"):
            r = getattr(VW, name_)(S, C, P, a.get("detail_voxel", 0.0105))
        elif name_ == "hair":
            r = PA.sculpt_hair(S).name
        elif name_ == "glasses":
            r = [o.name for o in PA.glasses(S, C)]
        else:
            r = getattr(VW, name_)(S, C, P)
        S.ckpt.create(name_)
        S.eng.on_dab = ses._on_dab
        done_before.append(name_)
        yield {"stage_done": name_, "result": jsonable(r)}
    S.vw_done = done_before
    rr = _render({"object": ob.name, "size": a.get("size", 320)}, ["front", "three_quarter_left", "left"], ["clay"])
    return {"object": ob.name, "stages_run": stages, "all_done": done_before, "loops": S.loops[-4:], **rr}


@tool
def overlay_reference(a):
    from ..perception.reference_overlay import overlay
    S = _S(a)
    objs = [S.obj] + [o for o in S.extras if o.name in bpy.data.objects]
    cam = _cams(S, ["front"], a.get("size", 320))["front"]
    buf = RN.raycast_render(objs, cam)
    lm = {}
    for nm, c in zip(("eye_L_center", "eye_R_center"), getattr(S, "eye_centers", [])):
        u, v, _ = cam.world_to_pixel([c])[0]
        lm[nm] = (float(u), float(v))
    if len(lm) < 2:
        raise ValueError("model has no eyeballs yet (run the eye stage first)")
    out = os.path.join(Session.get().out_dir, "renders", f"overlay_{int(time.time())}.png")
    res = overlay(a["path"], a["photo_landmarks"], RN.shade(buf, "clay"), lm, out)
    return {**res, "images": [out]}


@tool
def smooth_region(a):
    S = _S(a)
    if a.get("region"):
        w = S.sem.weights(a["region"])
    elif a.get("center") is not None:
        d = np.linalg.norm(S.st.co - np.asarray(a["center"], float), axis=1)
        from ..sculpt import falloff as FO
        w = FO.smooth(np.minimum(d / a.get("radius", 0.15), 1.0))
    else:
        w = None
    S.ckpt.snapshot("smooth_region")
    return S.eng.relax_smooth(w, iterations=a.get("iterations", 15), factor=a.get("factor", 0.5))


@tool
def run_creature_head(a):
    from ..anatomy.creature_sculptor import sculpt_creature, creature_map, QUADRUPED_HEAD
    ses = Session.get()
    ob = MR.create_primitive("ICOSPHERE", a.get("name", "BNM_Creature"), 1.0, detail=6, log=ses.log)
    for p in ob.data.polygons:
        p.use_smooth = True
    S = ses.sculptor(ob.name)
    schema = a.get("params", {}).get("schema", QUADRUPED_HEAD)
    reps = sculpt_creature(S, schema)
    S.sem = creature_map(ob, schema)
    S.sem.rebuild(S.st)
    r = _render({"object": ob.name}, ["front", "left", "three_quarter_left"], ["clay"])
    return {"object": ob.name, "parts": reps, **r}


@tool
def hard_surface_part(a):
    from ..geometry import hard_surface as HS
    ses = Session.get()
    fn = {"flanged_gear_hub": HS.flanged_gear_hub, "bracket": HS.bracket}[a.get("kind", "flanged_gear_hub")]
    ob = fn(name=a.get("name", "BNM_" + a.get("kind", "part")), log=ses.log, **a.get("params", {}))
    v = _validate(ob, check_self_intersection=False)
    ses.target = ob.name
    return {"object": ob.name, "vertices": len(ob.data.vertices), "validation": v}


@tool
def edit_mode_op(a):
    from ..geometry import hard_surface as HS
    ses = Session.get()
    ob = ses.obj(a.get("object"))
    r = HS.edit_mode_op(ob, a["op"], a.get("select"), a.get("params"), log=ses.log)
    if ob.name in ses.sculptors:
        ses.sculptors[ob.name].topology_changed()
    return r


@tool
def add_modifier(a):
    from ..geometry import hard_surface as HS
    ses = Session.get()
    ob = ses.obj(a.get("object"))
    r = HS.add_modifier(ob, a["type"], a.get("settings"), a.get("apply", False), log=ses.log)
    if a.get("apply") and ob.name in ses.sculptors:
        ses.sculptors[ob.name].topology_changed()
    return r


@tool
def curve_sweep(a):
    from ..geometry import hard_surface as HS
    ob = HS.curve_sweep(a.get("name", "BNM_Sweep"), a["points"], a.get("radius", 0.05), a.get("taper"), a.get("nurbs", False),
                        log=Session.get().log)
    return {"object": ob.name, "vertices": len(ob.data.vertices)}


@tool
def save_blend(a):
    bpy.ops.wm.save_as_mainfile(filepath=a["path"], copy=True)
    return {"saved": a["path"]}


def check_registry():
    missing = sorted(set(SCHEMAS) - set(REGISTRY))
    extra = sorted(set(REGISTRY) - set(SCHEMAS))
    return {"missing_implementations": missing, "unlisted_tools": extra}
