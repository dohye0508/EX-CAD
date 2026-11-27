"""Head sculpt that follows the professional timelapse workflow observed in knowledge/video_workflow_lAspS2BK2Vk.md.

Order: sphere -> egg face + round cranium (sculpted toward a primary-form target) -> soft eye sockets ->
nose ridge -> mouth barrel, chin, brow -> side planes (3/4) -> lip volumes, nostrils -> almond openings +
lid rims (+ eyeballs) -> profile check -> ears and neck from separate primitives merged in.  Every stage
ends with many-iteration region smoothing ("Shift after every stroke").  Creases only at the end, shallow.
"""
import numpy as np
import bpy

from . import eye_sculptor as EYE
from ..sculpt import multiresolution as MR
from ..sculpt import falloff as FO
from ..sculpt.stroke import catmull_rom, line
from ..geometry import sdf as SDF
from ..geometry import target_shape as TS
from ..geometry.sdf import sd_ellipsoid
from ..geometry.mesh_io import MeshState, get_coords
from ..geometry.semantic import face_map, ray_hit
from ..core.orchestrator import Sculptor

STAGES = ["v1_egg_cranium", "v2_sockets", "v3_nose_ridge", "v4_mouth_chin_brow", "v5_side_planes",
          "v6_lips_nostrils", "v7_eyes", "v8_profile", "v9_ears", "v10_neck", "v11_final_smooth"]


def _ball_weights(S, centers, radius):
    co = S.st.co
    w = np.zeros(len(co))
    for c in np.atleast_2d(centers):
        d = np.linalg.norm(co - c, axis=1)
        w = np.maximum(w, FO.smooth(np.minimum(d / radius, 1.0)))
    return w


def _smooth_area(S, xz_points, radius, iterations=15, factor=0.5):
    pts = S.surface_path(xz_points)
    if len(pts) == 0:
        return None
    pts = np.vstack([pts, pts * [-1, 1, 1]])
    return S.eng.relax_smooth(_ball_weights(S, pts, radius), iterations=iterations, factor=factor)


# ------------------------------------------------------------------ V1
def v1_egg_cranium(S, C, P):
    """Video 1:19-1:22: pull the lower front of the sphere down into a face mass, then refine toward the
    primary-form target (cranium + face egg + cheek masses + jaw).  Neck is excluded (separate primitive later)."""
    p = dict(P)
    p["neck_radius"] = 1e-3
    p["neck_top"] = p["neck_bottom"] = (0, 0.2, -0.6)
    field = TS.head_target(p)
    st = S.st
    j = ray_hit(st, (0, -3, -0.45), (0, 1, 0))
    a = st.co[j]
    S.stroke("ELASTIC_DEFORM", [a, a + [0, -0.06, -0.38]], 0.45, 1.0, region="face_mass")
    hist = []
    for it, (r, step) in enumerate([(0.45, 0.08), (0.4, 0.07), (0.35, 0.06), (0.3, 0.05), (0.25, 0.04), (0.22, 0.03),
                                    (0.2, 0.03), (0.18, 0.025), (0.16, 0.02), (0.14, 0.02)]):
        rep = S.eng.conform_to_field(field, r, step, tol=0.006, max_dabs=50, region="primary_forms")
        S.eng.relax_smooth(None, iterations=12, factor=0.5)
        hist.append(round(rep["rms_after"], 4))
        if it == 4:
            MR.sdf_remesh(S.obj, 0.022, log=S.log)   # even density after the big stretch
            for f in S.obj.data.polygons:
                f.use_smooth = True
            S.topology_changed()
    for _ in range(12):   # wide conform + smoothing removes the dab bumps
        rep = S.eng.conform_global(field, step=0.03, gain=0.6, region="primary_forms")
        S.eng.relax_smooth(None, iterations=4, factor=0.5)
    hist.append(round(rep["rms_after"], 4))
    S.log.record({"op": "loop_validate", "loop": "primary_target_rms", "history": hist})
    S.loops.append({"name": "primary_target_rms", "history": [{"iter": i, "abs": h} for i, h in enumerate(hist)]})
    S.field_head = field
    return {"target_rms_history": hist}


# ------------------------------------------------------------------ V2
def v2_sockets(S, C, P):
    """Video 1:22: two broad, soft hollows at eye level.  Carved toward a broad bowl volume with
    brush dabs (smooth edges by construction), then smoothed."""
    L, W = C["levels"], C["widths"]
    ex, ez = W["eye_center"], L["eye"]
    reps = []
    for sx in (1, -1):
        p = S.surface(sx * ex, ez)
        rel = S.st.co - p
        rel[:, 2] -= 0.015                                    # deepest just under the brow
        q = np.sqrt((rel[:, 0] / 0.27) ** 2 + (rel[:, 2] / 0.2) ** 2)
        front = np.clip((-S.st.normals[:, 1] - 0.1) / 0.3, 0, 1) * (np.abs(rel[:, 1]) < 0.25)
        w = np.clip((np.exp(-3.2 * q ** 2) - np.exp(-3.2)) / (1 - np.exp(-3.2)), 0, 1) * front   # gaussian bowl, exactly 0 at the rim
        idx = np.nonzero(w > 1e-3)[0]
        # broad, shallow hollow pushed back along the surface normals (no rims), done in 3 gentle passes
        for _ in range(3):
            reps.append(S.eng.offset_region(idx, -C["orbit_depth"] * 0.22, weights=w[idx], use_normals=True))
            S.eng.relax_smooth(_ball_weights(S, [p], 0.3), iterations=6, factor=0.4)
    _smooth_area(S, [(ex, ez)], 0.26, iterations=20)
    # brow mass above each socket (soft)
    S.stroke("CLAY", S.surface_path([(x, L["brow"] + 0.01) for x in np.linspace(0.12, 0.45, 6)]), 0.08, 0.22,
             symmetry_x=True, region="brow_ridge")
    _smooth_area(S, [(0.3, L["brow"])], 0.15, iterations=10)
    eye = S.surface(ex, ez)
    br = S.surface(ex - 0.02, L["brow"])
    return {"socket_depth_vs_brow": float(eye[1] - br[1]), "moved": [r["moved_vertices"] for r in reps]}


# ------------------------------------------------------------------ V3
def v3_nose_ridge(S, C, P):
    """Video 1:27-1:32: one soft ridge between the sockets -> nose mass.  Clay is added toward a soft
    nose-wedge volume (bridge widening to the tip, tip ball, alae) by brush dabs, then smoothed."""
    L, W, N = C["levels"], C["widths"], C["nose"]
    nas = S.surface(0, L["nasion"])
    sub = S.surface(0, L["subnasale"])
    cheek_y = S.front_y(W["ala"] + 0.1, L["nose_tip"] - 0.04)
    tip = np.array([0, cheek_y - N["tip_projection"] + 0.06, L["nose_tip"]])   # tip ball radius ~0.065
    ala_y = S.front_y(W["ala"], L["subnasale"] + 0.04) - 0.015
    vol = TS.nose_volume(nas + [0, -0.01, 0], tip, sub + [0, -0.03, 0], W["ala"] - 0.03, ala_y, L["subnasale"] + 0.035)
    co = S.st.co
    fx = FO.smooth(np.minimum(np.abs(co[:, 0]) / (W["ala"] + 0.09), 1.0))
    zt, zb = L["nasion"] + 0.06, L["subnasale"] - 0.06
    fz = FO.smooth(np.clip(np.maximum(co[:, 2] - zt, zb - co[:, 2]) / 0.05, 0, 1)) * ((co[:, 2] < zt + 0.05) & (co[:, 2] > zb - 0.05))
    allowed = fx * fz * (S.st.normals[:, 1] < -0.3)
    hist = []
    for it in range(14):
        rep = S.eng.grow_into_volume(vol, 0.1, 0.03, tmax=0.32, tol=0.003, allowed=allowed, wide=True,
                                     label="CLAY_INTO_NOSE_VOLUME", region="nose", direction=(0, -1, 0))
        S.eng.relax_smooth(allowed, iterations=3, factor=0.3)
        hist.append(round(rep["max_exit_before"], 4))
        if rep["max_exit_before"] < 0.004:
            break
    S.eng.relax_smooth(allowed, iterations=10, factor=0.4)   # melt tip and alae into one base
    MR.voxel_remesh(S.obj, 0.014, log=S.log)    # even out the stretched nose faces
    for f in S.obj.data.polygons:
        f.use_smooth = True
    S.topology_changed()
    _smooth_area(S, [(0, L["nose_tip"]), (0, L["nasion"]), (W["ala"] * 0.7, L["subnasale"] + 0.03)], 0.14, iterations=6, factor=0.35)
    S.loops.append({"name": "nose_volume", "history": [{"iter": i, "abs": h} for i, h in enumerate(hist)]})
    tip_now = S.surface(0, L["nose_tip"])
    return {"tip_projection_error": float(tip_now[1] - (tip[1] - 0.065)), "hist": hist}


# ------------------------------------------------------------------ V4
def v4_mouth_chin_brow(S, C, P):
    """Video 1:38: mouth barrel bulge, soft mouth line, chin mound; everything smoothed."""
    L, W = C["levels"], C["widths"]
    mb = float(C.get("mouth_barrel", 1.0))       # style: 1 = realistic dental arch, ~0.3 = flat anime face
    for z, s in ((L["upper_lip"] + 0.02, 0.35 * mb), (L["stomion"], 0.35 * mb), (L["lower_lip"] - 0.02, 0.3 * mb)):
        S.stroke("CLAY", S.surface_path([(x, z) for x in np.linspace(-0.25, 0.25, 7)]), 0.1, s, region="mouth_barrel")
    _smooth_area(S, [(0, L["stomion"]), (0.18, L["stomion"])], 0.18, iterations=10)
    mc = W["mouth_corner"]
    S.stroke("DRAW_SUBTRACT", S.surface_path([(x, L["stomion"] + 0.012 * (abs(x) / mc) ** 2) for x in np.linspace(-mc, mc, 11)]),
             0.035, 0.35, region="mouth", falloff="SMOOTHER")
    S.stroke("CLAY", S.surface_path([(-0.07, C["levels"]["chin"]), (0.07, C["levels"]["chin"])]), 0.14, 0.18, region="chin")
    S.stroke("DRAW_SUBTRACT", S.surface_path([(x, L["mentolabial"]) for x in np.linspace(-0.12, 0.12, 5)]), 0.05, 0.1, region="chin")

    def m_chin():
        ch = S.surface(0, L["chin"])
        ml = S.surface(0, L["mentolabial"])
        return np.array([(ml[1] - C["chin_projection"]) - ch[1]])

    def a_chin(e, g):
        ch = S.surface(0, L["chin"])
        S.stroke("GRAB", [ch, ch + [0, e[0] * g, 0]], 0.2, 1.0, region="chin")
    S.closed_loop("chin_projection", m_chin, a_chin, tol=0.01, max_iter=6)
    _smooth_area(S, [(0, L["chin"]), (0, L["mentolabial"]), (0.12, L["stomion"])], 0.12, iterations=8)
    return {"chin_error": float(m_chin()[0])}


# ------------------------------------------------------------------ V5
def v5_side_planes(S, C, P):
    """Video 1:45-1:52 (3/4 view): side planes of the face, cheek-jaw transition, temples."""
    L = C["levels"]
    # side planes come from the primary target; here only melt the face block into the cranium sides
    co = S.st.co
    w = FO.smooth(np.clip(np.abs(np.abs(co[:, 0]) - 0.55) / 0.2, 0, 1)) * (co[:, 1] < 0.1) * (co[:, 2] > -0.85) * (co[:, 2] < 0.5)
    S.eng.relax_smooth(w, iterations=25, factor=0.5)
    tp = S.surface(-0.3, 0.32, "side")
    if tp is not None:
        S.stroke("FLATTEN", [tp + [0, 0, 0.06], tp, tp - [0, 0, 0.06]], 0.15, 0.35, symmetry_x=True, region="temple")
    # cheekbone (zygoma) soft volume, high and lateral
    S.stroke("CLAY", S.surface_path([(0.33, -0.08), (0.45, -0.06)]), 0.1, 0.3 * float(C.get("cheekbone", 1.0)), symmetry_x=True, region="cheekbone")
    _smooth_area(S, [(0.45, -0.1), (0.45, -0.35), (0.42, -0.55), (0.3, -0.25)], 0.16, iterations=12)
    return {}


# ------------------------------------------------------------------ V6
def v6_lips_nostrils(S, C, P):
    """Video 2:00: lips as two soft volumes, small nostril subtractions."""
    L, W = C["levels"], C["widths"]
    mc = W["mouth_corner"]
    smile = float(C.get("smile", 0.6))          # 0 neutral .. 1 broad closed-mouth smile
    lift = 0.07 * smile                          # corner lift (object units, ~8 mm at smile=1)

    def line_z(x):                               # mouth line: straight centre, corners curving up
        return L["stomion"] + lift * (abs(x) / mc) ** 2.2

    def bow(x):                                  # cupid's bow: dip at the centre, peaks at +-0.045
        return 0.007 * np.exp(-((abs(x) - 0.045) / 0.025) ** 2) - 0.005 * np.exp(-(x / 0.018) ** 2)
    # smile pulls the cheeks up and forward beside the corners (do this before the lips so they sit on it)
    S.stroke("CLAY", S.surface_path([(0.3, -0.38), (0.34, -0.46), (0.36, -0.53)]), 0.12, 0.25 + 0.2 * smile,
             symmetry_x=True, region="cheek")
    S.stroke("INFLATE", S.surface_path([(mc + 0.07, line_z(mc) + 0.03)]), 0.08, 0.15 + 0.2 * smile, symmetry_x=True, region="cheek")
    _smooth_area(S, [(0.33, -0.45)], 0.16, iterations=8, factor=0.4)
    # lip volumes along the smiling line: thin upper lip with a bow, fuller lower lip (from the photo)
    xs_u = np.linspace(-mc * 0.94, mc * 0.94, 15)
    xs_l = np.linspace(-mc * 0.8, mc * 0.8, 13)
    up = [S.surface(x, line_z(x) + 0.021 + bow(x)) for x in xs_u]
    lo = [S.surface(x, line_z(x) - 0.04 + 0.006 * (abs(x) / mc)) for x in xs_l]
    up = np.array([p + [0, 0.004, 0] for p in up if p is not None])
    lo = np.array([p + [0, -0.004, 0] for p in lo if p is not None])
    ls = float(C.get("lip_scale", 1.0))
    vol = TS.lips_volume(up, lo, r_upper=0.03 * ls, r_lower=0.052 * ls)
    co = S.st.co
    allowed = FO.smooth(np.minimum(np.sqrt((co[:, 0] / (mc + 0.06)) ** 2 + ((co[:, 2] - L["stomion"]) / 0.1) ** 2), 1.0))         * (S.st.normals[:, 1] < -0.3)
    for it in range(10):
        rep = S.eng.grow_into_volume(vol, 0.05, 0.01, tmax=0.12, tol=0.0015, allowed=allowed, wide=True,
                                     label="CLAY_INTO_LIP_VOLUME", region="lips", direction=(0, -1, 0))
        S.eng.relax_smooth(allowed, iterations=2, factor=0.3)
        if rep["max_exit_before"] < 0.002:
            break
    # mouth line between the volumes, deepest at the corners (smile tucks them in)
    ml = [(x, line_z(x) - 0.004) for x in np.linspace(-mc, mc, 17)]
    S.stroke("DRAW_SUBTRACT", S.surface_path(ml), 0.016, 0.6, region="mouth", geodesic=True, spacing=0.05)
    mlp = S.surface_path(ml)
    band = _ball_weights(S, mlp, 0.03)
    S.eng.relax_smooth(band, iterations=5, factor=0.35)        # remove dab beading, keep the groove
    S.stroke("DRAW_SUBTRACT", S.surface_path([(mc + 0.008, line_z(mc) + 0.004)]), 0.022, 0.5 + 0.3 * smile,
             symmetry_x=True, region="mouth_corner")
    # vermilion border of the upper lip and philtrum (subtle)
    border = [(x, line_z(x) + 0.042 + bow(x) * 1.4) for x in np.linspace(-mc * 0.8, mc * 0.8, 15)]
    S.stroke("PINCH", S.surface_path(border), 0.014, 0.35, region="upper_lip", geodesic=True)
    for _ in range(2):
        S.stroke("CLAY_STRIPS", S.surface_path([(0.028, L["subnasale"] - 0.015), (0.03, line_z(0) + 0.05)]), 0.014, 0.35,
                 symmetry_x=True, region="philtrum")
    S.stroke("CREASE", S.surface_path([(0, L["subnasale"] - 0.015), (0, line_z(0) + 0.05)]), 0.012, 0.15, region="philtrum")
    # soft nasolabial folds from the alae, around the smiling cheeks, past the corners
    nl = [(W["ala"] + 0.035, L["subnasale"] + 0.03), (W["ala"] + 0.07, L["subnasale"] - 0.07),
          (mc + 0.02, line_z(mc) + 0.03), (mc + 0.045, line_z(mc) - 0.04)]
    S.stroke("CREASE", catmull_rom(S.surface_path(nl), 6), 0.03, 0.15 + 0.15 * smile, symmetry_x=True, region="cheek")
    _smooth_area(S, [(W["ala"] + 0.06, L["subnasale"] - 0.04), (mc + 0.03, line_z(mc))], 0.06, iterations=3, factor=0.3)
    S.eng.relax_smooth(allowed * 0.7, iterations=6, factor=0.3)
    # nostrils from below, columella protected
    st = S.st
    tip = S.surface(0, L["nose_tip"])
    sub = S.surface(0, L["subnasale"])
    allowed = np.abs(st.co[:, 0]) > 0.016
    for sx in (1, -1):
        pts = []
        for yy in np.linspace(tip[1] + 0.03, sub[1] - 0.015, 3):
            j = ray_hit(st, (sx * W["ala"] * 0.42, yy, L["subnasale"] - 0.2), (0, 0, 1))
            if j is not None:
                pts.append(st.co[j])
        if len(pts) >= 2:
            S.stroke("DRAW_SUBTRACT", np.array(pts), 0.022, 0.7, geodesic=True, allowed=allowed, region="nostril")
    S.stroke("CREASE", S.surface_path([(W["ala"] + 0.02, L["subnasale"] + 0.09), (W["ala"] + 0.035, L["subnasale"] + 0.04),
                                       (W["ala"] + 0.02, L["subnasale"])]), 0.02, 0.2, symmetry_x=True, geodesic=True, region="nasal_ala")
    return {}


# ------------------------------------------------------------------ V7
def v7_eyes(S, C, P):
    """Video 2:10-2:35: almond openings cut, lid rims built around them.  Realistic head: eyeballs inside."""
    S.sem = face_map(S.obj)
    S.sem.rebuild(S.st)
    EYE.place_eyeballs(S, C)
    EYE.conform_lids(S, C)
    E, a = C["eye"], C["widths"]["eye_half_width"]
    us = np.linspace(-0.95, 1.0, 12)
    for side, c in zip((1, -1), S.eye_centers):
        upc, loc = EYE.aperture_curves(us, E)
        x = c[0] + side * us * a
        S.stroke("CLAY", S.surface_path(list(zip(x, c[2] + upc + 0.03))), 0.035, 0.22, geodesic=True, region="upper_eyelid")
        S.stroke("CLAY", S.surface_path(list(zip(x[1:-1], c[2] + loc[1:-1] - 0.022))), 0.025, 0.18, geodesic=True, region="lower_eyelid")
        w = _ball_weights(S, [c + [0, -E["ball_radius"], 0]], 0.2)
        # smooth the lids but keep the margin (no smoothing inside 1.2 x ball radius of the opening)
        d = np.linalg.norm(S.st.co - c, axis=1)
        w = w * np.clip((d - E["ball_radius"] * 1.05) / 0.03, 0, 1)
        S.eng.relax_smooth(w, iterations=22, factor=0.45)          # lids must read as smooth skin
    # even out the triangles stretched by the radial wrap, then a light tangential relax around the eyes
    MR.voxel_remesh(S.obj, C.get("eye_voxel", 0.009), log=S.log)
    for f in S.obj.data.polygons:
        f.use_smooth = True
    S.topology_changed()
    for c in S.eye_centers:
        d = np.linalg.norm(S.st.co - c, axis=1)
        w = FO.smooth(np.clip((d - E["ball_radius"]) / 0.12, 0, 1)) * (d > E["ball_radius"] * 1.02)
        S.eng.relax_smooth(np.where(d < E["ball_radius"] + 0.12, 1 - w * 0 , 0) * (d > E["ball_radius"] * 1.02), iterations=6, factor=0.3)
    return EYE.measure_eyes(S, C)


# ------------------------------------------------------------------ V8
def v8_profile(S, C, P):
    """Video 2:48: profile check: nose tip, lips, chin; small brush corrections."""
    L = C["levels"]
    out = {}
    up = S.front_y(0, L["upper_lip"])
    lo = S.front_y(0, L["lower_lip"])
    if C.get("profile_fix", True) and up is not None and lo is not None and up > lo - 0.005:   # upper lip must not sit behind the lower lip
        p = S.surface(0, L["upper_lip"])
        S.stroke("GRAB", [p, p + [0, -(up - lo + 0.008), 0]], 0.13, 1.0, falloff="SMOOTHER", region="upper_lip")
        out["upper_lip_moved_forward"] = True
    out["profile_y"] = {k: S.front_y(0, L[k]) for k in ("brow", "nasion", "nose_tip", "subnasale", "upper_lip", "lower_lip", "mentolabial", "chin")}
    return out


# ------------------------------------------------------------------ V9 / V10 (separate primitives merged)
def _sculpt_ear(S_head, C, P, side):
    """Ear from its own sphere (video 3:02): flatten into a plate, rim, bowl, lobe; returns the object."""
    E = C["ear"]
    st = S_head.st
    j = ray_hit(st, (3 * side, E["y"], -0.06), (-side, 0, 0))
    base = st.co[j]
    es = float(C.get("ear_scale", 1.0))
    ob = MR.create_primitive("UV_SPHERE", f"BNM_Ear_{'L' if side > 0 else 'R'}", 0.2 * es,
                             tuple(base + [side * 0.045 * es, 0.0, 0.0]), detail=4, log=S_head.log)
    # shape with the same engine on the ear object
    ear = Sculptor(ob, log=S_head.log)
    k = es
    co = ear.st.co
    # make it an ear plate: squash across x (pose-like), taller than wide, tilted back like the nose bridge
    c = co.mean(0)
    co[:] = c + (co - c) * [0.26, 0.68, 1.38]          # ~6.5 cm tall, ~3.5 cm wide plate
    ang = np.radians(15)
    rel = co - c
    co[:, 1] = c[1] + rel[:, 1] * np.cos(ang) + rel[:, 2] * np.sin(ang)
    co[:, 2] = c[2] - rel[:, 1] * np.sin(ang) + rel[:, 2] * np.cos(ang)
    ear.st.update_normals()
    ear.st.write()
    out = np.array([side, 0, 0.0])
    # concha bowl (front-lower), rim (helix) build-up, lobe  (all offsets/radii scale with the ear: k)
    bowl = c + np.array([side * 0.05, -0.03, -0.04]) * k
    for r_, s_ in ((0.08, 1.0), (0.06, 1.0), (0.045, 0.9)):
        ear.stroke("DRAW_SUBTRACT", [bowl], r_ * k, s_, normal=out, frontface=False)
    # helix rim around the top and back, antihelix inside it, tragus in front of the bowl, lobe below
    rim = [c + np.array([side * 0.05, 0.1 * np.cos(t) + 0.01, 0.21 * np.sin(t) + 0.02]) * k for t in np.linspace(np.radians(-70), np.radians(215), 14)]
    for s_ in (0.6, 0.5):
        ear.stroke("CLAY_STRIPS", np.array(rim), 0.03 * k, s_, normal=out, frontface=False)
    anti = [c + np.array([side * 0.05, 0.05 * np.cos(t) + 0.01, 0.12 * np.sin(t) + 0.03]) * k for t in np.linspace(np.radians(-40), np.radians(150), 9)]
    ear.stroke("CLAY_STRIPS", np.array(anti), 0.022 * k, 0.45, normal=out, frontface=False)
    ear.stroke("INFLATE", [c + np.array([side * 0.04, -0.09, -0.05]) * k], 0.035 * k, 0.5, normal=out, frontface=False)
    ear.stroke("INFLATE", [c + np.array([side * 0.035, 0.01, -0.22]) * k], 0.06 * k, 0.45, normal=out, frontface=False)
    ear.eng.relax_smooth(None, iterations=6, factor=0.4)
    return ob


def merge_into(S, objs, voxel, blend=0.03, edit=None):
    """SDF smooth-union of separate primitives into the head (video: ears/neck attached)."""
    st = S.st
    lo_all, hi_all = st.co.min(0), st.co.max(0)
    for ob in objs:
        mw = np.array(ob.matrix_world)
        cw = MeshState(ob).co @ mw[:3, :3].T + mw[:3, 3]
        lo_all, hi_all = np.minimum(lo_all, cw.min(0)), np.maximum(hi_all, cw.max(0))
    hw = int(np.ceil((blend + 0.06) / voxel)) + 3
    g = SDF.mesh_to_sdf(st.co, st.tris, voxel, pad=6, half_width=hw, bounds=(lo_all - 6 * voxel, hi_all + 6 * voxel))
    for ob in objs:
        ms = MeshState(ob)
        mw = np.array(ob.matrix_world)
        co = ms.co @ mw[:3, :3].T + mw[:3, 3]
        g2 = SDF.mesh_to_sdf(co, ms.tris, voxel, pad=4, half_width=hw)
        # resample g2 into g's grid where they overlap (index offsets on the same voxel lattice)
        off = np.round((g2.origin - g.origin) / voxel).astype(int)
        lo = np.maximum(off, 0)
        hi = np.minimum(off + np.array(g2.v.shape), np.array(g.v.shape))
        if (hi <= lo).any():
            continue
        a = tuple(slice(l, h) for l, h in zip(lo, hi))
        b = tuple(slice(l - o, h - o) for l, h, o in zip(lo, hi, off))
        g.v[a] = SDF.smooth_union(g.v[a].astype(np.float64), g2.v[b].astype(np.float64), blend).astype(np.float32)
        bpy.data.objects.remove(ob)
    if edit:
        edit(g)
    v, q = g.extract()
    MR._replace_mesh(S.obj, v, q)
    for f in S.obj.data.polygons:
        f.use_smooth = True
    S.topology_changed()


def _cleanup_degenerate(S):
    """Dissolve zero-area faces/edges left by level-set extraction."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(S.obj.data)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bm.to_mesh(S.obj.data)
    bm.free()
    S.obj.data.update()
    S.topology_changed()


def v9_ears(S, C, P, voxel):
    ears = [_sculpt_ear(S, C, P, s) for s in (1, -1)]
    merge_into(S, ears, voxel, blend=0.035)
    _cleanup_degenerate(S)
    S.log.record({"op": "merge_primitives", "what": "ears (separate spheres sculpted, then SDF smooth union)"})
    return {"vertices": len(S.st.co)}


def v10_neck(S, C, P, voxel):
    """Video 3:30: neck = separate cylinder under/behind the jaw, merged."""
    top, bot, r = np.array(P["neck_top"]), np.array(P["neck_bottom"]), P["neck_radius"]
    # rounded neck column (stretched sphere: no hard cylinder rims), top buried inside the head
    nc = tuple(C.get("neck_center", (0.0, 0.16, -1.08)))
    r = float(C.get("neck_radius", r))
    ob = MR.create_primitive("UV_SPHERE", "BNM_Neck", r, nc, detail=5, log=S.log)
    ob.scale = (1.06, 0.95, 2.2)
    bpy.context.view_layer.update()
    cut = P["cut_z"]
    # need the grid to reach the bust cut: pad the head grid downward by extending with the neck object
    ch, ja = np.array(P["chin"], float), np.array(P["jaw_angle"], float)
    nfront = np.array(nc, float) + [0, -0.85 * r, 0]

    def edit(g):
        # rounded underside: submental slope chin -> throat, and jaw lines -> neck sides (no flat bib)
        sub_a = ch + [0, 0.14, 0.03]
        sub_b = np.array([0, nfront[1] + 0.08, -1.12])
        g.union_sdf(lambda Q: SDF.sd_capsule(Q, sub_a, sub_b, 0.15, 0.2), 0.12,
                    np.minimum(sub_a, sub_b) - 0.45, np.maximum(sub_a, sub_b) + 0.45)
        for sx in (1, -1):
            ja_s = ja * [sx, 1, 1] + [0, 0, -0.08]
            nb = np.array([sx * 0.7 * r, nc[1], -1.05])
            g.union_sdf(lambda Q, a_=ja_s, b_=nb: SDF.sd_capsule(Q, a_, b_, 0.11, 0.14), 0.12,
                        np.minimum(ja_s, nb) - 0.4, np.maximum(ja_s, nb) + 0.4)
        g.apply(lambda Q, V: np.maximum(V, cut - Q[..., 2]))
    merge_into(S, [ob], voxel, blend=P["blend"] * float(C.get("neck_blend_scale", 1.0)), edit=edit)
    S.log.record({"op": "merge_primitives", "what": "neck cylinder, SDF smooth union + bust cut"})
    _cleanup_degenerate(S)
    # soften the jaw-neck junction
    S.eng.relax_smooth(_ball_weights(S, [(0.0, -0.1, -0.8), (0.3, 0.0, -0.75), (-0.3, 0.0, -0.75), (0.0, 0.2, -0.7)], 0.4), iterations=25)
    return {"vertices": len(S.st.co)}


def v11_final_smooth(S, C, P):
    from ..geometry.curvature import feature_protection
    S.eng.protect[:] = feature_protection(S.st, strength=1.2, percentile=90)
    S.eng.relax_smooth(None, iterations=6, factor=0.35)
    S.eng.protect[:] = 0
    return {}


def run(S, C, on_stage=None, until=None, detail_voxel=0.0105):
    P = TS.head_params(C)
    S.canon = C
    res = {}
    for name in STAGES:
        S.log.set_stage(name)
        if name == "v1_egg_cranium":
            r = v1_egg_cranium(S, C, P)
        elif name == "v7_eyes":
            MR.voxel_remesh(S.obj, detail_voxel, log=S.log)
            for f in S.obj.data.polygons:
                f.use_smooth = True
            S.topology_changed()
            r = v7_eyes(S, C, P)
        elif name == "v9_ears":
            r = v9_ears(S, C, P, detail_voxel)
        elif name == "v10_neck":
            r = v10_neck(S, C, P, detail_voxel)
        else:
            r = globals()[name](S, C, P)
        res[name] = r
        S.ckpt.create(name)
        if on_stage:
            on_stage(name)
        if until == name:
            break
    return res
