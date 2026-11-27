"""Sphere -> Head scratch-sculpt pipeline (Benchmark A).

Every stage computes its strokes from *measurements of the current surface* (landmarks are
ray-cast each time) and from the proportion spec; stages with a measurable target run a closed
feedback loop.  Order follows the professional workflow: silhouette -> planes -> eyes -> nose ->
mouth -> ears/neck -> secondary -> tertiary -> review.  No external meshes are used.
"""
import numpy as np

from . import face_canon as FC
from . import eye_sculptor as EYE
from ..sculpt import multiresolution as MR
from ..sculpt.stroke import ellipse, line, catmull_rom
from ..geometry import sdf as SDF
from ..geometry.semantic import face_map, ray_hit
from ..geometry.mesh_io import MeshState


def _sym_xz(points):
    return [(x, z) for x, z in points]


# ------------------------------------------------------------------ Stage 0
def stage0_initial(S_factory, C, log, detail=6):
    ob = MR.create_primitive("ICOSPHERE", "BNM_Head", 1.0, detail=detail, log=log)
    for p in ob.data.polygons:
        p.use_smooth = True
    return ob


# ------------------------------------------------------------------ Stage 1
def stage1_skull(S, C, iters=10):
    sil = C["silhouette"]
    fp = C["front_plane"]

    def measure():
        co = S.st.co
        e = []
        for z, hw, fr, bk in sil:
            sel = np.abs(co[:, 2] - z) < 0.035
            P = co[sel]
            mid = np.abs(P[:, 0]) < 0.07
            e.append(hw - np.abs(P[:, 0]).max())
            e.append(fr - P[mid, 1].min() if mid.any() else 0.0)
            e.append((bk - P[mid, 1].max()) if (bk is not None and mid.any()) else 0.0)
        for z, y in fp["rows"]:
            sel = (np.abs(co[:, 2] - z) < 0.035) & (np.abs(co[:, 0] - fp["x"]) < 0.05) & (co[:, 1] < 0)
            e.append(y - co[sel, 1].min() if sel.any() else 0.0)
        return np.array(e)

    def act(e, g):
        co = S.st.co.copy()
        k = 0
        for z, hw, fr, bk in sil:
            e_hw, e_fr, e_bk = e[k:k + 3]
            k += 3
            sel = np.abs(co[:, 2] - z) < 0.035
            P = co[sel]
            mid = np.abs(P[:, 0]) < 0.07
            if abs(e_hw) > 0.008:
                p = P[np.argmax(P[:, 0])]
                S.stroke("ELASTIC_DEFORM", [p, p + [e_hw * g, 0, 0]], 0.24, 1.0, symmetry_x=True, region="skull")
            if abs(e_fr) > 0.008 and mid.any():
                p = P[mid][np.argmin(P[mid, 1])]
                S.stroke("ELASTIC_DEFORM", [p, p + [0, e_fr * g, 0]], 0.22, 1.0, region="face_front")
            if abs(e_bk) > 0.008 and mid.any():
                p = P[mid][np.argmax(P[mid, 1])]
                S.stroke("ELASTIC_DEFORM", [p, p + [0, e_bk * g, 0]], 0.26, 1.0, region="occiput")
        for (z, y), ee in zip(fp["rows"], e[k:]):
            if abs(ee) > 0.008:
                sel = (np.abs(co[:, 2] - z) < 0.035) & (np.abs(co[:, 0] - fp["x"]) < 0.05) & (co[:, 1] < 0)
                if sel.any():
                    P = co[sel]
                    p = P[np.argmin(P[:, 1])]
                    S.stroke("ELASTIC_DEFORM", [p, p + [0, ee * g, 0]], 0.2, 1.0, symmetry_x=True, region="face_plane")

    hist = S.closed_loop("stage1_silhouette", measure, act, tol=0.015, max_iter=iters, gain=0.45)
    # remove kelvinlet ripples: tangential relax keeps volume, then a light Laplacian pass
    allv = np.arange(len(S.st.co))
    S.eng.smooth_region(allv, iterations=8, factor=0.5, tangential=True)
    S.eng.smooth_region(allv, iterations=10, factor=0.45)
    return hist


def stage1_neck_and_remesh(S, C, voxel=0.02):
    N = C["neck"]

    def edit(g):
        a, b = np.array(N["top"]), np.array(N["bottom"])
        pad = N["radius"] + N["blend"] + 0.05
        lo = np.minimum(a, b) - pad
        hi = np.maximum(a, b) + pad
        g.union_sdf(lambda P: SDF.sd_capsule(P, a, b, N["radius"], N["radius"] * 1.08), N["blend"], lo, hi)
        cz = N["cut_z"]
        g.apply(lambda P, V: np.maximum(V, cz - P[..., 2]))   # flat bust cut
    # grid must reach the neck bottom: extend bbox by meshing a padded grid
    st = S.st
    hw = int(np.ceil((N["blend"] + 0.12) / voxel)) + 2   # narrow band must cover the blend zone
    g = SDF.mesh_to_sdf(st.co, st.tris, voxel, pad=4, half_width=hw)
    # enlarge grid downward to contain the neck
    need_lo = np.array([-0.8, -0.6, N["cut_z"] - 0.1])
    if (g.origin > need_lo).any():
        lo = np.minimum(g.origin, need_lo)
        shift = np.round((g.origin - lo) / voxel).astype(int)
        big = np.full(tuple(np.array(g.v.shape) + shift), float(g.v.max()), dtype=np.float32)  # same background -> no seam
        big[shift[0]:, shift[1]:, shift[2]:] = g.v
        g = SDF.SDFGrid(big, g.origin - shift * voxel, voxel)
    edit(g)
    v, q = g.extract()
    MR._replace_mesh(S.obj, v, q)
    for p in S.obj.data.polygons:
        p.use_smooth = True
    S.log.record({"op": "sdf_union", "what": "neck capsule (auxiliary volume)", "voxel": voxel,
                  "vertices_after": len(S.obj.data.vertices)})
    S.topology_changed()
    S.eng.smooth_region(np.arange(len(S.st.co)), iterations=2, factor=0.3)


# ------------------------------------------------------------------ Stage 2
def stage2_planes(S, C):
    L, W = C["levels"], C["widths"]
    # brow ridge band (slight arch)
    xs = np.linspace(-0.44, 0.44, 13)
    for dz, s in ((0.0, 0.5), (-0.03, 0.3)):
        brow = S.surface_path([(x, L["brow"] + dz + 0.025 * (1 - (x / 0.44) ** 2)) for x in xs])
        S.stroke("CLAY_STRIPS", brow, 0.075, s, region="brow_ridge")
    # nasion dip, glabella
    S.stroke("DRAW_SUBTRACT", S.surface_path([(0, L["nasion"])]), 0.06, 0.35, region="glabella")

    # orbits: closed loop on socket depth
    def m_orbit():
        e = []
        for sx in (1, -1):
            eye = S.surface(sx * W["eye_center"], L["eye"])
            br = S.surface(sx * 0.26, L["brow"])
            e.append(C["orbit_depth"] - (eye[1] - br[1]))
        return np.array(e)

    def a_orbit(e, g):
        err = float(np.mean(e))
        ring = [(W["eye_center"] + 0.04 * np.cos(t), L["eye"] + 0.03 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 10)]
        path = S.surface_path(ring)
        S.stroke("DRAW_SUBTRACT", path, 0.13, float(np.clip(err / 0.05 * g, 0.05, 1.0)), symmetry_x=True, region="orbit")
    S.closed_loop("orbit_depth", m_orbit, a_orbit, tol=0.012, max_iter=8)
    S.stroke("SMOOTH", S.surface_path([(W["eye_center"], L["eye"])]), 0.14, 0.4, symmetry_x=True, region="orbit")

    # cheekbones (zygoma): volume high and lateral, then hollow below
    for s in (0.45, 0.3):
        S.stroke("CLAY", S.surface_path([(0.36, -0.08), (0.46, -0.06), (0.55, -0.03)]), 0.1, s, symmetry_x=True, region="cheekbone")
    S.stroke("DRAW_SUBTRACT", S.surface_path([(0.44, -0.40), (0.47, -0.46)]), 0.12, 0.22, symmetry_x=True, region="cheek")
    # temples: flatten the side planes
    tp = S.surface(-0.32, 0.30, "side")
    if tp is not None:
        S.stroke("FLATTEN", [tp, tp + [0, 0, -0.08]], 0.16, 0.5, symmetry_x=True, region="temple")
    # forehead front plane
    fh = S.surface_path([(x, z) for z in (0.62, 0.5, 0.38) for x in np.linspace(-0.3, 0.3, 5)])
    S.stroke("FLATTEN", fh, 0.18, 0.25, region="forehead")
    # muzzle / dental arch barrel
    for z, s in ((-0.47, 0.3), (-0.57, 0.35), (-0.66, 0.25)):
        S.stroke("CLAY_STRIPS", S.surface_path([(x, z) for x in np.linspace(-0.26, 0.26, 9)]), 0.07, s, region="muzzle")
    # chin mound
    S.stroke("CLAY", S.surface_path([(-0.04, L["chin"]), (0.04, L["chin"])]), 0.1, 0.45, region="chin")
    # jaw line: build along side, flatten underneath
    jaw = [S.surface(y, z, "side") for y, z in ((0.10, -0.60), (-0.12, -0.72), (-0.35, -0.82), (-0.55, -0.88))]
    jaw = np.array([p for p in jaw if p is not None])
    if len(jaw) > 1:
        S.stroke("CLAY_STRIPS", catmull_rom(jaw, 6), 0.06, 0.35, symmetry_x=True, region="jaw")
        S.stroke("PINCH", catmull_rom(jaw - [0, 0, 0.03], 6), 0.05, 0.3, symmetry_x=True, region="jaw")
    st = S.st
    under = []
    for x, y in ((0.25, -0.35), (0.3, -0.1), (0.15, -0.5)):
        j = ray_hit(st, (x, y, -3.0), (0, 0, 1))
        if j is not None:
            under.append(st.co[j])
    if len(under) > 1:
        S.stroke("FLATTEN", np.array(under), 0.14, 0.3, symmetry_x=True, normal=(0, 0, -1), region="jaw")
    # unify
    face = S.surface_path([(x, z) for z in np.linspace(0.6, -0.8, 8) for x in (-0.3, 0.0, 0.3)])
    S.stroke("SURFACE_SMOOTH", face, 0.15, 0.4, region="face")
    S.sem.rebuild(S.st)
    S.sem.mark("forehead", "glabella", "brow_ridge", "orbit", "cheekbone", "cheek", "temple", "jaw", "chin")


# ------------------------------------------------------------------ Stage 3
def stage3_eyes(S, C):
    EYE.place_eyeballs(S, C)
    EYE.conform_lids(S, C)
    S.eng.st.update_normals()
    # relax the lid band without destroying the margin: tangential relax, geodesic
    for c, sx in zip(S.eye_centers, (1, -1)):
        S.stroke("RELAX", S.surface_path([(c[0], c[2] + 0.07), (c[0], c[2] - 0.06)]), 0.07, 0.3, geodesic=True, region="upper_eyelid")
    EYE.lid_details(S, C)
    S.sem.rebuild(S.st)
    S.sem.mark("upper_eyelid", "lower_eyelid", "orbit")
    return EYE.measure_eyes(S, C)


# ------------------------------------------------------------------ Stage 4
def stage4_nose(S, C):
    L, W, N = C["levels"], C["widths"], C["nose"]
    ref_y = lambda: S.front_y(0.2, -0.3)

    def m_tip():
        tip = S.surface(0, L["nose_tip"])
        return np.array([(ref_y() - N["tip_projection"]) - tip[1]])

    def a_tip(e, g):
        tip = S.surface(0, L["nose_tip"])
        S.stroke("GRAB", [tip, tip + [0, e[0] * g, -0.15 * e[0] * g]], 0.11, 1.0, region="nose_tip")
        S.stroke("SNAKE_HOOK", [tip, tip + [0, e[0] * g * 0.3, 0]], 0.06, 0.6, region="nose_tip")
    S.closed_loop("nose_tip_projection", m_tip, a_tip, tol=0.012, max_iter=10)

    def m_bridge():
        b = S.surface(0, -0.03)
        ic = S.surface(0.14, 0.0)
        return np.array([(ic[1] - N["bridge_projection"]) - b[1]])

    def a_bridge(e, g):
        path = S.surface_path([(0, z) for z in np.linspace(L["nasion"], L["nose_tip"] + 0.05, 8)])
        S.stroke("CLAY_STRIPS", path, 0.045, float(np.clip(-e[0] / 0.03 * g, 0.1, 1.0)), region="nose_bridge")
    S.closed_loop("nose_bridge_projection", m_bridge, a_bridge, tol=0.012, max_iter=8)
    # side planes of the nose
    for sx in (1,):
        side = S.surface_path([(sx * 0.055, z) for z in np.linspace(0.04, L["nose_tip"] + 0.04, 6)])
        S.stroke("FLATTEN", side, 0.04, 0.35, symmetry_x=True, normal=(0.75, -0.66, 0), region="nose_bridge")
    # tip ball + domes
    tip = S.surface(0, L["nose_tip"] - 0.01)
    S.stroke("INFLATE", [tip], N["tip_radius"], 0.6, region="nose_tip")
    S.stroke("CLAY", S.surface_path([(0.025, L["nose_tip"] + 0.01)]), 0.03, 0.4, symmetry_x=True, region="nose_tip")
    # alae (wings)
    for s in (0.7, 0.5):
        S.stroke("INFLATE", S.surface_path([(W["ala"] - 0.01, -0.31), (W["ala"], -0.34)]), 0.05, s, symmetry_x=True, region="nasal_ala")
    al = S.surface(W["ala"], -0.33)
    if al is not None:
        S.stroke("GRAB", [al, al + [0.015, -0.01, 0]], 0.05, 1.0, symmetry_x=True, region="nasal_ala")
    # alar groove around the wing
    groove = [(W["ala"] + 0.045 * np.cos(t), -0.33 + 0.045 * np.sin(t)) for t in np.linspace(np.radians(110), np.radians(-80), 8)]
    S.stroke("CREASE", S.surface_path(groove), 0.022, 0.55, symmetry_x=True, geodesic=True, region="nasal_ala")
    # nostrils: carve upward from below the tip
    st = S.st
    tip = S.surface(0, L["nose_tip"])
    sub = S.surface(0, L["subnasale"])
    allowed = np.abs(st.co[:, 0]) > 0.014                       # protect the columella
    for sx in (1, -1):
        pts = []
        for yy in np.linspace(tip[1] + 0.035, sub[1] - 0.015, 4):
            j = ray_hit(st, (sx * 0.042, yy, L["subnasale"] - 0.2), (0, 0, 1))
            if j is not None:
                pts.append(st.co[j])
        if len(pts) >= 2:
            for _ in range(3):
                S.stroke("DRAW_SUBTRACT", np.array(pts), 0.022, 0.9, geodesic=True, allowed=allowed, region="nostril")
    # blend nose into cheeks
    S.stroke("SURFACE_SMOOTH", S.surface_path([(0.16, 0.0), (0.18, -0.2), (0.17, -0.32)]), 0.05, 0.4, symmetry_x=True, region="nose_bridge")
    S.sem.rebuild(S.st)
    S.sem.mark("nose_bridge", "nose_tip", "nasal_ala", "nostril")


# ------------------------------------------------------------------ Stage 5
def stage5_mouth(S, C):
    L, W = C["levels"], C["widths"]
    mc = W["mouth_corner"]
    # philtrum columns + groove
    for _ in range(2):
        S.stroke("CLAY_STRIPS", S.surface_path([(0.03, -0.41), (0.032, -0.51)]), 0.016, 0.45, symmetry_x=True, region="philtrum")
    S.stroke("CREASE", S.surface_path([(0, -0.41), (0, -0.51)]), 0.016, 0.25, region="philtrum")
    # mouth line (stomion) with slight downturn at corners
    ml = [(x, L["stomion"] - 0.02 * (abs(x) / mc) ** 2) for x in np.linspace(-mc, mc, 13)]
    for s in (0.8, 0.6):
        S.stroke("CREASE", S.surface_path(ml), 0.032, s, pinch=0.8, region="mouth")
    # upper lip with cupid's bow
    bow = [(-mc + 0.01, -0.555), (-0.1, -0.53), (-0.035, -0.515), (0.0, -0.525), (0.035, -0.515), (0.1, -0.53), (mc - 0.01, -0.555)]
    upper = S.surface_path(bow)
    S.stroke("CLAY_STRIPS", catmull_rom(upper, 5), 0.035, 0.5, region="upper_lip", geodesic=True)
    border = S.surface_path([(x, z + 0.022) for x, z in bow])
    S.stroke("PINCH", catmull_rom(border, 5), 0.02, 0.5, region="upper_lip", geodesic=True)
    # lower lip fuller
    lower = S.surface_path([(x, L["lower_lip"] + 0.015 * (1 - (x / 0.16) ** 2)) for x in np.linspace(-0.16, 0.16, 9)])
    for s in (0.55, 0.4):
        S.stroke("CLAY_STRIPS", lower, 0.04, s, region="lower_lip", geodesic=True)
    S.stroke("INFLATE", S.surface_path([(0, L["lower_lip"])]), 0.06, 0.3, region="lower_lip", geodesic=True)
    # mouth corners
    S.stroke("DRAW_SUBTRACT", S.surface_path([(mc, L["stomion"] - 0.02)]), 0.03, 0.5, symmetry_x=True, region="mouth_corner")
    # mentolabial sulcus and chin projection loop
    S.stroke("DRAW_SUBTRACT", S.surface_path([(x, L["mentolabial"]) for x in np.linspace(-0.12, 0.12, 7)]), 0.05, 0.4, region="chin")

    def m_chin():
        ch = S.surface(0, L["chin"])
        ml_ = S.surface(0, L["mentolabial"])
        return np.array([(ml_[1] - C["chin_projection"]) - ch[1]])

    def a_chin(e, g):
        ch = S.surface(0, L["chin"])
        S.stroke("GRAB", [ch, ch + [0, e[0] * g, 0]], 0.13, 1.0, region="chin")
    S.closed_loop("chin_projection", m_chin, a_chin, tol=0.01, max_iter=6)
    # nasolabial folds (subtle) and cheek mass beside them
    nl = [(W["ala"] + 0.035, -0.34), (0.17, -0.45), (0.22, -0.56), (0.25, -0.64)]
    S.stroke("CREASE", catmull_rom(S.surface_path(nl), 5), 0.03, 0.18, symmetry_x=True, region="cheek")
    S.stroke("CLAY", catmull_rom(S.surface_path([(x + 0.06, z) for x, z in nl[:3]]), 5), 0.05, 0.2, symmetry_x=True, region="cheek")
    S.sem.rebuild(S.st)
    S.sem.mark("philtrum", "upper_lip", "lower_lip", "mouth_corner", "chin")


# ------------------------------------------------------------------ Stage 6
def stage6_ears_neck(S, C):
    E = C["ear"]
    # pull the ear mass out of the side of the head (snake hook), as a vertical plate
    for z in (0.1, -0.06, -0.22):
        p = S.surface(E["y"], z, "side")
        if p is not None:
            S.stroke("SNAKE_HOOK", line(p, p + [E["protrusion"], 0.04, 0], 10), 0.11, 0.9, symmetry_x=True, region="ear")
    ec = S.surface(E["y"], -0.06, "side")
    S.stroke("INFLATE", [ec + [0, 0, 0.1], ec, ec - [0, 0, 0.14]], 0.1, 0.35, symmetry_x=True, region="ear")
    ec = S.surface(E["y"], -0.06, "side")
    S.stroke("FLATTEN", [ec + [0, 0, 0.12], ec, ec - [0, 0, 0.16]], 0.13, 0.45, symmetry_x=True, normal=(0.94, -0.34, 0), region="ear")
    # helix rim, concha bowl, antihelix, tragus, lobe
    rim = [(E["y"] + 0.075 * np.cos(t) + 0.01, -0.05 + 0.2 * np.sin(t)) for t in np.linspace(np.radians(-80), np.radians(230), 14)]
    rp = np.array([p for p in (S.surface(a, z, "side") for a, z in rim) if p is not None])
    if len(rp) > 3:
        S.stroke("CLAY_STRIPS", rp, 0.025, 0.5, symmetry_x=True, region="ear")
    for _ in range(3):
        cc = S.surface(E["y"] - 0.01, -0.11, "side")
        S.stroke("DRAW_SUBTRACT", [cc], 0.045, 0.8, symmetry_x=True, region="ear")
    ah = [(E["y"] + 0.035 * np.cos(t), -0.05 + 0.11 * np.sin(t)) for t in np.linspace(np.radians(-30), np.radians(160), 8)]
    ap = np.array([p for p in (S.surface(a, z, "side") for a, z in ah) if p is not None])
    if len(ap) > 2:
        S.stroke("CLAY_STRIPS", ap, 0.016, 0.35, symmetry_x=True, region="ear")
    tr = S.surface(E["y"] - 0.085, -0.12, "side")
    if tr is not None:
        S.stroke("INFLATE", [tr], 0.03, 0.5, symmetry_x=True, region="ear")
    lb = S.surface(E["y"] + 0.01, E["z_bottom"] + 0.06, "side")
    if lb is not None:
        S.stroke("INFLATE", [lb], 0.045, 0.45, symmetry_x=True, region="ear")
    # neck: sternocleidomastoid from mastoid to sternum, smooth throat
    scm = np.array([p for p in (S.surface(y, z, "side") for y, z in ((0.28, -0.58), (0.12, -0.9), (-0.05, -1.2), (-0.18, -1.45))) if p is not None])
    if len(scm) > 1:
        S.stroke("CLAY_STRIPS", catmull_rom(scm, 6), 0.075, 0.35, symmetry_x=True, region="neck")
    throat = S.surface_path([(0, -0.95), (0, -1.2), (0, -1.45)])
    S.stroke("SURFACE_SMOOTH", throat, 0.12, 0.4, region="neck")
    S.sem.rebuild(S.st)
    S.sem.mark("ear", "neck")


# ------------------------------------------------------------------ Stage 7 / 8
def stage7_secondary(S, C):
    from ..geometry.curvature import feature_protection
    prot = feature_protection(S.st, strength=1.2, percentile=88)
    S.eng.protect[:] = prot
    face = S.surface_path([(x, z) for z in np.linspace(0.55, -0.85, 10) for x in (-0.42, -0.2, 0.0, 0.2, 0.42)])
    S.stroke("SURFACE_SMOOTH", face, 0.12, 0.35, region="face")
    S.stroke("RELAX", face, 0.12, 0.3, region="face")
    S.eng.protect[:] = 0
    # cheek-zygoma transition and jaw definition
    S.stroke("SMOOTH", S.surface_path([(0.46, -0.2), (0.4, -0.3)]), 0.1, 0.3, symmetry_x=True, region="cheek")
    jaw = np.array([p for p in (S.surface(y, z, "side") for y, z in ((0.05, -0.66), (-0.25, -0.8), (-0.5, -0.88))) if p is not None])
    if len(jaw) > 1:
        S.stroke("PINCH", catmull_rom(jaw - [0, 0, 0.04], 5), 0.04, 0.25, symmetry_x=True, region="jaw")


def stage8_tertiary(S, C):
    L = C["levels"]
    # lip texture: fine vertical creases, eyelid fold accents
    for x in np.linspace(-0.12, 0.12, 9):
        S.stroke("CREASE", S.surface_path([(x, L["lower_lip"] + 0.02), (x * 1.05, L["lower_lip"] - 0.02)]), 0.007, 0.12, region="lower_lip")
        S.stroke("CREASE", S.surface_path([(x, L["upper_lip"] - 0.005), (x * 1.05, L["upper_lip"] - 0.03)]), 0.006, 0.1, region="upper_lip")
    for c, sx in zip(S.eye_centers, (1, -1)):
        S.stroke("CREASE", S.surface_path([(c[0] - sx * 0.06, c[2] - 0.075), (c[0] + sx * 0.06, c[2] - 0.08)]), 0.01, 0.12, region="lower_eyelid")
    # alar crease accent and slight natural asymmetry
    S.stroke("CREASE", S.surface_path([(C["widths"]["ala"] + 0.04, -0.3), (C["widths"]["ala"] + 0.035, -0.36)]), 0.012, 0.25, region="nasal_ala")
    S.stroke("CLAY", S.surface_path([(0.46, -0.06)]), 0.08, 0.08, region="cheekbone")  # left zygoma 1-2 mm fuller


# ------------------------------------------------------------------ metrics
def face_metrics(S, C):
    L, W = C["levels"], C["widths"]
    co = S.st.co
    head = co[co[:, 2] > -0.95]
    top = float(head[:, 2].max())
    m = {}
    def fy(x, z):
        return S.front_y(x, z)
    tip, sub, cheek = fy(0, L["nose_tip"]), fy(0, L["subnasale"]), fy(0.2, -0.3)
    m["nose_tip_projection"] = cheek - tip if tip is not None else None
    m["nose_bridge_projection"] = (fy(0.14, 0.0) - fy(0, -0.03)) if fy(0, -0.03) is not None else None
    m["orbit_depth_skin"] = None
    m["upper_lip_projection"] = fy(0, L["subnasale"] - 0.03) - fy(0, L["upper_lip"])
    m["lower_lip_projection"] = fy(0, L["mentolabial"]) - fy(0, L["lower_lip"])
    m["chin_projection"] = fy(0, L["mentolabial"]) - fy(0, L["chin"])
    sl = np.abs(co[:, 2] - 0.0) < 0.03
    m["face_width_at_eyes"] = float(2 * np.abs(co[sl, 0]).max())
    m["head_height_top_to_menton"] = top - float(co[(np.abs(co[:, 0]) < 0.05) & (co[:, 1] < -0.4), 2].min())
    m["eye_line_ratio"] = (top - L["eye"]) / m["head_height_top_to_menton"]
    ear = np.abs(co[:, 0])[(np.abs(co[:, 2] + 0.06) < 0.1) & (np.abs(co[:, 1] - C["ear"]["y"]) < 0.12)].max()
    m["ear_protrusion_from_skull"] = float(ear - C["widths"]["jaw_angle"] - 0.15)
    targets = {"nose_tip_projection": C["nose"]["tip_projection"], "nose_bridge_projection": C["nose"]["bridge_projection"],
               "chin_projection": C["chin_projection"], "eye_line_ratio": 0.5}
    return {k: (round(float(v), 4) if v is not None else None) for k, v in m.items()}, targets


STAGES = ["stage0_initial", "stage1_skull", "stage2_planes", "stage3_eyes", "stage4_nose", "stage5_mouth",
          "stage6_ears_neck", "stage7_secondary", "stage8_tertiary", "stage9_review"]


def run(S, C, remesh_primary=0.02, remesh_detail=0.0105, until=None, on_stage=None):
    """Run the whole pipeline on a Sculptor whose object is the initial sphere."""
    results = {}
    S.sem = face_map(S.obj)
    S.sem.rebuild(S.st)

    def done(name, data=None):
        S.ckpt.create(name)
        if on_stage:
            on_stage(name)
        results[name] = data
        return until == name

    S.log.set_stage("stage0_initial")
    if done("stage0_initial", {"vertices": len(S.st.co)}):
        return results
    S.log.set_stage("stage1_skull")
    h = stage1_skull(S, C)
    stage1_neck_and_remesh(S, C, remesh_primary)
    if done("stage1_skull", {"loop": h[-1], "vertices": len(S.st.co)}):
        return results
    S.log.set_stage("stage2_planes")
    stage2_planes(S, C)
    if done("stage2_planes", {"vertices": len(S.st.co)}):
        return results
    S.log.set_stage("stage3_eyes")
    MR.voxel_remesh(S.obj, remesh_detail, log=S.log)
    for p in S.obj.data.polygons:
        p.use_smooth = True
    S.topology_changed()
    r3 = stage3_eyes(S, C)
    if done("stage3_eyes", r3):
        return results
    S.log.set_stage("stage4_nose")
    stage4_nose(S, C)
    if done("stage4_nose"):
        return results
    S.log.set_stage("stage5_mouth")
    stage5_mouth(S, C)
    if done("stage5_mouth"):
        return results
    S.log.set_stage("stage6_ears_neck")
    stage6_ears_neck(S, C)
    if done("stage6_ears_neck"):
        return results
    S.log.set_stage("stage7_secondary")
    stage7_secondary(S, C)
    if done("stage7_secondary"):
        return results
    S.log.set_stage("stage8_tertiary")
    stage8_tertiary(S, C)
    if done("stage8_tertiary"):
        return results
    S.log.set_stage("stage9_review")
    from ..validation.geometry_validator import validate_mesh
    v = validate_mesh(S.obj)
    if "self-intersections" in v["problems"]:
        # voxel remesh resolves interpenetrating sheets (nostrils/creases) at a finer voxel than detail
        MR.voxel_remesh(S.obj, remesh_detail * 0.8, log=S.log)
        for p in S.obj.data.polygons:
            p.use_smooth = True
        S.topology_changed()
        S.log.record({"op": "repair", "reason": "self-intersections", "pairs_before": v.get("self_intersecting_face_pairs")})
    results["stage9_review"] = face_metrics(S, C)
    done("stage9_review", results["stage9_review"])
    return results
