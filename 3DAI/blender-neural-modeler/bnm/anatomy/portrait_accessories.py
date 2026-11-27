"""Hair (sculpted from its own icosphere) and glasses (hard-surface curve sweeps) for a portrait head."""
import math
import numpy as np
import bpy
import bmesh

from ..core.orchestrator import Sculptor
from ..geometry import sdf as SDF
from ..geometry.mesh_io import MeshState
from ..geometry import hard_surface as HS
from ..sculpt.multiresolution import create_primitive

# hairline z by azimuth (deg; 0 = front, 90 = side above the ear, 180 = nape), read from the photo
HAIRLINE = [(0, 0.70), (25, 0.66), (45, 0.52), (70, 0.30), (82, 0.24), (118, 0.24), (140, 0.02), (160, -0.32), (180, -0.42)]


def hairline_z(phi_deg):
    xs, zs = zip(*HAIRLINE)
    return np.interp(np.abs(phi_deg), xs, zs)


def thickness(P):
    """Short hair: fuller on top (photo: volume on top), thin above the ears, medium at the back."""
    z = P[..., 2]
    phi = np.degrees(np.arctan2(P[..., 0], -P[..., 1]))
    top = np.clip((z - 0.25) / 0.65, 0, 1)
    back = np.clip((np.abs(phi) - 110) / 60, 0, 1)
    front_top = np.clip((z - 0.45) / 0.4, 0, 1) * np.clip(1 - np.abs(phi) / 60, 0, 1)   # volume above the forehead
    return 0.035 + 0.06 * top + 0.02 * back + 0.04 * front_top


def sculpt_hair(S, voxel=0.02, iterations=28):
    st = S.st
    g = SDF.mesh_to_sdf(st.co, st.tris, voxel, pad=14, half_width=14)
    ob = create_primitive("ICOSPHERE", "BNM_Hair", 1.3, detail=6, log=S.log)
    for f in ob.data.polygons:
        f.use_smooth = True
    H = Sculptor(ob, log=S.log)

    def field(P):
        return SDF.sdf_sample_mesh_distance(g, P.reshape(-1, 3)).reshape(P.shape[:-1]) - thickness(P)
    for it in range(iterations):
        H.eng.conform_global(field, step=0.06, gain=0.7, region="hair_mass")
        H.eng.relax_smooth(None, iterations=3, factor=0.4)
    # trim everything below the hairline (open shell, like a sculpted hair cap)
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    kill = []
    for f in bm.faces:
        c = f.calc_center_median()
        phi = math.degrees(math.atan2(c.x, -c.y))
        if c.z < hairline_z(phi):
            kill.append(f)
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    H.topology_changed()
    H.eng.relax_smooth(None, iterations=4, factor=0.3)
    # part on his right (-X), flow grooves sweeping toward +X (from the photo)
    def top_point(x, y):
        co = H.st.co
        sel = (np.abs(co[:, 0] - x) < 0.04) & (np.abs(co[:, 1] - y) < 0.04)
        return co[sel][np.argmax(co[sel, 2])] if sel.any() else None
    part = [p for p in (top_point(-0.28, y) for y in np.linspace(-0.62, 0.15, 7)) if p is not None]
    if len(part) > 1:
        H.stroke("CREASE", np.array(part), 0.025, 0.25, region="hair")
    for y0 in np.linspace(-0.6, 0.2, 7):
        pts = [p for p in (top_point(x, y0 + 0.12 * (x + 0.25)) for x in np.linspace(-0.2, 0.55, 7)) if p is not None]
        if len(pts) > 1:
            H.stroke("CREASE", np.array(pts), 0.02, 0.35, region="hair")
    # front fringe volume swept to his left
    fr = [p for p in (top_point(x, -0.6) for x in np.linspace(-0.25, 0.35, 5)) if p is not None]
    if len(fr) > 1:
        H.stroke("CLAY_STRIPS", np.array(fr), 0.08, 0.35, region="hair")
    H.eng.relax_smooth(None, iterations=2, factor=0.3)
    S.extras.append(ob)
    return ob


def glasses(S, C, rim_r=0.011):
    """Rectangular frames (photo): rims, bridge, temple arms as curve sweeps placed from measured surfaces."""
    ex = abs(S.eye_centers[0][0])
    ez = float(S.eye_centers[0][2])
    a, b, n = C["widths"]["eye_half_width"] + 0.07, 0.135, 5.0

    def rim(side):
        out = []
        for t in np.linspace(0, 2 * math.pi, 33):
            c, s = math.cos(t), math.sin(t)
            out.append((side * ex + a * np.sign(c) * abs(c) ** (2 / n), ez + 0.005 + b * np.sign(s) * abs(s) ** (2 / n)))
        return out
    ys = [S.front_y(x, z) for side in (1, -1) for x, z in rim(side)[::2]]
    y0 = min(y for y in ys if y is not None) - 0.035
    y_at = lambda x: y0 + 0.9 * max(abs(x) - 0.32, 0) ** 2
    objs = []
    for side, nm in ((1, "L"), (-1, "R")):
        objs.append(HS.curve_sweep(f"Glasses_Rim_{nm}", [[x, y_at(x), z] for x, z in rim(side)], rim_r, log=S.log))
    yb = S.front_y(0, ez + 0.07)
    objs.append(HS.curve_sweep("Glasses_Bridge", [[-ex + a * 0.98, y_at(ex - a), ez + 0.06], [0, min(y0, yb - 0.03), ez + 0.085],
                                                 [ex - a * 0.98, y_at(ex - a), ez + 0.06]], rim_r * 0.85, log=S.log))
    st = S.st
    from ..geometry.semantic import ray_hit
    for side, nm in ((1, "L"), (-1, "R")):
        hx = side * (ex + a)
        arm = [[hx, y_at(hx), ez + 0.08]]
        for y, z in ((-0.45, 0.11), (-0.2, 0.1), (0.02, 0.09), (0.12, 0.05), (0.2, -0.04)):
            j = ray_hit(st, (3 * side, y, z), (-side, 0, 0))
            if j is not None:
                arm.append([side * (abs(st.co[j][0]) + 0.02), y, z])
        objs.append(HS.curve_sweep(f"Glasses_Temple_{nm}", arm, rim_r * 0.75, log=S.log))
    S.extras.extend(objs)
    return objs
