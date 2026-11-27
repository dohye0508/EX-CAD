"""Eye region: eyeball placement, palpebral fissure (almond aperture), eyelids that wrap the ball
with real thickness, lid crease, canthi.  Works on any head mesh via the Sculptor context."""
import numpy as np

from ..sculpt.multiresolution import create_primitive
from ..sculpt import falloff as FO


def aperture_curves(u, E):
    """Upper/lower lid margins (dz) for lateral coordinate u in [-1, 1] (u<0 medial)."""
    u = np.clip(u, -1, 1)
    base = np.maximum(1 - u * u, 0)
    up = E["upper_arc"] * base ** 0.7 * (1 - 0.18 * u) + E["canthal_tilt"] * u
    lo = -E["lower_arc"] * base ** 0.85 * (1 + 0.22 * u) + E["canthal_tilt"] * u
    return up, lo


def place_eyeballs(S, C):
    E = C["eye"]
    W = C["widths"]
    cz = C["levels"]["eye"]
    balls = []
    import bpy
    for nm in ("BNM_Eye_L", "BNM_Eye_R"):                 # re-running the stage replaces the eyeballs
        for o in [o for o in bpy.data.objects if o.name == nm or o.name.startswith(nm + ".")]:
            if o in S.extras:
                S.extras.remove(o)
            bpy.data.objects.remove(o)
    for side, nm in ((1, "L"), (-1, "R")):
        x = side * W["eye_center"]
        brow = S.surface(side * 0.26, C["levels"]["brow"] - 0.01)
        y_cornea = brow[1] + E["cornea_behind_brow"]
        c = np.array([x, y_cornea + E["ball_radius"], cz])
        ob = create_primitive("UV_SPHERE", f"BNM_Eye_{nm}", E["ball_radius"], tuple(c), detail=3, log=S.log)
        for p in ob.data.polygons:
            p.use_smooth = True
        S.extras.append(ob)
        balls.append(c)
    S.eye_centers = balls
    return balls


def conform_lids(S, C):
    """Wrap the skin over the eyeballs: aperture pushed behind the ball surface, lids form a shell of
    lid_thickness outside it, blending back to the socket over the lid band."""
    E = C["eye"]
    a = C["widths"]["eye_half_width"]
    r = E["ball_radius"]
    reps = []
    for side, c in zip((1, -1), S.eye_centers):
        co = S.st.co
        rel = co - c
        dx, dz = rel[:, 0], rel[:, 2]
        zone = (np.abs(dx) < a * 1.6) & (dz > -0.16) & (dz < 0.17) & (rel[:, 1] < 0.02) & (S.st.normals[:, 1] < 0.3)
        idx = np.nonzero(zone)[0]
        u = (dx[idx] * side) / a
        up, lo = aperture_curves(u, E)
        z = dz[idx]
        # signed distance to the lid margin (negative inside the opening), in object units
        inside_u = np.abs(u) <= 1.0
        d_up = z - up
        d_lo = lo - z
        corner = np.maximum(np.abs(u) - 1.0, 0) * a
        inner = np.minimum(np.minimum(-d_up, -d_lo), (1 - np.abs(u)) * a)   # >0 inside the almond
        outside = np.sqrt(np.maximum(np.maximum(d_up, d_lo), 0) ** 2 + corner ** 2)
        sd = np.where(inside_u & (d_up < 0) & (d_lo < 0), -inner, outside)
        band = np.where(z >= 0, 0.055, 0.04)
        rho = np.linalg.norm(co[idx] - c, axis=1)
        dirv = (co[idx] - c) / np.maximum(rho, 1e-9)[:, None]
        # rounded lid margin: radius rolls from behind the ball (opening) to the lid shell over ~1.6 cm
        t = np.clip((sd + 0.012) / 0.016, 0, 1)
        t = t * t * (3 - 2 * t)
        r_in = r - E["aperture_depth"]
        r_out = r + E["lid_thickness"] + 0.2 * np.clip(sd, 0, None).clip(max=band)
        rho_t = r_in + (r_out - r_in) * t
        w = np.where(sd < 0, 1.0, FO.smooth(np.clip((sd - band * 0.6) / (band * 0.9), 0, 1)))
        # beyond the canthi the wrap fades out quickly (no groove along the canthal line)
        beyond = np.clip((np.abs(u) - 1.0) / 0.35, 0, 1)
        w *= 1 - beyond * beyond * (3 - 2 * beyond)
        # only pull skin that is in front of the ball's equator region
        w *= np.clip((-dirv[:, 1] - 0.05) / 0.25, 0, 1)
        targets = c + dirv * rho_t[:, None]
        reps.append(S.eng.blend_to(idx, targets, w, "CONFORM_EYELID", region="upper_eyelid"))
    return reps


def lid_details(S, C):
    E = C["eye"]
    a = C["widths"]["eye_half_width"]
    us = np.linspace(-0.95, 1.0, 14)
    for side, c in zip((1, -1), S.eye_centers):
        up, lo = aperture_curves(us, E)
        x = c[0] + side * us * a
        crease = S.surface_path(list(zip(x, c[2] + up + 0.042 + 0.012 * (1 - us ** 2))))
        if len(crease) > 2:
            S.stroke("CREASE", crease, 0.022, 0.45, region="upper_eyelid", geodesic=True)
        lower = S.surface_path(list(zip(x[1:-1], c[2] + lo[1:-1] - 0.022)))
        if len(lower) > 2:
            S.stroke("CLAY_STRIPS", lower, 0.018, 0.3, region="lower_eyelid", geodesic=True)
        tear = S.surface_path(list(zip(x[1:-1], c[2] + lo[1:-1] - 0.07)))
        if len(tear) > 2:
            S.stroke("CREASE", tear, 0.03, 0.12, region="lower_eyelid")
        med = S.surface(c[0] - side * a * 1.04, c[2] + E["canthal_tilt"] * -1)
        if med is not None:
            S.stroke("DRAW_SUBTRACT", [med], 0.016, 0.5, region="orbit")
    return True


def measure_eyes(S, C):
    """Numbers for validation: aperture visible, lid shell thickness above the ball."""
    out = {}
    E = C["eye"]
    for nm, c in zip(("L", "R"), S.eye_centers):
        p = S.surface(c[0], c[2])            # skin at eye centre (should be *behind* the cornea)
        cornea_y = c[1] - E["ball_radius"]
        up = S.surface(c[0], c[2] + E["upper_arc"] + 0.012)
        out[nm] = {"skin_y_at_center": None if p is None else float(p[1]), "cornea_y": float(cornea_y),
                   "aperture_open": bool(p is None or p[1] > cornea_y + 0.005),
                   "upper_lid_shell_offset": None if up is None else float(np.linalg.norm(up - c) - E["ball_radius"])}
    return out
