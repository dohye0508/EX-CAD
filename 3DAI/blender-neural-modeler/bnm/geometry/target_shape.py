"""Low-resolution primary-form target for a head (spec 3.1 step 3: 'represent the whole shape as a
low-resolution spatial function').

Only *primary masses* are modelled: cranium, face egg, cheek masses, jaw/chin, neck.  No eyes, nose,
mouth, ears - those are sculpted with brushes.  The target is never extracted as the result; it only
tells the brushes where volume is missing or in excess.

Frame: Z up, face -Y, +X model's left, head height 2 (top +1, menton -1).
"""
import numpy as np

from .sdf import sd_ellipsoid, sd_capsule, sd_sphere, smooth_union, smooth_intersection


def head_params(C=None):
    """Primary-form parameters (Loomis construction), from canon/reference overrides."""
    W = C.get("widths", {}) if C else {}
    fw = W.get("face_half_width", 0.66)
    jw = W.get("jaw_angle", 0.53)
    p = {
        # cranium ball, sides cut flat (Loomis)
        "cranium_center": (0.0, 0.05, 0.22), "cranium_radii": (fw + 0.06, 0.83, 0.78), "side_plane": fw,
        # face block in front of the ball: brow line down to the mouth
        "face_center": (0.0, -0.50, -0.14), "face_half": (fw * 0.80, 0.30, 0.44), "face_round": 0.24,
        "cheek_center": (fw * 0.64, -0.5, -0.3), "cheek_radii": (0.27, 0.26, 0.3),
        # mandible: jaw angles -> chin, mouth block, chin
        "jaw_angle": (jw * 0.9, 0.12, -0.42), "chin": (0.0, -0.66, -0.88), "chin_half_width": W.get("chin", 0.14),
        "jaw_radius": 0.14, "mouth_block": ((0.0, -0.45, -0.62), (fw * 0.7, 0.38, 0.29)),
        "neck_top": (0.0, 0.12, -0.25), "neck_bottom": (0.0, 0.26, -1.6), "neck_radius": 0.40,
        "cut_z": -1.55, "blend": 0.22,
    }
    for k, v in ((C or {}).get("target") or {}).items():   # per-character overrides of the primary forms
        p[k] = tuple(v) if isinstance(v, list) else v
    return p


def head_target(p):
    """Return field(P) -> signed distance (+ outside) of the primary-form target."""
    from .sdf import sd_box

    def field(P):
        P = np.asarray(P, float)
        cr = sd_ellipsoid(P, p["cranium_center"], p["cranium_radii"])
        cr = smooth_intersection(cr, np.abs(P[..., 0]) - p["side_plane"], 0.12)
        face = sd_box(P, p["face_center"], p["face_half"], p["face_round"])
        d = smooth_union(cr, face, 0.34)
        cx, cy, cz = p["cheek_center"]
        for s in (1, -1):
            d = smooth_union(d, sd_ellipsoid(P, (s * cx, cy, cz), p["cheek_radii"]), 0.14)
            if p.get("cheek_low_center") is not None:     # lower cheek fat pad (keeps the face from looking gaunt)
                lx, ly, lz = p["cheek_low_center"]
                d = smooth_union(d, sd_ellipsoid(P, (s * lx, ly, lz), p["cheek_low_radii"]), 0.16)
        ja, ch = np.array(p["jaw_angle"]), np.array(p["chin"])
        jaw = sd_ellipsoid(P, ch + [0, 0.02, 0.0], (p["chin_half_width"] + 0.07, 0.15, 0.13))
        for s in (1, -1):
            seg = sd_capsule(P, ja * [s, 1, 1], ch + [s * p["chin_half_width"], 0.06, 0.03], p["jaw_radius"], p["jaw_radius"])
            jaw = smooth_union(jaw, seg, 0.16)
        mc, mr = p["mouth_block"]
        jaw = smooth_union(jaw, sd_ellipsoid(P, mc, mr), 0.16)
        d = smooth_union(d, jaw, 0.24)
        neck = sd_capsule(P, p["neck_top"], p["neck_bottom"], p["neck_radius"], p["neck_radius"] * 1.06)
        d = smooth_union(d, neck, p["blend"])
        d = np.maximum(d, p["cut_z"] - P[..., 2])
        return d
    return field


def add_volume_field(vol):
    """Field for 'add clay where the feature volume is': vertices on the current surface (distance 0)
    move outward until they reach the feature volume's surface; elsewhere nothing changes."""
    def field(P):
        d = vol(P)
        return np.minimum(d, 0.0)
    return field


def carve_volume_field(vol):
    """Field for 'carve where the cutter volume is': vertices inside the cutter move inward to its surface."""
    def field(P):
        d = vol(P)
        return np.maximum(-d, 0.0)
    return field


def nose_volume(nasion, tip, subnasale, ala_x, ala_y, ala_z, tip_r=0.065, bridge_r=(0.032, 0.055)):
    """Soft nose wedge: bridge (capsule widening toward the tip), tip ball, alae, columella."""
    nasion, tip, sub = (np.asarray(v, float) for v in (nasion, tip, subnasale))

    def vol(P):
        d = sd_capsule(P, nasion, tip, bridge_r[0], bridge_r[1])
        d = smooth_union(d, sd_sphere(P, tip, tip_r), 0.04)
        d = smooth_union(d, sd_capsule(P, tip, sub + [0, -0.01, 0.01], 0.035, 0.03), 0.03)
        for s in (1, -1):
            ala = np.array([s * ala_x, ala_y, ala_z])
            d = smooth_union(d, sd_ellipsoid(P, ala, (0.05, 0.065, 0.045)), 0.07)
            d = smooth_union(d, sd_capsule(P, tip + [0, 0.03, -0.02], ala, 0.04, 0.035), 0.06)   # wing joins the tip
        return d
    return vol


def lips_volume(upper_pts, lower_pts, r_upper=0.03, r_lower=0.038):
    """Upper/lower lip masses as smooth capsule chains along the lip lines."""
    U, Lo = np.asarray(upper_pts, float), np.asarray(lower_pts, float)

    def vol(P):
        from .sdf import sd_path
        du = sd_path(P, U, np.linspace(r_upper * 0.5, r_upper, len(U)) * np.r_[np.linspace(0.6, 1, len(U) // 2), np.linspace(1, 0.6, len(U) - len(U) // 2)])
        dl = sd_path(P, Lo, r_lower * np.r_[np.linspace(0.55, 1, len(Lo) // 2), np.linspace(1, 0.55, len(Lo) - len(Lo) // 2)])
        return smooth_union(du, dl, 0.008)
    return vol


def debug_extract(field, voxel=0.04, lo=(-1.2, -1.4, -1.7), hi=(1.2, 1.4, 1.2)):
    """Extract the target as a mesh for inspection only (never submitted as a result)."""
    from .sdf import SDFGrid
    g = SDFGrid.empty(lo, hi, voxel)
    g.v = field(g.points()).astype(np.float32)
    return g.extract("nets")
