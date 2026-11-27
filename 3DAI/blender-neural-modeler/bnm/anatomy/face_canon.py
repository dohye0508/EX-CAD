"""Head/face proportion canon in object units.

Frame: Z up, face looks toward -Y, +X is the model's LEFT side.  Head height (vertex->chin) = 2.0,
so 1 unit ~ 11.5 cm of a real adult head (23 cm vertex-menton).  Sources of the ratios are
classical drawing canons (Loomis eye line at 1/2 of the head, facial thirds, 'five eyes' width)
and adult anthropometric averages; see knowledge/anatomy_rules.md.  Everything here is a *target
specification*; the sculpt pipeline measures the real mesh and closes the loop against it.
A reference image can override these values (reference/structural_specification.py).
"""
import copy

CM = 2.0 / 23.0  # object units per centimetre

CANON = {
    "head_height": 2.0,
    "levels": {  # z of horizontal landmarks
        "vertex": 1.0, "hairline": 0.72, "brow": 0.22, "nasion": 0.12, "eye": 0.0, "nose_tip": -0.27,
        "subnasale": -0.37, "upper_lip": -0.50, "stomion": -0.565, "lower_lip": -0.63, "mentolabial": -0.72,
        "chin": -0.86, "menton": -1.0,
    },
    "widths": {  # half widths (x)
        "eye_center": 0.27, "eye_half_width": 0.135, "ala": 0.115, "mouth_corner": 0.205,
        "cheekbone": 0.47, "jaw_angle": 0.53, "chin": 0.13,
    },
    "eye": {"ball_radius": 1.2 * CM, "cornea_behind_brow": 0.8 * CM, "upper_arc": 0.06, "lower_arc": 0.045,
            "canthal_tilt": 0.012, "lid_thickness": 0.012, "aperture_depth": 0.03},
    "nose": {"tip_projection": 2.3 * CM, "bridge_projection": 1.1 * CM, "tip_radius": 0.055},
    "lips": {"upper_projection": 0.9 * CM, "lower_projection": 0.7 * CM},
    "chin_projection": 0.6 * CM,     # chin in front of mentolabial fold
    "orbit_depth": 1.4 * CM,         # eye centre surface behind brow ridge (before the eyeball)
    # silhouette targets (Stage 1): z, half_width, front (midline y), back (midline y or None)
    "silhouette": [
        (0.92, 0.42, -0.42, 0.55), (0.75, 0.58, -0.66, 0.80), (0.50, 0.66, -0.82, 0.92),
        (0.25, 0.68, -0.86, 0.90), (0.00, 0.69, -0.80, 0.80), (-0.20, 0.67, -0.84, 0.62),
        (-0.40, 0.61, -0.86, None), (-0.60, 0.52, -0.88, None), (-0.78, 0.36, -0.84, None),
        (-0.92, 0.20, -0.74, None),
    ],
    # para-sagittal front plane (x = +-0.40): the face is flatter than a sphere
    "front_plane": {"x": 0.40, "rows": [(0.45, -0.70), (0.15, -0.70), (-0.15, -0.70), (-0.45, -0.66), (-0.70, -0.52)]},
    "neck": {"top": (0.0, 0.18, -0.55), "bottom": (0.0, 0.30, -1.65), "radius": 0.40, "blend": 0.22, "cut_z": -1.55},
    "ear": {"y": 0.10, "z_top": 0.20, "z_bottom": -0.36, "protrusion": 0.10},
}


# landmarks: name -> (x, z, ray) ; ray "front" casts +Y from the front, "side" casts -X from +X,
# "below" casts +Z from below.  Mirrored names (_L/_R) are generated automatically.
LANDMARKS = {
    "forehead": (0.0, 0.55, "front"), "glabella": (0.0, 0.20, "front"), "nasion": (0.0, 0.12, "front"),
    "brow_ridge_L": (0.26, 0.21, "front"), "eye_L": (0.27, 0.0, "front"), "inner_canthus_L": (0.14, -0.005, "front"),
    "outer_canthus_L": (0.40, 0.015, "front"), "nose_bridge": (0.0, -0.02, "front"), "nose_tip": (0.0, -0.27, "front"),
    "subnasale": (0.0, -0.39, "front"), "ala_L": (0.11, -0.32, "front"), "cheekbone_L": (0.46, -0.10, "front"),
    "cheek_L": (0.36, -0.38, "front"), "philtrum": (0.0, -0.46, "front"), "upper_lip": (0.0, -0.53, "front"),
    "stomion": (0.0, -0.565, "front"), "lower_lip": (0.0, -0.62, "front"), "mouth_corner_L": (0.205, -0.58, "front"),
    "mentolabial": (0.0, -0.72, "front"), "chin": (0.0, -0.86, "front"),
    "temple_L": (-0.30, 0.30, "side"), "jaw_angle_L": (0.05, -0.62, "side"), "ear_L": (0.10, -0.08, "side"),
}

# regions: name -> (landmark, radii (rx, ry, rz) object units, facing)
REGIONS = {
    "forehead": ("forehead", (0.45, 0.25, 0.28), "front"), "glabella": ("glabella", (0.09, 0.12, 0.08), "front"),
    "brow_ridge": ("brow_ridge_L", (0.17, 0.12, 0.07), "front"), "orbit": ("eye_L", (0.17, 0.2, 0.13), "front"),
    "upper_eyelid": ("eye_L", (0.15, 0.2, 0.07), "front"), "lower_eyelid": ("eye_L", (0.15, 0.2, 0.06), "front"),
    "nose_bridge": ("nose_bridge", (0.07, 0.3, 0.16), "front"), "nose_tip": ("nose_tip", (0.07, 0.3, 0.07), "front"),
    "nasal_ala": ("ala_L", (0.05, 0.3, 0.05), "front"), "nostril": ("ala_L", (0.06, 0.3, 0.04), "front"),
    "cheekbone": ("cheekbone_L", (0.14, 0.25, 0.1), "front"), "cheek": ("cheek_L", (0.15, 0.25, 0.15), "front"),
    "philtrum": ("philtrum", (0.05, 0.25, 0.06), "front"), "upper_lip": ("upper_lip", (0.21, 0.25, 0.04), "front"),
    "lower_lip": ("lower_lip", (0.19, 0.25, 0.045), "front"), "mouth_corner": ("mouth_corner_L", (0.05, 0.25, 0.05), "front"),
    "chin": ("chin", (0.16, 0.3, 0.12), "front"), "jaw": ("jaw_angle_L", (0.2, 0.25, 0.2), "side"),
    "temple": ("temple_L", (0.2, 0.15, 0.15), "side"), "ear": ("ear_L", (0.25, 0.2, 0.3), "side"),
    "neck": (None, None, None),
}
# which regions are mirrored: their landmark name ends with _L
ADJACENT = {
    "forehead": ["glabella", "brow_ridge", "temple"], "glabella": ["forehead", "nose_bridge", "brow_ridge"],
    "brow_ridge": ["forehead", "orbit", "upper_eyelid", "temple"], "orbit": ["brow_ridge", "upper_eyelid", "lower_eyelid", "cheekbone", "nose_bridge"],
    "upper_eyelid": ["orbit", "brow_ridge"], "lower_eyelid": ["orbit", "cheek", "cheekbone"],
    "nose_bridge": ["glabella", "orbit", "nose_tip"], "nose_tip": ["nose_bridge", "nasal_ala", "nostril"],
    "nasal_ala": ["nose_tip", "cheek", "nostril"], "nostril": ["nose_tip", "nasal_ala", "philtrum"],
    "cheekbone": ["orbit", "cheek", "temple"], "cheek": ["cheekbone", "nasal_ala", "mouth_corner", "jaw"],
    "philtrum": ["nostril", "upper_lip"], "upper_lip": ["philtrum", "lower_lip", "mouth_corner"],
    "lower_lip": ["upper_lip", "chin", "mouth_corner"], "mouth_corner": ["upper_lip", "lower_lip", "cheek"],
    "chin": ["lower_lip", "jaw"], "jaw": ["chin", "cheek", "ear", "neck"], "temple": ["forehead", "cheekbone", "ear"],
    "ear": ["temple", "jaw"], "neck": ["jaw"],
}
EDIT_DIRECTIONS = {  # semantic edit verbs allowed per region
    "nose_tip": ["forward", "back", "up", "down", "narrow", "widen", "round"], "nose_bridge": ["raise", "lower", "narrow"],
    "nasal_ala": ["narrow", "widen", "raise", "lower"], "upper_eyelid": ["thicken", "thin", "raise", "lower"],
    "lower_eyelid": ["thicken", "thin"], "cheekbone": ["add_volume", "reduce_volume", "raise"],
    "upper_lip": ["fuller", "thinner", "sharpen_border"], "lower_lip": ["fuller", "thinner", "sharpen_border"],
    "chin": ["forward", "back", "down", "up", "narrow", "widen"], "orbit": ["deepen", "shallow"],
    "jaw": ["narrow", "widen", "sharpen"], "cheek": ["add_volume", "hollow", "smooth"],
}
PROTECT = {  # anatomical features a region edit should not destroy
    "nose_tip": ["nostril", "philtrum"], "upper_lip": ["lower_lip", "philtrum"], "lower_lip": ["upper_lip"],
    "upper_eyelid": ["lower_eyelid", "brow_ridge"], "lower_eyelid": ["upper_eyelid"], "cheekbone": ["orbit", "lower_eyelid"],
    "chin": ["lower_lip"], "nose_bridge": ["orbit", "upper_eyelid"], "nasal_ala": ["cheek", "upper_lip"],
}


def canon(overrides=None):
    c = copy.deepcopy(CANON)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(c.get(k), dict):
            c[k].update(v)
        else:
            c[k] = v
    return c


def expanded_landmarks():
    out = {}
    for name, (x, z, ray) in LANDMARKS.items():
        if name.endswith("_L"):
            out[name] = (x, z, ray)
            out[name[:-2] + "_R"] = (-x if ray != "side" else x, z, ray + "_R" if ray == "side" else ray)
        else:
            out[name] = (x, z, ray)
    return out
