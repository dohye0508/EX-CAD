"""Quantitative form specification built from references.

Every value carries `source` (observed / inferred / canon) and `confidence`.  A single image never
yields absolute 3-D size: values are ratios relative to head height.  The spec converts into canon
overrides that the head pipeline consumes, so the sculpt reproduces the reference's own proportions.
"""
import json

# landmark names expected from the LLM / user on a FRONT image (pixel coords):
FRONT_LANDMARKS = ["vertex", "brow_mid", "eye_L_center", "eye_R_center", "eye_L_outer", "eye_L_inner", "nose_tip",
                   "ala_L", "ala_R", "subnasale", "mouth_L", "mouth_R", "stomion", "chin_bottom", "face_L", "face_R", "jaw_L", "jaw_R"]
# on a SIDE image (face pointing left):
SIDE_LANDMARKS = ["vertex", "glabella", "nasion", "nose_tip", "subnasale", "upper_lip", "lower_lip", "chin", "menton",
                  "back_of_head", "ear_top", "ear_bottom", "tragus"]


def _v(value, source, conf):
    return {"value": None if value is None else round(float(value), 4), "source": source, "confidence": conf}


def build_spec(front=None, side=None, notes=None):
    """front/side: outputs of image_analyzer.analyze_image (with landmarks_norm when available)."""
    spec = {"units": "fraction of head height (vertex->menton)", "values": {}, "unobserved": [], "notes": notes or []}
    V = spec["values"]
    if front and "landmarks_norm" in front:
        L = front["landmarks_norm"]
        top = L.get("vertex", [0, 0])[1]
        bot = L.get("chin_bottom", [0, 1])[1]
        h = max(bot - top, 1e-6)
        rel = lambda k: (L[k][1] - top) / h if k in L else None
        V["eye_line"] = _v(rel("eye_L_center"), "observed", 0.8)
        V["brow_line"] = _v(rel("brow_mid"), "observed", 0.6)
        V["nose_base_line"] = _v(rel("subnasale"), "observed", 0.75)
        V["mouth_line"] = _v(rel("stomion"), "observed", 0.8)
        if "eye_L_center" in L and "eye_R_center" in L:
            V["eye_center_half_width"] = _v(abs(L["eye_L_center"][0] - L["eye_R_center"][0]) / 2 / h, "observed", 0.8)
        if "eye_L_outer" in L and "eye_L_inner" in L:
            V["eye_half_width"] = _v(abs(L["eye_L_outer"][0] - L["eye_L_inner"][0]) / 2 / h, "observed", 0.6)
            dy = L["eye_L_outer"][1] - L["eye_L_inner"][1]
            V["canthal_tilt"] = _v(-dy / h, "observed", 0.5)
        if "ala_L" in L and "ala_R" in L:
            V["ala_half_width"] = _v(abs(L["ala_L"][0] - L["ala_R"][0]) / 2 / h, "observed", 0.7)
        if "mouth_L" in L and "mouth_R" in L:
            V["mouth_half_width"] = _v(abs(L["mouth_L"][0] - L["mouth_R"][0]) / 2 / h, "observed", 0.8)
        if "face_L" in L and "face_R" in L:
            V["face_half_width"] = _v(abs(L["face_L"][0] - L["face_R"][0]) / 2 / h, "observed", 0.7)
        if "jaw_L" in L and "jaw_R" in L:
            V["jaw_half_width"] = _v(abs(L["jaw_L"][0] - L["jaw_R"][0]) / 2 / h, "observed", 0.5)
        asym = []
        for a, b in (("eye_L_center", "eye_R_center"), ("mouth_L", "mouth_R"), ("ala_L", "ala_R")):
            if a in L and b in L:
                asym.append((L[a][1] - L[b][1]) / h)
        if asym:
            V["vertical_asymmetry"] = _v(max(asym, key=abs), "observed", 0.4)
    else:
        spec["unobserved"].append("front proportions")
    if side and "landmarks_norm" in side:
        L = side["landmarks_norm"]
        top = L.get("vertex", [0, 0])[1]
        bot = L.get("menton", [0, 1])[1]
        h = max(bot - top, 1e-6)
        fx = lambda k: L[k][0] / h if k in L else None    # facing left: smaller x = more forward
        if "nose_tip" in L and "subnasale" in L:
            V["nose_tip_projection"] = _v(fx("subnasale") - fx("nose_tip"), "observed", 0.75)
        if "nasion" in L and "glabella" in L:
            V["nasion_depth"] = _v(fx("nasion") - fx("glabella"), "observed", 0.5)
        if "chin" in L and "lower_lip" in L:
            V["chin_vs_lower_lip"] = _v(fx("lower_lip") - fx("chin"), "observed", 0.6)
        if "back_of_head" in L and "glabella" in L:
            V["head_depth"] = _v(fx("back_of_head") - fx("glabella"), "observed", 0.7)
    else:
        spec["unobserved"] += ["side profile (nose projection, chin projection, skull depth)",
                               "back of head"]
    if not side:
        spec["notes"].append("Depth values are inferred from canon, not observed.")
    return spec


def to_canon_overrides(spec):
    """Map spec ratios (head height = 1) to canon values (head height = 2, top z=+1)."""
    V = spec["values"]
    z = lambda r: 1.0 - 2.0 * r
    ov = {"levels": {}, "widths": {}, "eye": {}, "nose": {}}
    g = lambda k: V.get(k, {}).get("value")
    if g("eye_line") is not None:
        ov["levels"]["eye"] = z(g("eye_line"))
    if g("brow_line") is not None:
        ov["levels"]["brow"] = z(g("brow_line"))
    if g("nose_base_line") is not None:
        ov["levels"]["subnasale"] = z(g("nose_base_line"))
        ov["levels"]["nose_tip"] = z(g("nose_base_line")) + 0.1
    if g("mouth_line") is not None:
        ov["levels"]["stomion"] = z(g("mouth_line"))
        ov["levels"]["upper_lip"] = z(g("mouth_line")) + 0.065
        ov["levels"]["lower_lip"] = z(g("mouth_line")) - 0.065
    for k_spec, k_can in (("eye_center_half_width", "eye_center"), ("eye_half_width", "eye_half_width"),
                          ("ala_half_width", "ala"), ("mouth_half_width", "mouth_corner")):
        if g(k_spec) is not None:
            ov["widths"][k_can] = 2.0 * g(k_spec)
    if g("canthal_tilt") is not None:
        ov["eye"]["canthal_tilt"] = 2.0 * g("canthal_tilt")
    if g("nose_tip_projection") is not None:
        ov["nose"]["tip_projection"] = 2.0 * g("nose_tip_projection")
    if g("chin_vs_lower_lip") is not None:
        ov["chin_projection"] = max(0.0, 2.0 * g("chin_vs_lower_lip"))
    return {k: v for k, v in ov.items() if v != {}}


def save(spec, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1, ensure_ascii=False)
