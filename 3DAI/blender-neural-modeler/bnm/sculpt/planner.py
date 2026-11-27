"""Sculpt Stroke Planner.

LLM decides *what* (region + intent); this module computes *how*: brush, path on the live
surface, radius/strength from the region size, protection masks, then executes and verifies
(displacement inside vs outside the region, direction of change) and rolls back on failure.

Plan item schema (JSON-serialisable):
{"operation":"sculpt","target":obj,"region":r,"brush":B,"purpose":p,"coordinate_space":"OBJECT",
 "path":[[x,y,z],...],"radius":R,"strength":s,"falloff":"SMOOTH","direction":"SURFACE_NORMAL"|"ADD"|"SUBTRACT",
 "drag":[dx,dy,dz] (drag brushes),"symmetry":bool,"protect_regions":[...],"geodesic":bool,"verify_after":true}
"""
import re
import numpy as np

from .stroke import catmull_rom

REGION_WORDS = {
    "nose_tip": ["코끝", "nose tip", "tip of the nose"], "nose_bridge": ["콧대", "nose bridge", "bridge"],
    "nasal_ala": ["콧방울", "비익", "ala", "nostril wing", "alae"], "nostril": ["콧구멍", "nostril"],
    "upper_eyelid": ["윗눈꺼풀", "위눈꺼풀", "upper eyelid", "upper lid"], "lower_eyelid": ["아래눈꺼풀", "lower eyelid", "lower lid"],
    "orbit": ["안와", "눈 주위", "eye socket", "orbit"], "cheekbone": ["광대", "cheekbone", "zygoma"],
    "cheek": ["볼", "cheek"], "upper_lip": ["윗입술", "upper lip"], "lower_lip": ["아랫입술", "lower lip"],
    "mouth_corner": ["입꼬리", "mouth corner"], "philtrum": ["인중", "philtrum"], "chin": ["턱끝", "chin"],
    "jaw": ["턱선", "jaw", "jawline"], "forehead": ["이마", "forehead"], "brow_ridge": ["눈썹뼈", "brow ridge", "brow"],
    "glabella": ["미간", "glabella"], "temple": ["관자놀이", "temple"], "ear": ["귀", "ear"], "neck": ["목", "neck"],
}
LIPS = ["입술", "lips"]
ACTION_WORDS = [  # (intent, words) checked in order
    ("sharpen", ["선명", "sharpen", "define", "crisp", "경계"]),
    ("smooth", ["자연스럽", "부드럽", "연결", "smooth", "blend", "soften"]),
    ("deepen", ["깊게", "파내", "deepen", "carve", "deeper"]),
    ("up", ["위쪽", "위로", "들어 올", "올려", "lift", "raise", "up"]),
    ("down", ["아래로", "내려", "lower", "down"]),
    ("forward", ["앞으로", "돌출", "forward", "project", "protrude"]),
    ("back", ["뒤로", "들어가", "back", "recede", "retract"]),
    ("narrow", ["좁게", "좁히", "narrow", "slimmer", "thinner nose"]),
    ("widen", ["넓게", "넓히", "widen", "wider"]),
    ("add_volume", ["볼륨", "추가", "두께", "두껍", "도톰", "fuller", "thicken", "add volume", "build", "증가"]),
    ("reduce", ["줄여", "얇게", "감소", "thinner", "reduce", "remove volume", "hollow"]),
]
AMOUNT_WORDS = [(0.4, ["약간", "조금", "살짝", "slightly", "a bit", "a little"]), (1.6, ["많이", "크게", "much", "a lot", "significantly"])]
DRAG_DIR = {"up": (0, 0, 1), "down": (0, 0, -1), "forward": (0, -1, 0), "back": (0, 1, 0)}


def parse_command(text):
    t = text.lower()
    region = None
    best = 0
    for r, words in REGION_WORDS.items():
        for w in words:
            if w in t and len(w) > best:
                region, best = r, len(w)
    if region is None and any(w in t for w in LIPS):
        region = "upper_lip"
    intent = next((a for a, ws in ACTION_WORDS if any(w in t for w in ws)), "add_volume")
    amount = next((a for a, ws in AMOUNT_WORDS if any(w in t for w in ws)), 1.0)
    side = "L" if any(w in t for w in ("왼", "left")) else ("R" if any(w in t for w in ("오른", "right")) else "both")
    both_lips = region == "upper_lip" and any(w in t for w in LIPS) and "윗" not in t and "upper" not in t
    return {"region": region, "intent": intent, "amount": amount, "side": side, "both_lips": both_lips, "text": text}


def _region_geometry(sem, region, side):
    st = sem.st
    w = sem.weights(region, side=side)
    idx = np.nonzero(w > 0.05)[0]
    if len(idx) == 0:
        raise ValueError(f"region {region} has no vertices yet (status={sem.status.get(region)})")
    P = st.co[idx]
    ww = w[idx]
    c = (P * ww[:, None]).sum(0) / ww.sum()
    n = (st.normals[idx] * ww[:, None]).sum(0)
    n /= np.linalg.norm(n)
    Q = P - c
    Q = Q - np.outer(Q @ n, n)
    _, s, vt = np.linalg.svd(Q * np.sqrt(ww)[:, None], full_matrices=False)
    ax_major, ax_minor = vt[0], vt[1]
    ext_major = float(np.percentile(np.abs(Q @ ax_major), 90))
    ext_minor = float(np.percentile(np.abs(Q @ ax_minor), 90))
    # nearest real vertex to the weighted centre (keeps the path on the surface)
    c_surf = st.co[idx[np.argmin(np.linalg.norm(P - c, axis=1))]]
    return {"idx": idx, "w": w, "center": c_surf, "normal": n, "major": ax_major, "minor": ax_minor,
            "ext_major": max(ext_major, st.mean_edge * 3), "ext_minor": max(ext_minor, st.mean_edge * 2)}


def _boundary_contour(sem, g):
    """Ordered boundary loop of a region (for sharpening edges like the lip line)."""
    st = sem.st
    inside = np.zeros(len(st.co), dtype=bool)
    inside[g["idx"][g["w"][g["idx"]] > 0.35]] = True
    from ..geometry.mesh_io import gather_neighbors
    cand = np.nonzero(inside)[0]
    owner, flat, _ = gather_neighbors(cand, st.ptr, st.nbr)
    out_n = np.bincount(owner, weights=(~inside[flat]).astype(float), minlength=len(cand))
    b = cand[out_n > 0]
    if len(b) < 4:
        return None
    Q = st.co[b] - g["center"]
    ang = np.arctan2(Q @ g["minor"], Q @ g["major"])
    order = np.argsort(ang)
    pts = st.co[b[order]]
    step = max(1, len(pts) // 24)
    return np.vstack([pts[::step], pts[:1]])


def plan_edit(sem, region, intent, amount=1.0, side="both", target="active_head"):
    """Return a list of plan items (dicts) for one semantic edit."""
    g = _region_geometry(sem, region, side)
    c, n, M, m = g["center"], g["normal"], g["major"], g["minor"]
    em, en = g["ext_major"], g["ext_minor"]
    R = float(np.clip(min(em, en) * 0.9, sem.st.mean_edge * 3, 0.25))
    sym = side == "both" and abs(c[0]) > 0.03 and region in {"cheekbone", "cheek", "upper_eyelid", "lower_eyelid", "nasal_ala",
                                                              "orbit", "mouth_corner", "temple", "ear", "jaw", "brow_ridge", "nostril"}
    if sym and c[0] < 0:  # always plan on the +X side, mirror for the other
        c = c * [-1, 1, 1]
    protect = sem.protect.get(region, [])
    base = {"operation": "sculpt", "target": target, "region": region, "coordinate_space": "OBJECT", "falloff": "SMOOTH",
            "symmetry": bool(sym), "protect_regions": protect, "verify_after": True, "geodesic": region in
            {"upper_lip", "lower_lip", "upper_eyelid", "lower_eyelid", "nostril", "mouth_corner"}}
    items = []
    line_major = np.array([c - M * em * 0.8, c, c + M * em * 0.8])
    if intent in DRAG_DIR or intent in ("narrow", "widen"):
        if intent in ("narrow", "widen"):
            sgn = -1 if intent == "narrow" else 1
            p = c + np.array([1, 0, 0]) * (abs(em * M[0]) + abs(en * m[0])) * 0.5 if region not in ("chin",) else c
            drag = np.array([sgn * 0.04 * amount * max(R / 0.08, 0.5), 0, 0])
            items.append({**base, "brush": "GRAB", "purpose": intent, "path": [p.tolist()], "drag": drag.tolist(),
                          "radius": R, "strength": 1.0, "direction": "DRAG", "symmetry": True})
        else:
            mag = 0.025 * amount * max(R / 0.06, 0.6)
            drag = np.array(DRAG_DIR[intent]) * mag
            items.append({**base, "brush": "GRAB", "purpose": f"move_{intent}", "path": [c.tolist()], "drag": drag.tolist(),
                          "radius": R * 1.2, "strength": 1.0, "direction": "DRAG"})
    elif intent == "add_volume":
        paths = [line_major + m * o for o in (-en * 0.35, en * 0.35)] if en > R * 0.8 else [line_major]
        for pth in paths:
            items.append({**base, "brush": "CLAY_STRIPS", "purpose": "increase_volume", "path": catmull_rom(pth, 4).tolist(),
                          "radius": R * 0.8, "strength": float(np.clip(0.35 * amount, 0.05, 1)), "direction": "ADD"})
        items.append({**base, "brush": "SURFACE_SMOOTH", "purpose": "blend", "path": line_major.tolist(), "radius": R * 1.3,
                      "strength": 0.25, "direction": "ADD"})
    elif intent in ("reduce", "deepen"):
        brush = "DRAW_SUBTRACT" if intent == "deepen" else "SCRAPE"
        items.append({**base, "brush": brush, "purpose": intent, "path": catmull_rom(line_major, 4).tolist(), "radius": R,
                      "strength": float(np.clip(0.4 * amount, 0.05, 1)), "direction": "SUBTRACT"})
        items.append({**base, "brush": "SURFACE_SMOOTH", "purpose": "blend", "path": line_major.tolist(), "radius": R * 1.2,
                      "strength": 0.2, "direction": "ADD"})
    elif intent == "sharpen":
        loop = _boundary_contour(sem, g)
        pth = loop if loop is not None else line_major
        items.append({**base, "brush": "PINCH", "purpose": "sharpen_border", "path": pth.tolist(), "radius": max(R * 0.35, sem.st.mean_edge * 3),
                      "strength": float(np.clip(0.4 * amount, 0.05, 1)), "direction": "ADD"})
        items.append({**base, "brush": "CREASE", "purpose": "define_edge", "path": pth.tolist(), "radius": max(R * 0.25, sem.st.mean_edge * 2.5),
                      "strength": float(np.clip(0.15 * amount, 0.05, 1)), "direction": "SUBTRACT"})
    elif intent == "smooth":
        fill = [c + M * a + m * b for b in np.linspace(-en, en, 3) for a in (np.linspace(-em, em, 4) if b == 0 else np.linspace(em, -em, 4))]
        items.append({**base, "brush": "SURFACE_SMOOTH", "purpose": "smooth_transition", "path": np.array(fill).tolist(),
                      "radius": R * 1.2, "strength": float(np.clip(0.45 * amount, 0.05, 1)), "direction": "ADD", "protect_regions": []})
    return items


def plan_from_command(sem, text, target="active_head"):
    cmd = parse_command(text)
    if cmd["region"] is None:
        raise ValueError(f"could not identify a region in: {text}")
    items = plan_edit(sem, cmd["region"], cmd["intent"], cmd["amount"], cmd["side"], target)
    if cmd["both_lips"]:
        items += plan_edit(sem, "lower_lip", cmd["intent"], cmd["amount"], cmd["side"], target)
    return {"command": cmd, "plan": items}


def execute_plan(S, items, max_leak=0.35, rollback=True):
    """Execute plan items on Sculptor S with protection + verification; returns per-item reports."""
    sem = S.sem
    reports = []
    for it in items:
        st = S.st
        region_w = sem.weights(it["region"])
        prot = np.zeros(len(st.co))
        for r in it.get("protect_regions", []):
            if r in sem.region_defs and r != it["region"]:
                prot = np.maximum(prot, sem.weights(r))
        prot = prot * (1 - region_w)
        S.eng.protect[:] = prot
        S.ckpt.snapshot(f"plan_{it['region']}_{it['brush']}")
        co0 = st.co.copy()
        path = np.asarray(it["path"], float)
        kw = dict(symmetry_x=it.get("symmetry", False), geodesic=it.get("geodesic", False), falloff=it.get("falloff"))
        if it.get("drag") is not None:
            path = np.array([path[0], path[0] + np.asarray(it["drag"], float)])
        if it.get("direction") == "SUBTRACT" and it["brush"] in ("DRAW", "CLAY", "CLAY_STRIPS", "INFLATE", "LAYER"):
            kw["direction"] = -1
        rep = S.stroke(it["brush"], path, it["radius"], it["strength"], region=it["region"], **kw)
        S.eng.protect[:] = 0
        d = np.linalg.norm(S.st.co - co0, axis=1)
        tot = d.sum()
        reg = (S.sem.weights(it["region"]) > 0.02) if len(d) == len(region_w) else np.zeros(len(d), bool)
        leak = float(d[~reg].sum() / tot) if tot > 0 else 0.0
        if it.get("symmetry"):
            leak = 0.0 if tot == 0 else float(d[~(reg | (S.sem.weights(it["region"], side="R") > 0.02))].sum() / tot)
        ok = tot > 0 and (leak <= max_leak or it["brush"] in ("SURFACE_SMOOTH", "SMOOTH"))
        if it.get("drag") is not None and tot > 0:
            dv = (S.st.co - co0)[reg].sum(0)
            ok = ok and float(dv @ np.asarray(it["drag"])) > 0
        verdict = {"region": it["region"], "brush": it["brush"], "purpose": it["purpose"], "moved": rep.get("moved_vertices"),
                   "max_displacement": rep.get("max_displacement"), "leak_fraction_outside_region": round(leak, 3),
                   "accepted": bool(ok)}
        if not ok and rollback:
            S.ckpt.undo(1)
            S.eng.resync()
            verdict["rolled_back"] = True
        else:
            S.ckpt.discard_last()
        reports.append(verdict)
        S.log.record({"op": "plan_item", **verdict})
    S.sem.rebuild(S.st)
    return reports
