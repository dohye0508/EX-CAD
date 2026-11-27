"""Portrait sculpt following the observed video workflow, with per-stage renders and a photo overlay.

blender -b --factory-startup --python scripts/portrait_video_workflow.py -- --out DIR --overrides overrides.json
        --photo photo.png --photo-landmarks lm.json [--until v3_nose_ridge]
"""
import os
import sys
import json
import time
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from bnm.core.state_manager import ActionLog, audit_scene  # noqa: E402
from bnm.core.orchestrator import Sculptor  # noqa: E402
from bnm.anatomy import face_canon as FC  # noqa: E402
from bnm.anatomy import video_workflow as VW  # noqa: E402
from bnm.sculpt.multiresolution import create_primitive  # noqa: E402
from bnm.perception import render as RN  # noqa: E402
from bnm.perception.reference_overlay import overlay  # noqa: E402
from bnm.validation.geometry_validator import validate_mesh  # noqa: E402


def model_front_landmarks(S, cam, buf, C):
    """Measured (not design) landmarks on the front render, where possible."""
    lm = {}
    for nm, c in zip(("eye_L_center", "eye_R_center"), getattr(S, "eye_centers", [])):
        u, v, _ = cam.world_to_pixel([c])[0]
        lm[nm] = (float(u), float(v))
    if not lm:
        W = C["widths"]
        for nm, x in (("eye_L_center", W["eye_center"]), ("eye_R_center", -W["eye_center"])):
            p = S.surface(x, C["levels"]["eye"])
            u, v, _ = cam.world_to_pixel([p])[0]
            lm[nm] = (float(u), float(v))
    depth, mask = buf["depth"], buf["mask"]
    H, Wd = mask.shape
    col = int(round(cam.world_to_pixel([[0, 0, 0]])[0][0]))
    mouth_v = int(cam.world_to_pixel([[0, -1, C["levels"]["stomion"]]])[0][1])
    d = depth[:, col]
    jumps = [(d[v + 1] - d[v], v) for v in range(mouth_v + 5, H - 1) if np.isfinite(d[v]) and np.isfinite(d[v + 1])]
    if jumps:
        lm["chin_bottom"] = (float(col), float(max(jumps)[1]))
    nose_v0 = int(cam.world_to_pixel([[0, -1, C["levels"]["nasion"]]])[0][1])
    sub = depth[nose_v0:mouth_v, col - 3:col + 4]
    if np.isfinite(sub).any():
        v_rel = np.unravel_index(np.nanargmin(np.where(np.isfinite(sub), sub, np.nan)), sub.shape)[0]
        lm["nose_tip"] = (float(col), float(nose_v0 + v_rel))
    row = mask[mouth_v]
    xs = np.nonzero(row)[0]
    if len(xs):
        lm["jaw_R"], lm["jaw_L"] = (float(xs.min()), float(mouth_v)), (float(xs.max()), float(mouth_v))
    for nm, x in (("mouth_L", C["widths"]["mouth_corner"]), ("mouth_R", -C["widths"]["mouth_corner"])):
        p = S.surface(x, C["levels"]["stomion"])
        if p is not None:
            u, v, _ = cam.world_to_pixel([p])[0]
            lm[nm] = (float(u), float(v))     # design position (flagged)
    return lm


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--overrides")
    ap.add_argument("--photo")
    ap.add_argument("--photo-landmarks")
    ap.add_argument("--until")
    ap.add_argument("--size", type=int, default=320)
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    log = ActionLog(os.path.join(out, "action_log.jsonl"), mode="REFERENCE_SCULPT")
    log.set_stage("v0_sphere")
    head = create_primitive("ICOSPHERE", "Portrait_Head", 1.0, detail=6, log=log)
    for f in head.data.polygons:
        f.use_smooth = True
    S = Sculptor(head, out_dir=out, log=log, render_size=a.size)
    C = FC.canon(json.load(open(a.overrides)) if a.overrides else None)
    photo_lm = json.load(open(a.photo_landmarks)) if a.photo_landmarks else None
    S.observe("v0_sphere", workbench=False)
    t0 = time.time()
    timings = {}

    def on_stage(name):
        timings[name] = round(time.time() - t0, 1)
        S.observe(name, workbench=False)

    res = VW.run(S, C, on_stage=on_stage, until=a.until)
    if not a.until:
        from bnm.anatomy import portrait_accessories as PA
        log.set_stage("v12_hair")
        PA.sculpt_hair(S)
        on_stage("v12_hair")
        log.set_stage("v13_glasses")
        PA.glasses(S, C)
        on_stage("v13_glasses")
    # overlay with the reference photo
    ov = None
    if a.photo and photo_lm:
        objs = [S.obj] + [o for o in S.extras if o.name in bpy.data.objects and not o.name.startswith(("Glasses", "BNM_Hair"))]
        cam = RN.head_cameras(S.obj, a.size, views=("front",))["front"]
        buf = RN.raycast_render(objs, cam)
        img = RN.shade(buf, "clay")
        mlm = model_front_landmarks(S, cam, buf, C)
        ov = overlay(a.photo, photo_lm, img, mlm, os.path.join(out, "photo_overlay.png"))
        ov["model_landmarks"] = mlm
    stages = ["v0_sphere"] + [s for s in VW.STAGES + ["v12_hair", "v13_glasses"] if s in S.stage_renders]
    tiles = [RN.load_image(S.stage_renders[s][f"{v}_clay"]) for s in stages for v in ("front", "three_quarter_left", "left")]
    RN.contact_sheet(tiles, 3, os.path.join(out, "progression_sheet.png"))
    rep = {"stages": res, "timings": timings, "loops": S.loops, "overlay": ov, "validation": validate_mesh(head),
           "audit": audit_scene(log), "brush_totals": log.brush_stats(), "stage_summary": log.stage_summary()}
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1, default=str)
    final_views = ("front", "three_quarter_left", "left", "three_quarter_right", "back", "top")
    objs = [S.obj] + [o for o in S.extras if o.name in bpy.data.objects]
    cams = RN.head_cameras(S.obj, 480, views=final_views)
    r = RN.render_views(objs, out, "final", final_views, ("clay",), 480, cameras=cams)
    RN.contact_sheet([RN.load_image(r[f"{v}_clay"]) for v in final_views], 3, os.path.join(out, "final_sheet.png"))
    for v in ("front", "three_quarter_left"):
        RN.workbench_render(objs, cams[v], os.path.join(out, f"final_{v}_blender.png"))
    # mouth close-ups
    from bnm.perception.camera import CameraModel
    mz = C["levels"]["stomion"]
    mcams = {v: CameraModel.for_view(v, (0, -0.85, mz + 0.08), distance=5, ortho_scale=0.75, width=360, height=360)
             for v in ("front", "three_quarter_left", "left")}
    face_objs = [o for o in objs if not o.name.startswith("Glasses")]
    rm = RN.render_views(face_objs, out, "mouth", tuple(mcams), ("clay",), 360, cameras=mcams)
    RN.contact_sheet([RN.load_image(rm[f"{v}_clay"]) for v in mcams], 3, os.path.join(out, "mouth_closeup.png"))
    S.ckpt.cleanup()
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "portrait_video_workflow.blend"))
    print("DONE", json.dumps({"timings": timings, "valid": rep["validation"]["problems"], "audit": rep["audit"]["passed"],
                              "overlay": ov and ov["landmark_errors_photo_minus_model"]}, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
