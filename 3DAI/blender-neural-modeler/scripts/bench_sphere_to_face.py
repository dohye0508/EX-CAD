"""Benchmark A - Sphere to Face (run inside Blender):
  blender -b --factory-startup --python scripts/bench_sphere_to_face.py -- --out benchmarks/A_sphere_to_face [--no-feedback]
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
from bnm.anatomy import head_sculpt_pipeline as HP  # noqa: E402
from bnm.sculpt.multiresolution import create_primitive  # noqa: E402
from bnm.validation.geometry_validator import validate_mesh  # noqa: E402
from bnm.perception.render import load_image, contact_sheet  # noqa: E402


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "A_sphere_to_face"))
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--until", default=None)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--no-workbench", action="store_true")
    ap.add_argument("--save-blend", action="store_true")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    t0 = time.time()
    log = ActionLog(os.path.join(out, "action_log.jsonl"))
    log.set_stage("stage0_initial")
    head = create_primitive("ICOSPHERE", "BNM_Head", 1.0, detail=6, log=log)
    for p in head.data.polygons:
        p.use_smooth = True
    S = Sculptor(head, out_dir=out, log=log, feedback=not a.no_feedback, render_size=a.size)
    C = FC.canon()
    timings = {}

    def on_stage(name):
        timings[name] = round(time.time() - t0, 1)
        S.observe(name, workbench=not a.no_workbench)

    res = HP.run(S, C, until=a.until, on_stage=on_stage)
    # final multiview
    final_views = ("front", "back", "left", "right", "three_quarter_left", "three_quarter_right", "top", "bottom")
    S.observe("final", views=final_views, modes=("clay", "matcap", "wireframe", "depth", "normal", "curvature", "silhouette"),
              workbench=not a.no_workbench)
    # contact sheets: stage progression (front + 3/4), final multiview
    rd = os.path.join(out, "renders")
    stages = [s for s in HP.STAGES if s in S.stage_renders]
    tiles = []
    for s in stages:
        for v in ("front", "three_quarter_left", "left"):
            p = S.stage_renders[s].get(f"{v}_clay")
            if p:
                tiles.append(load_image(p))
    if tiles:
        contact_sheet(tiles, 3, os.path.join(out, "progression_sheet.png"))
    ft = [load_image(S.stage_renders["final"][f"{v}_clay"]) for v in final_views]
    contact_sheet(ft, 4, os.path.join(out, "final_multiview_sheet.png"))
    modes = [load_image(S.stage_renders["final"][f"three_quarter_left_{m}"]) for m in ("clay", "matcap", "wireframe", "depth", "normal", "curvature")]
    contact_sheet(modes, 3, os.path.join(out, "final_render_modes_sheet.png"))
    report = {"benchmark": "A_sphere_to_face", "feedback": not a.no_feedback, "blender": bpy.app.version_string,
              "seconds_total": round(time.time() - t0, 1), "stage_end_times": timings, "stages": {}}
    summ = log.stage_summary()
    for s in stages:
        report["stages"][s] = {**summ.get(s, {}), "renders": S.stage_renders.get(s)}
    report["stage_results"] = json.loads(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    report["feedback_loops"] = S.loops
    report["brush_totals"] = log.brush_stats()
    report["validation"] = validate_mesh(head)
    report["audit"] = audit_scene(log)
    report["engine_stats"] = S.eng.stats
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    if a.save_blend:
        S.ckpt.cleanup()
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "sphere_to_face.blend"))
    print("BENCH_DONE", json.dumps({"seconds": report["seconds_total"], "metrics": res.get("stage9_review"),
                                    "valid": report["validation"]["ok"], "audit": report["audit"]["passed"]}, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
