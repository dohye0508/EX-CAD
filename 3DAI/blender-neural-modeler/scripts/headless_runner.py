"""Run tool commands inside a background Blender:
  blender -b --factory-startup --python scripts/headless_runner.py -- --commands cmds.json --result out.json
          [--blend-in in.blend] [--blend-out out.blend]
cmds.json = [{"tool": "create_initial_sphere", "args": {...}}, ...]
"""
import os
import sys
import json
import argparse
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--commands", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--blend-in")
    ap.add_argument("--blend-out")
    a = ap.parse_args(argv)
    if a.blend_in:
        bpy.ops.wm.open_mainfile(filepath=a.blend_in)
    else:
        for ob in list(bpy.data.objects):
            bpy.data.objects.remove(ob)
    from bnm.blender import tools as T
    cmds = json.load(open(a.commands, encoding="utf-8"))
    results = []
    for c in cmds:
        try:
            r = T.run_to_completion(T.call(c["tool"], c.get("args", {})))
            results.append({"tool": c["tool"], "ok": True, "result": T.jsonable(r)})
        except Exception as e:
            results.append({"tool": c["tool"], "ok": False, "error": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()[-1200:]})
            if c.get("stop_on_error", True):
                break
    if a.blend_out:
        from bnm.blender.session import Session
        for S in Session.get().sculptors.values():
            S.ckpt.cleanup()
        bpy.ops.wm.save_as_mainfile(filepath=a.blend_out)
    with open(a.result, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, default=str)


if __name__ == "__main__":
    main()
