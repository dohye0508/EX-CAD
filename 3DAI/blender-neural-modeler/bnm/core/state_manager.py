"""Append-only action log: every primitive creation, stroke, remesh, render and checkpoint.

It is the evidence for Benchmark I (no external mesh) and for per-stage brush statistics.
"""
import json
import os
import time

FORBIDDEN_OPS_SCRATCH = {"import_mesh", "import_scene", "append_external_mesh", "link_library",
                         "download_asset", "neural_generate", "tripo_generate"}


class ActionLog:
    def __init__(self, path=None, mode="SCRATCH_SCULPT"):
        self.path = path
        self.mode = mode
        self.events = []
        self.t0 = time.time()
        self.stage = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            open(path, "w", encoding="utf-8").close()

    def record(self, event):
        ev = dict(event)
        ev.setdefault("t", round(time.time() - self.t0, 3))
        ev.setdefault("stage", self.stage)
        ev.setdefault("mode", self.mode)
        if self.mode == "SCRATCH_SCULPT" and ev.get("op") in FORBIDDEN_OPS_SCRATCH:
            raise PermissionError(f"operation {ev['op']} is not allowed in SCRATCH_SCULPT mode")
        self.events.append(ev)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, default=_json_default, ensure_ascii=False) + "\n")
        return ev

    __call__ = record

    def set_stage(self, name):
        self.stage = name
        self.record({"op": "stage_begin", "name": name})

    def brush_stats(self, stage=None):
        out = {}
        for e in self.events:
            if stage and e.get("stage") != stage:
                continue
            if e.get("op") in ("virtual_stroke", "native_stroke", "virtual_pose"):
                b = e.get("brush", "POSE")
                out[b] = out.get(b, 0) + 1
        return out

    def stage_summary(self):
        stages = {}
        for e in self.events:
            s = e.get("stage")
            if s is None:
                continue
            d = stages.setdefault(s, {"strokes": 0, "brushes": {}, "regions": set(), "t_start": e["t"], "t_end": e["t"]})
            d["t_end"] = e["t"]
            if e.get("op") in ("virtual_stroke", "native_stroke", "virtual_pose"):
                d["strokes"] += 1
                b = e.get("brush", "POSE")
                d["brushes"][b] = d["brushes"].get(b, 0) + 1
            if e.get("region"):
                d["regions"].add(e["region"])
        for d in stages.values():
            d["regions"] = sorted(d["regions"])
            d["seconds"] = round(d["t_end"] - d["t_start"], 2)
        return stages


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
    except Exception:
        pass
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def audit_scene(log=None):
    """Benchmark I: prove no externally produced mesh is in the scene."""
    import bpy
    report = {"mesh_objects": [], "violations": []}
    for lib in bpy.data.libraries:
        report["violations"].append(f"linked library present: {lib.filepath}")
    for ob in bpy.data.objects:
        if ob.type != "MESH":
            continue
        tag = ob.get("bnm_origin")
        entry = {"name": ob.name, "origin": tag, "vertices": len(ob.data.vertices),
                 "mesh_library": ob.data.library.filepath if ob.data.library else None}
        report["mesh_objects"].append(entry)
        if ob.data.library is not None:
            report["violations"].append(f"{ob.name}: mesh data comes from a library")
        if not tag:
            report["violations"].append(f"{ob.name}: no bnm provenance tag (not created by the engine)")
        elif not str(tag).startswith(("primitive:", "derived:")):
            report["violations"].append(f"{ob.name}: provenance {tag} is not a primitive/derived mesh")
    if log is not None:
        bad = [e for e in log.events if e.get("op") in FORBIDDEN_OPS_SCRATCH]
        report["forbidden_ops_in_log"] = len(bad)
        prim = [e for e in log.events if e.get("op") == "create_primitive"]
        report["primitives_created"] = [{"name": e.get("name"), "kind": e.get("kind"), "vertices": e.get("vertices")} for e in prim]
        report["strokes_logged"] = sum(1 for e in log.events if e.get("op") in ("virtual_stroke", "native_stroke", "virtual_pose"))
        if bad:
            report["violations"].append("forbidden operations found in log")
    report["passed"] = not report["violations"]
    return report
