"""Sculptor: the working context (mesh, engines, log, checkpoints, semantic map, observation) and
the closed feedback loop OBSERVE -> ANALYZE -> PLAN -> SCULPT -> VALIDATE -> REPLAN.

The loop is generic: a `measure` callable returns the current error vector (numbers measured on
the mesh), a `act` callable turns the error into strokes.  Every iteration is snapshot-protected:
if the error got worse the strokes are undone and the gain is halved.
"""
import os
import time
import numpy as np

from .state_manager import ActionLog
from .checkpoint_manager import CheckpointManager
from ..sculpt.virtual_brush import VirtualBrushEngine
from ..sculpt.native_brush import NativeBrushEngine
from ..geometry.mesh_io import MeshState
from ..geometry.semantic import surface_point, ray_hit


class Budget:
    def __init__(self, max_iterations=12, max_seconds=600.0):
        self.max_iterations = max_iterations
        self.max_seconds = max_seconds
        self.t0 = time.time()

    def expired(self):
        return time.time() - self.t0 > self.max_seconds


class Sculptor:
    def __init__(self, obj, out_dir=None, log=None, engine="virtual", feedback=True, render_size=256):
        self.obj = obj
        self.out_dir = out_dir
        self.log = log or ActionLog(os.path.join(out_dir, "action_log.jsonl") if out_dir else None)
        self.engine_pref = engine
        self.feedback = feedback
        self.render_size = render_size
        self.extras = []            # auxiliary objects that belong to the sculpture (eyeballs)
        self.eng = VirtualBrushEngine(obj, recorder=self.log)
        self.native = NativeBrushEngine(obj, recorder=self.log) if engine in ("native", "hybrid") else None
        self.ckpt = CheckpointManager(obj, os.path.join(out_dir, "checkpoints") if out_dir else None, log=self.log)
        self.sem = None
        self.loops = []
        self.stage_renders = {}

    # ---------------------------------------------------------- state
    @property
    def st(self):
        return self.eng.st

    def topology_changed(self):
        self.eng = VirtualBrushEngine(self.obj, recorder=self.log)
        if self.native:
            self.native.obj = self.obj
        if self.sem is not None:
            self.sem.obj = self.obj
            self.sem.rebuild(self.eng.st)

    def refresh_semantics(self):
        if self.sem is not None:
            self.sem.rebuild(self.eng.st)

    # ---------------------------------------------------------- sculpt
    def stroke(self, brush, path, radius, strength=0.5, region=None, engine=None, **kw):
        path = np.asarray(path, float)
        self.log.stage_region = region
        want = engine or self.engine_pref
        native_ok = ("GRAB", "DRAW", "CLAY", "CLAY_STRIPS", "INFLATE", "SMOOTH", "FLATTEN", "PINCH", "CREASE",
                     "SCRAPE", "DRAW_SUBTRACT", "ELASTIC_DEFORM", "SNAKE_HOOK", "SURFACE_SMOOTH", "LAYER", "BLOB",
                     "CLAY_THUMB", "DRAW_SHARP", "NUDGE", "THUMB", "FILL", "RELAX")
        if want in ("native", "hybrid") and self.native is None:
            self.native = NativeBrushEngine(self.obj, recorder=self.log)
        use_native = want in ("native", "hybrid") and self.native is not None and self.native.available             and brush.upper() in native_ok
        if want == "native" and not use_native:   # never fall back silently when the real brush was asked for
            raise RuntimeError(f"native brush unavailable for {brush} (UI viewport: "
                               f"{self.native.available if self.native else False})")
        if use_native:
            self.native.obj = self.obj
            nkw = {k: v for k, v in kw.items() if k in ("falloff", "spacing", "pressure", "direction", "symmetry_x")}
            if brush.upper() == "DRAW_SUBTRACT":
                brush, nkw["direction"] = "DRAW", -1
            rep = self.native.stroke(brush, path, radius, strength, **nkw)
            self.native.to_object_mode()
            self.eng.resync()
        else:
            rep = self.eng.stroke(brush, path, radius, strength, **kw)
            rep["engine"] = "virtual"
        if region:
            self.log.record({"op": "region_edit", "region": region, "brush": brush.upper(),
                             "moved": rep.get("moved_vertices")})
        return rep

    def surface(self, x, z, ray="front", y=None):
        hit = surface_point(self.st, x, z, ray, y)
        return None if hit is None else hit[0]

    def surface_normal(self, x, z, ray="front", y=None):
        hit = surface_point(self.st, x, z, ray, y)
        return None if hit is None else hit[1]

    def surface_path(self, xz, ray="front", lift=0.0):
        """Project (a, z) control points onto the surface -> object-space path (skip misses)."""
        out = []
        for a, z in xz:
            hit = surface_point(self.st, a, z, ray)
            if hit is not None:
                out.append(hit[0] + hit[1] * lift)
        return np.array(out)

    def front_y(self, x, z):
        p = self.surface(x, z, "front")
        return None if p is None else float(p[1])

    # ---------------------------------------------------------- feedback loop
    def closed_loop(self, name, measure, act, tol, max_iter=8, gain=1.0, min_gain=0.12):
        """measure() -> np.array error (target - current, signed);  act(error, gain) -> sculpts.

        Returns history.  With self.feedback == False runs exactly one open-loop act with gain 1.
        """
        hist = []
        e = np.atleast_1d(np.asarray(measure(), float))
        hist.append({"iter": 0, "error": e.round(5).tolist(), "abs": float(np.abs(e).max())})
        self.log.record({"op": "loop_observe", "loop": name, "iter": 0, "error": e.tolist()})
        if not self.feedback:
            act(e, 1.0)
            e2 = np.atleast_1d(np.asarray(measure(), float))
            hist.append({"iter": 1, "error": e2.round(5).tolist(), "abs": float(np.abs(e2).max()), "open_loop": True})
            self.loops.append({"name": name, "history": hist, "feedback": False})
            return hist
        g = gain
        for it in range(1, max_iter + 1):
            if np.abs(e).max() <= tol:
                break
            self.ckpt.snapshot(f"{name}_{it}")
            act(e, g)
            e_new = np.atleast_1d(np.asarray(measure(), float))
            score_old, score_new = float(np.linalg.norm(e)), float(np.linalg.norm(e_new))
            accepted = score_new < score_old
            if not accepted:
                self.ckpt.undo(1)
                self.eng.resync()
                g *= 0.5
                self.log.record({"op": "loop_reject", "loop": name, "iter": it, "gain": g})
                if g < min_gain:
                    break
            else:
                self.ckpt.discard_last()
                e = e_new
            hist.append({"iter": it, "error": e_new.round(5).tolist(), "abs": float(np.abs(e_new).max()),
                         "accepted": accepted, "gain": round(g, 3)})
            self.log.record({"op": "loop_validate", "loop": name, "iter": it, "error": e_new.tolist(), "accepted": accepted})
        self.loops.append({"name": name, "history": hist, "feedback": True})
        return hist

    # ---------------------------------------------------------- observe
    def observe(self, name, views=("front", "left", "three_quarter_left"), modes=("clay",), size=None,
                workbench=True):
        from ..perception.render import render_views, workbench_render, head_cameras
        if not self.out_dir:
            return {}
        d = os.path.join(self.out_dir, "renders")
        objs = [self.obj] + [o for o in self.extras if o.name in __import__("bpy").data.objects]
        size = size or self.render_size
        cams = head_cameras(self.obj, size, views=views)
        out = render_views(objs, d, name, views, modes, size, cameras=cams,
                           curvature_obj=self.obj if "curvature" in modes else None)
        if workbench:
            for v, cam in cams.items():
                p = os.path.join(d, f"{name}_{v}_blender.png")
                workbench_render(objs, cam, p, mode="clay")
                out[f"{v}_blender"] = p
        clean = {k: v for k, v in out.items() if not k.startswith("_buf")}
        self.stage_renders[name] = clean
        self.log.record({"op": "observe", "name": name, "files": clean})
        return out
