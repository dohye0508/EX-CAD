"""Per-Blender-process session: sculpt targets, brush defaults, mode guard, user-edit detection."""
import os
import tempfile
import numpy as np
import bpy

from ..core.state_manager import ActionLog
from ..core.task_router import ModeGuard
from ..core.orchestrator import Sculptor
from ..geometry.mesh_io import get_coords


class Session:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = Session()
        return cls._inst

    def __init__(self):
        self.out_dir = os.path.join(tempfile.gettempdir(), "bnm_session")
        os.makedirs(self.out_dir, exist_ok=True)
        self.log = ActionLog(os.path.join(self.out_dir, "action_log.jsonl"))
        self.guard = ModeGuard("SCRATCH_SCULPT")
        self.sculptors = {}
        self.target = None
        self.brush = {"brush": "CLAY_STRIPS", "radius": 0.08, "strength": 0.4, "falloff": "SMOOTH"}
        self.user_edits = []
        self.busy = False          # True while the AI is changing the mesh
        self.paused = False
        self.overlay = {"path": None, "dab": None, "radius": 0.0, "region_points": None, "stage": ""}

    # ------------------------------------------------------------- targets
    def obj(self, name=None):
        name = name or self.target
        if name is None:
            ob = bpy.context.view_layer.objects.active
            if ob is None or ob.type != "MESH":
                raise ValueError("no target object; call create_initial_sphere or set_target")
            name = ob.name
        ob = bpy.data.objects.get(name)
        if ob is None or ob.type != "MESH":
            raise ValueError(f"mesh object {name!r} not found")
        return ob

    def sculptor(self, name=None, schema=None):
        ob = self.obj(name)
        S = self.sculptors.get(ob.name)
        if S is None or S.obj != ob or ob.name not in bpy.data.objects:
            S = Sculptor(ob, out_dir=self.out_dir, log=self.log)
            S.eng.on_dab = self._on_dab
            self.sculptors[ob.name] = S
        self.target = ob.name
        if schema == "face":
            from ..geometry.semantic import face_map
            S.sem = face_map(ob)
            S.sem.rebuild(S.st)
        elif schema == "creature":
            from ..anatomy.creature_sculptor import creature_map
            S.sem = creature_map(ob)
            S.sem.rebuild(S.st)
        self.sync_external(S)
        return S

    def _on_dab(self, center, radius):
        self.overlay["dab"] = tuple(center)
        self.overlay["radius"] = float(radius)

    # ------------------------------------------------------------- co-editing
    def flush_sculpt_mode(self, ob):
        """Sculpt-mode edits live in the PBVH until the mode is left; flush them to mesh data."""
        if ob.mode in ("SCULPT", "EDIT") and not bpy.app.background:
            from ..sculpt.native_brush import find_view3d
            ctx = find_view3d()
            if ctx:
                win, area, region, _ = ctx
                mode = ob.mode
                with bpy.context.temp_override(window=win, area=area, region=region, active_object=ob, object=ob):
                    bpy.ops.object.mode_set(mode="OBJECT")
                return mode
        return None

    def restore_mode(self, ob, mode):
        if mode and not bpy.app.background:
            from ..sculpt.native_brush import find_view3d
            ctx = find_view3d()
            if ctx:
                win, area, region, _ = ctx
                with bpy.context.temp_override(window=win, area=area, region=region, active_object=ob, object=ob):
                    bpy.ops.object.mode_set(mode=mode)

    def sync_external(self, S):
        """Detect user edits made since the AI last touched the mesh. Never overwrite them: re-read
        the mesh, record the changed area, and report it so plans can avoid/recompute."""
        ob = S.obj
        me = ob.data
        st = S.eng.st
        if (len(me.vertices), len(me.edges), len(me.polygons)) != st.key:
            self.user_edits.append({"type": "topology", "vertices": len(me.vertices)})
            S.topology_changed()
            return True
        co = get_coords(ob)
        d = np.linalg.norm(co - st.co, axis=1)
        moved = d > 1e-6
        if moved.any():
            P = co[moved]
            ev = {"type": "geometry", "vertices_changed": int(moved.sum()), "bbox": [P.min(0).round(4).tolist(), P.max(0).round(4).tolist()],
                  "max_displacement": float(d.max())}
            self.user_edits.append(ev)
            self.log.record({"op": "user_edit_detected", **ev})
            S.eng.resync()
            if S.sem is not None:
                S.sem.rebuild(S.st)
            return True
        return False

    def conflicts(self, region_points, margin=0.05):
        """Does a planned edit overlap the most recent user edit?"""
        if not self.user_edits or region_points is None:
            return False
        ev = self.user_edits[-1]
        if ev.get("type") != "geometry":
            return False
        lo, hi = np.array(ev["bbox"][0]) - margin, np.array(ev["bbox"][1]) + margin
        P = np.atleast_2d(region_points)
        return bool(((P >= lo) & (P <= hi)).all(1).any())
