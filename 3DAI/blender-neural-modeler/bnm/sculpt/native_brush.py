"""Native Blender Sculpt Mode engine (Blender 5.2: brush assets + bpy.ops.sculpt.brush_stroke).

Verified on 5.2.1: brush_stroke(stroke=[{name, location, mouse, mouse_event, pressure, size, x_tilt,
y_tilt, time, is_start}], mode='NORMAL'|'INVERT') works only with a real window/VIEW_3D region
(poll() is False in background mode).  Brushes are activated from the ESSENTIALS asset library.

Like a human sculptor, the engine first orbits the viewport so it faces the surface it is about
to sculpt, converts the world radius to pixels, then drags the virtual pen along the path.
"""
import numpy as np
import bpy
from mathutils import Vector
from bpy_extras import view3d_utils

from . import falloff as FO
from .stroke import Stroke
from ..geometry.mesh_io import get_coords

ESSENTIALS = {
    "DRAW": "Draw", "DRAW_SHARP": "Draw Sharp", "DRAW_SUBTRACT": "Draw", "CLAY": "Clay", "CLAY_STRIPS": "Clay Strips",
    "CLAY_THUMB": "Clay Thumb", "INFLATE": "Inflate/Deflate", "BLOB": "Blob", "LAYER": "Layer",
    "CREASE": "Crease Sharp", "CREASE_POLISH": "Crease Polish", "SMOOTH": "Smooth", "SURFACE_SMOOTH": "Smooth",
    "POLISH": "Crease Polish", "RELAX": "Relax Slide", "FLATTEN": "Flatten/Contrast", "SCRAPE": "Scrape/Fill",
    "FILL": "Fill/Deepen", "MULTIPLANE_SCRAPE": "Scrape Multiplane", "PINCH": "Pinch/Magnify", "GRAB": "Grab",
    "ELASTIC_DEFORM": "Elastic Grab", "SNAKE_HOOK": "Snake Hook", "POSE": "Pose", "NUDGE": "Nudge",
    "THUMB": "Thumb", "BOUNDARY": "Boundary", "MASK": "Mask", "FACE_SETS": "Face Set Paint",
}
DRAG = {"GRAB", "ELASTIC_DEFORM", "SNAKE_HOOK", "POSE", "NUDGE", "THUMB", "BOUNDARY"}
LIB = "brushes/essentials_brushes-mesh_sculpt.blend/Brush/"


def find_view3d():
    wm = bpy.context.window_manager
    if bpy.app.background or not wm.windows:
        return None
    best = None
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                if best is None or area.width * area.height > best[1].width * best[1].height:
                    best = (win, area)
    if not best:
        return None
    win, area = best
    region = next(r for r in area.regions if r.type == "WINDOW")
    return win, area, region, area.spaces.active.region_3d


def _ups(ts):
    s = ts.sculpt
    return getattr(s, "unified_paint_settings", None) or getattr(ts, "unified_paint_settings", None)


class NativeBrushEngine:
    def __init__(self, obj, recorder=None):
        self.obj = obj
        self.recorder = recorder
        self.ctx = find_view3d()

    @property
    def available(self):
        return self.ctx is not None

    def override(self):
        win, area, region, _ = self.ctx
        return bpy.context.temp_override(window=win, area=area, region=region,
                                         active_object=self.obj, object=self.obj)

    def ensure_sculpt(self):
        with self.override():
            vl = bpy.context.view_layer
            vl.objects.active = self.obj
            self.obj.select_set(True)
            if self.obj.mode != "SCULPT":
                bpy.ops.object.mode_set(mode="SCULPT")

    def activate(self, brush):
        b = brush.upper()
        name = ESSENTIALS.get(b, brush)
        with self.override():
            r = bpy.ops.brush.asset_activate(asset_library_type="ESSENTIALS", relative_asset_identifier=LIB + name)
        br = bpy.context.tool_settings.sculpt.brush
        if b == "SURFACE_SMOOTH" and br is not None:
            br.smooth_deform_type = "SURFACE"
        elif b == "SMOOTH" and br is not None:
            br.smooth_deform_type = "LAPLACIAN"
        return r, br

    def look_at(self, center_w, normal_w, distance):
        _, area, region, rv3d = self.ctx
        rv3d.view_perspective = "ORTHO"
        rv3d.view_location = Vector(center_w)
        rv3d.view_rotation = Vector(normal_w).to_track_quat("Z", "Y")
        rv3d.view_distance = distance
        rv3d.update()
        area.tag_redraw()
        # sculpt ray casts use the view matrices of the last *drawn* frame: draw once now
        win = self.ctx[0]
        with bpy.context.temp_override(window=win, area=area, region=self.ctx[2]):
            bpy.ops.wm.redraw_timer(type="DRAW_WIN", iterations=1)

    def project(self, p_w):
        _, _, region, rv3d = self.ctx
        m = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector(p_w))
        return None if m is None else (m.x, m.y)

    def read_coords(self):
        """Sculpt-mode edits live in the PBVH; toggle to object mode to flush to mesh data."""
        with self.override():
            mode = self.obj.mode
            if mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            co = get_coords(self.obj)
            if mode != "OBJECT":
                bpy.ops.object.mode_set(mode=mode)
        return co

    def stroke(self, brush, path, radius, strength=0.5, falloff="SMOOTH", spacing=0.1, pressure="taper",
               direction=1, symmetry_x=False, view_normal=None, verify=True, auto_smooth=None):
        if not self.available:
            raise RuntimeError("native sculpt needs Blender with a window and a 3D Viewport (not -b)")
        b = brush.upper()
        mw = self.obj.matrix_world
        st = Stroke(path, radius, strength, spacing if b not in DRAG else 0.04,
                    pressure if b not in DRAG else "constant")
        pts_w = np.array([tuple(mw @ Vector(p)) for p in st.points])
        scale = float(np.mean(mw.to_scale()))
        r_w = radius * scale
        co_before = self.read_coords() if verify else None
        # surface normal at stroke centre (object space mean of nearby vertex normals)
        if view_normal is None:
            from ..geometry.mesh_io import MeshState
            ms = MeshState(self.obj)
            c = st.points[0] if b in DRAG else st.points.mean(0)
            d = np.linalg.norm(ms.co - c, axis=1)
            near = d < max(radius * 1.5, ms.mean_edge * 3)
            nrm = ms.normals[near].mean(0) if near.any() else (c - ms.co.mean(0))
            nrm_w = (mw.to_3x3() @ Vector(nrm)).normalized()
        else:
            nrm_w = Vector(view_normal).normalized()
        if b in DRAG and len(pts_w) > 1:
            # drag must lie in the view plane, otherwise the depth part of the mouse motion is lost:
            # look along the surface normal minus its component along the drag direction
            dvec = Vector(pts_w[-1] - pts_w[0])
            if dvec.length > 1e-6:
                dh = dvec.normalized()
                nv = Vector(nrm_w) - dh * Vector(nrm_w).dot(dh)
                if nv.length < 1e-3:
                    nv = dh.orthogonal()
                nrm_w = nv.normalized()
        extent = float(np.linalg.norm(pts_w.max(0) - pts_w.min(0)))
        self.ensure_sculpt()
        center_w = pts_w[0] if b in DRAG else pts_w.mean(0)
        self.look_at(center_w, nrm_w, max(extent * 1.8, r_w * 8, 0.1))
        rr, br = self.activate(b)
        ts = bpy.context.tool_settings
        br.strength = float(np.clip(strength, 0, 1))
        try:
            br.curve_distance_falloff_preset = FO.BLENDER_PRESET.get(str(falloff).upper(), "SMOOTH")
        except Exception:
            pass
        br.use_pressure_strength = True
        if auto_smooth is not None:
            br.auto_smooth_factor = auto_smooth
        self.obj.data.use_mirror_x = bool(symmetry_x)
        # world radius -> pixels at the stroke centre
        c2 = self.project(center_w)
        tvec = Vector(nrm_w).orthogonal().normalized()
        e2 = self.project(Vector(center_w) + tvec * r_w)
        px = max(2.0, float(np.hypot(e2[0] - c2[0], e2[1] - c2[1]))) if (c2 and e2) else 40.0
        # Blender 5.2 brush "size" is a DIAMETER in pixels (measured: size=px gave a footprint of ~r/2)
        br.size = int(round(2 * px))
        ups = _ups(ts)
        if ups is not None and getattr(ups, "use_unified_size", False):
            ups.size = int(round(2 * px))
        if ups is not None and getattr(ups, "use_unified_strength", False):
            ups.strength = br.strength
        items = []
        for k, p in enumerate(pts_w):
            m = self.project(p)
            if m is None:
                continue
            # sculpt stroke "location" is in OBJECT space (only the mouse position is screen space)
            items.append({"name": "", "location": tuple(st.points[k]), "mouse": m, "mouse_event": m,
                          "pressure": float(np.clip(st.pressure[k], 0, 1)), "size": 2.0 * px * float(st.radii[k] / radius),   # stroke size = diameter (measured)
                          "x_tilt": 0.0, "y_tilt": 0.0, "time": k * 0.016, "is_start": k == 0})
        with self.override():
            res = bpy.ops.sculpt.brush_stroke(stroke=items, mode="INVERT" if direction < 0 else "NORMAL")
        rep = {"engine": "native", "brush": b, "blender_brush": br.name, "operator_result": list(res),
               "dabs": len(items), "pixel_radius": px}
        if verify:
            co_after = self.read_coords()
            d = np.linalg.norm(co_after - co_before, axis=1)
            moved = d > 1e-7
            rep.update({"moved_vertices": int(moved.sum()), "max_displacement": float(d.max()),
                        "mesh_changed": bool(moved.any())})
            if moved.any():
                from ..geometry.mesh_io import MeshState
                ms = MeshState(self.obj)
                dv = co_after - co_before
                rep["mean_normal_displacement"] = float(np.einsum("ij,ij->i", dv[moved], ms.normals[moved]).mean())
                rep["change_centroid"] = ((co_before[moved] * d[moved, None]).sum(0) / d[moved].sum()).round(4).tolist()
        if self.recorder:
            self.recorder({"op": "native_stroke", "brush": b, **st.summary(), "report": rep})
        return rep

    def to_object_mode(self):
        if self.available:
            with self.override():
                if self.obj.mode != "OBJECT":
                    bpy.ops.object.mode_set(mode="OBJECT")
