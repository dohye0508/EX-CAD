# SPDX-FileCopyrightText: 2026 EX-CAD Authors
# SPDX-License-Identifier: GPL-2.0-or-later

import json
import math
import queue
import socket
import sys
import threading
from dataclasses import dataclass, field

import bpy
import bmesh
import numpy as np
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import Operator, Panel
from mathutils import Matrix, Vector


# ---------------------------------------------------------------------------
# 1. Fusion 360-Style 2D Parametric Sketch + NumPy Gauss-Newton Solver
# ---------------------------------------------------------------------------

@dataclass
class SketchPoint:
    pid: int
    u: float
    v: float
    fixed: bool = False


@dataclass
class SketchEntity:
    eid: int
    etype: str  # 'LINE' | 'CIRCLE'
    p0: int
    p1: int = -1
    radius: float = 0.5


@dataclass
class SketchConstraint:
    cid: int
    ctype: str
    a: int
    b: int = -1
    c: int = -1
    d: int = -1
    value: float = 0.0
    value2: float = 0.0


class SketchDocument:
    _active = None

    def __init__(self, plane: str = "XY"):
        self.plane = plane
        self.points: list[SketchPoint] = []
        self.entities: list[SketchEntity] = []
        self.constraints: list[SketchConstraint] = []

    @classmethod
    def active(cls):
        if cls._active is None:
            cls._active = cls("XY")
        return cls._active

    def reset(self, plane: str = "XY"):
        self.plane = plane
        self.points.clear()
        self.entities.clear()
        self.constraints.clear()

    def add_point(self, u: float, v: float, fixed: bool = False) -> int:
        pid = len(self.points)
        self.points.append(SketchPoint(pid, float(u), float(v), fixed))
        return pid

    def add_line_pts(self, p0: int, p1: int) -> int:
        eid = len(self.entities)
        self.entities.append(SketchEntity(eid, "LINE", p0, p1))
        return eid

    def add_line(self, u0: float, v0: float, u1: float, v1: float) -> int:
        p0 = self.add_point(u0, v0)
        p1 = self.add_point(u1, v1)
        return self.add_line_pts(p0, p1)

    def add_rect(self, u0: float, v0: float, width: float, height: float):
        p0 = self.add_point(u0, v0)
        p1 = self.add_point(u0 + width, v0)
        p2 = self.add_point(u0 + width, v0 + height)
        p3 = self.add_point(u0, v0 + height)
        self.add_line_pts(p0, p1)
        self.add_line_pts(p1, p2)
        self.add_line_pts(p2, p3)
        self.add_line_pts(p3, p0)
        self.add_constraint("HORIZONTAL", p0, p1)
        self.add_constraint("HORIZONTAL", p3, p2)
        self.add_constraint("VERTICAL", p1, p2)
        self.add_constraint("VERTICAL", p0, p3)
        self.add_constraint("DISTANCE", p0, p1, value=abs(width))
        self.add_constraint("DISTANCE", p1, p2, value=abs(height))

    def add_circle(self, cu: float, cv: float, radius: float) -> int:
        center = self.add_point(cu, cv)
        eid = len(self.entities)
        self.entities.append(SketchEntity(eid, "CIRCLE", center, -1, abs(radius)))
        self.add_constraint("RADIUS", eid, value=abs(radius))
        return eid

    def add_constraint(self, ctype: str, a: int, b: int = -1, c: int = -1, d: int = -1,
                       value: float = 0.0, value2: float = 0.0) -> int:
        cid = len(self.constraints)
        self.constraints.append(SketchConstraint(cid, ctype, a, b, c, d, float(value), float(value2)))
        return cid

    def _eval_residuals(self, state: np.ndarray) -> np.ndarray:
        n_pts = len(self.points)
        res = []
        for con in self.constraints:
            t = con.ctype
            if t == "COINCIDENT":
                res.append(state[2 * con.a] - state[2 * con.b])
                res.append(state[2 * con.a + 1] - state[2 * con.b + 1])
            elif t == "HORIZONTAL":
                res.append(state[2 * con.a + 1] - state[2 * con.b + 1])
            elif t == "VERTICAL":
                res.append(state[2 * con.a] - state[2 * con.b])
            elif t == "DISTANCE":
                du = state[2 * con.b] - state[2 * con.a]
                dv = state[2 * con.b + 1] - state[2 * con.a + 1]
                res.append(math.sqrt(du * du + dv * dv + 1e-12) - con.value)
            elif t == "RADIUS":
                res.append(state[2 * n_pts + con.a] - con.value)
            elif t == "PARALLEL":
                dx1 = state[2 * con.b] - state[2 * con.a]
                dy1 = state[2 * con.b + 1] - state[2 * con.a + 1]
                dx2 = state[2 * con.d] - state[2 * con.c]
                dy2 = state[2 * con.d + 1] - state[2 * con.c + 1]
                res.append(dx1 * dy2 - dy1 * dx2)
            elif t == "PERPENDICULAR":
                dx1 = state[2 * con.b] - state[2 * con.a]
                dy1 = state[2 * con.b + 1] - state[2 * con.a + 1]
                dx2 = state[2 * con.d] - state[2 * con.c]
                dy2 = state[2 * con.d + 1] - state[2 * con.c + 1]
                res.append(dx1 * dx2 + dy1 * dy2)
            elif t == "EQUAL_LENGTH":
                d1 = math.hypot(state[2 * con.b] - state[2 * con.a], state[2 * con.b + 1] - state[2 * con.a + 1])
                d2 = math.hypot(state[2 * con.d] - state[2 * con.c], state[2 * con.d + 1] - state[2 * con.c + 1])
                res.append(d1 - d2)
            elif t == "FIXED":
                res.append(state[2 * con.a] - con.value)
                res.append(state[2 * con.a + 1] - con.value2)
        return np.array(res, dtype=np.float64)

    def solve(self, max_iters: int = 80, tol: float = 1e-7) -> tuple[bool, int, float]:
        n_pts = len(self.points)
        n_ent = len(self.entities)
        n_vars = 2 * n_pts + n_ent
        if n_vars == 0 or not self.constraints:
            return True, 0, 0.0

        state = np.zeros(n_vars, dtype=np.float64)
        fixed = np.zeros(n_vars, dtype=bool)
        for i, p in enumerate(self.points):
            state[2 * i] = p.u
            state[2 * i + 1] = p.v
            if p.fixed:
                fixed[2 * i] = True
                fixed[2 * i + 1] = True
        for i, e in enumerate(self.entities):
            state[2 * n_pts + i] = e.radius
            if e.etype != "CIRCLE":
                fixed[2 * n_pts + i] = True

        eps = 1e-6
        converged = False
        iters = 0
        max_res = 0.0

        for it in range(max_iters):
            r0 = self._eval_residuals(state)
            iters = it
            if len(r0) == 0:
                converged = True
                break
            max_res = float(np.max(np.abs(r0)))
            if max_res < tol:
                converged = True
                break

            J = np.zeros((len(r0), n_vars), dtype=np.float64)
            for col in range(n_vars):
                if fixed[col]:
                    continue
                orig = state[col]
                state[col] = orig + eps
                r1 = self._eval_residuals(state)
                state[col] = orig
                J[:, col] = (r1 - r0) / eps

            A = J.T @ J + 1e-6 * np.eye(n_vars)
            b = -J.T @ r0
            dx = np.linalg.lstsq(A, b, rcond=None)[0]
            dx[fixed] = 0.0
            state += np.clip(dx, -1.0, 1.0)

        for i, p in enumerate(self.points):
            p.u = float(state[2 * i])
            p.v = float(state[2 * i + 1])
        for i, e in enumerate(self.entities):
            if e.etype == "CIRCLE":
                e.radius = max(1e-4, float(state[2 * n_pts + i]))
        return converged, iters, max_res

    def plane_to_world(self, u: float, v: float, depth: float) -> tuple[float, float, float]:
        if self.plane == "XZ":
            return (float(u), float(depth), float(v))
        if self.plane == "YZ":
            return (float(depth), float(u), float(v))
        return (float(u), float(v), float(depth))

    def build_2d_loops(self, circle_segments: int = 48):
        loops = []
        edges = []
        for e in self.entities:
            if e.etype == "CIRCLE":
                c = self.points[e.p0]
                loop = []
                for i in range(circle_segments):
                    ang = (2.0 * math.pi * i) / circle_segments
                    loop.append((c.u + e.radius * math.cos(ang), c.v + e.radius * math.sin(ang)))
                loops.append(loop)
            elif e.etype == "LINE":
                edges.append((e.p0, e.p1))

        used = [False] * len(edges)
        for i in range(len(edges)):
            if used[i]:
                continue
            used[i] = True
            chain = [edges[i][0], edges[i][1]]
            extended = True
            while extended:
                extended = False
                tail = chain[-1]
                if tail == chain[0] and len(chain) >= 4:
                    break
                for j in range(len(edges)):
                    if used[j]:
                        continue
                    if edges[j][0] == tail:
                        used[j] = True
                        chain.append(edges[j][1])
                        extended = True
                        break
                    if edges[j][1] == tail:
                        used[j] = True
                        chain.append(edges[j][0])
                        extended = True
                        break
            if len(chain) >= 4 and chain[0] == chain[-1]:
                chain.pop()
                loop = [(self.points[pid].u, self.points[pid].v) for pid in chain]
                area2 = sum(loop[k][0] * loop[(k + 1) % len(loop)][1] - loop[(k + 1) % len(loop)][0] * loop[k][1]
                            for k in range(len(loop)))
                if area2 < 0.0:
                    loop.reverse()
                loops.append(loop)
        return loops


def create_mesh_object(name: str, verts: list, faces: list):
    bm = bmesh.new()
    bm_verts = [bm.verts.new(v) for v in verts]
    for f in faces:
        if len(f) >= 3:
            try:
                bm.faces.new([bm_verts[idx] for idx in f])
            except ValueError:
                pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def extrude_sketch_doc(doc: SketchDocument, depth: float, symmetric: bool, name: str = "EXCAD_Extrude"):
    loops = doc.build_2d_loops(48)
    if not loops:
        return None
    d0 = -0.5 * depth if symmetric else 0.0
    d1 = 0.5 * depth if symmetric else depth
    z_min, z_max = min(d0, d1), max(d0, d1)

    verts, faces = [], []
    for loop in loops:
        n = len(loop)
        base = len(verts)
        for u, v in loop:
            verts.append(doc.plane_to_world(u, v, z_min))
        for u, v in loop:
            verts.append(doc.plane_to_world(u, v, z_max))
        faces.append([base + (n - 1 - i) for i in range(n)])
        faces.append([base + n + i for i in range(n)])
        for i in range(n):
            nxt = (i + 1) % n
            faces.append([base + i, base + nxt, base + n + nxt, base + n + i])
    return create_mesh_object(name, verts, faces)


def revolve_sketch_doc(doc: SketchDocument, angle_deg: float, segments: int, name: str = "EXCAD_Revolve"):
    loops = doc.build_2d_loops(max(8, segments))
    if not loops:
        return None
    segs = max(3, segments)
    clamped = max(1.0, min(360.0, angle_deg))
    full = clamped >= 359.99
    total_rad = math.radians(360.0 if full else clamped)
    rings = segs if full else (segs + 1)

    verts, faces = [], []
    for loop in loops:
        n = len(loop)
        base = len(verts)
        for r in range(rings):
            theta = (2.0 * math.pi * r / segs) if full else (total_rad * r / segs)
            ct, st = math.cos(theta), math.sin(theta)
            for radius, v in loop:
                verts.append(doc.plane_to_world(radius * ct, v, radius * st))
        for r in range(segs):
            rn = (r + 1) % rings
            for i in range(n):
                inext = (i + 1) % n
                faces.append([
                    base + r * n + i,
                    base + r * n + inext,
                    base + rn * n + inext,
                    base + rn * n + i,
                ])
        if not full:
            faces.append([base + (n - 1 - i) for i in range(n)])
            faces.append([base + (rings - 1) * n + i for i in range(n)])
    return create_mesh_object(name, verts, faces)


# ---------------------------------------------------------------------------
# 2. High-Precision Anatomical Humanoid & Custom SDF Mesh Generator
# ---------------------------------------------------------------------------

def _add_ellipsoid(bm, center, radii, u_seg=20, v_seg=12):
    ret = bmesh.ops.create_uvsphere(bm, u_segments=u_seg, v_segments=v_seg, radius=1.0)
    mat = Matrix.Translation(Vector(center)) @ Matrix.Diagonal((radii[0], radii[1], radii[2], 1.0))
    bmesh.ops.transform(bm, matrix=mat, verts=ret["verts"])


def _add_capsule(bm, a, b, ra, rb, segs=18, rings=8):
    va = Vector(a)
    vb = Vector(b)
    axis = vb - va
    length = axis.length
    if length < 1e-5:
        _add_ellipsoid(bm, a, (ra, ra, ra))
        return
    z_dir = axis.normalized()
    quat = z_dir.to_track_quat('Z', 'Y')
    rot_mat = quat.to_matrix().to_4x4()

    ring_verts = []
    for r in range(rings + 1):
        t = r / rings
        rad = ra + (rb - ra) * t
        z = t * length
        ring = []
        for s in range(segs):
            ang = 2.0 * math.pi * s / segs
            local = Vector((rad * math.cos(ang), rad * math.sin(ang), z))
            ring.append(bm.verts.new(va + (rot_mat @ local)))
        ring_verts.append(ring)

    for r in range(rings):
        for s in range(segs):
            sn = (s + 1) % segs
            bm.faces.new((ring_verts[r][s], ring_verts[r][sn], ring_verts[r + 1][sn], ring_verts[r + 1][s]))

    # End caps
    _add_ellipsoid(bm, a, (ra, ra, ra), u_seg=14, v_seg=8)
    _add_ellipsoid(bm, b, (rb, rb, rb), u_seg=14, v_seg=8)


def _remesh_smooth(obj, voxel_size: float):
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    obj.data.remesh_voxel_size = max(0.008, voxel_size)
    bpy.ops.object.voxel_remesh()

    mod = obj.modifiers.new(name="EXCAD_Smooth", type='SMOOTH')
    mod.factor = 0.5
    mod.iterations = 4
    bpy.ops.object.modifier_apply(modifier=mod.name)

    for poly in obj.data.polygons:
        poly.use_smooth = True


def generate_humanoid_mesh(height=1.75, muscularity=0.5, shoulder_ratio=1.0, hip_ratio=1.0,
                           resolution=72, name="EXCAD_Humanoid"):
    scale = max(0.5, min(3.0, height)) / 1.75
    musc = max(0.0, min(1.5, muscularity))
    sh = max(0.7, min(1.4, shoulder_ratio))
    hp = max(0.7, min(1.4, hip_ratio))

    bm = bmesh.new()

    # Head, jaw, neck
    _add_ellipsoid(bm, (0.0, 0.01, 1.63), (0.082, 0.098, 0.112))
    _add_ellipsoid(bm, (0.0, -0.028, 1.565), (0.058, 0.068, 0.072))
    _add_capsule(bm, (0.0, 0.0, 1.44), (0.0, 0.008, 1.54), 0.056, 0.048)

    # Torso: ribcage, abdomen, pelvis
    _add_ellipsoid(bm, (0.0, 0.005, 1.29), (0.145 * sh, 0.105 + 0.015 * musc, 0.165))
    _add_ellipsoid(bm, (0.0, -0.005, 1.09), (0.118 + 0.01 * musc, 0.092, 0.125))
    _add_ellipsoid(bm, (0.0, 0.0, 0.93), (0.142 * hp, 0.102, 0.095))

    # Symmetric bilateral anatomy
    for sx in (-1.0, 1.0):
        # Trapezius / clavicle
        _add_capsule(bm, (sx * 0.035, 0.01, 1.47), (sx * 0.175 * sh, 0.01, 1.41), 0.048, 0.038)
        # Pectorals & glutes
        _add_ellipsoid(bm, (sx * 0.072 * sh, -0.078, 1.32), (0.075, 0.045 + 0.018 * musc, 0.068))
        _add_ellipsoid(bm, (sx * 0.068 * hp, 0.055, 0.90), (0.072, 0.068, 0.085))

        # Arms
        shoulder = (sx * 0.195 * sh, 0.01, 1.40)
        elbow = (sx * 0.275 * sh, 0.025, 1.13)
        wrist = (sx * 0.335 * sh, -0.015, 0.87)
        hand_end = (sx * 0.355 * sh, -0.025, 0.77)
        _add_ellipsoid(bm, shoulder, (0.056 + 0.014 * musc,) * 3)
        _add_capsule(bm, shoulder, elbow, 0.048 + 0.012 * musc, 0.040 + 0.008 * musc)
        _add_capsule(bm, elbow, wrist, 0.041 + 0.008 * musc, 0.029)
        _add_capsule(bm, wrist, hand_end, 0.030, 0.022)

        # Legs & feet
        hip = (sx * 0.092 * hp, 0.0, 0.89)
        knee = (sx * 0.095, -0.008, 0.50)
        ankle = (sx * 0.092, 0.015, 0.09)
        heel = (sx * 0.092, 0.040, 0.035)
        toe = (sx * 0.092, -0.105, 0.025)
        _add_capsule(bm, hip, knee, 0.078 + 0.015 * musc, 0.054 + 0.006 * musc)
        _add_capsule(bm, knee, ankle, 0.054 + 0.008 * musc, 0.036)
        _add_capsule(bm, ankle, heel, 0.036, 0.034)
        _add_capsule(bm, heel, toe, 0.036, 0.026)

    bmesh.ops.transform(bm, matrix=Matrix.Diagonal((scale, scale, scale, 1.0)), verts=bm.verts)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    voxel_size = (1.85 * scale) / max(24, min(128, resolution))
    _remesh_smooth(obj, voxel_size)
    return obj


def generate_custom_sdf_mesh(spec: str, resolution: int = 64, name: str = "EXCAD_SDF"):
    bm = bmesh.new()
    for line in spec.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cmd = parts[0].lower()
        nums = [float(x) for x in parts[1:]]
        if cmd == "sphere" and len(nums) >= 4:
            _add_ellipsoid(bm, nums[0:3], (nums[3], nums[3], nums[3]))
        elif cmd == "capsule" and len(nums) >= 8:
            _add_capsule(bm, nums[0:3], nums[3:6], nums[6], nums[7])
        elif cmd == "ellipsoid" and len(nums) >= 6:
            _add_ellipsoid(bm, nums[0:3], nums[3:6])
        elif cmd == "box" and len(nums) >= 6:
            ret = bmesh.ops.create_cube(bm, size=2.0)
            mat = Matrix.Translation(Vector(nums[0:3])) @ Matrix.Diagonal((nums[3], nums[4], nums[5], 1.0))
            bmesh.ops.transform(bm, matrix=mat, verts=ret["verts"])

    if len(bm.verts) == 0:
        _add_ellipsoid(bm, (0.0, 0.0, 0.5), (0.5, 0.5, 0.5))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    _remesh_smooth(obj, 1.5 / max(24, min(128, resolution)))
    return obj


# ---------------------------------------------------------------------------
# 3. TCP JSON-RPC AI Direct Bridge (127.0.0.1:9877)
# ---------------------------------------------------------------------------

class AiBridgeServer:
    _instance = None

    def __init__(self):
        self.running = False
        self.port = 9877
        self._thread = None
        self._queue: queue.Queue = queue.Queue()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self, port: int = 9877):
        if self.running:
            return True
        self.port = port
        self.running = True
        self._thread = threading.Thread(target=self._server_loop, daemon=True)
        self._thread.start()
        if not bpy.app.timers.is_registered(self._timer_poll):
            bpy.app.timers.register(self._timer_poll, first_interval=0.05)
        return True

    def stop(self):
        self.running = False

    def _timer_poll(self):
        while not self._queue.empty():
            payload, holder, evt = self._queue.get_nowait()
            try:
                holder["resp"] = self._dispatch(payload)
            except Exception as exc:
                holder["resp"] = json.dumps({"ok": False, "error": str(exc)})
            finally:
                evt.set()
        return 0.05 if self.running else None

    def _dispatch(self, payload: str) -> str:
        data = json.loads(payload)
        cmd = data.get("cmd", "")
        doc = SketchDocument.active()

        if cmd == "ping":
            return json.dumps({"ok": True, "service": "EX-CAD", "port": self.port})
        if cmd == "sketch_new":
            plane = data.get("plane", "XY")
            doc.reset(plane)
            return json.dumps({"ok": True, "plane": plane})
        if cmd == "sketch_add_rect":
            doc.add_rect(data.get("u0", -0.5), data.get("v0", -0.5),
                         data.get("width", 1.0), data.get("height", 1.0))
            return json.dumps({"ok": True, "points": len(doc.points)})
        if cmd == "sketch_add_circle":
            doc.add_circle(data.get("cu", 0.0), data.get("cv", 0.0), data.get("radius", 0.5))
            return json.dumps({"ok": True, "entities": len(doc.entities)})
        if cmd == "sketch_add_line":
            doc.add_line(data.get("u0", 0.0), data.get("v0", 0.0),
                         data.get("u1", 1.0), data.get("v1", 0.0))
            return json.dumps({"ok": True, "entities": len(doc.entities)})
        if cmd == "sketch_solve":
            conv, iters, res = doc.solve()
            return json.dumps({"ok": True, "converged": conv, "iterations": iters, "max_residual": res})
        if cmd == "sketch_extrude":
            doc.solve()
            obj = extrude_sketch_doc(doc, float(data.get("depth", 1.0)),
                                     bool(data.get("symmetric", False)),
                                     data.get("name", "EXCAD_Extrude"))
            if obj is None:
                return json.dumps({"ok": False, "error": "No closed sketch profile to extrude"})
            return json.dumps({"ok": True, "verts": len(obj.data.vertices), "faces": len(obj.data.polygons)})
        if cmd == "sketch_revolve":
            doc.solve()
            obj = revolve_sketch_doc(doc, float(data.get("angle_deg", 360.0)),
                                     int(data.get("segments", 48)),
                                     data.get("name", "EXCAD_Revolve"))
            if obj is None:
                return json.dumps({"ok": False, "error": "No closed sketch profile to revolve"})
            return json.dumps({"ok": True, "verts": len(obj.data.vertices), "faces": len(obj.data.polygons)})
        if cmd == "sdf_humanoid":
            obj = generate_humanoid_mesh(
                float(data.get("height", 1.75)),
                float(data.get("muscularity", 0.5)),
                float(data.get("shoulder_ratio", 1.0)),
                float(data.get("hip_ratio", 1.0)),
                int(data.get("resolution", 72)),
                data.get("name", "EXCAD_Humanoid"),
            )
            return json.dumps({"ok": True, "verts": len(obj.data.vertices), "faces": len(obj.data.polygons)})
        if cmd == "sdf_custom":
            obj = generate_custom_sdf_mesh(
                data.get("spec", ""),
                int(data.get("resolution", 64)),
                data.get("name", "EXCAD_SDF"),
            )
            return json.dumps({"ok": True, "verts": len(obj.data.vertices), "faces": len(obj.data.polygons)})
        return json.dumps({"ok": False, "error": f"Unknown command: {cmd}"})

    def _server_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.2)
        try:
            sock.bind(("127.0.0.1", self.port))
            sock.listen(8)
        except OSError:
            self.running = False
            sock.close()
            return

        while self.running:
            try:
                client, _ = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with client:
                chunks = []
                while True:
                    chunk = client.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if b"\n" in chunk:
                        break
                payload = b"".join(chunks).decode("utf-8", errors="replace").strip()
                if payload:
                    holder = {}
                    evt = threading.Event()
                    self._queue.put((payload, holder, evt))
                    if evt.wait(timeout=15.0):
                        reply = holder.get("resp", '{"ok":false,"error":"Empty"}')
                    else:
                        reply = '{"ok":false,"error":"Timeout"}'
                    client.sendall((reply + "\n").encode("utf-8"))
        sock.close()


# ---------------------------------------------------------------------------
# 4. Blender Operators & 3D Viewport UI Panels
# ---------------------------------------------------------------------------

class EXCAD_OT_sketch_new(Operator):
    bl_idname = "excad.sketch_new"
    bl_label = "New Parametric Sketch"
    bl_options = {'REGISTER', 'UNDO'}

    plane: EnumProperty(
        name="Plane",
        items=[('XY', "XY", ""), ('XZ', "XZ", ""), ('YZ', "YZ", "")],
        default='XY',
    )

    def execute(self, context):
        SketchDocument.active().reset(self.plane)
        self.report({'INFO'}, f"EX-CAD: Started new {self.plane} sketch")
        return {'FINISHED'}


class EXCAD_OT_sketch_add_rect(Operator):
    bl_idname = "excad.sketch_add_rect"
    bl_label = "Sketch Rectangle"
    bl_options = {'REGISTER', 'UNDO'}

    u0: FloatProperty(name="U Origin", default=-0.5)
    v0: FloatProperty(name="V Origin", default=-0.5)
    width: FloatProperty(name="Width", default=1.0, min=0.001)
    height: FloatProperty(name="Height", default=1.0, min=0.001)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        SketchDocument.active().add_rect(self.u0, self.v0, self.width, self.height)
        self.report({'INFO'}, f"EX-CAD: Added rectangle ({self.width:.3f} x {self.height:.3f})")
        return {'FINISHED'}


class EXCAD_OT_sketch_add_circle(Operator):
    bl_idname = "excad.sketch_add_circle"
    bl_label = "Sketch Circle"
    bl_options = {'REGISTER', 'UNDO'}

    cu: FloatProperty(name="Center U", default=0.0)
    cv: FloatProperty(name="Center V", default=0.0)
    radius: FloatProperty(name="Radius", default=0.5, min=0.001)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        SketchDocument.active().add_circle(self.cu, self.cv, self.radius)
        self.report({'INFO'}, f"EX-CAD: Added circle (r={self.radius:.3f})")
        return {'FINISHED'}


class EXCAD_OT_sketch_add_line(Operator):
    bl_idname = "excad.sketch_add_line"
    bl_label = "Sketch Line"
    bl_options = {'REGISTER', 'UNDO'}

    u0: FloatProperty(name="U0", default=0.0)
    v0: FloatProperty(name="V0", default=0.0)
    u1: FloatProperty(name="U1", default=1.0)
    v1: FloatProperty(name="V1", default=0.0)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        SketchDocument.active().add_line(self.u0, self.v0, self.u1, self.v1)
        return {'FINISHED'}


class EXCAD_OT_sketch_solve(Operator):
    bl_idname = "excad.sketch_solve"
    bl_label = "Solve Sketch"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        conv, iters, res = SketchDocument.active().solve()
        self.report({'INFO'}, f"EX-CAD Solve: {'converged' if conv else 'max iters'} ({iters} iters, res={res:.2e})")
        return {'FINISHED'}


class EXCAD_OT_sketch_extrude(Operator):
    bl_idname = "excad.sketch_extrude"
    bl_label = "Extrude Sketch"
    bl_options = {'REGISTER', 'UNDO'}

    depth: FloatProperty(name="Depth", default=1.0, min=0.001)
    symmetric: BoolProperty(name="Symmetric", default=False)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        doc = SketchDocument.active()
        doc.solve()
        obj = extrude_sketch_doc(doc, self.depth, self.symmetric)
        if obj is None:
            self.report({'WARNING'}, "EX-CAD: Add a closed rectangle, circle, or loop first")
            return {'CANCELLED'}
        return {'FINISHED'}


class EXCAD_OT_sketch_revolve(Operator):
    bl_idname = "excad.sketch_revolve"
    bl_label = "Revolve Sketch"
    bl_options = {'REGISTER', 'UNDO'}

    angle_deg: FloatProperty(name="Angle (deg)", default=360.0, min=1.0, max=360.0)
    segments: IntProperty(name="Segments", default=48, min=8, max=256)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        doc = SketchDocument.active()
        doc.solve()
        obj = revolve_sketch_doc(doc, self.angle_deg, self.segments)
        if obj is None:
            self.report({'WARNING'}, "EX-CAD: Add a closed profile first")
            return {'CANCELLED'}
        return {'FINISHED'}


class EXCAD_OT_sdf_humanoid(Operator):
    bl_idname = "excad.sdf_humanoid"
    bl_label = "Generate Anatomical Humanoid (SDF)"
    bl_options = {'REGISTER', 'UNDO'}

    height: FloatProperty(name="Height (m)", default=1.75, min=0.5, max=3.0)
    muscularity: FloatProperty(name="Muscularity", default=0.5, min=0.0, max=1.5)
    shoulder_ratio: FloatProperty(name="Shoulder Ratio", default=1.0, min=0.7, max=1.4)
    hip_ratio: FloatProperty(name="Hip Ratio", default=1.0, min=0.7, max=1.4)
    resolution: IntProperty(name="Resolution", default=72, min=24, max=128)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = generate_humanoid_mesh(
            self.height, self.muscularity, self.shoulder_ratio, self.hip_ratio, self.resolution
        )
        self.report({'INFO'}, f"EX-CAD: Humanoid created ({len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces)")
        return {'FINISHED'}


class EXCAD_OT_sdf_custom(Operator):
    bl_idname = "excad.sdf_custom"
    bl_label = "Generate Custom SDF Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    spec: StringProperty(name="SDF Spec", default="sphere 0 0 0.5 0.5")
    resolution: IntProperty(name="Resolution", default=64, min=24, max=128)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = generate_custom_sdf_mesh(self.spec, self.resolution)
        self.report({'INFO'}, f"EX-CAD: SDF mesh created ({len(obj.data.vertices)} verts)")
        return {'FINISHED'}


class EXCAD_OT_bridge_start(Operator):
    bl_idname = "excad.bridge_start"
    bl_label = "Start AI Direct Bridge"

    port: IntProperty(name="Port", default=9877, min=1024, max=65535)

    def execute(self, context):
        if AiBridgeServer.instance().start(self.port):
            self.report({'INFO'}, f"EX-CAD: AI Bridge listening on 127.0.0.1:{self.port}")
            return {'FINISHED'}
        self.report({'ERROR'}, "EX-CAD: Failed to start AI Bridge")
        return {'CANCELLED'}


class EXCAD_OT_bridge_stop(Operator):
    bl_idname = "excad.bridge_stop"
    bl_label = "Stop AI Direct Bridge"

    def execute(self, context):
        AiBridgeServer.instance().stop()
        self.report({'INFO'}, "EX-CAD: AI Bridge stopped")
        return {'FINISHED'}


class VIEW3D_PT_excad_sketch(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "EX-CAD"
    bl_label = "Parametric Sketch (Fusion Style)"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="1. Sketch Plane:")
        row = col.row(align=True)
        for p in ("XY", "XZ", "YZ"):
            op = row.operator("excad.sketch_new", text=p)
            op.plane = p
        col.separator()
        col.label(text="2. Sketch Entities:")
        col.operator("excad.sketch_add_rect", text="Rectangle", icon='MESH_PLANE')
        col.operator("excad.sketch_add_circle", text="Circle", icon='MESH_CIRCLE')
        col.operator("excad.sketch_add_line", text="Line", icon='IPO_LINEAR')
        col.separator()
        col.label(text="3. Constraints & Solid:")
        col.operator("excad.sketch_solve", text="Solve Constraints", icon='CONSTRAINT')
        row = col.row(align=True)
        row.operator("excad.sketch_extrude", text="Extrude", icon='MOD_SOLIDIFY')
        row.operator("excad.sketch_revolve", text="Revolve", icon='MOD_SCREW')


class VIEW3D_PT_excad_sdf(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "EX-CAD"
    bl_label = "High-Precision SDF Mesh"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.operator("excad.sdf_humanoid", text="Anatomical Humanoid", icon='OUTLINER_OB_ARMATURE')
        col.operator("excad.sdf_custom", text="Custom SDF Spec", icon='META_BALL')


class VIEW3D_PT_excad_bridge(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "EX-CAD"
    bl_label = "AI Direct Bridge (Port 9877)"

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True)
        row.operator("excad.bridge_start", text="Start Bridge", icon='PLAY')
        row.operator("excad.bridge_stop", text="Stop Bridge", icon='PAUSE')


CLASSES = (
    EXCAD_OT_sketch_new,
    EXCAD_OT_sketch_add_rect,
    EXCAD_OT_sketch_add_circle,
    EXCAD_OT_sketch_add_line,
    EXCAD_OT_sketch_solve,
    EXCAD_OT_sketch_extrude,
    EXCAD_OT_sketch_revolve,
    EXCAD_OT_sdf_humanoid,
    EXCAD_OT_sdf_custom,
    EXCAD_OT_bridge_start,
    EXCAD_OT_bridge_stop,
    VIEW3D_PT_excad_sketch,
    VIEW3D_PT_excad_sdf,
    VIEW3D_PT_excad_bridge,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


if __name__ == "__main__":
    register()
    if "--self-test" in sys.argv:
        bpy.ops.excad.sketch_new(plane='XY')
        bpy.ops.excad.sketch_add_rect(u0=-1.0, v0=-0.5, width=2.0, height=1.0)
        bpy.ops.excad.sketch_solve()
        bpy.ops.excad.sketch_extrude(depth=0.8, symmetric=True)
        ext_obj = bpy.data.objects.get("EXCAD_Extrude")

        bpy.ops.excad.sdf_humanoid(height=1.75, muscularity=0.6, resolution=56)
        hum_obj = bpy.data.objects.get("EXCAD_Humanoid")

        print(
            f"EXCAD_SELF_TEST_OK extrude=({len(ext_obj.data.vertices)},{len(ext_obj.data.polygons)}) "
            f"humanoid=({len(hum_obj.data.vertices)},{len(hum_obj.data.polygons)})"
        )
