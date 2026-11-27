# SPDX-FileCopyrightText: 2026 EX-CAD Authors
# SPDX-License-Identifier: GPL-2.0-or-later

import math
from dataclasses import dataclass

import bpy
import bmesh
import numpy as np


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

    def add_constraint(
        self,
        ctype: str,
        a: int,
        b: int = -1,
        c: int = -1,
        d: int = -1,
        value: float = 0.0,
        value2: float = 0.0,
    ) -> int:
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
                area2 = sum(
                    loop[k][0] * loop[(k + 1) % len(loop)][1]
                    - loop[(k + 1) % len(loop)][0] * loop[k][1]
                    for k in range(len(loop))
                )
                if area2 < 0.0:
                    loop.reverse()
                loops.append(loop)
        return loops


def _create_mesh_object(name: str, verts: list, faces: list):
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
    return _create_mesh_object(name, verts, faces)


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
    return _create_mesh_object(name, verts, faces)
