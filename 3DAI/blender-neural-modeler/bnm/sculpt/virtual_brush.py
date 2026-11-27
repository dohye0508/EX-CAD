"""Independent mathematical brush engine (headless-capable, deterministic).

Model:  v_i' = v_i + s * w_i * D_i,   w_i = F(d_i / r) * pressure * (1 - mask_i) * (1 - protect_i)
D_i depends on the brush (area normal, vertex normal, plane projection, Laplacian, drag delta,
regularised Kelvinlet, ...).  Displacements are normalised by brush radius and stroke spacing so
that a stroke's total effect does not depend on mesh resolution or dab spacing.
"""
import time
import numpy as np

from . import falloff as FO
from .stroke import Stroke
from .geodesic import geodesic_ball
from ..geometry.mesh_io import MeshState, neighbor_mean, vertex_normals

SURFACE_BRUSHES = {"DRAW", "DRAW_SHARP", "DRAW_SUBTRACT", "CLAY", "CLAY_STRIPS", "CLAY_THUMB", "INFLATE", "BLOB",
                   "LAYER", "CREASE", "PINCH", "FLATTEN", "SCRAPE", "FILL", "SMOOTH", "SURFACE_SMOOTH",
                   "RELAX", "POLISH", "MASK", "UNMASK"}
DRAG_BRUSHES = {"GRAB", "ELASTIC_DEFORM", "SNAKE_HOOK", "NUDGE", "THUMB"}
ALL_BRUSHES = SURFACE_BRUSHES | DRAG_BRUSHES | {"POSE"}

# default (falloff, spacing, pressure profile)
DEFAULTS = {
    "DRAW": ("SMOOTH", 0.10, "taper"), "DRAW_SHARP": ("SHARP", 0.06, "taper"),
    "DRAW_SUBTRACT": ("SMOOTH", 0.10, "taper"),
    "CLAY": ("SMOOTH", 0.12, "taper"), "CLAY_STRIPS": ("SMOOTHER", 0.10, "taper"),
    "CLAY_THUMB": ("SMOOTH", 0.10, "taper"), "INFLATE": ("SMOOTH", 0.12, "taper"),
    "BLOB": ("SMOOTH", 0.10, "taper"), "LAYER": ("SMOOTH", 0.10, "taper"),
    "CREASE": ("SHARP", 0.06, "taper"), "PINCH": ("SMOOTH", 0.10, "taper"),
    "FLATTEN": ("SMOOTH", 0.12, "taper"), "SCRAPE": ("SMOOTH", 0.12, "taper"), "FILL": ("SMOOTH", 0.12, "taper"),
    "SMOOTH": ("SMOOTH", 0.12, "constant"), "SURFACE_SMOOTH": ("SMOOTH", 0.12, "constant"),
    "RELAX": ("SMOOTH", 0.12, "constant"), "POLISH": ("SMOOTH", 0.12, "constant"),
    "MASK": ("SMOOTH", 0.12, "constant"), "UNMASK": ("SMOOTH", 0.12, "constant"),
    "GRAB": ("SMOOTH", 0.04, "constant"), "ELASTIC_DEFORM": ("SMOOTH", 0.04, "constant"),
    "SNAKE_HOOK": ("SMOOTH", 0.05, "constant"), "NUDGE": ("SMOOTH", 0.06, "constant"),
    "THUMB": ("SMOOTH", 0.06, "constant"),
}
ALIASES = {"CLAYSTRIPS": "CLAY_STRIPS", "INFLATE_DEFLATE": "INFLATE", "ELASTIC": "ELASTIC_DEFORM",
           "ELASTIC_GRAB": "ELASTIC_DEFORM", "SNAKEHOOK": "SNAKE_HOOK", "SUBTRACT": "DRAW_SUBTRACT",
           "SCRAPE_FILL": "SCRAPE", "PINCH_MAGNIFY": "PINCH", "FLATTEN_CONTRAST": "FLATTEN"}


def canonical(brush):
    b = str(brush).upper().replace(" ", "_").replace("/", "_").replace("-", "_")
    b = ALIASES.get(b, b)
    if b not in ALL_BRUSHES:
        raise ValueError(f"unsupported virtual brush {brush!r}; supported: {sorted(ALL_BRUSHES)}")
    return b


def mesh_volume(co, tris):
    v0, v1, v2 = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def kelvinlet_weights(rvec, eps, nu=0.45):
    """Regularised Kelvinlet (de Goes & James 2017) grab kernel, normalised so u(0) = f.

    Returns a function f -> displacement for each row of rvec.
    """
    a = 1.0 / (4 * np.pi)
    b = a / (4 * (1 - nu))
    r2 = (rvec ** 2).sum(1)
    re = np.sqrt(r2 + eps * eps)
    c1 = (a - b) / re + 0.5 * a * eps * eps / re ** 3
    c2 = b / re ** 3
    norm = (1.5 * a - b) / eps

    def apply(f):
        f = np.asarray(f, float)
        return (c1[:, None] * f + c2[:, None] * rvec * (rvec @ f)[:, None]) / norm
    return apply


class VirtualBrushEngine:
    def __init__(self, obj, state=None, recorder=None):
        self.obj = obj
        self.st = state or MeshState(obj)
        n = len(self.st.co)
        self.mask = self._read_mask()
        self.protect = np.zeros(n)
        self.recorder = recorder        # callable(event_dict) for audit/telemetry
        self.on_dab = None              # callable(center, radius) for live overlay
        self.stats = {"strokes": 0, "dabs": 0, "by_brush": {}}

    # ------------------------------------------------------------ masks
    def _read_mask(self):
        me = self.obj.data
        n = len(me.vertices)
        at = me.attributes.get(".sculpt_mask")
        if at is not None and at.domain == "POINT" and len(at.data) == n:
            a = np.empty(n, dtype=np.float32)
            at.data.foreach_get("value", a)
            return a.astype(np.float64)
        return np.zeros(n)

    def write_mask(self):
        me = self.obj.data
        at = me.attributes.get(".sculpt_mask") or me.attributes.new(".sculpt_mask", "FLOAT", "POINT")
        at.data.foreach_set("value", self.mask.astype(np.float32))

    def resync(self):
        """Call after any external change of the mesh (user edit, remesh)."""
        old = len(self.st.co)
        self.st.sync()
        if len(self.st.co) != old:
            self.mask = self._read_mask()
            self.protect = np.zeros(len(self.st.co))

    def set_protect(self, idx=None, values=None, clear=True):
        if clear:
            self.protect[:] = 0
        if idx is not None:
            self.protect[np.asarray(idx, dtype=np.int64)] = 1.0 if values is None else values

    # ------------------------------------------------------------ public API
    def stroke(self, brush, path, radius, strength=0.5, falloff=None, spacing=None, pressure=None,
               direction=1, symmetry_x=False, geodesic=False, frontface=True, normal=None,
               plane_offset=0.0, allowed=None, radius_scale=None, strength_scale=None,
               pressure_values=None, pinch=0.5, normal_refresh=6, write=True):
        """Execute one continuous stroke.  `path` is an (M,3) object-space polyline
        (for drag brushes: the drag trajectory starting on the surface)."""
        b = canonical(brush)
        if b == "DRAW_SUBTRACT":
            b, direction = "DRAW", -abs(direction)
        fo_name, sp_def, pr_def = DEFAULTS.get(b, ("SMOOTH", 0.1, "taper"))
        F = FO.get(falloff or fo_name)
        sp = float(spacing or sp_def)
        st = Stroke(path, radius, strength, sp, pressure or pr_def, radius_scale=radius_scale,
                    strength_scale=strength_scale, pressure_values=pressure_values)
        co0 = self.st.co.copy()
        n0 = self.st.normals.copy()
        t0 = time.perf_counter()
        allowed_mask = None
        if allowed is not None:
            allowed_mask = np.zeros(len(co0), dtype=bool)
            a = np.asarray(allowed)
            allowed_mask[a if a.dtype != bool else np.nonzero(a)[0]] = True
        ctx = dict(F=F, sp=sp, dirn=1 if direction >= 0 else -1, geodesic=geodesic, frontface=frontface,
                   normal=None if normal is None else np.asarray(normal, float) / np.linalg.norm(normal),
                   plane_offset=plane_offset, allowed=allowed_mask, o=co0, n0=n0, pinch=pinch, missed=0)
        sides = [1.0, -1.0] if symmetry_x else [1.0]
        if b in DRAG_BRUSHES:
            for side in sides:
                self._drag_stroke(b, st, side, ctx)
        else:
            for k in range(len(st)):
                for side in sides:
                    m = np.array([side, 1.0, 1.0])
                    self._dab(b, st.points[k] * m, st.radii[k], st.strengths[k] * st.pressure[k],
                              st.tangents[k] * m, ctx, side)
                if normal_refresh and k % normal_refresh == normal_refresh - 1:
                    self.st.update_normals()
                if self.on_dab:
                    self.on_dab(st.points[k], st.radii[k])
        self.st.update_normals()
        if b in ("MASK", "UNMASK"):
            self.write_mask()
        elif write:
            self.st.write()
        rep = self._report(b, st, co0, n0, time.perf_counter() - t0, ctx)
        self.stats["strokes"] += 1
        self.stats["dabs"] += len(st) * len(sides)
        self.stats["by_brush"][b] = self.stats["by_brush"].get(b, 0) + 1
        if self.recorder:
            self.recorder({"op": "virtual_stroke", "brush": b, **st.summary(), "report": rep})
        return rep

    # ------------------------------------------------------------ dab core
    def _select(self, c, r, ctx, sel_r=None):
        co = self.st.co
        sel_r = sel_r or r
        diff = co - c
        d = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        if ctx["geodesic"]:
            idx, dd = geodesic_ball(co, self.st.ptr, self.st.nbr, c, sel_r, d, ctx["allowed"])
        else:
            m = d < sel_r
            if ctx["allowed"] is not None:
                m &= ctx["allowed"]
            idx = np.nonzero(m)[0]
            dd = d[idx]
        return idx, dd

    def _area(self, idx, w0, ctx):
        co, nr = self.st.co, self.st.normals
        ws = w0.sum()
        if ws <= 1e-12:
            return None, None
        c_a = (co[idx] * w0[:, None]).sum(0) / ws
        if ctx["normal"] is not None:
            n_a = ctx["normal"]
        else:
            n_a = (nr[idx] * w0[:, None]).sum(0)
            ln = np.linalg.norm(n_a)
            if ln < 1e-12:
                return c_a, None
            n_a = n_a / ln
        return c_a, n_a

    def _dab(self, b, c, r, s, tan, ctx, side):
        F, sp, dirn = ctx["F"], ctx["sp"], ctx["dirn"]
        sel_r = r * 1.42 if b == "CLAY_STRIPS" else r
        idx, d = self._select(c, r, ctx, sel_r)
        if len(idx) == 0:
            ctx["missed"] += 1
            return
        co, nr = self.st.co, self.st.normals
        w0 = F(np.minimum(d / r, 1.0))
        c_a, n_a = self._area(idx, w0, ctx)
        if n_a is None:
            return
        if ctx["normal"] is not None and side < 0:
            n_a = n_a * np.array([-1.0, 1, 1])
        if ctx["frontface"]:
            ff = nr[idx] @ n_a
            w0 = w0 * np.clip(ff * 2.0 + 0.5, 0, 1)
        if b == "CLAY_STRIPS":
            tan_p = tan - n_a * (tan @ n_a)
            ln = np.linalg.norm(tan_p)
            tan_p = tan_p / ln if ln > 1e-9 else np.cross(n_a, [0, 0, 1.0])
            bit = np.cross(n_a, tan_p)
            rel = co[idx] - c
            sq = np.maximum(np.abs(rel @ tan_p), np.abs(rel @ bit)) / r
            w0 = FO.smoother(np.minimum(sq, 1.0)) * np.clip((nr[idx] @ n_a) * 2 + 0.5, 0, 1)
        w = w0 * s * (1 - self.mask[idx]) * (1 - self.protect[idx])
        if b in ("MASK", "UNMASK"):
            sgn = 1 if b == "MASK" else -1
            self.mask[idx] = np.clip(self.mask[idx] + sgn * w0 * s * sp * 6, 0, 1)
            return
        if not np.any(w > 0):
            return
        P = co[idx]
        if b in ("DRAW", "DRAW_SHARP"):
            disp = dirn * n_a[None, :] * (w * r * 0.5 * sp)[:, None]
        elif b == "INFLATE":
            disp = dirn * nr[idx] * (w * r * 0.4 * sp)[:, None]
        elif b == "BLOB":
            v = c - P
            v -= np.outer(v @ n_a, n_a)
            disp = dirn * nr[idx] * (w * r * 0.4 * sp)[:, None] + v * np.clip(w * sp * 1.5, 0, 1)[:, None]
        elif b == "LAYER":
            o, n0 = ctx["o"][idx], ctx["n0"][idx]
            h = np.einsum("ij,ij->i", P - o, n0)
            target = dirn * r * 0.25 * w0 * s
            nh = np.where(dirn > 0, np.maximum(h, h + (target - h) * np.clip(sp * 4, 0, 1)),
                          np.minimum(h, h + (target - h) * np.clip(sp * 4, 0, 1)))
            disp = n0 * (nh - h)[:, None]
        elif b in ("CLAY", "CLAY_STRIPS", "CLAY_THUMB"):
            n_p = n_a
            if b == "CLAY_THUMB":
                n_p = n_a - tan * 0.35 * dirn
                n_p /= np.linalg.norm(n_p)
            p_t = c_a + n_p * dirn * r * (0.08 + 0.25 * min(s, 1.0)) + n_p * ctx["plane_offset"] * r
            h = (P - p_t) @ n_p
            sel = (h < 0) if dirn > 0 else (h > 0)
            frac = np.clip(w * sp * 3.0, 0, 1) * sel
            disp = -np.outer(h * frac, n_p)
        elif b in ("FLATTEN", "SCRAPE", "FILL", "POLISH"):
            p_t = c_a + n_a * ctx["plane_offset"] * r
            if b == "SCRAPE":
                p_t = p_t - n_a * r * 0.03 * s
            h = (P - p_t) @ n_a
            if b == "SCRAPE":
                sel = h > 0
            elif b == "FILL":
                sel = h < 0
            else:
                sel = np.ones_like(h, dtype=bool)
            frac = np.clip(w * sp * 3.0, 0, 1) * sel
            disp = -np.outer(h * frac, n_a)
            if b == "POLISH":
                lap = neighbor_mean(co, idx, self.st.ptr, self.st.nbr) - P
                lap -= nr[idx] * np.einsum("ij,ij->i", lap, nr[idx])[:, None]
                disp = disp * 0.4 + lap * np.clip(w * sp * 6, 0, 1)[:, None]
        elif b == "PINCH":
            v = c - P
            v -= np.outer(v @ n_a, n_a)
            disp = dirn * v * np.clip(w * sp * 2.0, 0, 0.5)[:, None]
        elif b == "CREASE":
            v = c - P
            v -= np.outer(v @ n_a, n_a)
            tp = tan - n_a * (tan @ n_a)
            ln = np.linalg.norm(tp)
            if ln > 1e-9:
                tp /= ln
                v -= np.outer(v @ tp, tp)
            disp = (-dirn) * n_a[None, :] * (w * r * 0.35 * sp)[:, None] \
                + v * np.clip(w * sp * 2.0 * ctx["pinch"], 0, 0.5)[:, None]
        elif b == "SMOOTH":
            lap = neighbor_mean(co, idx, self.st.ptr, self.st.nbr) - P
            disp = lap * np.clip(w * sp * 8.0, 0, 1)[:, None]
        elif b == "RELAX":
            lap = neighbor_mean(co, idx, self.st.ptr, self.st.nbr) - P
            lap -= nr[idx] * np.einsum("ij,ij->i", lap, nr[idx])[:, None]
            disp = lap * np.clip(w * sp * 8.0, 0, 1)[:, None]
        elif b == "SURFACE_SMOOTH":
            # HC-Laplacian (Vollmer et al. 1999): smooth while pushing back toward original shape
            alpha, beta = 0.1, 0.5
            o = ctx["o"]
            p = neighbor_mean(co, idx, self.st.ptr, self.st.nbr)
            bvec = p - (alpha * o[idx] + (1 - alpha) * P)
            bfull = np.zeros_like(co)
            bfull[idx] = bvec
            bmean = neighbor_mean(co, idx, self.st.ptr, self.st.nbr, values=bfull)
            q = p - (beta * bvec + (1 - beta) * bmean)
            disp = (q - P) * np.clip(w * sp * 8.0, 0, 1)[:, None]
        else:
            return
        co[idx] = P + disp

    # ------------------------------------------------------------ drag brushes
    def _drag_stroke(self, b, st, side, ctx):
        m = np.array([side, 1.0, 1.0])
        pts = st.points * m
        F = ctx["F"]
        co = self.st.co
        if len(pts) < 2:
            return
        r0 = st.radii[0]
        if b in ("GRAB", "ELASTIC_DEFORM"):
            c0 = pts[0]
            if b == "GRAB":
                idx, d = self._select(c0, r0, ctx)
                if len(idx) == 0:
                    ctx["missed"] += 1
                    return
                w0 = F(d / r0)
                c_a, n_a = self._area(idx, w0, ctx)
                if ctx["frontface"] and n_a is not None:
                    w0 = w0 * np.clip((self.st.normals[idx] @ n_a) * 2 + 0.5, 0, 1)
                w = w0 * st.strength * (1 - self.mask[idx]) * (1 - self.protect[idx])
                base = co[idx].copy()
                for k in range(1, len(pts)):
                    co[idx] = base + np.outer(w, pts[k] - c0)
                    if self.on_dab:
                        self.on_dab(pts[k], r0)
            else:
                eps = r0
                reach = 3.5 * eps
                d = np.linalg.norm(co - c0, axis=1)
                sel = d < reach
                if ctx["allowed"] is not None:
                    sel &= ctx["allowed"]
                idx = np.nonzero(sel)[0]
                base = co[idx].copy()
                K = kelvinlet_weights(base - c0, eps)
                win = FO.smooth(np.clip((d[idx] - 2.5 * eps) / eps, 0, 1))  # localise tail to 3.5*eps
                win = win * st.strength * (1 - self.mask[idx]) * (1 - self.protect[idx])
                for k in range(1, len(pts)):
                    co[idx] = base + K(pts[k] - c0) * win[:, None]
                    if self.on_dab:
                        self.on_dab(pts[k], r0)
            return
        # SNAKE_HOOK / NUDGE / THUMB: re-select every dab around the *previous* point
        for k in range(1, len(pts)):
            r = st.radii[k]
            prev, cur = pts[k - 1], pts[k]
            delta = cur - prev
            idx, d = self._select(prev, r, ctx)
            if len(idx) == 0:
                ctx["missed"] += 1
                continue
            w0 = F(d / r)
            c_a, n_a = self._area(idx, w0, ctx)
            w = w0 * st.strength * (1 - self.mask[idx]) * (1 - self.protect[idx])
            if b == "SNAKE_HOOK":
                v = prev - co[idx]
                if n_a is not None:
                    v -= np.outer(v @ n_a, n_a)
                co[idx] += np.outer(w, delta) + v * np.clip(w * 0.15 * ctx["pinch"], 0, 0.3)[:, None]
            else:
                dt = delta - (n_a * (delta @ n_a) if n_a is not None else 0)
                co[idx] += np.outer(w * (0.6 if b == "THUMB" else 1.0), dt)
            if k % 6 == 5:
                self.st.update_normals()
            if self.on_dab:
                self.on_dab(cur, r)

    # ------------------------------------------------------------ pose
    def pose_rotate(self, pivot, axis, angle_deg, region_idx, weights=None, write=True):
        """Pose brush: rigidly rotate a weighted region about an axis through `pivot`."""
        import mathutils
        co0 = self.st.co.copy()
        n0 = self.st.normals.copy()
        t0 = time.perf_counter()
        idx = np.asarray(region_idx, dtype=np.int64)
        w = np.ones(len(idx)) if weights is None else np.asarray(weights, float)
        w = w * (1 - self.mask[idx]) * (1 - self.protect[idx])
        ax = np.asarray(axis, float) / np.linalg.norm(axis)
        pv = np.asarray(pivot, float)
        P = self.st.co[idx] - pv
        # Rodrigues, per-vertex angle = w * angle
        th = np.radians(angle_deg) * w
        ct, stn = np.cos(th)[:, None], np.sin(th)[:, None]
        rot = P * ct + np.cross(ax, P) * stn + np.outer(P @ ax, ax) * (1 - ct)
        self.st.co[idx] = rot + pv
        self.st.update_normals()
        if write:
            self.st.write()
        st = Stroke(np.array([pv, pv]), 1.0, 1.0)
        rep = self._report("POSE", st, co0, n0, time.perf_counter() - t0, {"missed": 0})
        self.stats["by_brush"]["POSE"] = self.stats["by_brush"].get("POSE", 0) + 1
        if self.recorder:
            self.recorder({"op": "virtual_pose", "pivot": pv.round(4).tolist(), "axis": ax.round(4).tolist(),
                           "angle": angle_deg, "report": rep})
        return rep

    # ------------------------------------------------------------ conform brush
    def blend_to(self, idx, targets, weights, label="CONFORM", region=None, write=True):
        """Move weighted vertices toward explicit target positions (shrinkwrap/conform-style brush,
        e.g. wrapping eyelid skin onto an eyeball shell).  v' = v + w (t - v)."""
        co0 = self.st.co.copy()
        n0 = self.st.normals.copy()
        t0 = time.perf_counter()
        idx = np.asarray(idx, dtype=np.int64)
        w = np.clip(np.asarray(weights, float), 0, 1) * (1 - self.mask[idx]) * (1 - self.protect[idx])
        self.st.co[idx] += (np.asarray(targets, float) - self.st.co[idx]) * w[:, None]
        self.st.update_normals()
        if write:
            self.st.write()
        rep = self._report(label, Stroke(np.zeros((1, 3)), 1.0), co0, n0, time.perf_counter() - t0, {"missed": 0})
        self.stats["by_brush"][label] = self.stats["by_brush"].get(label, 0) + 1
        if self.recorder:
            self.recorder({"op": "virtual_stroke", "brush": label, "region": region, "report": rep})
        return rep

    # ------------------------------------------------------------ target-volume brush
    def conform_to_field(self, field, radius, max_step=0.03, tol=0.004, max_dabs=60, min_sep=None,
                         allowed=None, label="TARGET_CONFORM", region=None, write=True):
        """Push/pull the surface toward a target volume (spec 3.1: find missing/excess volume, then
        push, pull or carve).  `field(P)` = signed distance of points to the target surface
        (+ outside).  Dabs are placed greedily at the largest errors; each dab moves vertices along
        their normals by -w*clip(d) (pulls inward where there is excess, outward where volume is missing)."""
        co = self.st.co
        co0 = co.copy()
        n0 = self.st.normals.copy()
        t0 = time.perf_counter()
        d = field(co)
        err = np.abs(d)
        if allowed is not None:
            err = err * allowed
        rms0 = float(np.sqrt(np.mean(d ** 2)))
        order = np.argsort(-err)
        min_sep = min_sep or radius * 0.8
        centers = []
        for i in order:
            if err[i] < tol or len(centers) >= max_dabs:
                break
            p = co[i]
            if all(np.sum((p - co[j]) ** 2) > min_sep * min_sep for j in centers):
                centers.append(int(i))
        for j in centers:
            c = co[j].copy()
            dist = np.linalg.norm(co - c, axis=1)
            idx = np.nonzero(dist < radius)[0]
            w = FO.smooth(dist[idx] / radius) * (1 - self.mask[idx]) * (1 - self.protect[idx])
            if allowed is not None:
                w = w * allowed[idx]
            dd = np.clip(field(co[idx]), -max_step, max_step)
            co[idx] -= self.st.normals[idx] * (w * dd)[:, None]
            if self.on_dab:
                self.on_dab(c, radius)
        self.st.update_normals()
        if write:
            self.st.write()
        d1 = field(self.st.co)
        rep = self._report(label, Stroke(np.zeros((1, 3)), 1.0), co0, n0, time.perf_counter() - t0, {"missed": 0})
        rep.update({"dabs": len(centers), "rms_before": rms0, "rms_after": float(np.sqrt(np.mean(d1 ** 2))),
                    "max_error_after": float(np.abs(d1 if allowed is None else d1 * allowed).max())})
        self.stats["by_brush"][label] = self.stats["by_brush"].get(label, 0) + 1
        if self.recorder:
            self.recorder({"op": "virtual_stroke", "brush": label, "region": region, "report": rep})
        return rep

    def grow_into_volume(self, vol, radius, max_step=0.03, tmax=0.35, tol=0.003, max_dabs=60, allowed=None,
                         label="CLAY_INTO_VOLUME", region=None, write=True, direction=None, wide=False):
        """Build clay outward along the surface normals until the surface leaves a feature volume
        (e.g. a soft nose wedge in front of the face).  For each vertex the exit distance of its
        normal ray through `vol` is found by marching; dabs are placed at the largest exit distances."""
        st = self.st
        co0 = st.co.copy()
        n0 = st.normals.copy()
        t0 = time.perf_counter()
        cand = np.arange(len(st.co)) if allowed is None else np.nonzero(np.asarray(allowed) > 1e-3)[0]
        ts = np.linspace(0, tmax, int(tmax / 0.004) + 1)

        dvec = None if direction is None else np.asarray(direction, float) / np.linalg.norm(direction)

        def dirs(idx):
            return st.normals[idx] if dvec is None else np.tile(dvec, (len(idx), 1))

        def exit_dist(idx):
            P = st.co[idx][:, None, :] + dirs(idx)[:, None, :] * ts[None, :, None]
            inside = vol(P.reshape(-1, 3)).reshape(len(idx), len(ts)) < 0
            any_in = inside.any(1)
            last = len(ts) - 1 - np.argmax(inside[:, ::-1], axis=1)
            return np.where(any_in, ts[last], 0.0)
        ex = np.zeros(len(st.co))
        ex[cand] = exit_dist(cand)
        if allowed is not None:
            ex = ex * np.asarray(allowed)
        before = float(ex.max()) if len(ex) else 0.0
        if wide:   # one brush covering the whole allowed region: coherent growth, no dab spikes
            w = (1 - self.mask) * (1 - self.protect) * (np.ones(len(st.co)) if allowed is None else np.asarray(allowed))
            st.co += dirs(np.arange(len(st.co))) * (np.minimum(ex, max_step) * w)[:, None]
            max_dabs = 0
        order = np.argsort(-ex)
        centers = []
        for i in order:
            if ex[i] < tol or len(centers) >= max_dabs:
                break
            if all(np.sum((st.co[i] - st.co[j]) ** 2) > (radius * 0.7) ** 2 for j in centers):
                centers.append(int(i))
        for j in centers:
            c = st.co[j].copy()
            dist = np.linalg.norm(st.co - c, axis=1)
            idx = np.nonzero(dist < radius)[0]
            w = FO.smooth(dist[idx] / radius) * (1 - self.mask[idx]) * (1 - self.protect[idx])
            if allowed is not None:
                w = w * np.asarray(allowed)[idx]
            e = np.minimum(exit_dist(idx), max_step)
            st.co[idx] += dirs(idx) * (w * e)[:, None]
            if self.on_dab:
                self.on_dab(c, radius)
        st.update_normals()
        if write:
            st.write()
        rep = self._report(label, Stroke(np.zeros((1, 3)), 1.0), co0, n0, time.perf_counter() - t0, {"missed": 0})
        rep.update({"dabs": len(centers), "max_exit_before": before})
        self.stats["by_brush"][label] = self.stats["by_brush"].get(label, 0) + 1
        if self.recorder:
            self.recorder({"op": "virtual_stroke", "brush": label, "region": region, "report": rep})
        return rep

    def conform_global(self, field, step=0.03, gain=0.6, allowed=None, label="TARGET_CONFORM_WIDE", region=None, write=True):
        """Whole-region conform pass (a brush as large as the region): no dab bumps."""
        st = self.st
        co0 = st.co.copy()
        n0 = st.normals.copy()
        d = np.clip(field(st.co), -step, step) * gain
        w = (1 - self.mask) * (1 - self.protect)
        if allowed is not None:
            w = w * np.asarray(allowed)
        st.co -= st.normals * (d * w)[:, None]
        st.update_normals()
        if write:
            st.write()
        rep = self._report(label, Stroke(np.zeros((1, 3)), 1.0), co0, n0, 0.0, {"missed": 0})
        rep["rms_after"] = float(np.sqrt(np.mean(field(st.co) ** 2)))
        self.stats["by_brush"][label] = self.stats["by_brush"].get(label, 0) + 1
        if self.recorder:
            self.recorder({"op": "virtual_stroke", "brush": label, "region": region, "report": rep})
        return rep

    def relax_smooth(self, weights=None, iterations=20, factor=0.5, tangential_first=True, write=True, label="REGION_SMOOTH"):
        """Many-iteration region smoothing (the 'Shift after every stroke' habit): taubin-style
        (lambda/mu) so the region does not shrink, optional tangential relax first."""
        n = len(self.st.co)
        w = np.ones(n) if weights is None else np.clip(np.asarray(weights, float), 0, 1)
        w = w * (1 - self.mask) * (1 - self.protect)
        idx = np.nonzero(w > 1e-3)[0]
        co0 = self.st.co.copy()
        n0 = self.st.normals.copy()
        wi = w[idx][:, None]
        lam, mu = factor, -factor * 1.04
        for it in range(iterations):
            if tangential_first and it < iterations // 3:
                lap = neighbor_mean(self.st.co, idx, self.st.ptr, self.st.nbr) - self.st.co[idx]
                nn = self.st.normals[idx]
                lap -= nn * np.einsum("ij,ij->i", lap, nn)[:, None]
                self.st.co[idx] += lap * wi * lam
                continue
            for f in (lam, mu):
                lap = neighbor_mean(self.st.co, idx, self.st.ptr, self.st.nbr) - self.st.co[idx]
                self.st.co[idx] += lap * wi * f
            if it % 5 == 4:
                self.st.update_normals()
        self.st.update_normals()
        if write:
            self.st.write()
        rep = self._report(label, Stroke(np.zeros((1, 3)), 1.0), co0, n0, 0.0, {"missed": 0})
        rep["iterations"] = iterations
        self.stats["by_brush"][label] = self.stats["by_brush"].get(label, 0) + 1
        if self.recorder:
            self.recorder({"op": "virtual_stroke", "brush": label, "report": rep})
        return rep

    # ------------------------------------------------------------ region helpers
    def offset_region(self, idx, amount, weights=None, use_normals=True, direction=None, write=True):
        """Directly displace a weighted region (used by optimisers; logged like a stroke)."""
        co0 = self.st.co.copy()
        n0 = self.st.normals.copy()
        idx = np.asarray(idx, dtype=np.int64)
        w = np.ones(len(idx)) if weights is None else np.asarray(weights, float)
        w = w * (1 - self.mask[idx]) * (1 - self.protect[idx])
        D = self.st.normals[idx] if use_normals else np.tile(np.asarray(direction, float), (len(idx), 1))
        self.st.co[idx] += D * (w * amount)[:, None]
        self.st.update_normals()
        if write:
            self.st.write()
        return self._report("OFFSET", Stroke(np.zeros((1, 3)), 1.0), co0, n0, 0.0, {"missed": 0})

    def smooth_region(self, idx, iterations=3, factor=0.5, tangential=False, write=True):
        idx = np.asarray(idx, dtype=np.int64)
        co0 = self.st.co.copy()
        n0 = self.st.normals.copy()
        f = factor * (1 - self.mask[idx]) * (1 - self.protect[idx])
        for _ in range(iterations):
            lap = neighbor_mean(self.st.co, idx, self.st.ptr, self.st.nbr) - self.st.co[idx]
            if tangential:
                nn = self.st.normals[idx]
                lap -= nn * np.einsum("ij,ij->i", lap, nn)[:, None]
            self.st.co[idx] += lap * f[:, None]
        self.st.update_normals()
        if write:
            self.st.write()
        return self._report("SMOOTH_REGION", Stroke(np.zeros((1, 3)), 1.0), co0, n0, 0.0, {"missed": 0})

    # ------------------------------------------------------------ reporting
    def _report(self, b, st, co0, n0, secs, ctx):
        co = self.st.co
        dv = co - co0
        dist = np.linalg.norm(dv, axis=1)
        moved = dist > 1e-7
        rep = {"brush": b, "dabs": len(st), "moved_vertices": int(moved.sum()),
               "max_displacement": float(dist.max()) if len(dist) else 0.0,
               "mean_displacement": float(dist[moved].mean()) if moved.any() else 0.0,
               "missed_dabs": int(ctx.get("missed", 0)), "seconds": round(secs, 4)}
        if moved.any():
            wts = dist[moved]
            rep["change_centroid"] = (co0[moved] * wts[:, None]).sum(0).__truediv__(wts.sum()).round(4).tolist()
            rep["mean_normal_displacement"] = float(np.einsum("ij,ij->i", dv[moved], n0[moved]).mean())
            rep["volume_delta"] = mesh_volume(co, self.st.tris) - mesh_volume(co0, self.st.tris)
        return rep
