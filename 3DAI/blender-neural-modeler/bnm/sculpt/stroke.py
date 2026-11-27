"""Stroke path generation, interpolation, arc-length resampling and virtual pen pressure.

A stroke is a list of dabs; each dab = (position, pressure, radius, strength, tangent).
Paths are generated in object space and later snapped to the live surface per dab.
"""
import math
import numpy as np


# ---------------------------------------------------------------- curves
def line(a, b, n=16):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.linspace(0, 1, n)[:, None]
    return a + (b - a) * t


def bezier(p0, p1, p2, p3, n=32):
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    t = np.linspace(0, 1, n)[:, None]
    u = 1 - t
    return u ** 3 * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t ** 3 * p3


def quad_bezier(p0, p1, p2, n=24):
    p0, p1, p2 = (np.asarray(p, float) for p in (p0, p1, p2))
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2


def catmull_rom(points, samples_per_seg=12, alpha=0.5):
    """Centripetal Catmull-Rom spline through all control points (no overshoot/cusps)."""
    P = np.asarray(points, float)
    if len(P) < 2:
        return P.copy()
    if len(P) == 2:
        return line(P[0], P[1], samples_per_seg + 1)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0) ** alpha, 1e-9)
        t2 = t1 + max(np.linalg.norm(p2 - p1) ** alpha, 1e-9)
        t3 = t2 + max(np.linalg.norm(p3 - p2) ** alpha, 1e-9)
        ts = np.linspace(t1, t2, samples_per_seg, endpoint=(i == len(P) - 3))[:, None]
        a1 = (t1 - ts) / (t1 - t0) * p0 + (ts - t0) / (t1 - t0) * p1
        a2 = (t2 - ts) / (t2 - t1) * p1 + (ts - t1) / (t2 - t1) * p2
        a3 = (t3 - ts) / (t3 - t2) * p2 + (ts - t2) / (t3 - t2) * p3
        b1 = (t2 - ts) / (t2 - t0) * a1 + (ts - t0) / (t2 - t0) * a2
        b2 = (t3 - ts) / (t3 - t1) * a2 + (ts - t1) / (t3 - t1) * a3
        out.append((t2 - ts) / (t2 - t1) * b1 + (ts - t1) / (t2 - t1) * b2)
    return np.vstack(out)


def arc(center, normal, radius, start_dir, angle_deg, n=24):
    """Circular arc around `center` in the plane orthogonal to `normal`."""
    c, nrm, d0 = (np.asarray(v, float) for v in (center, normal, start_dir))
    nrm = nrm / np.linalg.norm(nrm)
    d0 = d0 - nrm * (d0 @ nrm)
    d0 /= np.linalg.norm(d0)
    d1 = np.cross(nrm, d0)
    a = np.radians(np.linspace(0, angle_deg, n))[:, None]
    return c + radius * (np.cos(a) * d0 + np.sin(a) * d1)


def ellipse(center, axis_u, axis_v, n=48, a0=0.0, a1=360.0):
    c, u, v = (np.asarray(x, float) for x in (center, axis_u, axis_v))
    a = np.radians(np.linspace(a0, a1, n))[:, None]
    return c + np.cos(a) * u + np.sin(a) * v


def parallel(path, offset_dir, count, spacing):
    """count copies of path offset along offset_dir, centred on the original."""
    path = np.asarray(path, float)
    d = np.asarray(offset_dir, float)
    d = d / np.linalg.norm(d)
    offs = (np.arange(count) - (count - 1) / 2.0) * spacing
    return [path + o * d for o in offs]


def radial(center, normal, length, count, start_dir=None, inward=False):
    """count straight strokes radiating from center in the tangent plane."""
    c, nrm = np.asarray(center, float), np.asarray(normal, float)
    nrm /= np.linalg.norm(nrm)
    ref = np.asarray(start_dir if start_dir is not None else ([1, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1, 0]), float)
    u = ref - nrm * (ref @ nrm)
    u /= np.linalg.norm(u)
    v = np.cross(nrm, u)
    paths = []
    for k in range(count):
        a = 2 * math.pi * k / count
        d = math.cos(a) * u + math.sin(a) * v
        p = line(c, c + d * length, 12)
        paths.append(p[::-1] if inward else p)
    return paths


def fill_region(center, normal, half_u, half_v, spacing, u_dir=None):
    """Zig-zag (boustrophedon) fill of a rectangle in the tangent plane -> one continuous path."""
    c, nrm = np.asarray(center, float), np.asarray(normal, float)
    nrm /= np.linalg.norm(nrm)
    ref = np.asarray(u_dir if u_dir is not None else ([1, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1, 0]), float)
    u = ref - nrm * (ref @ nrm)
    u /= np.linalg.norm(u)
    v = np.cross(nrm, u)
    rows = max(2, int(round(2 * half_v / spacing)) + 1)
    pts = []
    for i, tv in enumerate(np.linspace(-half_v, half_v, rows)):
        xs = np.linspace(-half_u, half_u, 8)
        if i % 2:
            xs = xs[::-1]
        for tu in xs:
            pts.append(c + tu * u + tv * v)
    return np.array(pts)


def resample(path, spacing):
    """Resample a polyline at uniform arc-length spacing (the 'Stroke Spacing')."""
    P = np.asarray(path, float)
    if len(P) < 2:
        return P.copy(), np.zeros(len(P))
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    if L <= 1e-12:
        return P[:1].copy(), np.zeros(1)
    n = max(2, int(math.ceil(L / max(spacing, 1e-9))) + 1)
    st = np.linspace(0, L, n)
    out = np.column_stack([np.interp(st, s, P[:, k]) for k in range(3)])
    return out, st / L


def tangents(path):
    P = np.asarray(path, float)
    if len(P) < 2:
        return np.tile([1.0, 0, 0], (len(P), 1))
    t = np.gradient(P, axis=0)
    ln = np.linalg.norm(t, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return t / ln


# ---------------------------------------------------------------- pressure
def pressure_profile(u, kind="taper", taper_in=0.2, taper_out=0.25, base=1.0, values=None):
    """Virtual pen pressure along the normalised arc-length u in [0,1].

    taper   : smooth ramp-in and ramp-out (like a lifted tablet pen)
    constant: flat pressure
    ramp_up / ramp_down: linear change across the stroke
    peak    : sin bump (strongest in the middle)
    custom  : interpolate `values` evenly over u
    """
    u = np.asarray(u, float)
    if kind == "constant":
        p = np.ones_like(u)
    elif kind == "ramp_up":
        p = 0.2 + 0.8 * u
    elif kind == "ramp_down":
        p = 1.0 - 0.8 * u
    elif kind == "peak":
        p = np.sin(np.pi * u) ** 0.8
    elif kind == "custom" and values is not None:
        p = np.interp(u, np.linspace(0, 1, len(values)), values)
    else:
        p = np.ones_like(u)
        if taper_in > 0:
            x = np.clip(u / taper_in, 0, 1)
            p *= x * x * (3 - 2 * x)
        if taper_out > 0:
            x = np.clip((1 - u) / taper_out, 0, 1)
            p *= x * x * (3 - 2 * x)
        p = 0.12 + 0.88 * p  # a real pen never reaches exactly 0 while touching
    return np.clip(p * base, 0, 1.5)


class Stroke:
    """A fully specified stroke: dab positions with per-dab pressure, radius and strength."""

    def __init__(self, path, radius, strength=0.5, spacing=0.12, pressure="taper",
                 radius_pressure=0.0, radius_scale=None, strength_scale=None, pressure_values=None,
                 taper_in=0.2, taper_out=0.25):
        self.control = np.asarray(path, float)
        self.radius = float(radius)
        self.strength = float(strength)
        self.spacing = float(spacing)
        pts, u = resample(self.control, self.spacing * self.radius) if len(self.control) > 1 else (self.control.copy(), np.zeros(1))
        self.points = pts
        self.u = u
        self.pressure = pressure_profile(u, pressure, taper_in, taper_out, values=pressure_values) if len(pts) > 1 else np.ones(1)
        rs = np.ones(len(pts)) if radius_scale is None else np.interp(u, np.linspace(0, 1, len(radius_scale)), radius_scale)
        ss = np.ones(len(pts)) if strength_scale is None else np.interp(u, np.linspace(0, 1, len(strength_scale)), strength_scale)
        # pressure can also modulate the size (tablet "size pressure")
        self.radii = self.radius * rs * (1 - radius_pressure + radius_pressure * self.pressure)
        self.strengths = self.strength * ss
        self.tangents = tangents(pts)

    def __len__(self):
        return len(self.points)

    def summary(self):
        return {"dabs": len(self), "start": self.points[0].round(4).tolist(), "end": self.points[-1].round(4).tolist(),
                "radius": round(self.radius, 4), "strength": round(self.strength, 3),
                "length": float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum()) if len(self) > 1 else 0.0}


def build_path(kind, **kw):
    """Factory used by plans/JSON: kind in line|polyline|bezier|catmull_rom|arc|ellipse|fill."""
    if kind == "line":
        return line(kw["start"], kw["end"], kw.get("n", 16))
    if kind == "polyline":
        return np.asarray(kw["points"], float)
    if kind == "bezier":
        return bezier(*kw["points"][:4], n=kw.get("n", 32))
    if kind in ("catmull_rom", "spline", "curve"):
        return catmull_rom(kw["points"], kw.get("samples_per_seg", 12))
    if kind == "arc":
        return arc(kw["center"], kw["normal"], kw["radius"], kw["start_dir"], kw["angle"], kw.get("n", 24))
    if kind == "ellipse":
        return ellipse(kw["center"], kw["axis_u"], kw["axis_v"], kw.get("n", 48), kw.get("a0", 0), kw.get("a1", 360))
    if kind == "fill":
        return fill_region(kw["center"], kw["normal"], kw["half_u"], kw["half_v"], kw["spacing"], kw.get("u_dir"))
    if kind == "dot":
        return np.asarray([kw["center"]], float)
    raise ValueError(f"unknown path kind {kind}")
