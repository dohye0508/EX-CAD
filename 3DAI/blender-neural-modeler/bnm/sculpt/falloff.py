"""Brush falloff functions F(t), t = d / r.

Every function returns 1 at t=0 and reaches exactly 0 at t>=1 with a continuous value,
so the brush boundary never produces a step in the surface.
"""
import numpy as np


def _clip(t):
    return np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)


def linear(t):
    return 1.0 - _clip(t)


def smooth(t):  # smoothstep, C1 at both ends (Blender "Smooth")
    t = _clip(t)
    return 1.0 - (3 * t * t - 2 * t ** 3)


def smoother(t):  # quintic smootherstep, C2
    t = _clip(t)
    return 1.0 - (6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3)


def gaussian(t, sigma=0.4):
    t = _clip(t)
    g = np.exp(-(t * t) / (2 * sigma * sigma))
    g0 = np.exp(-1.0 / (2 * sigma * sigma))
    return np.clip((g - g0) / (1 - g0), 0, 1)  # shifted so F(1)=0


def polynomial(t, power=2.0):
    return (1.0 - _clip(t) ** power)


def sharp(t):  # concentrated centre, (1-t)^2
    return (1.0 - _clip(t)) ** 2


def root(t):
    return np.sqrt(np.maximum(1.0 - _clip(t), 0))


def sphere(t):
    t = _clip(t)
    return np.sqrt(np.maximum(1.0 - t * t, 0.0))


def constant(t):
    # constant inside but softened in the outer 5% to avoid a hard step
    t = _clip(t)
    return np.where(t < 0.95, 1.0, (1.0 - t) / 0.05)


def make_custom(points):
    """points: [(t, value), ...] sorted by t; endpoint forced to 0 at t=1."""
    pts = sorted(points)
    xs = np.array([p[0] for p in pts] + [1.0])
    ys = np.array([p[1] for p in pts] + [0.0])

    def f(t):
        return np.interp(_clip(t), xs, ys)
    return f


FALLOFFS = {
    "LINEAR": linear, "SMOOTH": smooth, "SMOOTHER": smoother, "GAUSSIAN": gaussian,
    "POLYNOMIAL": polynomial, "SHARP": sharp, "ROOT": root, "SPHERE": sphere, "CONSTANT": constant,
}


def get(name_or_points):
    if callable(name_or_points):
        return name_or_points
    if isinstance(name_or_points, (list, tuple)):
        return make_custom(name_or_points)
    key = str(name_or_points).upper()
    if key not in FALLOFFS:
        raise ValueError(f"unknown falloff {name_or_points!r}; options: {sorted(FALLOFFS)}")
    return FALLOFFS[key]


# Blender curve_distance_falloff_preset names for the native engine
BLENDER_PRESET = {"LINEAR": "LIN", "SMOOTH": "SMOOTH", "SMOOTHER": "SMOOTHER", "SHARP": "SHARP",
                  "ROOT": "ROOT", "SPHERE": "SPHERE", "CONSTANT": "CONSTANT", "GAUSSIAN": "SMOOTH",
                  "POLYNOMIAL": "POW4"}
