"""Discrete curvature (Meyer et al. 2003 style): cotangent mean curvature, angle-deficit Gaussian curvature,
principal curvatures and normal variation."""
import numpy as np

from .mesh_io import neighbor_mean


def _cot(a, b):
    cr = np.linalg.norm(np.cross(a, b), axis=1)
    return np.einsum("ij,ij->i", a, b) / np.maximum(cr, 1e-12)


def curvature(st):
    co, t = st.co, st.tris
    n = len(co)
    i0, i1, i2 = t[:, 0], t[:, 1], t[:, 2]
    e0, e1, e2 = co[i2] - co[i1], co[i0] - co[i2], co[i1] - co[i0]   # opposite each corner
    # cot of angle at each corner
    c0 = _cot(co[i1] - co[i0], co[i2] - co[i0])
    c1 = _cot(co[i2] - co[i1], co[i0] - co[i1])
    c2 = _cot(co[i0] - co[i2], co[i1] - co[i2])
    area = 0.5 * np.linalg.norm(np.cross(co[i1] - co[i0], co[i2] - co[i0]), axis=1)
    A = np.zeros(n)
    for k in (i0, i1, i2):
        A += np.bincount(k, weights=area / 3.0, minlength=n)
    A = np.maximum(A, 1e-12)
    L = np.zeros_like(co)
    # edge (i1,i2) weighted by c0, (i2,i0) by c1, (i0,i1) by c2
    for (a, b, w) in ((i1, i2, c0), (i2, i0, c1), (i0, i1, c2)):
        d = co[b] - co[a]
        for c in range(3):
            L[:, c] += np.bincount(a, weights=w * d[:, c], minlength=n)
            L[:, c] -= np.bincount(b, weights=w * d[:, c], minlength=n)
    L = L / (2 * A[:, None])
    # Laplace-Beltrami of position = -2 H n  ->  H > 0 on convex regions (outward normals)
    H = -0.5 * np.einsum("ij,ij->i", L, st.normals)
    # angle deficit
    def ang(a, b):
        return np.arccos(np.clip(np.einsum("ij,ij->i", a, b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12), -1, 1))
    th = np.zeros(n)
    th += np.bincount(i0, weights=ang(co[i1] - co[i0], co[i2] - co[i0]), minlength=n)
    th += np.bincount(i1, weights=ang(co[i2] - co[i1], co[i0] - co[i1]), minlength=n)
    th += np.bincount(i2, weights=ang(co[i0] - co[i2], co[i1] - co[i2]), minlength=n)
    K = (2 * np.pi - th) / A
    disc = np.sqrt(np.maximum(H * H - K, 0))
    return {"mean": H, "gaussian": K, "k1": H + disc, "k2": H - disc}


def normal_variation(st):
    idx = np.arange(len(st.co))
    mn = neighbor_mean(st.normals, idx, st.ptr, st.nbr)
    ln = np.linalg.norm(mn, axis=1)
    return 1.0 - np.einsum("ij,ij->i", st.normals, mn) / np.maximum(ln, 1e-12)


def curvature_stats(st, robust=True):
    c = curvature(st)
    H = c["mean"]
    lo, hi = np.percentile(H, [5, 95])
    nv = normal_variation(st)
    return {"mean_curvature_p5_p50_p95": [round(float(lo), 4), round(float(np.median(H)), 4), round(float(hi), 4)],
            "gaussian_median": round(float(np.median(c["gaussian"])), 4),
            "sharp_vertex_fraction": round(float((nv > 0.02).mean()), 4),
            "noise_index": round(float(np.median(nv)), 6)}


def feature_protection(st, strength=1.0, percentile=85):
    """Weights in [0,1] that protect high-curvature anatomical features from smoothing."""
    H = np.abs(curvature(st)["mean"])
    thr = np.percentile(H, percentile)
    return np.clip((H / max(thr, 1e-9) - 0.5) * strength, 0, 1)
