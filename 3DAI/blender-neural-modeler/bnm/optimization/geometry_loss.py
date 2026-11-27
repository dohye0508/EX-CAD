"""Loss terms for shape optimisation.  L_total = sum_i w_i L_i.

Implemented (non-differentiable, evaluated numerically): silhouette, landmark, depth, normal,
curvature, laplacian, edge, preservation.  L_feature (DINOv2-style image features) needs torch and
is not available in this environment -> reported as unavailable, never faked.
"""
import numpy as np

from ..geometry.mesh_io import neighbor_mean


def silhouette_loss(mask_model, mask_ref):
    inter = (mask_model & mask_ref).sum()
    union = (mask_model | mask_ref).sum()
    return 1.0 - inter / max(union, 1)


def row_width_loss(mask_model, mask_ref):
    """Per-row signed half-width difference (model - ref) in pixels: tells WHERE to push/pull."""
    H = mask_model.shape[0]
    out = np.zeros(H)
    for y in range(H):
        a = np.nonzero(mask_model[y])[0]
        b = np.nonzero(mask_ref[y])[0]
        wa = (a.max() - a.min() + 1) / 2 if len(a) else 0
        wb = (b.max() - b.min() + 1) / 2 if len(b) else 0
        out[y] = wa - wb
    return out


def landmark_loss(pred, target):
    keys = [k for k in target if k in pred]
    if not keys:
        return 0.0
    return float(np.mean([np.linalg.norm(np.asarray(pred[k]) - np.asarray(target[k])) for k in keys]))


def depth_loss(depth_model, depth_ref, mask):
    a, b = depth_model[mask], depth_ref[mask]
    if len(a) == 0:
        return 0.0
    a, b = a - np.median(a), b - np.median(b)  # relative depth only
    return float(np.mean(np.abs(a - b)))


def normal_loss(n_model, n_ref, mask):
    if not mask.any():
        return 0.0
    return float(np.mean(1 - np.einsum("ij,ij->i", n_model[mask], n_ref[mask])))


def laplacian_energy(st, idx=None):
    idx = np.arange(len(st.co)) if idx is None else idx
    lap = neighbor_mean(st.co, idx, st.ptr, st.nbr) - st.co[idx]
    return float(np.mean(np.linalg.norm(lap, axis=1)) / max(st.mean_edge, 1e-9))


def edge_loss(st):
    e = np.linalg.norm(st.co[st.edges[:, 0]] - st.co[st.edges[:, 1]], axis=1)
    return float(np.std(e) / max(np.mean(e), 1e-9))


def curvature_loss(st, idx=None):
    from ..geometry.curvature import curvature
    H = curvature(st)["mean"]
    H = H if idx is None else H[idx]
    return float(np.percentile(np.abs(H), 98))


def preservation_loss(co_now, co_ref, keep_weights):
    d = np.linalg.norm(co_now - co_ref, axis=1)
    return float((d * keep_weights).sum() / max(keep_weights.sum(), 1e-9))


AVAILABLE = {"silhouette": True, "landmark": True, "depth": True, "normal": True, "feature": False,
             "curvature": True, "laplacian": True, "edge": True, "preservation": True}


def total(terms, weights):
    return float(sum(weights.get(k, 1.0) * v for k, v in terms.items() if v is not None))
