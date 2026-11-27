"""Compact spatial summaries of a mesh for the LLM (never dump all vertices)."""
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .mesh_io import MeshState, neighbor_mean


def bvh_from_state(st):
    return BVHTree.FromPolygons(st.co.tolist(), st.tris.tolist(), all_triangles=True)


def surface_area(co, tris):
    return float(0.5 * np.linalg.norm(np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]]), axis=1).sum())


def volume(co, tris):
    v0, v1, v2 = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def symmetry_error(st, bvh=None, axis=0, samples=3000, seed=0):
    """Mean distance from mirrored sample vertices to the surface, relative to bbox diagonal."""
    bvh = bvh or bvh_from_state(st)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(st.co), min(samples, len(st.co)), replace=False)
    m = st.co[idx].copy()
    m[:, axis] *= -1
    d = np.array([bvh.find_nearest(Vector(p))[3] for p in m])
    diag = float(np.linalg.norm(st.co.max(0) - st.co.min(0)))
    return {"mean": float(d.mean()), "p95": float(np.percentile(d, 95)), "relative": float(d.mean() / max(diag, 1e-9))}


def cross_section(st, origin, normal):
    """Intersect the mesh with a plane. Returns segment endpoints (K,2,3) + extents in-plane."""
    o, n = np.asarray(origin, float), np.asarray(normal, float) / np.linalg.norm(normal)
    s = (st.co - o) @ n
    t = st.tris
    st_ = s[t]
    pos = st_ > 0
    crossing = pos.any(1) & ~pos.all(1)
    segs = []
    for tri, sv in zip(t[crossing], st_[crossing]):
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (sv[a] > 0) != (sv[b] > 0):
                u = sv[a] / (sv[a] - sv[b])
                pts.append(st.co[tri[a]] + u * (st.co[tri[b]] - st.co[tri[a]]))
        if len(pts) == 2:
            segs.append(pts)
    segs = np.array(segs) if segs else np.zeros((0, 2, 3))
    out = {"segments": int(len(segs))}
    if len(segs):
        P = segs.reshape(-1, 3)
        ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        u = ref - n * (ref @ n)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        pu, pv = (P - o) @ u, (P - o) @ v
        L = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1).sum()
        out.update({"u_axis": u.round(4).tolist(), "v_axis": v.round(4).tolist(),
                    "extent_u": [float(pu.min()), float(pu.max())], "extent_v": [float(pv.min()), float(pv.max())],
                    "perimeter": float(L)})
    return out, segs


def slab_profile(co, z_levels, slab=0.03, mid_band=0.06, axis_up=2, axis_side=0, axis_depth=1):
    """Silhouette numbers per height level: half-width, front depth (min y on midline), back depth."""
    rows = []
    for z in z_levels:
        sel = np.abs(co[:, axis_up] - z) < slab
        if not sel.any():
            rows.append({"z": float(z), "half_width": 0.0, "front": None, "back": None})
            continue
        P = co[sel]
        mid = np.abs(P[:, axis_side]) < mid_band
        rows.append({"z": float(z), "half_width": float(np.abs(P[:, axis_side]).max()),
                     "front": float(P[mid, axis_depth].min()) if mid.any() else float(P[:, axis_depth].min()),
                     "back": float(P[mid, axis_depth].max()) if mid.any() else float(P[:, axis_depth].max())})
    return rows


def relief(st, iterations=20):
    """Signed height of each vertex above a heavily smoothed copy (protrusion > 0, depression < 0)."""
    sm = st.co.copy()
    idx = np.arange(len(sm))
    for _ in range(iterations):
        sm = sm + 0.6 * (neighbor_mean(sm, idx, st.ptr, st.nbr) - sm)
    return np.einsum("ij,ij->i", st.co - sm, st.normals)


def features(st, k=6, min_sep=None):
    """Top protrusions and depressions as compact landmarks."""
    h = relief(st)
    diag = float(np.linalg.norm(st.co.max(0) - st.co.min(0)))
    min_sep = min_sep or diag * 0.08

    def pick(order):
        out = []
        for i in order:
            p = st.co[i]
            if all(np.linalg.norm(p - st.co[j]) > min_sep for j in out):
                out.append(int(i))
            if len(out) >= k:
                break
        return [{"vertex": j, "position": st.co[j].round(4).tolist(), "height": round(float(h[j]), 5)} for j in out]
    return {"protrusions": pick(np.argsort(-h)), "depressions": pick(np.argsort(h))}


def connected_components(st):
    n = len(st.co)
    lab = -np.ones(n, dtype=np.int64)
    c = 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        stack = [s]
        lab[s] = c
        while stack:
            v = stack.pop()
            for u in st.nbr[st.ptr[v]:st.ptr[v + 1]]:
                if lab[u] < 0:
                    lab[u] = c
                    stack.append(int(u))
        c += 1
    return c, np.bincount(lab).tolist()


def analyze_mesh(obj, detail=False):
    st = MeshState(obj)
    co = st.co
    bmin, bmax = co.min(0), co.max(0)
    res = {"object": obj.name, "vertices": len(co), "faces": len(obj.data.polygons), "triangles": len(st.tris),
           "bbox_min": bmin.round(4).tolist(), "bbox_max": bmax.round(4).tolist(),
           "dimensions": (bmax - bmin).round(4).tolist(), "center": ((bmin + bmax) / 2).round(4).tolist(),
           "centroid": co.mean(0).round(4).tolist(), "surface_area": round(surface_area(co, st.tris), 5),
           "volume": round(volume(co, st.tris), 5), "mean_edge_length": round(st.mean_edge, 5),
           "origin_tag": obj.get("bnm_origin")}
    if detail:
        bvh = bvh_from_state(st)
        res["symmetry_x"] = symmetry_error(st, bvh)
        ncomp, sizes = connected_components(st)
        res["connected_regions"] = {"count": ncomp, "sizes": sizes[:10]}
        res["features"] = features(st)
        from .curvature import curvature_stats
        res["curvature"] = curvature_stats(st)
    return res
