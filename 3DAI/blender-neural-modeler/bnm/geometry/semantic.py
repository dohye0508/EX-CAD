"""Semantic Geometry Map: named regions + landmarks that are *measured* on the current surface.

Nothing is assumed to exist on the initial sphere: each region starts as 'planned' (a target
location from the canon/reference) and becomes 'measured' once the stage that forms it ran.
Membership is recomputed from geometry every time (rebuild after remesh), so vertex-index
changes never break it.  Schemas are plain dicts, so the same class serves faces, animals,
creatures or machines (see anatomy/creature_schema.py).
"""
import numpy as np

from .mesh_io import MeshState
from ..sculpt.geodesic import connected_component
from ..sculpt import falloff as FO

RAY_DIRS = {"front": (0, 1, 0), "back": (0, -1, 0), "side": (-1, 0, 0), "side_R": (1, 0, 0),
            "below": (0, 0, 1), "top": (0, 0, -1)}


def ray_hit(st, origin, direction, tol=None):
    """Fast vertex-level ray cast: nearest front-facing vertex close to the ray."""
    o = np.asarray(origin, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    rel = st.co - o
    t = rel @ d
    perp = np.linalg.norm(rel - np.outer(t, d), axis=1)
    tol = tol or max(2.5 * st.mean_edge, 1e-4)
    cand = (perp < tol) & (t > 0) & ((st.normals @ d) < 0.1)
    if not cand.any():
        cand = (perp < tol * 3) & (t > 0)
        if not cand.any():
            return None
    i = np.nonzero(cand)[0]
    j = i[np.argmin(t[i] + perp[i] * 0.5)]
    return int(j)


def surface_point(st, x, z, ray="front", y=None, far=4.0):
    dvec = np.array(RAY_DIRS[ray], float)
    if ray == "front":
        o = np.array([x, -far, z])
    elif ray == "back":
        o = np.array([x, far, z])
    elif ray == "side":
        o = np.array([far, x if y is None else y, z])   # for side rays, 'x' holds y
    elif ray == "side_R":
        o = np.array([-far, x if y is None else y, z])
    elif ray == "below":
        o = np.array([x, y if y is not None else 0.0, -far])
    else:
        o = np.array([x, y if y is not None else 0.0, far])
    j = ray_hit(st, o, dvec)
    return None if j is None else (st.co[j].copy(), st.normals[j].copy(), j)


class SemanticMap:
    def __init__(self, obj, landmarks, regions, adjacency=None, directions=None, protect=None):
        self.obj = obj
        self.landmark_defs = landmarks      # name -> (a, z, ray)
        self.region_defs = regions          # name -> (landmark, radii, facing)
        self.adjacency = adjacency or {}
        self.directions = directions or {}
        self.protect = protect or {}
        self.status = {r: "planned" for r in regions}
        self.landmarks = {}
        self.st = None

    def rebuild(self, st=None):
        self.st = st or MeshState(self.obj)
        self.landmarks = {}
        for name, (a, z, ray) in self.landmark_defs.items():
            hit = surface_point(self.st, a, z, ray)
            if hit is not None:
                self.landmarks[name] = {"position": hit[0], "normal": hit[1], "vertex": hit[2]}
        return self

    def mark(self, *regions, status="measured"):
        for r in regions:
            if r in self.status:
                self.status[r] = status

    def landmark(self, name):
        if self.st is None:
            self.rebuild()
        lm = self.landmarks.get(name)
        return None if lm is None else lm["position"]

    def _sides(self, region):
        lm = self.region_defs[region][0]
        if lm and lm.endswith("_L"):
            return [lm, lm[:-2] + "_R"]
        return [lm]

    def weights(self, region, side="both", connected=True, scale=1.0):
        """Per-vertex weights (N,) in [0,1]: smooth ellipsoidal falloff around measured landmarks."""
        if self.st is None:
            self.rebuild()
        st = self.st
        w = np.zeros(len(st.co))
        lm_name, radii, facing = self.region_defs[region]
        if lm_name is None:
            return w
        names = self._sides(region)
        if side == "L":
            names = names[:1]
        elif side == "R":
            names = names[-1:]
        for nm in names:
            lm = self.landmarks.get(nm)
            if lm is None:
                continue
            c = lm["position"]
            rr = np.asarray(radii, float) * scale
            q = np.sqrt((((st.co - c) / rr) ** 2).sum(1))
            wi = FO.smooth(np.minimum(q, 1.0))
            fn = lm["normal"]
            wi *= np.clip((st.normals @ fn) * 2.0 + 0.2, 0, 1)   # same-facing surface only
            if connected:
                comp = connected_component(st.ptr, st.nbr, lm["vertex"], wi > 0)
                wi *= comp
            w = np.maximum(w, wi)
        return w

    def indices(self, region, side="both", threshold=0.05, **kw):
        return np.nonzero(self.weights(region, side, **kw) > threshold)[0]

    def info(self, region):
        from .curvature import curvature
        if self.st is None:
            self.rebuild()
        idx = self.indices(region)
        out = {"region": region, "status": self.status.get(region), "vertices": int(len(idx)),
               "adjacent": self.adjacency.get(region, []), "edit_directions": self.directions.get(region, []),
               "protect": self.protect.get(region, [])}
        names = self._sides(region) if self.region_defs[region][0] else []
        out["landmarks"] = {n: self.landmarks[n]["position"].round(4).tolist() for n in names if n in self.landmarks}
        if len(idx):
            P = self.st.co[idx]
            out["center"] = P.mean(0).round(4).tolist()
            out["bbox"] = [P.min(0).round(4).tolist(), P.max(0).round(4).tolist()]
            n = self.st.normals[idx].mean(0)
            out["mean_normal"] = (n / max(np.linalg.norm(n), 1e-9)).round(3).tolist()
            out["mean_curvature"] = round(float(curvature(self.st)["mean"][idx].mean()), 3)
        return out

    def summary(self):
        return {"landmarks": {k: v["position"].round(3).tolist() for k, v in self.landmarks.items()},
                "regions": dict(self.status)}


def face_map(obj):
    from ..anatomy import face_canon as FC
    return SemanticMap(obj, FC.expanded_landmarks(), FC.REGIONS, FC.ADJACENT, FC.EDIT_DIRECTIONS, FC.PROTECT)
