"""Surface-distance (geodesic) brush support.

Dijkstra on the mesh edge graph, restricted to the Euclidean ball (geodesic >= Euclidean, so
nothing outside the ball can be within the geodesic radius).  This stops a stroke on the upper
lip from leaking into the lower lip, or an eyelid stroke from leaking across the eye opening.
"""
import heapq
import numpy as np


def geodesic_ball(co, ptr, nbr, center, radius, euclid_d=None, allowed=None):
    """Return (indices, geodesic_distances) of vertices within `radius` along the surface."""
    d = euclid_d if euclid_d is not None else np.linalg.norm(co - center, axis=1)
    cand = d < radius
    if allowed is not None:
        cand &= allowed
    cidx = np.nonzero(cand)[0]
    if len(cidx) == 0:
        return cidx, np.zeros(0)
    seed = int(cidx[np.argmin(d[cidx])])
    dist = {seed: float(d[seed])}
    heap = [(float(d[seed]), seed)]
    done = set()
    while heap:
        dv, v = heapq.heappop(heap)
        if v in done:
            continue
        done.add(v)
        cv = co[v]
        for u in nbr[ptr[v]:ptr[v + 1]]:
            u = int(u)
            if not cand[u] or u in done:
                continue
            nd = dv + float(np.sqrt(((co[u] - cv) ** 2).sum()))
            if nd < radius and nd < dist.get(u, 1e30):
                dist[u] = nd
                heapq.heappush(heap, (nd, u))
    idx = np.fromiter(dist.keys(), dtype=np.int64, count=len(dist))
    gd = np.fromiter(dist.values(), dtype=np.float64, count=len(dist))
    return idx, gd


def connected_component(ptr, nbr, seed, allowed):
    """Flood fill from seed through `allowed` vertices (Connected Region Mask)."""
    out = np.zeros(len(allowed), dtype=bool)
    if not allowed[seed]:
        return out
    stack = [int(seed)]
    out[seed] = True
    while stack:
        v = stack.pop()
        for u in nbr[ptr[v]:ptr[v + 1]]:
            if allowed[u] and not out[u]:
                out[u] = True
                stack.append(int(u))
    return out


def geodesic_field(co, ptr, nbr, seeds, max_dist=np.inf):
    """Multi-source Dijkstra distance field over the whole mesh (used for region growing)."""
    n = len(co)
    dist = np.full(n, np.inf)
    heap = []
    for s in np.atleast_1d(seeds):
        dist[int(s)] = 0.0
        heap.append((0.0, int(s)))
    heapq.heapify(heap)
    while heap:
        dv, v = heapq.heappop(heap)
        if dv > dist[v] or dv > max_dist:
            continue
        for u in nbr[ptr[v]:ptr[v + 1]]:
            nd = dv + float(np.linalg.norm(co[u] - co[v]))
            if nd < dist[u]:
                dist[u] = nd
                heapq.heappush(heap, (nd, int(u)))
    return dist
