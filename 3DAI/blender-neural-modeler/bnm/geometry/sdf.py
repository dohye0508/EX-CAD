"""SDF-based sculpt assist engine (inspired by TripoSG's SDF latent representation).

- mesh -> narrow-band signed distance grid via OpenVDB (bundled with Blender) or a BVH fallback
- analytic SDF brushes (sphere, capsule chain, ellipsoid, box) composed with smooth CSG
- zero level-set extraction: OpenVDB convertToQuads or a numpy Naive Surface Nets extractor
SDF is a volume tool for adding/removing clay and repairing surfaces, *not* a face generator.
"""
import numpy as np

try:
    import openvdb as vdb
except Exception:  # pragma: no cover
    vdb = None


# ------------------------------------------------------------ analytic SDFs (numpy, x: (...,3))
def sd_sphere(x, c, r):
    return np.linalg.norm(x - np.asarray(c, float), axis=-1) - r


def sd_ellipsoid(x, c, radii):
    # Inigo Quilez' bound-corrected ellipsoid approximation
    p = (x - np.asarray(c, float))
    rr = np.asarray(radii, float)
    k0 = np.linalg.norm(p / rr, axis=-1)
    k1 = np.linalg.norm(p / (rr * rr), axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-12)


def sd_box(x, c, half, radius=0.0):
    q = np.abs(x - np.asarray(c, float)) - (np.asarray(half, float) - radius)
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0) - radius


def sd_capsule(x, a, b, ra, rb=None):
    """Round cone / capsule from a (radius ra) to b (radius rb)."""
    rb = ra if rb is None else rb
    a, b = np.asarray(a, float), np.asarray(b, float)
    ba = b - a
    pa = x - a
    h = np.clip((pa @ ba) / max(ba @ ba, 1e-12), 0, 1)
    return np.linalg.norm(pa - h[..., None] * ba, axis=-1) - (ra + (rb - ra) * h)


def sd_path(x, path, radii):
    """Union of capsules along a brush path (a clay 'snake' along the stroke)."""
    path = np.asarray(path, float)
    radii = np.broadcast_to(np.asarray(radii, float), (len(path),))
    d = np.full(x.shape[:-1], np.inf)
    if len(path) == 1:
        return sd_sphere(x, path[0], radii[0])
    for i in range(len(path) - 1):
        d = np.minimum(d, sd_capsule(x, path[i], path[i + 1], radii[i], radii[i + 1]))
    return d


# ------------------------------------------------------------ smooth CSG (polynomial smin)
def smooth_union(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smooth_difference(a, b, k):
    """a minus b."""
    if k <= 0:
        return np.maximum(a, -b)
    h = np.clip(0.5 - 0.5 * (a + b) / k, 0, 1)
    return a * (1 - h) + (-b) * h + k * h * (1 - h)


def smooth_intersection(a, b, k):
    if k <= 0:
        return np.maximum(a, b)
    h = np.clip(0.5 - 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h + k * h * (1 - h)


# ------------------------------------------------------------ grid container
class SDFGrid:
    """Dense float grid with world mapping x = origin + ijk * voxel."""

    def __init__(self, values, origin, voxel):
        self.v = np.asarray(values, dtype=np.float32)
        self.origin = np.asarray(origin, float)
        self.voxel = float(voxel)

    @classmethod
    def empty(cls, bmin, bmax, voxel, fill=1e3):
        bmin, bmax = np.asarray(bmin, float), np.asarray(bmax, float)
        dims = np.maximum(np.ceil((bmax - bmin) / voxel).astype(int) + 1, 2)
        return cls(np.full(dims, fill, dtype=np.float32), bmin, voxel)

    def points(self, sub=None):
        nx, ny, nz = self.v.shape
        ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        return self.origin + np.stack([ii, jj, kk], -1) * self.voxel

    def apply(self, fn, bmin=None, bmax=None):
        """Evaluate fn(points, current_values)->new_values only inside an AABB (sparse update)."""
        if bmin is None:
            P = self.points()
            self.v = fn(P, self.v.astype(np.float64)).astype(np.float32)
            return
        lo = np.clip(np.floor((np.asarray(bmin) - self.origin) / self.voxel).astype(int), 0, np.array(self.v.shape) - 1)
        hi = np.clip(np.ceil((np.asarray(bmax) - self.origin) / self.voxel).astype(int) + 1, 1, np.array(self.v.shape))
        sl = tuple(slice(l, h) for l, h in zip(lo, hi))
        ii, jj, kk = np.meshgrid(*[np.arange(s.start, s.stop) for s in sl], indexing="ij")
        P = self.origin + np.stack([ii, jj, kk], -1) * self.voxel
        self.v[sl] = fn(P, self.v[sl].astype(np.float64)).astype(np.float32)

    # brush-style volume edits ------------------------------------------------
    def add_path(self, path, radius, k=None):
        path = np.asarray(path, float)
        r = np.max(np.atleast_1d(radius))
        k = r * 0.6 if k is None else k
        pad = r + k + 2 * self.voxel
        self.apply(lambda P, V: smooth_union(V, sd_path(P, path, radius), k), path.min(0) - pad, path.max(0) + pad)

    def remove_path(self, path, radius, k=None):
        path = np.asarray(path, float)
        r = np.max(np.atleast_1d(radius))
        k = r * 0.4 if k is None else k
        pad = r + k + 2 * self.voxel
        self.apply(lambda P, V: smooth_difference(V, sd_path(P, path, radius), k), path.min(0) - pad, path.max(0) + pad)

    def union_sdf(self, fn, k, bmin=None, bmax=None):
        self.apply(lambda P, V: smooth_union(V, fn(P), k), bmin, bmax)

    def offset(self, amount):
        """Positive amount inflates the surface, negative shrinks it."""
        self.v = self.v - np.float32(amount)

    def blur(self, iterations=1):
        """Local surface reconstruction: 3x3x3 box filter of the field (keeps zero set smooth)."""
        v = self.v
        for _ in range(iterations):
            p = np.pad(v, 1, mode="edge")
            acc = np.zeros_like(v)
            for dx in (0, 1, 2):
                for dy in (0, 1, 2):
                    for dz in (0, 1, 2):
                        acc += p[dx:dx + v.shape[0], dy:dy + v.shape[1], dz:dz + v.shape[2]]
            v = acc / 27.0
        self.v = v.astype(np.float32)

    # extraction --------------------------------------------------------------
    def extract(self, method="auto"):
        if method in ("auto", "vdb") and vdb is not None:
            v, q = _extract_vdb(self)
        else:
            v, q = surface_nets(self.v, self.origin, self.voxel)
        return v, orient_outward(v, q)


def orient_outward(verts, quads):
    """Guarantee outward winding: positive signed volume of the closed surface."""
    if len(quads) == 0:
        return quads
    t = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    a, b, c = verts[t[:, 0]], verts[t[:, 1]], verts[t[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    return quads[:, ::-1].copy() if vol < 0 else quads


# ------------------------------------------------------------ mesh -> SDF
def mesh_to_sdf(verts, tris, voxel, pad=4, half_width=None, bounds=None):
    """Narrow-band signed distance grid from a closed mesh (object space).

    Uses OpenVDB createLevelSetFromPolygons when available (robust inside/outside),
    dense numpy copy, outside = +, inside = -.
    """
    verts = np.asarray(verts, float)
    bmin = verts.min(0) - pad * voxel
    bmax = verts.max(0) + pad * voxel
    if bounds is not None:
        bmin = np.minimum(bmin, np.asarray(bounds[0], float))
        bmax = np.maximum(bmax, np.asarray(bounds[1], float))
    if vdb is None:
        return _mesh_to_sdf_bvh(verts, tris, voxel, bmin, bmax)
    hw = half_width or max(3, pad)
    xf = vdb.createLinearTransform(voxelSize=voxel)
    grid = vdb.FloatGrid.createLevelSetFromPolygons(np.ascontiguousarray(verts, dtype=np.float32),
                                                     triangles=np.ascontiguousarray(tris, dtype=np.int32),
                                                     transform=xf, halfWidth=hw)
    lo = np.floor(bmin / voxel).astype(int)
    hi = np.ceil(bmax / voxel).astype(int)
    dims = hi - lo + 1
    arr = np.empty(tuple(dims), dtype=np.float32)
    grid.copyToArray(arr, ijk=tuple(int(x) for x in lo))
    return SDFGrid(arr, lo * voxel, voxel)


def _mesh_to_sdf_bvh(verts, tris, voxel, bmin, bmax):
    from mathutils.bvhtree import BVHTree
    from mathutils import Vector
    bvh = BVHTree.FromPolygons(verts.tolist(), tris.tolist())
    g = SDFGrid.empty(bmin, bmax, voxel)
    P = g.points().reshape(-1, 3)
    out = np.empty(len(P), dtype=np.float32)
    for i, p in enumerate(P):
        loc, nrm, _, dist = bvh.find_nearest(Vector(p))
        s = 1.0 if (Vector(p) - loc).dot(nrm) >= 0 else -1.0
        out[i] = s * dist
    g.v = out.reshape(g.v.shape)
    return g


def _extract_vdb(g, iso=0.0, adaptivity=0.0):
    xf = vdb.createLinearTransform(voxelSize=g.voxel)
    grid = vdb.FloatGrid(background=float(3 * g.voxel))
    grid.transform = xf
    grid.gridClass = vdb.GridClass.LEVEL_SET
    lo = np.round(g.origin / g.voxel).astype(int)
    # copyFromArray writes into index space starting at ijk; clamp far values to keep it narrow band
    arr = np.clip(g.v, -3 * g.voxel, 3 * g.voxel).astype(np.float32)
    grid.copyFromArray(arr, ijk=tuple(int(x) for x in lo), tolerance=0.0)
    pts, quads = grid.convertToQuads(isovalue=iso)
    pts = np.asarray(pts, dtype=np.float64)
    quads = np.asarray(quads, dtype=np.int64)
    # OpenVDB emits outward quads with clockwise winding for Blender; flip to CCW
    return pts, quads[:, ::-1].copy()


def surface_nets(V, origin, voxel):
    """Naive Surface Nets (Gibson 1998) on a dense grid, fully vectorised. Returns (verts, quads)."""
    V = np.asarray(V, dtype=np.float64)
    nx, ny, nz = V.shape
    inside = V < 0
    corners = np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)])
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    cv = [V[c[0]:nx - 1 + c[0], c[1]:ny - 1 + c[1], c[2]:nz - 1 + c[2]] for c in corners]
    cin = [x < 0 for x in cv]
    anyin = np.zeros_like(cin[0])
    allin = np.ones_like(cin[0])
    for x in cin:
        anyin |= x
        allin &= x
    active = anyin & ~allin
    ai = np.argwhere(active)
    cell_index = -np.ones(active.shape, dtype=np.int64)
    cell_index[active] = np.arange(len(ai))
    acc = np.zeros((len(ai), 3))
    cnt = np.zeros(len(ai))
    for (a, b) in edges:
        va, vb = cv[a][active], cv[b][active]
        cross = (va < 0) != (vb < 0)
        t = np.where(cross, va / np.where(np.abs(va - vb) < 1e-12, 1e-12, va - vb), 0.0)
        p = corners[a] + t[:, None] * (corners[b] - corners[a])
        acc += p * cross[:, None]
        cnt += cross
    verts = origin + (ai + acc / np.maximum(cnt, 1)[:, None]) * voxel
    quads = []
    # for every grid edge with a sign change, connect the 4 surrounding cells
    for axis in range(3):
        o1, o2 = [a for a in range(3) if a != axis]
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[axis] = slice(0, V.shape[axis] - 1)
        sl1[axis] = slice(1, V.shape[axis])
        s0, s1 = inside[tuple(sl0)], inside[tuple(sl1)]
        ch = s0 != s1
        idx = np.argwhere(ch)
        # need 1 <= idx[o] <= dim-2 so that the four cells exist
        ok = (idx[:, o1] >= 1) & (idx[:, o2] >= 1) & (idx[:, o1] <= V.shape[o1] - 2) & (idx[:, o2] <= V.shape[o2] - 2)
        ok &= idx[:, axis] <= V.shape[axis] - 2
        idx = idx[ok]
        flip = s0[tuple(idx.T)]
        def cell(d1, d2):
            c = idx.copy()
            c[:, o1] -= d1
            c[:, o2] -= d2
            return cell_index[tuple(c.T)]
        q = np.stack([cell(1, 1), cell(0, 1), cell(0, 0), cell(1, 0)], 1)
        q[flip] = q[flip][:, ::-1]
        if axis == 1:
            q = q[:, ::-1]
        quads.append(q)
    quads = np.vstack(quads) if quads else np.zeros((0, 4), dtype=np.int64)
    quads = quads[(quads >= 0).all(1)]
    return verts, quads


def sdf_sample_mesh_distance(grid, points):
    """Trilinear lookup of grid values at arbitrary points (used for validation)."""
    p = (np.asarray(points, float) - grid.origin) / grid.voxel
    p = np.clip(p, 0, np.array(grid.v.shape) - 1.001)
    i = np.floor(p).astype(int)
    f = p - i
    V = grid.v
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out = out + w * V[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
    return out
