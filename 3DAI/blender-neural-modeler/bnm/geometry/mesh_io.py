"""Fast numpy <-> Blender mesh transfer and adjacency structures."""
import numpy as np
import bpy


def get_coords(obj):
    me = obj.data
    n = len(me.vertices)
    a = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", a)
    return a.reshape(n, 3).astype(np.float64)


def set_coords(obj, co):
    me = obj.data
    me.vertices.foreach_set("co", np.ascontiguousarray(co, dtype=np.float32).ravel())
    me.update()


def get_triangles(me):
    me.calc_loop_triangles()
    n = len(me.loop_triangles)
    t = np.empty(n * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", t)
    return t.reshape(n, 3).astype(np.int64)


def get_edges(me):
    n = len(me.edges)
    e = np.empty(n * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", e)
    return e.reshape(n, 2).astype(np.int64)


def get_polygons(me):
    n = len(me.polygons)
    ls = np.empty(n, dtype=np.int32)
    lt = np.empty(n, dtype=np.int32)
    me.polygons.foreach_get("loop_start", ls)
    me.polygons.foreach_get("loop_total", lt)
    lv = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", lv)
    return ls, lt, lv


def build_csr(n, edges):
    if len(edges) == 0:
        return np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(n, dtype=np.int64)
    a = np.concatenate([edges[:, 0], edges[:, 1]])
    b = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    deg = np.bincount(a, minlength=n).astype(np.int64)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=ptr[1:])
    return ptr, b.astype(np.int64), deg


def face_normals(co, tris):
    v0 = co[tris[:, 0]]
    return np.cross(co[tris[:, 1]] - v0, co[tris[:, 2]] - v0)


def vertex_normals(co, tris):
    fn = face_normals(co, tris)
    vn = np.zeros_like(co)
    n = len(co)
    for k in range(3):
        for c in range(3):
            vn[:, c] += np.bincount(tris[:, k], weights=fn[:, c], minlength=n)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return vn / ln


def gather_neighbors(idx, ptr, nbr):
    """Vectorised neighbour gather for a vertex subset. Returns (owner_pos, neighbour_idx, counts)."""
    idx = np.asarray(idx, dtype=np.int64)
    starts = ptr[idx]
    counts = ptr[idx + 1] - starts
    total = int(counts.sum())
    owner = np.repeat(np.arange(len(idx)), counts)
    offs = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    flat = nbr[np.repeat(starts, counts) + offs]
    return owner, flat, counts


def neighbor_mean(co, idx, ptr, nbr, values=None):
    """Umbrella-operator mean of neighbours for vertices idx (values defaults to co)."""
    vals = co if values is None else values
    owner, flat, counts = gather_neighbors(idx, ptr, nbr)
    m = len(idx)
    out = np.zeros((m, vals.shape[1]))
    for c in range(vals.shape[1]):
        out[:, c] = np.bincount(owner, weights=vals[flat, c], minlength=m)
    safe = np.maximum(counts, 1)[:, None]
    out = out / safe
    zero = counts == 0
    if zero.any():
        out[zero] = vals[np.asarray(idx)[zero]]
    return out


class MeshState:
    """Cached numpy view of a mesh object (object space)."""

    def __init__(self, obj):
        self.obj = obj
        self.refresh_topology()

    def topology_key(self):
        me = self.obj.data
        return (len(me.vertices), len(me.edges), len(me.polygons))

    def refresh_topology(self):
        me = self.obj.data
        self.co = get_coords(self.obj)
        self.tris = get_triangles(me)
        self.edges = get_edges(me)
        self.ptr, self.nbr, self.deg = build_csr(len(self.co), self.edges)
        self.key = self.topology_key()
        self.update_normals()
        if len(self.edges):
            el = np.linalg.norm(self.co[self.edges[:, 0]] - self.co[self.edges[:, 1]], axis=1)
            self.mean_edge = float(el.mean())
        else:
            self.mean_edge = 0.0

    def sync(self):
        """Re-read the mesh (after external edits). Rebuilds topology if it changed."""
        if self.topology_key() != self.key:
            self.refresh_topology()
        else:
            self.co = get_coords(self.obj)
            self.update_normals()

    def update_normals(self):
        self.normals = vertex_normals(self.co, self.tris)

    def write(self):
        set_coords(self.obj, self.co)

    def __len__(self):
        return len(self.co)


def new_mesh_object(name, verts, faces, collection=None, origin_tag=None):
    """Create a mesh object from numpy verts (N,3) and faces (list or (F,k) array)."""
    me = bpy.data.meshes.new(name)
    verts = np.asarray(verts, dtype=np.float32)
    if isinstance(faces, np.ndarray) and faces.ndim == 2:
        k = faces.shape[1]
        nf = len(faces)
        me.vertices.add(len(verts))
        me.vertices.foreach_set("co", verts.ravel())
        me.loops.add(nf * k)
        me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
        me.polygons.add(nf)
        me.polygons.foreach_set("loop_start", (np.arange(nf, dtype=np.int32) * k))
        me.update(calc_edges=True)
    else:
        me.from_pydata(verts.tolist(), [], [list(f) for f in faces])
        me.update(calc_edges=True)
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    if origin_tag:
        ob["bnm_origin"] = origin_tag
    return ob
