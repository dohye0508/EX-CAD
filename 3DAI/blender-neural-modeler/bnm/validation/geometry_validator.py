"""Measurable mesh-quality checks (kept strictly separate from qualitative judgement)."""
import numpy as np
import bmesh

from ..geometry.mesh_io import MeshState


def validate_mesh(obj, check_self_intersection=True, allow_open_boundary=False):
    me = obj.data
    st = MeshState(obj)
    co = st.co
    bm = bmesh.new()
    bm.from_mesh(me)
    nonmanifold = sum(1 for e in bm.edges if not e.is_manifold)
    boundary = sum(1 for e in bm.edges if e.is_boundary)
    wire = sum(1 for e in bm.edges if e.is_wire)
    loose_verts = sum(1 for v in bm.verts if not v.link_edges)
    degenerate = sum(1 for f in bm.faces if f.calc_area() < 1e-12)
    # inconsistent winding: manifold edge whose two faces traverse it in the same direction
    flipped = 0
    for e in bm.edges:
        if len(e.link_loops) == 2:
            l0, l1 = e.link_loops
            if l0.vert == l1.vert:
                flipped += 1
    bm.free()
    finite = bool(np.isfinite(co).all())
    # duplicate vertices (quantised)
    q = np.round(co / max(st.mean_edge * 1e-3, 1e-9)).astype(np.int64)
    _, cnt = np.unique(q, axis=0, return_counts=True)
    duplicates = int((cnt - 1).sum())
    tris = st.tris
    v0, v1, v2 = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    vol = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0) if finite else float("nan")
    res = {"vertices": len(co), "faces": len(me.polygons), "finite_coordinates": finite,
           "non_manifold_edges": nonmanifold, "boundary_edges": boundary, "wire_edges": wire,
           "loose_vertices": loose_verts, "degenerate_faces": degenerate, "inconsistent_winding_edges": flipped,
           "duplicate_vertices": duplicates, "signed_volume": vol, "inside_out": bool(finite and vol < 0)}
    if check_self_intersection and finite and len(co) < 400000:
        from mathutils.bvhtree import BVHTree
        bvh = BVHTree.FromPolygons(co.tolist(), tris.tolist(), all_triangles=True)
        pairs = bvh.overlap(bvh)
        # exclude pairs sharing a vertex
        bad = 0
        for a, b in pairs:
            if a < b and not (set(tris[a]) & set(tris[b])):
                bad += 1
        res["self_intersecting_face_pairs"] = bad
    problems = []
    if not finite:
        problems.append("non-finite coordinates")
    if nonmanifold - (boundary if allow_open_boundary else 0) > 0:
        problems.append("non-manifold edges")
    if degenerate:
        problems.append("degenerate faces")
    if flipped:
        problems.append("inconsistent normals")
    if res.get("inside_out"):
        problems.append("inside-out (negative volume)")
    if res.get("self_intersecting_face_pairs", 0) > max(10, len(tris) * 0.002):
        problems.append("self-intersections")
    res["problems"] = problems
    res["ok"] = not problems
    return res
