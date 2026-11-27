"""Resolution management: voxel remesh, subdivision, SDF remesh, primitives.

Policy: never raise density while primary-form errors remain; pick ONE of voxel remesh /
dyntopo / multires per phase.  Remesh changes vertex indices, so callers must rebuild
semantic regions and masks afterwards (SemanticMap.rebuild()).
"""
import bpy
import numpy as np

from ..geometry.mesh_io import MeshState, get_coords, new_mesh_object
from ..geometry import sdf as SDF

QUALITY = {  # phase -> target vertex range (guideline, not a rule)
    "blocking": (1000, 10000), "primary": (10000, 50000), "secondary": (50000, 250000),
    "detail": (250000, 1000000),
}


def create_primitive(kind="ICOSPHERE", name="BNM_Base", size=1.0, location=(0, 0, 0), detail=5, log=None):
    kind = kind.upper()
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    ops = {
        "ICOSPHERE": lambda: bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=detail, radius=size, location=location),
        "UV_SPHERE": lambda: bpy.ops.mesh.primitive_uv_sphere_add(segments=16 * detail, ring_count=8 * detail, radius=size, location=location),
        "CUBE": lambda: bpy.ops.mesh.primitive_cube_add(size=2 * size, location=location),
        "CYLINDER": lambda: bpy.ops.mesh.primitive_cylinder_add(vertices=8 * detail, radius=size, depth=2 * size, location=location),
        "PLANE": lambda: bpy.ops.mesh.primitive_plane_add(size=2 * size, location=location),
        "GRID": lambda: bpy.ops.mesh.primitive_grid_add(x_subdivisions=20 * detail, y_subdivisions=20 * detail, size=2 * size, location=location),
    }
    if kind not in ops:
        raise ValueError(f"primitive {kind} not supported: {sorted(ops)}")
    ops[kind]()
    ob = bpy.context.active_object
    ob.name = name
    ob.data.name = name
    ob["bnm_origin"] = f"primitive:{kind.lower()}"
    if log:
        log({"op": "create_primitive", "kind": kind, "name": name, "vertices": len(ob.data.vertices), "size": size})
    return ob


def _replace_mesh(obj, verts, faces):
    me_old = obj.data
    tmp = new_mesh_object("_bnm_tmp", verts, faces)
    me_new = tmp.data
    bpy.data.objects.remove(tmp)
    me_new.name = me_old.name
    for mat in me_old.materials:
        me_new.materials.append(mat)
    obj.data = me_new
    if me_old.users == 0:
        bpy.data.meshes.remove(me_old)
    obj["bnm_origin"] = "derived:" + str(obj.get("bnm_origin", "unknown")).replace("derived:", "")


def voxel_remesh(obj, voxel_size, adaptivity=0.0, log=None):
    """Voxel remesh via a REMESH modifier evaluated through the depsgraph (works headless)."""
    n0 = len(obj.data.vertices)
    mod = obj.modifiers.new("BNM_Remesh", "REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = voxel_size
    mod.adaptivity = adaptivity
    mod.use_smooth_shade = True
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, depsgraph=dg)
    obj.modifiers.remove(mod)
    old = obj.data
    me.name = old.name
    obj.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)
    obj["bnm_origin"] = "derived:" + str(obj.get("bnm_origin", "unknown")).replace("derived:", "")
    if log:
        log({"op": "voxel_remesh", "voxel_size": voxel_size, "vertices_before": n0, "vertices_after": len(me.vertices)})
    return len(me.vertices)


def voxel_size_for(obj, target_vertices):
    st = MeshState(obj)
    from ..geometry.analyzer import surface_area
    A = surface_area(st.co, st.tris)
    return float(np.sqrt(A / max(target_vertices, 1)))


def subdivide(obj, levels=1, simple=False, log=None):
    n0 = len(obj.data.vertices)
    mod = obj.modifiers.new("BNM_Subsurf", "SUBSURF")
    mod.levels = levels
    mod.render_levels = levels
    mod.subdivision_type = "SIMPLE" if simple else "CATMULL_CLARK"
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg), depsgraph=dg)
    obj.modifiers.remove(mod)
    old = obj.data
    me.name = old.name
    obj.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)
    if log:
        log({"op": "subdivide", "levels": levels, "vertices_before": n0, "vertices_after": len(me.vertices)})
    return len(me.vertices)


def adaptive_subdivide(obj, region_idx, cuts=1, log=None):
    """Local subdivision of faces fully inside a vertex region (bmesh)."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    sel = set(int(i) for i in region_idx)
    edges = [e for e in bm.edges if e.verts[0].index in sel and e.verts[1].index in sel]
    bmesh.ops.subdivide_edges(bm, edges=edges, cuts=cuts, use_grid_fill=True, smooth=1.0)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    n0 = len(obj.data.vertices)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    if log:
        log({"op": "adaptive_subdivide", "edges": len(edges), "vertices_before": n0, "vertices_after": len(obj.data.vertices)})


def set_multiresolution(obj, levels=2, log=None):
    mod = obj.modifiers.get("BNM_Multires") or obj.modifiers.new("BNM_Multires", "MULTIRES")
    with bpy.context.temp_override(object=obj, active_object=obj):
        for _ in range(levels):
            bpy.ops.object.multires_subdivide(modifier=mod.name, mode="CATMULL_CLARK")
    if log:
        log({"op": "set_multiresolution", "levels": levels})
    return mod.total_levels


def sdf_remesh(obj, voxel, edit_fn=None, log=None):
    """Mesh -> SDF -> (optional volume edits) -> zero level set -> mesh.  Repairs self-intersections."""
    st = MeshState(obj)
    g = SDF.mesh_to_sdf(st.co, st.tris, voxel, pad=6)
    if edit_fn:
        edit_fn(g)
    v, q = g.extract()
    n0 = len(st.co)
    _replace_mesh(obj, v, q)
    if log:
        log({"op": "sdf_remesh", "voxel": voxel, "vertices_before": n0, "vertices_after": len(obj.data.vertices)})
    return g
