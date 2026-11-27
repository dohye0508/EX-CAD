"""Hard-surface modelling: bmesh edit operations, modifiers, curve sweeps.
Machines and architecture are NOT sculpted: they use precise edit-mode/CAD-like operations."""
import math
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix

from ..sculpt.multiresolution import create_primitive


# ------------------------------------------------------------------ selection
def _select_faces(bm, sel):
    sel = sel or {"all": True}
    faces = list(bm.faces)
    if sel.get("all"):
        return faces
    if "face_indices" in sel:
        bm.faces.ensure_lookup_table()
        return [bm.faces[i] for i in sel["face_indices"]]
    out = faces
    if "normal" in sel:
        n = Vector(sel["normal"]).normalized()
        thr = sel.get("threshold", 0.9)
        out = [f for f in out if f.normal.dot(n) >= thr]
    if "z_min" in sel:
        out = [f for f in out if f.calc_center_median().z >= sel["z_min"]]
    if "z_max" in sel:
        out = [f for f in out if f.calc_center_median().z <= sel["z_max"]]
    if "every" in sel:  # every k-th face of the remaining list, sorted by angle around Z
        out = sorted(out, key=lambda f: math.atan2(f.calc_center_median().y, f.calc_center_median().x))
        out = out[sel.get("offset", 0)::sel["every"]]
    return out


def edit_mode_op(obj, op, select=None, params=None, log=None):
    """Edit-mode style operations via bmesh (no UI needed)."""
    p = params or {}
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.normal_update()
    faces = _select_faces(bm, select)
    res = {"op": op, "selected_faces": len(faces)}
    if op == "extrude":
        r = bmesh.ops.extrude_discrete_faces(bm, faces=faces) if p.get("individual") else \
            bmesh.ops.extrude_face_region(bm, geom=faces)
        new_faces = r["faces"] if "faces" in r else [g for g in r["geom"] if isinstance(g, bmesh.types.BMFace)]
        dist = p.get("distance", 0.1)
        for f in new_faces:
            f.normal_update()
        if p.get("individual"):
            for f in new_faces:
                d = f.normal * dist
                for v in f.verts:
                    v.co += d
                if p.get("tip_scale", 1.0) != 1.0:
                    c = f.calc_center_median()
                    for v in f.verts:
                        v.co = c + (v.co - c) * p["tip_scale"]
        else:
            verts = {v for f in new_faces for v in f.verts}
            n = Vector(p["direction"]) if "direction" in p else sum((f.normal for f in new_faces), Vector()).normalized()
            for v in verts:
                v.co += n * dist
            bmesh.ops.delete(bm, geom=[f for f in faces if f.is_valid], context="FACES")
        res["new_faces"] = len(new_faces)
    elif op == "inset":
        r = bmesh.ops.inset_region(bm, faces=faces, thickness=p.get("thickness", 0.05), depth=p.get("depth", 0.0),
                                   use_even_offset=True)
        res["new_faces"] = len(r["faces"])
    elif op == "bevel":
        edges = list({e for f in faces for e in f.edges}) if select and not select.get("all") else list(bm.edges)
        if p.get("sharp_only", True):
            edges = [e for e in edges if e.is_manifold and e.calc_face_angle(0) > math.radians(p.get("angle", 30))]
        r = bmesh.ops.bevel(bm, geom=edges, offset=p.get("width", 0.02), segments=p.get("segments", 2), profile=0.5,
                            affect="EDGES", clamp_overlap=True)
        res["beveled_edges"] = len(edges)
    elif op == "loop_cut":
        # cut an edge ring: edges selected by direction (e.g. vertical edges of a cylinder)
        axis = Vector(p.get("axis", (0, 0, 1))).normalized()
        ring = [e for e in bm.edges if abs((e.verts[1].co - e.verts[0].co).normalized().dot(axis)) > 0.95]
        bmesh.ops.subdivide_edges(bm, edges=ring, cuts=p.get("cuts", 1), use_grid_fill=True)
        res["ring_edges"] = len(ring)
    elif op == "bridge":
        edges = [e for e in bm.edges if e.is_boundary]
        r = bmesh.ops.bridge_loops(bm, edges=edges)
        res["new_faces"] = len(r["faces"])
    elif op == "merge":
        r = bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=p.get("distance", 1e-4))
    elif op == "dissolve":
        bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(p.get("angle", 1.0)), verts=bm.verts, edges=bm.edges)
    elif op == "spin":
        geom = list({v for f in faces for v in f.verts}) + list({e for f in faces for e in f.edges}) + faces
        bmesh.ops.spin(bm, geom=geom, cent=Vector(p.get("center", (0, 0, 0))), axis=Vector(p.get("axis", (0, 0, 1))),
                       angle=math.radians(p.get("angle", 90)), steps=p.get("steps", 8), use_duplicate=False)
    elif op == "shrink_fatten":
        verts = {v for f in faces for v in f.verts}
        for v in verts:
            v.co += v.normal * p.get("distance", 0.01)
    elif op == "knife_plane":
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        bmesh.ops.bisect_plane(bm, geom=geom, plane_co=Vector(p.get("co", (0, 0, 0))), plane_no=Vector(p.get("no", (0, 0, 1))))
    elif op == "proportional_move":
        c = Vector(p["center"])
        rad = p.get("radius", 0.5)
        d = Vector(p["offset"])
        for v in bm.verts:
            t = (v.co - c).length / rad
            if t < 1:
                v.co += d * (1 - (3 * t * t - 2 * t ** 3))
    else:
        bm.free()
        raise ValueError(f"unknown edit op {op}")
    bm.normal_update()
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    res["vertices"] = len(obj.data.vertices)
    if log:
        log({"op": "edit_mode_op", **res})
    return res


def add_modifier(obj, mtype, settings=None, apply=False, log=None):
    m = obj.modifiers.new(f"BNM_{mtype}", mtype)
    for k, v in (settings or {}).items():
        if k == "object" and isinstance(v, str):
            v = bpy.data.objects[v]
        setattr(m, k, v)
    res = {"modifier": m.name, "type": mtype}
    if apply:
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg), depsgraph=dg)
        obj.modifiers.remove(m)
        old = obj.data
        obj.data = me
        if old.users == 0:
            bpy.data.meshes.remove(old)
        res["applied"] = True
        res["vertices"] = len(me.vertices)
    if log:
        log({"op": "add_modifier", **res})
    return res


def curve_sweep(name, points, radius=0.05, taper=None, nurbs=False, log=None):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 4
    cu.use_fill_caps = True
    if nurbs:
        sp = cu.splines.new("NURBS")
        sp.points.add(len(points) - 1)
        for i, p in enumerate(points):
            sp.points[i].co = (*p, 1.0)
            if taper:
                sp.points[i].radius = float(np.interp(i / max(len(points) - 1, 1), np.linspace(0, 1, len(taper)), taper))
        sp.use_endpoint_u = True
        sp.order_u = min(4, len(points))
    else:
        sp = cu.splines.new("BEZIER")
        sp.bezier_points.add(len(points) - 1)
        for i, p in enumerate(points):
            bp = sp.bezier_points[i]
            bp.co = p
            bp.handle_left_type = bp.handle_right_type = "AUTO"
            if taper:
                bp.radius = float(np.interp(i / max(len(points) - 1, 1), np.linspace(0, 1, len(taper)), taper))
    ob = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(ob)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), depsgraph=dg)
    mob = bpy.data.objects.new(name + "_mesh", me)
    bpy.context.scene.collection.objects.link(mob)
    bpy.data.objects.remove(ob)
    mob["bnm_origin"] = "derived:curve_sweep"
    if log:
        log({"op": "curve_sweep", "name": mob.name, "points": len(points), "vertices": len(me.vertices)})
    return mob


# ------------------------------------------------------------------ recipes
def _faces_where(ob, pred):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.faces.ensure_lookup_table()
    out = [f.index for f in bm.faces if pred(f)]
    bm.free()
    return out


def flanged_gear_hub(name="BNM_GearHub", teeth=24, radius=1.0, thickness=0.3, log=None):
    """Spur gear with raised hub, bore, 6 bolt holes and bevelled edges, from a single cylinder."""
    bpy.ops.mesh.primitive_cylinder_add(vertices=teeth * 2, radius=radius, depth=thickness)
    ob = bpy.context.active_object
    ob.name = name
    ob["bnm_origin"] = "primitive:cylinder"
    if log:
        log({"op": "create_primitive", "kind": "CYLINDER", "name": name, "vertices": len(ob.data.vertices)})
    # teeth: every second side face extruded individually and tapered
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.faces.ensure_lookup_table()
    side = [f for f in bm.faces if abs(f.normal.z) < 0.1]
    side.sort(key=lambda f: math.atan2(f.calc_center_median().y, f.calc_center_median().x))
    tooth_faces = [f.index for f in side[::2]]
    bm.free()
    edit_mode_op(ob, "extrude", {"face_indices": tooth_faces}, {"individual": True, "distance": 0.16 * radius, "tip_scale": 0.62}, log)
    # hub: inset the top cap, raise the inner face
    edit_mode_op(ob, "inset", {"normal": (0, 0, 1), "threshold": 0.99}, {"thickness": 0.45 * radius}, log)
    top = _faces_where(ob, lambda f: f.normal.z > 0.99 and f.calc_center_median().xy.length < 0.2 * radius)
    edit_mode_op(ob, "extrude", {"face_indices": top}, {"distance": 0.25 * radius, "direction": (0, 0, 1)}, log)
    # bore + bolt circle: one joined cutter, exact boolean
    cutters = []
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.18 * radius, depth=2)
    cutters.append(bpy.context.active_object)
    for k in range(6):
        a = 2 * math.pi * k / 6
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.06 * radius, depth=2,
                                            location=(0.42 * radius * math.cos(a), 0.42 * radius * math.sin(a), 0))
        cutters.append(bpy.context.active_object)
    bpy.ops.object.select_all(action="DESELECT")
    for c in cutters:
        c.select_set(True)
    bpy.context.view_layer.objects.active = cutters[0]
    bpy.ops.object.join()
    cutter = bpy.context.active_object
    cutter.name = name + "_cutter"
    cutter["bnm_origin"] = "primitive:cylinder"
    add_modifier(ob, "BOOLEAN", {"operation": "DIFFERENCE", "object": cutter.name, "solver": "EXACT"}, apply=True, log=log)
    bpy.data.objects.remove(cutter)
    edit_mode_op(ob, "bevel", None, {"width": 0.012 * radius, "segments": 2, "angle": 40}, log)
    add_modifier(ob, "WEIGHTED_NORMAL", {"keep_sharp": True}, apply=False, log=log)
    for p in ob.data.polygons:
        p.use_smooth = True
    return ob


def bracket(name="BNM_Bracket", log=None):
    """L-bracket with a bolt hole from a cube: scale-in-edit, loop cut, extrude, boolean, bevel."""
    ob = create_primitive("CUBE", name, 0.5, log=log)
    for v in ob.data.vertices:
        v.co = Vector((v.co.x, v.co.y * 0.6, v.co.z * 0.08))
    ob.data.update()
    edit_mode_op(ob, "loop_cut", None, {"axis": (1, 0, 0), "cuts": 2}, log)
    up = _faces_where(ob, lambda f: f.normal.z > 0.99 and f.calc_center_median().x > 0.3)
    edit_mode_op(ob, "extrude", {"face_indices": up}, {"distance": 0.6, "direction": (0, 0, 1)}, log)
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.07, depth=1, location=(-0.2, 0, 0))
    c = bpy.context.active_object
    c["bnm_origin"] = "primitive:cylinder"
    add_modifier(ob, "BOOLEAN", {"operation": "DIFFERENCE", "object": c.name, "solver": "EXACT"}, apply=True, log=log)
    bpy.data.objects.remove(c)
    edit_mode_op(ob, "bevel", None, {"width": 0.01, "segments": 3, "angle": 30}, log)
    return ob
