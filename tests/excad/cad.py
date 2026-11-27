# SPDX-License-Identifier: GPL-2.0-or-later
import json
import os
from pathlib import Path

import bmesh
import bpy

from excad_addon.ai_bridge import AiBridgeServer
from excad_addon.sketch_engine import SketchDocument
from excad_addon.sketch_overlay import constrained_variables

bridge = AiBridgeServer.instance()
for obj in list(bpy.context.scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)


def run(cmd, **params):
    result = bridge._dispatch_dict(dict(cmd=cmd, **params))
    assert result["ok"], (cmd, result)
    return result


def volume(name):
    obj = bpy.data.objects[name]
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    assert bm.faces and bm.verts and all(e.is_manifold for e in bm.edges) and all(v.is_manifold for v in bm.verts), name
    result = abs(bm.calc_volume(signed=True))
    bm.free()
    return result


def sketch(name="profile", **args):
    run("sketch_new", sketch_id=name, **args)


def tree(name, sketches, features):
    return run("feature_tree", name=name, recipe={"version": 1, "sketches": sketches, "features": features})


sketch()
run("sketch_arc", points=[[1,0], [0,1], [-1,0]])
run("sketch_add_line", u0=-1,v0=0,u1=1,v1=0)
run("extrude", name="Half", depth=1)
assert abs(volume("Half")-1.5708) < 0.005
sketch()
run("sketch_arc", center=[0,0], radius=1, start_deg=0, angle_deg=180)
run("sketch_add_line", u0=-1,v0=0,u1=1,v1=0)
run("extrude", name="OtherHalf", depth=1, symmetric=True)
assert abs(volume("Half")-volume("OtherHalf")) < 1e-5
sketch()
run("sketch_regular_polygon", sides=6, radius=0.5, center=[1,0])
run("sketch_mirror2d", axis="Y")
run("extrude", name="Mirrored", depth=0.3)
assert volume("Mirrored") > 0.35
sketch()
run("sketch_add_rect", u0=-1,v0=-1,width=2,height=2)
run("sketch_offset_loop", distance=-0.2)
run("extrude", name="Ring", depth=1)
assert abs(volume("Ring")-(4-1.6**2)) < 1e-5
sketch()
run("sketch_spline", points=[[0,0],[0.5,-0.1],[0.8,0.3],[0.1,0.6]], closed=True)
run("extrude", name="Spline", depth=0.2, draft_angle=2)
assert volume("Spline") > 0.01

# Independent residual checks start from an incorrect solution.
for kind in ("MIDPOINT", "SYMMETRIC", "ANGLE", "TANGENT", "CONCENTRIC"):
    d = SketchDocument()
    if kind in {"TANGENT", "CONCENTRIC"}:
        if kind == "TANGENT":
            d.add_line(-2,0,2,0)
            d.points[0].fixed = d.points[1].fixed = True
            b = d.add_circle(0,0.7,1)
        else:
            d.add_circle(0,0,1)
            d.points[0].fixed=True
            b = d.add_circle(0.3,0.1,0.5)
        d.add_constraint(kind,0,b)
    else:
        for i,p in enumerate(((0.2,0.3),(-1,0),(1,0),(1,1))):
            d.add_point(*p, fixed=i>0)
        d.add_constraint(kind,0,1,2,3,value=45)
    solved, _, residual = d.solve()
    assert solved and residual < 1e-6, (kind, residual)

sketch()
run("sketch_add_circle", radius=0.5)
run("sketch_constraint", ctype="FIXED", a=0)
assert constrained_variables(SketchDocument.active()) == {0,1,2}

rect = {"shapes":[{"type":"RECT","width":3,"height":1.8}]}
tree("Housing", {"base":rect}, [{"id":"base","type":"EXTRUDE","sketch":"base","depth":0.65}])
for side in ("TOP","BOTTOM","FRONT","BACK","LEFT","RIGHT","LARGEST_PLANAR"):
    assert run("sketch_on_face", object="Housing", face=side, sketch_id="opening")["plane"] == "FACE"
run("sketch_on_face", object="Housing", face="TOP", sketch_id="opening")
run("sketch_slot", start=[-0.65,0], end=[0.65,0], width=0.48)
run("extrude", object="Housing", op="CUT", depth=-0.8)
v = volume("Housing")
assert abs(v - (3*1.8 - 1.3*0.48 - 3.14159*0.24**2)*0.65) < 0.001
run("fillet", object="Housing", edges="TOP", radius=0.065)
run("chamfer", object="Housing", edges="BOTTOM", width=0.035)
assert volume("Housing") < v
bpy.context.view_layer.update()
assert not bpy.data.objects["Housing"].ray_cast((0, 0, 2), (0, 0, -1))[0], "Slot roof obstructs through-cut"
out = Path(os.getenv("EXCAD_TEST_OUTPUT", bpy.app.tempdir))
out.mkdir(exist_ok=True, parents=True)
probe = run("render_inspect", target="Housing", path=str(out/"phase1-cad.png"))
assert not probe["warnings"], probe
print("CAD INSPECTION", probe)
recipe_path = Path(bpy.app.tempdir)/"feature_tree.json"
run("save_feature_tree", object="Housing", path=str(recipe_path))
before = volume("Housing")
run("rebuild_feature_tree", object="Housing", path=str(recipe_path), parameters={"base.depth":0.72})
assert volume("Housing") > before
bad = bridge._dispatch_dict({"cmd":"rebuild_feature_tree","object":"Housing","parameters":{"base.depth":0}})
assert not bad["ok"] and volume("Housing") > before
sketch()
run("sketch_add_rect", u0=0,v0=-1,width=2,height=2)
run("extrude", object="Housing", depth=2, op="INTERSECT")
assert 0 < volume("Housing") < before

tree("Case", {"base":rect}, [{"type":"EXTRUDE","sketch":"base","depth":1},
                             {"type":"SHELL","thickness":0.1,"open_faces":["TOP"]}])
assert 0 < volume("Case") < 2
sketch("bottom")
run("sketch_add_circle", radius=0.4)
sketch("top", offset=1)
run("sketch_regular_polygon", sides=6, radius=0.65)
run("loft", name="Vase", sketch_ids=["bottom","top"])
assert volume("Vase") > 0.25
sketch("profile", plane="XZ")
run("sketch_add_rect", u0=0,v0=0,width=0.7,height=0.35)
run("revolve", name="Dial", axis="Z", angle_deg=270, segments=40)
assert volume("Dial") > 0.35
sketch()
run("sketch_add_circle", radius=0.1)
curve = bpy.data.curves.new("HandlePath","CURVE")
curve.dimensions = "3D"
sp = curve.splines.new("POLY")
sp.points.add(3)
for p,co in zip(sp.points, [(0,0,0),(0,0,0.5),(1,0,0.5),(1,0,0)]):
    p.co = (*co,1)
path_obj = bpy.data.objects.new("HandlePath", curve)
bpy.context.collection.objects.link(path_obj)
run("sweep", name="Handle", profile_sketch="profile", path_curve=path_obj.name)
assert volume("Handle") > 0.04
sketch()
run("sketch_add_circle", cu=1, radius=0.10)
run("extrude", name="Bolts", depth=0.2)
unit = volume("Bolts")
run("pattern", object="Bolts", type="CIRCULAR", axis="Z", count=4)
run("pattern", object="Bolts", type="LINEAR", axis="X", spacing=3, count=2)
assert abs(volume("Bolts")/unit - 8) < 0.01
run("mirror3d", object="Bolts", axis="Y", offset=3)
assert abs(volume("Bolts")/unit - 16) < 0.02
assert not any(o.name.startswith("EXCAD_Construction") for o in bpy.data.objects)
assert len(bridge._dispatch('{"cmd":"schema"}').encode()) <= 1200
print("CAD PASS", bpy.app.version_string)
