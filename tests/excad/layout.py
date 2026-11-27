# SPDX-License-Identifier: GPL-2.0-or-later
"""Exercise layout, transactions, compact deltas, hot reload and the four-tool stdio interface."""

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import addon_utils
import bpy
from mathutils import Matrix, Vector

from excad_addon.ai_bridge import AiBridgeServer
from excad_addon.scene_layout import bounds

addon_utils.enable("excad_addon", default_set=False)
bridge = AiBridgeServer.instance()
for obj in list(bpy.context.scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)


def run(cmd, **kw):
    result = bridge._dispatch_dict(dict(cmd=cmd, **kw))
    assert result["ok"], (cmd, result)
    return result


def box(name, x, y, z):
    return run(
        "feature_tree",
        name=name,
        recipe={
            "sketches": {"s": {"shapes": [{"type": "RECT", "width": x, "height": y}]}},
            "features": [{"type": "EXTRUDE", "id": "height", "sketch": "s", "depth": z}],
        },
    )


def verify_collision_geometry():
    def cube(name):
        bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.object
        obj.name = name
        return obj

    a, b = cube("Fixed cube"), cube("Moving cube")
    rot = Matrix.Rotation(0.44, 4, "Y") @ Matrix.Rotation(0.27, 4, "Z")
    a.matrix_world = rot
    for x, expected in ((2, False), (2.001, False), (1.996, True), (0, True)):
        b.matrix_world = rot @ Matrix.Translation((x, 0, 0))
        hits = run("check_intersections", objects=[a.name, b.name])["intersections"]
        assert bool(hits) == expected, (x, hits)
    # Only the second component is enclosed, and the container has a reflected transform.
    a.matrix_world = Matrix.Diagonal((-1, 1, 1, 1))
    b.matrix_world = Matrix.Identity(4)
    vertices = [v.co * 0.2 + Vector((x, 0, 0)) for x in (3, 0) for v in a.data.vertices]
    faces = [tuple(i + offset for i in f.vertices) for offset in (0, 8) for f in a.data.polygons]
    islands = bpy.data.meshes.new("Disconnected mesh")
    islands.from_pydata(vertices, [], faces)
    old = b.data
    b.data = islands
    bpy.data.meshes.remove(old)
    hit = run("check_intersections", objects=[a.name, b.name])["intersections"]
    assert hit and hit[0]["kind"] == "contained", hit
    # Collision resolution moves the assembly root and preserves child transforms.
    parent = bpy.data.objects.new("Assembly", None)
    bpy.context.collection.objects.link(parent)
    b.parent = parent
    sibling = cube("Fitting")
    sibling.parent = parent
    sibling.location = (3, 0, 0)
    bpy.context.view_layer.update()
    original = {o.name: o.matrix_local.copy() for o in (b, sibling)}
    assert run("resolve_overlaps", objects=[a.name, parent.name], fixed=[a.name])["moved"] == [parent.name]
    assert all(o.matrix_local == original[o.name] for o in (b, sibling))
    assert not run("check_intersections")["intersections"]
    assert run("check_intersections", objects=[b.name, sibling.name])["intersections"]
    for obj in (a, b, sibling, parent):
        bpy.data.objects.remove(obj, do_unlink=True)


verify_collision_geometry()

# A bounding box overlaps this frame, but its opening cannot support the peg.
run(
    "feature_tree",
    name="Frame",
    recipe={
        "sketches": {
            "rim": {"shapes": [{"type": "RECT", "width": 2, "height": 2}, {"type": "RECT", "width": 1, "height": 1}]}
        },
        "features": [{"type": "EXTRUDE", "sketch": "rim", "depth": 0.4}],
    },
)
box("Peg", 0.25, 0.2, 0.1)
bpy.data.objects["Peg"].location.z = 0.3
unsupported = bridge._dispatch_dict({"cmd": "place_relative", "obj": "Peg", "target": "Frame", "gap": 0})
assert not unsupported["ok"] and abs(bpy.data.objects["Peg"].location.z - 0.3) < 1e-5
info = run("render_inspect", size=80)
assert any(w["kind"] == "above_floor" and w["object"] == "Peg" for w in info["warnings"])
assert not run("check_intersections")["intersections"]
for name in ("Peg", "Frame"):
    bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

box("Stand", 2.4, 1.8, 0.22)
box("Block", 0.4, 0.4, 0.6)
for relation, axis, sign in (("LEFT_OF", 0, -1), ("RIGHT_OF", 0, 1), ("IN_FRONT_OF", 1, -1), ("BEHIND", 1, 1)):
    run("place_relative", obj="Block", target="Stand", relation=relation, gap=0.07)
    a, b = bounds(bpy.data.objects["Block"]), bounds(bpy.data.objects["Stand"])
    assert abs((a[0, axis] - b[1, axis] if sign > 0 else b[0, axis] - a[1, axis]) - 0.07) < 1e-5
run("place_relative", obj="Block", target="Stand", gap=0)
assert abs(bounds(bpy.data.objects["Block"])[0, 2] - 0.22) < 1e-6
run("place_relative", obj="Block", relation="ALIGN_FLOOR")
box("Small", 0.1, 0.1, 0.1)
run("place_relative", obj="Small", target="Stand", relation="INSIDE", gap=0.01)
hits = run("check_intersections", objects=["Small", "Stand"])["intersections"]
assert hits and hits[0]["kind"] == "contained", hits
run("resolve_overlaps", objects=["Small", "Stand", "Block"], fixed=["Stand"])
assert not run("check_intersections")["intersections"]
run("look_at", obj="Block", target="Stand")
obj = bpy.data.objects["Block"]
aim = Vector(bounds(bpy.data.objects["Stand"]).mean(0)) - obj.matrix_world.translation
assert (obj.matrix_world.to_quaternion() @ Vector((0, -1, 0))).dot(aim.normalized()) > 0.999
# Surface sampling uses the evaluated transformed surface, with exact gaps in world space.
mesh = bpy.data.meshes.new("Slope")
mesh.from_pydata([(-1, -1, 0), (1, -1, 0.4), (1, 1, 0.4), (-1, 1, 0)], [], [(0, 1, 2, 3)])
slope = bpy.data.objects.new("Slope", mesh)
bpy.context.collection.objects.link(slope)
slope.location = (3, 0, 0)
placed = run("scatter_on_surface", asset="Small", surface_obj="Slope", count=10, min_dist=0.26, seed=17)["objects"]
positions = [bpy.data.objects[n].matrix_world.translation for n in placed]
assert min((p - q).length for i, p in enumerate(positions) for q in positions[i + 1 :]) >= 0.259
count = len(bpy.data.objects)
fail = bridge._dispatch_dict(dict(cmd="scatter_on_surface", asset="Small", surface_obj="Slope", count=8, min_dist=50))
assert not fail["ok"] and len(bpy.data.objects) == count
unrelated = bpy.data.scenes.new("Reference")
marker = bpy.data.objects.new("Reference marker", None)
unrelated.collection.objects.link(marker)
marker["note"] = "preserve"
shared = bpy.data.objects.new("Shared alignment marker", None)
bpy.context.collection.objects.link(shared)
unrelated.collection.objects.link(shared)
shared["setting"] = 17
bpy.context.scene["reference_scene"] = unrelated
saved_path = str(Path(bpy.app.tempdir) / "layout-session.blend")
bpy.ops.wm.save_as_mainfile(filepath=saved_path)
initial = run("scene_digest")
run("checkpoint", name="layout")
filepath = bpy.data.filepath
run("dsl", code='sketch_new {"sketch_id":"test"}; C 0 0 0.15; extrude {"name":"Transient","depth":0.25}')
delta = run("scene_digest", since_hash=initial["h"])
assert delta["delta"] and any(r[0] == "Transient" for r in delta["changed"]) and delta["sketch"]
run("undo")
assert "Transient" not in bpy.data.objects and bpy.data.filepath == filepath
check = run("scene_digest", since_hash=initial["h"])
assert check.get("unchanged"), check
run("place_relative", obj="Block", target="Small", relation="LEFT_OF")
unrelated = bpy.data.scenes["Reference"]
unrelated.objects["Reference marker"]["note"] = "independent edit"
unrelated.objects["Shared alignment marker"]["setting"] = 23
run("rollback", name="layout")
reference = bpy.data.scenes["Reference"]
assert bpy.data.filepath == saved_path and reference.objects["Reference marker"]["note"] == "independent edit"
shared = bpy.context.scene.objects["Shared alignment marker"]
assert shared == reference.objects[shared.name] and shared["setting"] == 17
assert bpy.context.scene["reference_scene"] == reference
assert len([o for o in bpy.data.objects if o.name.startswith("Reference marker")]) == 1
check = run("scene_digest", since_hash=initial["h"])
assert check.get("unchanged"), check
# Coordinate and shader changes can preserve vertex counts and bounding boxes.
obj = bpy.data.objects["Small"]
mat = bpy.data.materials.new("Paint")
mat.diffuse_color = (0.5, 0.1, 0.1, 1)
obj.data.materials.append(mat)
before = run("scene_digest")["h"]
mat.diffuse_color = (0.1, 0.3, 0.6, 1)
assert any(r[0] == "Small" for r in run("scene_digest", since_hash=before)["changed"])
uv = obj.data.uv_layers.new(name="Test")
linked_parts = {part.name for part in bpy.context.scene.objects if part.type == "MESH" and part.data == obj.data}
before = run("scene_digest")["h"]
uv.uv[0].vector = (0.25, 0.75)
assert {r[0] for r in run("scene_digest", since_hash=before)["changed"]} == linked_parts
color = obj.data.color_attributes.new(name="Paint", type="BYTE_COLOR", domain="POINT")
before = run("scene_digest")["h"]
color.data[0].color = (0.85, 0.15, 0.12, 1)
assert {r[0] for r in run("scene_digest", since_hash=before)["changed"]} == linked_parts

# A separate arrangement for the public four-view inspection.
for name in [*placed, "Slope", "Small"]:
    bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
# Assemble a glazed planter on a chamfered display plinth using semantic placement and finishes.
bpy.data.objects.remove(bpy.data.objects["Block"], do_unlink=True)
run("chamfer", object="Stand", edges="ALL", width=0.055)
planter = {
    "op": "smooth_subtract",
    "k": 0.035,
    "children": [
        {"type": "cone", "height": 0.88, "radius": 0.38, "top_radius": 0.57},
        {"type": "cone", "height": 0.93, "radius": 0.32, "top_radius": 0.52, "center": [0, 0, 0.10]},
    ],
}
run("sdf_generate", name="Planter", tree=planter, resolution=72)
run("place_relative", obj="Planter", target="Stand", gap=0)
run("set_material", object="Planter", preset="CERAMIC", color=[0.095, 0.40, 0.36])
run("set_material", object="Stand", preset="WOOD")
lights = run("light_studio", target="Planter", style="SOFT")["lights"]
assert len(lights) == 3
light_digest = run("scene_digest", include_all=True)["h"]
assert run("light_studio", target="Planter", style="PRODUCT", power=1.1)["lights"] == lights
assert len(run("scene_digest", include_all=True, since_hash=light_digest)["changed"]) == 3
out = Path(os.getenv("EXCAD_TEST_OUTPUT", bpy.app.tempdir))
out.mkdir(parents=True, exist_ok=True)
inspection = run("render_inspect", target="ALL", size=240, path=str(out / "phase4-layout.png"))
assert not inspection["warnings"], inspection
print("LAYOUT INSPECTION", inspection, flush=True)
# Drive the published MCP entry point in a separate stdio process, through the real
# loopback socket. The background fixture pumps the same main-thread timer as the UI.
bridge.start(int(os.getenv("EXCAD_TEST_PORT", "0")))
port = bridge.port
repo = Path(__file__).resolve().parents[2]
env = dict(os.environ, EXCAD_PORT=str(port))
python = Path(sys.prefix) / ("bin/python.exe" if os.name == "nt" else "bin/python3")
if not python.exists():
    python = next((Path(sys.prefix) / "bin").glob("python*"))
messages = [
    (
        "initialize",
        {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}},
    ),
    ("server/discover", {}),
    ("tools/list", {}),
    ("tools/call", {"name": "excad_schema", "arguments": {}}),
    ("tools/call", {"name": "excad_digest", "arguments": {}}),
    (
        "tools/call",
        {"name": "excad_run", "arguments": {"steps": [{"cmd": "checkpoint", "name": "tcp"}, {"cmd": "reload_addon"}]}},
    ),
    ("tools/call", {"name": "excad_inspect", "arguments": {"target": "Stand"}}),
    ("tools/call", {"name": "excad_run", "arguments": {"code": "DIGEST"}}),
]
input_data = "".join(
    json.dumps({"jsonrpc": "2.0", "id": i, "method": method, "params": args}) + "\n"
    for i, (method, args) in enumerate(messages)
)
result = {}


def client():
    proc = subprocess.run(
        [str(python), str(repo / "excad_mcp_server.py")],
        input=input_data,
        text=True,
        capture_output=True,
        timeout=45,
        env=env,
    )
    result.update(stdout=proc.stdout, stderr=proc.stderr, code=proc.returncode)


t = threading.Thread(target=client)
t.start()
deadline = time.monotonic() + 50
while t.is_alive() and time.monotonic() < deadline:
    bridge._timer_poll()
    time.sleep(0.005)
t.join(1)
bridge.stop()
assert result.get("code") == 0, result.get("stderr")
messages = [json.loads(s)["result"] for s in result["stdout"].splitlines()]
assert len(messages) == 8
assert {t["name"] for t in messages[2]["tools"]} == {"excad_schema", "excad_run", "excad_digest", "excad_inspect"}
assert all(not m.get("isError") for m in messages[3:]), [m["content"][0] for m in messages[3:]]
assert base64.b64decode(messages[6]["content"][1]["data"]).startswith(b"\x89PNG")
assert json.loads(messages[5]["content"][0]["text"])["last"]["port"] == port
assert len(bridge._dispatch('{"cmd":"schema"}').encode()) <= 1200
print("LAYOUT MCP PASS", bpy.app.version_string, port)
