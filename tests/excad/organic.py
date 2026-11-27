# SPDX-License-Identifier: GPL-2.0-or-later
import math
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading

import bpy
import bmesh
import numpy as np

from excad_addon import ai_3d_provider as providers
from excad_addon import polyhedron_engine
from excad_addon.ai_bridge import AiBridgeServer
from excad_addon.feature_tree import _solid
from excad_addon.inspection import _png

bridge = AiBridgeServer.instance()
for obj in list(bpy.context.scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)


def run(cmd, **kw):
    result = bridge._dispatch_dict(dict(cmd=cmd, **kw))
    assert result["ok"], (cmd, result)
    return result


def shape(t, **kw):
    return dict(type=t, **kw)


def sdf(name, tree, **args):
    result = run("sdf_generate", tree=tree, name=name, **args)
    obj = bpy.data.objects[result["name"]]
    _solid(obj.data)
    return obj


sphere = shape("sphere", radius=0.25)
capsule = shape("capsule", a=[0, 0, -0.4], b=[0, 0, 0.4], radius=0.15)
box = shape("round_box", size=[0.45, 0.22, 1], radius=0.06)
for i, tree in enumerate(
    [
        sphere,
        capsule,
        shape("torus", major_radius=0.4, minor_radius=0.1),
        shape("ellipsoid", radii=[0.42, 0.23, 0.35]),
        shape("cone", height=0.8, radius=0.35),
        shape("mirror_x", child=dict(sphere, center=[0.18, 0, 0])),
        shape("radial_array", count=5, child=dict(capsule, center=[0.4, 0, 0])),
        shape("twist", degrees_per_unit=190, child=box),
        shape("bend", radius=0.85, child=box),
        shape("smooth_intersect", children=[box, sphere], k=0.04),
    ]
):
    sdf("Sample " + str(i), tree, resolution=40, remesh="VOXEL" if i == 2 else "QUAD")
# A fully enclosed void must subtract from the signed volume of the outer surface.
hollow = sdf(
    "Hollow sphere",
    shape("smooth_subtract", k=0, children=[shape("sphere", radius=0.7), shape("sphere", radius=0.45)]),
    resolution=72,
)
bm = bmesh.new()
bm.from_mesh(hollow.data)
assert math.isclose(bm.calc_volume(signed=True), 4 * math.pi / 3 * (0.7**3 - 0.45**3), rel_tol=0.012)
bm.free()
# A glazed vase with a flared lip, tapered base and a genuinely hollow interior.
body = shape(
    "smooth_union",
    k=0.13,
    children=[
        shape("ellipsoid", radii=[0.46, 0.46, 0.58], center=[0, 0, 0.57]),
        shape("cone", radius=0.21, top_radius=0.27, height=0.30, center=[0, 0, 1.08]),
        shape("torus", major_radius=0.25, minor_radius=0.045, center=[0, 0, 1.22]),
    ],
)
inside = shape(
    "smooth_union",
    k=0.09,
    children=[
        shape("ellipsoid", radii=[0.37, 0.37, 0.49], center=[0, 0, 0.57]),
        shape("capsule", radius=0.20, a=[0, 0, 0.89], b=[0, 0, 1.42]),
    ],
)
vase = sdf("Glazed vase", shape("smooth_subtract", children=[body, inside], k=0.025), resolution=96)
vase.location.z = -min(v.co.z for v in vase.data.vertices)
mat = bpy.data.materials.new("Jade")
mat.diffuse_color = (0.055, 0.36, 0.28, 1)
vase.data.materials.append(mat)
out = Path(os.getenv("EXCAD_TEST_OUTPUT", bpy.app.tempdir))
out.mkdir(parents=True, exist_ok=True)
preview = run("render_inspect", target=vase.name, size=240, path=str(out / "phase3-sdf.png"))
print("SDF INSPECTION", preview, flush=True)
assert not preview["warnings"]
with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp)
    img = np.full((64, 64, 3), 255, np.uint8)
    img[8:56, 8:56] = [32, 128, 165]
    img[16:49, 19:44] = [225, 88, 50]
    for f in polyhedron_engine.get_polyhedron_faces("CUBE_6"):
        _png(directory / f["filename"], img)
        if f["normal"][2] > 0.99:
            depth = np.full_like(img, 32)
            depth[24:40, 24:40] = 128
            _png(directory / f["filename"].replace(".png", "_depth.png"), depth)
    opt = dict(image_dir=tmp, poly_type="CUBE_6", size=2, center=[0, 0, 0], resolution=38, project_colors=False)
    full = bpy.data.objects[run("poly_reconstruct", name="Hull", use_depth=False, **opt)["name"]]
    cut = bpy.data.objects[run("poly_reconstruct", name="Depth hull", use_depth=True, **opt)["name"]]
    bpy.context.view_layer.update()
    _solid(cut.data)
    assert full.ray_cast((0, 0, 2), (0, 0, -1))[1].z - cut.ray_cast((0, 0, 2), (0, 0, -1))[1].z > 0.4
    atlas = run("poly_bake", object=cut.name, image_dir=tmp, poly_type="CUBE_6", size=2, center=[0, 0, 0])
    tex = cut.data.materials[0].node_tree.nodes.get("Image Texture").image
    assert list(tex.size) == [2048, 2048] and cut.data.uv_layers["EXCAD Atlas"] and Path(atlas["path"]).is_file()
    values = np.empty(2048 * 2048 * 4, np.float32)
    tex.pixels.foreach_get(values)
    assert np.std(values.reshape(-1, 4)[:, :3]) > 0.10
    print("UV BAKE", atlas["resolution"], atlas["views_used"], flush=True)

    glb = directory / "fixture.glb"
    for o in bpy.context.selected_objects:
        o.select_set(False)
    vase.select_set(True)
    bpy.context.view_layer.objects.active = vase
    bpy.ops.export_scene.gltf(filepath=str(glb), use_selection=True, export_format="GLB")
    model = glb.read_bytes()
    calls = []
    sessions = {}
    inputs = [[], [1], [1, 2, 0, 3, 4, 5, 6, 7, 8], [0, 3, 4]]
    functions = ["start_session", "preprocess_image", "image_to_3d", "extract_glb"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, data):
            if not isinstance(data, bytes):
                data = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            calls.append(("GET", self.path))
            if self.path == "/result.glb":
                assert not self.headers.get("Authorization")
                return self.send(model)
            if self.path == "/config":
                return self.send(
                    {
                        "api_prefix": "/gradio_api",
                        "dependencies": [
                            {"id": i, "api_name": name, "inputs": inputs[i]} for i, name in enumerate(functions)
                        ],
                        "components": [{"id": i, "type": "state" if i == 0 else "value"} for i in range(9)],
                    }
                )
            if self.path.startswith("/gradio_api/queue/data?session_hash="):
                session = self.path.split("=")[-1]
                name = functions[sessions[session]]
                res = {
                    "start_session": [],
                    "preprocess_image": [{"path": "processed.png"}],
                    "image_to_3d": [{}],
                    "extract_glb": [{"url": base + "/result.glb"}],
                }[name]
                message = {"msg": "process_completed", "event_id": "event", "success": True, "output": {"data": res}}
                return self.send(("data: " + json.dumps(message) + "\n\n").encode())
            if self.path.startswith("/tripo/"):
                return self.send(
                    {"code": 0, "data": {"status": "success", "output": {"model_url": base + "/result.glb"}}}
                )
            self.send({"status": "SUCCEEDED", "model_urls": {"glb": base + "/result.glb"}})

        def do_POST(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(("POST", self.path))
            if self.path == "/generate":
                command = json.loads(data)
                assert command["type"] == "glb" and command["texture"] == ("image" in command)
                return self.send(model)
            if self.path.startswith("/gradio_api/"):
                if self.path.endswith("upload"):
                    return self.send(["upload.png"])
                body = json.loads(data)
                index = body["fn_index"]
                key = body["session_hash"]
                assert index == sessions.get(key, -1) + 1 and len(body["data"]) == len(inputs[index])
                assert all(body["data"][j] is None for j, c in enumerate(inputs[index]) if c == 0)
                sessions[key] = index
                return self.send({"event_id": "event"})
            assert self.headers["Authorization"] == "Bearer fixture-token"
            if self.path == "/tripo/files":
                assert b'name="file"' in data
                return self.send({"code": 0, "data": {"file_token": "file_test"}})
            if self.path.startswith("/tripo/"):
                return self.send({"code": 0, "data": {"task_id": "task"}})
            return self.send({"result": "task"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        os.environ.update(
            HUNYUAN3D_URL=base, TRELLIS_URL=base, TRIPO_API_KEY="fixture-token", MESHY_API_KEY="fixture-token"
        )
        providers.CLOUD.update(TRIPO=base + "/tripo", MESHY=base + "/meshy")
        for i, (provider, image) in enumerate(
            [
                ("MESHY", False),
                ("MESHY", True),
                ("TRIPO", False),
                ("TRIPO", True),
                ("HUNYUAN3D-2", False),
                ("HUNYUAN3D-2", True),
                ("TRELLIS", True),
            ]
        ):
            res = run(
                "generate_3d",
                prompt="vase",
                provider=provider,
                image_path=str(directory / "face_00.png") if image else None,
                height=0.65,
                remesh=(i == 4),
                texture=(i == 5),
                rig=(i == 5),
                name="Imported " + str(i),
            )
            assert res["objects"] and res["height"] == 0.65
            obj = bpy.data.objects[res["meshes"][0]]
            bpy.context.view_layer.update()
            bounds = [(obj.matrix_world @ v.co).z for v in obj.data.vertices]
            assert abs(min(bounds)) < 0.02 and abs(max(bounds) - 0.65) < 0.02, (provider, min(bounds), max(bounds))
            if i == 5:
                armature = obj.find_armature()
                assert armature and armature.pose.bones["Root"]
                assert all(v.groups for v in obj.data.vertices)
                run("place_relative", obj=res["name"], relation="ALIGN_FLOOR", gap=0.2)
                assert abs(providers._dimensions([obj])[0].z - 0.2) < 1e-5
        # Several unparented imported parts must behave as one placement target.
        for item in list(bpy.context.selected_objects):
            item.select_set(False)
        vase.select_set(True)
        second = vase.copy()
        bpy.context.collection.objects.link(second)
        second.location = (2, 0, 0.45)
        second.scale = (0.4, 0.4, 0.4)
        second.select_set(True)
        bundle = directory / "assembly.glb"
        bpy.ops.export_scene.gltf(filepath=str(bundle), use_selection=True, export_format="GLB")
        assembled = providers.import_model(bundle, name="Bundle", height=0.9, rig=True)
        parts = [bpy.data.objects[n] for n in assembled["meshes"]]
        assert len(parts) == 2 and len({p.find_armature() for p in parts}) == 1
        assert abs(providers._dimensions(parts)[1].z - 0.9) < 1e-5
        run("place_relative", obj=assembled["name"], relation="ALIGN_FLOOR", gap=0.17)
        assert abs(providers._dimensions(parts)[0].z - 0.17) < 1e-5
        print("PROVIDER PROTOCOLS", len(calls), "real loopback HTTP requests", flush=True)
    finally:
        server.shutdown()
        server.server_close()
assert len(bridge._dispatch('{"cmd":"schema"}').encode()) <= 1200
print("ORGANIC PASS", bpy.app.version_string)
