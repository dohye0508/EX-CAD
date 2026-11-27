# SPDX-License-Identifier: GPL-2.0-or-later
"""Provider protocols run without bpy. Only the final GLB import needs Blender's main thread."""

import base64
import json
import mimetypes
import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen
import uuid

CLOUD = {"MESHY": "https://api.meshy.ai/openapi", "TRIPO": "https://openapi.tripo3d.ai/v3"}


def request(url, data=None, *, token=None, content_type=None, timeout=90):
    headers = {}
    if token:
        headers["Authorization"] = "Bearer "+token
    if isinstance(data,dict):
        data = json.dumps(data).encode()
        content_type = "application/json"
    if content_type:
        headers["Content-Type"] = content_type
    try:
        with urlopen(Request(url,data=data,headers=headers),timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        raise RuntimeError(f"Provider HTTP {exc.code}: {exc.reason}") from exc


def multipart(path, field):
    boundary = "excad-"+uuid.uuid4().hex
    kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    safe_name = "input"+path.suffix.lower()
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{safe_name}"\r\n'
            f'Content-Type: {kind}\r\n\r\n').encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return body,"multipart/form-data; boundary="+boundary


def _poll(url, token, provider, deadline):
    while time.monotonic() < deadline:
        state = json.loads(request(url,token=token))
        if provider == "TRIPO":
            if state.get("code") != 0:
                raise RuntimeError(f"Tripo task query failed ({state.get('code')})")
            state = state["data"]
        status = state["status"].lower()
        if status in {"succeeded","success"}:
            if provider == "TRIPO":
                return state["output"]["model_url"]
            return state["model_urls"]["glb"]
        if status not in {"pending","queued","running","processing","in_progress","created"}:
            raise RuntimeError(f"{provider} task ended with status {status}")
        time.sleep(min(2.0,max(0,deadline-time.monotonic())))
    raise TimeoutError(f"{provider} generation timed out")


def _cloud(provider, prompt, image, deadline):
    token = os.getenv(provider+"_API_KEY")
    if not token:
        raise ValueError(f"Set {provider}_API_KEY to use this provider")
    base = CLOUD[provider]
    if provider == "MESHY":
        if image:
            mime = mimetypes.guess_type(image.name)[0] or "image/png"
            uri = "data:"+mime+";base64,"+base64.b64encode(image.read_bytes()).decode()
            endpoint,payload = "/v1/image-to-3d",{"image_url":uri}
        else:
            endpoint,payload = "/v2/text-to-3d",{"mode":"preview","prompt":prompt}
        url = base+endpoint
        task = json.loads(request(url,payload,token=token))["result"]
        download = _poll(url+"/"+quote(task,safe=""),token,provider,deadline)
        if not image:
            refine = json.loads(request(url,{"mode":"refine","preview_task_id":task},token=token))["result"]
            download = _poll(url+"/"+quote(refine,safe=""),token,provider,deadline)
    else:
        payload = {"model":"v3.1-20260211","texture":True,"pbr":True}
        if image:
            body,ctype = multipart(image,"file")
            upload = json.loads(request(base+"/files",body,token=token,content_type=ctype))
            payload["input"] = upload["data"]["file_token"]
            route = "/generation/image-to-model"
        else:
            payload["prompt"] = prompt
            route = "/generation/text-to-model"
        response = json.loads(request(base+route,payload,token=token))
        if response.get("code") != 0:
            raise RuntimeError(f"Tripo submission failed ({response.get('code')})")
        task = response["data"]["task_id"]
        download = _poll(base+"/tasks/"+quote(task,safe=""),token,provider,deadline)
    # Signed result URLs have their own authority. Only the API origin receives the key.
    return request(download,timeout=max(1,deadline-time.monotonic()))


def _trellis(base, image, deadline):
    if not image:
        raise ValueError("The upstream TRELLIS image server requires image_path")
    config = json.loads(request(base+"/config"))
    api = base+config.get("api_prefix","/gradio_api").rstrip("/")
    session = uuid.uuid4().hex
    def call(name, args):
        url = api+"/call/"+name
        event = json.loads(request(url,{"data":args,"session_hash":session}))["event_id"]
        stream = request(url+"/"+quote(event,safe=""),timeout=max(1,deadline-time.monotonic())).decode()
        for block in stream.replace("\r\n","\n").split("\n\n"):
            lines = block.splitlines()
            if "event: error" in lines:
                raise RuntimeError(f"TRELLIS {name} failed")
            if "event: complete" in lines:
                return json.loads(next(s[6:] for s in lines if s.startswith("data: ")))
        raise RuntimeError(f"TRELLIS {name} returned no result")
    body,ctype = multipart(image,"files")
    file = json.loads(request(api+"/upload",body,content_type=ctype))[0]
    call("start_session",[])
    processed = call("preprocess_image",[{"path":file,"meta":{"_type":"gradio.FileData"}}])[0]
    call("image_to_3d",[processed,[],0,7.5,12,3.0,12,"stochastic"])
    model = call("extract_glb",[0.95,1024])[0]
    url = model.get("url") or api+"/file="+quote(model["path"],safe="/:")
    return request(urljoin(base+"/",url),timeout=max(1,deadline-time.monotonic()))


def download_model(prompt="", image_path=None, provider="HUNYUAN3D-2", timeout=900):
    provider = provider.upper()
    image = Path(image_path) if image_path else None
    if not prompt and image is None:
        raise ValueError("Provide a prompt or an image")
    if image and not image.is_file():
        raise FileNotFoundError(image)
    deadline = time.monotonic()+timeout
    if provider in CLOUD:
        blob = _cloud(provider,prompt,image,deadline)
    elif provider in {"HUNYUAN3D-2","HUNYUAN"}:
        base = os.getenv("HUNYUAN3D_URL","http://127.0.0.1:8080").rstrip("/")
        payload = {"image":base64.b64encode(image.read_bytes()).decode()} if image else {"text":prompt}
        blob = request(base+"/generate",dict(payload,type="glb",texture=True),timeout=timeout)
    elif provider == "TRELLIS":
        blob = _trellis(os.getenv("TRELLIS_URL","http://127.0.0.1:7860").rstrip("/"),image,deadline)
    else:
        raise ValueError(f"Unknown 3D provider {provider}")
    if blob[:4] != b"glTF" or blob[4:8] != b"\x02\x00\x00\x00" or int.from_bytes(blob[8:12],"little") != len(blob):
        raise ValueError("Provider returned an invalid GLB 2.0 file")
    with tempfile.NamedTemporaryFile(prefix="excad-model-",suffix=".glb",delete=False) as out:
        out.write(blob)
        return Path(out.name)


def import_model(path, height=1.0, remesh=False, rig=False, name="Generated model"):
    import bpy
    from mathutils import Matrix, Vector
    from .asset_fetcher import import_asset_file, _world_bbox, _root_objects
    from .human_engine import auto_rig_any_human_mesh
    if height <= 0:
        raise ValueError("Normalized height must be positive")
    before = set(bpy.data.objects)
    try:
        imported = import_asset_file(str(path))
        meshes = [o for o in imported if o.type=="MESH"]
        if not meshes:
            raise ValueError("GLB contains no mesh")
        bpy.context.view_layer.update()
        low,high = _world_bbox(imported)
        if high.z-low.z <= 1e-8:
            raise ValueError("GLB has no height")
        center = Vector(((low.x+high.x)/2,(low.y+high.y)/2,low.z))
        matrix = Matrix.Scale(height/(high.z-low.z),4) @ Matrix.Translation(-center)
        roots = _root_objects(imported)
        for obj in roots:
            obj.matrix_world = matrix @ obj.matrix_world
        bpy.context.view_layer.update()
        for obj in meshes:
            if remesh:
                obj.data = obj.data.copy() if obj.data.users > 1 else obj.data
                obj.data.remesh_voxel_size = max(obj.dimensions)/max(obj.matrix_world.to_scale())/128
                with bpy.context.temp_override(object=obj,active_object=obj):
                    bpy.ops.object.voxel_remesh()
            if rig and not obj.find_armature():
                auto_rig_any_human_mesh(obj,name=obj.name+" rig")
        roots[0].name = name
        return {"name":roots[0].name,"objects":[o.name for o in imported],"height":height}
    except Exception:
        for obj in set(bpy.data.objects)-before:
            bpy.data.objects.remove(obj,do_unlink=True)
        raise


def generate_3d(prompt="", image_path=None, provider="HUNYUAN3D-2", **options):
    path = download_model(prompt,image_path,provider)
    try:
        return import_model(path,**options)
    finally:
        path.unlink(missing_ok=True)
