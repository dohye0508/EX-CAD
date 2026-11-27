"""Live bridge: lets an external agent (MCP server) sculpt in a *running* Blender.

* TCP server (127.0.0.1:BNM_PORT, default 9890), newline-delimited JSON:
    -> {"id": "...", "tool": "execute_brush_stroke", "args": {...}}
    <- {"id": "...", "ok": true, "result": {...}}  |  {"id": "...", "ok": false, "error": "..."}
  Special tools: ping, list_tools, cancel {"job": id}, job_status.
* All bpy work happens on the main thread inside a bpy.app.timers callback (never in the socket thread).
* Multi-stroke tools are generators stepped once per tick -> the viewport redraws between strokes.
* Viewport overlay (gpu): planned stroke path, current brush centre + radius ring.
* Co-editing: a depsgraph handler notices geometry updates made while the AI is idle; the next tool
  call re-reads the mesh (Session.sync_external) instead of overwriting the user's work, and a stroke
  that overlaps the user's edit is paused (returns paused=true) until re-planned.
"""
import json
import math
import queue
import socket
import threading
import traceback
import time

import bpy

from . import tools as TOOLS
from .session import Session

PORT = 9890
_state = {"server": None, "thread": None, "running": False, "queue": queue.Queue(), "job": None,
          "cancel": set(), "draw_handle": None, "jobs": {}, "port": PORT}


# ------------------------------------------------------------------ socket side (worker threads)
def _send(conn, obj):
    conn.sendall((json.dumps(obj, default=str) + "\n").encode("utf-8"))


def _client_thread(conn):
    buf = b""
    try:
        while _state["running"]:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    req = json.loads(line.decode("utf-8"))
                except Exception as e:
                    _send(conn, {"ok": False, "error": f"bad json: {e}"})
                    continue
                if req.get("tool") == "cancel":
                    _state["cancel"].add(req.get("args", {}).get("job"))
                    _send(conn, {"id": req.get("id"), "ok": True, "result": {"cancel_requested": True}})
                    continue
                if req.get("tool") == "ping":
                    _send(conn, {"id": req.get("id"), "ok": True, "result": {"pong": True, "blender": bpy.app.version_string}})
                    continue
                done = threading.Event()
                holder = {}
                _state["queue"].put((req, done, holder))
                done.wait()
                _send(conn, holder["resp"])
    except (ConnectionResetError, OSError):
        pass
    finally:
        conn.close()


def _server_thread(sock):
    while _state["running"]:
        try:
            conn, _ = sock.accept()
        except OSError:
            break
        threading.Thread(target=_client_thread, args=(conn,), daemon=True).start()


# ------------------------------------------------------------------ main thread
def _respond(holder, done, rid, ok, payload):
    holder["resp"] = {"id": rid, "ok": ok, ("result" if ok else "error"): payload}
    done.set()


def _tick():
    if not _state["running"]:
        return None
    job = _state["job"]
    if job is not None:
        gen, req, done, holder, t0 = job
        rid = req.get("id")
        if rid in _state["cancel"]:
            _state["cancel"].discard(rid)
            gen.close()
            _state["job"] = None
            _respond(holder, done, rid, False, "cancelled by user")
            return 0.02
        try:
            prog = next(gen)
            _state["jobs"][rid] = {"tool": req.get("tool"), "progress": TOOLS.jsonable(prog), "seconds": round(time.time() - t0, 1)}
            _redraw()
        except StopIteration as e:
            _state["job"] = None
            _state["jobs"].pop(rid, None)
            _respond(holder, done, rid, True, TOOLS.jsonable(e.value))
            _redraw()
        except Exception as e:
            _state["job"] = None
            _state["jobs"].pop(rid, None)
            _respond(holder, done, rid, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
        return 0.01
    try:
        req, done, holder = _state["queue"].get_nowait()
    except queue.Empty:
        return 0.05
    rid = req.get("id")
    name = req.get("tool")
    try:
        if name == "reload_engine":
            _respond(holder, done, rid, True, reload_engine())
        elif name == "list_tools":
            _respond(holder, done, rid, True, {"tools": sorted(TOOLS.REGISTRY), "registry_check": TOOLS.check_registry()})
        elif name == "job_status":
            _respond(holder, done, rid, True, {"jobs": _state["jobs"], "queued": _state["queue"].qsize()})
        else:
            res = TOOLS.call(name, req.get("args", {}))
            if hasattr(res, "__next__"):
                _state["job"] = (res, req, done, holder, time.time())
            else:
                _respond(holder, done, rid, True, TOOLS.jsonable(res))
                _redraw()
    except Exception as e:
        _respond(holder, done, rid, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
    return 0.01


def _redraw():
    wm = bpy.context.window_manager
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


# ------------------------------------------------------------------ overlay
def _draw_overlay():
    import gpu
    from gpu_extras.batch import batch_for_shader
    from mathutils import Vector
    ses = Session._inst
    if ses is None or not ses.target:
        return
    ob = bpy.data.objects.get(ses.target)
    if ob is None:
        return
    ov = ses.overlay
    mw = ob.matrix_world
    shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    region = bpy.context.region
    shader.uniform_float("viewportSize", (region.width, region.height))
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")
    if ov.get("path") and len(ov["path"]) > 1:
        pts = [tuple(mw @ Vector(p)) for p in ov["path"]]
        shader.uniform_float("lineWidth", 2.0)
        shader.uniform_float("color", (1.0, 0.55, 0.1, 0.9))
        batch_for_shader(shader, "LINE_STRIP", {"pos": pts}).draw(shader)
    if ov.get("dab") is not None and ov.get("radius"):
        c = mw @ Vector(ov["dab"])
        vr = bpy.context.region_data.view_rotation
        ax, ay = vr @ Vector((1, 0, 0)), vr @ Vector((0, 1, 0))
        r = ov["radius"] * max(mw.to_scale())
        ring = [tuple(c + (ax * math.cos(t) + ay * math.sin(t)) * r) for t in (i * 2 * math.pi / 40 for i in range(41))]
        shader.uniform_float("lineWidth", 1.5)
        shader.uniform_float("color", (0.2, 0.8, 1.0, 0.9))
        batch_for_shader(shader, "LINE_STRIP", {"pos": ring}).draw(shader)
    gpu.state.blend_set("NONE")


# ------------------------------------------------------------------ co-editing detection
def _depsgraph_handler(scene, depsgraph):
    ses = Session._inst
    if ses is None or ses.busy or not ses.target:
        return
    for u in depsgraph.updates:
        idd = getattr(u.id, "original", u.id)
        if isinstance(idd, bpy.types.Object) and idd.name == ses.target and u.is_updated_geometry:
            ses.pending_user_edit = True


def reload_engine():
    """Hot-reload the bnm engine modules (tools, sculpt, geometry, ...) without restarting Blender.
    The bridge itself keeps running; the session (targets, checkpoints) is rebuilt on next use."""
    global TOOLS, Session
    import sys
    import importlib
    keep = {__name__, "bnm", "bnm.blender"}
    for k in [k for k in sys.modules if k.startswith("bnm.") and k not in keep]:
        del sys.modules[k]
    TOOLS = importlib.import_module("bnm.blender.tools")
    Session = importlib.import_module("bnm.blender.session").Session
    return {"reloaded": True, "tools": len(TOOLS.REGISTRY), "registry_check": TOOLS.check_registry()}


# ------------------------------------------------------------------ control
def start(port=None):
    if _state["running"]:
        return {"running": True, "port": _state["port"]}
    port = int(port or PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(4)
    _state.update(server=sock, running=True, port=port)
    _state["thread"] = threading.Thread(target=_server_thread, args=(sock,), daemon=True)
    _state["thread"].start()
    bpy.app.timers.register(_tick, persistent=True)
    if not bpy.app.background and _state["draw_handle"] is None:
        _state["draw_handle"] = bpy.types.SpaceView3D.draw_handler_add(_draw_overlay, (), "WINDOW", "POST_VIEW")
    if _depsgraph_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_depsgraph_handler)
    Session.get()
    return {"running": True, "port": port}


def stop():
    _state["running"] = False
    if _state["server"]:
        try:
            _state["server"].close()
        except OSError:
            pass
    _state["server"] = None
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    if _state["draw_handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_state["draw_handle"], "WINDOW")
        _state["draw_handle"] = None
    if _depsgraph_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_depsgraph_handler)
    return {"running": False}


def is_running():
    return _state["running"]
