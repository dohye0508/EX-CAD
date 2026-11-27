#!/usr/bin/env python3
"""MCP server (stdio, JSON-RPC 2.0, stdlib only) for blender-neural-modeler.

Forwards tool calls to the BNM Live Bridge add-on running inside Blender (127.0.0.1:9890).
Render tools return their PNGs as MCP image content so the model can *look* at the sculpt.
If no live Blender is reachable, `headless_batch` runs commands in a background Blender on a
.blend file, and `launch_blender` starts Blender with the bridge add-on.
"""
import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bnm.blender.tool_schemas import TOOLS  # noqa: E402  (pure python, no bpy)

CFG_PATH = os.path.join(ROOT, "config", "backend_config.json")
CFG = json.load(open(CFG_PATH, encoding="utf-8")) if os.path.exists(CFG_PATH) else {}
HOST = os.environ.get("BNM_HOST", CFG.get("host", "127.0.0.1"))
PORT = int(os.environ.get("BNM_PORT", CFG.get("port", 9890)))
BLENDER = os.environ.get("BNM_BLENDER", CFG.get("blender_path", "blender"))
MAX_IMAGES = int(CFG.get("max_images_per_result", 6))

EXTRA = {
    "bridge_status": {"description": "Is a live Blender with the BNM bridge reachable? (ping)",
                      "inputSchema": {"type": "object", "properties": {}}},
    "launch_blender": {"description": "Start Blender (UI) with the BNM bridge add-on autostarting; optional .blend to open.",
                       "inputSchema": {"type": "object", "properties": {"blend": {"type": "string"}}}},
    "headless_batch": {"description": "Run a list of {tool,args} commands in a background Blender (no UI). State persists only "
                                      "through the .blend: blend_in is opened first, blend_out is saved at the end.",
                       "inputSchema": {"type": "object", "properties": {
                           "commands": {"type": "array", "items": {"type": "object"}}, "blend_in": {"type": "string"},
                           "blend_out": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["commands"]}},
    "reload_engine": {"description": "Hot-reload the bnm engine inside the running Blender after the skill code was updated.",
                      "inputSchema": {"type": "object", "properties": {}}},
    "job_status": {"description": "Progress of a running multi-stroke job in the live bridge.",
                   "inputSchema": {"type": "object", "properties": {}}},
    "cancel_job": {"description": "Cancel a running job by request id.",
                   "inputSchema": {"type": "object", "properties": {"job": {"type": "string"}}, "required": ["job"]}},
}


# ------------------------------------------------------------------ bridge client
def bridge_call(tool, args, timeout=3600):
    rid = uuid.uuid4().hex[:12]
    with socket.create_connection((HOST, PORT), timeout=5) as s:
        s.settimeout(timeout)
        s.sendall((json.dumps({"id": rid, "tool": tool, "args": args}) + "\n").encode("utf-8"))
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(1 << 20)
            if not chunk:
                raise ConnectionError("bridge closed the connection")
            buf += chunk
    return json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))


def bridge_alive():
    try:
        r = bridge_call("ping", {}, timeout=5)
        return r.get("ok", False), r
    except OSError as e:
        return False, {"error": str(e)}


def headless_batch(args):
    cmd_file = os.path.join(tempfile.gettempdir(), f"bnm_cmds_{uuid.uuid4().hex[:8]}.json")
    res_file = cmd_file.replace("cmds", "result")
    with open(cmd_file, "w", encoding="utf-8") as f:
        json.dump(args["commands"], f)
    cmd = [BLENDER, "-b", "--factory-startup", "--python", os.path.join(ROOT, "scripts", "headless_runner.py"), "--",
           "--commands", cmd_file, "--result", res_file]
    if args.get("blend_in"):
        cmd += ["--blend-in", args["blend_in"]]
    if args.get("blend_out"):
        cmd += ["--blend-out", args["blend_out"]]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.get("timeout", 1800))
    out = json.load(open(res_file, encoding="utf-8")) if os.path.exists(res_file) else {"error": p.stderr[-2000:]}
    for f in (cmd_file, res_file):
        if os.path.exists(f):
            os.remove(f)
    return out


def launch_blender(args):
    cmd = [BLENDER]
    if args.get("blend"):
        cmd.append(args["blend"])
    subprocess.Popen(cmd, close_fds=True, creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
    return {"launched": True, "note": "the BNM add-on autostarts the bridge ~1 s after Blender opens; call bridge_status"}


# ------------------------------------------------------------------ MCP plumbing
def _collect_images(obj, acc):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "images" and isinstance(v, list):
                acc.extend(p for p in v if isinstance(p, str))
            else:
                _collect_images(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _collect_images(v, acc)
    return acc


def to_content(result):
    content = [{"type": "text", "text": json.dumps(result, ensure_ascii=False, default=str)[:60000]}]
    imgs = [p for p in _collect_images(result, []) if os.path.exists(p)]
    # prefer clay/blender renders when there are many
    imgs.sort(key=lambda p: (0 if ("_clay" in p or "_blender" in p) else 1))
    for p in imgs[:MAX_IMAGES]:
        with open(p, "rb") as f:
            content.append({"type": "image", "data": base64.b64encode(f.read()).decode("ascii"), "mimeType": "image/png"})
    return content


def handle_call(name, args):
    if name == "bridge_status":
        ok, r = bridge_alive()
        return {"live_bridge": ok, "detail": r, "host": HOST, "port": PORT}
    if name == "launch_blender":
        return launch_blender(args)
    if name == "headless_batch":
        return headless_batch(args)
    if name == "cancel_job":
        return bridge_call("cancel", {"job": args["job"]})
    if name == "job_status":
        return bridge_call("job_status", {})
    if name == "reload_engine":
        r = bridge_call("reload_engine", {})
        if not r.get("ok"):
            raise RuntimeError(r.get("error"))
        return r["result"]
    if name not in TOOLS:
        raise ValueError(f"unknown tool {name}")
    try:
        r = bridge_call(name, args)
    except OSError as e:
        raise RuntimeError(f"Blender live bridge not reachable at {HOST}:{PORT} ({e}). Start Blender with the BNM add-on "
                           f"(tool launch_blender) or use headless_batch.")
    if not r.get("ok"):
        raise RuntimeError(r.get("error", "unknown bridge error"))
    return r["result"]


def main():
    tools = [{"name": k, **v} for k, v in {**TOOLS, **EXTRA}.items()]
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid = msg.get("id")
        method = msg.get("method")
        if mid is None:  # notification
            continue
        try:
            if method == "initialize":
                pv = msg.get("params", {}).get("protocolVersion", "2025-06-18")
                res = {"protocolVersion": pv, "capabilities": {"tools": {}},
                       "serverInfo": {"name": "blender-neural-modeler", "version": "0.1.0"},
                       "instructions": "Scratch-sculpt in Blender. Read the blender-neural-modeler skill first. "
                                       "Default mode SCRATCH_SCULPT: start from a primitive, never import finished meshes."}
            elif method == "tools/list":
                res = {"tools": tools}
            elif method == "tools/call":
                p = msg.get("params", {})
                try:
                    out = handle_call(p.get("name"), p.get("arguments") or {})
                    res = {"content": to_content(out), "isError": False}
                except Exception as e:
                    res = {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}
            elif method == "ping":
                res = {}
            else:
                raise ValueError(f"method not found: {method}")
            reply = {"jsonrpc": "2.0", "id": mid, "result": res}
        except Exception as e:
            reply = {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": str(e)}}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
