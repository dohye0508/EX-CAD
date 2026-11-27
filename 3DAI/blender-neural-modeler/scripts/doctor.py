"""Environment / installation check (system Python; launches Blender in background for engine checks).

  python scripts/doctor.py
"""
import json
import os
import shutil
import socket
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")


def main():
    cfg = json.load(open(os.path.join(ROOT, "config", "backend_config.json")))
    rep = {"skill_root": ROOT, "python": sys.version.split()[0], "os": sys.platform}
    blender = cfg.get("blender_path")
    rep["blender_found"] = bool(blender and os.path.exists(blender))
    rep["skill_installed"] = os.path.exists(os.path.join(HOME, ".claude", "skills", "blender-neural-modeler", "SKILL.md"))
    claude = shutil.which("claude") or os.path.join(HOME, ".local", "bin", "claude.exe")
    if os.path.exists(claude):
        r = subprocess.run([claude, "mcp", "get", "blender-neural-modeler"], capture_output=True, text=True)
        rep["mcp_registered"] = r.returncode == 0
    try:
        with socket.create_connection((cfg["host"], cfg["port"]), timeout=2) as s:
            s.sendall(b'{"id":"d","tool":"ping","args":{}}\n')
            rep["live_bridge"] = json.loads(s.recv(4096).decode().split("\n")[0])
    except OSError as e:
        rep["live_bridge"] = f"not reachable ({e}) - start Blender with the BNM add-on"
    sys.path.insert(0, ROOT)
    from bnm.blender.tool_schemas import TOOLS
    rep["mcp_tools"] = len(TOOLS)
    if rep["blender_found"]:
        expr = ("import sys, json; sys.path.insert(0, r'%s'); from bnm.core.capability_detector import detect; "
                "from bnm.blender.tools import check_registry; c=detect(); c.pop('essentials_sculpt_brushes', None); "
                "c.pop('brush_stroke_props', None); c.pop('sculpt_brush_types', None); "
                "print('BNMDOC' + json.dumps({'caps': c, 'registry': check_registry()}))") % ROOT
        out = subprocess.run([blender, "-b", "--factory-startup", "--python-expr", expr], capture_output=True, text=True, timeout=300).stdout
        line = next((l for l in out.splitlines() if l.startswith("BNMDOC")), None)
        rep["engine"] = json.loads(line[6:]) if line else {"error": out[-1500:]}
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
