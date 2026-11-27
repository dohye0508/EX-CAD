"""Install blender-neural-modeler for Claude Code.

  python scripts/install.py [--blender PATH] [--skills-dir DIR] [--no-mcp] [--no-addon]

1. Copies the skill (SKILL.md, bnm/, bnm_mcp/, knowledge/, config/, scripts/) to ~/.claude/skills/blender-neural-modeler
   (benchmarks/ and caches are not copied).
2. Installs the Blender add-on `bnm_live_bridge` into Blender's user add-on folder and enables it
   (runs Blender in background once to enable + save preferences).
3. Registers the MCP server with Claude Code (`claude mcp add --scope user`), stdio, system Python.
Existing skills/MCP servers are never removed; an existing install of this skill is replaced.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")
COPY = ["SKILL.md", "README.md", "requirements.txt", "pyproject.toml", "bnm", "bnm_mcp", "knowledge", "config", "scripts"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "benchmarks", "*.blend1")


def find_blender(explicit):
    cands = [explicit, os.environ.get("BNM_BLENDER"),
             json.load(open(os.path.join(SRC, "config", "backend_config.json"))).get("blender_path"),
             shutil.which("blender")]
    base = r"C:\Program Files\Blender Foundation"
    if os.path.isdir(base):
        for d in sorted(os.listdir(base), reverse=True):
            cands.append(os.path.join(base, d, "blender.exe"))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def blender_version(blender):
    out = subprocess.run([blender, "-b", "--factory-startup", "--python-expr",
                          "import bpy;print('BNMVER', bpy.app.version[0], bpy.app.version[1])"],
                         capture_output=True, text=True, timeout=180).stdout
    for line in out.splitlines():
        if line.startswith("BNMVER"):
            _, a, b = line.split()
            return f"{a}.{b}"
    return None


def install_skill(dst):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for item in COPY:
        s = os.path.join(SRC, item)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(dst, item), ignore=IGNORE)
        elif os.path.exists(s):
            shutil.copy2(s, os.path.join(dst, item))
    return dst


def install_addon(blender, skill_dir):
    ver = blender_version(blender)
    if not ver:
        return {"ok": False, "error": "could not query Blender version"}
    addons = os.path.join(os.environ.get("APPDATA", HOME), "Blender Foundation", "Blender", ver, "scripts", "addons")
    os.makedirs(addons, exist_ok=True)
    dst = os.path.join(addons, "bnm_live_bridge")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(skill_dir, "bnm", "blender", "addon", "bnm_live_bridge"), dst, ignore=IGNORE)
    with open(os.path.join(dst, "_root.txt"), "w", encoding="utf-8") as f:
        f.write(skill_dir)
    expr = ("import bpy, addon_utils; addon_utils.enable('bnm_live_bridge', default_set=True, persistent=True); "
            "bpy.ops.wm.save_userpref(); print('BNM_ADDON_ENABLED', 'bnm_live_bridge' in bpy.context.preferences.addons)")
    out = subprocess.run([blender, "-b", "--python-expr", expr], capture_output=True, text=True, timeout=180).stdout
    return {"ok": "BNM_ADDON_ENABLED True" in out, "path": dst, "blender_version": ver}


def register_mcp(skill_dir):
    claude = shutil.which("claude") or os.path.join(HOME, ".local", "bin", "claude.exe")
    if not os.path.exists(claude):
        return {"ok": False, "error": "claude CLI not found",
                "manual": f'claude mcp add --scope user blender-neural-modeler -- "{sys.executable}" "{os.path.join(skill_dir, "bnm_mcp", "server.py")}"'}
    subprocess.run([claude, "mcp", "remove", "--scope", "user", "blender-neural-modeler"], capture_output=True, text=True)
    r = subprocess.run([claude, "mcp", "add", "--scope", "user", "blender-neural-modeler", "--",
                        sys.executable, os.path.join(skill_dir, "bnm_mcp", "server.py")], capture_output=True, text=True)
    return {"ok": r.returncode == 0, "stdout": r.stdout.strip()[-400:], "stderr": r.stderr.strip()[-400:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender")
    ap.add_argument("--skills-dir", default=os.path.join(HOME, ".claude", "skills"))
    ap.add_argument("--no-mcp", action="store_true")
    ap.add_argument("--no-addon", action="store_true")
    a = ap.parse_args()
    report = {}
    blender = find_blender(a.blender)
    report["blender"] = blender
    skill_dir = install_skill(os.path.join(a.skills_dir, "blender-neural-modeler"))
    report["skill_dir"] = skill_dir
    if blender:
        cfgp = os.path.join(skill_dir, "config", "backend_config.json")
        cfg = json.load(open(cfgp))
        cfg["blender_path"] = blender.replace("\\", "/")
        json.dump(cfg, open(cfgp, "w"), indent=2)
    if not a.no_addon:
        report["addon"] = install_addon(blender, skill_dir) if blender else {"ok": False, "error": "Blender not found"}
    if not a.no_mcp:
        report["mcp"] = register_mcp(skill_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
