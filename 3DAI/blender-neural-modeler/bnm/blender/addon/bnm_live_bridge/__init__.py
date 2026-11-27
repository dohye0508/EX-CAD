"""BNM Live Bridge - Blender add-on for the blender-neural-modeler skill.

Installed by scripts/install.py into Blender's user add-on folder.  `_root.txt` (written by the
installer) points at the skill directory that contains the `bnm` engine package.
"""
bl_info = {
    "name": "BNM Live Bridge (blender-neural-modeler)",
    "author": "blender-neural-modeler",
    "version": (0, 1, 0),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > BNM",
    "description": "Lets an AI agent sculpt in this Blender session via a local socket (127.0.0.1)",
    "category": "Sculpt",
}

import os
import sys

import bpy

_HERE = os.path.dirname(os.path.abspath(__file__))


def _root():
    p = os.path.join(_HERE, "_root.txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read().strip()
    return os.environ.get("BNM_ROOT", "")


def _ensure_path():
    r = _root()
    if r and r not in sys.path:
        sys.path.insert(0, r)
    return r


class BNM_Prefs(bpy.types.AddonPreferences):
    bl_idname = __name__
    port: bpy.props.IntProperty(name="Port", default=9890, min=1024, max=65535)
    autostart: bpy.props.BoolProperty(name="Start bridge when Blender starts", default=True)

    def draw(self, context):
        self.layout.prop(self, "port")
        self.layout.prop(self, "autostart")
        self.layout.label(text=f"Engine path: {_root() or '(not set)'}")


class BNM_OT_start(bpy.types.Operator):
    bl_idname = "bnm.bridge_start"
    bl_label = "Start AI Bridge"

    def execute(self, context):
        _ensure_path()
        from bnm.blender import live_bridge
        prefs = context.preferences.addons[__name__].preferences
        r = live_bridge.start(prefs.port)
        self.report({"INFO"}, f"BNM bridge listening on 127.0.0.1:{r['port']}")
        return {"FINISHED"}


class BNM_OT_stop(bpy.types.Operator):
    bl_idname = "bnm.bridge_stop"
    bl_label = "Stop AI Bridge"

    def execute(self, context):
        _ensure_path()
        from bnm.blender import live_bridge
        live_bridge.stop()
        return {"FINISHED"}


class BNM_PT_panel(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "BNM"
    bl_label = "AI Sculptor Bridge"

    def draw(self, context):
        lay = self.layout
        try:
            _ensure_path()
            from bnm.blender import live_bridge
            from bnm.blender.session import Session
            running = live_bridge.is_running()
        except Exception as e:  # engine path missing
            lay.label(text=f"engine not found: {e}"[:60])
            return
        lay.label(text=("Running" if running else "Stopped") + f"  port {live_bridge._state['port']}")
        lay.operator("bnm.bridge_stop" if running else "bnm.bridge_start")
        ses = Session._inst
        if ses:
            lay.label(text=f"Target: {ses.target}")
            lay.label(text=f"Mode: {ses.guard.mode}")
            if live_bridge._state["jobs"]:
                for jid, j in list(live_bridge._state["jobs"].items())[:3]:
                    lay.label(text=f"{j.get('tool')}: {j.get('progress')}"[:60])
            if getattr(ses, "pending_user_edit", False):
                lay.label(text="User edit detected - AI will re-read mesh", icon="INFO")


def _autostart():
    try:
        prefs = bpy.context.preferences.addons[__name__].preferences
        if prefs.autostart and not bpy.app.background:
            _ensure_path()
            from bnm.blender import live_bridge
            live_bridge.start(prefs.port)
    except Exception as e:
        print("BNM autostart failed:", e)
    return None


classes = (BNM_Prefs, BNM_OT_start, BNM_OT_stop, BNM_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.app.timers.register(_autostart, first_interval=1.0)


def unregister():
    try:
        _ensure_path()
        from bnm.blender import live_bridge
        live_bridge.stop()
    except Exception:
        pass
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
