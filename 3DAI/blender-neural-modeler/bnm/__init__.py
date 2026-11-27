"""blender-neural-modeler (bnm): autonomous scratch-sculpting engine for Blender.

All geometry code runs inside Blender's bundled Python (bpy + numpy + mathutils).
"""
import os

__version__ = "0.1.0"
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PKG_DIR)
CONFIG_DIR = os.path.join(ROOT_DIR, "config")


def load_config(name):
    import json
    with open(os.path.join(CONFIG_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)
