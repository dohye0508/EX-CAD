# SPDX-License-Identifier: GPL-2.0-or-later
"""Run with Blender --background --factory-startup --python-exit-code 1 --python FILE."""

import json
import sys
from pathlib import Path

import bpy

from excad_addon.ai_bridge import AiBridgeServer

bridge = AiBridgeServer.instance()
run = bridge._dispatch_dict
schema = bridge._dispatch('{"cmd":"schema"}').encode("utf8")
assert len(schema) <= 1200, len(schema)
steps = [{"cmd": "ping"}, {"cmd": "absent"}, {"cmd": "list"}]
response = run({"cmd": "batch", "steps": steps, "return_all": True, "stop_on_error": False})
assert not response["ok"] and response["steps_run"] == 3 and response["last"]["ok"], response
assert run({"cmd": "batch", "steps": steps})["steps_run"] == 2
code = 'S XY; R 0 0 2 1; sketch_extrude {"depth":0.5,"name":"A; B"}; DIGEST'
result = run({"cmd": "dsl", "code": code})
assert result["ok"] and result["n"] == 4 and bpy.data.objects.get("A; B"), result
digest = run({"cmd": "scene_digest", "since_hash": result["last"]["h"]})
assert digest["unchanged"]
print("PROTOCOL PASS", bpy.app.version_string, len(schema), "schema bytes")
