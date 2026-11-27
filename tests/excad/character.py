# SPDX-License-Identifier: GPL-2.0-or-later
"""Set EXCAD_TEST_PHOTO to a local front-facing portrait and enable the MPFB extension."""

import json
import os
from pathlib import Path

import addon_utils
import bpy

addon_utils.enable("bl_ext.blender_org.mpfb", default_set=True)
from excad_addon.ai_bridge import AiBridgeServer
from excad_addon.human_engine import _import_mpfb

run_bridge = AiBridgeServer.instance()._dispatch_dict
for obj in list(bpy.context.scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)


def run(cmd, **args):
    res = run_bridge(dict(cmd=cmd, **args))
    assert res["ok"], (cmd,res)
    return res


def wear(**params):
    return run("character_stylize", object="Portrait", **params)


result = run("human_generate", name="Portrait", rig="game_engine", subdiv_level=0, apply_skin=False)
assert result["engine"] == "MPFB2" and result["rig"], result
human = bpy.data.objects["Portrait"]
rig = human.find_armature()
photo = os.environ["EXCAD_TEST_PHOTO"]
fit = run("fit_face_from_image", object=human.name, image_path=photo)
assert len(fit["measurements"]) == 8 and all(0.15 <= v <= 0.5 for v in fit["weights"].values())
for name, value in fit["weights"].items():
    assert abs(human.data.shape_keys.key_blocks[name].value-value) < 1e-5
    if "/l-" in name:
        assert fit["weights"][name.replace("/l-", "/r-")] == value
print("PHOTO FIT", fit, flush=True)
for tone in ("FAIR_IVORY", "TAN", "DEEP", "WARM_BEIGE"):
    wear(skin=tone)
assert human.data.color_attributes.get("EXCAD Lips")
for name in ("SMILE","SURPRISED","WINK","SERIOUS","SMILE"):
    wear(expression=name)
    assert json.loads(human["excad_expression"])
for name in ("FIST","POINT","OPEN","RELAXED"):
    assert wear(hands=name)["posed_fingers"] == 30
for name in ("system","shirts","pants","skirts"):
    dest = Path(run("install_asset_pack",pack=name)["path"])
    assert dest.is_dir() and any(dest.rglob("*.mhclo"))
assert run("equip_asset",human="Portrait",name="eyebrow001")["name"]

# Isolate outfits between trials, including MPFB's body occlusion masks.
# Reload the same fitted character so repeated poses do not mask other tests.
fixture = str(Path(bpy.app.tempdir)/"excad-wearable-test.blend")
bpy.ops.wm.save_as_mainfile(filepath=fixture, copy=True)
for style in ("SHORT","PONYTAIL","LONG_STRAIGHT","LONG_WAVY","CURTAIN_BANGS","BOB"):
    bpy.ops.wm.open_mainfile(filepath=fixture)
    added = wear(hair=style)["added"]
    for name in added:
        obj = bpy.data.objects[name]
        assert obj.data.polygons and obj.vertex_groups.get("head") and obj.find_armature(), name
    print("HAIR",style,flush=True)
for style in ("TSHIRT","VNECK_SWEATER","HOODIE","JACKET","PANTS","SKIRT","SHOES"):
    bpy.ops.wm.open_mainfile(filepath=fixture)
    obj = bpy.data.objects[wear(clothes=style)["added"][0]]
    assert obj.data.polygons and obj.find_armature() and any(v.groups for v in obj.data.vertices), style
    print("GARMENT",style,flush=True)
bpy.ops.wm.open_mainfile(filepath=fixture)
wear(hair="BOB", clothes=["TSHIRT","PANTS","SHOES"], iris_color=[0.15,0.075,0.025], lashes=True, teeth=True)
out = Path(os.getenv("EXCAD_TEST_OUTPUT",bpy.app.tempdir))
out.mkdir(parents=True, exist_ok=True)
inspect = run("render_inspect", target="Portrait", size=320, path=str(out/"phase2-character.png"))
print("CHARACTER INSPECTION", inspect, flush=True)
assert Path(inspect["path"]).stat().st_size > 1000
assert len(AiBridgeServer.instance()._dispatch('{"cmd":"schema"}').encode()) <= 1200
print("CHARACTER PASS", bpy.app.version_string)
