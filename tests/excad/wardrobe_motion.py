# SPDX-License-Identifier: GPL-2.0-or-later
"""Check assembled clothes and head-weighted hair across MPFB rigs and body proportions."""

import os
from pathlib import Path
import tempfile

import addon_utils
import bpy

from excad_addon.ai_bridge import AiBridgeServer
from excad_addon.oneshot_generators import _BONE_ALIASES

addon_utils.enable("bl_ext.blender_org.mpfb", default_set=True)
bridge = AiBridgeServer.instance()
cases = [
    ("REALISTIC_MALE", "default", "SHORT", ["TSHIRT", "JACKET", "PANTS", "SHOES"], "TAN", "SMILE", {}),
    ("HEROIC_SCULPT", "game_engine", "PONYTAIL", ["HOODIE", "PANTS", "SHOES"], "DEEP", "SERIOUS", {}),
    ("REALISTIC_FEMALE", "default", "LONG_WAVY", ["VNECK_SWEATER", "SKIRT", "SHOES"], "WARM_BEIGE", "WINK", {}),
    (
        "CUSTOM",
        "game_engine",
        "CURTAIN_BANGS",
        ["TSHIRT", "JACKET", "PANTS", "SHOES"],
        "FAIR_IVORY",
        "SURPRISED",
        {"gender": 0.6, "weight": 0.78, "muscle": 0.3, "height": 0.7, "age": 0.62},
    ),
    ("ATHLETIC_MALE", "default", "BOB", ["VNECK_SWEATER", "PANTS", "SHOES"], "DEEP", "SERIOUS", {}),
    ("ATHLETIC_FEMALE", "game_engine", "LONG_STRAIGHT", ["HOODIE", "SKIRT", "SHOES"], "FAIR_IVORY", "SMILE", {}),
]


def run(cmd, **args):
    response = bridge._dispatch_dict(dict(cmd=cmd, **args))
    assert response["ok"], (cmd, response)
    return response


with tempfile.TemporaryDirectory() as tmp:
    image_dir = Path(os.getenv("EXCAD_TEST_VARIANTS", tmp))
    image_dir.mkdir(parents=True, exist_ok=True)
    for obj in list(bpy.context.scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    fixture = str(Path(tmp) / "empty.blend")
    bpy.ops.wm.save_as_mainfile(filepath=fixture, copy=True)
    measured = []
    for preset, rig, hair, clothes, skin, expression, options in cases:
        bpy.ops.wm.open_mainfile(filepath=fixture)
        run(
            "batch",
            steps=[
                dict(
                    cmd="human_generate",
                    name="Person",
                    preset=preset,
                    rig=rig,
                    apply_skin=False,
                    subdiv_level=1,
                    **options,
                ),
                dict(cmd="fit_face_from_image", object="Person", image_path=os.environ["EXCAD_TEST_PHOTO"]),
                dict(
                    cmd="character_stylize",
                    object="Person",
                    hair=hair,
                    clothes=clothes,
                    skin=skin,
                    expression=expression,
                    iris_color=[0.18, 0.08, 0.035],
                    teeth=True,
                    lashes=True,
                    hands="RELAXED",
                ),
            ],
        )
        obj = bpy.context.scene.objects["Person"]
        bones = obj.find_armature().pose.bones

        def bone(role):
            return next(bones[name] for name in _BONE_ALIASES[role] if name in bones)

        for pose in ("RESET", "RELAXED", "HEROIC", "WALK", "WAVE"):
            assert run("oneshot_pose", object=obj.name, pose=pose)["posed_bones"] > 0
            run("place_relative", obj=obj.name, relation="ALIGN_FLOOR")
            preview = run("render_inspect", target=obj.name, path=str(image_dir / f"{preset}-{pose}.png"))
            assert not preview["warnings"], (preset, pose, preview["warnings"])
            measured.append(preview["elapsed_ms"])
        # The waving wrist belongs above and outside the right shoulder in both rigs.
        shoulder = bone("upperarm_r").head
        outward = (shoulder - bone("upperarm_l").head).normalized()
        up = bone("head").head - (shoulder + bone("upperarm_l").head) / 2
        wrist = bone("forearm_r").tail
        assert (wrist - shoulder).dot(up) > 0 and (wrist - shoulder).dot(outward) > 0
        if "JACKET" in clothes:
            shirt = next(o for o in bpy.context.scene.objects if o.get("excad_slot") == "top")
            covered = len(shirt.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices)
            assert 0 < covered < len(shirt.data.vertices)
            run("character_stylize", object=obj.name, remove=["outerwear"])
            assert not shirt.get("excad_underlay_group")
            assert len(shirt.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices) > covered
            assert not any(o.get("excad_slot") == "outerwear" for o in bpy.context.scene.objects)
        print("WARDROBE", preset, rig, flush=True)
    print("WARDROBE MOTION PASS", len(measured), "previews", max(measured), "max ms", flush=True)
