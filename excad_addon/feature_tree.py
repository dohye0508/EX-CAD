# SPDX-License-Identifier: GPL-2.0-or-later
"""Replay ordered CAD features in isolation before replacing the visible mesh."""

import copy
import json
import math
import os
from pathlib import Path
import tempfile

import bmesh
import bpy
from mathutils import Matrix, Vector

from . import feature_geometry as geo
from .sketch_engine import SketchDocument

FEATURES = {"EXTRUDE", "REVOLVE", "LOFT", "SWEEP", "FILLET", "CHAMFER", "SHELL", "PATTERN", "MIRROR3D"}
SHAPES = {"ARC", "SLOT", "REGULAR_POLYGON", "SPLINE", "OFFSET_LOOP", "MIRROR2D"}


def sketch_from_spec(spec, body, owner_name):
    if "entities" in spec:
        doc = SketchDocument.deserialize(spec)
    else:
        doc = SketchDocument(spec.get("plane", "XY"))
        doc.frame, doc.support = spec.get("frame"), spec.get("support")
        for shape in spec["shapes"]:
            args = dict(shape)
            kind = args.pop("type").upper()
            if kind == "RECT":
                width, height = args["width"], args["height"]
                center = args.get("center", (0, 0))
                doc.add_rect(center[0]-width/2, center[1]-height/2, width, height)
            elif kind == "CIRCLE":
                doc.add_circle(*args.get("center", (0, 0)), args["radius"])
            elif kind == "LINE":
                doc.add_line(*args["start"], *args["end"])
            else:
                doc.add_shape(kind, **args)
        for constraint in spec.get("constraints", []):
            doc.add_constraint(**constraint)
    if doc.support:
        name, side = doc.support["object"], doc.support["face"]
        target = body if name in {"SELF", owner_name} else bpy.context.scene.objects.get(name)
        if target is None:
            raise ValueError(f"Missing support body {name}")
        doc.frame = geo.selected_face(target, side)
    if spec.get("offset"):
        frame = Matrix(doc.frame) if doc.frame else Matrix((
            Vector(doc.plane_to_world(1, 0, 0)), Vector(doc.plane_to_world(0, 1, 0)),
            Vector(doc.plane_to_world(0, 0, 1)))).transposed().to_4x4()
        frame.translation += float(spec["offset"]) * frame.col[2].xyz
        doc.frame = [list(row) for row in frame]
    return doc


def _mesh_copy(objects, source, transform):
    return geo.new_mesh(objects, [transform @ v.co for v in source.vertices],
                        [tuple(p.vertices) for p in source.polygons])


def _pattern(objects, body, feature):
    kind = feature.get("pattern_type", feature.get("mode", "LINEAR")).upper()
    count = int(feature.get("count", 2))
    if not 2 <= count <= 256:
        raise ValueError("Pattern count must be between 2 and 256, including the original")
    axis = geo.axis_vector(feature.get("axis", "X" if kind == "LINEAR" else "Z"))
    pivot = Vector(feature.get("origin", (0, 0, 0)))
    original = body.data.copy()
    try:
        for i in range(1, count):
            if kind == "LINEAR":
                matrix = Matrix.Translation(axis * (float(feature.get("spacing", 1)) * i))
            elif kind == "CIRCULAR":
                angle = float(feature.get("angle_deg", 360))
                divisor = count if abs(angle) == 360 else count-1
                matrix = (Matrix.Translation(pivot) @ Matrix.Rotation(math.radians(angle)*i/divisor, 4, axis)
                          @ Matrix.Translation(-pivot))
            else:
                raise ValueError("Pattern type must be LINEAR or CIRCULAR")
            geo.boolean(body, _mesh_copy(objects, original, matrix), "ADD")
    finally:
        bpy.data.meshes.remove(original)


def _solid(mesh):
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        if not bm.faces or any(not e.is_manifold for e in bm.edges) or any(not v.is_manifold for v in bm.verts):
            raise ValueError("Feature produced an empty or non-manifold solid")
        if abs(bm.calc_volume(signed=True)) < 1e-12:
            raise ValueError("Feature produced zero volume")
    finally:
        bm.free()


class FeatureTree:
    def __init__(self, name, recipe=None):
        self.name = name
        self.owner = bpy.context.scene.objects.get(name)
        if self.owner and self.owner.type != "MESH":
            raise ValueError("Feature tree owner must be a mesh")
        if recipe is not None:
            self.recipe = copy.deepcopy(recipe)
        elif self.owner and "excad_tree" in self.owner:
            self.recipe = json.loads(self.owner["excad_tree"])
        else:
            self.recipe = {"version": 1, "sketches": {}, "features": []}
            if self.owner:
                mesh, matrix = self.owner.data, self.owner.matrix_world
                self.recipe["base"] = {"vertices": [list(matrix @ v.co) for v in mesh.vertices],
                                       "faces": [list(p.vertices) for p in mesh.polygons]}
                self.recipe["model_frame"] = [list(row) for row in matrix]
        if self.recipe.get("version", 1) != 1:
            raise ValueError("Unsupported feature tree version")

    def append(self, kind, args):
        feature = dict(args, type=kind)
        sketch = bpy.context.scene.get("excad_active_sketch", "Sketch")
        if kind in {"EXTRUDE", "REVOLVE", "SWEEP"}:
            feature.setdefault("sketch", feature.pop("profile_sketch", sketch))
        ids = feature.get("sketch_ids", []) if kind == "LOFT" else [feature["sketch"]] if "sketch" in feature else []
        for name in ids:
            record = bpy.context.scene.get("excad_sketches", {}).get(name)
            if name not in self.recipe["sketches"]:
                if not record:
                    raise ValueError(f"Unknown sketch {name}")
                self.recipe["sketches"][name] = json.loads(record)
            elif record and name == sketch:
                # A reset of the unnamed active sketch must not redefine an earlier feature.
                spec = json.loads(record)
                if spec != self.recipe["sketches"][name]:
                    new_name = f"{name}@{len(self.recipe['features'])+1}"
                    self.recipe["sketches"][new_name] = spec
                    if kind == "LOFT":
                        feature["sketch_ids"] = [new_name if n == name else n for n in ids]
                    else:
                        feature["sketch"] = new_name
        if kind == "SWEEP" and isinstance(feature.get("path_curve"), str):
            feature["path_curve"] = [list(p) for p in geo.path_points(feature["path_curve"])]
        feature.setdefault("id", f"F{len(self.recipe['features'])+1}")
        self.recipe["features"].append(feature)

    def rebuild(self):
        recipe = self.recipe
        sketches = recipe.get("sketches", {})
        with geo.construction() as objects:
            base = recipe.get("base")
            body = geo.new_mesh(objects, base["vertices"], base["faces"]) if base else None
            for i, feature in enumerate(recipe["features"]):
                kind = feature["type"].upper()
                try:
                    if kind in {"EXTRUDE", "REVOLVE", "SWEEP", "LOFT"}:
                        ids = feature["sketch_ids"] if kind == "LOFT" else [feature["sketch"]]
                        docs = [sketch_from_spec(sketches[key], body, self.name) for key in ids]
                        if kind == "EXTRUDE":
                            # Overlap the supporting face by a few mesh-coordinate ULPs.
                            # A microscopically recessed cutter otherwise leaves a zero-thickness lid.
                            overlap = 0
                            if body and docs[0].support:
                                overlap = max(max(body.dimensions), abs(feature["depth"]), 1e-3) * 2e-6
                            tool = geo.extrude(objects, docs[0], feature["depth"], feature.get("symmetric", False),
                                               feature.get("draft_angle", 0), overlap)
                        elif kind == "REVOLVE":
                            tool = geo.revolve(objects, docs[0], feature.get("axis"), feature.get("angle_deg", 360),
                                               feature.get("segments", 64), feature.get("origin"))
                        elif kind == "LOFT":
                            tool = geo.loft(objects, docs, feature.get("segments", 64))
                        else:
                            tool = geo.sweep(objects, docs[0], feature["path_curve"])
                        op = feature.get("op", "ADD").upper()
                        if body is None:
                            if op != "ADD":
                                raise ValueError("First feature must add a solid")
                            body = tool
                        else:
                            geo.boolean(body, tool, op)
                    elif body is None:
                        raise ValueError("Feature requires a base solid")
                    elif kind in {"FILLET", "CHAMFER"}:
                        width = feature["radius"] if kind == "FILLET" else feature["width"]
                        geo.finish_edges(body, feature.get("edges", "ALL"), width,
                                         feature.get("segments", 5) if kind == "FILLET" else 1)
                    elif kind == "SHELL":
                        geo.shell(body, feature["thickness"], feature.get("open_faces", ["TOP"]))
                    elif kind == "PATTERN":
                        _pattern(objects, body, feature)
                    elif kind == "MIRROR3D":
                        axis = geo.axis_vector(feature.get("axis", "X"))
                        mirror = Matrix.Identity(3)
                        for row in range(3):
                            for col in range(3):
                                mirror[row][col] -= 2*axis[row]*axis[col]
                        matrix = mirror.to_4x4()
                        matrix.translation = 2*float(feature.get("offset", 0))*axis
                        geo.boolean(body, _mesh_copy(objects, body.data, matrix), "ADD")
                    else:
                        raise ValueError(f"Unknown feature {kind}")
                    _solid(body.data)
                except (KeyError, IndexError, ValueError, RuntimeError) as exc:
                    raise ValueError(f"Feature {feature.get('id', i+1)} ({kind}): {exc}") from exc
            if body is None:
                raise ValueError("Feature tree is empty")
            raw = json.dumps(recipe, separators=(",", ":"), allow_nan=False)
            result = body.data.copy()
            if "model_frame" in recipe:
                result.transform(Matrix(recipe["model_frame"]).inverted())
            if self.owner is None:
                self.owner = bpy.data.objects.new(self.name, result)
                bpy.context.collection.objects.link(self.owner)
            else:
                old = self.owner.data
                for material in old.materials:
                    result.materials.append(material)
                self.owner.data = result
                if old.users == 0:
                    bpy.data.meshes.remove(old)
            result.name = f"{self.owner.name} Solid"
            self.owner["excad_tree"] = raw
            bpy.context.view_layer.objects.active = self.owner
            self.owner.select_set(True)
        return self.owner


def save_recipe(obj, path):
    path = Path(bpy.path.abspath(path)).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.loads(obj["excad_tree"])
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=path.parent, delete=False) as f:
        temp = f.name
        try:
            json.dump(raw, f, indent=2, allow_nan=False)
        except Exception:
            f.close()
            os.unlink(temp)
            raise
    try:
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
    return str(path)


def dispatch(cmd, data):
    args = {k: v for k, v in data.items() if k != "cmd"}
    doc = SketchDocument.active()
    if cmd == "sketch_on_face":
        name, face = args["object"], args.get("face", "TOP").upper()
        frame = geo.selected_face(bpy.context.scene.objects[name], face)
        bpy.context.scene["excad_active_sketch"] = args.get("sketch_id", "Sketch")
        doc = SketchDocument.active()
        doc.reset("FACE")
        doc.frame, doc.support = frame, {"object": name, "face": face}
        doc.save()
        return {"plane": doc.plane, "frame": frame}
    if cmd == "sketch_constraint":
        return {"constraint": doc.add_constraint(**args)}
    if cmd.startswith("sketch_") and cmd[7:].removeprefix("add_").upper() in SHAPES:
        return {"entity": doc.add_shape(cmd[7:].removeprefix("add_").upper(), **args)}
    if cmd in {"feature_tree", "rebuild_feature_tree", "save_feature_tree"} or cmd.upper() in FEATURES:
        name = args.pop("object", None) or args.pop("name", None)
        if not name:
            name = doc.support["object"] if doc.support else "Body1"
        if cmd == "save_feature_tree":
            path = args.get("path") or str(Path(bpy.data.filepath or bpy.app.tempdir).parent / "feature_tree.json")
            return {"path": save_recipe(bpy.context.scene.objects[name], path)}
        recipe = args.pop("recipe", None)
        if "path" in args:
            recipe = json.loads(Path(bpy.path.abspath(args.pop("path"))).read_text(encoding="utf8"))
        tree = FeatureTree(name, recipe)
        if cmd.upper() in FEATURES:
            if cmd == "pattern" and "type" in args:
                args["pattern_type"] = args.pop("type")
            tree.append(cmd.upper(), args)
        for key, value in args.get("parameters", {}).items():
            feature_id, field = key.split(".", 1)
            matches = [f for f in tree.recipe["features"] if f.get("id") == feature_id]
            if len(matches) != 1 or field not in matches[0] or field in {"type", "id"}:
                raise ValueError(f"Unknown feature parameter {key}")
            matches[0][field] = value
        obj = tree.rebuild()
        return {"name": obj.name, "features": len(tree.recipe["features"]), "verts": len(obj.data.vertices),
                "faces": len(obj.data.polygons)}
    return None
