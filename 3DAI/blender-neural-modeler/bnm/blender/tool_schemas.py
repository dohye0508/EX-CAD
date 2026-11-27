"""Tool names, descriptions and JSON schemas.  Pure Python (no bpy): imported by the MCP server
(outside Blender) and by the in-Blender tool registry (inside Blender)."""

V3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
PATH = {"type": "array", "items": V3, "description": "Object-space points; for drag brushes the drag trajectory starting on the surface"}
OBJ = {"type": "string", "description": "Mesh object name (default: active sculpt target)"}
REGION = {"type": "string", "description": "Semantic region name, e.g. nose_tip, upper_lip, orbit, cheekbone"}
VIEWS = {"type": "array", "items": {"type": "string", "enum": ["front", "back", "left", "right", "three_quarter_left",
                                                              "three_quarter_right", "top", "bottom", "low_front", "high_front"]}}
MODES = {"type": "array", "items": {"type": "string", "enum": ["clay", "matcap", "silhouette", "wireframe", "depth", "normal", "curvature"]}}
BRUSH = {"type": "string", "description": "DRAW, DRAW_SHARP, DRAW_SUBTRACT, CLAY, CLAY_STRIPS, CLAY_THUMB, INFLATE, BLOB, LAYER, CREASE, "
                                          "PINCH, FLATTEN, SCRAPE, FILL, SMOOTH, SURFACE_SMOOTH, RELAX, POLISH, MASK, UNMASK, GRAB, "
                                          "ELASTIC_DEFORM, SNAKE_HOOK, NUDGE, THUMB (POSE via deform_surface)"}
STROKE_PROPS = {
    "object": OBJ, "brush": BRUSH, "path": PATH,
    "path_spec": {"type": "object", "description": "Alternative to path: {kind: line|polyline|bezier|catmull_rom|arc|ellipse|fill|dot, ...}"},
    "surface_xz": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Alternative: (x,z) pairs projected onto the surface by front rays (Z up, face -Y)"},
    "radius": {"type": "number"}, "strength": {"type": "number"}, "falloff": {"type": "string"},
    "spacing": {"type": "number", "description": "dab spacing as fraction of radius"},
    "pressure": {"type": "string", "enum": ["taper", "constant", "ramp_up", "ramp_down", "peak", "custom"]},
    "pressure_values": {"type": "array", "items": {"type": "number"}},
    "radius_scale": {"type": "array", "items": {"type": "number"}}, "direction": {"type": "integer", "enum": [1, -1]},
    "symmetry_x": {"type": "boolean"}, "geodesic": {"type": "boolean"}, "frontface": {"type": "boolean"},
    "normal": V3, "region": REGION, "protect_regions": {"type": "array", "items": {"type": "string"}},
    "engine": {"type": "string", "enum": ["auto", "virtual", "native"]},
    "coordinate_space": {"type": "string", "enum": ["OBJECT", "WORLD"]},
}


def T(desc, props=None, required=()):
    return {"description": desc, "inputSchema": {"type": "object", "properties": props or {}, "required": list(required)}}


TOOLS = {
    # ---- session / environment
    "get_capabilities": T("Blender version, GPU, native-sculpt availability, brush assets, optional ML modules."),
    "get_status": T("Session state: target object, mode, user-edit detection, checkpoints, semantic regions, brush stats."),
    "set_mode": T("Execution mode. SCRATCH_SCULPT (default) forbids external meshes and generators.",
                  {"mode": {"type": "string"}, "user_permissions": {"type": "array", "items": {"type": "string"}}}, ["mode"]),
    "set_target": T("Choose the mesh object to sculpt and (optionally) attach a semantic schema (face|creature|none).",
                    {"object": OBJ, "schema": {"type": "string"}}, ["object"]),
    "audit_no_external_mesh": T("Benchmark I audit: every mesh has primitive/derived provenance, no linked libraries, no forbidden ops."),
    # ---- base mesh & structure
    "create_initial_sphere": T("Create an icosphere/UV sphere as the sculpt base (Stage 0).",
                               {"name": {"type": "string"}, "radius": {"type": "number"}, "kind": {"type": "string", "enum": ["ICOSPHERE", "UV_SPHERE"]},
                                "detail": {"type": "integer"}, "schema": {"type": "string"}}),
    "create_primitive": T("Create a primitive (ICOSPHERE, UV_SPHERE, CUBE, CYLINDER, PLANE, GRID).",
                          {"kind": {"type": "string"}, "name": {"type": "string"}, "size": {"type": "number"},
                           "location": V3, "detail": {"type": "integer"}}, ["kind"]),
    "analyze_mesh": T("Compact spatial summary (bbox, area, volume, symmetry, components, protrusions/depressions, curvature stats).",
                      {"object": OBJ, "detail": {"type": "boolean"}}),
    "analyze_surface": T("Local surface summary around a point: normal, curvature, relief, neighbourhood size.",
                         {"object": OBJ, "point": V3, "radius": {"type": "number"}}, ["point"]),
    "get_surface_region": T("Vertices/centre/normal of a sphere region on the surface (optionally geodesic).",
                            {"object": OBJ, "center": V3, "radius": {"type": "number"}, "geodesic": {"type": "boolean"}}, ["center", "radius"]),
    "extract_cross_section": T("Plane cross-section: perimeter and in-plane extents.",
                               {"object": OBJ, "origin": V3, "normal": V3}, ["origin", "normal"]),
    "estimate_curvature": T("Mean/Gaussian/principal curvature statistics (whole mesh or a semantic region).",
                            {"object": OBJ, "region": REGION}),
    # ---- brush state
    "set_sculpt_mode": T("Enter/leave Blender Sculpt Mode (native engine needs a UI session).", {"object": OBJ, "enable": {"type": "boolean"}}),
    "select_sculpt_brush": T("Set the default brush for following strokes.", {"brush": BRUSH}, ["brush"]),
    "set_brush_radius": T("Default brush radius (object units).", {"radius": {"type": "number"}}, ["radius"]),
    "set_brush_strength": T("Default brush strength 0..1.", {"strength": {"type": "number"}}, ["strength"]),
    "set_brush_falloff": T("Default falloff: LINEAR SMOOTH SMOOTHER GAUSSIAN POLYNOMIAL SHARP ROOT SPHERE CONSTANT or [[t,v],...].",
                           {"falloff": {}}, ["falloff"]),
    # ---- sculpting
    "execute_brush_stroke": T("Execute ONE continuous brush stroke along a surface path; returns measured change (moved vertices, "
                              "max/normal displacement, volume delta, change centroid).", STROKE_PROPS),
    "execute_stroke_sequence": T("Execute several strokes in order (shown live, one per viewport tick).",
                                 {"object": OBJ, "strokes": {"type": "array", "items": {"type": "object"}}}, ["strokes"]),
    "sculpt_along_curve": T("Stroke along a smooth Catmull-Rom/Bezier curve through control points.",
                            {**STROKE_PROPS, "control_points": PATH, "curve": {"type": "string", "enum": ["catmull_rom", "bezier"]}}, ["control_points"]),
    "sculpt_surface_region": T("Zig-zag fill strokes over a semantic region with a brush.",
                               {"object": OBJ, "region": REGION, "brush": BRUSH, "strength": {"type": "number"}, "radius": {"type": "number"}}, ["region", "brush"]),
    "add_surface_volume": T("Build clay on a semantic region or along a path (CLAY_STRIPS + blend).",
                            {"object": OBJ, "region": REGION, "path": PATH, "amount": {"type": "number"}, "radius": {"type": "number"}}),
    "remove_surface_volume": T("Carve/scrape volume on a region or path.", {"object": OBJ, "region": REGION, "path": PATH,
                                                                           "amount": {"type": "number"}, "radius": {"type": "number"}}),
    "deform_surface": T("Grab/elastic/snake-hook a point by a drag vector, or POSE-rotate a region.",
                        {"object": OBJ, "brush": {"type": "string", "enum": ["GRAB", "ELASTIC_DEFORM", "SNAKE_HOOK", "POSE"]},
                         "point": V3, "drag": V3, "radius": {"type": "number"}, "region": REGION, "pivot": V3, "axis": V3,
                         "angle": {"type": "number"}, "symmetry_x": {"type": "boolean"}}, ["brush"]),
    "smooth_surface": T("Smooth a region/path (SMOOTH, SURFACE_SMOOTH, RELAX) with optional curvature feature protection.",
                        {"object": OBJ, "region": REGION, "path": PATH, "kind": {"type": "string"}, "strength": {"type": "number"},
                         "radius": {"type": "number"}, "protect_features": {"type": "boolean"}}),
    "pinch_surface": T("Pinch along a path (sharpen an edge).", {"object": OBJ, "path": PATH, "radius": {"type": "number"}, "strength": {"type": "number"}}, ["path"]),
    "flatten_surface": T("Flatten/scrape/fill along a path (plane changes).", {"object": OBJ, "path": PATH, "radius": {"type": "number"},
                                                                               "strength": {"type": "number"}, "kind": {"type": "string"}, "normal": V3}, ["path"]),
    "sdf_volume_edit": T("SDF assist: add/remove a smooth clay tube along a path, or offset the surface, then re-extract the mesh.",
                         {"object": OBJ, "operation": {"type": "string", "enum": ["add", "remove", "offset", "repair"]}, "path": PATH,
                          "radius": {"type": "number"}, "blend": {"type": "number"}, "amount": {"type": "number"}, "voxel": {"type": "number"}}, ["operation"]),
    "undo_stroke": T("Undo the last N strokes (coordinate snapshot).", {"steps": {"type": "integer"}}),
    # ---- spatial understanding
    "raycast_surface": T("Ray cast in object space -> hit point/normal/vertex.", {"object": OBJ, "origin": V3, "direction": V3}, ["origin", "direction"]),
    "screen_to_surface": T("Pixel (u,v) in a rendered view -> 3D surface point.", {"object": OBJ, "view": {"type": "string"}, "u": {"type": "number"},
                                                                                     "v": {"type": "number"}, "size": {"type": "integer"}}, ["view", "u", "v"]),
    "project_landmark": T("Project landmarks (names or points) into a view's pixel coordinates.",
                          {"object": OBJ, "view": {"type": "string"}, "names": {"type": "array", "items": {"type": "string"}}, "size": {"type": "integer"}}, ["view"]),
    "get_surface_normal": T("Surface normal near a point.", {"object": OBJ, "point": V3}, ["point"]),
    "get_semantic_region": T("Region info: status (planned/measured), landmarks, centre, normal, curvature, adjacency, allowed edits.",
                             {"object": OBJ, "region": REGION}, ["region"]),
    "get_geodesic_region": T("Vertices within a surface (geodesic) distance from a point.", {"object": OBJ, "center": V3, "radius": {"type": "number"}}, ["center", "radius"]),
    "create_region_mask": T("Write a sculpt mask from a semantic region, a geodesic ball or an image area (view+circle/polygon).",
                            {"object": OBJ, "region": REGION, "view": {"type": "string"}, "circle": {"type": "array", "items": {"type": "number"}},
                             "polygon": {"type": "array"}, "center": V3, "radius": {"type": "number"}, "invert": {"type": "boolean"},
                             "clear": {"type": "boolean"}}),
    "protect_region": T("Protect regions from subsequent strokes (0..1 weights); empty list clears.",
                        {"object": OBJ, "regions": {"type": "array", "items": {"type": "string"}}}, ["regions"]),
    # ---- resolution
    "voxel_remesh": T("Voxel remesh (voxel size or target vertex count). Rebuilds semantic regions.",
                      {"object": OBJ, "voxel_size": {"type": "number"}, "target_vertices": {"type": "integer"}}),
    "adaptive_subdivide": T("Subdivide only a region.", {"object": OBJ, "region": REGION, "cuts": {"type": "integer"}}, ["region"]),
    "set_multiresolution": T("Add/increase a Multiresolution modifier.", {"object": OBJ, "levels": {"type": "integer"}}),
    "optimize_topology": T("Tangential relax over the mesh (even edge flow) without changing shape.", {"object": OBJ, "iterations": {"type": "integer"}}),
    "validate_mesh": T("Non-manifold, boundary, degenerate, duplicate, winding, inside-out, self-intersection checks.", {"object": OBJ}),
    # ---- visual feedback
    "capture_viewport": T("Screenshot of the live Blender 3D viewport (UI sessions only).", {"path": {"type": "string"}}),
    "render_front": T("Render front view.", {"object": OBJ, "modes": MODES, "size": {"type": "integer"}}),
    "render_side": T("Render left side view.", {"object": OBJ, "modes": MODES, "size": {"type": "integer"}}),
    "render_three_quarter": T("Render 3/4 view.", {"object": OBJ, "modes": MODES, "size": {"type": "integer"}}),
    "render_multiview": T("Render several views and modes (clay/matcap/silhouette/wireframe/depth/normal/curvature); returns image paths.",
                          {"object": OBJ, "views": VIEWS, "modes": MODES, "size": {"type": "integer"}, "blender_render": {"type": "boolean"}}),
    "render_depth": T("Depth render.", {"object": OBJ, "views": VIEWS, "size": {"type": "integer"}}),
    "render_normals": T("Normal render.", {"object": OBJ, "views": VIEWS, "size": {"type": "integer"}}),
    "analyze_reference_image": T("Analyse a user reference image: silhouette, width/side profile, normalised landmarks -> structural spec.",
                                 {"path": {"type": "string"}, "view": {"type": "string"}, "facing": {"type": "string"},
                                  "landmarks_px": {"type": "object"}, "side_path": {"type": "string"}, "side_landmarks_px": {"type": "object"}}, ["path"]),
    "compare_reference": T("Compare model silhouette to a reference image in the same view (IoU, per-row width error).",
                           {"object": OBJ, "path": {"type": "string"}, "view": {"type": "string"}, "size": {"type": "integer"}}, ["path"]),
    "identify_shape_errors": T("Measured error list vs the proportion spec (face) + protrusion/depression/symmetry anomalies.", {"object": OBJ}),
    # ---- iteration
    "plan_sculpt_stroke": T("Natural-language edit ('코끝을 약간 위로') or {region,intent,amount} -> executable stroke plan JSON.",
                            {"object": OBJ, "command": {"type": "string"}, "region": REGION, "intent": {"type": "string"},
                             "amount": {"type": "number"}, "execute": {"type": "boolean"}}),
    "execute_plan": T("Execute plan items with protection, leakage/direction verification and automatic rollback.",
                      {"object": OBJ, "plan": {"type": "array", "items": {"type": "object"}}, "max_leak": {"type": "number"}}, ["plan"]),
    "optimize_stroke_parameters": T("Candidate search over grab gain/radius to move a point to a target position.",
                                    {"object": OBJ, "point": V3, "target": V3}, ["point", "target"]),
    "evaluate_sculpt_result": T("Before/after metrics against the last checkpoint: displacement stats, volume, validity, region change.",
                                {"object": OBJ, "checkpoint": {"type": "string"}, "region": REGION}),
    "create_checkpoint": T("Full mesh checkpoint (survives remesh).", {"object": OBJ, "label": {"type": "string"}}, ["label"]),
    "restore_checkpoint": T("Restore a checkpoint.", {"object": OBJ, "label": {"type": "string"}}, ["label"]),
    "repeat_sculpt_pass": T("Repeat a stroke/plan N times, stopping when a measured metric stops improving.",
                            {"object": OBJ, "stroke": {"type": "object"}, "plan": {"type": "array"}, "times": {"type": "integer"}}),
    "run_head_stage": T("Run one stage (or all) of the scratch Sphere->Head pipeline on the target.",
                        {"object": OBJ, "stage": {"type": "string"}, "canon_overrides": {"type": "object"}, "render": {"type": "boolean"}}, ["stage"]),
    "run_video_workflow_head": T("Sculpt a head from a sphere following the observed professional workflow (knowledge/video_workflow_lAspS2BK2Vk.md): "
                                 "primary-form target -> sockets -> nose volume -> mouth/chin -> side planes -> lips -> eyes -> profile -> "
                                 "ears/neck from separate primitives -> optional hair + glasses. Renders after every stage.",
                                 {"name": {"type": "string"}, "canon_overrides": {"type": "object"}, "until": {"type": "string"},
                                  "hair": {"type": "boolean"}, "glasses": {"type": "boolean"}}),
    "overlay_reference": T("Align a reference photo to the model's front render by the eye centres and save [photo | model | blend] "
                           "with per-landmark errors (photo_landmarks: pixel coords incl. eye_L_center, eye_R_center).",
                           {"object": OBJ, "path": {"type": "string"}, "photo_landmarks": {"type": "object"}, "size": {"type": "integer"}},
                           ["path", "photo_landmarks"]),
    "smooth_region": T("Many-iteration taubin smoothing over a semantic region, a ball, or the whole mesh (no shrinkage).",
                       {"object": OBJ, "region": REGION, "center": V3, "radius": {"type": "number"}, "iterations": {"type": "integer"},
                        "factor": {"type": "number"}}),
    "run_creature_head": T("Organic workflow: sculpt a creature/animal head from a sphere with a schema-driven plan.",
                           {"name": {"type": "string"}, "params": {"type": "object"}}),
    "hard_surface_part": T("Hard-surface workflow from a cylinder/cube (edit-mode ops, bevel, boolean, array, mirror).",
                           {"kind": {"type": "string", "enum": ["flanged_gear_hub", "bracket"]}, "name": {"type": "string"}, "params": {"type": "object"}}),
    "edit_mode_op": T("bmesh edit operations on a face/edge/vertex selection: extrude, inset, bevel, loop_cut, bridge, merge, dissolve, spin, shrink_fatten.",
                      {"object": OBJ, "op": {"type": "string"}, "select": {"type": "object"}, "params": {"type": "object"}}, ["op"]),
    "add_modifier": T("Add (and optionally apply) a modifier: MIRROR, ARRAY, SOLIDIFY, SUBSURF, BEVEL, BOOLEAN, SHRINKWRAP, REMESH.",
                      {"object": OBJ, "type": {"type": "string"}, "settings": {"type": "object"}, "apply": {"type": "boolean"}}, ["type"]),
    "curve_sweep": T("Bezier/NURBS curve with a bevel profile and taper (sweep), converted to mesh.",
                     {"name": {"type": "string"}, "points": PATH, "radius": {"type": "number"}, "taper": {"type": "array", "items": {"type": "number"}},
                      "nurbs": {"type": "boolean"}}, ["points"]),
    "save_blend": T("Save the current .blend file.", {"path": {"type": "string"}}, ["path"]),
}
