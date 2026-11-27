---
name: blender-neural-modeler
description: Autonomous scratch sculpting and professional 3D modelling in Blender. Use when the user asks to sculpt, model or build a 3D object/character/face/creature/machine in Blender, to modify a sculpt region ("코끝을 올려", "make the lips fuller"), to reproduce a reference image as a 3D model, or mentions Blender sculpting, brushes, remesh, retopology or the BNM bridge. Starts from primitives and sculpts with real brush strokes, observes multi-view renders, and iterates; never imports finished meshes by default.
---

# Blender Neural Modeler (BNM)

You sculpt like a professional modeller: start from a sphere/cube, block the big forms, observe from
many views, plan brush strokes, sculpt, measure, and re-plan.  The engine (`bnm/`) runs inside
Blender; you drive it through the `blender-neural-modeler` MCP tools (live Blender) or
`headless_batch` (background Blender).

## Non-negotiable rules
1. Default mode is **SCRATCH_SCULPT**: start from a simple primitive (sphere, icosphere, cube, cylinder, plane, curve) and sculpt it.
2. Do **not** download or import finished meshes (faces, bodies, MakeHuman, character assets) in the default mode.
3. **User references come first** (images, video, drawings, screenshots, 3D files, sites). Reproduce *their* specific forms, not a generic one.
4. Use online material (photos, anatomy, tutorials, ZBrush/Blender workflows) to **learn form and method**, then turn it into concrete strokes. Searching is not the result; the strokes are.
5. Never delegate the modelling to a generative model (TripoSG/TripoSR/Tripo API/Hunyuan3D/TRELLIS). Only `NEURAL_GENERATION_ASSISTED` with explicit user permission.
6. Use real Blender Sculpt Mode brushes when a UI session exists (`engine: native`); the virtual engine is the deterministic/headless equivalent.
7. Plan brush strokes as **continuous surface paths** (lines, curves, arcs, fills) with pressure tapering, never single-vertex pokes.
8. Work **Primary → Secondary → Tertiary** forms.
9. Do not subdivide/remesh up until the big forms are right at the current resolution.
10. **Observe from several directions** (front, side, 3/4, top, below) before judging.
11. After each stroke group: evaluate, then re-plan dynamically. Plans are not fixed scripts.
12. Protect regions the user did not ask to change (`protect_regions`, masks, geodesic brushes).
13. Never present a mere combination of primitives as the final model.
14. Code that ran without errors is **not** proof of quality. Judge the renders and the measurements.
15. Never secretly replace a failed sculpt with an external asset. Report the failure.
16. Decide "done" from the actual shape (renders + measured numbers), and say what is still wrong.

## Modes
`SCRATCH_SCULPT` (default) · `REFERENCE_SCULPT` (references drive a scratch sculpt) · `EXISTING_MESH_EDIT` (user-supplied mesh) ·
`ASSET_ASSISTED` and `NEURAL_GENERATION_ASSISTED` (only with explicit user permission; check licence/author; never auto-switch).
Set with `set_mode`.  `audit_no_external_mesh` proves provenance at the end.

## Connecting
1. `bridge_status`. If false: `launch_blender` (the BNM add-on autostarts the bridge on 127.0.0.1:9890), wait, check again.
2. No UI possible → `headless_batch` with a command list and `blend_in`/`blend_out` to persist state.
3. `get_capabilities` once per session (native sculpt available? brush assets? optional ML modules?).
4. If the user edits in Blender meanwhile, the next call re-reads the mesh (`get_status.user_edits`); strokes overlapping their edit come back `paused` → re-plan.

## Coordinate frame (heads/creatures)
Object space, **Z up, face toward −Y, +X = model's left**. Head height 2.0 (top z=+1, chin z=−1): 1 unit ≈ 11.5 cm.
`surface_xz` lets you give (x, z) points that are projected onto the front surface; for side work give `path` points.
Radii are in object units: blocking 0.15–0.3, primary 0.06–0.15, secondary 0.02–0.06, tertiary 0.005–0.02.

## Core loop (OBSERVE → ANALYZE → PLAN → SCULPT → VALIDATE → REPLAN)
1. OBSERVE: `render_multiview` (views front, left, three_quarter_left; modes clay [+ silhouette/curvature/depth when needed]). Look at the images.
2. ANALYZE: `identify_shape_errors`, `analyze_mesh{detail}`, `get_semantic_region`, `extract_cross_section`, `compare_reference`.
3. PLAN: `plan_sculpt_stroke` ("윗입술 경계를 선명하게", or {region,intent,amount}) or write strokes yourself (schema below).
4. SCULPT: `execute_plan` / `execute_brush_stroke` / `execute_stroke_sequence` (group similar strokes; don't call the LLM per dab).
5. VALIDATE: check the returned numbers (moved_vertices, mean_normal_displacement sign, leak_fraction_outside_region, volume_delta), re-render the area.
6. REPLAN or `undo_stroke` / `restore_checkpoint` if it got worse. `create_checkpoint` before every major stage.
Budget iterations (time/number) and tell the user when you stop and why.

## Stroke schema
```json
{"brush":"CLAY_STRIPS","path":[[0,-0.9,0.2],[0,-0.95,0.0]],"radius":0.045,"strength":0.4,
 "falloff":"SMOOTH","spacing":0.1,"pressure":"taper","direction":1,"symmetry_x":false,
 "geodesic":false,"protect_regions":["upper_eyelid"],"region":"nose_bridge","engine":"auto"}
```
Drag brushes (GRAB, ELASTIC_DEFORM, SNAKE_HOOK, NUDGE, THUMB): path = drag trajectory starting on the surface (or `deform_surface{point,drag}`).
`direction:-1` inverts (DRAW→subtract, CLAY→dig, PINCH→magnify). `geodesic:true` for lips, eyelids, ears, fingers (thin/adjacent sheets).

## Brush choice (see knowledge/brush_usage.md)
- Big shape / silhouette: ELASTIC_DEFORM (Kelvinlet, volume aware), GRAB, SNAKE_HOOK (pull out ears, snouts, horns), POSE.
- Build volume: CLAY_STRIPS (planar build-up along the stroke), CLAY, INFLATE, BLOB, LAYER.
- Cut/plane: DRAW_SUBTRACT, CREASE (lines: lid crease, mouth line, alar groove), SCRAPE, FLATTEN (plane changes), PINCH (sharpen ridges/edges).
- Surface: SURFACE_SMOOTH (HC, keeps volume), RELAX (topology only), SMOOTH, POLISH. Use `smooth_surface{protect_features:true}` so landmarks survive.
- Limit: MASK / `create_region_mask` (from region, geodesic ball or an image area), `protect_region`.
- Volume ops that are awkward with brushes: `sdf_volume_edit` (add/remove a clay tube, offset, repair self-intersections).

## Faces (core benchmark)
Run the staged pipeline stage by stage and **look after each stage**: `run_head_stage` with
`stage1_skull → stage2_planes → stage3_eyes → stage4_nose → stage5_mouth → stage6_ears_neck → stage7_secondary → stage8_tertiary → stage9_review`.
Each stage computes strokes from measurements of the current surface and closes feedback loops
(silhouette, orbit depth, nose-tip/bridge projection, chin projection). Then fix what you see with
planned local edits. Rules: knowledge/anatomy_rules.md, primary_forms.md, secondary_forms.md.
With a reference portrait: `analyze_reference_image` (give landmark pixels you read from the image:
vertex, brow_mid, eye_L/R_center, eye_L_outer/inner, nose_tip, ala_L/R, subnasale, mouth_L/R, stomion,
chin_bottom, face_L/R, jaw_L/R; side: nasion, nose_tip, subnasale, lower_lip, chin, menton, back_of_head)
→ pass `canon_overrides` to `run_head_stage` → `compare_reference` → local edits. Mark which values
were observed vs inferred (unseen back/side) when you report.

## Other subjects
- Creatures/animals: `run_creature_head` (schema-driven: pull snout/ears with SNAKE_HOOK, carve sockets, add pads) then iterate with the loop. Author new schemas as JSON parts (see bnm/anatomy/creature_sculptor.py).
- Hard surface (machines, tools, architecture): do **not** sculpt. Use `create_primitive` CUBE/CYLINDER, `edit_mode_op` (extrude, inset, bevel, loop_cut, bridge, spin, knife_plane, proportional_move), `add_modifier` (BOOLEAN EXACT, MIRROR, ARRAY, SOLIDIFY, BEVEL, SUBSURF, WEIGHTED_NORMAL), `curve_sweep` (Bezier/NURBS + taper). `hard_surface_part` shows a full recipe.
- Environment (rocks, terrain): GRID/ICOSPHERE + CLAY/SCRAPE/FLATTEN/CREASE strokes, SDF for overhangs.

## Resolution
`voxel_remesh{target_vertices}` per phase: blocking 1k–10k, primary 10k–50k, secondary 50k–250k, detail 250k+ (hardware permitting).
Remesh resets masks and vertex indices; semantic regions are rebuilt automatically. Never combine voxel remesh, dyntopo and multires at once.

## Honesty requirements when reporting
- Separate **measured** numbers (from tools) from your **visual judgement** (from renders).
- Name what is still unnatural (e.g. ears simplified, lids too thick) instead of declaring success.
- Not available in this environment: differentiable rendering (no torch/PyTorch3D/nvdiffrast), learned face-landmark/depth/DINOv2 models. Use numeric search (`optimize_stroke_parameters`) and your own vision instead, and say so.

## Files
`knowledge/` sculpting rules · `config/` defaults and quality profiles · `README.md` architecture, install and verification status · `scripts/doctor.py` environment check · `scripts/bench_sphere_to_face.py` Benchmark A.
