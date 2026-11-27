# blender-neural-modeler (BNM)

Claude Code skill + Blender engine + MCP server for **scratch sculpting**: a primitive (sphere/cube) is
turned into a model with real brush strokes, multi-view observation and closed feedback loops. No finished
meshes are imported in the default mode; every stroke, primitive and remesh is logged for audit.

Target environment verified: Windows 11, Blender 5.2.1 LTS (bundled Python 3.13, numpy 2.3, openvdb),
Intel Arc iGPU (no CUDA), Claude Code (skill in `~/.claude/skills`, MCP via `claude mcp add`).

## Architecture
```
Claude (LLM) ── decides forms, regions, intents, judges renders
   │  MCP (stdio, stdlib only)            bnm_mcp/server.py  (+ headless_batch, launch_blender)
   ▼
Blender add-on bnm_live_bridge ── TCP 127.0.0.1:9890, command queue, main-thread timer, live overlay,
   │                               user-edit detection, cancel, chunked jobs
   ▼
bnm engine (inside Blender)
  sculpt/     virtual_brush (25 brushes, Kelvinlet elastic, HC smooth), native_brush (Blender brush assets +
              sculpt.brush_stroke), stroke (line/bezier/catmull-rom/arc/ellipse/parallel/radial/fill, arc-length
              spacing, virtual pressure), falloff, geodesic (Dijkstra ball, connected regions), planner
              (NL → plan JSON → verified execution/rollback), multiresolution (primitives, voxel/SDF remesh,
              subdivision, multires, adaptive subdivide)
  geometry/   mesh_io, analyzer (bbox/area/volume/symmetry/sections/relief/features), curvature (cotan H,
              angle-deficit K, principal, normal variation, feature protection), sdf (VDB level set, smooth CSG,
              capsule-path brushes, Surface Nets), semantic (measured landmarks + regions, schema-based),
              hard_surface (bmesh edit ops, modifiers, curve sweep, gear/bracket recipes)
  perception/ camera (intrinsics/extrinsics, world↔pixel, pixel→ray), render (BVH raycast: depth, normal,
              face-id, silhouette, clay, matcap, curvature, wireframe; Workbench renders), picker
              (screen→surface, image-area → vertex mask)
  anatomy/    face_canon, eye_sculptor (lids conformed to eyeball shell), head_sculpt_pipeline (stages 0–9
              with closed loops), creature_sculptor (schema-driven organic heads)
  reference/  image_analyzer (silhouette/profile/edges, numpy), structural_specification (observed/inferred +
              confidence → canon overrides), video_analyzer (ffmpeg frames)
  optimization/ geometry_loss, stroke_search (brush-parameter coordinate/grid search), iterative_optimizer
              (silhouette refinement vs reference)
  core/       orchestrator (Sculptor + OBSERVE→…→REPLAN loop with accept/rollback), state_manager (action log,
              audit), checkpoint_manager, capability_detector, task_router (modes, organic vs hard-surface)
  validation/ geometry_validator (non-manifold, degenerate, duplicates, winding, inside-out, self-intersection)
```
The spec's file list was used as a guide, not a checklist: modules are merged where that kept them real
(e.g. stroke interpolation + pressure live in `sculpt/stroke.py`). No placeholder files.

## Install
```
python scripts/install.py          # skill → ~/.claude/skills, add-on → Blender user add-ons (enabled), MCP → claude mcp add --scope user
python scripts/doctor.py           # check
```
Then open Blender normally (the add-on autostarts the bridge; panel: View3D ▸ Sidebar ▸ BNM) and ask Claude to sculpt.

## Status (honest)
Legend: **V** = implemented and executed/verified in Blender 5.2.1 during development · **I** = implemented, not executed
yet · **–** = not implemented (reason).

| Area | Item | Status | Evidence / note |
|---|---|---|---|
| Env | OS/Blender/GPU/Python/existing MCP survey | V | Blender 5.2.1, Arc iGPU, existing `mcp-for-blender` left untouched |
| Native | brush asset activation + `sculpt.brush_stroke` changes mesh | V | Draw, Clay Strips, Grab, Inflate, Smooth on a UV sphere (moved 221–586 verts) in UI mode |
| Native | `NativeBrushEngine` (view orbit to surface, px radius, pressure, verify by vertex diff) | I | wraps the verified calls; full class not yet run |
| Native | background mode | – | `brush_stroke.poll()` is False without a window → virtual engine used |
| Virtual | 25 brushes incl. Kelvinlet elastic, snake hook, HC smooth, layer, clay strips | V | smoke test: correct sign of normal displacement for every brush |
| Virtual | pose, conform (eyelid shell), mask attribute write | V/V/I | conform used in Benchmark A run; mask write not run |
| Strokes | curves, resampling, pressure profiles, symmetry, geodesic, front-face | V | used throughout Benchmark A run |
| SDF | VDB level set, smooth union/diff, VDB + Surface Nets extraction, SDF remesh | V | sphere round trip, manifold, 0 problems |
| Analysis | analyzer, curvature (sphere H=1.0), validator, cross-section | V/V/V/I | |
| Perception | raycast renders 7 modes, Workbench render (IoU 0.991 vs raycast), pixel↔surface round trip exact | V | |
| Semantic | face map, measured landmarks, regions, status planned→measured | V | used by the pipeline |
| Face | Sphere→Face stages 0–9, closed loops (silhouette, orbit, nose tip, bridge, chin) | V | 1 run, 170 s, 162k verts; loops converged (e.g. nose tip error 0.182→0.001) |
| Face | quality fixes after visual review (skull ripples, neck seam, eye aperture, ear mass, self-intersection repair) | I | code changed after the run; not re-run by request |
| Planner | NL command → plan JSON → protected execution, leakage/direction check, rollback | I | |
| Reference | image silhouette/profiles, spec with observed/inferred, canon overrides, silhouette refine loop | I | no reference portrait provided yet |
| Reference | learned landmark/depth/DINOv2 models | – | not installed (no torch/mediapipe); LLM vision supplies landmarks |
| Optimization | loss terms, coordinate/grid search over brush params | I | |
| Optimization | differentiable rendering (PyTorch3D/nvdiffrast) | – | not available; numeric search used instead |
| Bridge | TCP server, timer queue, generator jobs, cancel, overlay, user-edit detection, add-on panel | I | |
| MCP | stdio server, 66 engine tools + 5 server tools, image content, headless_batch, launch_blender | I | |
| Hard surface | bmesh ops, modifiers, curve sweep, gear hub / bracket recipes | I | |
| Creature | schema-driven quadruped head | I | |
| Tests/benchmarks | unit tests, Benchmarks B–J | – | skipped at the user's request; only Benchmark A was run once |

## Benchmark A (first run, before fixes)
Stage renders showed a recognisable head (brow, orbits with eyeballs inside lids, nose with nostrils,
lips, chin, ears, neck). Visual defects found: Kelvinlet ripples on the cranium, a seam ring on the
neck, small sunken eye openings, thin ear plates, 657 self-intersecting face pairs (mesh validator
failed). Fixes are in the code but have not been re-run.

Run: `blender -b --factory-startup --python scripts/bench_sphere_to_face.py -- --out benchmarks/A_sphere_to_face`
(outputs stage renders, progression sheet, 8-view final, render-mode sheet, action log, report.json with
loops, brush counts per stage, timings, validation and the no-external-mesh audit).
