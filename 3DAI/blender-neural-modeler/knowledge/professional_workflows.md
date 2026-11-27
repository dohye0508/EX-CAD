# Professional workflow notes and their translation into BNM

Sources consulted (Nov 2026): Proko "How to Sculpt the Loomis Head" (block from simple forms, measure landmarks);
Gnomon "Sculpting the Planes of the Head" (primary forms = outline and planes before secondary detail);
anatomy4sculptors "Forming the Head: Simple to Complex"; Tripo blog "Facial detail: planes and micro forms"
(order: planes → features → folds/fat pads → pores). Exact proportions come from classical canons (see
anatomy_rules.md), not from those pages. Rule: blocking → planes → landmarks → features → transitions → texture,
never eyes/lips before the skull is right.

## Order and timing extracted
| Phase | First form | Brushes | Direction of strokes | When to check | When to add resolution |
|---|---|---|---|---|---|
| Block | cranium + jaw + neck | Move/Elastic, Snake Hook | along the silhouette, large radius | front/side silhouette every pass | after silhouette error < tol |
| Planes | front/side/under planes | Flatten, Clay, Scrape | across planes, follow form | 3/4 view (planes read best) | — |
| Landmarks | brow, orbits, zygoma, nose wedge, mouth barrel, chin | Clay Strips, Draw (sub) | strips follow muscle/bone direction | front + profile | before eyes (needs ~1 cm/10 verts) |
| Features | eyes/lids, nose parts, lips, ears | Inflate, Crease, Pinch, Clay Strips | along feature contours (lid arc, cupid's bow) | close-up views | locally if needed |
| Transitions | cheeks, nasolabial, jaw | Surface smooth, Relax | broad, low strength | curvature render | — |
| Texture | creases, pores | Crease, Draw Sharp, alphas | perpendicular to tension lines | close-up, raking light | highest |

## Tripo/TripoSG principles applied (not the model)
- **SDF representation**: TripoSG's VAE encodes shapes as SDF supervised with normals and an eikonal term.
  BNM uses SDF for volume edits (smooth union/difference along a stroke), neck union and self-intersection
  repair, extracting the zero level set with OpenVDB (or numpy Surface Nets).
- **Coarse-to-fine / multi-resolution**: icosphere (10k) → SDF remesh (≈35k) → voxel remesh (≈160k).
- **Implicit, measured targets**: losses (silhouette, landmarks, depth, normals, curvature, Laplacian,
  preservation) evaluated numerically; brush *parameters* are searched (coordinate descent) because Blender
  brushes are not differentiable. Rectified-flow sampling / learned priors are **not** reproduced: without
  trained weights there is no generative shape prior, so generality comes from rules + iterative observation.

## Video analysis
When the user supplies a sculpting video: `bnm/reference/video_analyzer.extract_frames` (ffmpeg) → inspect
frames → fill WORKFLOW_OBSERVATION_TEMPLATE (initial mesh, brushes, stroke direction, order, subdivision
moments, check views) → add general rules here. If a video cannot be opened, say so.
