# Sculpting principles → executable rules

| Principle | What professionals do | Executable rule in BNM |
|---|---|---|
| Primary forms first | Block big masses (cranium, jaw, neck) before any feature | Stage 1 closed-loop silhouette fit (ELASTIC_DEFORM r 0.2–0.26) must converge (max error ≤ 0.015) before Stage 2 |
| Silhouette | Check the outline from front/side/top before details | `render_multiview` modes `silhouette`; `compare_reference` gives per-row width error → symmetric grabs at those heights |
| Plane changes | Faces read as planes (front, side, under-jaw, forehead) | FLATTEN with explicit `normal` along the plane; SCRAPE to cut plane breaks |
| Volume distribution | Masses overlap: zygoma over cheek, brow over orbit | CLAY/CLAY_STRIPS build on the upper mass, DRAW_SUBTRACT the hollow below |
| Negative space | Sockets, nasolabial, mentolabial define the positives | Carve hollows with measured depth targets (orbit depth loop) |
| Gesture / rhythm | Forms flow (jaw line to chin, brow to nose) | Catmull-Rom curves through landmarks, not straight segments; taper pressure |
| Big to small brush | Radius shrinks with each phase | blocking 0.15–0.3 → primary 0.06–0.15 → secondary 0.02–0.06 → tertiary 0.005–0.02 (head height 2) |
| Resolution when needed | Subdivide only when current density cannot hold the form | remesh_primary 0.02 after blocking; 0.0105 before eyes; repair remesh only when self-intersections |
| Constant checking | Rotate the model all the time | render after every stage: front, left, three_quarter_left; full 8-view at the end |
| Symmetry then asymmetry | Work symmetric, break symmetry last | `symmetry_x` on until Stage 8; small asymmetry strokes at the end |
| Smooth without erasing | Smooth transitions but keep landmarks | SURFACE_SMOOTH (HC) + `feature_protection` from curvature (85–88th percentile) |
| Thin structures | Lips, lids, ears are separate sheets | `geodesic: true` so the brush follows the surface and cannot jump the gap |

## Judging unnatural form (checklist used after each render)
1. Silhouette: crown round, back of skull projects behind ears, jaw angle below ear, chin not pointed.
2. Profile: forehead → nasion dip → nose bridge → tip → subnasale → upper lip ahead of lower lip → mentolabial dip → chin.
3. Eye line ≈ half head height; eyes one eye-width apart; eyeballs sit *inside* the lids, lids have thickness.
4. Nose base width ≈ inner-eye distance; nostrils are holes with rims, not paint.
5. Mouth corners on pupil lines (slightly inside); upper lip has cupid's bow; lower lip fuller.
6. Ears between brow and nose-base lines, behind the jaw midline, tilted like the nose bridge.
7. Surface: no ripples, no sharp pits outside intended creases (curvature render).
