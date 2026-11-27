# Brush usage (virtual engine ⇄ Blender 5.2 essentials asset)

Verified in Blender 5.2.1: brushes are assets activated with
`bpy.ops.brush.asset_activate(asset_library_type='ESSENTIALS', relative_asset_identifier='brushes/essentials_brushes-mesh_sculpt.blend/Brush/<Name>')`
and strokes run through `bpy.ops.sculpt.brush_stroke(stroke=[{location, mouse, mouse_event, pressure, size, ...}])`
(requires a window + VIEW_3D region; poll() is False in `-b` background mode).

| BNM brush | Blender asset | Displacement model (virtual) | Typical use | Radius (head=2) | Strength |
|---|---|---|---|---|---|
| DRAW | Draw | area normal × r·0.5 per stroke | general build, quick volume | 0.03–0.12 | 0.2–0.5 |
| DRAW_SHARP | Draw Sharp | sharp falloff, original normal | fine lines | 0.005–0.02 | 0.2–0.4 |
| DRAW_SUBTRACT | Draw (INVERT) | −area normal | sockets, nostrils, concha, sulci | 0.02–0.13 | 0.3–0.9 |
| CLAY | Clay | fill toward offset plane | masses: zygoma, chin, cheeks | 0.05–0.12 | 0.3–0.6 |
| CLAY_STRIPS | Clay Strips | square footprint along stroke | brow band, bridge, lips, muscles | 0.015–0.08 | 0.3–0.6 |
| CLAY_THUMB | Clay Thumb | tilted plane, piles ahead | pushing clay along | 0.03–0.08 | 0.3 |
| INFLATE | Inflate/Deflate | vertex normals | nose tip, alae, lobes, lips | 0.03–0.1 | 0.3–0.7 |
| BLOB | Blob | inflate + pinch | knuckles, nodules | 0.02–0.06 | 0.4 |
| LAYER | Layer | capped height over original | plates, scales | 0.02–0.08 | 0.4 |
| CREASE | Crease Sharp | cut + pinch toward stroke line | lid crease, mouth line, alar groove, wrinkles | 0.006–0.035 | 0.1–0.8 |
| PINCH | Pinch/Magnify | toward centre in tangent plane | sharpen ridges: lip border, jaw line | 0.02–0.05 | 0.25–0.5 |
| FLATTEN | Flatten/Contrast | project to area plane | plane changes, temples, forehead | 0.04–0.2 | 0.25–0.5 |
| SCRAPE / FILL | Scrape/Fill, Fill/Deepen | only above / below plane | remove bumps / fill pits | 0.04–0.2 | 0.3 |
| SMOOTH | Smooth | Laplacian | quick cleanup | any | 0.2–0.5 |
| SURFACE_SMOOTH | Smooth (deform SURFACE) | HC Laplacian (volume kept) | transitions without shrinkage | any | 0.2–0.4 |
| RELAX | Relax Slide | tangential Laplacian | even topology, keep shape | any | 0.3 |
| GRAB | Grab | captured weights × drag | move a feature (tip, chin) | 0.05–0.15 | 1.0 |
| ELASTIC_DEFORM | Elastic Grab | regularised Kelvinlet | blocking, proportions | 0.15–0.3 | 1.0 |
| SNAKE_HOOK | Snake Hook | re-grab every dab | ears, snouts, horns, limbs | 0.06–0.4 | 0.9 |
| NUDGE / THUMB | Nudge / Thumb | tangential smear | shift surface sideways | 0.03–0.1 | 0.5 |
| POSE | Pose | Rodrigues rotation of weighted region | bend ear/jaw/limb | region | angle |
| MASK | Mask | mask attribute `.sculpt_mask` | protect | any | — |

Pressure profile `taper` ramps in/out (smoothstep, min 0.12) so starts/ends never pop.
Spacing default ≈ 0.1 r (drag brushes 0.04 r); displacement is normalised by spacing so totals do not
depend on dab count.  Combination recipe (nose bridge): GRAB forward → CLAY_STRIPS down the bridge
→ FLATTEN side planes (normal ±(0.75,−0.66,0)) → PINCH ridge if needed → SURFACE_SMOOTH into the
cheeks → re-render front + side.
