# Observed workflow: YouTube lAspS2BK2Vk ("[blender] Chainsaw Man/power 팬아트", 27:36, timelapse)

Accessed 2026-11-25 in a browser; frames sampled every 5–15 s between 0:26 and 8:20. Observations below
are what was visible in those frames (two viewports: left = front orthographic overview, right = working
view; key overlay shows the modifier keys). Only the *method* is reused, not the character design.

| Time | What happens | Brush / tool (from overlay & result) | Rule extracted |
|---|---|---|---|
| 0:00–1:15 | finished renders (showcase) | – | – |
| 1:19 | plain sphere, large brush circle on the lower front | Grab / Elastic, radius ≈ 1/3 sphere | start from a sphere; first stroke pulls the lower front down into a face mass |
| 1:22 | egg-shaped face mask, two very soft broad hollows at eye level | Draw with Ctrl (subtract), big & soft | eye sockets are broad soft hollows, not holes |
| 1:27–1:32 | a single soft ridge pulled between the sockets down to the nose; mouth area bulges | Draw/Clay, then Smooth (Shift) | nose = one soft ridge first; smooth after every stroke |
| 1:38 | mouth line, chin, upper lip shape visible, everything soft | Ctrl (subtract) for mouth line, Shift smooth | features stay volumes; lines come late and shallow |
| 1:45–1:52 | 3/4 view: side planes of the face, cheek–jaw transition | Draw/Flatten + Smooth | turn the view to 3/4 to set side planes |
| 2:00 | lids/brow masses, lips as two soft volumes, small nostrils | Ctrl ×6, Left | lips are upper/lower *volumes*; nostrils small subtractions |
| 2:10–2:35 | almond eye openings cut deep, rim (lids) shaped around them | Ctrl subtract, Shift smooth ×2 | cut the almond opening, then build the lid rim around it |
| 2:48 | profile view: nose tip, lips, chin profile | Left bracket (smaller brush) | check profile; reduce brush size for details |
| 3:02–3:15 | Object mode: a new sphere placed at the ear position, moved with G | separate primitive | ear starts as its own sphere, sculpted, then attached |
| 3:30 | neck = separate cylinder under the head | separate primitive | neck is its own cylinder, placed behind/below the jaw |
| 3:45–4:45 | body blocked with spheres/cylinders (torso, pelvis, limbs), then sculpted | primitives + Draw/Smooth | block with primitives before sculpting anatomy |
| 6:40–8:20 | posing (seated), props from a subdivided cube cage | Edit mode + Subdivision | hard-surface props are box-modelled, not sculpted |

## Translated into BNM rules (anatomy/video_workflow.py)
1. Primary mass first and **smooth constantly**: every stroke group is followed by region smoothing (many
   iterations), so no stroke leaves a line.
2. Face mass = egg in front of a round cranium; shaped toward a low-resolution target volume.
3. Features in the order: sockets → nose ridge → mouth barrel/chin/brow → side planes (3/4) → lips volumes,
   nostrils → almond openings + lid rims (+ eyeballs for a realistic head) → profile check.
4. Ears and neck are **separate primitives**, sculpted separately and merged (SDF smooth union).
5. Lines (creases) only at the very end and shallow.

## Later steps (sampled 9:20–26:00)
| Time | What happens | Rule extracted |
|---|---|---|
| 9:20–12:40 | anatomy refined directly in the seated, crossed-leg pose; seat prop (box-modelled, subdivided) finished | pose first, then sculpt anatomy in the pose |
| 13:00–16:00 | clothing (long-sleeve sweater) sculpted as a separate shell over the body; feet/toes detailed | clothes = separate objects sculpted over the body |
| 16:00–19:20 | horns added; hair built from many curve tubes (bevelled curves) laid over the head | hair = curve strands, not a sculpted cap |
| 21:00–22:40 | vertex/texture paint: skin, sweater colour; face and eyes painted | colour after form |
| 24:20 | shirt print and denim shorts texture | – |
| 26:00 | hair coloured, floor tiles, final scene | – |
Character-specific design elements (horns, the shirt print, exact hairstyle) are not reproduced by BNM; only the method.
