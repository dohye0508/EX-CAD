# Secondary and tertiary forms

## Secondary (features, 50k–250k verts)
- **Planes**: brow band (CLAY_STRIPS r 0.075, arched path), nasion dip, orbit carve with depth loop, zygoma
  (CLAY diagonal), buccal hollow (DRAW_SUBTRACT below zygoma), temple FLATTEN, forehead FLATTEN, muzzle barrel
  (3 CLAY_STRIPS arcs), chin mound, jaw line (CLAY_STRIPS + PINCH), under-jaw FLATTEN normal −Z.
- **Eyes**: place eyeball spheres from measured brow depth; conform lids to the ball shell (aperture pushed
  behind the ball, lids = ball radius + 1.2 mm); RELAX geodesic; CREASE lid fold 4 mm above margin; CLAY_STRIPS
  lower lid; tear-trough CREASE light; medial canthus DRAW_SUBTRACT.
- **Nose**: GRAB tip forward (loop on projection) + small SNAKE_HOOK; CLAY_STRIPS bridge (loop on projection);
  FLATTEN side planes; INFLATE tip + domes; INFLATE alae + GRAB outward; CREASE alar groove (geodesic);
  nostrils DRAW_SUBTRACT from below with columella protected; SURFACE_SMOOTH nose–cheek.
- **Mouth**: philtral columns (CLAY_STRIPS) + groove (CREASE); stomion CREASE with downturned corners;
  upper lip along cupid's bow (CLAY_STRIPS, geodesic) + PINCH vermilion border; lower lip fuller (2× CLAY_STRIPS +
  INFLATE); corners DRAW_SUBTRACT; mentolabial DRAW_SUBTRACT; chin projection loop; subtle nasolabial CREASE.
- **Ears**: SNAKE_HOOK a vertical plate (3 pulls), INFLATE mass, FLATTEN outer face, CLAY_STRIPS helix rim,
  DRAW_SUBTRACT concha, CLAY_STRIPS antihelix, INFLATE tragus and lobe.
- **Neck**: SCM CLAY_STRIPS mastoid→sternum, SURFACE_SMOOTH throat.
- **Transitions**: SURFACE_SMOOTH + RELAX across the face with curvature feature protection.

## Tertiary (only when everything above reads right)
Fine lip creases (CREASE r 0.006–0.007, strength 0.1), lower-lid fold, alar crease accent, 1–2 mm asymmetry
(e.g. one zygoma fuller), skin texture/pores only if the user asks for that style (needs ≥ 500k verts or
displacement maps).
