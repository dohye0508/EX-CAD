# Primary forms (blocking)

Goal: correct silhouette and big masses at low density (1k–50k verts). No features yet.

1. **Cranium & face mass** – ELASTIC_DEFORM grabs at silhouette extremes per height level, closed loop on
   half-width / front / back targets (`stage1_skull`). Large radius (0.2–0.26) so neighbouring levels move
   together; gain ≈ 0.45; reject iterations that increase error.
2. **Front plane** – the face is flatter than a sphere: grab para-sagittal points (x = ±0.4) back to the
   target depths so the face has a front and side plane.
3. **Neck** – SDF smooth union of a capsule (r 0.4, blend 0.22) behind the jaw, flat bust cut, re-extract
   (keeps one closed manifold, no seam). Narrow band ≥ blend width.
4. **Clean** – tangential relax + light Laplacian to remove Kelvinlet ripples before remeshing.
5. **Check** – front & side silhouette vs targets; crown, occiput and chin must read correctly before Stage 2.

Common failures and fixes
- Ripples on the cranium → too-small grab radius or too many levels; widen radius, smooth more.
- Pointed chin → lower levels' half-width targets too small or front target too far forward.
- Seam ring on neck → SDF background mismatch (fill with the grid background value) or band too narrow.
