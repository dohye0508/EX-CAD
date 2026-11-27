# Head anatomy rules (adult, neutral) — targets used by `anatomy/face_canon.py`

Frame: head height vertex→menton = 2.0 units (≈ 23 cm, 1 unit ≈ 11.5 cm). Z up, face −Y, +X = model's left.
These are canon averages for a generic adult; a reference portrait overrides them (structural_specification.py).

## Vertical (fraction of head height from the top)
| Landmark | fraction | z | note |
|---|---|---|---|
| hairline | ~0.14 | 0.72 | facial thirds start |
| brow ridge / glabella | ~0.39 | 0.22 | |
| nasion | ~0.44 | 0.12 | deepest point between brow and nose |
| eye line | 0.50 | 0.00 | Loomis: eyes at half the head |
| nose tip | ~0.64 | −0.27 | |
| subnasale (nose base) | ~0.69 | −0.37 | hairline→brow ≈ brow→nose base ≈ nose base→chin |
| stomion (mouth line) | ~0.78 | −0.565 | ~1/3 of nose-base→chin |
| mentolabial sulcus | ~0.86 | −0.72 | |
| chin / menton | 0.93 / 1.0 | −0.86 / −1.0 | |

## Horizontal
- Head width ≈ 0.68–0.7 × height at the temples/zygomatic arches ("five eyes" wide).
- Eye width = 1/5 head width (≈ 0.27); inner canthi one eye-width apart; eye centres x = ±0.27.
- Nose base (alae) width ≈ intercanthal distance (alae half-width ≈ 0.115).
- Mouth corners ≈ pupil lines, slightly inside (±0.205).
- Jaw angle half-width ≈ 0.53; chin half-width ≈ 0.13.

## Depth (relative projections, measured on the surface)
- Eyeball diameter ≈ 24 mm (radius 0.104); cornea ≈ 8 mm behind the brow ridge; lids are a 1–1.5 mm-thick shell over the ball (0.012), aperture almond 30×10 mm, lateral canthus slightly higher (tilt +0.012).
- Orbit (skin at eye centre before eyeball) ≈ 14 mm behind brow ridge.
- Nose tip projects ≈ 23 mm in front of the cheek plane at the alar base; bridge ≈ 11 mm in front of inner canthi.
- Upper lip ahead of lower lip; chin ≈ 6 mm in front of the mentolabial fold.
- Ears: from brow line to nose-base line, just behind the head's mid-depth (y ≈ +0.1), ~0.56 tall, protrude ~1.5–2 cm, tilt parallel to the nose bridge.
- Neck: cylinder r ≈ 0.4, set behind the jaw (centre y ≈ +0.2), sternocleidomastoids from mastoid (behind ear) to the sternum.

## Structure order (what forms what)
skull & jaw → brow/orbits/zygoma/maxilla planes → eyeballs then lids → nose (bridge, tip, alae, nostrils) →
mouth barrel then lips (philtrum, cupid's bow, corners, mentolabial) → ears/neck → transitions → texture.

## Region semantics
Regions (forehead, glabella, brow_ridge, orbit, upper/lower_eyelid, nose_bridge, nose_tip, nasal_ala, nostril,
cheekbone, cheek, philtrum, upper/lower_lip, mouth_corner, chin, jaw, temple, ear, neck) are ellipsoids around
landmarks that are *ray-cast on the current mesh* every time; each has adjacency, allowed edit directions and
features to protect (e.g. upper_lip protects lower_lip and philtrum).
