"""Reference-driven iterative refinement (REFERENCE_SCULPT):
render the model from the reference's view -> compare silhouettes -> per-row width error ->
symmetric Grab strokes at the matching heights -> re-render -> accept/rollback.
"""
import numpy as np

from ..perception.render import raycast_render, scene_bvh
from ..perception.camera import CameraModel
from .geometry_loss import silhouette_loss, row_width_loss


def fit_mask_to_camera(ref_mask, cam: CameraModel, model_mask):
    """Align the reference silhouette to the model render: scale/translate by bbox height & centre."""
    ys, xs = np.nonzero(ref_mask)
    ym, xm = np.nonzero(model_mask)
    s = (ym.max() - ym.min() + 1) / (ys.max() - ys.min() + 1)
    H, W = model_mask.shape
    out = np.zeros((H, W), dtype=bool)
    cy_r, cx_r = ys.min(), (xs.min() + xs.max()) / 2
    cy_m, cx_m = ym.min(), (xm.min() + xm.max()) / 2
    vv, uu = np.nonzero(np.ones((H, W), dtype=bool))
    sy = np.round((vv - cy_m) / s + cy_r).astype(int)
    sx = np.round((uu - cx_m) / s + cx_r).astype(int)
    ok = (sy >= 0) & (sy < ref_mask.shape[0]) & (sx >= 0) & (sx < ref_mask.shape[1])
    out[vv[ok], uu[ok]] = ref_mask[sy[ok], sx[ok]]
    return out


def silhouette_refine(S, ref_mask, cam: CameraModel, iterations=6, rows=12, radius=0.14, gain=0.6, z_range=None):
    """Front-view silhouette fitting by symmetric lateral grabs.  Returns loss history."""
    hist = []

    def render_mask():
        objs = [S.obj]
        return raycast_render(objs, cam, bvh_pack=scene_bvh(objs))["mask"]

    m = render_mask()
    ref = fit_mask_to_camera(ref_mask, cam, m)
    loss = silhouette_loss(m, ref)
    hist.append({"iter": 0, "silhouette_loss": round(loss, 4)})
    px = cam.pixel_size
    for it in range(1, iterations + 1):
        dw = row_width_loss(m, ref)
        ys = np.nonzero(m.any(1) | ref.any(1))[0]
        S.ckpt.snapshot(f"sil_{it}")
        for y in np.linspace(ys.min() + 2, ys.max() - 2, rows).astype(int):
            err = float(np.mean(dw[max(0, y - 2):y + 3])) * px
            if abs(err) < px * 1.5:
                continue
            o, d = cam.pixel_to_ray(np.array([cam.W / 2]), np.array([y + 0.5]))
            z = float(o[0][2])
            if z_range and not (z_range[0] <= z <= z_range[1]):
                continue
            co = S.st.co
            sel = np.abs(co[:, 2] - z) < 0.03
            if not sel.any():
                continue
            P = co[sel]
            p = P[np.argmax(P[:, 0])]
            S.stroke("ELASTIC_DEFORM", [p, p - [err * gain, 0, 0]], radius, 1.0, symmetry_x=True, region="silhouette")
        m2 = render_mask()
        l2 = silhouette_loss(m2, ref)
        acc = l2 < loss
        if acc:
            S.ckpt.discard_last()
            m, loss = m2, l2
        else:
            S.ckpt.undo(1)
            S.eng.resync()
            gain *= 0.5
        hist.append({"iter": it, "silhouette_loss": round(l2, 4), "accepted": acc})
        S.log.record({"op": "silhouette_refine", "iter": it, "loss": l2, "accepted": acc})
        if loss < 0.01 or gain < 0.1:
            break
    return hist
