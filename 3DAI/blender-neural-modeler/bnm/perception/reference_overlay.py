"""Overlay a reference photo on the model's front render.

The photo is aligned to the render with a similarity transform (rotation, scale, translation) that maps
the photo's two eye centres onto the model's projected eye centres; the remaining landmarks (nose, mouth,
chin, face edges) are then compared in pixels and in head-height fractions.  Output: [photo | model | 50%
blend] image + numbers.  This catches 'wrong face' problems that silhouette IoU cannot.
"""
import numpy as np

from .render import save_png, load_image


def similarity_from_eyes(src_l, src_r, dst_l, dst_r):
    s0 = np.asarray(src_r, float) - np.asarray(src_l, float)
    d0 = np.asarray(dst_r, float) - np.asarray(dst_l, float)
    scale = np.linalg.norm(d0) / max(np.linalg.norm(s0), 1e-9)
    ang = np.arctan2(d0[1], d0[0]) - np.arctan2(s0[1], s0[0])
    c, s = np.cos(ang), np.sin(ang)
    R = scale * np.array([[c, -s], [s, c]])
    t = np.asarray(dst_l, float) - R @ np.asarray(src_l, float)
    return R, t


def warp_photo(photo, R, t, W, H):
    """Inverse-map every output pixel into the photo (nearest neighbour)."""
    Ri = np.linalg.inv(R)
    vv, uu = np.mgrid[0:H, 0:W]
    pts = np.stack([uu.ravel() + 0.5, vv.ravel() + 0.5], 1)
    src = (pts - t) @ Ri.T
    x = np.clip(src[:, 0].astype(int), 0, photo.shape[1] - 1)
    y = np.clip(src[:, 1].astype(int), 0, photo.shape[0] - 1)
    inside = (src[:, 0] >= 0) & (src[:, 0] < photo.shape[1]) & (src[:, 1] >= 0) & (src[:, 1] < photo.shape[0])
    out = photo[y, x].reshape(H, W, 3)
    out[~inside.reshape(H, W)] = 0.08
    return out


def overlay(photo_path, photo_lm, model_img, model_lm, out_path):
    """photo_lm / model_lm: dict name -> (u, v) pixels; must both contain eye_L_center and eye_R_center.
    Returns metrics: per-landmark pixel error after eye alignment, in fractions of model eye-to-chin height."""
    photo = load_image(photo_path)
    H, W = model_img.shape[:2]
    R, t = similarity_from_eyes(photo_lm["eye_R_center"], photo_lm["eye_L_center"],
                                model_lm["eye_R_center"], model_lm["eye_L_center"])
    aligned = warp_photo(photo, R, t, W, H)
    blend = 0.5 * aligned + 0.5 * model_img
    sheet = np.concatenate([aligned, np.full((H, 4, 3), 0.02), model_img, np.full((H, 4, 3), 0.02), blend], 1)
    save_png(sheet, out_path)
    eye_mid = (np.asarray(model_lm["eye_L_center"]) + np.asarray(model_lm["eye_R_center"])) / 2
    ref_len = None
    if "chin_bottom" in model_lm:
        ref_len = abs(model_lm["chin_bottom"][1] - eye_mid[1])
    errs = {}
    for k, p in photo_lm.items():
        if k in model_lm:
            q = R @ np.asarray(p, float) + t
            e = q - np.asarray(model_lm[k], float)
            errs[k] = {"du_px": round(float(e[0]), 1), "dv_px": round(float(e[1]), 1),
                       "frac": round(float(np.linalg.norm(e) / ref_len), 3) if ref_len else None}
    return {"image": out_path, "landmark_errors_photo_minus_model": errs,
            "note": "dv>0: the photo landmark is LOWER than the model's; du>0: further to the image right"}
