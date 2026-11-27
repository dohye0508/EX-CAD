"""Reference image analysis with numpy only (Blender has no PIL/OpenCV/torch here).

Extracts: foreground silhouette (border-colour background model + largest component),
per-row width profile, side-view facial profile, Sobel edges.  Facial landmarks come from the
LLM (vision) or the user as pixel coordinates; this module normalises them.  Learned detectors
(MediaPipe/DINOv2/depth estimators) are optional and only used when installed (they are not).
"""
import numpy as np

from ..perception.render import load_image


def foreground_mask(img, thr=0.12, min_frac=0.002):
    H, W, _ = img.shape
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    bg = np.median(border, 0)
    spread = np.median(np.abs(border - bg), 0).max()
    dist = np.linalg.norm(img - bg, axis=2)
    m = dist > max(thr, 4 * spread)
    m = largest_component(m)
    m = fill_holes(m)
    if m.sum() < min_frac * H * W:
        raise ValueError("no foreground found; supply a mask or a cleaner background")
    return m


def largest_component(m):
    H, W = m.shape
    lab = np.zeros((H, W), dtype=np.int32)
    best, best_n, cur = 0, 0, 0
    for y0, x0 in zip(*np.nonzero(m)):
        if lab[y0, x0]:
            continue
        cur += 1
        stack = [(y0, x0)]
        lab[y0, x0] = cur
        n = 0
        while stack:
            y, x = stack.pop()
            n += 1
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < H and 0 <= xx < W and m[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = cur
                    stack.append((yy, xx))
        if n > best_n:
            best, best_n = cur, n
    return lab == best


def fill_holes(m):
    """Background = everything connected to the border through non-mask pixels."""
    inv = ~m
    H, W = m.shape
    seen = np.zeros_like(m)
    stack = [(y, x) for y in range(H) for x in (0, W - 1) if inv[y, x]] + [(y, x) for x in range(W) for y in (0, H - 1) if inv[y, x]]
    for y, x in stack:
        seen[y, x] = True
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= yy < H and 0 <= xx < W and inv[yy, xx] and not seen[yy, xx]:
                seen[yy, xx] = True
                stack.append((yy, xx))
    return ~seen


def sobel_edges(img):
    g = img.mean(2)
    p = np.pad(g, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return np.hypot(gx, gy)


def width_profile(mask, rows=40):
    """Normalised silhouette: for each relative height (0 top .. 1 bottom) the half-width and centre,
    in units of silhouette height (scale-free, because a single photo has no absolute size)."""
    ys, xs = np.nonzero(mask)
    top, bot = ys.min(), ys.max()
    Hs = bot - top + 1
    cx = np.median(xs)
    out = []
    for t in np.linspace(0, 1, rows):
        y = int(round(top + t * (Hs - 1)))
        r = np.nonzero(mask[y])[0]
        if len(r) == 0:
            out.append((float(t), 0.0, 0.0))
            continue
        out.append((float(t), float((r.max() - r.min() + 1) / 2 / Hs), float(((r.max() + r.min()) / 2 - cx) / Hs)))
    return {"rows": out, "height_px": int(Hs), "bbox": [int(xs.min()), int(top), int(xs.max()), int(bot)]}


def side_profile(mask, facing="left", rows=60):
    """Facial profile from a side photo: for each relative height, how far forward the silhouette is
    (relative to the rearmost point) in units of silhouette height.  facing: direction the face points."""
    ys, xs = np.nonzero(mask)
    top, bot = ys.min(), ys.max()
    Hs = bot - top + 1
    out = []
    for t in np.linspace(0, 1, rows):
        y = int(round(top + t * (Hs - 1)))
        r = np.nonzero(mask[y])[0]
        if len(r) == 0:
            out.append((float(t), None))
            continue
        front = r.min() if facing == "left" else r.max()
        back = xs.max() if facing == "left" else xs.min()
        out.append((float(t), float(abs(back - front) / Hs)))
    return {"rows": out, "height_px": int(Hs)}


def analyze_image(path, view="front", facing="left", landmarks_px=None):
    img = load_image(path)
    mask = foreground_mask(img)
    res = {"path": path, "size": [img.shape[1], img.shape[0]], "view": view, "silhouette_area_px": int(mask.sum())}
    res["width_profile"] = width_profile(mask)
    if view in ("left", "right", "side"):
        res["side_profile"] = side_profile(mask, facing)
    if landmarks_px:
        x0, y0, x1, y1 = res["width_profile"]["bbox"]
        Hs = y1 - y0 + 1
        cx = (x0 + x1) / 2
        res["landmarks_norm"] = {k: [(u - cx) / Hs, (v - y0) / Hs] for k, (u, v) in landmarks_px.items()}
    res["edge_strength_mean"] = float(sobel_edges(img)[mask].mean())
    return res, mask
