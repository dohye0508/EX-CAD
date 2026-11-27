"""Multi-view observation.

* raycast_render: BVH ray casting -> depth, normal, face-id/object-id buffers, silhouette, clay,
  matcap, curvature and wireframe images, all in the same pixel frame as CameraModel.
  These buffers are also the surface-ID map used to turn image regions into mesh masks.
* workbench_render: real Blender (Workbench) renders for the human-readable stage record.
No materials, textures or flattering lights: shape errors must stay visible.
"""
import os
import numpy as np
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .camera import CameraModel, VIEWS
from ..geometry.mesh_io import MeshState


def scene_bvh(objects):
    """World-space BVH over several mesh objects. Returns (bvh, tri_owner, tri_local_vertices, states)."""
    verts, tris, owner, local = [], [], [], []
    states = []
    off = 0
    for oi, ob in enumerate(objects):
        st = MeshState(ob)
        mw = np.array(ob.matrix_world)
        co = st.co @ mw[:3, :3].T + mw[:3, 3]
        verts.append(co)
        tris.append(st.tris + off)
        owner.append(np.full(len(st.tris), oi))
        local.append(st.tris)
        states.append((st, co))
        off += len(co)
    V = np.vstack(verts)
    T = np.vstack(tris)
    bvh = BVHTree.FromPolygons(V.tolist(), T.tolist(), all_triangles=True)
    return bvh, np.concatenate(owner), np.vstack(local), states, V, T


def raycast_render(objects, cam: CameraModel, curvature_obj=None, bvh_pack=None):
    bvh, owner, local, states, V, T = bvh_pack or scene_bvh(objects)
    W, H = cam.W, cam.H
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    O, D = cam.pixel_to_ray(uu.ravel(), vv.ravel())
    depth = np.full(W * H, np.inf)
    nrm = np.zeros((W * H, 3))
    fid = -np.ones(W * H, dtype=np.int64)
    pos = np.full((W * H, 3), np.nan)
    rc = bvh.ray_cast
    for i in range(W * H):
        loc, n, f, dist = rc(Vector(O[i]), Vector(D[i]), 50.0)
        if loc is not None:
            depth[i] = dist
            nrm[i] = n
            fid[i] = f
            pos[i] = loc
    hit = fid >= 0
    # make normals face the camera (open meshes / back faces)
    flip = (nrm @ cam.fwd) > 0
    nrm[flip] *= -1
    buf = {"depth": depth.reshape(H, W), "normal": nrm.reshape(H, W, 3), "face": fid.reshape(H, W),
           "object": np.where(hit, owner[np.maximum(fid, 0)], -1).reshape(H, W),
           "position": pos.reshape(H, W, 3), "mask": hit.reshape(H, W), "camera": cam}
    if curvature_obj is not None:
        from ..geometry.curvature import curvature
        oi = objects.index(curvature_obj)
        st, _ = states[oi]
        Hc = curvature(st)["mean"]
        cmap = np.full(W * H, np.nan)
        sel = hit & (owner[np.maximum(fid, 0)] == oi)
        tri_v = local[fid[sel]]
        cmap[sel] = Hc[tri_v].mean(1)
        buf["curvature"] = cmap.reshape(H, W)
    return buf


# ---------------------------------------------------------------- shading
def shade(buf, mode="clay"):
    cam = buf["camera"]
    n = buf["normal"]
    m = buf["mask"]
    H, W = m.shape
    img = np.zeros((H, W, 3))
    bg = np.array([0.08, 0.08, 0.09])
    img[:] = bg
    if mode == "silhouette":
        img[m] = 1.0
        img[~m] = 0.0
        return img
    if mode == "depth":
        d = buf["depth"]
        if m.any():
            lo, hi = d[m].min(), d[m].max()
            g = 1 - (d - lo) / max(hi - lo, 1e-9)
            img[m] = (0.15 + 0.85 * g[m])[:, None]
        return img
    if mode == "normal":
        # camera-space normal encoded as RGB
        cn = n @ cam.R.T
        img[m] = 0.5 + 0.5 * cn[m]
        return img
    if mode == "curvature" and "curvature" in buf:
        c = buf["curvature"]
        ok = m & np.isfinite(c)
        if ok.any():
            s = np.percentile(np.abs(c[ok]), 95) or 1.0
            t = np.clip(c / s, -1, 1)
            base = 0.55
            img[ok] = np.stack([base + 0.45 * np.clip(t[ok], 0, 1), base - 0.35 * np.abs(t[ok]),
                                base + 0.45 * np.clip(-t[ok], 0, 1)], 1)
        return img
    # clay / matcap / wireframe base shading: key + fill + rim, all in camera space
    cn = n @ cam.R.T
    key = np.array([-0.45, 0.55, 0.7])
    key /= np.linalg.norm(key)
    fill = np.array([0.6, -0.1, 0.8])
    fill /= np.linalg.norm(fill)
    if mode == "matcap":
        # procedural clay matcap: lookup by camera-space normal xy
        x, y = cn[..., 0], cn[..., 1]
        lam = np.clip(0.5 + 0.5 * (x * -0.5 + y * 0.6 + cn[..., 2] * 0.6), 0, 1)
        spec = np.clip((cn @ key), 0, 1) ** 24
        val = 0.2 + 0.7 * lam + 0.25 * spec
        col = np.array([0.78, 0.66, 0.58])
    else:
        lam = np.clip(cn @ key, 0, 1)
        fl = np.clip(cn @ fill, 0, 1)
        rim = (1 - np.clip(cn[..., 2], 0, 1)) ** 3
        val = 0.12 + 0.72 * lam + 0.18 * fl + 0.15 * rim
        col = np.array([0.82, 0.80, 0.77])
    shaded = np.clip(val[..., None] * col, 0, 1)
    # eyes / secondary objects tinted so they are distinguishable
    obj = buf["object"]
    tint = (obj > 0)[..., None]
    shaded = np.where(tint, shaded * np.array([0.85, 0.9, 1.05]), shaded)
    img[m] = shaded[m]
    if mode == "wireframe":
        f = buf["face"]
        edge = np.zeros_like(m)
        edge[:, 1:] |= f[:, 1:] != f[:, :-1]
        edge[1:, :] |= f[1:, :] != f[:-1, :]
        img[edge & m] = img[edge & m] * 0.72
    return img


def save_png(img, path):
    H, W = img.shape[:2]
    rgba = np.ones((H, W, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(img, 0, 1)
    rgba = rgba[::-1]  # Blender images are bottom-up
    name = "_bnm_save"
    im = bpy.data.images.get(name)
    if im is not None and (im.size[0] != W or im.size[1] != H):
        bpy.data.images.remove(im)
        im = None
    if im is None:
        im = bpy.data.images.new(name, W, H, alpha=True)
    im.pixels.foreach_set(rgba.ravel())
    im.filepath_raw = path
    im.file_format = "PNG"
    im.save()
    return path


def load_image(path):
    """Load any Blender-readable image as float RGB (H,W,3) top-down."""
    im = bpy.data.images.load(path, check_existing=False)
    W, H = im.size
    a = np.empty(W * H * 4, dtype=np.float32)
    im.pixels.foreach_get(a)
    bpy.data.images.remove(im)
    return a.reshape(H, W, 4)[::-1, :, :3].astype(np.float64)


def contact_sheet(images, cols, path, labels=None):
    """Tile equally sized images into one sheet (no text rendering available, labels go in json)."""
    H, W = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    sheet = np.full((rows * H + (rows - 1) * 4, cols * W + (cols - 1) * 4, 3), 0.02)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        sheet[r * (H + 4):r * (H + 4) + H, c * (W + 4):c * (W + 4) + W] = im
    return save_png(sheet, path)


# ---------------------------------------------------------------- high level
def head_cameras(obj_or_center, size=256, ortho_scale=None, views=("front", "left", "three_quarter_left")):
    if hasattr(obj_or_center, "matrix_world"):
        ob = obj_or_center
        co = MeshState(ob).co
        mw = np.array(ob.matrix_world)
        cw = co @ mw[:3, :3].T + mw[:3, 3]
        center = (cw.min(0) + cw.max(0)) / 2
        ext = float((cw.max(0) - cw.min(0)).max())
    else:
        center = np.asarray(obj_or_center, float)
        ext = 2.5
    osc = ortho_scale or ext * 1.15
    return {v: CameraModel.for_view(v, center, distance=ext * 3 + 2, ortho_scale=osc, width=size, height=size) for v in views}


def render_views(objects, out_dir, prefix, views=("front", "left", "three_quarter_left"), modes=("clay",),
                 size=256, cameras=None, curvature_obj=None):
    os.makedirs(out_dir, exist_ok=True)
    cams = cameras or head_cameras(objects[0], size, views=views)
    pack = scene_bvh(objects)
    out = {}
    for v, cam in cams.items():
        buf = raycast_render(objects, cam, curvature_obj=curvature_obj if "curvature" in modes else None, bvh_pack=pack)
        for mode in modes:
            p = os.path.join(out_dir, f"{prefix}_{v}_{mode}.png")
            save_png(shade(buf, mode), p)
            out[f"{v}_{mode}"] = p
        out[f"_buf_{v}"] = buf
    return out


def workbench_render(objects, cam: CameraModel, path, mode="clay", cavity=True):
    """Real Blender render (Workbench).  mode: clay | matcap | silhouette | flat"""
    sc = bpy.context.scene
    cam.to_blender(sc)
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.color_type = "SINGLE"
    sh.single_color = (0.8, 0.78, 0.75)
    sh.show_cavity = cavity and mode != "silhouette"
    if sh.show_cavity:
        sh.cavity_type = "BOTH"
    if mode == "matcap":
        sh.light = "MATCAP"
        names = [s.name for s in bpy.context.preferences.studio_lights if s.type == "MATCAP"]
        pick = next((n for n in names if "clay" in n.lower()), names[0] if names else None)
        if pick:
            sh.studio_light = pick
    elif mode in ("silhouette", "flat"):
        sh.light = "FLAT"
        sh.single_color = (1, 1, 1)
    else:
        sh.light = "STUDIO"
    if sc.world is None:
        sc.world = bpy.data.worlds.new("BNM_World")
    sc.world.color = (0, 0, 0) if mode == "silhouette" else (0.05, 0.05, 0.055)
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    hidden = []
    for ob in sc.objects:
        if ob.type == "MESH" and ob not in objects and not ob.hide_render:
            ob.hide_render = True
            hidden.append(ob)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    for ob in hidden:
        ob.hide_render = False
    return path
