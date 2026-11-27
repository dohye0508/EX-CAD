"""Image-space <-> mesh-space linking (screen_to_surface, region masks from image areas)."""
import numpy as np
from mathutils import Vector

from .camera import CameraModel
from .render import scene_bvh


def screen_to_surface(objects, cam: CameraModel, u, v, bvh_pack=None):
    """Pixel -> first surface hit: world position, normal, object, face, local position."""
    bvh, owner, local, states, V, T = bvh_pack or scene_bvh(objects)
    o, d = cam.pixel_to_ray(np.array([u]), np.array([v]))
    loc, n, f, dist = bvh.ray_cast(Vector(o[0]), Vector(d[0]), 100.0)
    if loc is None:
        return None
    oi = int(owner[f])
    ob = objects[oi]
    lp = ob.matrix_world.inverted() @ loc
    tri = local[f]
    return {"object": ob.name, "face_triangle": int(f), "triangle_vertices": tri.tolist(),
            "world": list(loc), "local": list(lp), "normal": list(n), "distance": dist}


def visible_vertices(obj, cam: CameraModel, depth_buffer=None, tol=None, bvh_pack=None):
    """Vertices of obj that are visible from cam (projected + z-buffer test)."""
    from ..geometry.mesh_io import MeshState
    st = MeshState(obj)
    mw = np.array(obj.matrix_world)
    cw = st.co @ mw[:3, :3].T + mw[:3, 3]
    uvd = cam.world_to_pixel(cw)
    W, H = cam.W, cam.H
    inside = (uvd[:, 0] >= 0) & (uvd[:, 0] < W) & (uvd[:, 1] >= 0) & (uvd[:, 1] < H)
    vis = np.zeros(len(cw), dtype=bool)
    if depth_buffer is None:
        from .render import raycast_render
        depth_buffer = raycast_render([obj], cam, bvh_pack=bvh_pack)["depth"]
    tol = tol or 3 * st.mean_edge
    iu = np.clip(uvd[:, 0].astype(int), 0, W - 1)
    iv = np.clip(uvd[:, 1].astype(int), 0, H - 1)
    zb = depth_buffer[iv, iu]
    vis[inside] = np.abs(uvd[inside, 2] - zb[inside]) < tol
    return vis, uvd


def region_mask_from_pixels(obj, cam: CameraModel, polygon=None, circle=None, depth_buffer=None):
    """Editable vertex mask for an image region the LLM points at.

    polygon: [(u,v),...] in pixels;  circle: (u, v, radius_px).  Only visible vertices are selected,
    so pointing at the upper lid in a front image does not select the back of the head.
    """
    vis, uvd = visible_vertices(obj, cam, depth_buffer)
    u, v = uvd[:, 0], uvd[:, 1]
    if circle is not None:
        cu, cv, cr = circle
        sel = (u - cu) ** 2 + (v - cv) ** 2 <= cr * cr
    else:
        P = np.asarray(polygon, float)
        sel = np.zeros(len(u), dtype=bool)
        j = len(P) - 1
        for i in range(len(P)):  # even-odd rule
            xi, yi = P[i]
            xj, yj = P[j]
            cond = ((yi > v) != (yj > v)) & (u < (xj - xi) * (v - yi) / np.where(yj - yi == 0, 1e-12, yj - yi) + xi)
            sel ^= cond
            j = i
    idx = np.nonzero(sel & vis)[0]
    return idx


def project_landmarks(landmarks, cam: CameraModel, matrix_world=None):
    """Object/world-space landmarks -> pixel coords."""
    names = list(landmarks)
    P = np.array([landmarks[k] for k in names], float)
    if matrix_world is not None:
        mw = np.array(matrix_world)
        P = P @ mw[:3, :3].T + mw[:3, 3]
    uvd = cam.world_to_pixel(P)
    return {k: [round(float(a), 2), round(float(b), 2)] for k, (a, b, _) in zip(names, uvd)}
