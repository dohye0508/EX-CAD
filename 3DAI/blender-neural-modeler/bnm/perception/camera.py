"""Camera intrinsics/extrinsics shared by Blender renders and the raycast renderer, so a pixel the
LLM points at in an image maps to exactly one 3-D ray on the mesh.

Pixel convention: (u, v) with origin at the image top-left, u to the right, v downward.
"""
import math
import numpy as np
from mathutils import Vector, Matrix

# camera direction (from target toward camera).  'left' = camera on the model's left (+X).
VIEWS = {
    "front": (0, -1, 0), "back": (0, 1, 0), "left": (1, 0, 0), "right": (-1, 0, 0),
    "three_quarter_left": (0.7071, -0.7071, 0), "three_quarter_right": (-0.7071, -0.7071, 0),
    "top": (0, 0, 1), "bottom": (0, 0, -1), "low_front": (0, -0.94, -0.34), "high_front": (0, -0.87, 0.5),
}


class CameraModel:
    def __init__(self, location, target, up=(0, 0, 1), ortho_scale=3.0, width=256, height=256,
                 perspective=False, lens_mm=50.0, sensor_mm=36.0):
        self.loc = np.asarray(location, float)
        self.target = np.asarray(target, float)
        self.W, self.H = int(width), int(height)
        self.persp = perspective
        self.ortho_scale = float(ortho_scale)
        self.lens, self.sensor = lens_mm, sensor_mm
        f = self.target - self.loc
        f /= np.linalg.norm(f)
        upv = np.asarray(up, float)
        if abs(f @ upv) > 0.99:
            upv = np.array([0, 1.0, 0]) if abs(f[1]) < 0.99 else np.array([1.0, 0, 0])
        r = np.cross(f, upv)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.right, self.up, self.fwd = r, u, f
        # extrinsics: world -> camera (x right, y up, z backward like Blender)
        self.R = np.stack([r, u, -f])          # rows
        self.t = -self.R @ self.loc

    @classmethod
    def for_view(cls, view, center, distance=6.0, ortho_scale=3.0, width=256, height=256, **kw):
        d = np.asarray(VIEWS[view] if isinstance(view, str) else view, float)
        d /= np.linalg.norm(d)
        up = (0, 1, 0) if abs(d[2]) > 0.95 else (0, 0, 1)
        if abs(d[2]) > 0.95:
            up = (0, 1, 0) if d[2] > 0 else (0, 1, 0)
        return cls(np.asarray(center, float) + d * distance, center, up, ortho_scale, width, height, **kw)

    # intrinsics -------------------------------------------------------------
    @property
    def pixel_size(self):  # world units per pixel (ortho), along the larger dimension like Blender
        return self.ortho_scale / max(self.W, self.H)

    @property
    def fx(self):
        return self.lens / self.sensor * max(self.W, self.H)

    def intrinsics(self):
        if self.persp:
            return {"type": "PERSP", "fx": self.fx, "fy": self.fx, "cx": self.W / 2, "cy": self.H / 2}
        return {"type": "ORTHO", "pixel_size": self.pixel_size, "cx": self.W / 2, "cy": self.H / 2}

    def extrinsics(self):
        return {"R_world_to_cam": self.R.round(6).tolist(), "t": self.t.round(6).tolist(),
                "location": self.loc.round(6).tolist()}

    # projection -------------------------------------------------------------
    def world_to_pixel(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        c = P @ self.R.T + self.t
        depth = -c[:, 2]
        if self.persp:
            u = self.W / 2 + self.fx * c[:, 0] / depth
            v = self.H / 2 - self.fx * c[:, 1] / depth
        else:
            u = self.W / 2 + c[:, 0] / self.pixel_size
            v = self.H / 2 - c[:, 1] / self.pixel_size
        return np.stack([u, v, depth], 1)

    def pixel_to_ray(self, u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        if self.persp:
            x = (u - self.W / 2) / self.fx
            y = -(v - self.H / 2) / self.fx
            d = x[..., None] * self.right + y[..., None] * self.up + self.fwd
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            o = np.broadcast_to(self.loc, d.shape).copy()
        else:
            x = (u - self.W / 2) * self.pixel_size
            y = -(v - self.H / 2) * self.pixel_size
            o = self.loc + x[..., None] * self.right + y[..., None] * self.up
            d = np.broadcast_to(self.fwd, o.shape).copy()
        return o, d

    def to_blender(self, scene, name="BNM_Camera"):
        import bpy
        cam = bpy.data.objects.get(name)
        if cam is None:
            cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
            scene.collection.objects.link(cam)
        cam.location = Vector(self.loc)
        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = Matrix(self.R.T.tolist()).to_quaternion()
        cd = cam.data
        cd.type = "PERSP" if self.persp else "ORTHO"
        cd.ortho_scale = self.ortho_scale
        cd.lens = self.lens
        cd.sensor_width = self.sensor
        cd.sensor_fit = "AUTO"
        cd.clip_start, cd.clip_end = 0.01, 100.0
        scene.camera = cam
        scene.render.resolution_x, scene.render.resolution_y = self.W, self.H
        scene.render.resolution_percentage = 100
        scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
        return cam

    def describe(self):
        return {"intrinsics": self.intrinsics(), "extrinsics": self.extrinsics(), "width": self.W, "height": self.H}
