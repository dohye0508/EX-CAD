"""Checkpoints and undo.

* Stroke-level undo: in-memory coordinate snapshots (topology unchanged) -> instant restore.
* Structural checkpoints: full mesh datablock copies (survive remesh), optional .npz on disk.
"""
import os
import numpy as np
import bpy

from ..geometry.mesh_io import get_coords, set_coords


class CheckpointManager:
    def __init__(self, obj, directory=None, max_coord_snapshots=64, log=None):
        self.obj = obj
        self.dir = directory
        self.snaps = []  # (label, topology_key, coords)
        self.meshes = {}  # label -> mesh datablock copy
        self.max = max_coord_snapshots
        self.log = log

    def _key(self):
        me = self.obj.data
        return (len(me.vertices), len(me.edges), len(me.polygons))

    # ---------- light snapshots
    def snapshot(self, label="snap"):
        self.snaps.append((label, self._key(), get_coords(self.obj)))
        if len(self.snaps) > self.max:
            self.snaps.pop(0)
        return len(self.snaps) - 1

    def undo(self, steps=1):
        """Restore the coordinate snapshot `steps` back (only if topology unchanged)."""
        if len(self.snaps) < steps:
            return False
        label, key, co = self.snaps[-steps]
        if key != self._key():
            return False
        set_coords(self.obj, co)
        del self.snaps[-steps:]
        if self.log:
            self.log({"op": "undo", "label": label})
        return True

    def discard_last(self):
        if self.snaps:
            self.snaps.pop()

    # ---------- full checkpoints
    def create(self, label):
        old = self.meshes.pop(label, None)
        if old is not None and old.users == 0:
            bpy.data.meshes.remove(old)
        base = f"BNM_CKPT_{label}"
        for m in [m for m in bpy.data.meshes if (m.name == base or m.name.startswith(base + ".")) and m.users <= 1]:
            m.use_fake_user = False
            if m.users == 0:
                bpy.data.meshes.remove(m)          # older checkpoint with the same label (e.g. before an engine reload)
        cp = self.obj.data.copy()
        cp.name = base
        cp.use_fake_user = True
        self.meshes[label] = cp
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)
            np.savez_compressed(os.path.join(self.dir, f"{label}.npz"), co=get_coords(self.obj))
        if self.log:
            self.log({"op": "create_checkpoint", "label": label, "vertices": len(cp.vertices)})
        return label

    def restore(self, label):
        cp = self.meshes.get(label) or bpy.data.meshes.get(f"BNM_CKPT_{label}")   # survives engine reloads
        if cp is None:
            raise KeyError(f"no checkpoint {label!r}; have {list(self.meshes)}")
        cur = self.obj.data
        new = cp.copy()
        new.name = cur.name
        self.obj.data = new
        if cur.users == 0:
            bpy.data.meshes.remove(cur)
        self.snaps.clear()
        if self.log:
            self.log({"op": "restore_checkpoint", "label": label})
        return True

    def labels(self):
        return list(self.meshes)

    def cleanup(self):
        for m in list(self.meshes.values()):
            m.use_fake_user = False
            if m.users == 0:
                bpy.data.meshes.remove(m)
        self.meshes.clear()
