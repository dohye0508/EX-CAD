# SPDX-License-Identifier: GPL-2.0-or-later
"""Named scene checkpoints and bounded request-level undo, without switching .blend files."""

from collections import deque
from pathlib import Path
import tempfile
import uuid

import bpy


class SceneHistory:
    def __init__(self):
        self.storage = tempfile.TemporaryDirectory(prefix="excad-history-")
        self.named = {}
        self.history = deque()
        self.owner = None

    def scope(self):
        uid = bpy.context.scene.session_uid
        if self.owner != uid:
            for path in set(self.named.values()) | set(self.history):
                path.unlink(missing_ok=True)
            self.named.clear()
            self.history.clear()
            self.owner = uid

    def capture(self):
        self.scope()
        path = Path(self.storage.name)/(uuid.uuid4().hex+".blend")
        scene = bpy.context.scene
        bpy.data.libraries.write(str(path),{scene},path_remap="ABSOLUTE",compress=True)
        return path

    def checkpoint(self,name):
        if not isinstance(name,str) or not name.strip():
            raise ValueError("Checkpoint name must be nonempty")
        snapshot = self.capture()
        old = self.named.get(name)
        self.named[name] = snapshot
        if old:
            old.unlink(missing_ok=True)
        return {"name":name}

    def remember(self):
        self.history.append(self.capture())
        if len(self.history)>12:
            self.history.popleft().unlink(missing_ok=True)

    def undo(self):
        self.scope()
        if not self.history:
            raise ValueError("No bridge edit to undo")
        snapshot = self.history[-1]
        self.restore(snapshot)
        self.history.pop().unlink(missing_ok=True)
        return {}

    def rollback(self,name):
        self.scope()
        if name not in self.named:
            raise ValueError(f"Unknown checkpoint {name}")
        self.remember()
        self.restore(self.named[name])
        return {"name":name}

    def restore(self,path):
        if bpy.context.mode!="OBJECT":
            raise ValueError("Switch to Object Mode before restoring a scene")
        current = bpy.context.scene
        other_scenes = [s for s in bpy.data.scenes if s!=current]
        shared = {o for scene in other_scenes for o in scene.objects}
        shared.update(c for scene in other_scenes for c in scene.collection.children_recursive)
        old_objects = set(current.objects)-shared
        old_collections = set(current.collection.children_recursive)-shared
        users = bpy.data.user_map()
        owners = {current,*old_objects,*old_collections}
        # Only remove data previously reachable from this scene which loses its last user.
        dependencies = {}
        for dep,refs in users.items():
            for ref in refs:
                dependencies.setdefault(ref,set()).add(dep)
        queue = list(owners)
        while queue:
            additional = dependencies.get(queue.pop(),set())-owners
            owners.update(additional)
            queue.extend(additional)
        # References from other scenes (including modifier and collection instances)
        # must survive a restore even when their objects are not linked there directly.
        queue = list(other_scenes)
        protected = set(queue)
        while queue:
            additional = dependencies.get(queue.pop(),set())-protected
            protected.update(additional)
            queue.extend(additional)
        old_objects -= protected
        old_collections -= protected
        disposable = owners-{current}-old_objects-old_collections-protected-shared
        requested = {}
        with bpy.data.libraries.load(str(path),link=False) as (source,dest):
            if len(source.scenes)!=1:
                raise ValueError("Invalid scene checkpoint")
            for key in ("scenes","objects","collections","meshes","curves","cameras","lights","materials",
                        "armatures","actions","images","node_groups","textures","worlds","metaballs","fonts",
                        "volumes","pointclouds","hair_curves","lattices"):
                names = list(getattr(source,key,()))
                if names:
                    requested[key] = names
                    setattr(dest,key,names.copy())
        restored = dest.scenes[0]
        for win in bpy.context.window_manager.windows:
            if win.scene==current:
                win.scene=restored
        bpy.data.scenes.remove(current)
        bpy.data.batch_remove(old_objects | old_collections)
        # Remove zero-user dependencies in order without purging unrelated orphan assets.
        remaining = list(disposable)
        while True:
            unused = {i for i in remaining if not i.library and i.users==0 and not i.use_fake_user}
            if not unused:
                break
            remaining = [i for i in remaining if i not in unused]
            bpy.data.batch_remove(unused)
        for key,names in requested.items():
            for name,block in zip(names,getattr(dest,key)):
                if block and not block.library:
                    block.name=name
        self.owner = restored.session_uid
        bpy.context.view_layer.update()
