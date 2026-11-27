# SPDX-License-Identifier: GPL-2.0-or-later
"""Semantic placement using evaluated world-space geometry and explicit contact gaps."""

import itertools
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import scene_mesh


def _items(obj):
    result = scene_mesh.surfaces(obj.name)
    if not result:
        raise ValueError(f"{obj.name} has no visible surface")
    return result


def bounds(obj):
    items = _items(obj)
    return np.stack((np.min([s.bounds[0] for s in items],0),np.max([s.bounds[1] for s in items],0)))


def _translate(obj, delta):
    matrix = obj.matrix_world.copy()
    matrix.translation += Vector(delta)
    obj.matrix_world = matrix
    bpy.context.view_layer.update()


def place_relative(obj, target=None, relation="ON_TOP_OF", gap=None):
    obj = bpy.context.scene.objects[obj]
    relation = relation.upper()
    gap = float(gap if gap is not None else 0 if relation == "ALIGN_FLOOR" else 0.05)
    a = bounds(obj)
    delta = np.zeros(3)
    if relation == "ALIGN_FLOOR":
        delta[2] = gap-a[0,2]
    else:
        other = bpy.context.scene.objects[target]
        if obj == other or other in obj.children_recursive:
            raise ValueError("The support must be independent of the moving object")
        b = bounds(other)
        delta = b.mean(0)-a.mean(0)
        if relation == "ON_TOP_OF":
            delta[2] = b[1,2]+gap-a[0,2]
        elif relation == "INSIDE":
            if np.any(np.ptp(a,axis=0)+2*gap > np.ptp(b,axis=0)+1e-6):
                raise ValueError("Object does not fit inside target bounds with the requested clearance")
        else:
            axis,sign = {"LEFT_OF":(0,-1),"RIGHT_OF":(0,1),"IN_FRONT_OF":(1,-1),"BEHIND":(1,1)}[relation]
            low,high = (1,0) if sign>0 else (0,1)
            delta[axis] = b[low,axis]+sign*gap-a[high,axis]
            delta[2] = b[0,2]-a[0,2]
    _translate(obj,delta)
    return {"object":obj.name,"bbox":(a+delta).round(5).tolist()}


def look_at(obj,target):
    source = bpy.context.scene.objects[obj]
    dest = bpy.context.scene.objects[target]
    position,_,scale = source.matrix_world.decompose()
    center = Vector(bounds(dest).mean(0)) if dest.type in {"MESH","CURVE"} else dest.matrix_world.translation
    direction = center-position
    if direction.length < 1e-8:
        raise ValueError("Look-at target coincides with the object origin")
    track = "-Z" if source.type in {"CAMERA","LIGHT"} else "-Y"
    rotation = direction.to_track_quat(track,"Y" if track=="-Z" else "Z")
    source.matrix_world = Matrix.LocRotScale(position,rotation,scale)
    bpy.context.view_layer.update()
    return {"object":source.name}


def _duplicate(asset):
    source = [asset,*asset.children_recursive]
    copies = {}
    try:
        for obj in source:
            copy = obj.copy()
            bpy.context.collection.objects.link(copy)
            copies[obj] = copy
        for obj,copy in copies.items():
            if obj == asset:
                copy.parent = None
                copy.matrix_world = asset.matrix_world
            else:
                copy.parent = copies[obj.parent]
            for mod in copy.modifiers:
                if mod.type == "ARMATURE" and mod.object in copies:
                    mod.object = copies[mod.object]
        return copies[asset]
    except Exception:
        for copy in copies.values():
            bpy.data.objects.remove(copy,do_unlink=True)
        raise


def scatter_on_surface(asset,surface_obj,count,min_dist=0.1,align_normal=True,seed=0):
    scene = bpy.context.scene
    source, surface = scene.objects[asset],scene.objects[surface_obj]
    if count<1 or count>10000 or min_dist<0:
        raise ValueError("Choose 1..10000 instances and nonnegative spacing")
    items = _items(surface)
    triangles = np.concatenate([s.points[s.triangles] for s in items])
    cross = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    areas = np.linalg.norm(cross,axis=1)
    if areas.sum()<=0:
        raise ValueError("Surface is degenerate")
    probability = areas/areas.sum(dtype=np.float64)
    rng = np.random.default_rng(seed)
    cell = max(min_dist,1e-6)
    lookup,placements = {},[]
    for _ in range(24):
        sample_count = max(32,(count-len(placements))*8)
        choices = rng.choice(len(triangles),sample_count,p=probability)
        for fi,(r,s) in zip(choices,rng.random((sample_count,2))):
            a,b,c = triangles[fi]
            sqrt_r = math.sqrt(r)
            point = a*(1-sqrt_r)+sqrt_r*((1-s)*b+s*c)
            key = tuple(np.floor(point/cell).astype(int))
            nearby = (q for delta in itertools.product((-1,0,1), repeat=3)
                      for q in lookup.get(tuple(i+j for i,j in zip(key,delta)),()))
            if any(np.dot(point-q,point-q)<min_dist**2 for q in nearby):
                continue
            lookup.setdefault(key,[]).append(point)
            placements.append((point, cross[fi]/areas[fi]))
            if len(placements)==count:
                break
        if len(placements)==count:
            break
    if len(placements)<count:
        raise ValueError("Surface cannot accommodate the requested count and minimum distance")
    cloud = np.concatenate([s.points for s in _items(source)])
    origin,rotation,scale = source.matrix_world.decompose()
    up = rotation @ Vector((0,0,1))
    copies = []
    try:
        for point,normal in placements:
            n = Vector(normal)
            rotation_delta = up.rotation_difference(n) if align_normal else Matrix.Identity(3).to_quaternion()
            verts = (cloud-np.array(origin)) @ np.array(rotation_delta.to_matrix()).T
            # Project the center of the footprint onto the sampled surface.
            center = (verts.min(0)+verts.max(0))/2
            anchor = center + normal*(float(np.min(verts @ normal))-float(center @ normal))
            copy = _duplicate(source)
            copies.append(copy)
            copy.matrix_world = Matrix.LocRotScale(Vector(point-anchor),rotation_delta @ rotation,scale)
        bpy.context.view_layer.update()
    except Exception:
        for root in copies:
            for obj in [*root.children_recursive,root]:
                bpy.data.objects.remove(obj,do_unlink=True)
        raise
    return {"objects":[o.name for o in copies],"count":len(copies)}


def _meshes(names):
    items = scene_mesh.surfaces()
    if names is not None:
        wanted = set(names)
        items = [s for s in items if s.obj.name in wanted]
        if {s.obj.name for s in items} != wanted:
            raise ValueError("Unknown or non-mesh collision object")
    return items


def check_intersections(objects=None):
    return {"intersections":scene_mesh.intersections(_meshes(objects))}


def resolve_overlaps(objects=None,fixed=(),gap=0.01,max_steps=64):
    if gap<0 or max_steps<1:
        raise ValueError("Use a nonnegative gap and positive step limit")
    items = _meshes(objects)
    locked = set(fixed)
    original = {s.obj:s.obj.matrix_world.copy() for s in items}
    ordered = sorted(items,key=lambda s:(s.obj.name not in locked,s.obj.name))
    moved = set()
    steps = 0
    try:
        for i,item in enumerate(ordered):
            previous = ordered[:i]
            while True:
                touching = [s for s in previous if scene_mesh.intersections([s,item])]
                if not touching:
                    break
                if item.obj.name in locked or steps>=max_steps:
                    raise ValueError("Unable to separate all collision objects within the step limit")
                # Each candidate clears every currently intersecting obstacle on that axis.
                candidates = []
                for axis in (0,1):
                    positive = max(s.bounds[1,axis]-item.bounds[0,axis] for s in touching)+gap
                    negative = min(s.bounds[0,axis]-item.bounds[1,axis] for s in touching)-gap
                    for distance in (positive,negative):
                        delta = np.zeros(3)
                        delta[axis] = distance
                        trial = item.bounds+delta
                        hits = sum(np.all(np.minimum(s.bounds[1],trial[1])-np.maximum(s.bounds[0],trial[0])>1e-5)
                                   for s in previous)
                        candidates.append((hits,abs(distance),axis,delta))
                _,_,_,delta = min(candidates,key=lambda c:c[:3])
                _translate(item.obj,delta)
                item = scene_mesh.read_surface(item.obj,bpy.context.evaluated_depsgraph_get())
                ordered[i] = item
                moved.add(item.obj.name)
                steps += 1
        remaining = check_intersections([s.obj.name for s in items])["intersections"]
        if remaining:
            raise ValueError("A parent transform introduced another collision")
    except Exception:
        for obj,matrix in original.items():
            obj.matrix_world=matrix
        bpy.context.view_layer.update()
        raise
    return {"moved":sorted(moved),"steps":steps}
