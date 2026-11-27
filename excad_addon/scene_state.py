# SPDX-License-Identifier: GPL-2.0-or-later
"""Compact content fingerprints for evaluated scene geometry, transforms, and materials."""

import hashlib
import json

import bpy
import numpy as np


def _feed(h, value):
    h.update(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode())


def _material(mat):
    h = hashlib.blake2s(digest_size=8)
    _feed(h,list(mat.diffuse_color))
    if mat.use_nodes and mat.node_tree:
        for node in mat.node_tree.nodes:
            values = []
            for inp in node.inputs:
                if inp.is_linked:
                    values.append([(l.from_node.name,l.from_socket.identifier) for l in inp.links])
                elif hasattr(inp,"default_value"):
                    val = inp.default_value
                    if isinstance(val,bpy.types.ID):
                        val = val.name
                    elif not isinstance(val,(str,int,float,bool,type(None))):
                        val = list(val)
                    values.append(val)
            image = getattr(node,"image",None)
            _feed(h,[node.name,node.bl_idname,values,image.filepath if image else None])
    return h.hexdigest()


def rows(include_all=False):
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    result = []
    materials = {}
    for obj in sorted(bpy.context.scene.objects,key=lambda o:o.name):
        if not include_all and obj.type in {"CAMERA","LIGHT"}:
            continue
        ev = obj.evaluated_get(graph)
        h = hashlib.blake2s(digest_size=8)
        h.update(np.asarray(ev.matrix_world,dtype="<f4").tobytes())
        _feed(h,[obj.parent.name if obj.parent else "",obj.hide_viewport,obj.hide_render,
                 obj.get("excad_tree",""),list(obj.color)])
        verts = faces = 0
        if ev.type in {"MESH","CURVE","SURFACE","FONT","META"}:
            mesh = ev.to_mesh()
            try:
                if mesh:
                    verts,faces = len(mesh.vertices),len(mesh.polygons)
                    for items,prop,dtype,n in ((mesh.vertices,"co",np.float32,3),(mesh.loops,"vertex_index",np.int32,1),
                                               (mesh.polygons,"material_index",np.int32,1)):
                        data = np.empty(len(items)*n,dtype=dtype)
                        items.foreach_get(prop,data)
                        h.update(data.tobytes())
                    for mat in mesh.materials:
                        if mat:
                            if mat.name not in materials:
                                materials[mat.name] = _material(mat)
                            _feed(h,[mat.name,materials[mat.name]])
            finally:
                ev.to_mesh_clear()
        elif ev.type=="ARMATURE":
            for b in ev.pose.bones:
                h.update(np.asarray(b.matrix,dtype="<f4").tobytes())
        elif ev.type=="LIGHT":
            _feed(h,[ev.data.type,ev.data.energy,list(ev.data.color)])
        elif ev.type=="CAMERA":
            _feed(h,[ev.data.type,ev.data.lens,ev.data.ortho_scale])
        result.append([obj.name,obj.type[0],verts,faces,[round(v,3) for v in ev.matrix_world.translation],
                       [round(v,3) for v in ev.dimensions],h.hexdigest()])
    scene = bpy.context.scene
    sketches = {key:hashlib.blake2s(raw.encode(),digest_size=6).hexdigest()
                for key,raw in scene.get("excad_sketches",{}).items()}
    state = {"active":scene.get("excad_active_sketch","Sketch"),"profiles":sketches}
    return result,state
