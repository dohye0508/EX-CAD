# SPDX-License-Identifier: GPL-2.0-or-later
"""Bake multi-view image detail into a Smart UV atlas, independently of vertex density."""

from pathlib import Path

import bpy
import numpy as np

from .inspection import _png
from .polyhedron_engine import _load_rgba_and_mask, get_polyhedron_faces


def read_depth(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        image.colorspace_settings.name = "Non-Color"
        w,h = image.size
        pixels = np.empty((h,w,4), np.float32)
        image.pixels.foreach_get(pixels.ravel())
        return pixels[:,:,0].copy()
    finally:
        bpy.data.images.remove(image)


def smart_uv(obj):
    layer = obj.data.uv_layers.get("EXCAD Atlas")
    if layer:
        obj.data.uv_layers.active = layer
        return layer
    if bpy.context.mode != "OBJECT":
        raise ValueError("Switch to Object Mode before unwrapping")
    active = bpy.context.view_layer.objects.active
    selected = list(bpy.context.selected_objects)
    layer = obj.data.uv_layers.new(name="EXCAD Atlas")
    obj.data.uv_layers.active = layer
    try:
        for other in selected:
            other.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(island_margin=0.012, scale_to_bounds=True)
    finally:
        if obj.mode == "EDIT":
            bpy.ops.object.mode_set(mode="OBJECT")
        obj.select_set(False)
        for other in selected:
            other.select_set(True)
        bpy.context.view_layer.objects.active = active
    return obj.data.uv_layers["EXCAD Atlas"]


def rasterize(uv, loops, size):
    ids = np.full((size,size), -1, np.int32)
    weights = np.zeros((size,size,2), np.float32)
    for i,(a,b,c) in enumerate(uv[loops]*size - 0.5):
        lo = np.maximum(0,np.ceil(np.minimum(np.minimum(a,b),c)).astype(int))
        hi = np.minimum(size-1,np.floor(np.maximum(np.maximum(a,b),c)).astype(int))
        x0,y0 = lo
        x1,y1 = hi
        if x1<x0 or y1<y0:
            continue
        u,v = b-a,c-a
        det = u[0]*v[1]-v[0]*u[1]
        if abs(det) < 1e-8:
            continue
        x = np.arange(x0,x1+1,dtype=np.float32)-a[0]
        y = np.arange(y0,y1+1,dtype=np.float32)[:,None]-a[1]
        wb,wc = (x*v[1]-y*v[0])/det, (u[0]*y-u[1]*x)/det
        inside = (wb>=-1e-5) & (wc>=-1e-5) & (wb+wc<=1.00001)
        region = np.s_[y0:y1+1,x0:x1+1]
        ids[region][inside] = i
        weights[region][inside] = np.stack((wb[inside],wc[inside]),axis=1)
    return ids,weights


def _pad(image, valid, steps=8):
    """Extend observed colors into unseen seam texels and island gutters without wrapping."""
    for _ in range(steps):
        old = valid.copy()
        for dst,src in ((np.s_[1:,:],np.s_[:-1,:]),(np.s_[:-1,:],np.s_[1:,:]),
                        (np.s_[:,1:],np.s_[:,:-1]),(np.s_[:,:-1],np.s_[:,1:])):
            fill = ~valid[dst] & old[src]
            image[dst][fill] = image[src][fill]
            valid[dst][fill] = True


def bake(obj, image_dir, poly_type="DODECAHEDRON_12", center=(0,0,0.9), size=2,
         texture_size=2048, path=None, threshold=0.15):
    if texture_size not in {256,512,1024,2048,4096} or size <= 0:
        raise ValueError("Invalid atlas resolution or projection size")
    directory = Path(bpy.path.abspath(image_dir))
    views = []
    for face in get_polyhedron_faces(poly_type):
        file = directory/face["filename"]
        if file.is_file():
            rgb,mask = _load_rgba_and_mask(str(file),threshold)
            depth_file = file.with_name(file.stem+"_depth.png")
            depth = read_depth(depth_file) if depth_file.is_file() else None
            if depth is not None and depth.shape != mask.shape:
                raise ValueError(f"Depth dimensions differ for {file.name}")
            views.append((face,rgb,mask,depth))
    if not views:
        raise ValueError("No multiview images available")
    mesh = obj.data
    layer = smart_uv(obj)
    mesh.calc_loop_triangles()
    uv = np.empty((len(mesh.loops),2),np.float32)
    layer.uv.foreach_get("vector",uv.ravel())
    loops = np.empty((len(mesh.loop_triangles),3),np.int32)
    mesh.loop_triangles.foreach_get("loops",loops.ravel())
    triangles = np.empty_like(loops)
    mesh.loop_triangles.foreach_get("vertices",triangles.ravel())
    coords = np.empty((len(mesh.vertices),3),np.float32)
    mesh.vertices.foreach_get("co",coords.ravel())
    matrix = np.asarray(obj.matrix_world,dtype=np.float32)
    points = coords @ matrix[:3,:3].T + matrix[:3,3] - np.asarray(center)
    normals = np.empty((len(mesh.corner_normals),3),np.float32)
    mesh.corner_normals.foreach_get("vector",normals.ravel())
    normals = normals @ np.linalg.inv(matrix[:3,:3])
    ids,bary = rasterize(uv,loops,texture_size)
    atlas = np.full((texture_size,texture_size,3),0.5,np.float32)
    valid = np.zeros_like(ids,dtype=bool)
    for row in range(0,texture_size,32):
        y,x = np.nonzero(ids[row:row+32]>=0)
        y += row
        fi = ids[y,x]
        wb,wc = bary[y,x].T
        weights = np.stack((1-wb-wc,wb,wc),axis=1)
        at = np.einsum("ij,ijk->ik",weights,points[triangles[fi]])
        n = np.einsum("ij,ijk->ik",weights,normals[loops[fi]])
        n /= np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-8)
        colors = np.zeros((len(x),3),np.float32)
        totals = np.zeros(len(x),np.float32)
        for face,rgb,mask,depth in views:
            h,w = mask.shape
            px = np.rint((at @ face["right"]/size + 0.5)*(w-1)).astype(int)
            py = np.rint((at @ face["up"]/size + 0.5)*(h-1)).astype(int)
            facing = np.maximum(0,n @ face["normal"])
            inside = (px>=0)&(py>=0)&(px<w)&(py<h)&(facing>0)
            idx = np.flatnonzero(inside)
            idx = idx[mask[py[idx],px[idx]]]
            if depth is not None:
                d = 0.5-at[idx] @ face["normal"]/size
                idx = idx[d <= depth[py[idx],px[idx]] + 2/min(w,h)]
            weight = facing[idx]**3
            totals[idx] += weight
            colors[idx] += rgb[py[idx],px[idx]]*weight[:,None]
        seen = totals>1e-8
        atlas[y[seen],x[seen]] = colors[seen]/totals[seen,None]
        valid[y[seen],x[seen]] = True
    if not valid.any():
        raise ValueError("No projected texels hit the UV atlas")
    _pad(atlas,valid)
    dest = Path(bpy.path.abspath(path)) if path else directory/(bpy.path.clean_name(obj.name)+"_albedo.png")
    dest.parent.mkdir(parents=True,exist_ok=True)
    srgb = np.where(atlas<=0.0031308,12.92*atlas,1.055*np.maximum(0,atlas)**(1/2.4)-0.055)
    _png(dest,np.rint(np.clip(srgb[::-1],0,1)*255).astype(np.uint8))
    image = bpy.data.images.load(str(dest.resolve()),check_existing=True)
    image.reload()
    mat = bpy.data.materials.new(obj.name+" Baked")
    mat.use_nodes = True
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = image
    uv_node = mat.node_tree.nodes.new("ShaderNodeUVMap")
    uv_node.uv_map = layer.name
    mat.node_tree.links.new(uv_node.outputs["UV"],tex.inputs["Vector"])
    mat.node_tree.links.new(tex.outputs["Color"],mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    mesh.materials.clear()
    mesh.materials.append(mat)
    return {"path":str(dest.resolve()),"resolution":texture_size,"views_used":len(views)}
