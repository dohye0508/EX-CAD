"""Detect what this Blender build/session can actually do (no guessing of APIs)."""
import importlib
import os


def detect():
    import bpy
    caps = {"blender_version": bpy.app.version_string, "background": bool(bpy.app.background),
            "python": __import__("sys").version.split()[0]}
    try:
        import gpu
        caps["gpu_backend"] = gpu.platform.backend_type_get()
        caps["gpu_renderer"] = gpu.platform.renderer_get()
    except Exception as e:
        caps["gpu_renderer"] = f"unavailable ({e})"
    mods = {}
    for m in ("numpy", "openvdb", "torch", "pytorch3d", "nvdiffrast", "scipy", "PIL", "cv2", "mediapipe"):
        try:
            mod = importlib.import_module(m)
            mods[m] = getattr(mod, "__version__", "yes")
        except Exception:
            mods[m] = None
    caps["modules"] = mods
    op = bpy.ops.sculpt.brush_stroke.get_rna_type()
    caps["brush_stroke_props"] = [p.identifier for p in op.properties if p.identifier != "rna_type"]
    caps["sculpt_brush_types"] = [e.identifier for e in bpy.types.Brush.bl_rna.properties["sculpt_brush_type"].enum_items]
    p = os.path.join(os.path.dirname(bpy.app.binary_path), f"{bpy.app.version[0]}.{bpy.app.version[1]}",
                     "datafiles", "assets", "brushes", "essentials_brushes-mesh_sculpt.blend")
    caps["essentials_sculpt_library"] = p if os.path.exists(p) else None
    if caps["essentials_sculpt_library"]:
        with bpy.data.libraries.load(p, assets_only=True) as (src, _dst):
            caps["essentials_sculpt_brushes"] = list(src.brushes)
    caps["native_sculpt_available"] = (not bpy.app.background) and bool(bpy.context.window_manager.windows)
    caps["differentiable_rendering"] = bool(mods.get("torch") and (mods.get("pytorch3d") or mods.get("nvdiffrast")))
    caps["sdf_backend"] = "openvdb" if mods.get("openvdb") else "numpy-bvh"
    return caps
