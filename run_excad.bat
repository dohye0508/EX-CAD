@echo off
call "%~dp0install_blender_addon.bat" || exit /b 1
"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --python-expr "import bpy; bpy.ops.preferences.addon_enable(module='excad_addon'); bpy.ops.wm.save_userpref(); bpy.ops.excad.bridge_start()"
