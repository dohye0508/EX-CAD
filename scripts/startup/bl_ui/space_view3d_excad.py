# SPDX-FileCopyrightText: 2026 EX-CAD Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

import bpy
from bpy.types import Panel


class VIEW3D_PT_excad_sketch(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "EX-CAD"
    bl_label = "Parametric Sketch (Fusion Style)"

    def draw(self, context):
        layout = self.layout

        col = layout.column(align=True)
        col.label(text="1. Sketch Plane:")
        row = col.row(align=True)
        op = row.operator("excad.sketch_new", text="XY")
        op.plane = 'XY'
        op = row.operator("excad.sketch_new", text="XZ")
        op.plane = 'XZ'
        op = row.operator("excad.sketch_new", text="YZ")
        op.plane = 'YZ'

        col.separator()
        col.label(text="2. Sketch Entities:")
        col.operator("excad.sketch_add_rect", text="Rectangle", icon='MESH_PLANE')
        col.operator("excad.sketch_add_circle", text="Circle", icon='MESH_CIRCLE')
        col.operator("excad.sketch_add_line", text="Line", icon='IPO_LINEAR')

        col.separator()
        col.label(text="3. Constraints & Solid:")
        col.operator("excad.sketch_solve", text="Solve Constraints", icon='CONSTRAINT')
        row = col.row(align=True)
        row.operator("excad.sketch_extrude", text="Extrude", icon='MOD_SOLIDIFY')
        row.operator("excad.sketch_revolve", text="Revolve", icon='MOD_SCREW')


class VIEW3D_PT_excad_sdf(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "EX-CAD"
    bl_label = "High-Precision SDF Mesh (C++)"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.operator("excad.sdf_humanoid", text="Anatomical Humanoid", icon='OUTLINER_OB_ARMATURE')
        col.operator("excad.sdf_custom", text="Custom SDF Spec", icon='META_BALL')


class VIEW3D_PT_excad_bridge(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "EX-CAD"
    bl_label = "AI Direct Bridge (Port 9877)"

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True)
        row.operator("excad.bridge_start", text="Start Bridge", icon='PLAY')
        row.operator("excad.bridge_stop", text="Stop Bridge", icon='PAUSE')


classes = (
    VIEW3D_PT_excad_sketch,
    VIEW3D_PT_excad_sdf,
    VIEW3D_PT_excad_bridge,
)

if __name__ == "__main__":  # only for live edit.
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
