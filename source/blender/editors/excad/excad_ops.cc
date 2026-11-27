/* SPDX-FileCopyrightText: 2026 EX-CAD Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "MEM_guardedalloc.h"

#include "DNA_mesh_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"

#include "BKE_context.hh"
#include "BKE_lib_id.hh"
#include "BKE_mesh.h"
#include "BKE_report.hh"

#include "DEG_depsgraph.hh"

#include "ED_excad.hh"
#include "ED_object.hh"
#include "ED_screen.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"

#include "WM_api.hh"
#include "WM_types.hh"

#include "bmesh.hh"

#include "excad_ai_bridge.hh"
#include "excad_intern.hh"
#include "excad_sdf_mesh.hh"
#include "excad_sketch_solver.hh"

#include <string>
#include <vector>

namespace blender::ed::excad {

Object *create_mesh_object_from_raw(bContext *C, const char *name, const RawMeshData &raw)
{
  Main *bmain = CTX_data_main(C);

  BMeshCreateParams bm_create_params{};
  BMesh *bm = BM_mesh_create(&bm_mesh_allocsize_default, &bm_create_params);

  std::vector<BMVert *> bm_verts;
  bm_verts.reserve(raw.verts.size());
  for (const auto &v : raw.verts) {
    bm_verts.push_back(BM_vert_create(bm, v.data(), nullptr, BM_CREATE_NOP));
  }

  std::vector<BMVert *> face_verts;
  for (const auto &f : raw.faces) {
    if (f.size() < 3) {
      continue;
    }
    face_verts.clear();
    face_verts.reserve(f.size());
    bool valid = true;
    for (int vid : f) {
      if (vid < 0 || vid >= static_cast<int>(bm_verts.size())) {
        valid = false;
        break;
      }
      face_verts.push_back(bm_verts[vid]);
    }
    if (valid) {
      BM_face_create_verts(
          bm, face_verts.data(), static_cast<int>(face_verts.size()), nullptr, BM_CREATE_NOP, true);
    }
  }

  BMO_op_callf(bm, 0, "recalc_face_normals faces=%af", bm);

  Mesh *me = BKE_mesh_add(bmain, name);
  BMeshToMeshParams to_mesh_params{};
  to_mesh_params.calc_object_remap = false;
  BM_mesh_bm_to_me(bmain, bm, me, &to_mesh_params);
  BM_mesh_free(bm);

  const float loc[3] = {0.0f, 0.0f, 0.0f};
  const float rot[3] = {0.0f, 0.0f, 0.0f};
  Object *ob = ed::object::add_type_with_obdata(
      C, OB_MESH, name, loc, rot, false, 0, &me->id);
  id_us_min(&me->id);

  DEG_id_tag_update(&ob->id, ID_RECALC_TRANSFORM | ID_RECALC_GEOMETRY);
  WM_event_add_notifier(C, NC_OBJECT | ND_DRAW, ob);
  return ob;
}

/* -------------------------------------------------------------------- */
/* Sketch Operators                                                     */
/* -------------------------------------------------------------------- */

static wmOperatorStatus excad_sketch_new_exec(bContext *C, wmOperator *op)
{
  const int plane_int = RNA_enum_get(op->ptr, "plane");
  SketchDocument::active().reset(static_cast<SketchPlane>(plane_int));
  BKE_reportf(op->reports, RPT_INFO, "EX-CAD: Started new parametric sketch");
  WM_event_add_notifier(C, NC_WINDOW, nullptr);
  return OPERATOR_FINISHED;
}

static void EXCAD_OT_sketch_new(wmOperatorType *ot)
{
  static const EnumPropertyItem plane_items[] = {
      {static_cast<int>(SketchPlane::XY), "XY", 0, "XY Plane", "Top XY sketch plane"},
      {static_cast<int>(SketchPlane::XZ), "XZ", 0, "XZ Plane", "Front XZ sketch plane"},
      {static_cast<int>(SketchPlane::YZ), "YZ", 0, "YZ Plane", "Side YZ sketch plane"},
      {0, nullptr, 0, nullptr, nullptr},
  };
  ot->name = "New Parametric Sketch";
  ot->idname = "EXCAD_OT_sketch_new";
  ot->description = "Reset active 2D parametric sketch on the selected plane";
  ot->exec = excad_sketch_new_exec;
  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  RNA_def_enum(ot->srna, "plane", plane_items, static_cast<int>(SketchPlane::XY), "Plane", "");
}

static wmOperatorStatus excad_sketch_add_rect_exec(bContext * /*C*/, wmOperator *op)
{
  const double u0 = RNA_float_get(op->ptr, "u0");
  const double v0 = RNA_float_get(op->ptr, "v0");
  const double width = RNA_float_get(op->ptr, "width");
  const double height = RNA_float_get(op->ptr, "height");
  SketchDocument::active().add_rect(u0, v0, width, height);
  BKE_reportf(op->reports, RPT_INFO, "EX-CAD: Added constrained rectangle (%.3f x %.3f)", width, height);
  return OPERATOR_FINISHED;
}

static void EXCAD_OT_sketch_add_rect(wmOperatorType *ot)
{
  ot->name = "Sketch Rectangle";
  ot->idname = "EXCAD_OT_sketch_add_rect";
  ot->description = "Add a Fusion-style H/V/dimension-constrained rectangle to the active sketch";
  ot->exec = excad_sketch_add_rect_exec;
  ot->invoke = WM_operator_props_popup_confirm;
  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  RNA_def_float(ot->srna, "u0", -0.5f, -1000.0f, 1000.0f, "U Origin", "", -100.0f, 100.0f);
  RNA_def_float(ot->srna, "v0", -0.5f, -1000.0f, 1000.0f, "V Origin", "", -100.0f, 100.0f);
  RNA_def_float(ot->srna, "width", 1.0f, 0.001f, 1000.0f, "Width", "", 0.01f, 100.0f);
  RNA_def_float(ot->srna, "height", 1.0f, 0.001f, 1000.0f, "Height", "", 0.01f, 100.0f);
}

static wmOperatorStatus excad_sketch_add_circle_exec(bContext * /*C*/, wmOperator *op)
{
  const double cu = RNA_float_get(op->ptr, "cu");
  const double cv = RNA_float_get(op->ptr, "cv");
  const double radius = RNA_float_get(op->ptr, "radius");
  SketchDocument::active().add_circle(cu, cv, radius);
  BKE_reportf(op->reports, RPT_INFO, "EX-CAD: Added circle (r=%.3f)", radius);
  return OPERATOR_FINISHED;
}

static void EXCAD_OT_sketch_add_circle(wmOperatorType *ot)
{
  ot->name = "Sketch Circle";
  ot->idname = "EXCAD_OT_sketch_add_circle";
  ot->description = "Add a radius-constrained circle to the active sketch";
  ot->exec = excad_sketch_add_circle_exec;
  ot->invoke = WM_operator_props_popup_confirm;
  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  RNA_def_float(ot->srna, "cu", 0.0f, -1000.0f, 1000.0f, "Center U", "", -100.0f, 100.0f);
  RNA_def_float(ot->srna, "cv", 0.0f, -1000.0f, 1000.0f, "Center V", "", -100.0f, 100.0f);
  RNA_def_float(ot->srna, "radius", 0.5f, 0.001f, 1000.0f, "Radius", "", 0.01f, 100.0f);
}

static wmOperatorStatus excad_sketch_add_line_exec(bContext * /*C*/, wmOperator *op)
{
  const double u0 = RNA_float_get(op->ptr, "u0");
  const double v0 = RNA_float_get(op->ptr, "v0");
  const double u1 = RNA_float_get(op->ptr, "u1");
  const double v1 = RNA_float_get(op->ptr, "v1");
  SketchDocument::active().add_line(u0, v0, u1, v1);
  return OPERATOR_FINISHED;
}

static void EXCAD_OT_sketch_add_line(wmOperatorType *ot)
{
  ot->name = "Sketch Line";
  ot->idname = "EXCAD_OT_sketch_add_line";
  ot->description = "Add a 2D line segment to the active sketch";
  ot->exec = excad_sketch_add_line_exec;
  ot->invoke = WM_operator_props_popup_confirm;
  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  RNA_def_float(ot->srna, "u0", 0.0f, -1000.0f, 1000.0f, "U0", "", -100.0f, 100.0f);
  RNA_def_float(ot->srna, "v0", 0.0f, -1000.0f, 1000.0f, "V0", "", -100.0f, 100.0f);
  RNA_def_float(ot->srna, "u1", 1.0f, -1000.0f, 1000.0f, "U1", "", -100.0f, 100.0f);
  RNA_def_float(ot->srna, "v1", 0.0f, -1000.0f, 1000.0f, "V1", "", -100.0f, 100.0f);
}

static wmOperatorStatus excad_sketch_add_constraint_exec(bContext * /*C*/, wmOperator *op)
{
  const int ctype = RNA_enum_get(op->ptr, "constraint_type");
  const int a = RNA_int_get(op->ptr, "a");
  const int b = RNA_int_get(op->ptr, "b");
  const double value = RNA_float_get(op->ptr, "value");
  SketchDocument::active().add_constraint(
      static_cast<SketchConstraintType>(ctype), a, b, -1, -1, value, 0.0);
  return OPERATOR_FINISHED;
}

static void EXCAD_OT_sketch_add_constraint(wmOperatorType *ot)
{
  static const EnumPropertyItem constraint_items[] = {
      {static_cast<int>(SketchConstraintType::COINCIDENT), "COINCIDENT", 0, "Coincident", ""},
      {static_cast<int>(SketchConstraintType::HORIZONTAL), "HORIZONTAL", 0, "Horizontal", ""},
      {static_cast<int>(SketchConstraintType::VERTICAL), "VERTICAL", 0, "Vertical", ""},
      {static_cast<int>(SketchConstraintType::DISTANCE), "DISTANCE", 0, "Distance", ""},
      {static_cast<int>(SketchConstraintType::RADIUS), "RADIUS", 0, "Radius", ""},
      {0, nullptr, 0, nullptr, nullptr},
  };
  ot->name = "Add Sketch Constraint";
  ot->idname = "EXCAD_OT_sketch_add_constraint";
  ot->description = "Add a geometric or dimensional constraint to the active sketch";
  ot->exec = excad_sketch_add_constraint_exec;
  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  RNA_def_enum(ot->srna,
               "constraint_type",
               constraint_items,
               static_cast<int>(SketchConstraintType::DISTANCE),
               "Type",
               "");
  RNA_def_int(ot->srna, "a", 0, 0, 10000, "Index A", "", 0, 1000);
  RNA_def_int(ot->srna, "b", 1, -1, 10000, "Index B", "", -1, 1000);
  RNA_def_float(ot->srna, "value", 1.0f, 0.0f, 10000.0f, "Value", "", 0.0f, 100.0f);
}

static wmOperatorStatus excad_sketch_solve_exec(bContext * /*C*/, wmOperator *op)
{
  const SolveResult res = SketchDocument::active().solve();
  BKE_reportf(op->reports,
              RPT_INFO,
              "EX-CAD Solve: %s (%d iters, residual=%.2e)",
              res.converged ? "converged" : "max iters",
              res.iterations,
              res.max_residual);
  return OPERATOR_FINISHED;
}

static void EXCAD_OT_sketch_solve(wmOperatorType *ot)
{
  ot->name = "Solve Sketch";
  ot->idname = "EXCAD_OT_sketch_solve";
  ot->description = "Solve all geometric and dimensional constraints in the active sketch";
  ot->exec = excad_sketch_solve_exec;
  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;
}

static wmOperatorStatus excad_sketch_extrude_exec(bContext *C, wmOperator *op)
{
  SketchDocument &doc = SketchDocument::active();
  doc.solve();
  const float depth = RNA_float_get(op->ptr, "depth");
  const bool symmetric = RNA_boolean_get(op->ptr, "symmetric");
  const RawMeshData raw = extrude_sketch(doc, depth, symmetric);
  if (raw.verts.empty()) {
    BKE_report(op->reports, RPT_WARNING, "EX-CAD: Add a closed rectangle, circle, or loop first");
    return OPERATOR_CANCELLED;
  }
  create_mesh_object_from_raw(C, "EXCAD_Extrude", raw);
  return OPERATOR_FINISHED;
}

static void EXCAD_OT_sketch_extrude(wmOperatorType *ot)
{
  ot->name = "Extrude Sketch";
  ot->idname = "EXCAD_OT_sketch_extrude";
  ot->description = "Extrude the active solved 2D sketch into a 3D solid mesh";
  ot->exec = excad_sketch_extrude_exec;
  ot->invoke = WM_operator_props_popup_confirm;
  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  RNA_def_float(ot->srna, "depth", 1.0f, 0.001f, 1000.0f, "Depth", "", 0.01f, 100.0f);
  RNA_def_boolean(ot->srna, "symmetric", false, "Symmetric", "");
}

static wmOperatorStatus excad_sketch_revolve_exec(bContext *C, wmOperator *op)
{
  SketchDocument &doc = SketchDocument::active();
  doc.solve();
  const float angle_deg = RNA_float_get(op->ptr, "angle_deg");
  const int segments = RNA_int_get(op->ptr, "segments");
  const RawMeshData raw = revolve_sketch(doc, angle_deg, segments);
  if (raw.verts.empty()) {
    BKE_report(op->reports, RPT_WARNING, "EX-CAD: Add a closed profile first");
    return OPERATOR_CANCELLED;
  }
  create_mesh_object_from_raw(C, "EXCAD_Revolve", raw);
  return OPERATOR_FINISHED;
}

static void EXCAD_OT_sketch_revolve(wmOperatorType *ot)
{
  ot->name = "Revolve Sketch";
  ot->idname = "EXCAD_OT_sketch_revolve";
  ot->description = "Revolve the active solved 2D sketch around the V-axis";
  ot->exec = excad_sketch_revolve_exec;
  ot->invoke = WM_operator_props_popup_confirm;
  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  RNA_def_float(ot->srna, "angle_deg", 360.0f, 1.0f, 360.0f, "Angle (deg)", "", 1.0f, 360.0f);
  RNA_def_int(ot->srna, "segments", 48, 8, 256, "Segments", "", 8, 128);
}

/* -------------------------------------------------------------------- */
/* Native C++ SDF + Marching Cubes Operators                            */
/* -------------------------------------------------------------------- */

static wmOperatorStatus excad_sdf_humanoid_exec(bContext *C, wmOperator *op)
{
  HumanoidParams hp;
  hp.height = RNA_float_get(op->ptr, "height");
  hp.muscularity = RNA_float_get(op->ptr, "muscularity");
  hp.shoulder_ratio = RNA_float_get(op->ptr, "shoulder_ratio");
  hp.hip_ratio = RNA_float_get(op->ptr, "hip_ratio");
  hp.resolution = RNA_int_get(op->ptr, "resolution");

  const RawMeshData raw = generate_humanoid_sdf(hp);
  create_mesh_object_from_raw(C, "EXCAD_Humanoid", raw);
  BKE_reportf(op->reports,
              RPT_INFO,
              "EX-CAD: Generated anatomical SDF humanoid (%zu verts, %zu faces)",
              raw.verts.size(),
              raw.faces.size());
  return OPERATOR_FINISHED;
}

static void EXCAD_OT_sdf_humanoid(wmOperatorType *ot)
{
  ot->name = "Generate Anatomical Humanoid (SDF)";
  ot->idname = "EXCAD_OT_sdf_humanoid";
  ot->description = "Generate a smooth watertight anatomical humanoid mesh using C++ SDF + Marching Cubes";
  ot->exec = excad_sdf_humanoid_exec;
  ot->invoke = WM_operator_props_popup_confirm;
  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  RNA_def_float(ot->srna, "height", 1.75f, 0.5f, 3.0f, "Height (m)", "", 1.0f, 2.5f);
  RNA_def_float(ot->srna, "muscularity", 0.5f, 0.0f, 1.5f, "Muscularity", "", 0.0f, 1.2f);
  RNA_def_float(ot->srna, "shoulder_ratio", 1.0f, 0.7f, 1.4f, "Shoulder Ratio", "", 0.8f, 1.3f);
  RNA_def_float(ot->srna, "hip_ratio", 1.0f, 0.7f, 1.4f, "Hip Ratio", "", 0.8f, 1.3f);
  RNA_def_int(ot->srna, "resolution", 72, 24, 128, "Grid Resolution", "", 32, 112);
}

static wmOperatorStatus excad_sdf_custom_exec(bContext *C, wmOperator *op)
{
  char spec_buf[4096];
  RNA_string_get(op->ptr, "spec", spec_buf);
  const int resolution = RNA_int_get(op->ptr, "resolution");
  const RawMeshData raw = generate_custom_sdf_from_spec(std::string(spec_buf), resolution);
  create_mesh_object_from_raw(C, "EXCAD_SDF", raw);
  return OPERATOR_FINISHED;
}

static void EXCAD_OT_sdf_custom(wmOperatorType *ot)
{
  ot->name = "Generate Custom SDF Mesh";
  ot->idname = "EXCAD_OT_sdf_custom";
  ot->description = "Generate a smooth SDF mesh from primitive specification lines";
  ot->exec = excad_sdf_custom_exec;
  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  RNA_def_string(ot->srna,
                 "spec",
                 "sphere 0 0 0.5 0.5 0.05",
                 4096,
                 "SDF Spec",
                 "Newline-separated SDF primitives");
  RNA_def_int(ot->srna, "resolution", 64, 24, 128, "Resolution", "", 24, 128);
}

/* -------------------------------------------------------------------- */
/* Native C++ AI Direct Bridge Operators                               */
/* -------------------------------------------------------------------- */

struct BridgeModalData {
  wmTimer *timer = nullptr;
};

static wmOperatorStatus excad_bridge_start_modal(bContext *C, wmOperator *op, const wmEvent *event)
{
  BridgeModalData *data = static_cast<BridgeModalData *>(op->customdata);
  if (!AiBridgeServer::instance().is_running()) {
    if (data && data->timer) {
      WM_event_timer_remove(CTX_wm_manager(C), CTX_wm_window(C), data->timer);
    }
    MEM_delete(data);
    op->customdata = nullptr;
    return OPERATOR_FINISHED;
  }

  if (event->type == TIMER && data && event->customdata == data->timer) {
    AiBridgeServer::instance().process_pending(C);
  }
  return OPERATOR_PASS_THROUGH;
}

static wmOperatorStatus excad_bridge_start_invoke(bContext *C,
                                                  wmOperator *op,
                                                  const wmEvent * /*event*/)
{
  const int port = RNA_int_get(op->ptr, "port");
  if (!AiBridgeServer::instance().start(port)) {
    BKE_report(op->reports, RPT_ERROR, "EX-CAD: Failed to start AI Direct Bridge");
    return OPERATOR_CANCELLED;
  }

  BridgeModalData *data = MEM_new<BridgeModalData>(__func__);
  data->timer = WM_event_timer_add(CTX_wm_manager(C), CTX_wm_window(C), TIMER, 0.05);
  op->customdata = data;

  WM_event_add_modal_handler(C, op);
  BKE_reportf(op->reports, RPT_INFO, "EX-CAD: AI Direct Bridge listening on 127.0.0.1:%d", port);
  return OPERATOR_RUNNING_MODAL;
}

static void EXCAD_OT_bridge_start(wmOperatorType *ot)
{
  ot->name = "Start AI Direct Bridge";
  ot->idname = "EXCAD_OT_bridge_start";
  ot->description = "Start the native C++ TCP JSON-RPC bridge for direct AI mesh & sketch control";
  ot->invoke = excad_bridge_start_invoke;
  ot->modal = excad_bridge_start_modal;
  ot->flag = OPTYPE_REGISTER;

  RNA_def_int(ot->srna, "port", 9877, 1024, 65535, "Port", "", 1024, 65535);
}

static wmOperatorStatus excad_bridge_stop_exec(bContext * /*C*/, wmOperator *op)
{
  AiBridgeServer::instance().stop();
  BKE_report(op->reports, RPT_INFO, "EX-CAD: AI Direct Bridge stopped");
  return OPERATOR_FINISHED;
}

static void EXCAD_OT_bridge_stop(wmOperatorType *ot)
{
  ot->name = "Stop AI Direct Bridge";
  ot->idname = "EXCAD_OT_bridge_stop";
  ot->description = "Stop the native C++ TCP JSON-RPC bridge";
  ot->exec = excad_bridge_stop_exec;
  ot->flag = OPTYPE_REGISTER;
}

void operatortypes_excad()
{
  WM_operatortype_append(EXCAD_OT_sketch_new);
  WM_operatortype_append(EXCAD_OT_sketch_add_rect);
  WM_operatortype_append(EXCAD_OT_sketch_add_circle);
  WM_operatortype_append(EXCAD_OT_sketch_add_line);
  WM_operatortype_append(EXCAD_OT_sketch_add_constraint);
  WM_operatortype_append(EXCAD_OT_sketch_solve);
  WM_operatortype_append(EXCAD_OT_sketch_extrude);
  WM_operatortype_append(EXCAD_OT_sketch_revolve);
  WM_operatortype_append(EXCAD_OT_sdf_humanoid);
  WM_operatortype_append(EXCAD_OT_sdf_custom);
  WM_operatortype_append(EXCAD_OT_bridge_start);
  WM_operatortype_append(EXCAD_OT_bridge_stop);
}

}  // namespace blender::ed::excad
