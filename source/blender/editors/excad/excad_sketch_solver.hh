/* SPDX-FileCopyrightText: 2026 EX-CAD Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <array>
#include <vector>

#include "excad_mesh_types.hh"

namespace blender::ed::excad {

enum class SketchPlane : int {
  XY = 0,
  XZ = 1,
  YZ = 2,
};

struct SketchPoint {
  int id = 0;
  double u = 0.0;
  double v = 0.0;
  bool fixed = false;
};

enum class SketchEntityType : int {
  LINE = 0,
  CIRCLE = 1,
};

struct SketchEntity {
  int id = 0;
  SketchEntityType type = SketchEntityType::LINE;
  int p0 = -1;
  int p1 = -1;
  double radius = 0.5;
};

enum class SketchConstraintType : int {
  COINCIDENT = 0,
  HORIZONTAL = 1,
  VERTICAL = 2,
  DISTANCE = 3,
  RADIUS = 4,
  PARALLEL = 5,
  PERPENDICULAR = 6,
  EQUAL_LENGTH = 7,
  FIXED = 8,
};

struct SketchConstraint {
  int id = 0;
  SketchConstraintType type = SketchConstraintType::COINCIDENT;
  int a = -1;
  int b = -1;
  int c = -1;
  int d = -1;
  double value = 0.0;
  double value2 = 0.0;
};

struct SolveResult {
  bool converged = false;
  int iterations = 0;
  double max_residual = 0.0;
};

class SketchDocument {
 public:
  SketchPlane plane = SketchPlane::XY;
  std::vector<SketchPoint> points;
  std::vector<SketchEntity> entities;
  std::vector<SketchConstraint> constraints;

  void reset(SketchPlane new_plane);
  int add_point(double u, double v, bool fixed = false);
  int add_line(double u0, double v0, double u1, double v1);
  int add_line_pts(int p0, int p1);
  void add_rect(double u0, double v0, double width, double height);
  int add_circle(double cu, double cv, double radius);
  int add_constraint(SketchConstraintType type,
                     int a,
                     int b = -1,
                     int c = -1,
                     int d = -1,
                     double value = 0.0,
                     double value2 = 0.0);

  SolveResult solve(int max_iters = 80, double tol = 1e-7);

  static SketchDocument &active();
};

std::array<float, 3> plane_to_world(SketchPlane plane, double u, double v, double depth);

RawMeshData extrude_sketch(const SketchDocument &doc,
                           float depth,
                           bool symmetric,
                           int circle_segments = 48);

RawMeshData revolve_sketch(const SketchDocument &doc,
                           float angle_deg,
                           int segments = 48);

}  // namespace blender::ed::excad
