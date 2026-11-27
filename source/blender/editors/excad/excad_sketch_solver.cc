/* SPDX-FileCopyrightText: 2026 EX-CAD Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "excad_sketch_solver.hh"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

namespace blender::ed::excad {

namespace {

constexpr double kPi = 3.14159265358979323846;

bool solve_linear_system(std::vector<std::vector<double>> &A,
                         std::vector<double> &b,
                         std::vector<double> &x)
{
  const int n = static_cast<int>(b.size());
  x.assign(n, 0.0);
  if (n == 0) {
    return true;
  }

  for (int i = 0; i < n; ++i) {
    int pivot = i;
    double max_val = std::abs(A[i][i]);
    for (int r = i + 1; r < n; ++r) {
      const double v = std::abs(A[r][i]);
      if (v > max_val) {
        max_val = v;
        pivot = r;
      }
    }
    if (max_val < 1e-14) {
      continue;
    }
    if (pivot != i) {
      std::swap(A[i], A[pivot]);
      std::swap(b[i], b[pivot]);
    }
    const double diag = A[i][i];
    for (int r = i + 1; r < n; ++r) {
      const double factor = A[r][i] / diag;
      if (factor == 0.0) {
        continue;
      }
      for (int c = i; c < n; ++c) {
        A[r][c] -= factor * A[i][c];
      }
      b[r] -= factor * b[i];
    }
  }

  for (int i = n - 1; i >= 0; --i) {
    double sum = b[i];
    for (int c = i + 1; c < n; ++c) {
      sum -= A[i][c] * x[c];
    }
    if (std::abs(A[i][i]) < 1e-14) {
      x[i] = 0.0;
    }
    else {
      x[i] = sum / A[i][i];
    }
  }
  return true;
}

void eval_residuals(const SketchDocument &doc,
                    const std::vector<double> &state,
                    std::vector<double> &res)
{
  res.clear();
  const int n_pts = static_cast<int>(doc.points.size());
  auto get_u = [&](int pid) -> double { return state[2 * pid]; };
  auto get_v = [&](int pid) -> double { return state[2 * pid + 1]; };
  auto get_r = [&](int eid) -> double { return state[2 * n_pts + eid]; };

  for (const SketchConstraint &c : doc.constraints) {
    switch (c.type) {
      case SketchConstraintType::COINCIDENT: {
        if (c.a >= 0 && c.a < n_pts && c.b >= 0 && c.b < n_pts) {
          res.push_back(get_u(c.a) - get_u(c.b));
          res.push_back(get_v(c.a) - get_v(c.b));
        }
        break;
      }
      case SketchConstraintType::HORIZONTAL: {
        if (c.a >= 0 && c.a < n_pts && c.b >= 0 && c.b < n_pts) {
          res.push_back(get_v(c.a) - get_v(c.b));
        }
        break;
      }
      case SketchConstraintType::VERTICAL: {
        if (c.a >= 0 && c.a < n_pts && c.b >= 0 && c.b < n_pts) {
          res.push_back(get_u(c.a) - get_u(c.b));
        }
        break;
      }
      case SketchConstraintType::DISTANCE: {
        if (c.a >= 0 && c.a < n_pts && c.b >= 0 && c.b < n_pts) {
          const double du = get_u(c.b) - get_u(c.a);
          const double dv = get_v(c.b) - get_v(c.a);
          const double d = std::sqrt(du * du + dv * dv + 1e-12);
          res.push_back(d - c.value);
        }
        break;
      }
      case SketchConstraintType::RADIUS: {
        if (c.a >= 0 && c.a < static_cast<int>(doc.entities.size())) {
          res.push_back(get_r(c.a) - c.value);
        }
        break;
      }
      case SketchConstraintType::PARALLEL: {
        if (c.a >= 0 && c.b >= 0 && c.c >= 0 && c.d >= 0 && c.a < n_pts && c.b < n_pts &&
            c.c < n_pts && c.d < n_pts)
        {
          const double dx1 = get_u(c.b) - get_u(c.a);
          const double dy1 = get_v(c.b) - get_v(c.a);
          const double dx2 = get_u(c.d) - get_u(c.c);
          const double dy2 = get_v(c.d) - get_v(c.c);
          res.push_back(dx1 * dy2 - dy1 * dx2);
        }
        break;
      }
      case SketchConstraintType::PERPENDICULAR: {
        if (c.a >= 0 && c.b >= 0 && c.c >= 0 && c.d >= 0 && c.a < n_pts && c.b < n_pts &&
            c.c < n_pts && c.d < n_pts)
        {
          const double dx1 = get_u(c.b) - get_u(c.a);
          const double dy1 = get_v(c.b) - get_v(c.a);
          const double dx2 = get_u(c.d) - get_u(c.c);
          const double dy2 = get_v(c.d) - get_v(c.c);
          res.push_back(dx1 * dx2 + dy1 * dy2);
        }
        break;
      }
      case SketchConstraintType::EQUAL_LENGTH: {
        if (c.a >= 0 && c.b >= 0 && c.c >= 0 && c.d >= 0 && c.a < n_pts && c.b < n_pts &&
            c.c < n_pts && c.d < n_pts)
        {
          const double d1 = std::hypot(get_u(c.b) - get_u(c.a), get_v(c.b) - get_v(c.a));
          const double d2 = std::hypot(get_u(c.d) - get_u(c.c), get_v(c.d) - get_v(c.c));
          res.push_back(d1 - d2);
        }
        break;
      }
      case SketchConstraintType::FIXED: {
        if (c.a >= 0 && c.a < n_pts) {
          res.push_back(get_u(c.a) - c.value);
          res.push_back(get_v(c.a) - c.value2);
        }
        break;
      }
    }
  }
}

void build_2d_loops(const SketchDocument &doc,
                    std::vector<std::vector<std::array<double, 2>>> &loops,
                    int circle_segments)
{
  loops.clear();
  std::vector<std::pair<int, int>> edges;
  for (const SketchEntity &e : doc.entities) {
    if (e.type == SketchEntityType::CIRCLE && e.p0 >= 0 &&
        e.p0 < static_cast<int>(doc.points.size()))
    {
      const SketchPoint &center = doc.points[e.p0];
      std::vector<std::array<double, 2>> loop;
      loop.reserve(circle_segments);
      for (int i = 0; i < circle_segments; ++i) {
        const double ang = (2.0 * kPi * i) / circle_segments;
        loop.push_back({center.u + e.radius * std::cos(ang), center.v + e.radius * std::sin(ang)});
      }
      loops.push_back(std::move(loop));
    }
    else if (e.type == SketchEntityType::LINE && e.p0 >= 0 && e.p1 >= 0 &&
             e.p0 < static_cast<int>(doc.points.size()) &&
             e.p1 < static_cast<int>(doc.points.size()))
    {
      edges.emplace_back(e.p0, e.p1);
    }
  }

  /* Extract closed polygon loops from line segments. */
  std::vector<bool> used(edges.size(), false);
  for (size_t i = 0; i < edges.size(); ++i) {
    if (used[i]) {
      continue;
    }
    used[i] = true;
    std::vector<int> chain = {edges[i].first, edges[i].second};
    bool extended = true;
    while (extended) {
      extended = false;
      const int tail = chain.back();
      if (tail == chain.front() && chain.size() >= 4) {
        break;
      }
      for (size_t j = 0; j < edges.size(); ++j) {
        if (used[j]) {
          continue;
        }
        if (edges[j].first == tail) {
          used[j] = true;
          chain.push_back(edges[j].second);
          extended = true;
          break;
        }
        if (edges[j].second == tail) {
          used[j] = true;
          chain.push_back(edges[j].first);
          extended = true;
          break;
        }
      }
    }
    if (chain.size() >= 4 && chain.front() == chain.back()) {
      chain.pop_back();
      std::vector<std::array<double, 2>> loop;
      loop.reserve(chain.size());
      for (int pid : chain) {
        loop.push_back({doc.points[pid].u, doc.points[pid].v});
      }
      /* Ensure counter-clockwise winding in (u, v). */
      double area2 = 0.0;
      for (size_t k = 0; k < loop.size(); ++k) {
        const auto &a = loop[k];
        const auto &b = loop[(k + 1) % loop.size()];
        area2 += (a[0] * b[1] - b[0] * a[1]);
      }
      if (area2 < 0.0) {
        std::reverse(loop.begin(), loop.end());
      }
      loops.push_back(std::move(loop));
    }
  }
}

}  // namespace

SketchDocument &SketchDocument::active()
{
  static SketchDocument instance;
  return instance;
}

void SketchDocument::reset(SketchPlane new_plane)
{
  plane = new_plane;
  points.clear();
  entities.clear();
  constraints.clear();
}

int SketchDocument::add_point(double u, double v, bool fixed)
{
  const int id = static_cast<int>(points.size());
  points.push_back(SketchPoint{id, u, v, fixed});
  return id;
}

int SketchDocument::add_line_pts(int p0, int p1)
{
  const int id = static_cast<int>(entities.size());
  entities.push_back(SketchEntity{id, SketchEntityType::LINE, p0, p1, 0.0});
  return id;
}

int SketchDocument::add_line(double u0, double v0, double u1, double v1)
{
  const int p0 = add_point(u0, v0, false);
  const int p1 = add_point(u1, v1, false);
  return add_line_pts(p0, p1);
}

void SketchDocument::add_rect(double u0, double v0, double width, double height)
{
  const int p0 = add_point(u0, v0, false);
  const int p1 = add_point(u0 + width, v0, false);
  const int p2 = add_point(u0 + width, v0 + height, false);
  const int p3 = add_point(u0, v0 + height, false);

  add_line_pts(p0, p1);
  add_line_pts(p1, p2);
  add_line_pts(p2, p3);
  add_line_pts(p3, p0);

  /* Fusion 360 style auto-constraints: Horizontal, Vertical, Width & Height dimensions. */
  add_constraint(SketchConstraintType::HORIZONTAL, p0, p1);
  add_constraint(SketchConstraintType::HORIZONTAL, p3, p2);
  add_constraint(SketchConstraintType::VERTICAL, p1, p2);
  add_constraint(SketchConstraintType::VERTICAL, p0, p3);
  add_constraint(SketchConstraintType::DISTANCE, p0, p1, -1, -1, std::abs(width));
  add_constraint(SketchConstraintType::DISTANCE, p1, p2, -1, -1, std::abs(height));
}

int SketchDocument::add_circle(double cu, double cv, double radius)
{
  const int center = add_point(cu, cv, false);
  const int id = static_cast<int>(entities.size());
  entities.push_back(SketchEntity{id, SketchEntityType::CIRCLE, center, -1, std::abs(radius)});
  add_constraint(SketchConstraintType::RADIUS, id, -1, -1, -1, std::abs(radius));
  return id;
}

int SketchDocument::add_constraint(SketchConstraintType type,
                                   int a,
                                   int b,
                                   int c,
                                   int d,
                                   double value,
                                   double value2)
{
  const int id = static_cast<int>(constraints.size());
  constraints.push_back(SketchConstraint{id, type, a, b, c, d, value, value2});
  return id;
}

SolveResult SketchDocument::solve(int max_iters, double tol)
{
  SolveResult result;
  const int n_pts = static_cast<int>(points.size());
  const int n_ent = static_cast<int>(entities.size());
  const int n_vars = 2 * n_pts + n_ent;
  if (n_vars == 0 || constraints.empty()) {
    result.converged = true;
    return result;
  }

  std::vector<double> state(n_vars, 0.0);
  std::vector<bool> var_fixed(n_vars, false);
  for (int i = 0; i < n_pts; ++i) {
    state[2 * i] = points[i].u;
    state[2 * i + 1] = points[i].v;
    if (points[i].fixed) {
      var_fixed[2 * i] = true;
      var_fixed[2 * i + 1] = true;
    }
  }
  for (int i = 0; i < n_ent; ++i) {
    state[2 * n_pts + i] = entities[i].radius;
    if (entities[i].type != SketchEntityType::CIRCLE) {
      var_fixed[2 * n_pts + i] = true;
    }
  }

  std::vector<double> r0, r1;
  const double eps = 1e-6;
  const double damping = 1e-6;

  for (int iter = 0; iter < max_iters; ++iter) {
    eval_residuals(*this, state, r0);
    const int m = static_cast<int>(r0.size());
    double max_res = 0.0;
    for (double v : r0) {
      max_res = std::max(max_res, std::abs(v));
    }
    result.iterations = iter;
    result.max_residual = max_res;
    if (max_res < tol || m == 0) {
      result.converged = true;
      break;
    }

    /* Compute finite-difference Jacobian J (m x n_vars). */
    std::vector<std::vector<double>> J(m, std::vector<double>(n_vars, 0.0));
    for (int col = 0; col < n_vars; ++col) {
      if (var_fixed[col]) {
        continue;
      }
      const double orig = state[col];
      state[col] = orig + eps;
      eval_residuals(*this, state, r1);
      state[col] = orig;
      for (int row = 0; row < m; ++row) {
        J[row][col] = (r1[row] - r0[row]) / eps;
      }
    }

    /* Normal equations: (J^T J + damping * I) * dx = -J^T * r0. */
    std::vector<std::vector<double>> A(n_vars, std::vector<double>(n_vars, 0.0));
    std::vector<double> b(n_vars, 0.0);
    for (int i = 0; i < n_vars; ++i) {
      A[i][i] = damping;
      for (int row = 0; row < m; ++row) {
        b[i] -= J[row][i] * r0[row];
        const double Ji = J[row][i];
        if (Ji == 0.0) {
          continue;
        }
        for (int j = 0; j < n_vars; ++j) {
          A[i][j] += Ji * J[row][j];
        }
      }
    }

    std::vector<double> dx;
    solve_linear_system(A, b, dx);
    for (int i = 0; i < n_vars; ++i) {
      if (!var_fixed[i]) {
        state[i] += std::clamp(dx[i], -1.0, 1.0);
      }
    }
  }

  for (int i = 0; i < n_pts; ++i) {
    points[i].u = state[2 * i];
    points[i].v = state[2 * i + 1];
  }
  for (int i = 0; i < n_ent; ++i) {
    if (entities[i].type == SketchEntityType::CIRCLE) {
      entities[i].radius = std::max(1e-4, state[2 * n_pts + i]);
    }
  }
  return result;
}

std::array<float, 3> plane_to_world(SketchPlane plane, double u, double v, double depth)
{
  switch (plane) {
    case SketchPlane::XY:
      return {static_cast<float>(u), static_cast<float>(v), static_cast<float>(depth)};
    case SketchPlane::XZ:
      return {static_cast<float>(u), static_cast<float>(depth), static_cast<float>(v)};
    case SketchPlane::YZ:
      return {static_cast<float>(depth), static_cast<float>(u), static_cast<float>(v)};
  }
  return {static_cast<float>(u), static_cast<float>(v), static_cast<float>(depth)};
}

RawMeshData extrude_sketch(const SketchDocument &doc,
                           float depth,
                           bool symmetric,
                           int circle_segments)
{
  RawMeshData mesh;
  std::vector<std::vector<std::array<double, 2>>> loops;
  build_2d_loops(doc, loops, std::max(8, circle_segments));

  const double d0 = symmetric ? -0.5 * static_cast<double>(depth) : 0.0;
  const double d1 = symmetric ? 0.5 * static_cast<double>(depth) : static_cast<double>(depth);
  const double z_min = std::min(d0, d1);
  const double z_max = std::max(d0, d1);

  for (const auto &loop : loops) {
    const int n = static_cast<int>(loop.size());
    if (n < 3) {
      continue;
    }
    const int base_idx = static_cast<int>(mesh.verts.size());
    for (int i = 0; i < n; ++i) {
      mesh.verts.push_back(plane_to_world(doc.plane, loop[i][0], loop[i][1], z_min));
    }
    for (int i = 0; i < n; ++i) {
      mesh.verts.push_back(plane_to_world(doc.plane, loop[i][0], loop[i][1], z_max));
    }

    /* Bottom cap (reversed winding) */
    std::vector<int> bottom_face(n);
    for (int i = 0; i < n; ++i) {
      bottom_face[i] = base_idx + (n - 1 - i);
    }
    mesh.faces.push_back(std::move(bottom_face));

    /* Top cap */
    std::vector<int> top_face(n);
    for (int i = 0; i < n; ++i) {
      top_face[i] = base_idx + n + i;
    }
    mesh.faces.push_back(std::move(top_face));

    /* Side quads */
    for (int i = 0; i < n; ++i) {
      const int next = (i + 1) % n;
      mesh.faces.push_back(
          {base_idx + i, base_idx + next, base_idx + n + next, base_idx + n + i});
    }
  }
  return mesh;
}

RawMeshData revolve_sketch(const SketchDocument &doc, float angle_deg, int segments)
{
  RawMeshData mesh;
  std::vector<std::vector<std::array<double, 2>>> loops;
  build_2d_loops(doc, loops, std::max(8, segments));

  const int segs = std::max(3, segments);
  const double clamped_deg = std::clamp(static_cast<double>(angle_deg), 1.0, 360.0);
  const bool full_circle = (clamped_deg >= 359.99);
  const double total_rad = (full_circle ? 360.0 : clamped_deg) * (kPi / 180.0);
  const int rings = full_circle ? segs : (segs + 1);

  for (const auto &loop : loops) {
    const int n = static_cast<int>(loop.size());
    if (n < 3) {
      continue;
    }
    const int base_idx = static_cast<int>(mesh.verts.size());

    for (int r = 0; r < rings; ++r) {
      const double theta = full_circle ? ((2.0 * kPi * r) / segs) : ((total_rad * r) / segs);
      const double ct = std::cos(theta);
      const double st = std::sin(theta);
      for (int i = 0; i < n; ++i) {
        const double radius = loop[i][0];
        const double v = loop[i][1];
        const double u = radius * ct;
        const double depth = radius * st;
        mesh.verts.push_back(plane_to_world(doc.plane, u, v, depth));
      }
    }

    const int ring_pairs = full_circle ? segs : segs;
    for (int r = 0; r < ring_pairs; ++r) {
      const int r_next = (r + 1) % rings;
      for (int i = 0; i < n; ++i) {
        const int i_next = (i + 1) % n;
        const int v00 = base_idx + r * n + i;
        const int v01 = base_idx + r * n + i_next;
        const int v11 = base_idx + r_next * n + i_next;
        const int v10 = base_idx + r_next * n + i;
        mesh.faces.push_back({v00, v01, v11, v10});
      }
    }

    if (!full_circle) {
      std::vector<int> start_cap(n);
      std::vector<int> end_cap(n);
      for (int i = 0; i < n; ++i) {
        start_cap[i] = base_idx + (n - 1 - i);
        end_cap[i] = base_idx + (rings - 1) * n + i;
      }
      mesh.faces.push_back(std::move(start_cap));
      mesh.faces.push_back(std::move(end_cap));
    }
  }
  return mesh;
}

}  // namespace blender::ed::excad
