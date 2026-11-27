/* SPDX-FileCopyrightText: 2026 EX-CAD Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * EX-CAD Standalone: native C++ Win32 + OpenGL application.
 * Runs the EX-CAD core (parametric sketch solver, SDF + Marching Cubes mesher)
 * and the AI Direct Bridge (TCP 127.0.0.1:9877) without Blender.
 */

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <commdlg.h>
#include <shellapi.h>
#include <GL/gl.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "excad_sdf_mesh.hh"
#include "excad_sketch_solver.hh"

#ifndef GL_BGR_EXT
#  define GL_BGR_EXT 0x80E0
#endif

using namespace blender::ed::excad;

/* ------------------------------------------------------------------------- */
/* Scene                                                                     */
/* ------------------------------------------------------------------------- */

struct SceneObject {
  std::string name;
  RawMeshData mesh;
  std::vector<std::array<float, 3>> fnormals;
  std::vector<std::array<float, 3>> vnormals;
  bool smooth = true;
  float color[3] = {0.8f, 0.8f, 0.8f};
  GLuint list = 0;
  GLuint wire_list = 0;
};

static std::vector<SceneObject> g_scene;
static std::vector<GLuint> g_lists_to_delete;
static SketchDocument g_sketch;

static HINSTANCE g_inst = nullptr;
static HWND g_main = nullptr;
static HWND g_gl = nullptr;
static HDC g_hdc = nullptr;
static HGLRC g_glrc = nullptr;
static HFONT g_font = nullptr;
static HFONT g_font_bold = nullptr;
static std::map<int, HWND> g_ctrl;

static float g_yaw = -0.9f;
static float g_pitch = 0.25f;
static float g_dist = 4.2f;
static float g_target[3] = {0.0f, 0.0f, 0.85f};
static bool g_wire = false;
static POINT g_last_mouse{};
static int g_drag = 0; /* 0 none, 1 orbit, 2 pan */

static std::wstring g_screenshot_path;

static std::wstring widen(const std::string &s)
{
  return std::wstring(s.begin(), s.end());
}

static std::string narrow(const std::wstring &w)
{
  if (w.empty()) {
    return {};
  }
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
  std::string out(static_cast<size_t>(std::max(0, n - 1)), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, out.data(), n, nullptr, nullptr);
  return out;
}

static double signed_volume(const RawMeshData &m)
{
  double vol = 0.0;
  for (const auto &f : m.faces) {
    if (f.size() < 3) {
      continue;
    }
    const auto &a = m.verts[f[0]];
    for (size_t i = 1; i + 1 < f.size(); ++i) {
      const auto &b = m.verts[f[i]];
      const auto &c = m.verts[f[i + 1]];
      vol += double(a[0]) * (double(b[1]) * c[2] - double(b[2]) * c[1]) -
             double(a[1]) * (double(b[0]) * c[2] - double(b[2]) * c[0]) +
             double(a[2]) * (double(b[0]) * c[1] - double(b[1]) * c[0]);
    }
  }
  return vol / 6.0;
}

static void ensure_outward(RawMeshData &m)
{
  if (signed_volume(m) < 0.0) {
    for (auto &f : m.faces) {
      std::reverse(f.begin(), f.end());
    }
  }
}

static size_t count_open_edges(const RawMeshData &m)
{
  std::map<std::pair<int, int>, int> edges;
  for (const auto &f : m.faces) {
    for (size_t i = 0; i < f.size(); ++i) {
      int a = f[i];
      int b = f[(i + 1) % f.size()];
      if (a > b) {
        std::swap(a, b);
      }
      edges[{a, b}]++;
    }
  }
  size_t open = 0;
  for (const auto &kv : edges) {
    if (kv.second != 2) {
      ++open;
    }
  }
  return open;
}

static void compute_normals(SceneObject &o)
{
  const auto &m = o.mesh;
  o.fnormals.assign(m.faces.size(), {0.0f, 0.0f, 1.0f});
  o.vnormals.assign(m.verts.size(), {0.0f, 0.0f, 0.0f});
  for (size_t fi = 0; fi < m.faces.size(); ++fi) {
    const auto &f = m.faces[fi];
    double nx = 0, ny = 0, nz = 0;
    for (size_t i = 0; i < f.size(); ++i) {
      const auto &a = m.verts[f[i]];
      const auto &b = m.verts[f[(i + 1) % f.size()]];
      nx += (double(a[1]) - b[1]) * (double(a[2]) + b[2]);
      ny += (double(a[2]) - b[2]) * (double(a[0]) + b[0]);
      nz += (double(a[0]) - b[0]) * (double(a[1]) + b[1]);
    }
    for (int vid : f) {
      o.vnormals[vid][0] += float(nx);
      o.vnormals[vid][1] += float(ny);
      o.vnormals[vid][2] += float(nz);
    }
    const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-20) {
      o.fnormals[fi] = {float(nx / len), float(ny / len), float(nz / len)};
    }
  }
  for (auto &n : o.vnormals) {
    const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len > 1e-20f) {
      n[0] /= len;
      n[1] /= len;
      n[2] /= len;
    }
    else {
      n = {0.0f, 0.0f, 1.0f};
    }
  }
}

static void update_title();
static void redraw()
{
  if (g_gl) {
    InvalidateRect(g_gl, nullptr, FALSE);
  }
}

static std::string unique_name(const std::string &base)
{
  std::string name = base.empty() ? "Object" : base;
  int suffix = 1;
  auto exists = [&](const std::string &n) {
    for (const auto &o : g_scene) {
      if (o.name == n) {
        return true;
      }
    }
    return false;
  };
  std::string candidate = name;
  while (exists(candidate)) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), ".%03d", suffix++);
    candidate = name + buf;
  }
  return candidate;
}

static SceneObject &add_object(const std::string &name, RawMeshData &&mesh, bool smooth)
{
  static const float palette[][3] = {
      {0.82f, 0.72f, 0.62f},
      {0.55f, 0.68f, 0.85f},
      {0.70f, 0.82f, 0.58f},
      {0.88f, 0.62f, 0.55f},
      {0.75f, 0.65f, 0.88f},
      {0.85f, 0.82f, 0.55f},
  };
  ensure_outward(mesh);
  SceneObject o;
  o.name = unique_name(name);
  o.mesh = std::move(mesh);
  o.smooth = smooth;
  const auto &c = palette[g_scene.size() % 6];
  o.color[0] = c[0];
  o.color[1] = c[1];
  o.color[2] = c[2];
  compute_normals(o);
  g_scene.push_back(std::move(o));
  update_title();
  redraw();
  return g_scene.back();
}

static void release_object_gl(SceneObject &o)
{
  if (o.list) {
    g_lists_to_delete.push_back(o.list);
  }
  if (o.wire_list) {
    g_lists_to_delete.push_back(o.wire_list);
  }
  o.list = o.wire_list = 0;
}

static void clear_scene()
{
  for (auto &o : g_scene) {
    release_object_gl(o);
  }
  g_scene.clear();
  update_title();
  redraw();
}

static bool delete_last()
{
  if (g_scene.empty()) {
    return false;
  }
  release_object_gl(g_scene.back());
  g_scene.pop_back();
  update_title();
  redraw();
  return true;
}

static void frame_all()
{
  if (g_scene.empty()) {
    g_target[0] = 0.0f;
    g_target[1] = 0.0f;
    g_target[2] = 0.5f;
    g_dist = 4.0f;
    redraw();
    return;
  }
  float mn[3] = {1e30f, 1e30f, 1e30f};
  float mx[3] = {-1e30f, -1e30f, -1e30f};
  for (const auto &o : g_scene) {
    for (const auto &v : o.mesh.verts) {
      for (int k = 0; k < 3; ++k) {
        mn[k] = std::min(mn[k], v[k]);
        mx[k] = std::max(mx[k], v[k]);
      }
    }
  }
  float radius = 0.0f;
  for (int k = 0; k < 3; ++k) {
    g_target[k] = 0.5f * (mn[k] + mx[k]);
    radius = std::max(radius, 0.5f * (mx[k] - mn[k]));
  }
  g_dist = std::max(0.5f, radius * 3.0f);
  redraw();
}

static bool export_obj(const std::wstring &path)
{
  std::ofstream f(path.c_str());
  if (!f) {
    return false;
  }
  f << "# EX-CAD Standalone OBJ export\n";
  size_t offset = 1;
  for (const auto &o : g_scene) {
    f << "o " << o.name << "\n";
    for (const auto &v : o.mesh.verts) {
      f << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
    }
    for (const auto &face : o.mesh.faces) {
      f << "f";
      for (int vid : face) {
        f << " " << (vid + offset);
      }
      f << "\n";
    }
    offset += o.mesh.verts.size();
  }
  return true;
}

static bool export_stl(const std::wstring &path)
{
  std::ofstream f(path.c_str(), std::ios::binary);
  if (!f) {
    return false;
  }
  char header[80] = "EX-CAD Standalone binary STL";
  f.write(header, 80);
  uint32_t tri_count = 0;
  for (const auto &o : g_scene) {
    for (const auto &face : o.mesh.faces) {
      if (face.size() >= 3) {
        tri_count += uint32_t(face.size() - 2);
      }
    }
  }
  f.write(reinterpret_cast<const char *>(&tri_count), 4);
  for (const auto &o : g_scene) {
    for (size_t fi = 0; fi < o.mesh.faces.size(); ++fi) {
      const auto &face = o.mesh.faces[fi];
      for (size_t i = 1; i + 1 < face.size(); ++i) {
        f.write(reinterpret_cast<const char *>(o.fnormals[fi].data()), 12);
        f.write(reinterpret_cast<const char *>(o.mesh.verts[face[0]].data()), 12);
        f.write(reinterpret_cast<const char *>(o.mesh.verts[face[i]].data()), 12);
        f.write(reinterpret_cast<const char *>(o.mesh.verts[face[i + 1]].data()), 12);
        const uint16_t attr = 0;
        f.write(reinterpret_cast<const char *>(&attr), 2);
      }
    }
  }
  return true;
}

/* ------------------------------------------------------------------------- */
/* Rendering                                                                 */
/* ------------------------------------------------------------------------- */

static void look_at(const float eye[3], const float target[3])
{
  float f[3] = {target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]};
  float fl = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
  for (float &v : f) {
    v /= fl;
  }
  const float up[3] = {0.0f, 0.0f, 1.0f};
  float s[3] = {f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]};
  float sl = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
  for (float &v : s) {
    v /= sl;
  }
  const float u[3] = {s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0]};
  GLfloat m[16] = {s[0], u[0], -f[0], 0, s[1], u[1], -f[1], 0, s[2], u[2], -f[2], 0, 0, 0, 0, 1};
  glMultMatrixf(m);
  glTranslatef(-eye[0], -eye[1], -eye[2]);
}

static void build_lists(SceneObject &o)
{
  const auto &m = o.mesh;
  o.list = glGenLists(1);
  glNewList(o.list, GL_COMPILE);
  glBegin(GL_TRIANGLES);
  for (size_t fi = 0; fi < m.faces.size(); ++fi) {
    const auto &f = m.faces[fi];
    for (size_t i = 1; i + 1 < f.size(); ++i) {
      const int idx[3] = {f[0], f[i], f[i + 1]};
      for (int k : idx) {
        glNormal3fv(o.smooth ? o.vnormals[k].data() : o.fnormals[fi].data());
        glVertex3fv(m.verts[k].data());
      }
    }
  }
  glEnd();
  glEndList();

  o.wire_list = glGenLists(1);
  glNewList(o.wire_list, GL_COMPILE);
  for (const auto &f : m.faces) {
    glBegin(GL_LINE_LOOP);
    for (int vid : f) {
      glVertex3fv(m.verts[vid].data());
    }
    glEnd();
  }
  glEndList();
}

static void draw_grid()
{
  glLineWidth(1.0f);
  glBegin(GL_LINES);
  for (int i = -10; i <= 10; ++i) {
    const float t = i * 0.5f;
    const float c = (i == 0) ? 0.0f : ((i % 2 == 0) ? 0.33f : 0.27f);
    if (i == 0) {
      continue;
    }
    glColor3f(c, c, c + 0.02f);
    glVertex3f(t, -5.0f, 0.0f);
    glVertex3f(t, 5.0f, 0.0f);
    glVertex3f(-5.0f, t, 0.0f);
    glVertex3f(5.0f, t, 0.0f);
  }
  glEnd();
  glLineWidth(2.0f);
  glBegin(GL_LINES);
  glColor3f(0.85f, 0.25f, 0.25f);
  glVertex3f(-5.0f, 0.0f, 0.0f);
  glVertex3f(5.0f, 0.0f, 0.0f);
  glColor3f(0.35f, 0.78f, 0.30f);
  glVertex3f(0.0f, -5.0f, 0.0f);
  glVertex3f(0.0f, 5.0f, 0.0f);
  glColor3f(0.30f, 0.45f, 0.90f);
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(0.0f, 0.0f, 1.0f);
  glEnd();
  glLineWidth(1.0f);
}

static void draw_sketch()
{
  const SketchDocument &doc = g_sketch;
  if (doc.points.empty()) {
    return;
  }
  glDisable(GL_DEPTH_TEST);
  glLineWidth(2.0f);
  glColor3f(1.0f, 0.82f, 0.25f);
  for (const auto &e : doc.entities) {
    if (e.type == SketchEntityType::LINE) {
      const auto &a = doc.points[e.p0];
      const auto &b = doc.points[e.p1];
      glBegin(GL_LINES);
      glVertex3fv(plane_to_world(doc.plane, a.u, a.v, 0.0).data());
      glVertex3fv(plane_to_world(doc.plane, b.u, b.v, 0.0).data());
      glEnd();
    }
    else {
      const auto &c = doc.points[e.p0];
      glBegin(GL_LINE_LOOP);
      for (int i = 0; i < 64; ++i) {
        const double ang = 2.0 * 3.14159265358979 * i / 64.0;
        glVertex3fv(plane_to_world(doc.plane,
                                   c.u + e.radius * std::cos(ang),
                                   c.v + e.radius * std::sin(ang),
                                   0.0)
                        .data());
      }
      glEnd();
    }
  }
  glPointSize(6.0f);
  glColor3f(1.0f, 1.0f, 1.0f);
  glBegin(GL_POINTS);
  for (const auto &p : doc.points) {
    glVertex3fv(plane_to_world(doc.plane, p.u, p.v, 0.0).data());
  }
  glEnd();
  glLineWidth(1.0f);
  glEnable(GL_DEPTH_TEST);
}

static void render_scene()
{
  if (!g_gl || !g_glrc) {
    return;
  }
  wglMakeCurrent(g_hdc, g_glrc);
  if (!g_lists_to_delete.empty()) {
    for (GLuint l : g_lists_to_delete) {
      glDeleteLists(l, 1);
    }
    g_lists_to_delete.clear();
  }

  RECT rc;
  GetClientRect(g_gl, &rc);
  const int w = std::max<int>(1, rc.right);
  const int h = std::max<int>(1, rc.bottom);
  glViewport(0, 0, w, h);

  /* Gradient background */
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glBegin(GL_QUADS);
  glColor3f(0.11f, 0.12f, 0.14f);
  glVertex2f(-1, -1);
  glVertex2f(1, -1);
  glColor3f(0.24f, 0.26f, 0.30f);
  glVertex2f(1, 1);
  glVertex2f(-1, 1);
  glEnd();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  const float aspect = float(w) / float(h);
  const float znear = 0.01f;
  const float zfar = 500.0f;
  const float top = znear * std::tan(0.5f * 40.0f * 3.14159265f / 180.0f);
  glFrustum(-top * aspect, top * aspect, -top, top, znear, zfar);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  /* Headlight in eye space */
  const GLfloat head_pos[4] = {0.3f, 0.5f, 1.0f, 0.0f};
  const GLfloat head_diff[4] = {0.78f, 0.78f, 0.76f, 1.0f};
  const GLfloat head_spec[4] = {0.35f, 0.35f, 0.35f, 1.0f};
  glLightfv(GL_LIGHT0, GL_POSITION, head_pos);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, head_diff);
  glLightfv(GL_LIGHT0, GL_SPECULAR, head_spec);

  const float cp = std::cos(g_pitch);
  const float eye[3] = {
      g_target[0] + g_dist * cp * std::cos(g_yaw),
      g_target[1] + g_dist * cp * std::sin(g_yaw),
      g_target[2] + g_dist * std::sin(g_pitch),
  };
  look_at(eye, g_target);

  /* Rim/fill light in world space */
  const GLfloat fill_pos[4] = {-0.6f, 0.8f, 0.4f, 0.0f};
  const GLfloat fill_diff[4] = {0.30f, 0.34f, 0.42f, 1.0f};
  glLightfv(GL_LIGHT1, GL_POSITION, fill_pos);
  glLightfv(GL_LIGHT1, GL_DIFFUSE, fill_diff);
  const GLfloat ambient[4] = {0.18f, 0.18f, 0.20f, 1.0f};
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);
  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  draw_grid();

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHT1);
  glEnable(GL_NORMALIZE);
  glEnable(GL_COLOR_MATERIAL);
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  const GLfloat mat_spec[4] = {0.25f, 0.25f, 0.25f, 1.0f};
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_spec);
  glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 40.0f);
  glShadeModel(GL_SMOOTH);

  glEnable(GL_POLYGON_OFFSET_FILL);
  glPolygonOffset(1.0f, 1.0f);
  for (auto &o : g_scene) {
    if (!o.list) {
      build_lists(o);
    }
    glColor3fv(o.color);
    glCallList(o.list);
  }
  glDisable(GL_POLYGON_OFFSET_FILL);
  glDisable(GL_LIGHTING);

  if (g_wire) {
    glColor4f(0.05f, 0.05f, 0.06f, 1.0f);
    for (auto &o : g_scene) {
      glCallList(o.wire_list);
    }
  }

  draw_sketch();
  glFinish();
}

static bool save_screenshot_bmp(const std::wstring &path)
{
  RECT rc;
  GetClientRect(g_gl, &rc);
  const int w = std::max<int>(1, rc.right);
  const int h = std::max<int>(1, rc.bottom);
  const int row = ((w * 3) + 3) & ~3;
  std::vector<unsigned char> pixels(size_t(row) * h);
  glPixelStorei(GL_PACK_ALIGNMENT, 4);
  glReadBuffer(GL_BACK);
  glReadPixels(0, 0, w, h, GL_BGR_EXT, GL_UNSIGNED_BYTE, pixels.data());

  BITMAPFILEHEADER bf{};
  BITMAPINFOHEADER bi{};
  bi.biSize = sizeof(bi);
  bi.biWidth = w;
  bi.biHeight = h;
  bi.biPlanes = 1;
  bi.biBitCount = 24;
  bi.biCompression = BI_RGB;
  bi.biSizeImage = DWORD(pixels.size());
  bf.bfType = 0x4D42;
  bf.bfOffBits = sizeof(bf) + sizeof(bi);
  bf.bfSize = bf.bfOffBits + bi.biSizeImage;

  std::ofstream f(path.c_str(), std::ios::binary);
  if (!f) {
    return false;
  }
  f.write(reinterpret_cast<const char *>(&bf), sizeof(bf));
  f.write(reinterpret_cast<const char *>(&bi), sizeof(bi));
  f.write(reinterpret_cast<const char *>(pixels.data()), std::streamsize(pixels.size()));
  return true;
}

/* ------------------------------------------------------------------------- */
/* Core command helpers (shared by UI + AI bridge)                           */
/* ------------------------------------------------------------------------- */

static std::string mesh_stats_json(const SceneObject &o)
{
  std::ostringstream oss;
  oss << "{\"ok\":true,\"name\":\"" << o.name << "\",\"verts\":" << o.mesh.verts.size()
      << ",\"faces\":" << o.mesh.faces.size() << "}";
  return oss.str();
}

static SceneObject *do_extrude(float depth, bool symmetric, const std::string &name)
{
  g_sketch.solve();
  RawMeshData raw = extrude_sketch(g_sketch, depth, symmetric);
  if (raw.verts.empty()) {
    return nullptr;
  }
  return &add_object(name, std::move(raw), false);
}

static SceneObject *do_revolve(float angle, int segs, const std::string &name)
{
  g_sketch.solve();
  RawMeshData raw = revolve_sketch(g_sketch, angle, segs);
  if (raw.verts.empty()) {
    return nullptr;
  }
  return &add_object(name, std::move(raw), true);
}

static SceneObject &do_humanoid(const HumanoidParams &hp, const std::string &name)
{
  return add_object(name, generate_humanoid_sdf(hp), true);
}

static SceneObject &do_custom_sdf(const std::string &spec, int res, const std::string &name)
{
  return add_object(name, generate_custom_sdf_from_spec(spec, res), true);
}

/* ------------------------------------------------------------------------- */
/* Minimal JSON helpers                                                      */
/* ------------------------------------------------------------------------- */

static size_t json_value_pos(const std::string &json, const std::string &key)
{
  const std::string pattern = "\"" + key + "\"";
  size_t pos = json.find(pattern);
  if (pos == std::string::npos) {
    return std::string::npos;
  }
  pos = json.find(':', pos + pattern.size());
  if (pos == std::string::npos) {
    return std::string::npos;
  }
  ++pos;
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
    ++pos;
  }
  return pos;
}

static std::string json_str(const std::string &json, const std::string &key, const std::string &def = "")
{
  size_t pos = json_value_pos(json, key);
  if (pos == std::string::npos || pos >= json.size() || json[pos] != '"') {
    return def;
  }
  ++pos;
  std::string out;
  while (pos < json.size() && json[pos] != '"') {
    if (json[pos] == '\\' && pos + 1 < json.size()) {
      ++pos;
      switch (json[pos]) {
        case 'n':
          out.push_back('\n');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'r':
          break;
        default:
          out.push_back(json[pos]);
          break;
      }
    }
    else {
      out.push_back(json[pos]);
    }
    ++pos;
  }
  return out;
}

static double json_num(const std::string &json, const std::string &key, double def)
{
  const size_t pos = json_value_pos(json, key);
  if (pos == std::string::npos || pos >= json.size()) {
    return def;
  }
  char *end = nullptr;
  const double v = std::strtod(json.c_str() + pos, &end);
  return (end == json.c_str() + pos) ? def : v;
}

static bool json_bool(const std::string &json, const std::string &key, bool def)
{
  const size_t pos = json_value_pos(json, key);
  if (pos == std::string::npos) {
    return def;
  }
  if (json.compare(pos, 4, "true") == 0) {
    return true;
  }
  if (json.compare(pos, 5, "false") == 0) {
    return false;
  }
  return def;
}

static std::string json_escape(const std::string &s)
{
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
      out.push_back(c);
    }
    else if (c == '\n') {
      out += "\\n";
    }
    else {
      out.push_back(c);
    }
  }
  return out;
}

static SketchPlane parse_plane(const std::string &s)
{
  if (s == "XZ") {
    return SketchPlane::XZ;
  }
  if (s == "YZ") {
    return SketchPlane::YZ;
  }
  return SketchPlane::XY;
}

static void set_status(const std::wstring &text);

static std::string dispatch_command(const std::string &payload)
{
  const std::string cmd = json_str(payload, "cmd");

  if (cmd == "ping") {
    return "{\"ok\":true,\"service\":\"EX-CAD Standalone\",\"objects\":" +
           std::to_string(g_scene.size()) + "}";
  }
  if (cmd == "sketch_new") {
    const std::string plane = json_str(payload, "plane", "XY");
    g_sketch.reset(parse_plane(plane));
    redraw();
    return "{\"ok\":true,\"plane\":\"" + json_escape(plane) + "\"}";
  }
  if (cmd == "sketch_add_rect") {
    g_sketch.add_rect(json_num(payload, "u0", -0.5),
                      json_num(payload, "v0", -0.5),
                      json_num(payload, "width", 1.0),
                      json_num(payload, "height", 1.0));
    redraw();
    return "{\"ok\":true,\"points\":" + std::to_string(g_sketch.points.size()) + "}";
  }
  if (cmd == "sketch_add_circle") {
    g_sketch.add_circle(json_num(payload, "cu", 0.0),
                        json_num(payload, "cv", 0.0),
                        json_num(payload, "radius", 0.5));
    redraw();
    return "{\"ok\":true,\"entities\":" + std::to_string(g_sketch.entities.size()) + "}";
  }
  if (cmd == "sketch_add_line") {
    g_sketch.add_line(json_num(payload, "u0", 0.0),
                      json_num(payload, "v0", 0.0),
                      json_num(payload, "u1", 1.0),
                      json_num(payload, "v1", 0.0));
    redraw();
    return "{\"ok\":true,\"entities\":" + std::to_string(g_sketch.entities.size()) + "}";
  }
  if (cmd == "sketch_add_constraint") {
    static const std::map<std::string, SketchConstraintType> types = {
        {"COINCIDENT", SketchConstraintType::COINCIDENT},
        {"HORIZONTAL", SketchConstraintType::HORIZONTAL},
        {"VERTICAL", SketchConstraintType::VERTICAL},
        {"DISTANCE", SketchConstraintType::DISTANCE},
        {"RADIUS", SketchConstraintType::RADIUS},
        {"PARALLEL", SketchConstraintType::PARALLEL},
        {"PERPENDICULAR", SketchConstraintType::PERPENDICULAR},
        {"EQUAL_LENGTH", SketchConstraintType::EQUAL_LENGTH},
        {"FIXED", SketchConstraintType::FIXED},
    };
    const auto it = types.find(json_str(payload, "type", "DISTANCE"));
    if (it == types.end()) {
      return "{\"ok\":false,\"error\":\"Unknown constraint type\"}";
    }
    const int id = g_sketch.add_constraint(it->second,
                                           int(json_num(payload, "a", 0)),
                                           int(json_num(payload, "b", -1)),
                                           int(json_num(payload, "c", -1)),
                                           int(json_num(payload, "d", -1)),
                                           json_num(payload, "value", 0.0),
                                           json_num(payload, "value2", 0.0));
    return "{\"ok\":true,\"constraint_id\":" + std::to_string(id) + "}";
  }
  if (cmd == "sketch_solve") {
    const SolveResult r = g_sketch.solve();
    redraw();
    std::ostringstream oss;
    oss << "{\"ok\":true,\"converged\":" << (r.converged ? "true" : "false")
        << ",\"iterations\":" << r.iterations << ",\"max_residual\":" << r.max_residual << "}";
    return oss.str();
  }
  if (cmd == "sketch_get") {
    std::ostringstream oss;
    oss << "{\"ok\":true,\"points\":[";
    for (size_t i = 0; i < g_sketch.points.size(); ++i) {
      oss << (i ? "," : "") << "[" << g_sketch.points[i].u << "," << g_sketch.points[i].v << "]";
    }
    oss << "],\"entities\":" << g_sketch.entities.size()
        << ",\"constraints\":" << g_sketch.constraints.size() << "}";
    return oss.str();
  }
  if (cmd == "sketch_extrude") {
    SceneObject *o = do_extrude(float(json_num(payload, "depth", 1.0)),
                                json_bool(payload, "symmetric", false),
                                json_str(payload, "name", "Extrude"));
    return o ? mesh_stats_json(*o) : "{\"ok\":false,\"error\":\"No closed sketch profile\"}";
  }
  if (cmd == "sketch_revolve") {
    SceneObject *o = do_revolve(float(json_num(payload, "angle_deg", 360.0)),
                                int(json_num(payload, "segments", 64)),
                                json_str(payload, "name", "Revolve"));
    return o ? mesh_stats_json(*o) : "{\"ok\":false,\"error\":\"No closed sketch profile\"}";
  }
  if (cmd == "sdf_humanoid") {
    HumanoidParams hp;
    hp.height = float(json_num(payload, "height", 1.75));
    hp.muscularity = float(json_num(payload, "muscularity", 0.5));
    hp.shoulder_ratio = float(json_num(payload, "shoulder_ratio", 1.0));
    hp.hip_ratio = float(json_num(payload, "hip_ratio", 1.0));
    hp.resolution = int(json_num(payload, "resolution", 96));
    return mesh_stats_json(do_humanoid(hp, json_str(payload, "name", "Humanoid")));
  }
  if (cmd == "sdf_custom") {
    return mesh_stats_json(do_custom_sdf(json_str(payload, "spec"),
                                         int(json_num(payload, "resolution", 80)),
                                         json_str(payload, "name", "SDF")));
  }
  if (cmd == "list") {
    std::ostringstream oss;
    oss << "{\"ok\":true,\"objects\":[";
    for (size_t i = 0; i < g_scene.size(); ++i) {
      oss << (i ? "," : "") << "{\"name\":\"" << json_escape(g_scene[i].name)
          << "\",\"verts\":" << g_scene[i].mesh.verts.size()
          << ",\"faces\":" << g_scene[i].mesh.faces.size() << "}";
    }
    oss << "]}";
    return oss.str();
  }
  if (cmd == "clear") {
    clear_scene();
    return "{\"ok\":true}";
  }
  if (cmd == "delete_last") {
    return delete_last() ? "{\"ok\":true}" : "{\"ok\":false,\"error\":\"Scene is empty\"}";
  }
  if (cmd == "frame_all") {
    frame_all();
    return "{\"ok\":true}";
  }
  if (cmd == "export_obj" || cmd == "export_stl") {
    const std::string path = json_str(payload, "path");
    if (path.empty()) {
      return "{\"ok\":false,\"error\":\"Missing path\"}";
    }
    const std::wstring wpath = widen(path);
    const bool ok = (cmd == "export_obj") ? export_obj(wpath) : export_stl(wpath);
    return ok ? "{\"ok\":true}" : "{\"ok\":false,\"error\":\"Write failed\"}";
  }
  if (cmd == "screenshot") {
    const std::string path = json_str(payload, "path");
    if (path.empty() || !g_gl) {
      return "{\"ok\":false,\"error\":\"Missing path or viewport\"}";
    }
    render_scene();
    const bool ok = save_screenshot_bmp(widen(path));
    SwapBuffers(g_hdc);
    return ok ? "{\"ok\":true}" : "{\"ok\":false,\"error\":\"Write failed\"}";
  }
  return "{\"ok\":false,\"error\":\"Unknown command: " + json_escape(cmd) + "\"}";
}

/* ------------------------------------------------------------------------- */
/* AI Direct Bridge (TCP JSON lines on 127.0.0.1)                            */
/* ------------------------------------------------------------------------- */

#define WM_EXCAD_BRIDGE (WM_APP + 1)
#define WM_EXCAD_SCREENSHOT (WM_APP + 2)

struct PendingRequest {
  std::string payload;
  std::shared_ptr<std::promise<std::string>> promise;
};

static std::mutex g_queue_mutex;
static std::deque<PendingRequest> g_queue;
static std::atomic<bool> g_bridge_running{false};
static std::atomic<bool> g_bridge_done{true};
static std::atomic<bool> g_bridge_failed{false};
static std::thread g_bridge_thread;
static int g_bridge_port = 9877;

static void process_bridge_queue()
{
  for (;;) {
    PendingRequest req;
    {
      std::lock_guard<std::mutex> lock(g_queue_mutex);
      if (g_queue.empty()) {
        return;
      }
      req = std::move(g_queue.front());
      g_queue.pop_front();
    }
    std::string reply;
    try {
      reply = dispatch_command(req.payload);
    }
    catch (const std::exception &e) {
      reply = std::string("{\"ok\":false,\"error\":\"") + json_escape(e.what()) + "\"}";
    }
    req.promise->set_value(reply);
    set_status(L"AI bridge: " + widen(json_str(req.payload, "cmd", "?")) + L" -> done");
  }
}

static std::string submit_and_wait(const std::string &payload)
{
  auto prom = std::make_shared<std::promise<std::string>>();
  auto fut = prom->get_future();
  {
    std::lock_guard<std::mutex> lock(g_queue_mutex);
    g_queue.push_back(PendingRequest{payload, prom});
  }
  PostMessageW(g_main, WM_EXCAD_BRIDGE, 0, 0);
  if (fut.wait_for(std::chrono::seconds(120)) == std::future_status::ready) {
    return fut.get();
  }
  return "{\"ok\":false,\"error\":\"Timeout\"}";
}

static void handle_client(SOCKET client)
{
  std::string buf;
  char chunk[8192];
  while (g_bridge_running.load()) {
    size_t nl;
    while ((nl = buf.find('\n')) != std::string::npos) {
      std::string line = buf.substr(0, nl);
      buf.erase(0, nl + 1);
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      if (line.empty()) {
        continue;
      }
      const std::string reply = submit_and_wait(line) + "\n";
      send(client, reply.data(), int(reply.size()), 0);
    }
    fd_set rf;
    FD_ZERO(&rf);
    FD_SET(client, &rf);
    timeval tv{0, 100000};
    const int r = select(0, &rf, nullptr, nullptr, &tv);
    if (r < 0) {
      return;
    }
    if (r == 0) {
      continue;
    }
    const int n = recv(client, chunk, sizeof(chunk), 0);
    if (n <= 0) {
      if (!buf.empty()) {
        const std::string reply = submit_and_wait(buf) + "\n";
        send(client, reply.data(), int(reply.size()), 0);
      }
      return;
    }
    buf.append(chunk, size_t(n));
  }
}

static void bridge_loop(int port)
{
  SOCKET ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (ls == INVALID_SOCKET) {
    g_bridge_failed = true;
    g_bridge_running = false;
    g_bridge_done = true;
    return;
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(u_short(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(ls, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 || listen(ls, 8) != 0) {
    closesocket(ls);
    g_bridge_failed = true;
    g_bridge_running = false;
    g_bridge_done = true;
    return;
  }
  while (g_bridge_running.load()) {
    fd_set rf;
    FD_ZERO(&rf);
    FD_SET(ls, &rf);
    timeval tv{0, 100000};
    if (select(0, &rf, nullptr, nullptr, &tv) <= 0) {
      continue;
    }
    SOCKET client = accept(ls, nullptr, nullptr);
    if (client == INVALID_SOCKET) {
      continue;
    }
    handle_client(client);
    closesocket(client);
  }
  closesocket(ls);
  g_bridge_done = true;
}

static void update_bridge_label();

static void bridge_start(int port)
{
  if (g_bridge_running.load()) {
    return;
  }
  if (g_bridge_thread.joinable()) {
    g_bridge_thread.join();
  }
  g_bridge_port = port;
  g_bridge_failed = false;
  g_bridge_done = false;
  g_bridge_running = true;
  g_bridge_thread = std::thread(bridge_loop, port);
  Sleep(150);
  update_bridge_label();
  update_title();
}

static void bridge_stop()
{
  g_bridge_running = false;
  while (!g_bridge_done.load()) {
    process_bridge_queue();
    Sleep(10);
  }
  if (g_bridge_thread.joinable()) {
    g_bridge_thread.join();
  }
  update_bridge_label();
  update_title();
}

/* ------------------------------------------------------------------------- */
/* UI                                                                        */
/* ------------------------------------------------------------------------- */

enum : int {
  ID_PLANE = 100,
  ID_NEW_SKETCH,
  ID_R_U0,
  ID_R_V0,
  ID_R_W,
  ID_R_H,
  ID_ADD_RECT,
  ID_C_U,
  ID_C_V,
  ID_C_R,
  ID_ADD_CIRCLE,
  ID_L_U0,
  ID_L_V0,
  ID_L_U1,
  ID_L_V1,
  ID_ADD_LINE,
  ID_SOLVE,
  ID_EX_DEPTH,
  ID_EX_SYM,
  ID_EXTRUDE,
  ID_RV_ANGLE,
  ID_RV_SEGS,
  ID_REVOLVE,
  ID_H_HEIGHT,
  ID_H_MUSC,
  ID_H_SH,
  ID_H_HIP,
  ID_H_RES,
  ID_HUMANOID,
  ID_SPEC,
  ID_SPEC_RES,
  ID_CUSTOM,
  ID_EXPORT_OBJ,
  ID_EXPORT_STL,
  ID_DELETE_LAST,
  ID_CLEAR,
  ID_FRAME,
  ID_WIRE,
  ID_BRIDGE_PORT,
  ID_BRIDGE_START,
  ID_BRIDGE_STOP,
  ID_BRIDGE_STATUS,
  ID_STATUS,
};

constexpr int PANEL_W = 282;

static HWND mk(DWORD ex, const wchar_t *cls, const wchar_t *text, DWORD style, int x, int y, int w, int h, int id, bool bold = false)
{
  HWND hw = CreateWindowExW(ex,
                            cls,
                            text,
                            WS_CHILD | WS_VISIBLE | style,
                            x,
                            y,
                            w,
                            h,
                            g_main,
                            reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                            g_inst,
                            nullptr);
  SendMessageW(hw, WM_SETFONT, reinterpret_cast<WPARAM>(bold ? g_font_bold : g_font), TRUE);
  if (id > 0) {
    g_ctrl[id] = hw;
  }
  return hw;
}

static void label(const wchar_t *text, int x, int y, int w, bool bold = false)
{
  mk(0, L"STATIC", text, SS_LEFT, x, y, w, bold ? 18 : 15, 0, bold);
}

static void edit(int id, const wchar_t *text, int x, int y, int w)
{
  mk(WS_EX_CLIENTEDGE, L"EDIT", text, ES_AUTOHSCROLL | WS_TABSTOP, x, y, w, 22, id);
}

static void button(int id, const wchar_t *text, int x, int y, int w, int h = 25)
{
  mk(0, L"BUTTON", text, BS_PUSHBUTTON | WS_TABSTOP, x, y, w, h, id);
}

static void checkbox(int id, const wchar_t *text, int x, int y, int w)
{
  mk(0, L"BUTTON", text, BS_AUTOCHECKBOX | WS_TABSTOP, x, y, w, 22, id);
}

static double get_num(int id, double def)
{
  wchar_t buf[64] = {0};
  GetWindowTextW(g_ctrl[id], buf, 63);
  wchar_t *end = nullptr;
  const double v = std::wcstod(buf, &end);
  return (end == buf) ? def : v;
}

static bool get_check(int id)
{
  return SendMessageW(g_ctrl[id], BM_GETCHECK, 0, 0) == BST_CHECKED;
}

static std::string get_text(int id)
{
  const int len = GetWindowTextLengthW(g_ctrl[id]);
  std::wstring w(size_t(len) + 1, L'\0');
  GetWindowTextW(g_ctrl[id], w.data(), len + 1);
  w.resize(size_t(len));
  return narrow(w);
}

static void set_status(const std::wstring &text)
{
  if (g_ctrl.count(ID_STATUS)) {
    SetWindowTextW(g_ctrl[ID_STATUS], text.c_str());
  }
}

static void update_bridge_label()
{
  if (!g_ctrl.count(ID_BRIDGE_STATUS)) {
    return;
  }
  std::wstring text;
  if (g_bridge_running.load()) {
    text = L"Bridge: LISTENING on 127.0.0.1:" + std::to_wstring(g_bridge_port);
  }
  else if (g_bridge_failed.load()) {
    text = L"Bridge: FAILED (port in use?)";
  }
  else {
    text = L"Bridge: stopped";
  }
  SetWindowTextW(g_ctrl[ID_BRIDGE_STATUS], text.c_str());
}

static void update_title()
{
  if (!g_main) {
    return;
  }
  size_t faces = 0;
  for (const auto &o : g_scene) {
    faces += o.mesh.faces.size();
  }
  std::wstring t = L"EX-CAD Standalone  |  objects: " + std::to_wstring(g_scene.size()) +
                   L"  faces: " + std::to_wstring(faces) + L"  |  AI bridge: " +
                   (g_bridge_running.load() ? (L"ON :" + std::to_wstring(g_bridge_port)) : L"off");
  SetWindowTextW(g_main, t.c_str());
}

static void build_panel()
{
  int y = 8;
  const int x = 10;
  label(L"1. Parametric Sketch (Fusion-style)", x, y, 262, true);
  y += 24;
  HWND combo = mk(0, L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP, x, y, 70, 120, ID_PLANE);
  SendMessageW(combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"XY"));
  SendMessageW(combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"XZ"));
  SendMessageW(combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"YZ"));
  SendMessageW(combo, CB_SETCURSEL, 0, 0);
  button(ID_NEW_SKETCH, L"New Sketch", x + 76, y - 1, 186);
  y += 32;

  label(L"Rectangle   u0 / v0 / width / height", x, y, 262);
  y += 16;
  edit(ID_R_U0, L"-0.5", x, y, 60);
  edit(ID_R_V0, L"-0.3", x + 67, y, 60);
  edit(ID_R_W, L"1.0", x + 134, y, 60);
  edit(ID_R_H, L"0.6", x + 201, y, 61);
  y += 26;
  button(ID_ADD_RECT, L"Add Rectangle", x, y, 262);
  y += 32;

  label(L"Circle   cu / cv / radius", x, y, 262);
  y += 16;
  edit(ID_C_U, L"0.0", x, y, 60);
  edit(ID_C_V, L"0.0", x + 67, y, 60);
  edit(ID_C_R, L"0.25", x + 134, y, 60);
  button(ID_ADD_CIRCLE, L"Add Circle", x + 200, y - 1, 62);
  y += 30;

  label(L"Line   u0 / v0 / u1 / v1", x, y, 262);
  y += 16;
  edit(ID_L_U0, L"0.0", x, y, 60);
  edit(ID_L_V0, L"0.0", x + 67, y, 60);
  edit(ID_L_U1, L"1.0", x + 134, y, 60);
  edit(ID_L_V1, L"0.0", x + 201, y, 61);
  y += 26;
  button(ID_ADD_LINE, L"Add Line", x, y, 128);
  button(ID_SOLVE, L"Solve Constraints", x + 134, y, 128);
  y += 34;

  label(L"Extrude depth   |   Revolve angle / segments", x, y, 262);
  y += 16;
  edit(ID_EX_DEPTH, L"0.4", x, y, 60);
  checkbox(ID_EX_SYM, L"Symmetric", x + 66, y, 84);
  button(ID_EXTRUDE, L"Extrude", x + 152, y - 1, 110);
  y += 28;
  edit(ID_RV_ANGLE, L"360", x, y, 60);
  edit(ID_RV_SEGS, L"64", x + 67, y, 60);
  button(ID_REVOLVE, L"Revolve (V-axis)", x + 134, y - 1, 128);
  y += 38;

  label(L"2. High-Precision SDF Mesh (C++ Marching Cubes)", x, y, 262, true);
  y += 24;
  label(L"height / muscle / shoulder / hip / resolution", x, y, 262);
  y += 16;
  edit(ID_H_HEIGHT, L"1.75", x, y, 48);
  edit(ID_H_MUSC, L"0.5", x + 53, y, 48);
  edit(ID_H_SH, L"1.0", x + 106, y, 48);
  edit(ID_H_HIP, L"1.0", x + 159, y, 48);
  edit(ID_H_RES, L"110", x + 212, y, 50);
  y += 26;
  button(ID_HUMANOID, L"Generate Anatomical Humanoid", x, y, 262);
  y += 32;
  label(L"Custom SDF spec (sphere/capsule/ellipsoid/box/sub_sphere)", x, y, 262);
  y += 16;
  mk(WS_EX_CLIENTEDGE,
     L"EDIT",
     L"sphere 0 0 0.6 0.35 0.05\r\ncapsule 0 0 0.6 0 0 1.2 0.18 0.1 0.12\r\nsub_sphere 0 -0.3 0.6 0.18 0.04",
     ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN | WS_VSCROLL | WS_TABSTOP,
     x,
     y,
     262,
     78,
     ID_SPEC);
  y += 84;
  edit(ID_SPEC_RES, L"90", x, y, 60);
  button(ID_CUSTOM, L"Generate Custom SDF", x + 67, y - 1, 195);
  y += 36;

  label(L"3. Scene", x, y, 262, true);
  y += 24;
  button(ID_EXPORT_OBJ, L"Export OBJ", x, y, 128);
  button(ID_EXPORT_STL, L"Export STL", x + 134, y, 128);
  y += 30;
  button(ID_DELETE_LAST, L"Delete Last", x, y, 84);
  button(ID_CLEAR, L"Clear", x + 89, y, 84);
  button(ID_FRAME, L"Frame All", x + 178, y, 84);
  y += 30;
  checkbox(ID_WIRE, L"Wireframe overlay", x, y, 200);
  y += 32;

  label(L"4. AI Direct Bridge (TCP JSON lines)", x, y, 262, true);
  y += 24;
  edit(ID_BRIDGE_PORT, L"9877", x, y, 60);
  button(ID_BRIDGE_START, L"Start", x + 67, y - 1, 95);
  button(ID_BRIDGE_STOP, L"Stop", x + 167, y - 1, 95);
  y += 28;
  mk(0, L"STATIC", L"Bridge: stopped", SS_LEFT, x, y, 262, 16, ID_BRIDGE_STATUS);
  y += 22;
  mk(0,
     L"STATIC",
     L"Ready. Mouse: L-drag orbit, R/M-drag pan, wheel zoom.",
     SS_LEFT,
     x,
     y,
     262,
     48,
     ID_STATUS);
}

static std::wstring save_dialog(const wchar_t *filter, const wchar_t *def_ext, const wchar_t *def_name)
{
  wchar_t file[MAX_PATH];
  wcsncpy(file, def_name, MAX_PATH - 1);
  file[MAX_PATH - 1] = 0;
  OPENFILENAMEW ofn{};
  ofn.lStructSize = sizeof(ofn);
  ofn.hwndOwner = g_main;
  ofn.lpstrFilter = filter;
  ofn.lpstrFile = file;
  ofn.nMaxFile = MAX_PATH;
  ofn.lpstrDefExt = def_ext;
  ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;
  if (GetSaveFileNameW(&ofn)) {
    return file;
  }
  return {};
}

static void report_object(const wchar_t *what, const SceneObject *o)
{
  if (!o) {
    set_status(std::wstring(what) + L": no closed profile in sketch.");
    return;
  }
  set_status(std::wstring(what) + L": " + widen(o->name) + L"  (" +
             std::to_wstring(o->mesh.verts.size()) + L" verts, " +
             std::to_wstring(o->mesh.faces.size()) + L" faces)");
}

static void on_command(int id)
{
  HCURSOR old_cursor = SetCursor(LoadCursor(nullptr, IDC_WAIT));
  switch (id) {
    case ID_NEW_SKETCH: {
      const int sel = int(SendMessageW(g_ctrl[ID_PLANE], CB_GETCURSEL, 0, 0));
      g_sketch.reset(static_cast<SketchPlane>(std::clamp(sel, 0, 2)));
      set_status(L"New sketch started.");
      redraw();
      break;
    }
    case ID_ADD_RECT:
      g_sketch.add_rect(get_num(ID_R_U0, -0.5), get_num(ID_R_V0, -0.3), get_num(ID_R_W, 1.0), get_num(ID_R_H, 0.6));
      set_status(L"Rectangle added (H/V + width/height constraints).");
      redraw();
      break;
    case ID_ADD_CIRCLE:
      g_sketch.add_circle(get_num(ID_C_U, 0.0), get_num(ID_C_V, 0.0), get_num(ID_C_R, 0.25));
      set_status(L"Circle added (radius constraint).");
      redraw();
      break;
    case ID_ADD_LINE:
      g_sketch.add_line(get_num(ID_L_U0, 0.0), get_num(ID_L_V0, 0.0), get_num(ID_L_U1, 1.0), get_num(ID_L_V1, 0.0));
      set_status(L"Line added.");
      redraw();
      break;
    case ID_SOLVE: {
      const SolveResult r = g_sketch.solve();
      wchar_t buf[160];
      std::swprintf(buf, 160, L"Solve: %ls, %d iters, residual %.2e", r.converged ? L"converged" : L"not converged", r.iterations, r.max_residual);
      set_status(buf);
      redraw();
      break;
    }
    case ID_EXTRUDE:
      report_object(L"Extrude", do_extrude(float(get_num(ID_EX_DEPTH, 0.4)), get_check(ID_EX_SYM), "Extrude"));
      break;
    case ID_REVOLVE:
      report_object(L"Revolve", do_revolve(float(get_num(ID_RV_ANGLE, 360.0)), int(get_num(ID_RV_SEGS, 64)), "Revolve"));
      break;
    case ID_HUMANOID: {
      HumanoidParams hp;
      hp.height = float(get_num(ID_H_HEIGHT, 1.75));
      hp.muscularity = float(get_num(ID_H_MUSC, 0.5));
      hp.shoulder_ratio = float(get_num(ID_H_SH, 1.0));
      hp.hip_ratio = float(get_num(ID_H_HIP, 1.0));
      hp.resolution = int(get_num(ID_H_RES, 110));
      report_object(L"Humanoid", &do_humanoid(hp, "Humanoid"));
      break;
    }
    case ID_CUSTOM:
      report_object(L"Custom SDF", &do_custom_sdf(get_text(ID_SPEC), int(get_num(ID_SPEC_RES, 90)), "SDF"));
      break;
    case ID_EXPORT_OBJ: {
      const std::wstring p = save_dialog(L"Wavefront OBJ\0*.obj\0", L"obj", L"excad_scene.obj");
      if (!p.empty()) {
        set_status(export_obj(p) ? L"Exported OBJ: " + p : L"OBJ export failed.");
      }
      break;
    }
    case ID_EXPORT_STL: {
      const std::wstring p = save_dialog(L"STL (binary)\0*.stl\0", L"stl", L"excad_scene.stl");
      if (!p.empty()) {
        set_status(export_stl(p) ? L"Exported STL: " + p : L"STL export failed.");
      }
      break;
    }
    case ID_DELETE_LAST:
      set_status(delete_last() ? L"Deleted last object." : L"Scene is empty.");
      break;
    case ID_CLEAR:
      clear_scene();
      set_status(L"Scene cleared.");
      break;
    case ID_FRAME:
      frame_all();
      break;
    case ID_WIRE:
      g_wire = get_check(ID_WIRE);
      redraw();
      break;
    case ID_BRIDGE_START:
      bridge_start(int(get_num(ID_BRIDGE_PORT, 9877)));
      set_status(g_bridge_running.load() ? L"AI bridge started. Send JSON lines, e.g. {\"cmd\":\"ping\"}" : L"AI bridge failed to start.");
      break;
    case ID_BRIDGE_STOP:
      bridge_stop();
      set_status(L"AI bridge stopped.");
      break;
    default:
      break;
  }
  SetCursor(old_cursor);
}

static LRESULT CALLBACK gl_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
  switch (msg) {
    case WM_CREATE: {
      g_hdc = GetDC(hwnd);
      PIXELFORMATDESCRIPTOR pfd{};
      pfd.nSize = sizeof(pfd);
      pfd.nVersion = 1;
      pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
      pfd.iPixelType = PFD_TYPE_RGBA;
      pfd.cColorBits = 32;
      pfd.cDepthBits = 24;
      pfd.iLayerType = PFD_MAIN_PLANE;
      const int pf = ChoosePixelFormat(g_hdc, &pfd);
      SetPixelFormat(g_hdc, pf, &pfd);
      g_glrc = wglCreateContext(g_hdc);
      wglMakeCurrent(g_hdc, g_glrc);
      glEnable(GL_LINE_SMOOTH);
      return 0;
    }
    case WM_ERASEBKGND:
      return 1;
    case WM_PAINT: {
      PAINTSTRUCT ps;
      BeginPaint(hwnd, &ps);
      render_scene();
      SwapBuffers(g_hdc);
      EndPaint(hwnd, &ps);
      return 0;
    }
    case WM_SIZE:
      redraw();
      return 0;
    case WM_LBUTTONDOWN:
    case WM_RBUTTONDOWN:
    case WM_MBUTTONDOWN:
      SetFocus(hwnd);
      SetCapture(hwnd);
      g_drag = (msg == WM_LBUTTONDOWN) ? 1 : 2;
      g_last_mouse = {short(LOWORD(lp)), short(HIWORD(lp))};
      return 0;
    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
    case WM_MBUTTONUP:
      g_drag = 0;
      ReleaseCapture();
      return 0;
    case WM_MOUSEMOVE: {
      if (!g_drag) {
        return 0;
      }
      const POINT p{short(LOWORD(lp)), short(HIWORD(lp))};
      const float dx = float(p.x - g_last_mouse.x);
      const float dy = float(p.y - g_last_mouse.y);
      g_last_mouse = p;
      if (g_drag == 1) {
        g_yaw -= dx * 0.008f;
        g_pitch = std::clamp(g_pitch + dy * 0.008f, -1.55f, 1.55f);
      }
      else {
        const float cp = std::cos(g_pitch);
        const float f[3] = {-cp * std::cos(g_yaw), -cp * std::sin(g_yaw), -std::sin(g_pitch)};
        float s[3] = {f[1], -f[0], 0.0f};
        const float sl = std::max(1e-6f, std::sqrt(s[0] * s[0] + s[1] * s[1]));
        s[0] /= sl;
        s[1] /= sl;
        const float u[3] = {s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0]};
        const float k = g_dist * 0.0016f;
        for (int i = 0; i < 3; ++i) {
          g_target[i] += (-dx * s[i] + dy * u[i]) * k;
        }
      }
      redraw();
      return 0;
    }
    case WM_MOUSEWHEEL: {
      const int delta = GET_WHEEL_DELTA_WPARAM(wp);
      g_dist = std::clamp(g_dist * std::pow(0.88f, delta / 120.0f), 0.05f, 200.0f);
      redraw();
      return 0;
    }
    case WM_DESTROY:
      wglMakeCurrent(nullptr, nullptr);
      if (g_glrc) {
        wglDeleteContext(g_glrc);
      }
      ReleaseDC(hwnd, g_hdc);
      return 0;
    default:
      break;
  }
  return DefWindowProcW(hwnd, msg, wp, lp);
}

static LRESULT CALLBACK main_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
  switch (msg) {
    case WM_CREATE:
      g_main = hwnd;
      build_panel();
      g_gl = CreateWindowExW(0, L"EXCAD_GL", L"", WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN, PANEL_W, 0, 100, 100, hwnd, nullptr, g_inst, nullptr);
      SetTimer(hwnd, 1, 50, nullptr);
      return 0;
    case WM_SIZE:
      if (g_gl) {
        MoveWindow(g_gl, PANEL_W, 0, std::max(1, int(LOWORD(lp)) - PANEL_W), std::max(1, int(HIWORD(lp))), TRUE);
      }
      return 0;
    case WM_COMMAND:
      if (HIWORD(wp) == BN_CLICKED) {
        on_command(LOWORD(wp));
      }
      return 0;
    case WM_TIMER:
    case WM_EXCAD_BRIDGE:
      process_bridge_queue();
      if (msg == WM_TIMER) {
        update_bridge_label();
      }
      return 0;
    case WM_EXCAD_SCREENSHOT:
      frame_all();
      render_scene();
      save_screenshot_bmp(g_screenshot_path);
      SwapBuffers(g_hdc);
      PostMessageW(hwnd, WM_CLOSE, 0, 0);
      return 0;
    case WM_DESTROY:
      KillTimer(hwnd, 1);
      bridge_stop();
      PostQuitMessage(0);
      return 0;
    default:
      break;
  }
  return DefWindowProcW(hwnd, msg, wp, lp);
}

/* ------------------------------------------------------------------------- */
/* Headless self-test                                                        */
/* ------------------------------------------------------------------------- */

static std::wstring exe_dir()
{
  wchar_t buf[MAX_PATH];
  GetModuleFileNameW(nullptr, buf, MAX_PATH);
  std::wstring p(buf);
  const size_t slash = p.find_last_of(L"\\/");
  return (slash == std::wstring::npos) ? L"." : p.substr(0, slash);
}

static int run_self_test()
{
  std::ostringstream log;
  bool ok = true;
  auto check = [&](bool cond, const char *what) {
    log << (cond ? "[PASS] " : "[FAIL] ") << what << "\n";
    ok = ok && cond;
  };

  /* 1. Constraint solver restores a perturbed rectangle. */
  {
    SketchDocument d;
    d.reset(SketchPlane::XY);
    d.add_rect(-1.0, -0.5, 2.0, 1.0);
    d.points[1].u += 0.35;
    d.points[2].v += 0.20;
    const SolveResult r = d.solve();
    const double w = d.points[1].u - d.points[0].u;
    const double h = d.points[2].v - d.points[1].v;
    log << "solver: converged=" << r.converged << " iters=" << r.iterations << " residual=" << r.max_residual
        << " width=" << w << " height=" << h << "\n";
    check(r.converged && std::abs(w - 2.0) < 1e-5 && std::abs(h - 1.0) < 1e-5, "sketch solver restores dimensions");
  }

  /* 2. Extrude */
  {
    SketchDocument d;
    d.reset(SketchPlane::XY);
    d.add_rect(-1.0, -0.5, 2.0, 1.0);
    d.solve();
    RawMeshData m = extrude_sketch(d, 0.8f, true);
    ensure_outward(m);
    const double vol = signed_volume(m);
    log << "extrude: verts=" << m.verts.size() << " faces=" << m.faces.size() << " open_edges=" << count_open_edges(m)
        << " volume=" << vol << "\n";
    check(m.faces.size() == 6 && count_open_edges(m) == 0 && std::abs(vol - 1.6) < 1e-4, "extrude watertight, volume 1.6");
  }

  /* 3. Revolve (tube) on XZ plane */
  {
    SketchDocument d;
    d.reset(SketchPlane::XZ);
    d.add_rect(0.3, 0.0, 0.2, 1.0);
    d.solve();
    RawMeshData m = revolve_sketch(d, 360.0f, 64);
    ensure_outward(m);
    const double vol = signed_volume(m);
    const double expected = 3.14159265 * (0.25 - 0.09) * 64.0 / (2.0 * 3.14159265) * std::sin(2.0 * 3.14159265 / 64.0);
    log << "revolve: verts=" << m.verts.size() << " faces=" << m.faces.size() << " open_edges=" << count_open_edges(m)
        << " volume=" << vol << " expected=" << expected << "\n";
    check(count_open_edges(m) == 0 && std::abs(vol - expected) < 1e-3, "revolve watertight, correct volume");
  }

  /* 4. Anatomical humanoid via SDF + Marching Cubes */
  RawMeshData human;
  {
    HumanoidParams hp;
    hp.resolution = 110;
    const auto t0 = std::chrono::steady_clock::now();
    human = generate_humanoid_sdf(hp);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ensure_outward(human);
    const size_t open = count_open_edges(human);
    const double vol = signed_volume(human);
    float zmin = 1e9f, zmax = -1e9f;
    for (const auto &v : human.verts) {
      zmin = std::min(zmin, v[2]);
      zmax = std::max(zmax, v[2]);
    }
    log << "humanoid: verts=" << human.verts.size() << " faces=" << human.faces.size() << " open_edges=" << open
        << " volume_m3=" << vol << " height=" << (zmax - zmin) << " time_ms=" << ms << "\n";
    check(human.faces.size() > 10000, "humanoid has high-res mesh");
    check(open == 0, "humanoid mesh is watertight (closed manifold)");
    check(vol > 0.03 && vol < 0.15, "humanoid volume is anatomically plausible");
    check(zmax - zmin > 1.6f && zmax - zmin < 1.9f, "humanoid height ~1.75m");
  }

  /* 5. Custom SDF */
  {
    RawMeshData m = generate_custom_sdf_from_spec(
        "sphere 0 0 0.6 0.35 0.05\ncapsule 0 0 0.6 0 0 1.2 0.18 0.1 0.12\nsub_sphere 0 -0.3 0.6 0.18 0.04", 90);
    ensure_outward(m);
    log << "custom_sdf: verts=" << m.verts.size() << " faces=" << m.faces.size() << " open_edges=" << count_open_edges(m) << "\n";
    check(m.faces.size() > 1000 && count_open_edges(m) == 0, "custom SDF watertight");
  }

  const std::wstring dir = exe_dir();
  g_scene.clear();
  add_object("Humanoid", std::move(human), true);
  export_obj(dir + L"\\selftest_humanoid.obj");
  log << "exported: selftest_humanoid.obj\n";
  log << (ok ? "EXCAD_STANDALONE_SELF_TEST_OK" : "EXCAD_STANDALONE_SELF_TEST_FAIL") << "\n";

  const std::wstring log_path = dir + L"\\excad_selftest.log";
  std::ofstream(log_path.c_str()) << log.str();
  return ok ? 0 : 1;
}

/* ------------------------------------------------------------------------- */
/* Entry point                                                               */
/* ------------------------------------------------------------------------- */

static void build_demo_scene()
{
  HumanoidParams hp;
  hp.resolution = 110;
  do_humanoid(hp, "Humanoid");
  g_sketch.reset(SketchPlane::XY);
  g_sketch.add_rect(0.7, -0.35, 0.7, 0.7);
  g_sketch.add_circle(-1.0, 0.0, 0.28);
  do_extrude(0.5f, false, "Extrude");
  g_sketch.reset(SketchPlane::XZ);
  g_sketch.add_rect(0.18, 0.0, 0.08, 0.35);
  do_revolve(360.0f, 64, "Revolve");
  for (auto &v : g_scene.back().mesh.verts) {
    v[1] += 1.1f;
  }
  compute_normals(g_scene.back());
  g_sketch.reset(SketchPlane::XY);
}

int WINAPI WinMain(HINSTANCE inst, HINSTANCE, LPSTR, int show)
{
  g_inst = inst;
  int argc = 0;
  LPWSTR *argv = CommandLineToArgvW(GetCommandLineW(), &argc);
  bool self_test = false, demo = false, auto_bridge = false;
  for (int i = 1; i < argc; ++i) {
    const std::wstring a = argv[i];
    if (a == L"--self-test") {
      self_test = true;
    }
    else if (a == L"--demo") {
      demo = true;
    }
    else if (a == L"--bridge") {
      auto_bridge = true;
    }
    else if (a == L"--screenshot" && i + 1 < argc) {
      g_screenshot_path = argv[++i];
    }
  }
  LocalFree(argv);

  if (self_test) {
    return run_self_test();
  }

  WSADATA wsa{};
  WSAStartup(MAKEWORD(2, 2), &wsa);

  g_font = CreateFontW(-13, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
  g_font_bold = CreateFontW(-13, 0, 0, 0, FW_BOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");

  WNDCLASSEXW gl{};
  gl.cbSize = sizeof(gl);
  gl.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
  gl.lpfnWndProc = gl_proc;
  gl.hInstance = inst;
  gl.hCursor = LoadCursor(nullptr, IDC_ARROW);
  gl.lpszClassName = L"EXCAD_GL";
  RegisterClassExW(&gl);

  WNDCLASSEXW wc{};
  wc.cbSize = sizeof(wc);
  wc.lpfnWndProc = main_proc;
  wc.hInstance = inst;
  wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
  wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
  wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
  wc.lpszClassName = L"EXCAD_MAIN";
  RegisterClassExW(&wc);

  HWND hwnd = CreateWindowExW(0, L"EXCAD_MAIN", L"EX-CAD Standalone", WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN, CW_USEDEFAULT, CW_USEDEFAULT, 1440, 930, nullptr, nullptr, inst, nullptr);
  ShowWindow(hwnd, show);
  update_title();

  if (demo) {
    build_demo_scene();
    frame_all();
    set_status(L"Demo scene: SDF humanoid + extrude + revolve.");
  }
  if (auto_bridge) {
    bridge_start(9877);
  }
  UpdateWindow(hwnd);
  if (!g_screenshot_path.empty()) {
    PostMessageW(hwnd, WM_EXCAD_SCREENSHOT, 0, 0);
  }

  MSG msg;
  while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
    if (!IsDialogMessageW(hwnd, &msg)) {
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }
  }
  WSACleanup();
  return 0;
}
