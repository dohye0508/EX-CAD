/* SPDX-FileCopyrightText: 2026 EX-CAD Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "excad_sdf_mesh.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace blender::ed::excad {

namespace {

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

inline Vec3 operator+(const Vec3 &a, const Vec3 &b)
{
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
inline Vec3 operator-(const Vec3 &a, const Vec3 &b)
{
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline Vec3 operator*(const Vec3 &a, float s)
{
  return {a.x * s, a.y * s, a.z * s};
}
inline float dot3(const Vec3 &a, const Vec3 &b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline float len3(const Vec3 &v)
{
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline float smin_poly(float a, float b, float k)
{
  if (k <= 1e-6f) {
    return std::min(a, b);
  }
  const float h = std::max(k - std::abs(a - b), 0.0f) / k;
  return std::min(a, b) - h * h * k * 0.25f;
}

inline float smax_poly(float a, float b, float k)
{
  return -smin_poly(-a, -b, k);
}

inline float sdf_sphere(const Vec3 &p, const Vec3 &c, float r)
{
  return len3(p - c) - r;
}

inline float sdf_ellipsoid(const Vec3 &p, const Vec3 &c, const Vec3 &r)
{
  const Vec3 q{
      (p.x - c.x) / std::max(r.x, 1e-5f),
      (p.y - c.y) / std::max(r.y, 1e-5f),
      (p.z - c.z) / std::max(r.z, 1e-5f),
  };
  const float k0 = len3(q);
  if (k0 < 1e-6f) {
    return -std::min({r.x, r.y, r.z});
  }
  const Vec3 q2{
      q.x / std::max(r.x, 1e-5f),
      q.y / std::max(r.y, 1e-5f),
      q.z / std::max(r.z, 1e-5f),
  };
  const float k1 = len3(q2);
  return k0 * (k0 - 1.0f) / std::max(k1, 1e-6f);
}

inline float sdf_capsule(const Vec3 &p, const Vec3 &a, const Vec3 &b, float ra, float rb)
{
  const Vec3 pa = p - a;
  const Vec3 ba = b - a;
  const float baba = dot3(ba, ba);
  const float h = (baba > 1e-8f) ? std::clamp(dot3(pa, ba) / baba, 0.0f, 1.0f) : 0.0f;
  const float r = ra + (rb - ra) * h;
  return len3(pa - ba * h) - r;
}

inline float sdf_round_box(const Vec3 &p, const Vec3 &c, const Vec3 &half_ext, float r)
{
  const Vec3 q{
      std::abs(p.x - c.x) - half_ext.x + r,
      std::abs(p.y - c.y) - half_ext.y + r,
      std::abs(p.z - c.z) - half_ext.z + r,
  };
  const Vec3 q_pos{std::max(q.x, 0.0f), std::max(q.y, 0.0f), std::max(q.z, 0.0f)};
  return len3(q_pos) + std::min(std::max({q.x, q.y, q.z}), 0.0f) - r;
}

/* Standard Paul Bourke / Lorensen-Cline Marching Cubes tables. */
static const int MC_EDGE_TABLE[256] = {
    0x0,   0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a,
    0xd03, 0xe09, 0xf00, 0x190, 0x99,  0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895,
    0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x33,  0x13a, 0x636, 0x73f, 0x435,
    0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0xaa,
    0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460,
    0x569, 0x663, 0x76a, 0x66,  0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963,
    0xa69, 0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff,  0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff,
    0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55,  0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6,
    0x2cf, 0x1c5, 0xcc,  0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9,
    0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc,  0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9,
    0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55,  0x35f, 0x256,
    0x55a, 0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc,
    0x3f5, 0xff,  0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f,
    0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x66,  0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3,
    0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa,  0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a,
    0x33,  0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795,
    0x49f, 0x596, 0x29a, 0x393, 0x99,  0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905,
    0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
};

static const int MC_TRI_TABLE[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
    {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
    {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
    {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
    {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
    {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
    {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
    {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
    {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
    {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
    {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
    {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
    {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
    {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
    {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
    {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
    {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
    {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
    {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
    {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
    {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
    {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
    {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
    {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
    {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
    {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
    {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
    {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
    {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
    {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
    {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
    {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
    {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
    {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
    {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
    {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
    {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
    {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
    {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
    {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
    {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
    {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
    {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
    {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
    {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
    {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
    {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
    {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
    {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
    {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
    {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
    {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
};

void taubin_smooth(RawMeshData &mesh, int iterations = 2)
{
  const int n = static_cast<int>(mesh.verts.size());
  if (n == 0 || mesh.faces.empty()) {
    return;
  }

  std::vector<std::vector<int>> adj(n);
  for (const auto &f : mesh.faces) {
    const int fn = static_cast<int>(f.size());
    for (int i = 0; i < fn; ++i) {
      const int a = f[i];
      const int b = f[(i + 1) % fn];
      adj[a].push_back(b);
      adj[b].push_back(a);
    }
  }
  for (auto &nbrs : adj) {
    std::sort(nbrs.begin(), nbrs.end());
    nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
  }

  auto laplacian_step = [&](float factor) {
    std::vector<std::array<float, 3>> next_verts = mesh.verts;
    for (int i = 0; i < n; ++i) {
      if (adj[i].empty()) {
        continue;
      }
      float sx = 0.0f, sy = 0.0f, sz = 0.0f;
      for (int nb : adj[i]) {
        sx += mesh.verts[nb][0];
        sy += mesh.verts[nb][1];
        sz += mesh.verts[nb][2];
      }
      const float inv = 1.0f / static_cast<float>(adj[i].size());
      const float dx = sx * inv - mesh.verts[i][0];
      const float dy = sy * inv - mesh.verts[i][1];
      const float dz = sz * inv - mesh.verts[i][2];
      next_verts[i][0] += factor * dx;
      next_verts[i][1] += factor * dy;
      next_verts[i][2] += factor * dz;
    }
    mesh.verts = std::move(next_verts);
  };

  for (int it = 0; it < iterations; ++it) {
    laplacian_step(0.50f);
    laplacian_step(-0.53f);
  }
}

RawMeshData extract_isosurface_mc(const std::function<float(const Vec3 &)> &sdf,
                                  const Vec3 &bmin,
                                  const Vec3 &bmax,
                                  int res_x,
                                  int res_y,
                                  int res_z)
{
  RawMeshData mesh;
  const int nx = std::max(8, res_x);
  const int ny = std::max(8, res_y);
  const int nz = std::max(8, res_z);

  const float dx = (bmax.x - bmin.x) / nx;
  const float dy = (bmax.y - bmin.y) / ny;
  const float dz = (bmax.z - bmin.z) / nz;

  const int sx = nx + 1;
  const int sy = ny + 1;
  std::vector<float> field(static_cast<size_t>(sx) * sy * (nz + 1));

  auto idx = [&](int x, int y, int z) -> size_t {
    return (static_cast<size_t>(z) * sy + y) * sx + x;
  };

  for (int z = 0; z <= nz; ++z) {
    const float pz = bmin.z + z * dz;
    for (int y = 0; y <= ny; ++y) {
      const float py = bmin.y + y * dy;
      for (int x = 0; x <= nx; ++x) {
        const float px = bmin.x + x * dx;
        field[idx(x, y, z)] = sdf({px, py, pz});
      }
    }
  }

  std::unordered_map<uint64_t, int> edge_vert_map;
  edge_vert_map.reserve(static_cast<size_t>(nx) * ny * 2);

  /* Canonical edge key between two grid points (x0,y0,z0) and (x1,y1,z1). */
  auto make_edge_key = [&](int x0, int y0, int z0, int x1, int y1, int z1) -> uint64_t {
    uint64_t a = (static_cast<uint64_t>(z0) << 20) | (static_cast<uint64_t>(y0) << 10) | x0;
    uint64_t b = (static_cast<uint64_t>(z1) << 20) | (static_cast<uint64_t>(y1) << 10) | x1;
    if (a > b) {
      std::swap(a, b);
    }
    /* Exact, collision-free key: each endpoint index fits in 30 bits. */
    return (a << 30) | b;
  };

  static const int corner_offsets[8][3] = {
      {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
  };
  static const int edge_corners[12][2] = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
      {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
  };

  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        float val[8];
        int cube_idx = 0;
        for (int c = 0; c < 8; ++c) {
          val[c] = field[idx(x + corner_offsets[c][0],
                             y + corner_offsets[c][1],
                             z + corner_offsets[c][2])];
          if (val[c] < 0.0f) {
            cube_idx |= (1 << c);
          }
        }
        const int edge_mask = MC_EDGE_TABLE[cube_idx];
        if (edge_mask == 0) {
          continue;
        }

        int vert_list[12];
        for (int e = 0; e < 12; ++e) {
          if ((edge_mask & (1 << e)) == 0) {
            continue;
          }
          const int c0 = edge_corners[e][0];
          const int c1 = edge_corners[e][1];
          const int x0 = x + corner_offsets[c0][0];
          const int y0 = y + corner_offsets[c0][1];
          const int z0 = z + corner_offsets[c0][2];
          const int x1 = x + corner_offsets[c1][0];
          const int y1 = y + corner_offsets[c1][1];
          const int z1 = z + corner_offsets[c1][2];

          const uint64_t key = make_edge_key(x0, y0, z0, x1, y1, z1);
          auto it = edge_vert_map.find(key);
          if (it != edge_vert_map.end()) {
            vert_list[e] = it->second;
          }
          else {
            const float v0 = val[c0];
            const float v1 = val[c1];
            const float denom = v1 - v0;
            const float t = (std::abs(denom) > 1e-7f) ? std::clamp(-v0 / denom, 0.0f, 1.0f) : 0.5f;
            const float px = bmin.x + (x0 + t * (x1 - x0)) * dx;
            const float py = bmin.y + (y0 + t * (y1 - y0)) * dy;
            const float pz = bmin.z + (z0 + t * (z1 - z0)) * dz;
            const int vid = static_cast<int>(mesh.verts.size());
            mesh.verts.push_back({px, py, pz});
            edge_vert_map.emplace(key, vid);
            vert_list[e] = vid;
          }
        }

        for (int i = 0; MC_TRI_TABLE[cube_idx][i] != -1; i += 3) {
          const int i0 = vert_list[MC_TRI_TABLE[cube_idx][i]];
          const int i1 = vert_list[MC_TRI_TABLE[cube_idx][i + 1]];
          const int i2 = vert_list[MC_TRI_TABLE[cube_idx][i + 2]];
          if (i0 != i1 && i1 != i2 && i2 != i0) {
            /* Reverse winding for outward normals when inside is < 0. */
            mesh.faces.push_back({i0, i2, i1});
          }
        }
      }
    }
  }

  taubin_smooth(mesh, 2);
  return mesh;
}

}  // namespace

RawMeshData generate_humanoid_sdf(const HumanoidParams &params)
{
  const float scale = std::clamp(params.height, 0.5f, 3.0f) / 1.75f;
  const float musc = std::clamp(params.muscularity, 0.0f, 1.5f);
  const float sh = std::clamp(params.shoulder_ratio, 0.7f, 1.4f);
  const float hp = std::clamp(params.hip_ratio, 0.7f, 1.4f);
  const int res = std::clamp(params.resolution, 24, 128);

  auto eval = [=](const Vec3 &wp) -> float {
    /* Normalize to 1.75m anatomical space, Z-up, Y-back (facing -Y). */
    const Vec3 p{wp.x / scale, wp.y / scale, wp.z / scale};
    const Vec3 ps{std::abs(p.x), p.y, p.z}; /* Bilaterally symmetric lookup */

    /* Head & Neck */
    float d = sdf_ellipsoid(p, {0.0f, 0.01f, 1.63f}, {0.082f, 0.098f, 0.112f});
    d = smin_poly(d, sdf_ellipsoid(p, {0.0f, -0.028f, 1.565f}, {0.058f, 0.068f, 0.072f}), 0.035f);
    d = smin_poly(d, sdf_capsule(p, {0.0f, 0.0f, 1.44f}, {0.0f, 0.008f, 1.54f}, 0.056f, 0.048f), 0.03f);

    /* Trapezius & Clavicles */
    d = smin_poly(d,
                  sdf_capsule(ps, {0.035f, 0.01f, 1.47f}, {0.175f * sh, 0.01f, 1.41f}, 0.048f, 0.038f),
                  0.04f);

    /* Ribcage, Pectorals, Abdomen, Pelvis */
    const float rib = sdf_ellipsoid(
        p, {0.0f, 0.005f, 1.29f}, {0.145f * sh, 0.105f + 0.015f * musc, 0.165f});
    d = smin_poly(d, rib, 0.05f);

    const float pec = sdf_ellipsoid(
        ps, {0.072f * sh, -0.078f, 1.32f}, {0.075f, 0.045f + 0.018f * musc, 0.068f});
    d = smin_poly(d, pec, 0.035f);

    const float abs_core = sdf_ellipsoid(
        p, {0.0f, -0.005f, 1.09f}, {0.118f + 0.01f * musc, 0.092f, 0.125f});
    d = smin_poly(d, abs_core, 0.045f);

    const float pelvis = sdf_ellipsoid(
        p, {0.0f, 0.0f, 0.93f}, {0.142f * hp, 0.102f, 0.095f});
    d = smin_poly(d, pelvis, 0.05f);

    const float glute = sdf_ellipsoid(
        ps, {0.068f * hp, 0.055f, 0.90f}, {0.072f, 0.068f, 0.085f});
    d = smin_poly(d, glute, 0.04f);

    /* Shoulders & Arms (natural relaxed A-pose) */
    const Vec3 shoulder{0.195f * sh, 0.01f, 1.40f};
    const Vec3 elbow{0.275f * sh, 0.025f, 1.13f};
    const Vec3 wrist{0.335f * sh, -0.015f, 0.87f};
    const Vec3 hand_end{0.355f * sh, -0.025f, 0.77f};

    d = smin_poly(d, sdf_sphere(ps, shoulder, 0.056f + 0.014f * musc), 0.035f);
    d = smin_poly(
        d, sdf_capsule(ps, shoulder, elbow, 0.048f + 0.012f * musc, 0.040f + 0.008f * musc), 0.03f);
    d = smin_poly(
        d, sdf_capsule(ps, elbow, wrist, 0.041f + 0.008f * musc, 0.029f), 0.025f);
    d = smin_poly(d, sdf_capsule(ps, wrist, hand_end, 0.030f, 0.022f), 0.02f);

    /* Legs & Feet */
    const Vec3 hip{0.092f * hp, 0.0f, 0.89f};
    const Vec3 knee{0.095f, -0.008f, 0.50f};
    const Vec3 ankle{0.092f, 0.015f, 0.09f};
    const Vec3 heel{0.092f, 0.040f, 0.035f};
    const Vec3 toe{0.092f, -0.105f, 0.025f};

    d = smin_poly(
        d, sdf_capsule(ps, hip, knee, 0.078f + 0.015f * musc, 0.054f + 0.006f * musc), 0.045f);
    d = smin_poly(
        d, sdf_capsule(ps, knee, ankle, 0.054f + 0.008f * musc, 0.036f), 0.03f);
    d = smin_poly(d, sdf_capsule(ps, ankle, heel, 0.036f, 0.034f), 0.02f);
    d = smin_poly(d, sdf_capsule(ps, heel, toe, 0.036f, 0.026f), 0.02f);

    return d * scale;
  };

  const Vec3 bmin{-0.55f * scale, -0.35f * scale, -0.05f * scale};
  const Vec3 bmax{0.55f * scale, 0.35f * scale, 1.85f * scale};
  const int res_x = std::max(24, static_cast<int>(res * 0.65f));
  const int res_y = std::max(20, static_cast<int>(res * 0.45f));
  return extract_isosurface_mc(eval, bmin, bmax, res_x, res_y, res);
}

namespace {

struct SdfPrimOp {
  enum class Kind { SPHERE, CAPSULE, ELLIPSOID, BOX };
  enum class Combine { UNION, SUBTRACT };

  Kind kind = Kind::SPHERE;
  Combine combine = Combine::UNION;
  float k = 0.04f;
  Vec3 a{};
  Vec3 b{};
  float r0 = 0.5f;
  float r1 = 0.5f;
};

}  // namespace

RawMeshData generate_custom_sdf_from_spec(const std::string &spec, int resolution)
{
  std::vector<SdfPrimOp> ops;
  std::istringstream iss(spec);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream ls(line);
    std::string cmd;
    ls >> cmd;
    SdfPrimOp op;
    if (cmd == "sphere") {
      op.kind = SdfPrimOp::Kind::SPHERE;
      ls >> op.a.x >> op.a.y >> op.a.z >> op.r0 >> op.k;
      ops.push_back(op);
    }
    else if (cmd == "capsule") {
      op.kind = SdfPrimOp::Kind::CAPSULE;
      ls >> op.a.x >> op.a.y >> op.a.z >> op.b.x >> op.b.y >> op.b.z >> op.r0 >> op.r1 >> op.k;
      ops.push_back(op);
    }
    else if (cmd == "ellipsoid") {
      op.kind = SdfPrimOp::Kind::ELLIPSOID;
      ls >> op.a.x >> op.a.y >> op.a.z >> op.b.x >> op.b.y >> op.b.z >> op.k;
      ops.push_back(op);
    }
    else if (cmd == "box") {
      op.kind = SdfPrimOp::Kind::BOX;
      ls >> op.a.x >> op.a.y >> op.a.z >> op.b.x >> op.b.y >> op.b.z >> op.r0 >> op.k;
      ops.push_back(op);
    }
    else if (cmd == "sub_sphere") {
      op.kind = SdfPrimOp::Kind::SPHERE;
      op.combine = SdfPrimOp::Combine::SUBTRACT;
      ls >> op.a.x >> op.a.y >> op.a.z >> op.r0 >> op.k;
      ops.push_back(op);
    }
  }

  if (ops.empty()) {
    SdfPrimOp fallback;
    fallback.kind = SdfPrimOp::Kind::SPHERE;
    fallback.a = {0.0f, 0.0f, 0.5f};
    fallback.r0 = 0.5f;
    ops.push_back(fallback);
  }

  Vec3 bmin{1e9f, 1e9f, 1e9f};
  Vec3 bmax{-1e9f, -1e9f, -1e9f};
  for (const auto &op : ops) {
    const float pad = std::max(op.r0, op.r1) + 0.35f;
    bmin.x = std::min(bmin.x, op.a.x - pad);
    bmin.y = std::min(bmin.y, op.a.y - pad);
    bmin.z = std::min(bmin.z, op.a.z - pad);
    bmax.x = std::max(bmax.x, op.a.x + pad);
    bmax.y = std::max(bmax.y, op.a.y + pad);
    bmax.z = std::max(bmax.z, op.a.z + pad);
    if (op.kind == SdfPrimOp::Kind::CAPSULE) {
      bmin.x = std::min(bmin.x, op.b.x - pad);
      bmin.y = std::min(bmin.y, op.b.y - pad);
      bmin.z = std::min(bmin.z, op.b.z - pad);
      bmax.x = std::max(bmax.x, op.b.x + pad);
      bmax.y = std::max(bmax.y, op.b.y + pad);
      bmax.z = std::max(bmax.z, op.b.z + pad);
    }
    else if (op.kind == SdfPrimOp::Kind::ELLIPSOID || op.kind == SdfPrimOp::Kind::BOX) {
      bmin.x = std::min(bmin.x, op.a.x - op.b.x - 0.3f);
      bmin.y = std::min(bmin.y, op.a.y - op.b.y - 0.3f);
      bmin.z = std::min(bmin.z, op.a.z - op.b.z - 0.3f);
      bmax.x = std::max(bmax.x, op.a.x + op.b.x + 0.3f);
      bmax.y = std::max(bmax.y, op.a.y + op.b.y + 0.3f);
      bmax.z = std::max(bmax.z, op.a.z + op.b.z + 0.3f);
    }
  }

  auto eval = [ops](const Vec3 &p) -> float {
    float d = 1e6f;
    for (size_t i = 0; i < ops.size(); ++i) {
      const auto &op = ops[i];
      float prim = 1e6f;
      switch (op.kind) {
        case SdfPrimOp::Kind::SPHERE:
          prim = sdf_sphere(p, op.a, op.r0);
          break;
        case SdfPrimOp::Kind::CAPSULE:
          prim = sdf_capsule(p, op.a, op.b, op.r0, op.r1);
          break;
        case SdfPrimOp::Kind::ELLIPSOID:
          prim = sdf_ellipsoid(p, op.a, op.b);
          break;
        case SdfPrimOp::Kind::BOX:
          prim = sdf_round_box(p, op.a, op.b, op.r0);
          break;
      }
      if (i == 0) {
        d = prim;
      }
      else if (op.combine == SdfPrimOp::Combine::SUBTRACT) {
        d = smax_poly(d, -prim, op.k);
      }
      else {
        d = smin_poly(d, prim, op.k);
      }
    }
    return d;
  };

  const int res = std::clamp(resolution, 24, 128);
  return extract_isosurface_mc(eval, bmin, bmax, res, res, res);
}

}  // namespace blender::ed::excad
