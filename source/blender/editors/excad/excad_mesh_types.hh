/* SPDX-FileCopyrightText: 2026 EX-CAD Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <array>
#include <vector>

namespace blender::ed::excad {

struct RawMeshData {
  std::vector<std::array<float, 3>> verts;
  std::vector<std::vector<int>> faces;
};

}  // namespace blender::ed::excad
