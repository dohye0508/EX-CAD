/* SPDX-FileCopyrightText: 2026 EX-CAD Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <string>

#include "excad_mesh_types.hh"

namespace blender::ed::excad {

struct HumanoidParams {
  float height = 1.75f;
  float muscularity = 0.5f;
  float shoulder_ratio = 1.0f;
  float hip_ratio = 1.0f;
  int resolution = 72;
};

RawMeshData generate_humanoid_sdf(const HumanoidParams &params);

RawMeshData generate_custom_sdf_from_spec(const std::string &spec, int resolution = 64);

}  // namespace blender::ed::excad
