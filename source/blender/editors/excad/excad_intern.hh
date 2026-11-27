/* SPDX-FileCopyrightText: 2026 EX-CAD Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "excad_mesh_types.hh"

namespace blender {
struct bContext;
struct Object;
}  // namespace blender

namespace blender::ed::excad {

Object *create_mesh_object_from_raw(bContext *C, const char *name, const RawMeshData &raw);

}  // namespace blender::ed::excad
