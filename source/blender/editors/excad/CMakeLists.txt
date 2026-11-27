# SPDX-FileCopyrightText: 2026 EX-CAD Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

set(INC
  ../include
  ../../makesrna

  # RNA_prototypes.hh
  ${CMAKE_BINARY_DIR}/source/blender/makesrna
)

set(INC_SYS
)

set(SRC
  excad_ai_bridge.cc
  excad_ops.cc
  excad_sdf_mesh.cc
  excad_sketch_solver.cc

  excad_ai_bridge.hh
  excad_intern.hh
  excad_mesh_types.hh
  excad_sdf_mesh.hh
  excad_sketch_solver.hh
)

set(LIB
  PRIVATE bf::blenkernel
  PRIVATE bf::blenlib
  PRIVATE bf::blentranslation
  PRIVATE bf::bmesh
  PRIVATE bf::depsgraph
  PRIVATE bf::dna
  bf_editor_object
  PRIVATE bf::intern::guardedalloc
  PRIVATE bf::windowmanager
)

if(WIN32)
  list(APPEND LIB ws2_32)
endif()

blender_add_lib(bf_editor_excad "${SRC}" "${INC}" "${INC_SYS}" "${LIB}")

# RNA_prototypes.hh dna_type_offsets.h
add_dependencies(bf_editor_excad bf_rna)
