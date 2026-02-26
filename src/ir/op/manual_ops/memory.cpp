/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * @file manual_ops/memory.cpp
 * @brief Manual (non-SSA) memory operations: load, move, ub_copy, full, fillpad.
 *
 * Each "manual" op receives the pre-allocated output tile as its last argument
 * and returns that tile's type rather than creating a fresh SSA result type.
 * This mirrors the hardware semantics where the programmer explicitly manages
 * tile buffers.
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

// ---------------------------------------------------------------------------
// Common helpers
// ---------------------------------------------------------------------------

/// Return the TileType of the last argument (the pre-allocated output tile).
static TypePtr DeduceManualOutTileType(const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs,
                                       const std::string& op_name, size_t expected_args) {
  CHECK(args.size() == expected_args)
      << "The operator " << op_name << " requires exactly " << expected_args << " arguments, but got "
      << args.size();
  auto out_type = As<TileType>(args.back()->GetType());
  CHECK(out_type) << "The operator " << op_name
                  << " requires last argument (out) to be TileType, but got "
                  << args.back()->GetType()->TypeName();
  return out_type;
}

// ---------------------------------------------------------------------------
// Op registration
// ---------------------------------------------------------------------------

// manual.load: (tensor, offsets, shapes, out) -> TileType (out's type)
REGISTER_OP("manual.load")
    .set_op_category("ManualOp")
    .set_description(
        "Manual load: copy data from a global tensor into a pre-allocated tile. "
        "The output tile (last arg) defines the destination buffer; its type is returned.")
    .add_argument("tensor", "Source tensor (TensorType)")
    .add_argument("offsets", "Offset tuple per dimension (MakeTuple)")
    .add_argument("shapes", "Size tuple per dimension (MakeTuple)")
    .add_argument("out", "Pre-allocated destination tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.load", 4);
    });

// manual.move: (src_tile, out) -> TileType (out's type)
REGISTER_OP("manual.move")
    .set_op_category("ManualOp")
    .set_description(
        "Manual move: transfer a tile between memory levels into a pre-allocated buffer.")
    .add_argument("src", "Source tile (TileType)")
    .add_argument("out", "Pre-allocated destination tile (TileType)")
    .set_attr<MemorySpace>("target_memory")
    .set_attr<bool>("transpose")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.move", 2);
    });

// manual.ub_copy: (src_tile, out) -> TileType (out's type)
REGISTER_OP("manual.ub_copy")
    .set_op_category("ManualOp")
    .set_description(
        "Manual UB-to-UB copy: copy a tile within unified buffer into a pre-allocated buffer.")
    .add_argument("src", "Source UB tile (TileType)")
    .add_argument("out", "Pre-allocated destination UB tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.ub_copy", 2);
    });

// manual.full: (scalar, out) -> TileType (out's type)
// Fills the pre-allocated tile with a scalar value. Shape comes from out's TileType.
REGISTER_OP("manual.full")
    .set_op_category("ManualOp")
    .set_description(
        "Manual fill: broadcast a scalar value across a pre-allocated tile (out = scalar).")
    .add_argument("scalar", "Fill value (ScalarType or constant)")
    .add_argument("out", "Pre-allocated tile to fill (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.full", 2);
    });

// manual.fillpad: (src_tile, out) -> TileType (out's type)
REGISTER_OP("manual.fillpad")
    .set_op_category("ManualOp")
    .set_description(
        "Manual fill-with-padding: copy src tile into out and pad remaining elements.")
    .add_argument("src", "Source tile (TileType)")
    .add_argument("out", "Pre-allocated destination tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.fillpad", 2);
    });

}  // namespace ir
}  // namespace pypto
