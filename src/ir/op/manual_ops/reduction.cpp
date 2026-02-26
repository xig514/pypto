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
 * @file manual_ops/reduction.cpp
 * @brief Manual (non-SSA) reduction and broadcast operations.
 *
 * Reduction ops: row_sum, row_max, row_min  (tile, tmp, out)
 * Broadcast ops: row_expand, col_expand, row_expand_*, col_expand_*
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

static TypePtr DeduceManualOutTileType(const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs,
                                       const std::string& op_name, size_t expected_args) {
  CHECK(args.size() == expected_args)
      << "The operator " << op_name << " requires exactly " << expected_args
      << " arguments, but got " << args.size();
  auto out_type = As<TileType>(args.back()->GetType());
  CHECK(out_type) << "The operator " << op_name
                  << " requires last argument (out) to be TileType, but got "
                  << args.back()->GetType()->TypeName();
  return out_type;
}

// ---------------------------------------------------------------------------
// Reduction operations: (tile, tmp, out) -> out's type
// ---------------------------------------------------------------------------

REGISTER_OP("manual.row_sum")
    .set_op_category("ManualOp")
    .set_description("Manual row-wise sum reduction: out[i,0] = sum_j(tile[i,j])")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .add_argument("out", "Pre-allocated output row vector tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.row_sum", 3);
    });

REGISTER_OP("manual.row_max")
    .set_op_category("ManualOp")
    .set_description("Manual row-wise max reduction: out[i,0] = max_j(tile[i,j])")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .add_argument("out", "Pre-allocated output row vector tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.row_max", 3);
    });

REGISTER_OP("manual.row_min")
    .set_op_category("ManualOp")
    .set_description("Manual row-wise min reduction: out[i,0] = min_j(tile[i,j])")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .add_argument("out", "Pre-allocated output row vector tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.row_min", 3);
    });

// ---------------------------------------------------------------------------
// Broadcast / expansion operations
// ---------------------------------------------------------------------------

// row_expand (src, out): unary broadcast.
REGISTER_OP("manual.row_expand")
    .set_op_category("ManualOp")
    .set_description("Manual row broadcast: out[i,j] = src[i,0] for all j")
    .add_argument("src", "Source tile [M,1] (TileType)")
    .add_argument("out", "Pre-allocated output tile [M,N] (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.row_expand", 2);
    });

// col_expand (col_vec, out): unary broadcast.
REGISTER_OP("manual.col_expand")
    .set_op_category("ManualOp")
    .set_description("Manual column broadcast: out[i,j] = col_vec[0,j] for all i")
    .add_argument("col_vec", "Source column vector [1,N] (TileType)")
    .add_argument("out", "Pre-allocated output tile [M,N] (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.col_expand", 2);
    });

// row_expand_* (tile, row_vec, out): binary broadcast-arithmetic.
#define REGISTER_MANUAL_ROW_EXPAND(op_suffix, description)                                       \
  REGISTER_OP("manual.row_expand_" #op_suffix)                                                  \
      .set_op_category("ManualOp")                                                               \
      .set_description("Manual row broadcast " description)                                      \
      .add_argument("tile", "Input tile [M,N] (TileType)")                                       \
      .add_argument("row_vec", "Row vector [M,1] (TileType)")                                    \
      .add_argument("out", "Pre-allocated output tile [M,N] (TileType)")                         \
      .f_deduce_type([](const std::vector<ExprPtr>& args,                                        \
                        const std::vector<std::pair<std::string, std::any>>& kwargs) {           \
        return DeduceManualOutTileType(args, kwargs, "manual.row_expand_" #op_suffix, 3);        \
      })

REGISTER_MANUAL_ROW_EXPAND(add, "add: out = tile + broadcast(row_vec)");
REGISTER_MANUAL_ROW_EXPAND(sub, "sub: out = tile - broadcast(row_vec)");
REGISTER_MANUAL_ROW_EXPAND(mul, "mul: out = tile * broadcast(row_vec)");
REGISTER_MANUAL_ROW_EXPAND(div, "div: out = tile / broadcast(row_vec)");

#undef REGISTER_MANUAL_ROW_EXPAND

// col_expand_* (tile, col_vec, out): binary broadcast-arithmetic.
#define REGISTER_MANUAL_COL_EXPAND(op_suffix, description)                                       \
  REGISTER_OP("manual.col_expand_" #op_suffix)                                                  \
      .set_op_category("ManualOp")                                                               \
      .set_description("Manual column broadcast " description)                                   \
      .add_argument("tile", "Input tile [M,N] (TileType)")                                       \
      .add_argument("col_vec", "Column vector [1,N] (TileType)")                                 \
      .add_argument("out", "Pre-allocated output tile [M,N] (TileType)")                         \
      .f_deduce_type([](const std::vector<ExprPtr>& args,                                        \
                        const std::vector<std::pair<std::string, std::any>>& kwargs) {           \
        return DeduceManualOutTileType(args, kwargs, "manual.col_expand_" #op_suffix, 3);        \
      })

REGISTER_MANUAL_COL_EXPAND(mul, "mul: out = tile * broadcast(col_vec)");
REGISTER_MANUAL_COL_EXPAND(div, "div: out = tile / broadcast(col_vec)");
REGISTER_MANUAL_COL_EXPAND(sub, "sub: out = tile - broadcast(col_vec)");

#undef REGISTER_MANUAL_COL_EXPAND

}  // namespace ir
}  // namespace pypto
