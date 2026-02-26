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
 * @file manual_ops/matmul.cpp
 * @brief Manual (non-SSA) matrix multiplication operations.
 *
 * All operations receive a pre-allocated output tile as the last argument.
 *   manual.matmul       (lhs, rhs, out)
 *   manual.matmul_acc   (acc, lhs, rhs, out)
 *   manual.matmul_bias  (lhs, rhs, bias, out)
 *   manual.gemv         (lhs, rhs, out)
 *   manual.gemv_acc     (acc, lhs, rhs, out)
 *   manual.gemv_bias    (lhs, rhs, bias, out)
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
// Op registration
// ---------------------------------------------------------------------------

// manual.matmul: (lhs, rhs, out) -> out's type
REGISTER_OP("manual.matmul")
    .set_op_category("ManualOp")
    .set_description("Manual matrix multiplication: out = lhs @ rhs")
    .add_argument("lhs", "Left matrix tile [M,K] (TileType)")
    .add_argument("rhs", "Right matrix tile [K,N] (TileType)")
    .add_argument("out", "Pre-allocated output tile [M,N] (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.matmul", 3);
    });

// manual.matmul_acc: (acc, lhs, rhs, out) -> out's type
REGISTER_OP("manual.matmul_acc")
    .set_op_category("ManualOp")
    .set_description("Manual matmul with accumulation: out = acc + lhs @ rhs")
    .add_argument("acc", "Accumulator tile [M,N] (TileType)")
    .add_argument("lhs", "Left matrix tile [M,K] (TileType)")
    .add_argument("rhs", "Right matrix tile [K,N] (TileType)")
    .add_argument("out", "Pre-allocated output tile [M,N] (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.matmul_acc", 4);
    });

// manual.matmul_bias: (lhs, rhs, bias, out) -> out's type
REGISTER_OP("manual.matmul_bias")
    .set_op_category("ManualOp")
    .set_description("Manual matmul with bias: out = lhs @ rhs + bias")
    .add_argument("lhs", "Left matrix tile [M,K] (TileType)")
    .add_argument("rhs", "Right matrix tile [K,N] (TileType)")
    .add_argument("bias", "Bias tile [1,N] (TileType)")
    .add_argument("out", "Pre-allocated output tile [M,N] (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.matmul_bias", 4);
    });

// manual.gemv: (lhs, rhs, out) -> out's type
REGISTER_OP("manual.gemv")
    .set_op_category("ManualOp")
    .set_description("Manual GEMV: out[1,N] = lhs[1,K] @ rhs[K,N]")
    .add_argument("lhs", "Row vector tile [1,K] (TileType)")
    .add_argument("rhs", "Matrix tile [K,N] (TileType)")
    .add_argument("out", "Pre-allocated output tile [1,N] (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.gemv", 3);
    });

// manual.gemv_acc: (acc, lhs, rhs, out) -> out's type
REGISTER_OP("manual.gemv_acc")
    .set_op_category("ManualOp")
    .set_description("Manual GEMV with accumulation: out += lhs @ rhs")
    .add_argument("acc", "Accumulator tile [1,N] (TileType)")
    .add_argument("lhs", "Row vector tile [1,K] (TileType)")
    .add_argument("rhs", "Matrix tile [K,N] (TileType)")
    .add_argument("out", "Pre-allocated output tile [1,N] (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.gemv_acc", 4);
    });

// manual.gemv_bias: (lhs, rhs, bias, out) -> out's type
REGISTER_OP("manual.gemv_bias")
    .set_op_category("ManualOp")
    .set_description("Manual GEMV with bias: out = lhs @ rhs + bias")
    .add_argument("lhs", "Row vector tile [1,K] (TileType)")
    .add_argument("rhs", "Matrix tile [K,N] (TileType)")
    .add_argument("bias", "Bias tile [1,N] (TileType)")
    .add_argument("out", "Pre-allocated output tile [1,N] (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutTileType(args, kwargs, "manual.gemv_bias", 4);
    });

}  // namespace ir
}  // namespace pypto
