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
 * @file manual_ops/elementwise.cpp
 * @brief Manual (non-SSA) element-wise operations.
 *
 * All ops accept a pre-allocated output tile as the last argument and return
 * that tile's type.  This file covers:
 *   - Tile x Tile binary: add, sub, mul, div, rem, maximum, minimum, and, or, shl, shr
 *   - Tile x Scalar binary: adds, subs, muls, divs, rems, ands, ors, shls, shrs, maxs, mins, lrelu
 *   - Unary: neg, exp, sqrt, rsqrt, recip, log, abs, relu, not, cast
 *   - Ternary: xor/xors (with tmp), prelu (with tmp), addc, subc, addsc, subsc, sel, sels
 *   - Comparison: cmp, cmps
 *   - Scalar-to-tile: expands
 *   - Layout: reshape, transpose
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
// Shared type deduction helpers
// ---------------------------------------------------------------------------

/// Return the TileType of the last argument (the pre-allocated output tile).
static TypePtr DeduceManualOutType(const std::vector<ExprPtr>& args,
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

// Validate that args[idx] is TileType.
static void CheckTileArg(const std::vector<ExprPtr>& args, size_t idx, const std::string& op_name) {
  CHECK(As<TileType>(args[idx]->GetType()))
      << "The operator " << op_name << " requires argument " << idx
      << " to be TileType, but got " << args[idx]->GetType()->TypeName();
}

// Validate that args[idx] is ScalarType.
static void CheckScalarArg(const std::vector<ExprPtr>& args, size_t idx, const std::string& op_name) {
  CHECK(As<ScalarType>(args[idx]->GetType()))
      << "The operator " << op_name << " requires argument " << idx
      << " to be ScalarType, but got " << args[idx]->GetType()->TypeName();
}

// Type deduction for (TileType, TileType, out:TileType) -> out.
static TypePtr DeduceManualBinaryTile(const std::vector<ExprPtr>& args,
                                      const std::vector<std::pair<std::string, std::any>>& kwargs,
                                      const std::string& op_name) {
  CHECK(args.size() == 3) << op_name << " requires 3 arguments (lhs, rhs, out)";
  CheckTileArg(args, 0, op_name);
  CheckTileArg(args, 1, op_name);
  return DeduceManualOutType(args, kwargs, op_name, 3);
}

// Type deduction for (TileType, ScalarType, out:TileType) -> out.
static TypePtr DeduceManualBinaryScalar(const std::vector<ExprPtr>& args,
                                        const std::vector<std::pair<std::string, std::any>>& kwargs,
                                        const std::string& op_name) {
  CHECK(args.size() == 3) << op_name << " requires 3 arguments (tile, scalar, out)";
  CheckTileArg(args, 0, op_name);
  CheckScalarArg(args, 1, op_name);
  return DeduceManualOutType(args, kwargs, op_name, 3);
}

// Type deduction for (TileType, out:TileType) -> out  (unary).
static TypePtr DeduceManualUnary(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs,
                                 const std::string& op_name) {
  CHECK(args.size() == 2) << op_name << " requires 2 arguments (src, out)";
  CheckTileArg(args, 0, op_name);
  return DeduceManualOutType(args, kwargs, op_name, 2);
}

// ---------------------------------------------------------------------------
// Tile x Tile binary operations
// ---------------------------------------------------------------------------

#define REGISTER_MANUAL_BINARY_TILE(name)                                                        \
  REGISTER_OP("manual." #name)                                                                   \
      .set_op_category("ManualOp")                                                               \
      .set_description("Manual element-wise " #name ": out = lhs " #name " rhs")                \
      .add_argument("lhs", "Left tile (TileType)")                                               \
      .add_argument("rhs", "Right tile (TileType)")                                              \
      .add_argument("out", "Pre-allocated output tile (TileType)")                               \
      .f_deduce_type([](const std::vector<ExprPtr>& args,                                        \
                        const std::vector<std::pair<std::string, std::any>>& kwargs) {           \
        return DeduceManualBinaryTile(args, kwargs, "manual." #name);                            \
      })

REGISTER_MANUAL_BINARY_TILE(add);
REGISTER_MANUAL_BINARY_TILE(sub);
REGISTER_MANUAL_BINARY_TILE(mul);
REGISTER_MANUAL_BINARY_TILE(div);
REGISTER_MANUAL_BINARY_TILE(rem);
REGISTER_MANUAL_BINARY_TILE(maximum);
REGISTER_MANUAL_BINARY_TILE(minimum);

// Bitwise tile-tile ops (integer only; validated at the Python layer).
REGISTER_MANUAL_BINARY_TILE(and);
REGISTER_MANUAL_BINARY_TILE(or);
REGISTER_MANUAL_BINARY_TILE(shl);
REGISTER_MANUAL_BINARY_TILE(shr);

#undef REGISTER_MANUAL_BINARY_TILE

// ---------------------------------------------------------------------------
// Tile x Scalar binary operations
// ---------------------------------------------------------------------------

#define REGISTER_MANUAL_BINARY_SCALAR(name)                                                      \
  REGISTER_OP("manual." #name)                                                                   \
      .set_op_category("ManualOp")                                                               \
      .set_description("Manual tile-scalar " #name ": out = tile " #name " scalar")             \
      .add_argument("tile", "Input tile (TileType)")                                             \
      .add_argument("scalar", "Scalar operand (ScalarType)")                                     \
      .add_argument("out", "Pre-allocated output tile (TileType)")                               \
      .f_deduce_type([](const std::vector<ExprPtr>& args,                                        \
                        const std::vector<std::pair<std::string, std::any>>& kwargs) {           \
        return DeduceManualBinaryScalar(args, kwargs, "manual." #name);                          \
      })

REGISTER_MANUAL_BINARY_SCALAR(adds);
REGISTER_MANUAL_BINARY_SCALAR(subs);
REGISTER_MANUAL_BINARY_SCALAR(muls);
REGISTER_MANUAL_BINARY_SCALAR(divs);
REGISTER_MANUAL_BINARY_SCALAR(rems);
REGISTER_MANUAL_BINARY_SCALAR(ands);
REGISTER_MANUAL_BINARY_SCALAR(ors);
REGISTER_MANUAL_BINARY_SCALAR(shls);
REGISTER_MANUAL_BINARY_SCALAR(shrs);
REGISTER_MANUAL_BINARY_SCALAR(maxs);
REGISTER_MANUAL_BINARY_SCALAR(mins);
REGISTER_MANUAL_BINARY_SCALAR(lrelu);

#undef REGISTER_MANUAL_BINARY_SCALAR

// ---------------------------------------------------------------------------
// Unary operations
// ---------------------------------------------------------------------------

#define REGISTER_MANUAL_UNARY(name)                                                              \
  REGISTER_OP("manual." #name)                                                                   \
      .set_op_category("ManualOp")                                                               \
      .set_description("Manual unary " #name ": out = " #name "(src)")                          \
      .add_argument("src", "Input tile (TileType)")                                              \
      .add_argument("out", "Pre-allocated output tile (TileType)")                               \
      .f_deduce_type([](const std::vector<ExprPtr>& args,                                        \
                        const std::vector<std::pair<std::string, std::any>>& kwargs) {           \
        return DeduceManualUnary(args, kwargs, "manual." #name);                                 \
      })

REGISTER_MANUAL_UNARY(neg);
REGISTER_MANUAL_UNARY(exp);
REGISTER_MANUAL_UNARY(sqrt);
REGISTER_MANUAL_UNARY(rsqrt);
REGISTER_MANUAL_UNARY(recip);
REGISTER_MANUAL_UNARY(log);
REGISTER_MANUAL_UNARY(abs);
REGISTER_MANUAL_UNARY(relu);
REGISTER_MANUAL_UNARY(not);

#undef REGISTER_MANUAL_UNARY

// manual.cast: (src_tile, out) -> out's type; carries target_type and mode attrs.
REGISTER_OP("manual.cast")
    .set_op_category("ManualOp")
    .set_description("Manual type-cast: out = cast(src, target_dtype, rounding_mode)")
    .add_argument("src", "Input tile (TileType)")
    .add_argument("out", "Pre-allocated output tile with target dtype (TileType)")
    .set_attr<DataType>("target_type")
    .set_attr<std::string>("mode")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualUnary(args, kwargs, "manual.cast");
    });

// ---------------------------------------------------------------------------
// Ternary / multi-input operations
// ---------------------------------------------------------------------------

// XOR with tmp buffer (tile, tile, tmp, out): 4 args.
REGISTER_OP("manual.xor")
    .set_op_category("ManualOp")
    .set_description("Manual bitwise XOR: out = lhs ^ rhs (integer tiles; tmp is scratch buffer)")
    .add_argument("lhs", "Left tile (TileType)")
    .add_argument("rhs", "Right tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutType(args, kwargs, "manual.xor", 4);
    });

// XOR-scalar with tmp buffer (tile, scalar, tmp, out): 4 args.
REGISTER_OP("manual.xors")
    .set_op_category("ManualOp")
    .set_description("Manual bitwise XOR with scalar: out = lhs ^ scalar (integer tiles)")
    .add_argument("lhs", "Input tile (TileType)")
    .add_argument("scalar", "Scalar operand (ScalarType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutType(args, kwargs, "manual.xors", 4);
    });

// prelu with tmp buffer (tile, slope, tmp, out): 4 args.
REGISTER_OP("manual.prelu")
    .set_op_category("ManualOp")
    .set_description("Manual parametric ReLU: out = prelu(tile, slope); tmp is scratch buffer")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("slope", "Slope tile (TileType)")
    .add_argument("tmp", "Scratch tile required by hardware (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutType(args, kwargs, "manual.prelu", 4);
    });

// Three-tile arithmetic (tile, tile, tile, out): 4 args.
REGISTER_OP("manual.addc")
    .set_op_category("ManualOp")
    .set_description("Manual three-tile add: out = lhs + rhs + rhs2")
    .add_argument("lhs", "First tile (TileType)")
    .add_argument("rhs", "Second tile (TileType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutType(args, kwargs, "manual.addc", 4);
    });

REGISTER_OP("manual.subc")
    .set_op_category("ManualOp")
    .set_description("Manual three-tile sub: out = lhs - rhs - rhs2")
    .add_argument("lhs", "First tile (TileType)")
    .add_argument("rhs", "Second tile (TileType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutType(args, kwargs, "manual.subc", 4);
    });

// (tile, scalar, tile, out): 4 args.
REGISTER_OP("manual.addsc")
    .set_op_category("ManualOp")
    .set_description("Manual tile+scalar+tile add: out = lhs + scalar + rhs2")
    .add_argument("lhs", "First tile (TileType)")
    .add_argument("scalar", "Scalar operand (ScalarType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutType(args, kwargs, "manual.addsc", 4);
    });

REGISTER_OP("manual.subsc")
    .set_op_category("ManualOp")
    .set_description("Manual tile-scalar-tile sub: out = lhs - scalar - rhs2")
    .add_argument("lhs", "First tile (TileType)")
    .add_argument("scalar", "Scalar operand (ScalarType)")
    .add_argument("rhs2", "Third tile (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutType(args, kwargs, "manual.subsc", 4);
    });

// Selection (mask, lhs, rhs, out): 4 args.
REGISTER_OP("manual.sel")
    .set_op_category("ManualOp")
    .set_description("Manual per-element selection: out[i]=lhs[i] if mask[i] else rhs[i]")
    .add_argument("mask", "Predicate mask tile (TileType)")
    .add_argument("lhs", "True-branch tile (TileType)")
    .add_argument("rhs", "False-branch tile (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutType(args, kwargs, "manual.sel", 4);
    });

// sels (lhs, rhs, scalar_mode, out): 4 args.
REGISTER_OP("manual.sels")
    .set_op_category("ManualOp")
    .set_description("Manual mode-based selection: out = sels(lhs, rhs, mode)")
    .add_argument("lhs", "First tile (TileType)")
    .add_argument("rhs", "Second tile (TileType)")
    .add_argument("select_mode", "Scalar mode (ScalarType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutType(args, kwargs, "manual.sels", 4);
    });

// ---------------------------------------------------------------------------
// Comparison operations
// ---------------------------------------------------------------------------

REGISTER_OP("manual.cmp")
    .set_op_category("ManualOp")
    .set_description("Manual element-wise tile comparison: out = (lhs cmp_op rhs)")
    .add_argument("lhs", "Left tile (TileType)")
    .add_argument("rhs", "Right tile (TileType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .set_attr<int>("cmp_type")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualBinaryTile(args, kwargs, "manual.cmp");
    });

REGISTER_OP("manual.cmps")
    .set_op_category("ManualOp")
    .set_description("Manual element-wise tile-scalar comparison: out = (tile cmp_op scalar)")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("scalar", "Scalar comparand (ScalarType)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .set_attr<int>("cmp_type")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualBinaryScalar(args, kwargs, "manual.cmps");
    });

// ---------------------------------------------------------------------------
// Scalar-to-tile broadcast
// ---------------------------------------------------------------------------

REGISTER_OP("manual.expands")
    .set_op_category("ManualOp")
    .set_description("Manual scalar broadcast: fill out tile with scalar value (out[i,j] = scalar)")
    .add_argument("scalar", "Fill value (ScalarType or constant)")
    .add_argument("out", "Pre-allocated output tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutType(args, kwargs, "manual.expands", 2);
    });

// ---------------------------------------------------------------------------
// Layout operations
// ---------------------------------------------------------------------------

// manual.reshape: (src_tile, shape_tuple, out) -> out's type.
REGISTER_OP("manual.reshape")
    .set_op_category("ManualOp")
    .set_description("Manual reshape: reinterpret src tile layout into out's shape")
    .add_argument("src", "Source tile (TileType)")
    .add_argument("shape", "New shape dimensions (MakeTuple)")
    .add_argument("out", "Pre-allocated output tile with target shape (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualOutType(args, kwargs, "manual.reshape", 3);
    });

// manual.transpose: (src_tile, out) -> out's type; axis attrs.
REGISTER_OP("manual.transpose")
    .set_op_category("ManualOp")
    .set_description("Manual transpose: swap two axes of src tile into out")
    .add_argument("src", "Source tile (TileType)")
    .add_argument("out", "Pre-allocated output tile with transposed shape (TileType)")
    .set_attr<int>("axis1")
    .set_attr<int>("axis2")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceManualUnary(args, kwargs, "manual.transpose");
    });

}  // namespace ir
}  // namespace pypto
