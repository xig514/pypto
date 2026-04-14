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
 * @file backend_910b_cce_manual_ops.cpp
 * @brief CCE backend op registration for manual (non-SSA) operations.
 *
 * Manual ops carry an explicit pre-allocated output tile as their last argument.
 * Convention: args = [input0, input1, ..., output_tile]
 * The helpers here extract output from args.back() instead of GetCurrentResultTarget().
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

#include "pypto/backend/910B_CCE/backend_910b_cce.h"
#include "pypto/backend/common/backend.h"
#include "pypto/codegen/cce/cce_codegen.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

// ============================================================================
// Helper: Manual binary op — args = [lhs, rhs, dst]
// Emits: OP(dst, lhs, rhs);
// ============================================================================
static std::string MakeManualBinaryCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                               codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << cce_op_name << ": expected 3 args (lhs, rhs, dst), got " << op->args_.size();
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit(cce_op_name + "(" + dst + ", " + lhs + ", " + rhs + ");");
  return "";
}

// ============================================================================
// Helper: Manual unary op — args = [src, dst]
// Emits: OP(dst, src);
// ============================================================================
static std::string MakeManualUnaryCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                              codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << cce_op_name << ": expected 2 args (src, dst), got " << op->args_.size();
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetExprAsCode(op->args_[1]);
  codegen.Emit(cce_op_name + "(" + dst + ", " + src + ");");
  return "";
}

// ============================================================================
// Helper: Manual scalar op — args = [tile, scalar, dst]
// Emits: OP(dst, tile, scalar);
// ============================================================================
static std::string MakeManualScalarCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                               codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << cce_op_name << ": expected 3 args (tile, scalar, dst), got "
                               << op->args_.size();
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string scalar = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit(cce_op_name + "(" + dst + ", " + tile + ", " + scalar + ");");
  return "";
}

// ============================================================================
// Helper: Manual reduction op — args = [tile, tmp, dst]
// Emits: OP(dst, tile, tmp);
// Used for TROWMAX, TROWSUM, TROWMIN (3-arg form).
// ============================================================================
static std::string MakeManualRowReductionCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                                     codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << cce_op_name << ": expected 3 args (tile, tmp, dst), got "
                               << op->args_.size();
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string tmp = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit(cce_op_name + "(" + dst + ", " + tile + ", " + tmp + ");");
  return "";
}

// ============================================================================
// Helper: Manual col-reduction op — args = [tile, tmp, dst]
// TCOLMAX emits: TCOLMAX(dst, tile);          — 2-arg form (no tmp)
// TCOLSUM emits: TCOLSUM(dst, tile, tmp, false); — 4-arg form
// ============================================================================
static std::string MakeManualColMaxCodegenCCE(const ir::CallPtr& op,
                                              codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "TCOLMAX: expected 3 args (tile, tmp, dst), got "
                               << op->args_.size();
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  // args_[1] is tmp — not used by TCOLMAX
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit("TCOLMAX(" + dst + ", " + tile + ");");
  return "";
}

static std::string MakeManualColSumCodegenCCE(const ir::CallPtr& op,
                                              codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "TCOLSUM: expected 3 args (tile, tmp, dst), got "
                               << op->args_.size();
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string tmp = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit("TCOLSUM(" + dst + ", " + tile + ", " + tmp + ", false);");
  return "";
}

// ============================================================================
// Helper: Manual row-expand op — args = [tile, reduction_tile, dst]
// Emits: OP(dst, tile, reduction_tile);
// ============================================================================
static std::string MakeManualRowExpandCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                                  codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << cce_op_name << ": expected 3 args (tile, red, dst), got "
                               << op->args_.size();
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string red = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit(cce_op_name + "(" + dst + ", " + tile + ", " + red + ");");
  return "";
}

// Helper function to compute stride-based offset
// In single-file mode: compute strides from IR tensor shape (no Tensor struct available)
// In normal mode: use Tensor struct strides (same as backend_910b_cce_ops.cpp)
static std::string ComputeManualOffset(codegen::CCECodegen& codegen, const std::string& tensor_var_name,
                                       const ir::MakeTuplePtr& offsets,
                                       const ir::TensorTypePtr& tensor_type) {
  if (codegen.IsSingleFileMode()) {
    return codegen.ComputeIRBasedOffset(tensor_type, offsets);
  }
  // Normal mode: use Tensor struct strides
  std::string tensor_struct = codegen.GetTensorStruct(tensor_var_name);
  std::ostringstream offset_computation;
  offset_computation << "(" << tensor_struct << "->start_offset";
  for (size_t i = 0; i < offsets->elements_.size(); ++i) {
    offset_computation << " + " << codegen.GetExprAsCode(offsets->elements_[i]) << " * " << tensor_struct
                       << "->strides[" << i << "]";
  }
  offset_computation << ")";
  return offset_computation.str();
}

// ============================================================================
// manual.load — args = [tensor, offsets, out_tile]
// Emits: TASSIGN(tensor_global, ptr + offset); TLOAD(out_tile, tensor_global);
// ============================================================================
static std::string MakeManualLoadCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "manual.load requires 3 arguments: tensor, offsets, out_tile";

  auto src_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
  CHECK(src_tensor_var_ptr != nullptr) << "manual.load source tensor must be a Var";

  auto offsets_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
  CHECK(offsets_tuple != nullptr) << "manual.load second argument must be a tuple (offsets)";

  auto out_tile = std::dynamic_pointer_cast<const ir::Var>(op->args_[2]);
  CHECK(out_tile != nullptr) << "manual.load third argument (out) must be a Var";

  std::string src_tensor_var = codegen.GetVarName(src_tensor_var_ptr);
  auto src_tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(src_tensor_var_ptr->GetType());
  CHECK(src_tensor_type != nullptr) << "manual.load source must be TensorType";

  std::string offset = ComputeManualOffset(codegen, src_tensor_var, offsets_tuple, src_tensor_type);
  std::string src_ptr = codegen.GetPointer(src_tensor_var);
  std::string out_name = codegen.GetExprAsCode(op->args_[2]);

  codegen.Emit("TASSIGN(" + src_tensor_var + ", " + src_ptr + " + " + offset + ");");
  codegen.Emit("TLOAD(" + out_name + ", " + src_tensor_var + ");");
  return "";
}

// ============================================================================
// manual.store — args = [tile, offsets, output_tensor]
// Emits: TASSIGN(tensor_global, ptr + offset); TSTORE(tensor_global, tile);
// ============================================================================
static std::string MakeManualStoreCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "manual.store requires 3 arguments: tile, offsets, output_tensor";

  std::string src_tile = codegen.GetExprAsCode(op->args_[0]);

  auto offsets_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
  CHECK(offsets_tuple != nullptr) << "manual.store second argument must be a tuple (offsets)";

  auto dst_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[2]);
  CHECK(dst_tensor_var_ptr != nullptr) << "manual.store destination tensor must be a Var";

  std::string dst_tensor_var = codegen.GetVarName(dst_tensor_var_ptr);
  auto dst_tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(dst_tensor_var_ptr->GetType());
  CHECK(dst_tensor_type != nullptr) << "manual.store destination must be TensorType";

  std::string offset = ComputeManualOffset(codegen, dst_tensor_var, offsets_tuple, dst_tensor_type);
  std::string dst_ptr = codegen.GetPointer(dst_tensor_var);

  codegen.Emit("TASSIGN(" + dst_tensor_var + ", " + dst_ptr + " + " + offset + ");");
  codegen.Emit("TSTORE(" + dst_tensor_var + ", " + src_tile + ");");
  return "";
}

// ============================================================================
// manual.make_tile — no-op (tile already declared in prologue)
// ============================================================================
static std::string MakeManualMakeTileCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  (void)op;
  (void)codegen_base;
  return "";
}

// ============================================================================
// manual.insert — args = [src, index_row, index_col, dst] or [src, index_row, index_col, offset, dst]
// Emits TINSERT(dst, src, indexRow, indexCol);
// With offset: Emits TASSIGN(dst, base + offset); TINSERT(...); TASSIGN(dst, base);
// ============================================================================
static std::string MakeManualInsertCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 4 || op->args_.size() == 5)
      << "manual.insert: expected 4 or 5 args, got " << op->args_.size();

  bool has_offset = (op->args_.size() == 5);
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string index_row = codegen.GetExprAsCode(op->args_[1]);
  std::string index_col = codegen.GetExprAsCode(op->args_[2]);
  std::string dst, offset, base_addr;

  if (has_offset) {
    offset = codegen.GetExprAsCode(op->args_[3]);
    dst = codegen.GetExprAsCode(op->args_[4]);
    base_addr = codegen.GetTileAddress(dst);
    codegen.Emit("TASSIGN(" + dst + ", " + base_addr + " + " + offset + ");");
  } else {
    dst = codegen.GetExprAsCode(op->args_[3]);
  }

  codegen.Emit("TINSERT(" + dst + ", " + src + ", " + index_row + ", " + index_col + ");");

  if (has_offset) {
    codegen.Emit("TASSIGN(" + dst + ", " + base_addr + ");");
  }

  return "";
}

// ============================================================================
// manual.move — args = [src, dst]
// Emits TMOV(dst, src);
// ============================================================================
static std::string MakeManualMoveCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2)
      << "manual.move: expected 2 args, got " << op->args_.size();

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetExprAsCode(op->args_[1]);

  // Emit TMOV (with or without AccToVecMode)
  if (op->HasKwarg("acc_to_vec_mode")) {
    const std::string& mode_str = op->GetKwarg<std::string>("acc_to_vec_mode");
    std::string mode_enum;
    if (mode_str == "single_vec0") {
      mode_enum = "AccToVecMode::SingleModeVec0";
    } else if (mode_str == "single_vec1") {
      mode_enum = "AccToVecMode::SingleModeVec1";
    } else if (mode_str == "dual_split_m") {
      mode_enum = "AccToVecMode::DualModeSplitM";
    } else if (mode_str == "dual_split_n") {
      mode_enum = "AccToVecMode::DualModeSplitN";
    } else {
      throw pypto::ValueError("Invalid acc_to_vec_mode: " + mode_str);
    }
    codegen.Emit("TMOV<decltype(" + dst + "), decltype(" + src + "), " + mode_enum + ">(" + dst + ", " + src + ");");
  } else {
    codegen.Emit("TMOV(" + dst + ", " + src + ");");
  }

  return "";
}

// ============================================================================
// manual.set_validshape — args = [row, col, tile]
// Emits: tile.SetValidShape(row, col);
// ============================================================================
static std::string MakeManualSetValidShapeCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "manual.set_validshape: expected 3 args (row, col, tile), got "
                               << op->args_.size();
  std::string row = codegen.GetExprAsCode(op->args_[0]);
  std::string col = codegen.GetExprAsCode(op->args_[1]);
  std::string tile = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit(tile + ".SetValidShape(" + row + ", " + col + ");");
  return "";
}

// ============================================================================
// manual.matmul — args = [left, right, dst]
// Emits: TMATMUL(dst, left, right);
// ============================================================================
static std::string MakeManualMatmulCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "manual.matmul: expected 3 args (left, right, dst), got "
                               << op->args_.size();
  std::string left = codegen.GetExprAsCode(op->args_[0]);
  std::string right = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit("TMATMUL(" + dst + ", " + left + ", " + right + ");");
  return "";
}

// manual.matmul_acc — args = [acc, left, right, dst]
static std::string MakeManualMatmulAccCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 4) << "manual.matmul_acc: expected 4 args, got " << op->args_.size();
  std::string acc = codegen.GetExprAsCode(op->args_[0]);
  std::string left = codegen.GetExprAsCode(op->args_[1]);
  std::string right = codegen.GetExprAsCode(op->args_[2]);
  std::string dst = codegen.GetExprAsCode(op->args_[3]);
  codegen.Emit("TMATMUL_ACC(" + dst + ", " + acc + ", " + left + ", " + right + ");");
  return "";
}

// manual.cast — args = [src, dst] + mode kwarg
static std::string MakeManualCastCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "manual.cast: expected 2 args (src, dst), got " << op->args_.size();
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetExprAsCode(op->args_[1]);

  // Get round mode from kwargs — manual.cast uses a string mode
  const std::string& mode_str = op->GetKwarg<std::string>("mode");
  static const std::vector<std::string> kModeNames = {"none", "rint",  "round", "floor",
                                                       "ceil", "trunc", "odd",   "cast_rint"};
  int mode_idx = -1;
  for (int i = 0; i < static_cast<int>(kModeNames.size()); ++i) {
    if (kModeNames[i] == mode_str) {
      mode_idx = i;
      break;
    }
  }
  CHECK(mode_idx >= 0) << "manual.cast: unknown round mode '" << mode_str << "'";

  codegen.Emit("TCVT(" + dst + ", " + src + ", " +
               codegen.GetTypeConverter().ConvertCastRoundMode(mode_idx) + ");");
  return "";
}

// manual.full/expands — args = [scalar, dst]
static std::string MakeManualExpandsCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "manual.full/expands: expected 2 args (scalar, dst), got "
                               << op->args_.size();
  std::string scalar = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetExprAsCode(op->args_[1]);
  codegen.Emit("TEXPANDS(" + dst + ", " + scalar + ");");
  return "";
}

// manual.fillpad — args = [src, dst]
static std::string MakeManualFillpadCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  return MakeManualUnaryCodegenCCE("TFILLPAD", op, codegen_base);
}

// manual.fillpad_expand — args = [src, dst]
static std::string MakeManualFillpadExpandCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  return MakeManualUnaryCodegenCCE("TFILLPAD_EXPAND", op, codegen_base);
}

// manual.reshape — args = [src, shape, dst]
static std::string MakeManualReshapeCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "manual.reshape: expected 3 args (src, shape, dst), got "
                               << op->args_.size();
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit("TRESHAPE(" + dst + ", " + src + ");");
  return "";
}

// manual.transpose — args = [src, dst]
static std::string MakeManualTransposeCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "manual.transpose: expected 2 args (src, dst), got " << op->args_.size();
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetExprAsCode(op->args_[1]);
  codegen.Emit("TTRANS(" + dst + ", " + src + ");");
  return "";
}

// manual.cmp — args = [lhs, rhs, dst] + cmp_type kwarg
static std::string MakeManualCmpCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "manual.cmp: expected 3 args (lhs, rhs, dst), got " << op->args_.size();
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  int cmp_type = op->GetKwarg<int>("cmp_type");
  codegen.Emit("TCMP(" + dst + ", " + lhs + ", " + rhs + ", " + std::to_string(cmp_type) + ");");
  return "";
}

// manual.cmps — args = [tile, scalar, dst] + cmp_type kwarg
static std::string MakeManualCmpsCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "manual.cmps: expected 3 args (tile, scalar, dst), got "
                               << op->args_.size();
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string scalar = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  int cmp_type = op->GetKwarg<int>("cmp_type");
  codegen.Emit("TCMPS(" + dst + ", " + tile + ", " + scalar + ", " + std::to_string(cmp_type) + ");");
  return "";
}

// manual.ub_copy — args = [src, dst] (Vec→Vec copy)
static std::string MakeManualUbCopyCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  return MakeManualMoveCodegenCCE(op, codegen_base);
}

// manual.col_expand — args = [src, dst]
static std::string MakeManualColExpandCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "manual.col_expand: expected 2 args (src, dst), got " << op->args_.size();
  // TCOLEXPAND is inplace: dst and src share same buffer
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetExprAsCode(op->args_[1]);
  codegen.Emit("TCOLEXPAND(" + dst + ", " + src + ");");
  return "";
}

// manual.row_expand — args = [src, dst]
static std::string MakeManualRowExpandUnaryCodegenCCE(const ir::CallPtr& op,
                                                       codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "manual.row_expand: expected 2 args (src, dst), got " << op->args_.size();
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetExprAsCode(op->args_[1]);
  codegen.Emit("TROWEXPAND(" + dst + ", " + src + ");");
  return "";
}

// ============================================================================
// Helper: Manual ternary op — args = [a, b, c, dst]
// Emits: OP(dst, a, b, c);
// ============================================================================
static std::string MakeManualTernaryCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                               codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 4) << cce_op_name << ": expected 4 args (a, b, c, dst), got " << op->args_.size();
  std::string a   = codegen.GetExprAsCode(op->args_[0]);
  std::string b   = codegen.GetExprAsCode(op->args_[1]);
  std::string c   = codegen.GetExprAsCode(op->args_[2]);
  std::string dst = codegen.GetExprAsCode(op->args_[3]);
  codegen.Emit(cce_op_name + "(" + dst + ", " + a + ", " + b + ", " + c + ");");
  return "";
}

// ============================================================================
// Helper: Manual quaternary op — args = [a, b, c, d, dst]
// Emits: OP(dst, a, b, c, d);
// ============================================================================
static std::string MakeManualQuaternaryCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                                  codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 5) << cce_op_name << ": expected 5 args (a, b, c, d, dst), got " << op->args_.size();
  std::string a   = codegen.GetExprAsCode(op->args_[0]);
  std::string b   = codegen.GetExprAsCode(op->args_[1]);
  std::string c   = codegen.GetExprAsCode(op->args_[2]);
  std::string d   = codegen.GetExprAsCode(op->args_[3]);
  std::string dst = codegen.GetExprAsCode(op->args_[4]);
  codegen.Emit(cce_op_name + "(" + dst + ", " + a + ", " + b + ", " + c + ", " + d + ");");
  return "";
}

// ============================================================================
// Helper: Binary + ReLU — args = [lhs, rhs, dst]
// Semantics: dst = max(0, OP(lhs, rhs))
// Emits: OP(lhs, lhs, rhs); TRELU(dst, lhs);
// ============================================================================
static std::string MakeManualBinaryReluCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                                   codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << cce_op_name << "+relu: expected 3 args (lhs, rhs, dst), got " << op->args_.size();
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit(cce_op_name + "(" + lhs + ", " + lhs + ", " + rhs + ");");
  codegen.Emit("TRELU(" + dst + ", " + lhs + ");");
  return "";
}

// ============================================================================
// Helper: Binary + ReLU + Cast — args = [lhs, rhs, dst] + mode kwarg
// Semantics: dst = cast(max(0, OP(lhs, rhs)), mode)
// Emits: OP(lhs, lhs, rhs); TRELU(lhs, lhs); TCVT(dst, lhs, mode);
// ============================================================================
static std::string MakeManualBinaryReluCastCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                                       codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << cce_op_name << "+relu+cast: expected 3 args (lhs, rhs, dst), got " << op->args_.size();
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  const std::string& mode_str = op->GetKwarg<std::string>("mode");
  static const std::vector<std::string> kModeNames = {"none", "rint", "round", "floor",
                                                       "ceil", "trunc", "odd", "cast_rint"};
  int mode_idx = -1;
  for (int i = 0; i < static_cast<int>(kModeNames.size()); ++i) {
    if (kModeNames[i] == mode_str) { mode_idx = i; break; }
  }
  CHECK(mode_idx >= 0) << cce_op_name << "+relu+cast: unknown round mode '" << mode_str << "'";
  codegen.Emit(cce_op_name + "(" + lhs + ", " + lhs + ", " + rhs + ");");
  codegen.Emit("TRELU(" + lhs + ", " + lhs + ");");
  codegen.Emit("TCVT(" + dst + ", " + lhs + ", " +
               codegen.GetTypeConverter().ConvertCastRoundMode(mode_idx) + ");");
  return "";
}

// ============================================================================
// Helper: Binary + Cast — args = [lhs, rhs, dst] + mode kwarg
// Semantics: dst = cast(OP(lhs, rhs), mode)
// Emits: OP(lhs, lhs, rhs); TCVT(dst, lhs, mode);
// ============================================================================
static std::string MakeManualBinaryCastCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                                   codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << cce_op_name << "+cast: expected 3 args (lhs, rhs, dst), got " << op->args_.size();
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  const std::string& mode_str = op->GetKwarg<std::string>("mode");
  static const std::vector<std::string> kModeNames = {"none", "rint", "round", "floor",
                                                       "ceil", "trunc", "odd", "cast_rint"};
  int mode_idx = -1;
  for (int i = 0; i < static_cast<int>(kModeNames.size()); ++i) {
    if (kModeNames[i] == mode_str) { mode_idx = i; break; }
  }
  CHECK(mode_idx >= 0) << cce_op_name << "+cast: unknown round mode '" << mode_str << "'";
  codegen.Emit(cce_op_name + "(" + lhs + ", " + lhs + ", " + rhs + ");");
  codegen.Emit("TCVT(" + dst + ", " + lhs + ", " +
               codegen.GetTypeConverter().ConvertCastRoundMode(mode_idx) + ");");
  return "";
}

// ============================================================================
// manual.mul_add_dst — args = [lhs, rhs, out]
// Semantics: out = (lhs * rhs) + out
// Emits: TMUL(lhs, lhs, rhs); TADD(out, lhs, out);
// ============================================================================
static std::string MakeManualMulAddDstCodegenCCE(const ir::CallPtr& op,
                                                  codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "manual.mul_add_dst: expected 3 args (lhs, rhs, out), got " << op->args_.size();
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string out = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit("TMUL(" + lhs + ", " + lhs + ", " + rhs + ");");
  codegen.Emit("TADD(" + out + ", " + lhs + ", " + out + ");");
  return "";
}

// ============================================================================
// manual.fused_mul_add — args = [lhs, rhs, out]
// Semantics: out = (lhs * out) + rhs
// Emits: TMUL(lhs, lhs, out); TADD(out, lhs, rhs);
// ============================================================================
static std::string MakeManualFusedMulAddCodegenCCE(const ir::CallPtr& op,
                                                    codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "manual.fused_mul_add: expected 3 args (lhs, rhs, out), got " << op->args_.size();
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string out = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit("TMUL(" + lhs + ", " + lhs + ", " + out + ");");
  codegen.Emit("TADD(" + out + ", " + lhs + ", " + rhs + ");");
  return "";
}

// ============================================================================
// manual.fused_mul_add_relu — args = [lhs, rhs, out]
// Semantics: out = max(0, (lhs * out) + rhs)
// Emits: TMUL(lhs, lhs, out); TADD(lhs, lhs, rhs); TRELU(out, lhs);
// ============================================================================
static std::string MakeManualFusedMulAddReluCodegenCCE(const ir::CallPtr& op,
                                                        codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "manual.fused_mul_add_relu: expected 3 args (lhs, rhs, out), got " << op->args_.size();
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string out = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit("TMUL(" + lhs + ", " + lhs + ", " + out + ");");
  codegen.Emit("TADD(" + lhs + ", " + lhs + ", " + rhs + ");");
  codegen.Emit("TRELU(" + out + ", " + lhs + ");");
  return "";
}

// ============================================================================
// manual.gather — args = [src, indices, out] or [src, indices, tmp, out]
// Emits: TGATHER(out, src, indices) or TGATHER(out, src, indices, tmp);
// ============================================================================
static std::string MakeManualGatherCodegenCCE(const ir::CallPtr& op,
                                               codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  if (op->args_.size() == 3) {
    std::string src     = codegen.GetExprAsCode(op->args_[0]);
    std::string indices = codegen.GetExprAsCode(op->args_[1]);
    std::string out     = codegen.GetExprAsCode(op->args_[2]);
    codegen.Emit("TGATHER(" + out + ", " + src + ", " + indices + ");");
  } else {
    CHECK(op->args_.size() == 4) << "manual.gather: expected 3 or 4 args, got " << op->args_.size();
    std::string src     = codegen.GetExprAsCode(op->args_[0]);
    std::string indices = codegen.GetExprAsCode(op->args_[1]);
    std::string tmp     = codegen.GetExprAsCode(op->args_[2]);
    std::string out     = codegen.GetExprAsCode(op->args_[3]);
    codegen.Emit("TGATHER(" + out + ", " + src + ", " + indices + ", " + tmp + ");");
  }
  return "";
}

// ============================================================================
// Op registrations — Memory
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.load")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualLoadCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.store")
    .set_pipe(ir::PipeType::MTE3)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualStoreCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.make_tile")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualMakeTileCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.set_validshape")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualSetValidShapeCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.move")
    .set_pipe(ir::PipeType::MTE1)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualMoveCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.insert")
    .set_pipe(ir::PipeType::MTE1)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualInsertCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.ub_copy")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUbCopyCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.full")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualExpandsCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.fillpad")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualFillpadCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.fillpad_expand")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualFillpadExpandCodegenCCE(op, codegen);
    });

// ============================================================================
// Op registrations — Tile x Tile binary
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.rem")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TREM", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.maximum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TMAX", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.minimum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TMIN", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.and")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TAND", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.or")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TOR", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.shl")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TSHL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.shr")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TSHR", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.add_relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryReluCodegenCCE("TADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.sub_relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryReluCodegenCCE("TSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.add_relu_cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryReluCastCodegenCCE("TADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.sub_relu_cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryReluCastCodegenCCE("TSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.mul_cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCastCodegenCCE("TMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.mul_add_dst")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualMulAddDstCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.fused_mul_add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualFusedMulAddCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.fused_mul_add_relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualFusedMulAddReluCodegenCCE(op, codegen);
    });

// ============================================================================
// Op registrations — Tile x Scalar binary
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.adds")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TADDS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.subs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TSUBS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.muls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TMULS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.divs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TDIVS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.rems")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TREMS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.ands")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TANDS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.ors")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TORS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.shls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TSHLS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.shrs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TSHRS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.maxs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TMAXS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.mins")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TMINS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.lrelu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TLRELU", op, codegen);
    });

// ============================================================================
// Op registrations — Unary
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.neg")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TNEG", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.exp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TEXP", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.sqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TSQRT", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.rsqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TRSQRT", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.recip")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TRECIP", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.log")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TLOG", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.abs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TABS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TRELU", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.not")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TNOT", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualCastCodegenCCE(op, codegen);
    });

// ============================================================================
// Op registrations — Comparison
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.cmp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualCmpCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.cmps")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualCmpsCodegenCCE(op, codegen);
    });

// ============================================================================
// Op registrations — Scalar broadcast
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.expands")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualExpandsCodegenCCE(op, codegen);
    });

// ============================================================================
// Op registrations — Reductions (tile, tmp, out)
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.row_sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowReductionCodegenCCE("TROWSUM", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.row_max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowReductionCodegenCCE("TROWMAX", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.row_min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowReductionCodegenCCE("TROWMIN", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.col_max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualColMaxCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.col_sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualColSumCodegenCCE(op, codegen);
    });

// ============================================================================
// Op registrations — Broadcast expansion
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.row_expand")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowExpandUnaryCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.row_expand_add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowExpandCodegenCCE("TROWEXPANDADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.row_expand_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowExpandCodegenCCE("TROWEXPANDSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.row_expand_mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowExpandCodegenCCE("TROWEXPANDMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.row_expand_div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowExpandCodegenCCE("TROWEXPANDDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.col_expand")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualColExpandCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.col_expand_mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowExpandCodegenCCE("TCOLEXPANDMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.col_expand_div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowExpandCodegenCCE("TCOLEXPANDDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.col_expand_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowExpandCodegenCCE("TCOLEXPANDSUB", op, codegen);
    });

// ============================================================================
// Op registrations — Matrix multiplication
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.matmul")
    .set_pipe(ir::PipeType::M)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualMatmulCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.matmul_acc")
    .set_pipe(ir::PipeType::M)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualMatmulAccCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.matmul_bias")
    .set_pipe(ir::PipeType::M)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualQuaternaryCodegenCCE("TMATMUL_BIAS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.gemv")
    .set_pipe(ir::PipeType::M)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TGEMV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.gemv_acc")
    .set_pipe(ir::PipeType::M)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryCodegenCCE("TGEMV_ACC", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.gemv_bias")
    .set_pipe(ir::PipeType::M)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryCodegenCCE("TGEMV_BIAS", op, codegen);
    });

// ============================================================================
// Op registrations — Layout operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.reshape")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualReshapeCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.transpose")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTransposeCodegenCCE(op, codegen);
    });

// ============================================================================
// Op registrations — Ternary / multi-input
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.xor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryCodegenCCE("TXOR", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.xors")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryCodegenCCE("TXORS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.prelu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryCodegenCCE("TPRELU", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.addc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryCodegenCCE("TADDC", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.subc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryCodegenCCE("TSUBC", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.addsc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryCodegenCCE("TADDSC", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.subsc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryCodegenCCE("TSUBSC", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.sel")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryCodegenCCE("TSEL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.sels")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryCodegenCCE("TSELS", op, codegen);
    });

// ============================================================================
// Op registrations — Gather / scatter
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.gather")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualGatherCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.gatherb")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TGATHERB", op, codegen);
    });

// ============================================================================
// Op registrations — Struct array (C++ struct + array for pl.struct lists)
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "struct.declare")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
      std::string arr_name = op->GetKwarg<std::string>("array");
      int size = op->GetKwarg<int>("size");
      std::string fields_csv = op->GetKwarg<std::string>("fields");

      // Deduplicate: same fields → same struct type
      std::string type_name = codegen.GetOrCreateStructType(fields_csv, arr_name);

      // Emit variable declaration only (type definition already emitted)
      if (size == 1) {
        codegen.Emit(type_name + " " + arr_name + " = {};");
      } else {
        codegen.Emit(type_name + " " + arr_name + "[" + std::to_string(size) + "] = {};");
      }
      return std::string("");
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "struct.get")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
      CHECK(op->args_.size() == 1) << "struct.get requires 1 arg (index)";
      std::string arr_name = op->GetKwarg<std::string>("array");
      std::string field = op->GetKwarg<std::string>("field");
      std::string ref = op->GetKwarg<std::string>("ref");
      // If a C++ reference exists, use it: "ref.field"
      if (!ref.empty()) {
        return ref + "." + field;
      }
      // For constant-0 index (single struct), emit "name.field" instead of "name[0].field"
      auto cint = ir::As<ir::ConstInt>(op->args_[0]);
      if (cint && cint->value_ == 0) {
        return arr_name + "." + field;
      }
      std::string idx = codegen.GetExprAsCode(op->args_[0]);
      return arr_name + "[" + idx + "]." + field;
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "struct.set")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
      CHECK(op->args_.size() == 2) << "struct.set requires 2 args (index, value)";
      std::string val = codegen.GetExprAsCode(op->args_[1]);
      std::string arr_name = op->GetKwarg<std::string>("array");
      std::string field = op->GetKwarg<std::string>("field");
      std::string ref = op->GetKwarg<std::string>("ref");
      // If a C++ reference exists, use it: "ref.field = val;"
      if (!ref.empty()) {
        codegen.Emit(ref + "." + field + " = " + val + ";");
        return std::string("");
      }
      // For constant-0 index (single struct), emit "name.field = val;"
      auto cint = ir::As<ir::ConstInt>(op->args_[0]);
      if (cint && cint->value_ == 0) {
        codegen.Emit(arr_name + "." + field + " = " + val + ";");
      } else {
        std::string idx = codegen.GetExprAsCode(op->args_[0]);
        codegen.Emit(arr_name + "[" + idx + "]." + field + " = " + val + ";");
      }
      return std::string("");
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "struct.ref")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
      CHECK(op->args_.size() == 1) << "struct.ref requires 1 arg (index)";
      std::string idx = codegen.GetExprAsCode(op->args_[0]);
      std::string arr_name = op->GetKwarg<std::string>("array");
      std::string var_name = op->GetKwarg<std::string>("var");
      codegen.Emit("auto& " + var_name + " = " + arr_name + "[" + idx + "];");
      return std::string("");
    });

// ============================================================================
// Op registrations — Sorting
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.sort32")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
      CHECK(op->args_.size() == 3 || op->args_.size() == 4)
          << "manual.sort32: expected 3 or 4 args (src, idx, dst[, tmp]), got "
          << op->args_.size();
      
      std::string src = codegen.GetExprAsCode(op->args_[0]);
      std::string idx = codegen.GetExprAsCode(op->args_[1]);
      std::string dst = codegen.GetExprAsCode(op->args_[2]);
      
      if (op->args_.size() == 4) {
        std::string tmp = codegen.GetExprAsCode(op->args_[3]);
        codegen.Emit("TSORT32(" + dst + ", " + src + ", " + idx + ", " + tmp + ");");
      } else {
        codegen.Emit("TSORT32(" + dst + ", " + src + ", " + idx + ");");
      }
      return "";
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.mrgsort")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
      CHECK(op->args_.size() == 2)
          << "manual.mrgsort: expected 2 args (src, dst), got "
          << op->args_.size();
      
      std::string src = codegen.GetExprAsCode(op->args_[0]);
      std::string dst = codegen.GetExprAsCode(op->args_[1]);
      int block_len = op->GetKwarg<int>("block_len");
      
      codegen.Emit("TMRGSORT(" + dst + ", " + src + ", " + std::to_string(block_len) + ");");
      return "";
    });

}  // namespace backend
}  // namespace pypto
