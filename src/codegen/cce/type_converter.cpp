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

#include "pypto/codegen/cce/type_converter.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {

namespace codegen {

namespace {

// Resolve a valid_shape ExprPtr to a string for use in the Tile<> type template.
// Only ConstInt values are supported (Var → compile-time unknown, skip).
// Returns "" if the expression cannot be resolved to a constant.
std::string ResolveValidShapeDim(const ir::ExprPtr& expr, int64_t fallback) {
  if (!expr) return std::to_string(fallback);
  if (auto c = ir::As<ir::ConstInt>(expr)) {
    return (c->value_ == -1) ? std::to_string(fallback) : std::to_string(c->value_);
  }
  // Var or other dynamic expression: cannot embed in template parameter.
  return "";
}

std::string ConvertTilePadToPTOValue(ir::TilePad pad) {
  switch (pad) {
    case ir::TilePad::null:
      return "";
    case ir::TilePad::zero:
      return "PadValue::Zero";
    case ir::TilePad::max:
      return "PadValue::Max";
    case ir::TilePad::min:
      return "PadValue::Min";
    default:
      throw pypto::ValueError("Invalid TilePad value");
  }
}

}  // namespace

std::string TypeConverter::ConvertTileType(const ir::TileTypePtr& tile_type, int64_t rows,
                                           int64_t cols) const {
  std::ostringstream type_alias;
  if (!tile_type->memref_.has_value()) {
    // No memref: IfStmt return variable or intermediate tile.
    // Still generate proper Tile type using tile_view if available.
    std::string tile_type_str = "TileType::Vec";
    std::string BLayout = "RowMajor";
    std::string SLayout = "NoneBox";
    std::string fractal = "512";
    std::string pad_value;
    if (cols == 1) {
      BLayout = "ColMajor";
    }
    // Compute type-template rows/cols: use valid_shape if present and constant.
    std::string type_rows = std::to_string(rows);
    std::string type_cols = std::to_string(cols);
    if (tile_type->tile_view_.has_value()) {
      const auto& tv = tile_type->tile_view_.value();
      BLayout = ConvertTileLayout(tv.blayout);
      SLayout = ConvertTileLayout(tv.slayout);
      fractal = std::to_string(tv.fractal);
      pad_value = ConvertTilePadToPTOValue(tv.pad);
      // Infer TileType from tile_view layout
      using TL = ir::TileLayout;
      if (tv.blayout == TL::col_major && tv.slayout == TL::row_major) tile_type_str = "TileType::Mat";
      else if (tv.blayout == TL::row_major && tv.slayout == TL::row_major) tile_type_str = "TileType::Left";
      else if (tv.blayout == TL::row_major && tv.slayout == TL::col_major) tile_type_str = "TileType::Right";
      // Override rows/cols from valid_shape if available.
      if (!tv.valid_shape.empty()) {
        std::string r = tv.valid_shape.size() >= 1 ? ResolveValidShapeDim(tv.valid_shape[0], rows) : "";
        std::string c = tv.valid_shape.size() >= 2 ? ResolveValidShapeDim(tv.valid_shape[1], cols) : "";
        if (!r.empty()) type_rows = r;
        if (!c.empty()) type_cols = c;
      }
    }
    type_alias << "Tile<" << tile_type_str << ", " << tile_type->dtype_.ToCTypeString() << ", " << rows
               << ", " << cols << ", BLayout::" << BLayout << ", " << type_rows << ", " << type_cols << ", SLayout::" << SLayout << ", "
               << fractal;
    if (!pad_value.empty()) {
      type_alias << ", " << pad_value;
    }
    type_alias << ">";
    return type_alias.str();
  }
  ir::MemorySpace space = (*tile_type->memref_)->memory_space_;  // NOLINT(bugprone-unchecked-optional-access)
  std::string tile_type_str = ConvertMemorySpaceToTileType(space);

  // TODO(YunjiQin): BLayout and SLayout should be determined by the tile format
  std::string BLayout = "RowMajor";
  std::string SLayout = "NoneBox";
  std::string fractal = "512";
  std::string pad_value;
  // Compute type-template rows/cols: use valid_shape if present and constant.
  std::string type_rows = std::to_string(rows);
  std::string type_cols = std::to_string(cols);

  if (cols == 1) {
    BLayout = "ColMajor";
  } else if (tile_type->tile_view_.has_value()) {
    const auto& tv = tile_type->tile_view_.value();
    BLayout = ConvertTileLayout(tv.blayout);
    SLayout = ConvertTileLayout(tv.slayout);
    fractal = std::to_string(tv.fractal);
    pad_value = ConvertTilePadToPTOValue(tv.pad);
  }
  // Override rows/cols from valid_shape if available (independent of cols==1 branch).
  if (tile_type->tile_view_.has_value()) {
    const auto& tv = tile_type->tile_view_.value();
    if (!tv.valid_shape.empty()) {
      std::string r = tv.valid_shape.size() >= 1 ? ResolveValidShapeDim(tv.valid_shape[0], rows) : "";
      std::string c = tv.valid_shape.size() >= 2 ? ResolveValidShapeDim(tv.valid_shape[1], cols) : "";
      if (!r.empty()) type_rows = r;
      if (!c.empty()) type_cols = c;
    }
  }
  type_alias << "Tile<" << tile_type_str << ", " << tile_type->dtype_.ToCTypeString() << ", " << rows << ", "
             << cols << ", BLayout::" << BLayout << ", " << type_rows << ", " << type_cols << ", SLayout::" << SLayout << ", " << fractal;
  if (!pad_value.empty()) {
    type_alias << ", " << pad_value;
  }
  type_alias << ">";

  return type_alias.str();
}

std::string TypeConverter::ConvertMemorySpaceToTileType(ir::MemorySpace space) const {
  switch (space) {
    case ir::MemorySpace::Left:
      return "TileType::Left";
    case ir::MemorySpace::Right:
      return "TileType::Right";
    case ir::MemorySpace::Acc:
      return "TileType::Acc";
    case ir::MemorySpace::Mat:
      return "TileType::Mat";
    case ir::MemorySpace::Vec:
      return "TileType::Vec";
    case ir::MemorySpace::DDR:
      // DDR is for GlobalTensor, not Tile - should not reach here
      throw pypto::ValueError("DDR is for GlobalTensor, not Tile");
    default:
      throw pypto::ValueError("Invalid MemorySpace value");
  }
}

std::string TypeConverter::ConvertPipeType(ir::PipeType pipe) const {
  switch (pipe) {
    case ir::PipeType::MTE1:
      return "PIPE_MTE1";
    case ir::PipeType::MTE2:
      return "PIPE_MTE2";
    case ir::PipeType::MTE3:
      return "PIPE_MTE3";
    case ir::PipeType::M:
      return "PIPE_M";
    case ir::PipeType::V:
      return "PIPE_V";
    case ir::PipeType::S:
      return "PIPE_S";
    case ir::PipeType::FIX:
      return "PIPE_FIX";
    case ir::PipeType::ALL:
      return "PIPE_ALL";
    default:
      throw pypto::ValueError("Invalid PipeType value");
  }
}

std::string TypeConverter::ConvertEventId(int event_id) const {
  CHECK(event_id >= 0 && event_id <= 7) << "Event ID must be in range [0, 7], got " << event_id;
  return "EVENT_ID" + std::to_string(event_id);
}

std::string TypeConverter::ConvertCastRoundMode(int mode) const {
  switch (mode) {
    case 0:
      return "RoundMode::CAST_NONE";
    case 1:
      return "RoundMode::CAST_RINT";
    case 2:
      return "RoundMode::CAST_ROUND";
    case 3:
      return "RoundMode::CAST_FLOOR";
    case 4:
      return "RoundMode::CAST_CEIL";
    case 5:
      return "RoundMode::CAST_TRUNC";
    case 6:
      return "RoundMode::CAST_ODD";
    default:
      throw pypto::ValueError("Cast round mode must be in range [0, 6], got " + std::to_string(mode));
  }
}

std::string TypeConverter::ConvertTileLayout(ir::TileLayout layout) const {
  switch (layout) {
    case ir::TileLayout::none_box:
      return "NoneBox";
    case ir::TileLayout::row_major:
      return "RowMajor";
    case ir::TileLayout::col_major:
      return "ColMajor";
    default:
      throw pypto::ValueError("Invalid TileLayout value");
  }
}

std::string TypeConverter::GenerateShapeType(const std::vector<int64_t>& dims) const {
  CHECK(!dims.empty()) << "Cannot generate Shape type for empty dimensions";

  std::ostringstream oss;
  oss << "pto::Shape<";

  // Pad to 5 dimensions with leading 1s
  const size_t target_dims = 5;
  CHECK(dims.size() <= target_dims) << "Cannot generate Shape with more than " << target_dims
                                    << " dimensions, got " << dims.size();

  // Add leading 1s for padding
  for (size_t i = 0; i < target_dims - dims.size(); ++i) {
    oss << "1, ";
  }

  // Add actual dimensions
  for (size_t i = 0; i < dims.size(); ++i) {
    oss << dims[i];
    if (i < dims.size() - 1) {
      oss << ", ";
    }
  }

  oss << ">";
  return oss.str();
}

std::string TypeConverter::GenerateStrideType(const std::vector<int64_t>& shape) const {
  CHECK(!shape.empty()) << "Cannot generate Stride type for empty shape";

  std::ostringstream oss;
  oss << "pto::Stride<";

  // Pad to 5 dimensions with leading 1s
  const size_t target_dims = 5;
  CHECK(shape.size() <= target_dims) << "Cannot generate Stride with more than " << target_dims
                                     << " dimensions, got " << shape.size();

  // Add leading 1s for padding
  for (size_t i = 0; i < target_dims - shape.size(); ++i) {
    oss << "1, ";
  }

  // set dynamic strides, will get from runtime
  for (size_t i = 0; i < shape.size(); ++i) {
    oss << "-1";
    if (i < shape.size() - 1) {
      oss << ", ";
    }
  }

  oss << ">";
  return oss.str();
}

}  // namespace codegen

}  // namespace pypto
