# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Op-to-pipe and op-tile-access metadata tables."""

from __future__ import annotations

from pypto.pypto_core.ir import MemorySpace, PipeType

from .data_structures import TileAccessPattern

# ---------------------------------------------------------------------------
# Op-to-pipe mapping
# ---------------------------------------------------------------------------

_OP_TO_PIPE: dict[str, PipeType] = {
    # Memory
    "load": PipeType.MTE2,
    "load_tile": PipeType.MTE2,
    # "store" and "store_tile" are handled by get_store_pipe() — see below
    # Matrix
    "matmul": PipeType.M,
    "matmul_acc": PipeType.M,
    "matmul_bias": PipeType.M,
    "gemv": PipeType.M,
    "gemv_acc": PipeType.M,
    "gemv_bias": PipeType.M,
    # Vector – unary
    "neg": PipeType.V,
    "exp": PipeType.V,
    "sqrt": PipeType.V,
    "rsqrt": PipeType.V,
    "recip": PipeType.V,
    "log": PipeType.V,
    "abs": PipeType.V,
    "relu": PipeType.V,
    "not_": PipeType.V,
    "cast": PipeType.V,
    # Vector – binary tile × tile
    "add": PipeType.V,
    "sub": PipeType.V,
    "mul": PipeType.V,
    "div": PipeType.V,
    "rem": PipeType.V,
    "maximum": PipeType.V,
    "minimum": PipeType.V,
    "and_": PipeType.V,
    "or_": PipeType.V,
    "shl": PipeType.V,
    "shr": PipeType.V,
    # Vector – binary tile × scalar
    "adds": PipeType.V,
    "subs": PipeType.V,
    "muls": PipeType.V,
    "divs": PipeType.V,
    "rems": PipeType.V,
    "ands": PipeType.V,
    "ors": PipeType.V,
    "shls": PipeType.V,
    "shrs": PipeType.V,
    "maxs": PipeType.V,
    "mins": PipeType.V,
    "lrelu": PipeType.V,
    # Vector – ternary / multi-input
    "xor": PipeType.V,
    "xors": PipeType.V,
    "prelu": PipeType.V,
    "addc": PipeType.V,
    "subc": PipeType.V,
    "addsc": PipeType.V,
    "subsc": PipeType.V,
    "sel": PipeType.V,
    "sels": PipeType.V,
    # Vector – comparison
    "cmp": PipeType.V,
    "cmps": PipeType.V,
    # Vector – reduction
    "row_max": PipeType.V,
    "row_sum": PipeType.V,
    "col_max": PipeType.V,
    "col_sum": PipeType.V,
    "row_min": PipeType.V,
    # Vector – broadcast
    "row_expand": PipeType.V,
    "row_expand_add": PipeType.V,
    "row_expand_sub": PipeType.V,
    "row_expand_mul": PipeType.V,
    "row_expand_div": PipeType.V,
    "col_expand": PipeType.V,
    "col_expand_mul": PipeType.V,
    "col_expand_div": PipeType.V,
    "col_expand_sub": PipeType.V,
    "expands": PipeType.V,
    # Vector – layout / memory
    "reshape": PipeType.V,
    "transpose": PipeType.V,
    "ub_copy": PipeType.V,
    "full": PipeType.V,
    "fillpad": PipeType.V,
    "fillpad_expand": PipeType.V,
    # "move" is handled specially via get_move_pipe()
}


def get_move_pipe(
    src_memory: MemorySpace | None,
    target_memory: MemorySpace | None,
) -> PipeType:
    """Determine the pipeline for a ``move`` operation.

    Rules from hardware documentation:
    - Mat → Left / Right  →  MTE1
    - Mat → Vec           →  V
    - Vec → Mat           →  FIX
    - Mat → S / other     →  FIX
    """
    if src_memory == MemorySpace.Mat:
        if target_memory in (MemorySpace.Left, MemorySpace.Right):
            return PipeType.MTE1
        if target_memory == MemorySpace.Vec:
            return PipeType.V
        return PipeType.FIX
    if src_memory == MemorySpace.Vec and target_memory == MemorySpace.Mat:
        return PipeType.FIX
    return PipeType.V  # conservative fallback


def get_store_pipe(src_memory: MemorySpace | None) -> PipeType:
    """Determine the pipeline for a ``store`` / ``store_tile`` operation.

    Rules from PTOAS:
    - Store from Acc (L0C) → PIPE_FIX  (TSTORE_ACC)
    - Store from Vec (UB)  → PIPE_MTE3 (TSTORE_VEC)
    - Unknown              → PIPE_MTE3 (conservative default, most common)
    """
    if src_memory == MemorySpace.Acc:
        return PipeType.FIX
    return PipeType.MTE3


# ---------------------------------------------------------------------------
# Op tile-access mapping  (DSL arg indices — arg0 is output for generic ops)
# ---------------------------------------------------------------------------

_OP_TILE_ACCESS: dict[str, TileAccessPattern] = {
    # Memory  – load(out, tensor, offsets, ...)  [ir_op.manual handler]
    "load":      TileAccessPattern([], [0]),
    "load_tile": TileAccessPattern([], [0]),
    # store(tensor, tile, offsets, ...)  [ir_op.manual handler]
    "store":      TileAccessPattern([1], []),
    "store_tile": TileAccessPattern([1], []),
    # Binary tile × tile  – DSL: op(out, lhs, rhs)
    "add":     TileAccessPattern([1, 2], [0]),
    "sub":     TileAccessPattern([1, 2], [0]),
    "mul":     TileAccessPattern([1, 2], [0]),
    "div":     TileAccessPattern([1, 2], [0]),
    "rem":     TileAccessPattern([1, 2], [0]),
    "maximum": TileAccessPattern([1, 2], [0]),
    "minimum": TileAccessPattern([1, 2], [0]),
    "and_":    TileAccessPattern([1, 2], [0]),
    "or_":     TileAccessPattern([1, 2], [0]),
    "shl":     TileAccessPattern([1, 2], [0]),
    "shr":     TileAccessPattern([1, 2], [0]),
    # Binary tile × scalar  – DSL: op(out, tile, scalar)
    "adds":  TileAccessPattern([1], [0]),
    "subs":  TileAccessPattern([1], [0]),
    "muls":  TileAccessPattern([1], [0]),
    "divs":  TileAccessPattern([1], [0]),
    "rems":  TileAccessPattern([1], [0]),
    "ands":  TileAccessPattern([1], [0]),
    "ors":   TileAccessPattern([1], [0]),
    "shls":  TileAccessPattern([1], [0]),
    "shrs":  TileAccessPattern([1], [0]),
    "maxs":  TileAccessPattern([1], [0]),
    "mins":  TileAccessPattern([1], [0]),
    "lrelu": TileAccessPattern([1], [0]),
    # Unary  – DSL: op(out, tile)
    "neg":   TileAccessPattern([1], [0]),
    "exp":   TileAccessPattern([1], [0]),
    "sqrt":  TileAccessPattern([1], [0]),
    "rsqrt": TileAccessPattern([1], [0]),
    "recip": TileAccessPattern([1], [0]),
    "log":   TileAccessPattern([1], [0]),
    "abs":   TileAccessPattern([1], [0]),
    "relu":  TileAccessPattern([1], [0]),
    "not_":  TileAccessPattern([1], [0]),
    # cast  – DSL: cast(out, tile, target_type)
    "cast":  TileAccessPattern([1], [0]),
    # Ternary  – DSL: xor(out, lhs, rhs, tmp)
    "xor":   TileAccessPattern([1, 2], [0, 3]),
    "xors":  TileAccessPattern([1],    [0, 3]),
    "prelu": TileAccessPattern([1, 2], [0, 3]),
    # DSL: addc(out, lhs, rhs, rhs2)
    "addc":  TileAccessPattern([1, 2, 3], [0]),
    "subc":  TileAccessPattern([1, 2, 3], [0]),
    # DSL: addsc(out, lhs, scalar, rhs2)
    "addsc": TileAccessPattern([1, 3], [0]),
    "subsc": TileAccessPattern([1, 3], [0]),
    # DSL: sel(out, mask, lhs, rhs)
    "sel":   TileAccessPattern([1, 2, 3], [0]),
    # DSL: sels(out, lhs, rhs, mode)
    "sels":  TileAccessPattern([1, 2], [0]),
    # Comparison  – DSL: cmp(out, lhs, rhs)
    "cmp":   TileAccessPattern([1, 2], [0]),
    "cmps":  TileAccessPattern([1],    [0]),
    # Reduction  – DSL: row_max(out, tile, tmp)
    "row_max": TileAccessPattern([1], [0, 2]),
    "row_sum": TileAccessPattern([1], [0, 2]),
    "row_min": TileAccessPattern([1], [0, 2]),
    "col_max": TileAccessPattern([1], [0, 2]),
    "col_sum": TileAccessPattern([1], [0, 2]),
    # Broadcast  – DSL: row_expand(out, src)
    "row_expand":     TileAccessPattern([1],    [0]),
    "row_expand_add": TileAccessPattern([1, 2], [0]),
    "row_expand_sub": TileAccessPattern([1, 2], [0]),
    "row_expand_mul": TileAccessPattern([1, 2], [0]),
    "row_expand_div": TileAccessPattern([1, 2], [0]),
    "col_expand":     TileAccessPattern([1],    [0]),
    "col_expand_mul": TileAccessPattern([1, 2], [0]),
    "col_expand_div": TileAccessPattern([1, 2], [0]),
    "col_expand_sub": TileAccessPattern([1, 2], [0]),
    "expands":        TileAccessPattern([],     [0]),
    # Matrix  – DSL: matmul(out, lhs, rhs)
    "matmul":      TileAccessPattern([1, 2],    [0]),
    "matmul_acc":  TileAccessPattern([1, 2, 3], [0]),
    "matmul_bias": TileAccessPattern([1, 2, 3], [0]),
    "gemv":        TileAccessPattern([1, 2],    [0]),
    "gemv_acc":    TileAccessPattern([1, 2, 3], [0]),
    "gemv_bias":   TileAccessPattern([1, 2, 3], [0]),
    # Layout  – DSL: reshape(out, tile, shape)
    "reshape":   TileAccessPattern([1], [0]),
    # DSL: transpose(out, tile, axis1, axis2)
    "transpose": TileAccessPattern([1], [0]),
    # Memory helpers  – DSL: move(out, tile, ...)
    "move":    TileAccessPattern([1], [0]),
    "ub_copy": TileAccessPattern([1], [0]),
    "full":    TileAccessPattern([],  [0]),
    "fillpad": TileAccessPattern([1], [0]),
    "fillpad_expand": TileAccessPattern([1], [0]),
}
