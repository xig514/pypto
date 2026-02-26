# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO Manual (non-SSA) Language module.

Provides an imperative, non-SSA tile programming interface.  The user explicitly
allocates output tile buffers with :func:`create_tile` and then calls operations
that write results into those pre-allocated buffers.  All compute operations
accept the output tile as their last argument and return ``None``.

Example::

    import pypto.language.manual as pm
    from pypto.pypto_core import DataType
    from pypto.pypto_core.ir import MemorySpace

    # Allocate tiles
    a   = pm.create_tile([64, 64], DataType.FP32)
    b   = pm.create_tile([64, 64], DataType.FP32)
    out = pm.create_tile([64, 64], DataType.FP32)

    # Load inputs
    pm.load(tensor_a, [0, 0], [64, 64], a)
    pm.load(tensor_b, [0, 0], [64, 64], b)

    # Compute
    pm.add(a, b, out)

    # Store result
    pm.store(out, [0, 0], [64, 64], result_tensor)
"""

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import MemorySpace

from ..typing import DynVar, Scalar, Tensor, Tile, dynamic  # noqa: E402
from .op.manual_ops import (
    abs,
    add,
    addc,
    addsc,
    adds,
    and_,
    ands,
    cast,
    cmps,
    cmp,
    col_expand,
    col_expand_div,
    col_expand_mul,
    col_expand_sub,
    create_tile,
    div,
    divs,
    exp,
    expands,
    fillpad,
    full,
    gemv,
    gemv_acc,
    gemv_bias,
    get_block_idx,
    l0c_store,
    lrelu,
    load,
    log,
    matmul,
    matmul_acc,
    matmul_bias,
    maximum,
    maxs,
    minimum,
    mins,
    move,
    mul,
    muls,
    neg,
    not_,
    or_,
    ors,
    prelu,
    recip,
    relu,
    rem,
    rems,
    reshape,
    row_expand,
    row_expand_add,
    row_expand_div,
    row_expand_mul,
    row_expand_sub,
    row_max,
    row_min,
    row_sum,
    rsqrt,
    sel,
    sels,
    shl,
    shls,
    shr,
    shrs,
    sqrt,
    store,
    sub,
    subc,
    subsc,
    subs,
    transpose,
    ub_copy,
    xor,
    xors,
)

# Re-export DataType constants for convenience
FP4 = DataType.FP4
FP8E4M3FN = DataType.FP8E4M3FN
FP8E5M2 = DataType.FP8E5M2
FP16 = DataType.FP16
FP32 = DataType.FP32
BF16 = DataType.BF16
HF4 = DataType.HF4
HF8 = DataType.HF8
INT4 = DataType.INT4
INT8 = DataType.INT8
INT16 = DataType.INT16
INT32 = DataType.INT32
INT64 = DataType.INT64
UINT4 = DataType.UINT4
UINT8 = DataType.UINT8
UINT16 = DataType.UINT16
UINT32 = DataType.UINT32
UINT64 = DataType.UINT64
BOOL = DataType.BOOL
INDEX = DataType.INDEX

__all__ = [
    # Types
    "Tensor", "Tile", "Scalar", "DynVar", "dynamic",
    "MemorySpace",
    # Allocation
    "create_tile",
    # Memory
    "load", "store", "l0c_store", "move", "ub_copy", "full", "fillpad", "get_block_idx",
    # Tile x Tile binary
    "add", "sub", "mul", "div", "rem", "maximum", "minimum",
    "and_", "or_", "shl", "shr",
    # Tile x Scalar binary
    "adds", "subs", "muls", "divs", "rems",
    "ands", "ors", "shls", "shrs",
    "maxs", "mins", "lrelu",
    # Unary
    "neg", "exp", "sqrt", "rsqrt", "recip", "log", "abs", "relu", "not_", "cast",
    # Ternary
    "xor", "xors", "prelu", "addc", "subc", "addsc", "subsc", "sel", "sels",
    # Comparison
    "cmp", "cmps",
    # Reduction
    "row_max", "row_sum", "row_min",
    # Broadcast
    "row_expand", "row_expand_add", "row_expand_sub", "row_expand_mul", "row_expand_div",
    "col_expand", "col_expand_mul", "col_expand_div", "col_expand_sub",
    "expands",
    # Matrix
    "matmul", "matmul_acc", "matmul_bias",
    "gemv", "gemv_acc", "gemv_bias",
    # Layout
    "reshape", "transpose",
    # DataType constants
    "FP4", "FP8E4M3FN", "FP8E5M2", "FP16", "FP32", "BF16", "HF4", "HF8",
    "INT4", "INT8", "INT16", "INT32", "INT64",
    "UINT4", "UINT8", "UINT16", "UINT32", "UINT64",
    "BOOL", "INDEX",
]
