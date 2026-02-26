# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Manual (non-SSA) block operations for PyPTO Language DSL.

Unlike the SSA-style block_ops where operations return new Tile values, all
operations here take a pre-allocated output Tile as the last argument and return
None.  The output Tile's internal SSA binding is updated in-place so that
subsequent uses of the same Tile object see the latest written value.

Typical usage::

    import pypto.language.manual as pm

    out = pm.create_tile([64, 64], pm.FP32)
    a   = pm.create_tile([64, 64], pm.FP32)
    b   = pm.create_tile([64, 64], pm.FP32)

    pm.load(tensor_a, [0, 0], [64, 64], a)
    pm.load(tensor_b, [0, 0], [64, 64], b)
    pm.add(a, b, out)
    pm.store(out, [0, 0], [64, 64], result_tensor)
"""

from collections.abc import Sequence
from typing import Literal, Optional, Sequence, Union

from pypto.ir.op import block_ops as _ir_block_ops
from pypto.ir.utils import _to_make_tuple
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Expr, MemorySpace, Span

from ...typing import Scalar, Tensor, Tile

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _span() -> Span:
    return Span.unknown()


def _op(name: str, ins: list, out: Tile, **kwargs) -> None:
    """Create a manual IR op and rebind *out* to the resulting SSA value.

    Args:
        name: The IR op name (e.g. ``"manual.add"``).
        ins: List of input IR expressions (already unwrapped).
        out: The pre-allocated output Tile; its ``_expr`` is rebound.
        **kwargs: Op keyword attributes forwarded to ``create_op_call``.
    """
    out._expr = _ir_core.create_op_call(name, ins + [out.unwrap()], kwargs, _span())


# ---------------------------------------------------------------------------
# Allocation / creation
# ---------------------------------------------------------------------------
def create_tile(
    shape: list[int],
    dtype: DataType,
    target_memory: MemorySpace = MemorySpace.UB,
    addr: Optional[Union[int, Expr]] = None,
    size: Optional[int] = None,
) -> Tile:
    """Allocate a tile buffer.

    This is identical to the SSA ``block.create_tile`` op.  The returned Tile
    must subsequently be passed as the ``out`` argument to load/compute ops.

    Args:
        shape: Tile shape.
        dtype: Element data type.
        target_memory: Memory space for the tile (default UB).
        addr: Optional memory address (int or Expr)
        size: Optional memory size in bytes

    Returns:
        Tile wrapping the allocation expression.
    """
    return Tile(expr=_ir_block_ops.create_tile(shape, dtype, target_memory, addr, size))


# ---------------------------------------------------------------------------
# Memory operations
# ---------------------------------------------------------------------------

def load(
    tensor: Tensor,
    offsets: Sequence[int | Expr],
    shapes: Sequence[int | Expr],
    out: Tile
) -> None:
    """Load data from a global tensor into a pre-allocated tile.

    Args:
        tensor: Source global tensor.
        offsets: Per-dimension offsets into the tensor.
        shapes: Number of elements to load in each dimension.
        out: Pre-allocated destination tile; rebound on return.
    """
    _op(
        "manual.load",
        [tensor.unwrap(), _to_make_tuple(offsets), _to_make_tuple(shapes)],
        out
    )


def store(
    tile: Tile,
    offsets: Sequence[int | Expr],
    shapes: Sequence[int | Expr],
    output_tensor: Tensor,
) -> Tensor:
    """Store data from a tile back to a global tensor.

    This operation is identical to the SSA ``block.store``; the output is the
    updated tensor expression rather than a tile, so no explicit ``out`` tile
    is needed.

    Args:
        tile: Source tile.
        offsets: Per-dimension offsets into the output tensor.
        shapes: Shape of the region to store.
        output_tensor: Destination tensor.

    Returns:
        Tensor wrapping the store result.
    """
    return Tensor(expr=_ir_block_ops.store(tile.unwrap(), offsets, shapes, output_tensor.unwrap()))


def l0c_store(
    tile: Tile,
    offsets: Sequence[int | Expr],
    shapes: Sequence[int | Expr],
    output_tensor: Tensor,
) -> Tensor:
    """Store from an L0C tile to a global tensor.

    Args:
        tile: Source L0C tile.
        offsets: Per-dimension offsets.
        shapes: Region shape.
        output_tensor: Destination tensor.

    Returns:
        Tensor wrapping the l0c_store result.
    """
    return Tensor(expr=_ir_block_ops.l0c_store(tile.unwrap(), offsets, shapes, output_tensor.unwrap()))


def move(tile: Tile, target_memory: MemorySpace, out: Tile, transpose: bool = False) -> None:
    """Move a tile between memory levels, writing into a pre-allocated buffer.

    Args:
        tile: Source tile.
        target_memory: Destination memory space.
        out: Pre-allocated output tile; rebound on return.
        transpose: Whether to transpose while moving.
    """
    _op("manual.move", [tile.unwrap()], out, target_memory=target_memory, transpose=transpose)


def ub_copy(tile: Tile, out: Tile) -> None:
    """Copy a tile within UB memory into a pre-allocated buffer.

    Args:
        tile: Source UB tile.
        out: Pre-allocated destination UB tile; rebound on return.
    """
    _op("manual.ub_copy", [tile.unwrap()], out)


def full(value: int | float | Expr | Scalar, out: Tile) -> None:
    """Fill a pre-allocated tile with a scalar value.

    Args:
        value: Fill value (scalar constant or Scalar expression).
        out: Pre-allocated tile to fill; rebound on return.
    """
    val_expr = value.unwrap() if isinstance(value, Scalar) else value
    _op("manual.full", [val_expr], out)


def fillpad(tile: Tile, out: Tile) -> None:
    """Fill a pre-allocated tile with padding from *tile*.

    Args:
        tile: Source tile providing the data.
        out: Pre-allocated destination tile; rebound on return.
    """
    _op("manual.fillpad", [tile.unwrap()], out)


def get_block_idx() -> Scalar:
    """Return the current block index (unchanged from SSA style).

    Returns:
        Scalar wrapping the block index (UINT64 type).
    """
    from pypto.language.op.block_ops import get_block_idx as _get_block_idx
    return _get_block_idx()


# ---------------------------------------------------------------------------
# Element-wise Tile x Tile binary operations
# ---------------------------------------------------------------------------

def add(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Element-wise addition: out = lhs + rhs."""
    _op("manual.add", [lhs.unwrap(), rhs.unwrap()], out)


def sub(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Element-wise subtraction: out = lhs - rhs."""
    _op("manual.sub", [lhs.unwrap(), rhs.unwrap()], out)


def mul(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Element-wise multiplication: out = lhs * rhs."""
    _op("manual.mul", [lhs.unwrap(), rhs.unwrap()], out)


def div(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Element-wise division: out = lhs / rhs."""
    _op("manual.div", [lhs.unwrap(), rhs.unwrap()], out)


def rem(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Element-wise remainder: out = lhs % rhs."""
    _op("manual.rem", [lhs.unwrap(), rhs.unwrap()], out)


def maximum(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Element-wise maximum: out = max(lhs, rhs)."""
    _op("manual.maximum", [lhs.unwrap(), rhs.unwrap()], out)


def minimum(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Element-wise minimum: out = min(lhs, rhs)."""
    _op("manual.minimum", [lhs.unwrap(), rhs.unwrap()], out)


def and_(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Element-wise bitwise AND: out = lhs & rhs (integer tiles)."""
    _op("manual.and", [lhs.unwrap(), rhs.unwrap()], out)


def or_(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Element-wise bitwise OR: out = lhs | rhs (integer tiles)."""
    _op("manual.or", [lhs.unwrap(), rhs.unwrap()], out)


def shl(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Element-wise left shift: out = lhs << rhs (integer tiles)."""
    _op("manual.shl", [lhs.unwrap(), rhs.unwrap()], out)


def shr(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Element-wise right shift: out = lhs >> rhs (integer tiles)."""
    _op("manual.shr", [lhs.unwrap(), rhs.unwrap()], out)


# ---------------------------------------------------------------------------
# Element-wise Tile x Scalar binary operations
# ---------------------------------------------------------------------------

def _scalar_expr(v: int | float | Expr | Scalar) -> Expr:
    return v.unwrap() if isinstance(v, Scalar) else v


def adds(lhs: Tile, rhs: int | float | Expr | Scalar, out: Tile) -> None:
    """Element-wise tile + scalar: out = lhs + rhs."""
    _op("manual.adds", [lhs.unwrap(), _scalar_expr(rhs)], out)


def subs(lhs: Tile, rhs: int | float | Expr | Scalar, out: Tile) -> None:
    """Element-wise tile - scalar: out = lhs - rhs."""
    _op("manual.subs", [lhs.unwrap(), _scalar_expr(rhs)], out)


def muls(lhs: Tile, rhs: int | float | Expr | Scalar, out: Tile) -> None:
    """Element-wise tile * scalar: out = lhs * rhs."""
    _op("manual.muls", [lhs.unwrap(), _scalar_expr(rhs)], out)


def divs(lhs: Tile, rhs: int | float | Expr | Scalar, out: Tile) -> None:
    """Element-wise tile / scalar: out = lhs / rhs."""
    _op("manual.divs", [lhs.unwrap(), _scalar_expr(rhs)], out)


def rems(lhs: Tile, rhs: int | float | Expr | Scalar, out: Tile) -> None:
    """Element-wise tile % scalar: out = lhs % rhs."""
    _op("manual.rems", [lhs.unwrap(), _scalar_expr(rhs)], out)


def ands(lhs: Tile, rhs: int | Expr | Scalar, out: Tile) -> None:
    """Element-wise tile & scalar (integer): out = lhs & rhs."""
    _op("manual.ands", [lhs.unwrap(), _scalar_expr(rhs)], out)


def ors(lhs: Tile, rhs: int | Expr | Scalar, out: Tile) -> None:
    """Element-wise tile | scalar (integer): out = lhs | rhs."""
    _op("manual.ors", [lhs.unwrap(), _scalar_expr(rhs)], out)


def shls(lhs: Tile, rhs: int | Expr | Scalar, out: Tile) -> None:
    """Element-wise tile << scalar (integer): out = lhs << rhs."""
    _op("manual.shls", [lhs.unwrap(), _scalar_expr(rhs)], out)


def shrs(lhs: Tile, rhs: int | Expr | Scalar, out: Tile) -> None:
    """Element-wise tile >> scalar (integer): out = lhs >> rhs."""
    _op("manual.shrs", [lhs.unwrap(), _scalar_expr(rhs)], out)


def maxs(lhs: Tile, rhs: int | float | Expr | Scalar, out: Tile) -> None:
    """Element-wise max(tile, scalar): out = max(lhs, rhs)."""
    _op("manual.maxs", [lhs.unwrap(), _scalar_expr(rhs)], out)


def mins(lhs: Tile, rhs: int | float | Expr | Scalar, out: Tile) -> None:
    """Element-wise min(tile, scalar): out = min(lhs, rhs)."""
    _op("manual.mins", [lhs.unwrap(), _scalar_expr(rhs)], out)


def lrelu(tile: Tile, slope: int | float | Expr | Scalar, out: Tile) -> None:
    """Leaky ReLU with scalar slope: out = max(tile, slope * tile)."""
    _op("manual.lrelu", [tile.unwrap(), _scalar_expr(slope)], out)


# ---------------------------------------------------------------------------
# Unary operations
# ---------------------------------------------------------------------------

def neg(tile: Tile, out: Tile) -> None:
    """Element-wise negation: out = -tile."""
    _op("manual.neg", [tile.unwrap()], out)


def exp(tile: Tile, out: Tile) -> None:
    """Element-wise exponential: out = exp(tile)."""
    _op("manual.exp", [tile.unwrap()], out)


def sqrt(tile: Tile, out: Tile) -> None:
    """Element-wise square root: out = sqrt(tile)."""
    _op("manual.sqrt", [tile.unwrap()], out)


def rsqrt(tile: Tile, out: Tile) -> None:
    """Element-wise reciprocal square root: out = 1 / sqrt(tile)."""
    _op("manual.rsqrt", [tile.unwrap()], out)


def recip(tile: Tile, out: Tile) -> None:
    """Element-wise reciprocal: out = 1 / tile."""
    _op("manual.recip", [tile.unwrap()], out)


def log(tile: Tile, out: Tile) -> None:
    """Element-wise natural logarithm: out = log(tile)."""
    _op("manual.log", [tile.unwrap()], out)


def abs(tile: Tile, out: Tile) -> None:
    """Element-wise absolute value: out = |tile|."""
    _op("manual.abs", [tile.unwrap()], out)


def relu(tile: Tile, out: Tile) -> None:
    """Element-wise ReLU: out = max(0, tile)."""
    _op("manual.relu", [tile.unwrap()], out)


def not_(tile: Tile, out: Tile) -> None:
    """Element-wise bitwise NOT (integer tiles): out = ~tile."""
    _op("manual.not", [tile.unwrap()], out)


def cast(
    tile: Tile,
    target_type: int | DataType,
    out: Tile,
    mode: Literal["none", "rint", "round", "floor", "ceil", "trunc", "odd"] = "round",
) -> None:
    """Cast tile elements to *target_type*: out = cast(tile, dtype, mode).

    Args:
        tile: Source tile.
        target_type: Target DataType.
        out: Pre-allocated output tile with the desired result dtype.
        mode: Rounding mode string.
    """
    _op("manual.cast", [tile.unwrap()], out, target_type=target_type, mode=mode)


# ---------------------------------------------------------------------------
# Ternary / multi-input operations
# ---------------------------------------------------------------------------

def xor(lhs: Tile, rhs: Tile, tmp: Tile, out: Tile) -> None:
    """Element-wise XOR: out = lhs ^ rhs (integer tiles; *tmp* is a scratch buffer)."""
    _op("manual.xor", [lhs.unwrap(), rhs.unwrap(), tmp.unwrap()], out)


def xors(lhs: Tile, rhs: int | Expr | Scalar, tmp: Tile, out: Tile) -> None:
    """Element-wise XOR with scalar: out = lhs ^ rhs (integer tiles)."""
    _op("manual.xors", [lhs.unwrap(), _scalar_expr(rhs), tmp.unwrap()], out)


def prelu(tile: Tile, slope: Tile, tmp: Tile, out: Tile) -> None:
    """Parametric ReLU: out = prelu(tile, slope)."""
    _op("manual.prelu", [tile.unwrap(), slope.unwrap(), tmp.unwrap()], out)


def addc(lhs: Tile, rhs: Tile, rhs2: Tile, out: Tile) -> None:
    """Three-tile addition: out = lhs + rhs + rhs2."""
    _op("manual.addc", [lhs.unwrap(), rhs.unwrap(), rhs2.unwrap()], out)


def subc(lhs: Tile, rhs: Tile, rhs2: Tile, out: Tile) -> None:
    """Three-tile subtraction: out = lhs - rhs - rhs2."""
    _op("manual.subc", [lhs.unwrap(), rhs.unwrap(), rhs2.unwrap()], out)


def addsc(lhs: Tile, rhs: int | float | Expr | Scalar, rhs2: Tile, out: Tile) -> None:
    """Tile + scalar + tile: out = lhs + rhs + rhs2."""
    _op("manual.addsc", [lhs.unwrap(), _scalar_expr(rhs), rhs2.unwrap()], out)


def subsc(lhs: Tile, rhs: int | float | Expr | Scalar, rhs2: Tile, out: Tile) -> None:
    """Tile - scalar - tile: out = lhs - rhs - rhs2."""
    _op("manual.subsc", [lhs.unwrap(), _scalar_expr(rhs), rhs2.unwrap()], out)


def sel(mask: Tile, lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Per-element selection: out[i] = lhs[i] if mask[i] else rhs[i]."""
    _op("manual.sel", [mask.unwrap(), lhs.unwrap(), rhs.unwrap()], out)


def sels(lhs: Tile, rhs: Tile, select_mode: int | float | Expr | Scalar, out: Tile) -> None:
    """Mode-based selection: out = sels(lhs, rhs, mode)."""
    _op("manual.sels", [lhs.unwrap(), rhs.unwrap(), _scalar_expr(select_mode)], out)


# ---------------------------------------------------------------------------
# Comparison operations
# ---------------------------------------------------------------------------

def cmp(lhs: Tile, rhs: Tile, out: Tile, cmp_type: int = 0) -> None:
    """Element-wise tile comparison.

    Args:
        lhs: Left tile.
        rhs: Right tile.
        out: Pre-allocated output tile; rebound on return.
        cmp_type: EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5.
    """
    _op("manual.cmp", [lhs.unwrap(), rhs.unwrap()], out, cmp_type=cmp_type)


def cmps(lhs: Tile, rhs: int | float | Expr | Scalar, out: Tile, cmp_type: int = 0) -> None:
    """Element-wise tile-scalar comparison.

    Args:
        lhs: Left tile.
        rhs: Scalar comparand.
        out: Pre-allocated output tile; rebound on return.
        cmp_type: EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5.
    """
    _op("manual.cmps", [lhs.unwrap(), _scalar_expr(rhs)], out, cmp_type=cmp_type)


# ---------------------------------------------------------------------------
# Reduction operations (require a temporary tile)
# ---------------------------------------------------------------------------

def row_max(tile: Tile, tmp: Tile, out: Tile) -> None:
    """Row-wise max reduction: out[i, 0] = max_j(tile[i, j])."""
    _op("manual.row_max", [tile.unwrap(), tmp.unwrap()], out)


def row_sum(tile: Tile, tmp: Tile, out: Tile) -> None:
    """Row-wise sum reduction: out[i, 0] = sum_j(tile[i, j])."""
    _op("manual.row_sum", [tile.unwrap(), tmp.unwrap()], out)


def row_min(tile: Tile, tmp: Tile, out: Tile) -> None:
    """Row-wise min reduction: out[i, 0] = min_j(tile[i, j])."""
    _op("manual.row_min", [tile.unwrap(), tmp.unwrap()], out)


# ---------------------------------------------------------------------------
# Broadcast / expansion operations
# ---------------------------------------------------------------------------

def row_expand(src: Tile, out: Tile) -> None:
    """Row broadcast: out[i, j] = src[i, 0] for all j."""
    _op("manual.row_expand", [src.unwrap()], out)


def row_expand_add(tile: Tile, row_vec: Tile, out: Tile) -> None:
    """Broadcast row vector and add: out = tile + broadcast(row_vec)."""
    _op("manual.row_expand_add", [tile.unwrap(), row_vec.unwrap()], out)


def row_expand_sub(tile: Tile, row_vec: Tile, out: Tile) -> None:
    """Broadcast row vector and subtract: out = tile - broadcast(row_vec)."""
    _op("manual.row_expand_sub", [tile.unwrap(), row_vec.unwrap()], out)


def row_expand_mul(tile: Tile, row_vec: Tile, out: Tile) -> None:
    """Broadcast row vector and multiply: out = tile * broadcast(row_vec)."""
    _op("manual.row_expand_mul", [tile.unwrap(), row_vec.unwrap()], out)


def row_expand_div(tile: Tile, row_vec: Tile, out: Tile) -> None:
    """Broadcast row vector and divide: out = tile / broadcast(row_vec)."""
    _op("manual.row_expand_div", [tile.unwrap(), row_vec.unwrap()], out)


def col_expand(col_vec: Tile, out: Tile) -> None:
    """Column broadcast: out[i, j] = col_vec[0, j] for all i."""
    _op("manual.col_expand", [col_vec.unwrap()], out)


def col_expand_mul(tile: Tile, col_vec: Tile, out: Tile) -> None:
    """Broadcast column vector and multiply: out = tile * broadcast(col_vec)."""
    _op("manual.col_expand_mul", [tile.unwrap(), col_vec.unwrap()], out)


def col_expand_div(tile: Tile, col_vec: Tile, out: Tile) -> None:
    """Broadcast column vector and divide: out = tile / broadcast(col_vec)."""
    _op("manual.col_expand_div", [tile.unwrap(), col_vec.unwrap()], out)


def col_expand_sub(tile: Tile, col_vec: Tile, out: Tile) -> None:
    """Broadcast column vector and subtract: out = tile - broadcast(col_vec)."""
    _op("manual.col_expand_sub", [tile.unwrap(), col_vec.unwrap()], out)


def expands(scalar: int | float | Expr | Scalar, out: Tile) -> None:
    """Expand a scalar to fill the output tile: out[i, j] = scalar."""
    _op("manual.expands", [_scalar_expr(scalar)], out)


# ---------------------------------------------------------------------------
# Matrix multiplication operations
# ---------------------------------------------------------------------------

def matmul(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Matrix multiplication: out = lhs @ rhs."""
    _op("manual.matmul", [lhs.unwrap(), rhs.unwrap()], out)


def matmul_acc(acc: Tile, lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Matrix multiplication with accumulation: out = acc + lhs @ rhs."""
    _op("manual.matmul_acc", [acc.unwrap(), lhs.unwrap(), rhs.unwrap()], out)


def matmul_bias(lhs: Tile, rhs: Tile, bias: Tile, out: Tile) -> None:
    """Matrix multiplication with bias: out = lhs @ rhs + bias."""
    _op("manual.matmul_bias", [lhs.unwrap(), rhs.unwrap(), bias.unwrap()], out)


def gemv(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """General matrix-vector multiply: out[1,N] = lhs[1,K] @ rhs[K,N]."""
    _op("manual.gemv", [lhs.unwrap(), rhs.unwrap()], out)


def gemv_acc(acc: Tile, lhs: Tile, rhs: Tile, out: Tile) -> None:
    """GEMV with accumulation: out += lhs @ rhs."""
    _op("manual.gemv_acc", [acc.unwrap(), lhs.unwrap(), rhs.unwrap()], out)


def gemv_bias(lhs: Tile, rhs: Tile, bias: Tile, out: Tile) -> None:
    """GEMV with bias: out = lhs @ rhs + bias."""
    _op("manual.gemv_bias", [lhs.unwrap(), rhs.unwrap(), bias.unwrap()], out)


# ---------------------------------------------------------------------------
# Layout operations
# ---------------------------------------------------------------------------

def reshape(tile: Tile, shape: list[int | Expr], out: Tile) -> None:
    """Reshape *tile* to *shape*, writing into *out*.

    Args:
        tile: Source tile.
        shape: Target shape dimensions.
        out: Pre-allocated output tile with the target shape; rebound on return.
    """
    shape_tuple = _to_make_tuple(shape)
    _op("manual.reshape", [tile.unwrap(), shape_tuple], out)


def transpose(tile: Tile, axis1: int, axis2: int, out: Tile) -> None:
    """Transpose *tile* by swapping *axis1* and *axis2*, writing into *out*.

    Args:
        tile: Source tile.
        axis1: First axis index.
        axis2: Second axis index.
        out: Pre-allocated transposed output tile; rebound on return.
    """
    _op("manual.transpose", [tile.unwrap()], out, axis1=axis1, axis2=axis2)


__all__ = [
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
]
