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

    import pypto.language.op.manual as pm

    tile_ty = pm.TileType(shape=[64, 64], dtype=pm.FP32)
    a = pm.make_tile(tile_ty, addr=0x0000, size=16384)
    b = pm.make_tile(tile_ty, addr=0x4000, size=16384)
    out = pm.make_tile(tile_ty, addr=0x8000, size=16384)

    pm.load(a, tensor_a, [0, 0])
    pm.load(b, tensor_b, [0, 0])
    pm.add(a, b, out)
    pm.store(result_tensor, out, [0, 0])
"""

from collections.abc import Sequence
from typing import Literal, Optional, Sequence, Union

from dataclasses import dataclass
from pypto.ir.op import block_ops as _ir_block_ops
from pypto.ir.op import manual_ops as _ir_manual
from pypto.ir.utils import _to_make_tuple
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Expr, MemorySpace, Span

from ....typing import Scalar, Tensor, Tile


# ---------------------------------------------------------------------------
# TileType descriptor
# ---------------------------------------------------------------------------

@dataclass
class TileType:
    """Tile type descriptor containing shape, dtype, and TileView parameters.

    This class encapsulates all the type information for a tile, which can then
    be used to create an actual tile with memory allocation via make_tile().

    Args:
        shape: Tile shape dimensions.
        dtype: Element data type.
        target_memory: Memory space for the tile (default Vec).
        valid_shape: Valid shape dimensions (optional).
        blayout: Block layout (0=none_box, 1=row_major, 2=col_major, optional).
            Auto-filled per memory space if omitted.
        slayout: Scatter layout (0=none_box, 1=row_major, 2=col_major, optional).
            Auto-filled per memory space if omitted.
        fractal: Fractal size (optional). Auto-filled: 512 for FP16, 1024 for FP32 ACC.
        pad: Pad mode (TilePad.null/zero/max/min or integer encoding, optional).
        compact: Compact mode (0=null, 1=normal, 2=row_plus_one, optional).
    """
    shape: Sequence[int] | _ir_core.MakeTuple
    dtype: DataType
    target_memory: MemorySpace = MemorySpace.Vec
    valid_shape: Optional[Sequence[int]] = None
    blayout: Optional[int] = None
    slayout: Optional[int] = None
    fractal: Optional[int] = None
    pad: Optional[int | _ir_core.TilePad] = None
    compact: Optional[int] = None

    def __post_init__(self):
        self.pad = _normalize_tile_pad(self.pad)
        _apply_default_layout(self)


# Hardware-required layouts per memory space for Cube matmul.
# {MemorySpace: (blayout, slayout)}
_REQUIRED_LAYOUTS: dict[MemorySpace, tuple[int, int]] = {
    MemorySpace.Mat:   (2, 1),  # NZ format: col_major block, row_major scatter (default)
    MemorySpace.Left:  (1, 1),  # row_major block, row_major scatter (a3/a2); a5 also supports (2, 1)
    MemorySpace.Right: (1, 2),  # row_major block, col_major scatter
    MemorySpace.Acc:   (2, 1),  # NZ format: col_major block, row_major scatter
}

# a5 Left also supports col_major block layout
_LEFT_A5_LAYOUT: tuple[int, int] = (2, 1)

# MAT also supports DN layout (row_major block, col_major scatter) for DN TLOAD.
_MAT_DN_LAYOUT: tuple[int, int] = (1, 2)

_LAYOUT_NAMES = {0: "none_box", 1: "row_major", 2: "col_major"}
_PAD_VALUES = {
    _ir_core.TilePad.null: 0,
    _ir_core.TilePad.zero: 1,
    _ir_core.TilePad.max: 2,
    _ir_core.TilePad.min: 3,
}


def _normalize_tile_pad(pad: int | _ir_core.TilePad | None) -> int | None:
    if pad is None:
        return None
    if pad in _PAD_VALUES:
        return _PAD_VALUES[pad]
    if isinstance(pad, int):
        if pad not in _PAD_VALUES.values():
            raise ValueError("TileType.pad must be one of TilePad.null/zero/max/min")
        return pad
    raise TypeError("TileType.pad must be a TilePad or integer 0/1/2/3")


def _apply_default_layout(tt: "TileType") -> None:
    """Auto-fill and validate blayout/slayout/fractal for Cube memory spaces."""
    required = _REQUIRED_LAYOUTS.get(tt.target_memory)
    if required is None:
        return  # Vec or other spaces: no constraints

    req_b, req_s = required
    space_name = tt.target_memory.name

    # Auto-fill if not specified
    if tt.blayout is None:
        tt.blayout = req_b
    if tt.slayout is None:
        tt.slayout = req_s

    # Validate against hardware requirements
    actual = (tt.blayout, tt.slayout)
    if tt.target_memory == MemorySpace.Mat:
        # MAT supports both ND (2,1) and DN (1,2) layouts
        if actual != required and actual != _MAT_DN_LAYOUT:
            raise ValueError(
                f"{space_name} tiles require blayout/slayout={required} (ND) or "
                f"{_MAT_DN_LAYOUT} (DN), got ({tt.blayout}, {tt.slayout})"
            )
    else:
        if tt.target_memory == MemorySpace.Left:
            # Left supports (1,1) on a3/a2 and (2,1) on a5
            if actual != required and actual != _LEFT_A5_LAYOUT:
                raise ValueError(
                    f"{space_name} tiles require blayout/slayout={required} (a3/a2) or "
                    f"{_LEFT_A5_LAYOUT} (a5), got ({tt.blayout}, {tt.slayout})"
                )
        else:
            if tt.blayout != req_b:
                raise ValueError(
                    f"{space_name} tiles require blayout={req_b} ({_LAYOUT_NAMES[req_b]}), "
                    f"got blayout={tt.blayout} ({_LAYOUT_NAMES.get(tt.blayout, '?')})"
                )
            if tt.slayout != req_s:
                raise ValueError(
                    f"{space_name} tiles require slayout={req_s} ({_LAYOUT_NAMES[req_s]}), "
                    f"got slayout={tt.slayout} ({_LAYOUT_NAMES.get(tt.slayout, '?')})"
            )

    # Auto-fill fractal for FP32 ACC
    if tt.target_memory == MemorySpace.Acc and tt.fractal is None:
        if tt.dtype in (DataType.FP32, DataType.INT32):
            tt.fractal = 1024
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
def make_tile(
    tile_type: TileType,
    addr: int | Expr,  
    size: int, 
) -> Tile:
    """Allocate a tile buffer.

    This is identical to the SSA ``block.make_tile`` op.  The returned Tile
    must subsequently be passed as the ``out`` argument to load/compute ops.

    Args:
        tile_type: Tile type descriptor containing shape, dtype, and TileView params.
        addr: Memory address (int or Expr). Required for manual mode.
        size: Memory size in bytes. Required for manual mode.

    Returns:
        Tile wrapping the allocation expression.
    """
    return Tile(expr=_ir_block_ops.make_tile(shape=tile_type.shape,
        dtype=tile_type.dtype,
        target_memory=tile_type.target_memory,
        addr=addr,
        size=size,
        valid_shape=tile_type.valid_shape,
        blayout=tile_type.blayout,
        slayout=tile_type.slayout,
        fractal=tile_type.fractal,
        pad=tile_type.pad,
        compact=tile_type.compact))


# ---------------------------------------------------------------------------
# Memory operations
# ---------------------------------------------------------------------------

def _check_nd_load_bounds(out: "Tile", tensor: "Tensor", tile_dims: "list[int] | None" = None) -> None:
    """Validate that tile dims do not exceed tensor dims for ND loads.

    Compares static (ConstInt) shape dimensions between the tile and the
    corresponding tensor dimensions.  For an M-D tile on an N-D tensor (N >= M),
    tile dimensions are aligned with the **last M** tensor dimensions by default,
    or with the dimensions specified by ``tile_dims``.

    Raises ValueError if a tile dimension exceeds the corresponding tensor
    dimension, which would cause an out-of-bounds partition_view.
    """
    tile_type = out.unwrap().type
    tensor_type = tensor.unwrap().type
    tile_shape = getattr(tile_type, "shape", None)
    tensor_shape = getattr(tensor_type, "shape", None)
    if tile_shape is None or tensor_shape is None:
        return
    tile_ndim = len(tile_shape)
    tensor_ndim = len(tensor_shape)
    if tile_dims is None:
        tile_dims = list(range(tensor_ndim - tile_ndim, tensor_ndim))
    for tile_d, tensor_d in enumerate(tile_dims):
        if tensor_d >= tensor_ndim or tile_d >= tile_ndim:
            continue
        t_val = getattr(tile_shape[tile_d], "value", None)
        s_val = getattr(tensor_shape[tensor_d], "value", None)
        if t_val is not None and s_val is not None and t_val > s_val:
            raise ValueError(
                f"manual.load: tile dimension {tile_d} ({t_val}) exceeds tensor "
                f"dimension {tensor_d} ({s_val}). If the tensor needs transposing, "
                f'use layout="dn".'
            )


def load(
    out: Tile,
    tensor: Tensor,
    offsets: Sequence[int | Expr],
    layout: str | None = None,
) -> None:
    """Load data from a global tensor into a pre-allocated tile.

    Args:
        out: Pre-allocated destination tile; rebound on return.
        tensor: Source global tensor.
        offsets: Per-dimension offsets into the tensor.
        layout: Tensor memory layout. ``"dn"`` for column-major (DN) layout,
            which lets TLOAD transpose on-chip. Default is row-major (ND).
    """
    if layout != "dn":
        _check_nd_load_bounds(out, tensor)
    kwargs: dict = {}
    if layout is not None:
        kwargs["layout"] = layout
    out._expr = _ir_core.create_op_call(
        "manual.load",
        [tensor.unwrap(), _to_make_tuple(offsets), out.unwrap()],
        kwargs,
        _span(),
    )


def load_tile(
    out: Tile,
    tensor: Tensor,
    tile_offsets: Sequence[int | Expr],
    layout: str | None = None,
    tile_dims: list[int] | None = None,
) -> None:
    """Load data from a global tensor into a pre-allocated tile using tile-relative offsets.

    For N-D tensor with M-D tile (N >= M):
    - First (N-M) offsets are used directly (batch, head indices, etc.)
    - Last M offsets are multiplied by tile shape

    Example for 4D tensor [B, N, S, D] with 2D tile [TS, TD]:
    - ``load_tile(tile, tensor, [b, n, s_tile, d_tile])``
    - ``b`` and ``n`` are used directly
    - ``s_tile * TS`` and ``d_tile * TD`` are computed for the last two dims

    Example for BSND layout [B, S, N, D] with 2D tile [TS, TD]:
    - ``load_tile(tile, tensor, [b, s_tile, n, d_tile], tile_dims=[1, 3])``
    - ``b`` and ``n`` are used directly
    - ``s_tile * TS`` and ``d_tile * TD`` are computed for dimensions 1 and 3

    Args:
        out: Pre-allocated destination tile; rebound on return.
        tensor: Source global tensor.
        tile_offsets: Per-dimension offsets. First (N-M) are direct offsets,
            last M are tile-relative offsets.
        layout: Tensor memory layout. ``"dn"`` for column-major (DN) layout,
            which lets TLOAD transpose on-chip. Default is row-major (ND).
        tile_dims: Optional list of dimension indices that tile corresponds to.
            If None, defaults to last M dimensions (BNSD layout).
            For BSND layout [B, S, N, D] with tile [TS, TD], use tile_dims=[1, 3].
    """
    out._expr = _ir_manual.load_tile(out.unwrap(), tensor.unwrap(), _to_make_tuple(tile_offsets), layout=layout, tile_dims=tile_dims)


def store(
    output_tensor: Tensor,
    tile: Tile,
    offsets: Sequence[int | Expr],
) -> Tensor:
    """Store data from a tile back to a global tensor.

    Args:
        output_tensor: Destination tensor.
        tile: Source tile.
        offsets: Per-dimension offsets into the output tensor.

    Returns:
        Tensor wrapping the store result.
    """
    span = _span()
    offsets_tuple = _to_make_tuple(offsets)
    return Tensor(expr=_ir_core.create_op_call(
        "manual.store",
        [tile.unwrap(), offsets_tuple, output_tensor.unwrap()],
        {},
        span,
    ))


def store_tile(
    output_tensor: Tensor,
    tile: Tile,
    tile_offsets: Sequence[int | Expr],
    tile_dims: list[int] | None = None,
) -> Tensor:
    """Store data from a tile back to a global tensor using tile-relative offsets.

    For N-D tensor with M-D tile (N >= M):
    - First (N-M) offsets are used directly (batch, head indices, etc.)
    - Last M offsets are multiplied by tile shape

    Example for 4D tensor [B, N, S, D] with 2D tile [TS, TD]:
    - ``store_tile(tensor, tile, [b, n, s_tile, d_tile])``
    - ``b`` and ``n`` are used directly
    - ``s_tile * TS`` and ``d_tile * TD`` are computed for the last two dims

    Example for BSND layout [B, S, N, D] with 2D tile [TS, TD]:
    - ``store_tile(tensor, tile, [b, s_tile, n, d_tile], tile_dims=[1, 3])``
    - ``b`` and ``n`` are used directly
    - ``s_tile * TS`` and ``d_tile * TD`` are computed for dimensions 1 and 3

    Args:
        output_tensor: Destination tensor.
        tile: Source tile.
        tile_offsets: Per-dimension offsets. First (N-M) are direct offsets,
            last M are tile-relative offsets.
        tile_dims: Optional list of dimension indices that tile corresponds to.
            If None, defaults to last M dimensions (BNSD layout).
            For BSND layout [B, S, N, D] with tile [TS, TD], use tile_dims=[1, 3].

    Returns:
        Tensor wrapping the store result.
    """
    return Tensor(expr=_ir_manual.store_tile(output_tensor.unwrap(), tile.unwrap(), _to_make_tuple(tile_offsets), tile_dims=tile_dims))


def move(
    out: Tile,
    tile: Tile,
    acc_to_vec_mode: Literal["single_vec0", "single_vec1", "dual_split_m", "dual_split_n"] | None = None,
) -> None:
    """Move a tile between memory levels, writing into a pre-allocated buffer.

    The TMOV variant (M2L, M2B, etc.) is determined by the output tile's
    memory space, which was set during make_tile().

    Args:
        out: Pre-allocated output tile; rebound on return.
        tile: Source tile.
        acc_to_vec_mode: AccToVecMode for Acc→Vec transfers (optional).
            - "single_vec0": SingleModeVec0 (default)
            - "single_vec1": SingleModeVec1
            - "dual_split_m": DualModeSplitM
            - "dual_split_n": DualModeSplitN
    """
    kwargs = {}
    if acc_to_vec_mode is not None:
        kwargs["acc_to_vec_mode"] = acc_to_vec_mode
    _op("manual.move", [tile.unwrap()], out, **kwargs)


def insert(
    out: Tile,
    tile: Tile,
    index_row: int | Expr = 0,
    index_col: int | Expr = 0,
    offset: int | Expr | None = None,
) -> None:
    """Insert a source sub-tile into destination tile at (index_row, index_col).

    Corresponds to pto-isa TINSERT instruction for UB→L1 transfer.

    Args:
        out: Pre-allocated destination tile (L1/Mat memory); rebound on return.
        tile: Source sub-tile (UB/Vec memory).
        index_row: Row index where insertion begins.
        index_col: Column index where insertion begins.
        offset: Optional byte offset for destination tile base address.
    """
    row_expr = _normalize_expr(index_row)
    col_expr = _normalize_expr(index_col)
    if offset is not None:
        _op("manual.insert", [tile.unwrap(), row_expr, col_expr, _normalize_expr(offset)], out)
    else:
        _op("manual.insert", [tile.unwrap(), row_expr, col_expr], out)


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


def fillpad(out: Tile, tile: Tile) -> None:
    """Fill a pre-allocated tile with padding from *tile*.

    Args:
        out: Pre-allocated destination tile; rebound on return.
            The destination tile's ``TileType`` determines the post-fill buffer
            shape/metadata, and ``TileType.pad`` determines the padding mode.
        tile: Source tile providing the data.
    """
    _op("manual.fillpad", [tile.unwrap()], out)


def get_block_idx() -> Scalar:
    """Return the current block index (unchanged from SSA style).

    Returns:
        Scalar wrapping the block index (UINT64 type).
    """
    from pypto.language.op.auto.op.auto_ops import get_block_idx as _get_block_idx
    return _get_block_idx()


def get_subblock_idx() -> Scalar:
    """Return to current subblock index (unchanged from SSA style).

    Returns:
        Scalar wrapping to subblock index (UINT64 type).
        >>>     ...
    """
    from pypto.language.op.auto.op.auto_ops import get_subblock_idx as _get_subblock_idx
    return _get_subblock_idx()


def get_block_num() -> Scalar:
    """Return to current block number (unchanged from SSA style).

    Returns:
        Scalar wrapping to block number (UINT64 type).

    Example:
        >>> block_num = pm.get_block_num()
        >>> if block_num < 5:
        >>>     # Process first 5 blocks differently
        >>>     ...
        >>>     ...
    """
    from pypto.language.op.auto.op.auto_ops import get_block_num as _get_block_num
    return _get_block_num()


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

def add_relu(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Element-wise addition RELU: out = max(0, (lhs + rhs))."""
    _op("manual.add_relu", [lhs.unwrap(), rhs.unwrap()], out)

def sub_relu(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Element-wise subtraction RELU: out = max(0, (lhs - rhs))."""
    _op("manual.sub_relu", [lhs.unwrap(), rhs.unwrap()], out)

def add_relu_cast(
    lhs: Tile,
    rhs: Tile,
    out: Tile,
    target_type: int | DataType,
    mode: Literal["none", "rint", "round", "floor", "ceil", "trunc", "odd"] = "round",
) -> None:
    """Element-wise addition RELU with cast: out = cast(max(0, (lhs + rhs)), target_type, mode).

    Args:
        lhs: Left tile.
        rhs: Right tile.
        out: Pre-allocated output tile with desired result dtype.
        target_type: Target DataType.
        mode: Rounding mode string.
    """
    _op("manual.add_relu_cast", [lhs.unwrap(), rhs.unwrap()], out, target_type=target_type, mode=mode)

def sub_relu_cast(
    lhs: Tile,
    rhs: Tile,
    out: Tile,
    target_type: int | DataType,
    mode: Literal["none", "rint", "round", "floor", "ceil", "trunc", "odd"] = "round",
) -> None:
    """Element-wise subtraction RELU with cast: out = cast(max(0, (lhs - rhs)), target_type, mode).

    Args:
        lhs: Left tile.
        rhs: Right tile.
        out: Pre-allocated output tile with desired result dtype.
        target_type: Target DataType.
        mode: Rounding mode string.
    """
    _op("manual.sub_relu_cast", [lhs.unwrap(), rhs.unwrap()], out, target_type=target_type, mode=mode)

def mul_cast(
    lhs: Tile,
    rhs: Tile,
    out: Tile,
    target_type: int | DataType,
    mode: Literal["none", "rint", "round", "floor", "ceil", "trunc", "odd"] = "round",
) -> None:
    """Element-wise multiplication with cast: out = cast(lhs * rhs, target_type, mode).

    Args:
        lhs: Left tile.
        rhs: Right tile.
        out: Pre-allocated output tile with desired result dtype.
        target_type: Target DataType.
        mode: Rounding mode string.
    """
    _op("manual.mul_cast", [lhs.unwrap(), rhs.unwrap()], out, target_type=target_type, mode=mode)

def mul_add_dst(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Element-wise multiplication with destination accumulation: out = (lhs * rhs) + out.

    Args:
        lhs: Left tile.
        rhs: Right tile.
        out: Pre-allocated output tile; used as both accumulator and destination.
    """
    _op("manual.mul_add_dst", [lhs.unwrap(), rhs.unwrap()], out)

def fused_mul_add(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Element-wise fused multiply-add: out = (lhs * out) + rhs.

    Args:
        lhs: Left tile (multiplied with).
        rhs: Right tile (added to product).
        out: Pre-allocated output tile; used as both multiplier and destination.
    """
    _op("manual.fused_mul_add", [lhs.unwrap(), rhs.unwrap()], out)

def fused_mul_add_relu(lhs: Tile, rhs: Tile, out: Tile) -> None:
    """Element-wise fused multiply-add-relu: out = max(0, (lhs * out) + rhs).

    Args:
        lhs: Left tile (multiplied with).
        rhs: Right tile (added to product).
        out: Pre-allocated output tile; used as both multiplier and destination.
    """
    _op("manual.fused_mul_add_relu", [lhs.unwrap(), rhs.unwrap()], out)

def gather(src: Tile, indices: Tile, out: Tile, tmp: Tile | None = None) -> None:
    """Gather elements from source tile using indices: out[i] = src[indices[i]].

    Args:
        src: Source tile.
        indices: Index tile.
        out: Pre-allocated output tile.
        tmp: Optional temporary tile for index gather.
    """
    if tmp is None:
        _op("manual.gather", [src.unwrap(), indices.unwrap()], out)
    else:
        _op("manual.gather", [src.unwrap(), indices.unwrap(), tmp.unwrap()], out)

def gatherb(src: Tile, offsets: Tile, out: Tile) -> None:
    """Gather elements using per-element byte offsets: out[i] = src[byte_offsets[i]].

    Args:
        src: Source tile.
        offsets: Byte offset tile.
        out: Pre-allocated output tile.
    """
    _op("manual.gatherb", [src.unwrap(), offsets.unwrap()], out)


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


def col_max(tile: Tile, tmp: Tile, out: Tile) -> None:
    """Col-wise max reduction: out[0, j] = max_i(tile[i, j])."""
    _op("manual.col_max", [tile.unwrap(), tmp.unwrap()], out)


def col_sum(tile: Tile, tmp: Tile, out: Tile) -> None:
    """Col-wise sum reduction: out[0, j] = sum_i(tile[i, j])."""
    _op("manual.col_sum", [tile.unwrap(), tmp.unwrap()], out)


# ---------------------------------------------------------------------------
# Sorting operations
# ---------------------------------------------------------------------------

def sort32(src: Tile, idx: Tile, dst: Tile, tmp: Tile | None = None) -> None:
    """Sort fixed-size 32-element blocks and produce an index mapping.

    This operation sorts each 32-element block of the input tile and produces
    two outputs: the sorted values and the permutation indices.

    Args:
        src: Input tile with loc=vec, blayout=row_major
        idx: Input/output tile for permutation indices (dtype=UINT32, modified in-place)
        dst: Output tile for sorted values (same dtype as src)
        tmp: Optional scratch tile for temporary storage
    """
    if tmp is not None:
        dst._expr = _ir_core.create_op_call(
            "manual.sort32", [src.unwrap(), idx.unwrap(), dst.unwrap(), tmp.unwrap()], {}, _span()
        )
    else:
        dst._expr = _ir_core.create_op_call(
            "manual.sort32", [src.unwrap(), idx.unwrap(), dst.unwrap()], {}, _span()
        )


def mrgsort(src: Tile, dst: Tile, block_len: int) -> None:
    """Perform merge sort on sorted lists (format1).

    This operation performs a merge sort on the input tile, merging sorted
    blocks of the specified length.

    Note: Only format1 is supported (single src, single dst).
    format2 (multi-list variant) is not exposed in the pypto API.

    Args:
        src: Input tile with loc=vec, blayout=row_major, rows=1
        dst: Output tile (same dtype as src)
        block_len: Block length for merge (must be multiple of 64)
    """
    dst._expr = _ir_core.create_op_call(
        "manual.mrgsort", [src.unwrap(), dst.unwrap()], {"block_len": block_len}, _span()
    )


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


def set_validshape(
    tile: Tile,
    row: int | Expr | Scalar,
    col: int | Expr | Scalar,
) -> None:
    """Update valid-shape metadata on a dynamic tile in place.

    Args:
        tile: Dynamic tile buffer to update.
        row: Runtime valid row count.
        col: Runtime valid column count.
    """
    row_expr = _scalar_expr(row)
    col_expr = _scalar_expr(col)
    tile._valid_shape = (row_expr, col_expr)
    _op("manual.set_validshape", [row_expr, col_expr], tile)


__all__ = [
    # Allocation
    "make_tile",
    # Memory
    "load", "load_tile", "store", "store_tile", "move", "ub_copy", "full", "fillpad", "get_block_idx", 
    "get_block_num", "get_subblock_idx",
    # Tile x Tile binary
    "add", "sub", "mul", "div", "rem", "maximum", "minimum",
    "and_", "or_", "shl", "shr", "add_relu", "sub_relu", "add_relu_cast", "sub_relu_cast", "mul_cast",
    "mul_add_dst", "fused_mul_add", "fused_mul_add_relu",
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
    # gather
    "gather", "gatherb",
    # Reduction
    "row_max", "row_sum", "row_min", "col_max", "col_sum",
    # Sorting
    "sort32", "mrgsort",
    # Broadcast
    "row_expand", "row_expand_add", "row_expand_sub", "row_expand_mul", "row_expand_div",
    "col_expand", "col_expand_mul", "col_expand_div", "col_expand_sub",
    "expands",
    # Matrix
    "matmul", "matmul_acc", "matmul_bias",
    "gemv", "gemv_acc", "gemv_bias",
    # Layout
    "reshape", "transpose",
    # Metadata
    "set_validshape",
]
