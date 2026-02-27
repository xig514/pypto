# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend tests for the @fe.kernel decorator with manual (non-SSA) plm.* ops.

Each test kernel is compiled to PTO MLIR and the output is checked for
correct pto.alloc_tile / pto.tload / pto.tadd / pto.tmul patterns.
"""

import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm
from pypto import backend
from pypto.backend import BackendType
from pypto.pypto_core.codegen import PTOCodegen

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _compile_to_mlir(prog) -> str:
    """Compile an ir.Program to PTO MLIR without running external tools."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)
    codegen = PTOCodegen()
    result = codegen.generate(prog)
    return result if isinstance(result, str) else "".join(result.values())


# ---------------------------------------------------------------------------
# Test kernels — defined at module level so @fe.kernel runs at import time
# ---------------------------------------------------------------------------

# Kernel 1: two loads only (baseline)
@fe.kernel
def load_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_a = plm.create_tile([64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.UB,
                             addr=0x0000, size=16384)
    plm.load(a, [0, 0], [64, 128], out=tile_a)

    tile_b = plm.create_tile([64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.UB,
                             addr=0x4000, size=16384)
    plm.load(b, [0, 0], [64, 128], out=tile_b)
    return b


# Kernel 2: load two tiles and add them
@fe.kernel
def add_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_a = plm.create_tile([64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.UB,
                             addr=0x0000, size=16384)
    plm.load(a, [0, 0], [64, 128], out=tile_a)

    tile_b = plm.create_tile([64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.UB,
                             addr=0x4000, size=16384)
    plm.load(b, [0, 0], [64, 128], out=tile_b)

    tile_c = plm.create_tile([64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.UB,
                             addr=0x8000, size=16384)
    plm.add(tile_a, tile_b, out=tile_c)
    return b


# Kernel 3: load two tiles and multiply them
@fe.kernel
def mul_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_a = plm.create_tile([64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.UB,
                             addr=0x0000, size=16384)
    plm.load(a, [0, 0], [64, 128], out=tile_a)

    tile_b = plm.create_tile([64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.UB,
                             addr=0x4000, size=16384)
    plm.load(b, [0, 0], [64, 128], out=tile_b)

    tile_c = plm.create_tile([64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.UB,
                             addr=0x8000, size=16384)
    plm.mul(tile_a, tile_b, out=tile_c)
    return b


# Kernel 4: unary neg
@fe.kernel
def neg_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_a = plm.create_tile([64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.UB,
                             addr=0x0000, size=16384)
    plm.load(a, [0, 0], [64, 128], out=tile_a)

    tile_b = plm.create_tile([64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.UB,
                             addr=0x4000, size=16384)
    plm.neg(tile_a, out=tile_b)
    return a


# ---------------------------------------------------------------------------
# Test functions (run with: python test_kernel.py)
# ---------------------------------------------------------------------------

def test_load_kernel():
    """Two loads: verify alloc_tile and tload patterns."""
    mlir = _compile_to_mlir(load_kernel)
    print("\n=== test_load_kernel MLIR ===")
    print(mlir)

    assert "pto.alloc_tile" in mlir, "Expected pto.alloc_tile"
    assert "dtype=f16" in mlir, "Expected f16 dtype"
    assert "rows=64, cols=128" in mlir, "Expected 64x128 tile shape"
    # tile_b is at 0x4000 = 16384 — should carry explicit base_addr
    assert "base_addr = 16384" in mlir, "Expected base_addr = 16384 for tile_b"
    assert "pto.partition_view" in mlir, "Expected pto.partition_view for load"
    assert "pto.tload" in mlir, "Expected pto.tload"
    assert mlir.count("pto.tload") == 2, f"Expected 2 tloads, got {mlir.count('pto.tload')}"


def test_add_kernel():
    """Two loads + add: verify tadd and three alloc_tile."""
    mlir = _compile_to_mlir(add_kernel)
    print("\n=== test_add_kernel MLIR ===")
    print(mlir)

    # 3 tiles allocated
    n_alloc = mlir.count("pto.alloc_tile")
    assert n_alloc == 3, f"Expected 3 pto.alloc_tile, got {n_alloc}"
    # tile_b at 0x4000 = 16384
    assert "base_addr = 16384" in mlir, "Expected base_addr = 16384 for tile_b"
    # tile_c at 0x8000 = 32768
    assert "base_addr = 32768" in mlir, "Expected base_addr = 32768 for tile_c"
    # 2 loads and 1 add
    assert mlir.count("pto.tload") == 2, f"Expected 2 tloads, got {mlir.count('pto.tload')}"
    assert "pto.tadd" in mlir, "Expected pto.tadd"


def test_mul_kernel():
    """Two loads + mul: verify tmul."""
    mlir = _compile_to_mlir(mul_kernel)
    print("\n=== test_mul_kernel MLIR ===")
    print(mlir)

    assert mlir.count("pto.alloc_tile") == 3, "Expected 3 pto.alloc_tile"
    assert mlir.count("pto.tload") == 2, "Expected 2 pto.tload"
    assert "pto.tmul" in mlir, "Expected pto.tmul"


def test_neg_kernel():
    """Load + unary neg: verify tneg."""
    mlir = _compile_to_mlir(neg_kernel)
    print("\n=== test_neg_kernel MLIR ===")
    print(mlir)

    assert mlir.count("pto.alloc_tile") == 2, "Expected 2 pto.alloc_tile"
    assert "pto.tload" in mlir, "Expected pto.tload"
    assert "pto.tneg" in mlir, "Expected pto.tneg"


if __name__ == "__main__":
    test_load_kernel()
    test_add_kernel()
    test_mul_kernel()
    test_neg_kernel()
    print("\nAll tests passed!")
