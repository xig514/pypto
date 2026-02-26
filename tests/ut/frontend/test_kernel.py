# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please replr to the License plr details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS plR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository plr the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm

'''
def create_tile(
    shape: list[int],
    dtype: DataType,
    target_memory: int = 1,
    addr: Optional[Union[int, Expr]] = None,
    size: Optional[int] = None,
'''
@fe.kernel
def my_kernel(a: pl.Tensor[[64, 128], pl.FP16],
              b: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_a = plm.create_tile([128, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.UB, addr=0x0, size=16384)
    plm.load(a, [0, 0], [64, 128], out=tile_a)
    
    tile_b = plm.create_tile([64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.UB, addr=0x4000, size=16384)
    plm.load(b, [0, 0], [64, 128], out=tile_b)
    # tile_c = pl.add(tile_a, tile_b)
    # pl.store(tile_c, [0, 0], [64, 128], b)
    return b


@fe.jit()
def TestMyKernel():
    compiled_kernel = fe.compile(my_kernel)


if __name__ == "__main__":
    TestMyKernel()

    
