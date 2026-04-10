# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend tests for plm.mrgsort function."""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.op.manual as plm


@fe.kernel
def mrgsort_kernel(
    a: pl.Tensor[[1, 1024], pl.FP16],
    sorted_out: pl.Tensor[[1, 1024], pl.FP16],
) -> pl.Tensor[[1, 1024], pl.FP16]:
    """Merge sort using mrgsort with blockLen=256.
    
    Input must be pre-sorted in blocks of 64 elements in value-index pair format.
    For FP16, each element occupies 4 FP16 values (value + pad + idx_low + idx_high).
    
    From PTO_IR_manual.md constraints:
    - blockLen must be a multiple of 64 (in FP16 columns)
    - src valid column must be an integer multiple of blockLen * 4
    - repeatTimes = src valid column / (blockLen * 4) must be in [1, 255]
    
    For our case:
    - Input: 256 elements in value-index pair format = 1024 FP16 columns
    - Each block: 64 elements = 256 FP16 columns
    - blockLen = 256
    - blockLen * 4 = 1024 FP16 columns
    - src valid column = 1024 FP16 columns
    - repeatTimes = 1024 / 1024 = 1 block (but actually merges 4 pre-sorted 64-element blocks)
    
    Note: blockLen in PTO IR is in FP16 columns, while gen_data.py uses element count.
    For value-index pairs: blockLen_in_columns = blockLen_in_elements * 4
    """
    pl.system.bar_all()
    
    tile_src = plm.make_tile(
        plm.TileType(shape=[1, 1024], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0000, size=2048
    )
    tile_dst = plm.make_tile(
        plm.TileType(shape=[1, 1024], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0800, size=2048
    )
    
    with pl.section_vector():
        plm.load(tile_src, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.mrgsort(tile_dst, tile_src, block_len=256)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(sorted_out, tile_dst, [0, 0])
    
    return sorted_out


@fe.jit()
def test_mrgsort():
    """Test mrgsort with blockLen=64.
    
    mrgsort merges multiple pre-sorted blocks into one fully sorted output.
    blockLen=64 means each block has 64 elements.
    With 256 total elements, we have 4 blocks of 64 elements each.
    
    Data format for FP16:
    - Each element occupies 4 FP16 values: [value, pad, idx_low, idx_high]
    - idx is stored as two uint16 values (little-endian)
    
    Test procedure:
    1. Generate random data
    2. Sort each block internally in descending order
    3. Convert to value-index pair format
    4. mrgsort merges blocks into fully sorted output
    5. Verify output matches expected fully sorted result
    """
    device = "npu:7"
    torch.npu.set_device(device)

    torch.manual_seed(0)
    dtype = torch.float16
    
    block_len = 64
    num_blocks = 4
    total_elements = block_len * num_blocks
    
    a = torch.rand([1, total_elements], device=device, dtype=dtype)
    
    print("Original data:")
    print("a[0, :16] =", a[0, :16])
    
    for i in range(num_blocks):
        start = i * block_len
        end = (i + 1) * block_len
        block = a[0, start:end]
        sorted_block, _ = torch.sort(block, descending=True)
        a[0, start:end] = sorted_block
    
    print("\nAfter sorting each block internally (descending):")
    for i in range(num_blocks):
        start = i * block_len
        end = (i + 1) * block_len
        print(f"Block {i}: {a[0, start:start+8]}...")
    
    input_pairs = torch.zeros([1, total_elements * 4], device=device, dtype=dtype)
    for i in range(total_elements):
        input_pairs[0, i*4] = a[0, i]
        idx_val = i
        input_pairs[0, i*4 + 2] = torch.tensor(idx_val & 0xFFFF, dtype=torch.uint16).view(dtype)
        input_pairs[0, i*4 + 3] = torch.tensor((idx_val >> 16) & 0xFFFF, dtype=torch.uint16).view(dtype)
    
    sorted_out = torch.zeros([1, total_elements * 4], device=device, dtype=dtype)
    
    expected_sorted, expected_indices = torch.sort(a, dim=1, descending=True)
    print("\nExpected fully sorted output (descending):")
    print("expected_sorted[0, :16] =", expected_sorted[0, :16])

    compiled_lib = fe.compile(mrgsort_kernel, arch="a3")
    print("compiled lib path:", compiled_lib.lib_path)
    fe.launch(None, 1, compiled_lib, input_pairs, sorted_out)
    
    torch.npu.synchronize()
    
    print("\n***********npu output***********")
    print("sorted_out shape:", sorted_out.shape)
    
    values = sorted_out[0, 0::4]
    print("\nExtracted values[:16]:", values[:16])
    
    sorted_diff = torch.abs(values - expected_sorted).max().item()
    print(f"\nMax diff: {sorted_diff}")
    
    assert sorted_diff < 1e-3, f"mrgsort failed: max diff {sorted_diff}"

    idx_low = sorted_out[0, 2::4].view(torch.uint16)
    idx_high = sorted_out[0, 3::4].view(torch.uint16)
    extracted_indices = (idx_high.to(torch.int32) << 16) | idx_low.to(torch.int32)
    
    print("\nExtracted indices[:16]:", extracted_indices[:16])
    print("Expected indices[:16]:", expected_indices[0, :16])
    
    indices_match = torch.all(extracted_indices == expected_indices[0, :])
    print(f"\nIndices match: {indices_match}")
    
    assert indices_match, "mrgsort failed: indices do not match"
    print("mrgsort test passed!")


if __name__ == "__main__":
    test_mrgsort()
    print("\nAll tests passed!")
