"""Lightning Indexer kernel using PyPTO IR manual (non-SSA) mode.

Lightning Indexer computes Top-k indices for sparse attention:
    Indices = Top-k(W ⊙ ReLU(Q @ K^T))

This is a fused operator combining:
  1. MatMul: Q @ K^T (Cube operation)
  2. ReLU + weight scaling (Vector operation)
  3. Top-k via sort32 (per 256-element block) + mrgsort (Vector operation)

Usage:
    python3 tests/ut/frontend/lightning_indexer/test_lightning_indexer.py
"""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.op.manual as plm

# Tile dimensions
TS        = 128          # Sq / Sk tile size
TD        = 128          # head dimension
TOPK      = 128
SK_FIXED  = 128         # fixed Sk (key sequence length)
# FP32 format: [value, idx, value, idx, ...] per element = 2 FP32
SK2_FIXED = SK_FIXED * 2 # 2048 * 2 = 4096 FP32 columns for sorted_workspace

# Cube tile byte sizes
Q_F16      = TS * TD * 2     # 32768  — [128,128] FP16
K_F16      = TD * TS * 2     # 32768  — [128,128] FP16
SCORES_F32 = TS * TS * 4     # 65536  — [128,128] FP32

# MAT addresses (512KB budget)
MA0 = 0          # q_mat  [128,128] FP16 → 32 KB
MA1 = Q_F16      # k_mat  [128,128] FP16 → 32 KB

# LEFT / RIGHT / ACC
LA0 = 0    # left  [128,128] FP16 → 32 KB
RA0 = 0    # right [128,128] FP16 → 32 KB
CA0 = 0    # acc   [128,128] FP32 → 64 KB

# VEC tile byte sizes (plain int constants — BinOp exprs are not allowed in size= kwargs)
VB4              = TS * TS * 4       # 65536  — [TS, TS]       FP32
WEIGHTS_FP32_SZ  = TS * 1 * 4       # 512    — [TS, 1]        FP32
SORT_SZ          = 32                # elements per sort32 call (sort32 sorts exactly 32 elements)
SORT_SZ_BYTES    = SORT_SZ * 4       # 128    — [1, 32]         FP32
SORT_IDX_BYTES   = SORT_SZ * 4       # 128    — [1, 32]         UINT32
SORT_DST_COLS    = SORT_SZ * 2       # 64     — 32 elements * 2 FP32 per element
SORT_DST_BYTES   = SORT_DST_COLS * 4 # 256    — [1, 64]         FP32
MRGSORT_COLS     = SK2_FIXED         # 4096   — 2048 elements * 2 FP32 per element
MRGSORT_BYTES    = MRGSORT_COLS * 4  # 16384  — [1, 4096]       FP32

# VEC addresses (192KB budget on a3)
VA0  = 0                       # scores_vec    [TS, TS]        FP32  —  65536 B
VA1  = VA0 + VB4               # tmp_vec       [TS, TS]        FP32  —  65536 B
VA2  = VA1 + VB4               # weights_fp32  [TS, 1]         FP32  —    512 B
VA3  = VA2 + WEIGHTS_FP32_SZ   # sort_src      [1, 32]         FP32  —    128 B
VA4  = VA3 + SORT_SZ_BYTES     # sort_idx      [1, 32]        UINT32 —    128 B
VA5  = VA4 + SORT_IDX_BYTES    # sort_dst      [1, 64]         FP32  —    256 B
VA6  = VA5 + SORT_DST_BYTES    # mrgsort_src   [1, 4096]       FP32  —  16384 B
VA7  = VA6 + MRGSORT_BYTES     # mrgsort_dst   [1, 4096]       FP32  —  16384 B
# Total: ~164 KB < 192 KB ✓

# Cross-core sync event ID: Cube signals Vector that scores_workspace row is ready
SCORES_READY = 5

Sq  = pl.DynVar('Sq')
Sk  = pl.DynVar('Sk')
Sk2 = pl.DynVar('Sk2')
D   = pl.DynVar('D')


def alloc_cube_buffer():
    q_mat = plm.make_tile(
        plm.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat),
        addr=MA0, size=Q_F16)
    k_mat = plm.make_tile(
        plm.TileType(shape=[TD, TS], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat, blayout=1, slayout=2),
        addr=MA1, size=K_F16)
    left = plm.make_tile(
        plm.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Left),
        addr=LA0, size=Q_F16)
    right = plm.make_tile(
        plm.TileType(shape=[TD, TS], dtype=pl.FP16, target_memory=pl.MemorySpace.Right),
        addr=RA0, size=K_F16)
    acc = plm.make_tile(
        plm.TileType(shape=[TS, TS], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc),
        addr=CA0, size=SCORES_F32)
    return q_mat, k_mat, left, right, acc


def alloc_vector_buffer():
    scores_vec = plm.make_tile(
        plm.TileType(shape=[TS, TS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=VA0, size=VB4)
    tmp_vec = plm.make_tile(
        plm.TileType(shape=[TS, TS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=VA1, size=VB4)
    weights_fp32 = plm.make_tile(
        plm.TileType(shape=[TS, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2),
        addr=VA2, size=WEIGHTS_FP32_SZ)
    sort_src = plm.make_tile(
        plm.TileType(shape=[1, SORT_SZ], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=VA3, size=SORT_SZ_BYTES)
    sort_idx = plm.make_tile(
        plm.TileType(shape=[1, SORT_SZ], dtype=pl.UINT32, target_memory=pl.MemorySpace.Vec),
        addr=VA4, size=SORT_IDX_BYTES)
    sort_dst = plm.make_tile(
        plm.TileType(shape=[1, SORT_DST_COLS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=VA5, size=SORT_DST_BYTES)
    mrgsort_src = plm.make_tile(
        plm.TileType(shape=[1, MRGSORT_COLS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=VA6, size=MRGSORT_BYTES)
    mrgsort_dst = plm.make_tile(
        plm.TileType(shape=[1, MRGSORT_COLS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=VA7, size=MRGSORT_BYTES)
    return (scores_vec, tmp_vec, weights_fp32,
            sort_src, sort_idx, sort_dst, mrgsort_src, mrgsort_dst)


@fe.kernel
def lightning_indexer_kernel(
    query: pl.Tensor[[Sq, D], pl.FP16],
    key_tensor: pl.Tensor[[Sk, D], pl.FP16],
    weights: pl.Tensor[[Sq, 1], pl.FP32],
    scores_workspace: pl.Tensor[[Sq, Sk], pl.FP32],
    idx_workspace: pl.Tensor[[Sq, Sk], pl.UINT32],
    sorted_workspace: pl.Tensor[[Sq, SK2_FIXED], pl.FP32],
) -> pl.Tensor[[Sq, SK2_FIXED], pl.FP32]:
    """Lightning Indexer kernel: compute Top-128 indices for sparse attention.

    Algorithm:
    1. Cube: Q @ K^T → scores_workspace
    2. Vector:
       - ReLU + weight scaling on scores
       - sort32 per SORT_SZ-element block → [value, idx] FP32 pairs in sorted_workspace
       - mrgsort to merge all blocks into fully sorted [value, idx] pairs
       - sorted_workspace[:, 1::2].view(int32) gives top-TOPK indices on host
    """
    sq_dim = Sq
    sk_dim = Sk
    sq_tiles = (sq_dim + (TS - 1)) // TS
    sk_tiles = (sk_dim + (TS - 1)) // TS
    sk_sort_blocks = (sk_dim + (SORT_SZ - 1)) // SORT_SZ
    num_cores = pl.block.index_cast(pl.block.get_block_num())
    core_id   = pl.block.index_cast(pl.block.get_block_idx())

    pl.system.bar_all()

    # =================== CUBE SECTION ===================
    with pl.section_cube():
        q_mat, k_mat, left, right, acc = alloc_cube_buffer()

        for sqi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = sqi * TS

            plm.load(q_mat, query, [sq_off, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
            for ski in pl.range(0, sk_tiles):
                sk_off = ski * TS
                # Ensure previous matmul finished reading L0A/L0B before overwriting
                pl.system.sync_src(set_pipe=pl.PipeType.M,    wait_pipe=pl.PipeType.MTE1, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.M,    wait_pipe=pl.PipeType.MTE1, event_id=1)
                plm.load(k_mat, key_tensor, [sk_off, 0], layout="dn")
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
                plm.move(left, q_mat)
                plm.move(right, k_mat)
                pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M,    event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M,    event_id=1)
                plm.matmul(acc, left, right)
                pl.system.sync_src(set_pipe=pl.PipeType.M,    wait_pipe=pl.PipeType.FIX,  event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.M,    wait_pipe=pl.PipeType.FIX,  event_id=0)
                plm.l0c_store(acc, [sq_off, sk_off], [TS, TS], scores_workspace)
                pl.system.sync_src(set_pipe=pl.PipeType.FIX,  wait_pipe=pl.PipeType.M,    event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.FIX,  wait_pipe=pl.PipeType.M,    event_id=1)
            # Signal Vector core: all scores for this sqi row are ready
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=SCORES_READY)
    
    # =================== VECTOR SECTION ===================
    with pl.section_vector():
        (scores_vec, tmp_vec, weights_fp32,
         sort_src, sort_idx, sort_dst, mrgsort_src, mrgsort_dst) = alloc_vector_buffer()

        for sqi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = sqi * TS

            # Load weights FP32 directly
            plm.load(weights_fp32, weights, [sq_off, 0], layout="dn")
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            # Wait for Cube to finish writing scores_workspace for this sqi row
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=SCORES_READY)
            # ReLU + weight scaling per Sk tile
            for ski in pl.range(0, sk_tiles):
                sk_off = ski * TS
                plm.load(scores_vec, scores_workspace, [sq_off, sk_off])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
                plm.relu(tmp_vec, scores_vec)
                pl.system.bar_v()
                plm.row_expand_mul(tmp_vec, tmp_vec, weights_fp32)
                pl.system.bar_v()
                pl.system.sync_src(set_pipe=pl.PipeType.V,    wait_pipe=pl.PipeType.MTE3, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V,    wait_pipe=pl.PipeType.MTE3, event_id=0)
                plm.store(scores_workspace, tmp_vec, [sq_off, sk_off])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=1)
            # Ensure last ski-loop MTE3 store completes before bi-loop MTE2 loads
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)

            # sort32: sort each SORT_SZ-element block; output is [value, idx] FP32 pairs
            for row in pl.range(0, TS):
                for blk in pl.range(0, sk_sort_blocks):
                    blk_off = blk * SORT_SZ
                    plm.load(sort_src, scores_workspace, [sq_off + row, blk_off])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    plm.load(sort_idx, idx_workspace,    [sq_off + row, blk_off])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
                    plm.sort32(sort_dst, sort_src, sort_idx)
                    pl.system.bar_v()
                    pl.system.sync_src(set_pipe=pl.PipeType.V,    wait_pipe=pl.PipeType.MTE3, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V,    wait_pipe=pl.PipeType.MTE3, event_id=0)
                    plm.store(sorted_workspace, sort_dst, [sq_off + row, blk_off * 2])
                    # Backward: MTE3 done, next iteration's MTE2 load can overwrite sort_src/sort_idx
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=1)
                
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=2)

                # mrgsort: merge all sorted blocks into one fully sorted sequence.
                # Each sort32 block = 32 elements × 2 FP32/elem = 64 FP32 cols → block_len=64
                plm.load(mrgsort_src, sorted_workspace, [sq_off + row, 0])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                plm.mrgsort(mrgsort_dst, mrgsort_src, block_len=64)
                pl.system.bar_v()
                pl.system.sync_src(set_pipe=pl.PipeType.V,    wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V,    wait_pipe=pl.PipeType.MTE3, event_id=1)
                plm.store(sorted_workspace, mrgsort_dst, [sq_off + row, 0])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)

    return sorted_workspace


# ================================================================
#  Reference + Tests
# ================================================================

def reference_lightning_indexer(weighted, topk=TOPK):
    """Reference: sorted values and indices of top-k over precomputed weighted scores."""
    sorted_values, sorted_indices = torch.sort(weighted, dim=1, descending=True)
    return sorted_values[:, :topk], sorted_indices[:, :topk]


def _verify_outputs(tag, indices_out, ref_weighted, ref_values, ref_indices):
    """Verify indices_out against reference.

    ref_weighted : [S1, S2] CPU FP32 — W ⊙ ReLU(Q @ K^T)
    ref_values   : [S1, topk] CPU FP32 — top-k values (sorted descending)
    ref_indices  : [S1, topk] CPU INT64 — top-k indices

    Two checks:
    1. Index match rate > 99%  (strict equality may fail on ties due to FP16 matmul precision)
    2. Values at NPU-chosen indices ≈ ref_values  (measures how close NPU top-k is to reference)
       Tolerance accounts for FP16 matmul precision (~0.5 in score magnitude).
    """
    indices_out_cpu = indices_out.cpu()

    # 打印第0行的前32个值和索引
    npu_values = ref_weighted[0:1].gather(1, indices_out_cpu[0:1].long())[0, :32]
    npu_indices = indices_out_cpu[0, :32]
    ref_values_row = ref_values[0, :32]
    ref_indices_row = ref_indices[0, :32]
    print(f"  [{tag}] NPU values[:32]  : {npu_values.tolist()}")
    print(f"  [{tag}] NPU indices[:32] : {npu_indices.tolist()}")
    print(f"  [ref] ref values[:32]  : {ref_values_row.tolist()}")
    print(f"  [ref] ref indices[:32] : {ref_indices_row.tolist()}")

    # Check 1: index match rate (tolerates tie-breaking differences)
    match = (indices_out_cpu == ref_indices).float().mean().item()
    print(f"  index match rate={match:.4f}")
    assert match > 0.99, f"lightning_indexer {tag} failed: index match={match:.4f}"

    # Check 2: values at NPU-chosen positions vs reference top-k values
    # got_values[i,k] = ref_weighted[i, indices_out[i,k]]
    # diff > 0 only when NPU picks a different element than reference (tie-breaking)
    got_values = ref_weighted.gather(1, indices_out_cpu.long())
    values_diff = torch.abs(got_values - ref_values).max().item()
    print(f"  values max diff={values_diff:.6f}")
    assert values_diff < 0.5, f"lightning_indexer {tag} failed: values max diff {values_diff}"

    print(f"{tag} PASS")


def test_lightning_indexer():
    compiled_cce = fe.compile(lightning_indexer_kernel, arch="a3", codegen_mode="cce")
    compiled_pto = fe.compile(lightning_indexer_kernel, arch="a3", codegen_mode="pto")
    device = "npu:7"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    for S1, S2, D, topk, num_cores in [
        (128, 128, TD, TOPK, 1),
    ]:
        print(f"\nLightning-Indexer ({S1},{S2},{D}) topk={topk} cores={num_cores}")
        query   = torch.randn([S1, D], device=device, dtype=torch.float16)
        key     = torch.randn([S2, D], device=device, dtype=torch.float16)
        weights = torch.rand([S1, 1], device=device, dtype=torch.float32).abs() + 0.1

        # Precompute reference once (shared by CCE and PTO)
        ref_weighted = torch.relu(torch.matmul(query.float(), key.float().T)) * weights
        ref_values, ref_indices = reference_lightning_indexer(ref_weighted, topk)
        ref_weighted = ref_weighted.cpu()
        ref_values   = ref_values.cpu()
        ref_indices  = ref_indices.cpu().int()

        idx_workspace = torch.arange(S2, device=device, dtype=torch.int32) \
                            .unsqueeze(0).expand(S1, -1).contiguous()

        def make_workspaces():
            return (
                torch.zeros([S1, S2],     device=device, dtype=torch.float32),
                torch.zeros([S1, S2 * 2], device=device, dtype=torch.float32),)
        # --- CCE ---
        scores_workspace, sorted_workspace = make_workspaces()
        fe.launch(None, num_cores, compiled_cce,
                  query, key, weights,
                  scores_workspace, idx_workspace, sorted_workspace)
        torch.npu.synchronize()

        # 提取索引：取奇数列(idx字段) reinterpret 为 int32，再拼接高低16位
        raw_cce = sorted_workspace[:, 1::2].contiguous().view(torch.int32)[:, :topk]
        low = raw_cce & 0xFFFF
        high = (raw_cce >> 16) & 0xFFFF
        indices_out_cce = (high << 16) | low
        _verify_outputs("CCE", indices_out_cce, ref_weighted, ref_values, ref_indices)

        # --- PTO ---
        scores_workspace, sorted_workspace = make_workspaces()
        fe.launch(None, num_cores, compiled_pto,
                  query, key, weights,
                  scores_workspace, idx_workspace, sorted_workspace)
        torch.npu.synchronize()
        raw_pto = sorted_workspace[:, 1::2].contiguous().view(torch.int32)[:, :topk]
        low = raw_pto & 0xFFFF
        high = (raw_pto >> 16) & 0xFFFF
        indices_out_pto = (high << 16) | low
        _verify_outputs("PTO", indices_out_pto, ref_weighted, ref_values, ref_indices)

if __name__ == "__main__":
    print("Lightning Indexer: sort32 (256-block) + mrgsort, single-core")
    print("=" * 60)
    test_lightning_indexer()
    print("\nAll lightning_indexer tests passed!")