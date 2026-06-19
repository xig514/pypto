"""FlashAttention kernel with Causal Mask - BN axis version.

Causal (lower triangular) attention: each position can only attend to itself and previous positions.

Features:
  1. Causal mask: Q[i] can only attend to K/V[0:i+1]
  2. Optimized KV loop: only compute necessary KV tiles (j <= qi)
  3. External mask: mask tensor is passed from outside the kernel
  4. Mask format: [Sq, Skv] FP32 tensor, 1.0 = keep, 0.0 = mask out (set to -inf)

Tensor shapes:
  - Q/K/V/O: [B, N, Sq, D], qk_buf: [B, N, Sq, Skv], p_buf: [B, N, Sq, Skv]
  - attn_mask: [Sq, Skv] - external attention mask tensor

Mask usage:
  - attn_mask[i, j] = 1.0: position i can attend to position j (keep original value)
  - attn_mask[i, j] = 0.0: position i cannot attend to position j (set to -inf)
  - For causal attention: lower triangular mask (i >= j -> 1.0, i < j -> 0.0)

Usage:
    python3 tests/ut/frontend/flash_attention/test_fa_BNSD_causal.py
"""

import math
import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.op.manual as plm

TS = 128
TKV = 128
TD = 128
TS_HALF = TS // 2
SCALE = 1.0 / math.sqrt(TD)

Q_F16   = TS * TD * 2
KT_F16  = TD * TKV * 2
QK_F16  = TS * TKV * 2
V_F16   = TKV * TD * 2
P_F16   = TS * TKV * 2
QK_F32  = TS * TKV * 4
PV_F32  = TS * TD * 4

MA0 = 0
MA0_PONG = MA0 + Q_F16
MA1 = Q_F16 * 2
MA1_PONG = MA1 + KT_F16
MA2 = MA1 + KT_F16 * 2
MA2_PONG = MA2 + P_F16
MA3 = MA2 + P_F16  * 2
MA3_PONG = MA3 + V_F16

LA0 = 0
LA1 = Q_F16

RA0 = 0
RA1 = KT_F16

CA0 = 0
CA1 = QK_F32

VB4_KV = TS_HALF * TKV * 4
VB2_KV = TS_HALF * TKV * 2
VB1_KV = TS_HALF * TKV * 1
VB4    = TS_HALF * TD * 4
VB2    = TS_HALF * TD * 2
VB_RED = TS_HALF * 1 * 4

VA0 = 0
VA1 = VA0 + VB4_KV
VA2 = VA1 + VB4_KV
VA3 = VA2 + VB2_KV
VA4 = VA3 + VB1_KV
VA5 = VA4 + VB4_KV
VA6 = VA5 + VB4
VA7 = VA6 + VB4
VA8 = VA7 + VB2
VA9 = VA8 + VB_RED
VA10 = VA9 + VB_RED
VA11 = VA10 + VB_RED

QK_READY = 0
P_READY = 1
PV_READY = 2

B = pl.DynVar('B')
N = pl.DynVar('N')
Sq2 = pl.DynVar('Sq')
Skv2 = pl.DynVar('Skv')
D2 = pl.DynVar('D')
NumRanges = pl.DynVar('NumRanges')

NEG_INF = -1e9


@fe.kernel
def fa_causal_kernel_bn(
    q: pl.Tensor[[B, N, Sq2, D2], pl.FP16],
    k: pl.Tensor[[B, N, Skv2, D2], pl.FP16],
    v: pl.Tensor[[B, N, Skv2, D2], pl.FP16],
    o: pl.Tensor[[B, N, Sq2, D2], pl.FP16],
    qk_buf: pl.Tensor[[B, N, Sq2, Skv2], pl.FP32],
    p_buf: pl.Tensor[[B, N, Sq2, Skv2], pl.FP16],
    pv_buf: pl.Tensor[[48 * TS, D2], pl.FP32],
    work_ranges: pl.Tensor[[NumRanges, 2], pl.INT32],
    attn_mask: pl.Tensor[[Sq2, Skv2], pl.FP32],
) -> pl.Tensor[[B, N, Sq2, D2], pl.FP16]:
    with pl.section_cube():
        sq_dim = Sq2
        skv_dim = Skv2
        sq_tiles = (sq_dim + (TS - 1)) // TS
        skv_tiles = (skv_dim + (TKV - 1)) // TKV
        num_cores = pl.block.index_cast(pl.block.get_block_num())
        core_id = pl.block.index_cast(pl.block.get_block_idx())
        n_dim = N

        q_mat_type = plm.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
        q_mat_0 = plm.make_tile(q_mat_type, addr=MA0, size=Q_F16)
        q_mat_1 = plm.make_tile(q_mat_type, addr=MA0_PONG, size=Q_F16)
        q_mat_buf = (q_mat_0, q_mat_1)

        k_mat_type = plm.TileType(shape=[TD, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat, blayout=1, slayout=2)
        k_mat_0 = plm.make_tile(k_mat_type, addr=MA1, size=KT_F16)
        k_mat_1 = plm.make_tile(k_mat_type, addr=MA1_PONG, size=KT_F16)
        k_mat_buf = (k_mat_0, k_mat_1)

        p_mat_0 = plm.make_tile(plm.TileType(shape=[TS, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat), addr=MA2, size=P_F16)
        p_mat_1 = plm.make_tile(plm.TileType(shape=[TS, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat), addr=MA2_PONG, size=P_F16)
        p_mat_buf = (p_mat_0, p_mat_1)

        v_mat_0 = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat), addr=MA3, size=V_F16)
        v_mat_1 = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat), addr=MA3_PONG, size=V_F16)
        v_mat_buf = (v_mat_0, v_mat_1)

        left_0 = plm.make_tile(plm.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Left), addr=LA0, size=Q_F16)
        left_1 = plm.make_tile(plm.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Left), addr=LA1, size=Q_F16)
        left_buf = (left_0, left_1)

        right_0 = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Right), addr=RA0, size=KT_F16)
        right_1 = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Right), addr=RA1, size=KT_F16)
        right_buf = (right_0, right_1)

        acc_0 = plm.make_tile(plm.TileType(shape=[TS, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc), addr=CA0, size=QK_F32)
        acc_1 = plm.make_tile(plm.TileType(shape=[TS, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc), addr=CA1, size=PV_F32)
        acc_buf = (acc_0, acc_1)

        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)

        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)

        q_count = 0
        left_index = 0
        right_index = 0
        work_start = pl.block.index_cast(pl.read(work_ranges, [core_id, 0]))
        work_end = pl.block.index_cast(pl.read(work_ranges, [core_id, 1]))
        for work_id in pl.range(work_start, work_end):
            b_idx = work_id // n_dim
            n_idx = work_id % n_dim
            for qi in pl.range(0, sq_tiles):
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)
                sq_off = qi * TS
                q_mat_idx = q_count % 2
                plm.load_tile(q_mat_buf[q_mat_idx], q, [b_idx, n_idx, qi, 0])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

                causal_kv_tiles = qi + 1
                for j in pl.range(0, causal_kv_tiles):
                    buf_idx = (q_count * skv_tiles + j) % 2
                    skv_off = j * TKV

                    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
                    plm.move(left_buf[left_index], q_mat_buf[q_mat_idx])
                    if j == causal_kv_tiles - 1:
                        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)

                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    plm.load_tile(k_mat_buf[buf_idx], k, [b_idx, n_idx, j, 0], layout="dn")
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)

                    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
                    plm.move(right_buf[right_index], k_mat_buf[buf_idx])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2)

                    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
                    plm.matmul(acc_buf[buf_idx], left_buf[left_index], right_buf[right_index])
                    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
                    right_index = 1 - right_index
                    left_index = 1 - left_index
                    plm.store_tile(qk_buf, acc_buf[buf_idx], [b_idx, n_idx, qi, j])
                    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)

                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY)
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=P_READY)

                    buf_idx_pv = (q_count * skv_tiles + j) % 2
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)
                    plm.load_tile(p_mat_buf[buf_idx], p_buf, [b_idx, n_idx, qi, j])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)
                    plm.load_tile(v_mat_buf[buf_idx], v, [b_idx, n_idx, j, 0])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)

                    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
                    plm.move(left_buf[left_index], p_mat_buf[buf_idx])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)

                    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
                    plm.move(right_buf[right_index], v_mat_buf[buf_idx])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)

                    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
                    plm.matmul(acc_buf[buf_idx_pv], left_buf[left_index], right_buf[right_index])
                    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
                    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
                    right_index = 1 - right_index
                    left_index = 1 - left_index
                    plm.store_tile(pv_buf, acc_buf[buf_idx_pv], [core_id * 2 + q_mat_idx, 0])
                    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
                    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY)
                q_count = q_count + 1
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)

    with pl.section_vector():
        sq_dim = Sq2
        skv_dim = Skv2
        sq_tiles = (sq_dim + (TS - 1)) // TS
        skv_tiles = (skv_dim + (TKV - 1)) // TKV
        num_cores = pl.block.index_cast(pl.block.get_block_num())
        core_id = pl.block.index_cast(pl.block.get_block_idx())
        sub_id = pl.block.index_cast(pl.block.get_subblock_idx())
        row_off = sub_id * TS_HALF

        qk_vec = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA0, size=VB4_KV)
        tmp_vec = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA1, size=VB4_KV)
        p_f16 = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA2, size=VB2_KV)
        mask_vec = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.INT8, target_memory=pl.MemorySpace.Vec), addr=VA3, size=VB1_KV)
        neg_inf_vec = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA4, size=VB4_KV)

        running_o = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA5, size=VB4)
        pv_vec = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA6, size=VB4)
        o_f16 = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA7, size=VB2)

        reduce_dst = plm.make_tile(plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA8, size=VB_RED)
        global_max = plm.make_tile(plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA9, size=VB_RED)
        global_sum = plm.make_tile(plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA10, size=VB_RED)
        exp_corr = plm.make_tile(plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA11, size=VB_RED)

        reduce_dst_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA8, size=VB_RED)
        global_max_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA9, size=VB_RED)
        global_sum_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA10, size=VB_RED)
        exp_corr_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA11, size=VB_RED)
        
        q_count = 0
        n_dim = N
        work_start = pl.block.index_cast(pl.read(work_ranges, [core_id, 0]))
        work_end = pl.block.index_cast(pl.read(work_ranges, [core_id, 1]))
        for work_id in pl.range(work_start, work_end):
            b_idx = work_id // n_dim
            n_idx = work_id % n_dim
            for qi in pl.range(0, sq_tiles):
                sq_off = qi * TS
                q_mat_idx = q_count % 2
                causal_kv_tiles = qi + 1
                
                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=QK_READY)
                plm.load_tile(qk_vec, qk_buf, [b_idx, n_idx, qi * 2 + sub_id, 0])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                
                plm.load_tile(mask_vec, attn_mask, [qi * 2 + sub_id, 0])
                pl.system.bar_v()
                plm.expands(neg_inf_vec, NEG_INF)
                plm.sel(qk_vec, mask_vec, neg_inf_vec, tmp_vec, qk_vec)
                pl.system.bar_v()
                
                plm.row_max(reduce_dst, qk_vec, tmp_vec)
                pl.system.bar_v()
                plm.row_expand_sub(tmp_vec, qk_vec, reduce_dst)
                plm.muls(global_max, reduce_dst, 1.0)
                plm.muls(tmp_vec, tmp_vec, SCALE)
                plm.exp(qk_vec, tmp_vec)
                pl.system.bar_v()
                plm.row_sum(reduce_dst, qk_vec, tmp_vec)
                pl.system.bar_v()
                plm.muls(global_sum, reduce_dst, 1.0)
                plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                plm.store_tile(p_buf, p_f16, [b_idx, n_idx, qi * 2 + sub_id, 0])
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY)

                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=PV_READY)
                plm.load_tile(running_o, pv_buf, [core_id * 4 + q_mat_idx * 2 + sub_id, 0])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

                for j in pl.range(1, causal_kv_tiles):
                    skv_off = j * TKV
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=QK_READY)
                    plm.load_tile(qk_vec, qk_buf, [b_idx, n_idx, qi * 2 + sub_id, j])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

                    plm.load_tile(mask_vec, attn_mask, [qi * 2 + sub_id, j])
                    pl.system.bar_v()
                    plm.expands(neg_inf_vec, NEG_INF)
                    plm.sel(qk_vec, mask_vec, neg_inf_vec, tmp_vec, qk_vec)
                    pl.system.bar_v()

                    plm.row_max(reduce_dst, qk_vec, tmp_vec)
                    pl.system.bar_v()
                    plm.maximum(reduce_dst_rm, reduce_dst_rm, global_max_rm)
                    pl.system.bar_v()
                    plm.sub(exp_corr_rm, global_max_rm, reduce_dst_rm)
                    pl.system.bar_v()
                    plm.muls(global_max_rm, reduce_dst_rm, 1.0)
                    pl.system.bar_v()
                    plm.row_expand_sub(tmp_vec, qk_vec, reduce_dst)
                    plm.muls(exp_corr_rm, exp_corr_rm, SCALE)
                    plm.muls(tmp_vec, tmp_vec, SCALE)
                    plm.exp(exp_corr_rm, exp_corr_rm)
                    plm.exp(qk_vec, tmp_vec)
                    plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
                    pl.system.bar_v()
                    plm.mul(global_sum_rm, global_sum_rm, exp_corr_rm)
                    plm.row_sum(reduce_dst, qk_vec, tmp_vec)
                    pl.system.bar_v()
                    plm.add(global_sum_rm, global_sum_rm, reduce_dst_rm)

                    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                    plm.store_tile(p_buf, p_f16, [b_idx, n_idx, qi * 2 + sub_id, j])
                    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY)

                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=PV_READY)
                    plm.load_tile(pv_vec, pv_buf, [core_id * 4 + q_mat_idx * 2 + sub_id, 0])
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                    plm.row_expand_mul(running_o, running_o, exp_corr)
                    plm.add(running_o, running_o, pv_vec)
                q_count = q_count + 1

                plm.row_expand_div(running_o, running_o, global_sum)
                plm.cast(o_f16, running_o, target_type=pl.FP16, mode="round")
                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                plm.store_tile(o, o_f16, [b_idx, n_idx, qi * 2 + sub_id, 0])
    return o


def flash_attention_causal_ref_bn(q, k, v, d):
    scale_val = 1.0 / math.sqrt(d)
    b, n, sq, d_ = q.shape
    _, _, skv, _ = k.shape
    o_ref = torch.zeros_like(q)
    for bi in range(b):
        for ni in range(n):
            qk = torch.matmul(q[bi, ni].float(), k[bi, ni].float().T) * scale_val
            causal_mask = torch.triu(torch.ones(sq, skv, dtype=torch.bool, device=q.device), diagonal=1)
            qk = qk.masked_fill(causal_mask, float('-inf'))
            attn = torch.softmax(qk, dim=-1)
            o_ref[bi, ni] = torch.matmul(attn, v[bi, ni].float()).half()
    return o_ref


def test_fa_causal_bn():
    compiled = fe.compile(fa_causal_kernel_bn, arch="a3")
    print("compiled:", compiled.lib_path)
    device = "npu:5"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    
    for b, n, sq, skv, d, num_cores in [
        (1, 1, 256, 256, TD, 1),
        (2, 2, 512, 512, TD, 4),
        (1, 4, 1024, 1024, TD, 4),
        (4, 2, 512, 512, TD, 8),
    ]:
        print(f"\nFA-Causal-BN (b={b}, n={n}, sq={sq}, skv={skv}, d={d}) cores={num_cores}")
        q = torch.rand((b, n, sq, d), device=device, dtype=torch.float16)
        k = torch.rand((b, n, skv, d), device=device, dtype=torch.float16)
        v = torch.rand((b, n, skv, d), device=device, dtype=torch.float16)
        o = torch.zeros((b, n, sq, d), device=device, dtype=torch.float16)
        qk_buf = torch.zeros((b, n, sq, skv), device=device, dtype=torch.float32)
        p_buf = torch.zeros((b, n, sq, skv), device=device, dtype=torch.float16)
        pv_buf = torch.zeros((48 * TS, d), device=device, dtype=torch.float32)
        
        attn_mask = torch.ones((sq, skv), device=device, dtype=torch.int8)
        causal_mask = torch.triu(torch.ones(sq, skv, dtype=torch.bool, device=device), diagonal=1)
        attn_mask[causal_mask] = 0
        
        total_work = b * n
        work_ranges = torch.zeros((num_cores, 2), device=device, dtype=torch.int32)
        
        work_per_core = (total_work + num_cores - 1) // num_cores
        for core in range(num_cores):
            work_start = core * work_per_core
            work_end = min((core + 1) * work_per_core, total_work)
            work_ranges[core, 0] = work_start
            work_ranges[core, 1] = work_end
        
        actual_num_cores = min(num_cores, total_work)
        fe.launch(None, actual_num_cores, compiled, q, k, v, o, qk_buf, p_buf, pv_buf, work_ranges, attn_mask)
        torch.npu.synchronize()

        print(f"  Q stats: min={q.min().item():.4f}, max={q.max().item():.4f}, mean={q.mean().item():.4f}")
        print(f"  K stats: min={k.min().item():.4f}, max={k.max().item():.4f}, mean={k.mean().item():.4f}")
        print(f"  V stats: min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}")
        print(f"  QK_buf stats: min={qk_buf.min().item():.4f}, max={qk_buf.max().item():.4f}, mean={qk_buf.mean().item():.4f}")
        print(f"  P_buf stats: min={p_buf.min().item():.4f}, max={p_buf.max().item():.4f}, mean={p_buf.mean().item():.4f}")
        print(f"  O stats: min={o.min().item():.4f}, max={o.max().item():.4f}, mean={o.mean().item():.4f}")
        
        if torch.isnan(qk_buf).any():
            print(f"  WARNING: QK_buf contains NaN!")
        if torch.isinf(qk_buf).any():
            print(f"  WARNING: QK_buf contains Inf!")
        if torch.isnan(p_buf).any():
            print(f"  WARNING: P_buf contains NaN!")
        if torch.isinf(p_buf).any():
            print(f"  WARNING: P_buf contains Inf!")
        if torch.isnan(o).any():
            print(f"  WARNING: O contains NaN!")
        if torch.isinf(o).any():
            print(f"  WARNING: O contains Inf!")
        
        o_ref = flash_attention_causal_ref_bn(q, k, v, d)
        print(f"  O_ref stats: min={o_ref.min().item():.4f}, max={o_ref.max().item():.4f}, mean={o_ref.mean().item():.4f}")
        
        diff = (o - o_ref).abs().max().item()
        print(f"  max|diff|={diff:.4f}")
        
        for ni in range(n):
            diff_head = (o[:, ni, :, :] - o_ref[:, ni, :, :]).abs().max().item()
            print(f"  Head {ni} max|diff|={diff_head:.4f}")
        
        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
        print("  PASS")


if __name__ == "__main__":
    print("FA Causal Attention - DN layout for K, multi-core + Q tiling + double buffer + BN axis")
    print("=" * 80)
    test_fa_causal_bn()
    print("\nAll FlashAttention Causal tests passed!")