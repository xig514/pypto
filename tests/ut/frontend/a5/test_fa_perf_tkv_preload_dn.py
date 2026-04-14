"""FlashAttention DN-mode performance kernel using PyPTO IR manual (non-SSA) mode.

Double-buffered cross-core communication + QK pre-compute pattern, DN layout.
Reference: fa_performance_dn_kernel.cpp

DN mode differences vs ND:
  - Q loaded with DN layout (transposed in hardware): GM[TS, TD] -> Mat[TD, TS]
  - QK acc result: [TKV, TS] (transposed from ND's [TS, TKV])
  - ACC->Vec via dual_split_n: qk_vec shape [TKV, TS_HALF] (vs ND's [TS_HALF, TKV])
  - Softmax: column-wise reduction (col_max/col_sum) on [TKV, TS_HALF] data
  - Global max/sum: shape [1, TS_HALF] row vectors (vs ND's [TS_HALF, 1] col vectors)
  - P mat for PV: [TS, TKV], loaded from GM with DN layout

Usage:
    python3 tests/ut/frontend/a5/test_fa_perf_tkv_preload_dn.py
"""

import math
import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.op.manual as plm

# ================================================================
#  Configuration — change QK_PRELOAD to tune pre-compute depth
# ================================================================
QK_PRELOAD = 1          # How many KV tiles to pre-compute QK ahead
FIFO_SIZE = QK_PRELOAD + 1  # Exp-corr FIFO depth (avoids read/write collision)

# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 128;  TKV = 128;  TD = 128
TS_HALF = TS // 2
SCALE = 1.0 / math.sqrt(TD)

# Cube tiles
Q_F16 = TS * TD * 2;   KT_F16 = TD * TKV * 2;  V_F16 = TKV * TD * 2
# DN: QK acc is [TKV, TS], same byte count as ND [TS, TKV]
P_F16 = TS * TKV * 2;  QK_HALF_F32 = TKV * TS * 4;  PV_HALF_F32 = TS * TD * 4

# ---- MAT (512KB) ----
MA0 = 0;                  MA0_PONG = MA0 + Q_F16
MA1 = Q_F16 * 2;          MA1_PONG = MA1 + KT_F16
MA2 = MA1 + KT_F16 * 2;   MA2_PONG = MA2 + P_F16
MA3 = MA2 + P_F16 * 2;    MA3_PONG = MA3 + V_F16
LA0 = 0;  LA1 = Q_F16
RA0 = 0;  RA1 = KT_F16
CA0 = 0;  CA1 = QK_HALF_F32

# ---- VEC addresses (248KB on a5) ----
# DN: qk_vec shape [TKV, TS_HALF], same byte count as ND [TS_HALF, TKV]
VB4_KV = TKV * TS_HALF * 4;  VB2_KV = TKV * TS_HALF * 2
VB4    = TS_HALF * TD * 4;   VB2    = TS_HALF * TD * 2
VB6    = (TKV + 1) * TS_HALF * 2   # DN tile_nz [TKV+1, TS_HALF] FP16
# DN: reduce shape [1, TS_HALF] — same byte count as ND [TS_HALF, 1]
VB_RED = 1 * TS_HALF * 4     # 256B — [1, TS_HALF] FP32

VA0  = 0                       # qk_vec [TKV, TS_HALF]
VA1  = VA0 + VB4_KV            # tmp_vec [TKV, TS_HALF]
VA2  = VA1 + VB4_KV            # p_f16 [TKV, TS_HALF]
VA3  = VA2 + VB2_KV            # reduce_dst [1, TS_HALF]
# global_max x 2 (by q_count % 2)
VA_GMAX0 = VA3 + VB_RED;  VA_GMAX1 = VA_GMAX0 + VB_RED
# global_sum x 2 (by q_count % 2)
VA_GSUM0 = VA_GMAX1 + VB_RED;  VA_GSUM1 = VA_GSUM0 + VB_RED
# exp_corr x FIFO_SIZE (by task_id % FIFO_SIZE)
VA_EXP_BASE = VA_GSUM1 + VB_RED
EXP_CORR_ADDRS = [VA_EXP_BASE + i * VB_RED for i in range(FIFO_SIZE)]
VA_AFTER_EXP = VA_EXP_BASE + FIFO_SIZE * VB_RED
VA7  = VA_AFTER_EXP            # running_o [TS_HALF, TD]
VA8  = VA7 + VB4               # pv_vec [TS_HALF, TD]
VA9  = VA8 + VB4               # o_f16 [TS_HALF, TD]
VA10 = VA9 + VB2
VA11 = VA10 + VB6       # qk_vec1 [TKV, TS_HALF]
VA12 = VA11 + VB4_KV    # pv_vec1 [TS_HALF, TD]
assert VA12 + VB4 <= 248 * 1024, f"VEC overflow: {VA12 + VB4} > {248*1024}"

event_ids_01 = (0, 1)
event_ids_23 = (2, 3)

# Cross-core event IDs
QK_READY_IDS = tuple(range(0, FIFO_SIZE))
P_READY_IDS  = tuple(range(FIFO_SIZE, 2 * FIFO_SIZE))
PV_READY_IDS = tuple(range(2 * FIFO_SIZE, 3 * FIFO_SIZE))
assert 3 * FIFO_SIZE <= 16, f"Too many cross-core event IDs: need {3*FIFO_SIZE}, max 16"
QK_MAX_EID = FIFO_SIZE
P_MAX_EID  = 2 * FIFO_SIZE
PV_MAX_EID = 3 * FIFO_SIZE

PV_CORE_STRIDE = 2 * FIFO_SIZE * TS

Sq2      = pl.DynVar('Sq')
Sq_fifo  = pl.DynVar('SqFifo')
Skv2     = pl.DynVar('Skv')
D2       = pl.DynVar('D')


# ================================================================
# Generate exp_corr FIFO allocator (static addresses, no subscript in kernel)
# ================================================================
import tempfile as _tf, importlib.util as _ilu, os as _os

def _gen_alloc_exp_corr():
    lines = ["import pypto.language as pl", "import pypto.language.op.manual as plm", ""]
    lines.append("def alloc_exp_corr_fifo():")
    names, col_names = [], []
    for i, addr in enumerate(EXP_CORR_ADDRS):
        lines.append(f"    ec{i} = plm.make_tile(plm.TileType(shape=[1, {TS_HALF}], dtype=pl.FP32, "
                     f"target_memory=pl.MemorySpace.Vec), addr={addr}, size={VB_RED})")
        lines.append(f"    ec{i}_col = plm.make_tile(plm.TileType(shape=[{TS_HALF}, 1], dtype=pl.FP32, "
                     f"target_memory=pl.MemorySpace.Vec, blayout=2), addr={addr}, size={VB_RED})")
        names.append(f"ec{i}"); col_names.append(f"ec{i}_col")
    lines.append(f"    return ({', '.join(names)}), ({', '.join(col_names)})")
    src = "\n".join(lines) + "\n"
    tmp = _os.path.join(_tf.gettempdir(), "_alloc_exp_corr_dn_fifo.py")
    with open(tmp, "w") as f:
        f.write(src)
    spec = _ilu.spec_from_file_location("_alloc_exp_corr_dn_fifo", tmp)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.alloc_exp_corr_fifo

alloc_exp_corr_fifo = _gen_alloc_exp_corr()


# ================================================================
def alloc_cube_buffer():
    # DN mode: Q mat is [TD, TS] (transposed from ND [TS, TD])
    q_mat_type = plm.TileType(shape=[TD, TS], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat, blayout=2, slayout=1)
    q_mat_0 = plm.make_tile(q_mat_type, addr=MA0, size=Q_F16)
    q_mat_1 = plm.make_tile(q_mat_type, addr=MA0_PONG, size=Q_F16)

    k_mat_type = plm.TileType(shape=[TD, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat, blayout=1, slayout=2)
    k_mat_0 = plm.make_tile(k_mat_type, addr=MA1, size=KT_F16)
    k_mat_1 = plm.make_tile(k_mat_type, addr=MA1_PONG, size=KT_F16)

    v_mat_type = plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
    v_mat_0 = plm.make_tile(v_mat_type, addr=MA3, size=V_F16)
    v_mat_1 = plm.make_tile(v_mat_type, addr=MA3_PONG, size=V_F16)

    left_0 = plm.make_tile(plm.TileType(shape=[TD, TS], dtype=pl.FP16, target_memory=pl.MemorySpace.Left, blayout=2, slayout=1), addr=LA0, size=Q_F16)
    left_1 = plm.make_tile(plm.TileType(shape=[TD, TS], dtype=pl.FP16, target_memory=pl.MemorySpace.Left, blayout=2, slayout=1), addr=LA1, size=Q_F16)
    right_0 = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Right, blayout=1, slayout=2), addr=RA0, size=KT_F16)
    right_1 = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Right, blayout=1, slayout=2), addr=RA1, size=KT_F16)

    # DN QK acc: [TKV, TS]
    acc_buf1 = plm.make_tile(plm.TileType(shape=[TKV, TS], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, blayout=2, slayout=1, fractal=1024), addr=CA0, size=QK_HALF_F32)
    acc_buf2 = plm.make_tile(plm.TileType(shape=[TKV, TS], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, blayout=2, slayout=1, fractal=1024), addr=CA1, size=QK_HALF_F32)

    return q_mat_0, q_mat_1, k_mat_0, k_mat_1, v_mat_0, v_mat_1, left_0, left_1, right_0, right_1, acc_buf1, acc_buf2


def compute_qk(ctx, state, const_info):
    """DN QK = Q^T * K -> ACC[TKV,TS] -> Vec[TKV,TS_HALF] via dual_split_n."""
    qk_fifo_slot = ctx.task_id % FIFO_SIZE
    skv_off = ctx.ki * TKV
    buf_idx = (ctx.q_count * ctx.skv_tiles + ctx.ki) % 2
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_01[buf_idx])
    if ctx.ki == 0:
        if ctx.q_count % 2 == 0:
            plm.load(q_mat_0, q, [ctx.sq_off, 0], layout="dn")
        else:
            plm.load(q_mat_1, q, [ctx.sq_off, 0], layout="dn")
    if buf_idx == 0:
        plm.load(k_mat_0, k, [skv_off, 0], layout="dn")
    else:
        plm.load(k_mat_1, k, [skv_off, 0], layout="dn")
    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[state.l0ab_idx])
    if state.l0ab_idx == 0:
        if ctx.q_count % 2 == 0:
            plm.move(left_0, q_mat_0)
        else:
            plm.move(left_0, q_mat_1)
    else:
        if ctx.q_count % 2 == 0:
            plm.move(left_1, q_mat_0)
        else:
            plm.move(left_1, q_mat_1)
    if state.l0ab_idx == 0:
        if buf_idx == 0:
            plm.move(right_0, k_mat_0)
        else:
            plm.move(right_0, k_mat_1)
    else:
        if buf_idx == 0:
            plm.move(right_1, k_mat_0)
        else:
            plm.move(right_1, k_mat_1)
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_01[buf_idx])
    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[state.l0c_idx])
    if state.l0c_idx == 0:
        if state.l0ab_idx == 0:
            plm.matmul(acc_buf1, left_0, right_0)
        else:
            plm.matmul(acc_buf1, left_1, right_1)
    else:
        if state.l0ab_idx == 0:
            plm.matmul(acc_buf2, left_0, right_0)
        else:
            plm.matmul(acc_buf2, left_1, right_1)
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[state.l0ab_idx])
    # DN: dual_split_n -> qk_vec shape [TKV, TS_HALF]
    if qk_fifo_slot == 0:
        if state.l0c_idx == 0:
            plm.move(qk_vec, acc_buf1, acc_to_vec_mode="dual_split_n")
        else:
            plm.move(qk_vec, acc_buf2, acc_to_vec_mode="dual_split_n")
    else:
        if state.l0c_idx == 0:
            plm.move(qk_vec1, acc_buf1, acc_to_vec_mode="dual_split_n")
        else:
            plm.move(qk_vec1, acc_buf2, acc_to_vec_mode="dual_split_n")
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[state.l0c_idx])
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_IDS[qk_fifo_slot], max_event_id=QK_MAX_EID)
    state.l0ab_idx = 1 - state.l0ab_idx
    state.l0c_idx = 1 - state.l0c_idx
    return


def compute_pv(ctx, state, const_info):
    """PV = P * V -> ACC[TS,TD] -> Vec[TS_HALF,TD] via dual_split_m (same as ND)."""
    pv_task_slot = ctx.task_id % FIFO_SIZE
    sv_off = ctx.ki * TKV
    pv_fifo_slot = ctx.task_id % FIFO_SIZE
    buf_idx = (ctx.q_count * ctx.skv_tiles + ctx.ki) % 2
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_23[buf_idx])
    if buf_idx == 0:
        plm.load(v_mat_0, v, [sv_off, 0])
    else:
        plm.load(v_mat_1, v, [sv_off, 0])
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_IDS[pv_fifo_slot], max_event_id=P_MAX_EID)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[state.l0ab_idx])
    if buf_idx == 0:
        if state.l0ab_idx == 0:
            plm.move(left_0, p_mat_buf1)
        else:
            plm.move(left_1, p_mat_buf1)
    else:
        if state.l0ab_idx == 0:
            plm.move(left_0, p_mat_buf2)
        else:
            plm.move(left_1, p_mat_buf2)
    if state.l0ab_idx == 0:
        if buf_idx == 0:
            plm.move(right_0, v_mat_0)
        else:
            plm.move(right_0, v_mat_1)
    else:
        if buf_idx == 0:
            plm.move(right_1, v_mat_0)
        else:
            plm.move(right_1, v_mat_1)
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_23[buf_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[state.l0c_idx])
    if state.l0c_idx == 0:
        if state.l0ab_idx == 0:
            plm.matmul(acc_buf_pv, left_0, right_0)
        else:
            plm.matmul(acc_buf_pv, left_1, right_1)
    else:
        if state.l0ab_idx == 0:
            plm.matmul(acc_buf_pv2, left_0, right_0)
        else:
            plm.matmul(acc_buf_pv2, left_1, right_1)
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[state.l0ab_idx])
    if pv_task_slot == 0:
        if state.l0c_idx == 0:
            plm.move(pv_vec, acc_buf_pv, acc_to_vec_mode="dual_split_m")
        else:
            plm.move(pv_vec, acc_buf_pv2, acc_to_vec_mode="dual_split_m")
    else:
        if state.l0c_idx == 0:
            plm.move(pv_vec1, acc_buf_pv, acc_to_vec_mode="dual_split_m")
        else:
            plm.move(pv_vec1, acc_buf_pv2, acc_to_vec_mode="dual_split_m")
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[state.l0c_idx])
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_IDS[pv_task_slot], max_event_id=PV_MAX_EID)
    state.l0ab_idx = 1 - state.l0ab_idx
    state.l0c_idx = 1 - state.l0c_idx
    return


@pl.inline
def softmax_body(ctx, sq_dim, row_off):
    """DN softmax body. qk_vec shape [TKV, TS_HALF]; use col_max/col_sum.
    Accesses gmax_0/1, gsum_0/1, exp_corr_0/1 from enclosing scope.
    """
    p_fifo_slot = ctx.task_id % FIFO_SIZE
    skv_off = ctx.ki * TKV
    q_idx = ctx.q_count % 2
    buf_idx = (ctx.q_count * ctx.skv_tiles + ctx.ki) % 2
    sub_id = pl.block.index_cast(pl.block.get_subblock_idx())
    if q_idx == 0:
        if p_fifo_slot == 0:
            if ctx.ki == 0:
                # col_max(out, tile, tmp): out = col_max(qk_vec) into reduce_dst [1,TS_HALF]
                plm.col_max(reduce_dst, qk_vec, tmp_vec)
                pl.system.bar_v()
                # col_expand_sub(out, tile, col_vec): tmp_vec = qk_vec - broadcast_col(reduce_dst)
                plm.col_expand_sub(tmp_vec, qk_vec, reduce_dst)
                plm.muls(gmax_0, reduce_dst, 1.0)
                plm.muls(tmp_vec, tmp_vec, SCALE)
                plm.exp(qk_vec, tmp_vec)
                pl.system.bar_v()
                plm.col_sum(reduce_dst, qk_vec, tmp_vec)
                pl.system.bar_v()
                plm.muls(gsum_0, reduce_dst, 1.0)
                plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
            if ctx.ki > 0:
                plm.col_max(reduce_dst, qk_vec, tmp_vec)
                pl.system.bar_v()
                plm.maximum(reduce_dst, reduce_dst, gmax_0)
                pl.system.bar_v()
                plm.sub(exp_corr_0, gmax_0, reduce_dst)
                pl.system.bar_v()
                plm.muls(gmax_0, reduce_dst, 1.0)
                pl.system.bar_v()
                plm.col_expand_sub(tmp_vec, qk_vec, reduce_dst)
                plm.muls(exp_corr_0, exp_corr_0, SCALE)
                plm.muls(tmp_vec, tmp_vec, SCALE)
                plm.exp(exp_corr_0, exp_corr_0)
                plm.exp(qk_vec, tmp_vec)
                plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
                pl.system.bar_v()
                plm.mul(gsum_0, gsum_0, exp_corr_0)
                plm.col_sum(reduce_dst, qk_vec, tmp_vec)
                pl.system.bar_v()
                plm.add(gsum_0, gsum_0, reduce_dst)
        elif p_fifo_slot == 1:
            if ctx.ki == 0:
                plm.col_max(reduce_dst, qk_vec1, tmp_vec)
                pl.system.bar_v()
                plm.col_expand_sub(tmp_vec, qk_vec1, reduce_dst)
                plm.muls(gmax_0, reduce_dst, 1.0)
                plm.muls(tmp_vec, tmp_vec, SCALE)
                plm.exp(qk_vec1, tmp_vec)
                pl.system.bar_v()
                plm.col_sum(reduce_dst, qk_vec1, tmp_vec)
                pl.system.bar_v()
                plm.muls(gsum_0, reduce_dst, 1.0)
                plm.cast(p_f16, qk_vec1, target_type=pl.FP16, mode="round")
            if ctx.ki > 0:
                plm.col_max(reduce_dst, qk_vec1, tmp_vec)
                pl.system.bar_v()
                plm.maximum(reduce_dst, reduce_dst, gmax_0)
                pl.system.bar_v()
                plm.sub(exp_corr_1, gmax_0, reduce_dst)
                pl.system.bar_v()
                plm.muls(gmax_0, reduce_dst, 1.0)
                pl.system.bar_v()
                plm.col_expand_sub(tmp_vec, qk_vec1, reduce_dst)
                plm.muls(exp_corr_1, exp_corr_1, SCALE)
                plm.muls(tmp_vec, tmp_vec, SCALE)
                plm.exp(exp_corr_1, exp_corr_1)
                plm.exp(qk_vec1, tmp_vec)
                plm.cast(p_f16, qk_vec1, target_type=pl.FP16, mode="round")
                pl.system.bar_v()
                plm.mul(gsum_0, gsum_0, exp_corr_1)
                plm.col_sum(reduce_dst, qk_vec1, tmp_vec)
                pl.system.bar_v()
                plm.add(gsum_0, gsum_0, reduce_dst)
    else:
        if p_fifo_slot == 0:
            if ctx.ki == 0:
                plm.col_max(reduce_dst, qk_vec, tmp_vec)
                pl.system.bar_v()
                plm.col_expand_sub(tmp_vec, qk_vec, reduce_dst)
                plm.muls(gmax_1, reduce_dst, 1.0)
                plm.muls(tmp_vec, tmp_vec, SCALE)
                plm.exp(qk_vec, tmp_vec)
                pl.system.bar_v()
                plm.col_sum(reduce_dst, qk_vec, tmp_vec)
                pl.system.bar_v()
                plm.muls(gsum_1, reduce_dst, 1.0)
                plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
            if ctx.ki > 0:
                plm.col_max(reduce_dst, qk_vec, tmp_vec)
                pl.system.bar_v()
                plm.maximum(reduce_dst, reduce_dst, gmax_1)
                pl.system.bar_v()
                plm.sub(exp_corr_0, gmax_1, reduce_dst)
                pl.system.bar_v()
                plm.muls(gmax_1, reduce_dst, 1.0)
                pl.system.bar_v()
                plm.col_expand_sub(tmp_vec, qk_vec, reduce_dst)
                plm.muls(exp_corr_0, exp_corr_0, SCALE)
                plm.muls(tmp_vec, tmp_vec, SCALE)
                plm.exp(exp_corr_0, exp_corr_0)
                plm.exp(qk_vec, tmp_vec)
                plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
                pl.system.bar_v()
                plm.mul(gsum_1, gsum_1, exp_corr_0)
                plm.col_sum(reduce_dst, qk_vec, tmp_vec)
                pl.system.bar_v()
                plm.add(gsum_1, gsum_1, reduce_dst)
        elif p_fifo_slot == 1:
            if ctx.ki == 0:
                plm.col_max(reduce_dst, qk_vec1, tmp_vec)
                pl.system.bar_v()
                plm.col_expand_sub(tmp_vec, qk_vec1, reduce_dst)
                plm.muls(gmax_1, reduce_dst, 1.0)
                plm.muls(tmp_vec, tmp_vec, SCALE)
                plm.exp(qk_vec1, tmp_vec)
                pl.system.bar_v()
                plm.col_sum(reduce_dst, qk_vec1, tmp_vec)
                pl.system.bar_v()
                plm.muls(gsum_1, reduce_dst, 1.0)
                plm.cast(p_f16, qk_vec1, target_type=pl.FP16, mode="round")
            if ctx.ki > 0:
                plm.col_max(reduce_dst, qk_vec1, tmp_vec)
                pl.system.bar_v()
                plm.maximum(reduce_dst, reduce_dst, gmax_1)
                pl.system.bar_v()
                plm.sub(exp_corr_1, gmax_1, reduce_dst)
                pl.system.bar_v()
                plm.muls(gmax_1, reduce_dst, 1.0)
                pl.system.bar_v()
                plm.col_expand_sub(tmp_vec, qk_vec1, reduce_dst)
                plm.muls(exp_corr_1, exp_corr_1, SCALE)
                plm.muls(tmp_vec, tmp_vec, SCALE)
                plm.exp(exp_corr_1, exp_corr_1)
                plm.exp(qk_vec1, tmp_vec)
                plm.cast(p_f16, qk_vec1, target_type=pl.FP16, mode="round")
                pl.system.bar_v()
                plm.mul(gsum_1, gsum_1, exp_corr_1)
                plm.col_sum(reduce_dst, qk_vec1, tmp_vec)
                pl.system.bar_v()
                plm.add(gsum_1, gsum_1, reduce_dst)
    plm.move(tile_nz, p_f16)
    pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
    if buf_idx == 0:
        plm.insert(p_mat_buf1, tile_nz, offset=(64 * sub_id * 32))
    else:
        plm.insert(p_mat_buf2, tile_nz, offset=(64 * sub_id * 32))
    return


@pl.inline
def compute_p(ctx, sq_dim, row_off):
    """Softmax on QK tile -> P. Includes cross-core sync."""
    p_fifo_slot = ctx.task_id % FIFO_SIZE
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_IDS[p_fifo_slot], max_event_id=QK_MAX_EID)
    softmax_body(ctx, sq_dim, row_off)
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_IDS[p_fifo_slot], max_event_id=P_MAX_EID)
    return


def compute_gu(ctx, row_off):
    """GU: running output update. exp_corr_col shape [TS_HALF,1] for row_expand_mul."""
    pv_slot = ctx.task_id % FIFO_SIZE
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_IDS[pv_slot], max_event_id=PV_MAX_EID)
    if pv_slot == 0:
        if ctx.ki == 0:
            plm.move(running_o, pv_vec)
        if ctx.ki > 0:
            # exp_corr_col_0 is [TS_HALF, 1] alias of exp_corr_0 [1, TS_HALF]
            plm.row_expand_mul(running_o, running_o, exp_corr_col_0)
            plm.add(running_o, running_o, pv_vec)
    else:
        if ctx.ki == 0:
            plm.move(running_o, pv_vec1)
        if ctx.ki > 0:
            plm.row_expand_mul(running_o, running_o, exp_corr_col_1)
            plm.add(running_o, running_o, pv_vec1)
    if ctx.ki == ctx.skv_tiles - 1:
        # gsum_col_0/1 is [TS_HALF, 1] alias for row_expand_div
        if ctx.q_count % 2 == 0:
            plm.row_expand_div(running_o, running_o, gsum_col_0)
        else:
            plm.row_expand_div(running_o, running_o, gsum_col_1)
        plm.cast(o_f16, running_o, target_type=pl.FP16, mode="round")
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        plm.store(o, o_f16, [ctx.sq_off + row_off, 0])
    return


# ================================================================
#  Kernel
# ================================================================
@fe.kernel
def fa_perf_tkv_preload_dn_kernel(
    q: pl.Tensor[[Sq2, D2], pl.FP16],
    k: pl.Tensor[[Skv2, D2], pl.FP16],
    v: pl.Tensor[[Skv2, D2], pl.FP16],
    o: pl.Tensor[[Sq2, D2], pl.FP16],
    qk_buf: pl.Tensor[[Sq_fifo, Skv2], pl.FP32],
    p_buf:  pl.Tensor[[Sq_fifo, Skv2], pl.FP16],
    pv_buf: pl.Tensor[[48 * PV_CORE_STRIDE, D2], pl.FP32],
) -> pl.Tensor[[Sq2, D2], pl.FP16]:

    sq_dim = Sq2
    skv_dim = Skv2
    sq_tiles = (sq_dim + (TS - 1)) // TS
    skv_tiles = (skv_dim + (TKV - 1)) // TKV
    num_cores = pl.block.index_cast(pl.block.get_block_num())
    core_id = pl.block.index_cast(pl.block.get_block_idx())

    # DN: qk_vec shape [TKV, TS_HALF]
    qk_vec  = plm.make_tile(plm.TileType(shape=[TKV, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA0,  size=VB4_KV)
    qk_vec1 = plm.make_tile(plm.TileType(shape=[TKV, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA11, size=VB4_KV)
    pv_vec  = plm.make_tile(plm.TileType(shape=[TS_HALF, TD],  dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA8,  size=VB4)
    pv_vec1 = plm.make_tile(plm.TileType(shape=[TS_HALF, TD],  dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA12, size=VB4)
    running_o = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA7, size=VB4)

    p_mat_type = plm.TileType(shape=[TS, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat, blayout=2, slayout=1)
    p_mat_buf1 = plm.make_tile(p_mat_type, addr=MA2,      size=P_F16)
    p_mat_buf2 = plm.make_tile(p_mat_type, addr=MA2_PONG, size=P_F16)

    # =================== CUBE SECTION ===================
    with pl.section_cube():
        q_mat_0, q_mat_1, k_mat_0, k_mat_1, v_mat_0, v_mat_1, left_0, left_1, right_0, right_1, acc_buf1, acc_buf2 = alloc_cube_buffer()
        # PV acc: [TS, TD]
        acc_buf_pv  = plm.make_tile(plm.TileType(shape=[TS, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, blayout=2, slayout=1, fractal=1024), addr=0x10000, size=PV_HALF_F32)
        acc_buf_pv2 = plm.make_tile(plm.TileType(shape=[TS, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, blayout=2, slayout=1, fractal=1024), addr=0x0,     size=PV_HALF_F32)

        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=1)

        task_id = 0
        q_count = 0
        const_info = pl.struct(sq_dim=sq_dim, core_id=core_id)
        state = pl.struct(l0ab_idx=0, l0c_idx=0)
        ctx_arr = pl.StructArray(3, sq_off=0, task_id=0, qi=0, ki=0, skv_tiles=0, q_count=0)
        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS
            for ki in pl.range(0, skv_tiles):
                ctx_curr = ctx_arr[task_id % 3]
                ctx_curr.sq_off = sq_off
                ctx_curr.task_id = task_id
                ctx_curr.qi = qi
                ctx_curr.ki = ki
                ctx_curr.skv_tiles = skv_tiles
                ctx_curr.q_count = q_count
                compute_qk(ctx_curr, state, const_info)

                ctx_pre = ctx_arr[(task_id + 2) % 3]
                if task_id > 0:
                    compute_pv(ctx_pre, state, const_info)
                task_id = task_id + 1
            q_count = q_count + 1

        compute_pv(ctx_arr[(task_id + 2) % 3], state, const_info)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=1)

    # =================== VECTOR SECTION ===================
    with pl.section_vector():
        # DN: tmp_vec and p_f16 match qk_vec shape [TKV, TS_HALF]
        tmp_vec    = plm.make_tile(plm.TileType(shape=[TKV, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA1, size=VB4_KV)
        p_f16      = plm.make_tile(plm.TileType(shape=[TKV, TS_HALF], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA2, size=VB2_KV)
        # DN: reduce_dst shape [1, TS_HALF] (row vector — result of col_max/col_sum)
        reduce_dst = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA3, size=VB_RED)

        red_type     = plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        red_col_type = plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2)

        # Double-buffered global_max / global_sum — shape [1, TS_HALF]
        gmax_0 = plm.make_tile(red_type, addr=VA_GMAX0, size=VB_RED)
        gmax_1 = plm.make_tile(red_type, addr=VA_GMAX1, size=VB_RED)
        gsum_0 = plm.make_tile(red_type, addr=VA_GSUM0, size=VB_RED)
        gsum_1 = plm.make_tile(red_type, addr=VA_GSUM1, size=VB_RED)
        # Column-vector views of gsum for row_expand_div in GU
        gsum_col_0 = plm.make_tile(red_col_type, addr=VA_GSUM0, size=VB_RED)
        gsum_col_1 = plm.make_tile(red_col_type, addr=VA_GSUM1, size=VB_RED)

        # FIFO exp_corr — shape [1, TS_HALF] for element-wise mul, [TS_HALF, 1] for row_expand_mul
        exp_corr_fifo, exp_corr_col_fifo = alloc_exp_corr_fifo()
        exp_corr_0     = exp_corr_fifo[0]
        exp_corr_col_0 = exp_corr_col_fifo[0]
        exp_corr_1     = exp_corr_fifo[1]
        exp_corr_col_1 = exp_corr_col_fifo[1]

        o_f16     = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA9, size=VB2)
        tile_type_nz = plm.TileType(shape=[TKV + 1, TS_HALF], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec, valid_shape=[TKV, TS_HALF], blayout=2, slayout=1)
        tile_nz = plm.make_tile(tile_type_nz, addr=VA10, size=VB6)

        task_id = 0
        q_count = 0
        sub_id = pl.block.index_cast(pl.block.get_subblock_idx())
        ctx_arr = pl.StructArray(3, sq_off=0, task_id=0, qi=0, ki=0, skv_tiles=0, q_count=0, core_id=0)

        row_off = sub_id * TS_HALF
        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS
            for ki in pl.range(0, skv_tiles):
                ctx_curr = ctx_arr[task_id % 3]
                ctx_curr.sq_off = sq_off
                ctx_curr.task_id = task_id
                ctx_curr.qi = qi
                ctx_curr.ki = ki
                ctx_curr.skv_tiles = skv_tiles
                ctx_curr.q_count = q_count
                ctx_curr.core_id = core_id
                if task_id > 0:
                    compute_p(ctx_arr[(task_id + 2) % 3], sq_dim, row_off)
                if task_id > 1:
                    compute_gu(ctx_arr[(task_id + 1) % 3], row_off)
                task_id = task_id + 1
            q_count = q_count + 1

        compute_p(ctx_arr[(task_id + 2) % 3], sq_dim, row_off)
        if task_id > 1:
            compute_gu(ctx_arr[(task_id + 1) % 3], row_off)
        task_id = task_id + 1
        compute_gu(ctx_arr[(task_id + 1) % 3], row_off)
    return o


# ================================================================
#  Reference + Tests
# ================================================================
def flash_attention_ref(q, k, v, d):
    scale_val = 1.0 / math.sqrt(d)
    qk = torch.matmul(q.float(), k.float().T)
    scale = qk * scale_val
    max_val = torch.max(scale, dim=-1, keepdim=True).values
    x_sub = scale - max_val
    x_exp = torch.exp(x_sub)
    attn = x_exp / torch.sum(x_exp, dim=-1, keepdim=True)
    return qk, x_exp, torch.matmul(attn, v.float()).half()


def test_fa_perf_dn():
    compiled = fe.compile(fa_perf_tkv_preload_dn_kernel, arch="a5", codegen_mode="cce")
    print("compiled:", compiled.lib_path)
    device = "npu:0"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    for sq, skv, d, num_cores in [
        (8192, 8192, TD, 28),
    ]:
        print(f"\nFA-Perf DN ({sq},{skv},{d}) cores={num_cores}  QK_PRELOAD={QK_PRELOAD}")
        q_t = torch.rand((sq, d), device=device, dtype=torch.float16)
        k_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        v_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        o_t = torch.zeros((sq, d), device=device, dtype=torch.float16)
        qk_t = torch.zeros((sq * FIFO_SIZE, skv), device=device, dtype=torch.float32)
        p_t  = torch.zeros((sq * FIFO_SIZE, skv), device=device, dtype=torch.float16)
        pv_t = torch.zeros((48 * PV_CORE_STRIDE, d), device=device, dtype=torch.float32)
        fe.launch(None, num_cores, compiled, q_t, k_t, v_t, o_t, qk_t, p_t, pv_t)
        torch.npu.synchronize()
        qk_ref, x_exp_ref, o_ref = flash_attention_ref(q_t, k_t, v_t, d)
        diff = (o_t - o_ref).abs().max().item()
        print(f"  max|diff|={diff:.4f}")
        torch.testing.assert_close(o_t, o_ref, rtol=5e-3, atol=5e-3)
        print("  PASS")


if __name__ == "__main__":
    print(f"FA perf DN: double-buffer + QK pre-compute (QK_PRELOAD={QK_PRELOAD}, FIFO={FIFO_SIZE})")
    print("=" * 60)
    test_fa_perf_dn()
    print("\nAll FlashAttention DN tests passed!")
