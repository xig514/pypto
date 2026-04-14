"""FlashAttention performance kernel using PyPTO IR manual (non-SSA) mode.

Double-buffered cross-core communication + QK pre-compute pattern.
Reference: fa_performance_kernel.cpp

Features:
  1. Multi-core: each Cube core processes multiple Q tiles via strided loop
  2. Double buffer: L1 ping/pong for K/P/V MAT tiles
  3. FIFO cross-core GM buffers (qk_buf, p_buf) with configurable depth
  4. Cross-core event ID ping/pong (QK_READY_0/1, P_READY_0/1, PV_READY_0/1)
  5. QK pre-compute: Cube runs QK_PRELOAD tiles ahead, then QK[i+preload] + PV[i]
  6. Vector: FIFO exp_corr (by task_id % FIFO_SIZE),
            double-buffered global_max/global_sum (by q_count % 2)

Usage:
    python3 tests/ut/frontend/flash_attention/test_fa_performance.py
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
                        # (configurable, max ~6; requires skv_tiles >= QK_PRELOAD)
FIFO_SIZE = QK_PRELOAD + 1  # Exp-corr FIFO depth (avoids read/write collision)

# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 256;  TKV = 1024;  TD = 128
baseM = 128;  baseN = 128;  baseK = 128
CUBE_CORE_NUM = 24        # cube 核数（写死）
TS_HALF = TS // 2
M_BLOCKS = TS // baseM            # q_mat_buf sub-tile count per double-buffer slot
SOFTMAX_ROWS = 8                  # softmax 每次处理行数
GU_ROWS = 64                      # compute_gu 每次处理行数
VEC_ROW_BLOCKS = TS_HALF // SOFTMAX_ROWS  # 16
GU_BLOCKS = TS_HALF // GU_ROWS           # 2
SCALE = 1.0 / math.sqrt(TD)

# ---- MAT buffer sizes (per base-block) ----
Q_SUB_F16 = baseM * TD * 2          # 每个 q_mat 子 tile 32KB
KT_F16 = TD * baseN * 2             # k_mat: 32KB
P_F16  = baseM * baseK * 2          # p_mat: 32KB
V_F16  = baseK * TD * 2             # v_mat: 32KB
QK_HALF_F32 = baseM * baseN * 4     # acc for QK: 64KB
PV_HALF_F32 = baseM * TD * 4        # acc for PV: 64KB

# ---- MAT addresses (512KB) ----
MA0_0    = 0                                        # q_mat slot 0, sub-tile 0
MA0_1    = MA0_0 + Q_SUB_F16                        # q_mat slot 0, sub-tile 1
MA0_PONG_0 = MA0_1 + Q_SUB_F16                     # q_mat slot 1, sub-tile 0
MA0_PONG_1 = MA0_PONG_0 + Q_SUB_F16                # q_mat slot 1, sub-tile 1
MA1      = MA0_PONG_1 + Q_SUB_F16                   # k_mat slot 0
MA1_PONG = MA1 + KT_F16                            # k_mat slot 1
MA2      = MA1_PONG + KT_F16                       # p_mat slot 0
MA2_PONG = MA2 + P_F16                             # p_mat slot 1
MA3      = MA2_PONG + P_F16                        # v_mat slot 0
MA3_PONG = MA3 + V_F16                             # v_mat slot 1
assert MA3_PONG + V_F16 <= 512 * 1024, f"MAT overflow: {MA3_PONG + V_F16} > {512*1024}"
LEFT_SIZE = baseM * baseK * 2
RIGHT_SIZE = baseN * baseK * 2
LA0 = 0;  LA1 = LEFT_SIZE                          # LEFT: 2 × [baseM, baseK] FP16
RA0 = 0;  RA1 = RIGHT_SIZE                         # RIGHT: 2 × [baseN, baseK] FP16
CA0 = 0;  CA1 = QK_HALF_F32                        # ACC: 2 × [baseM, baseN] FP32

# ---- VEC addresses (192KB) ----
VB4_KV = SOFTMAX_ROWS * TKV * 4     # qk_vec/tmp_vec: 8*1024*4 = 32KB
VB2_KV = SOFTMAX_ROWS * TKV * 2     # p_f16: 8*1024*2 = 16KB
VB_RED = SOFTMAX_ROWS * 1 * 4       # reduce_dst: 8*4 = 32B
VB4    = GU_ROWS * TD * 4           # running_o/pv_vec: 64*128*4 = 32KB
VB2    = GU_ROWS * TD * 2           # o_f16: 64*128*2 = 16KB

# Double-buffered qk_vec: 2 × [SOFTMAX_ROWS, TKV] FP32
VA_QK0 = 0                            # qk_vec slot 0
VA_QK1 = VA_QK0 + VB4_KV              # qk_vec slot 1
VA1  = VA_QK1 + VB4_KV                # tmp_vec (single buffer, shared with pv_vec)
# Double-buffered p_f16: 2 × [SOFTMAX_ROWS, TKV] FP16
VA_P0 = VA1 + VB4_KV                  # p_f16 slot 0
VA_P1 = VA_P0 + VB2_KV                # p_f16 slot 1
VA3  = VA_P1 + VB2_KV                 # reduce_dst [SOFTMAX_ROWS, 1] FP32
# global_max: 2 × VEC_ROW_BLOCKS 个 [SOFTMAX_ROWS, 1] 子 tile (by q_count % 2, by ri)
VA_GMAX_BASE = VA3 + VB_RED
VA_GMAX0 = [VA_GMAX_BASE + i * VB_RED for i in range(VEC_ROW_BLOCKS)]
VA_GMAX1 = [VA_GMAX0[-1] + VB_RED + i * VB_RED for i in range(VEC_ROW_BLOCKS)]
# global_sum: 同结构
VA_GSUM_BASE = VA_GMAX1[-1] + VB_RED
VA_GSUM0 = [VA_GSUM_BASE + i * VB_RED for i in range(VEC_ROW_BLOCKS)]
VA_GSUM1 = [VA_GSUM0[-1] + VB_RED + i * VB_RED for i in range(VEC_ROW_BLOCKS)]
# exp_corr: FIFO_SIZE × VEC_ROW_BLOCKS 个 [SOFTMAX_ROWS, 1] 子 tile
VA_EXP_BASE = VA_GSUM1[-1] + VB_RED
EXP_CORR_STRIDE = VEC_ROW_BLOCKS * VB_RED   # 每个 fifo slot 的 exp_corr 总大小
EXP_CORR_ADDRS = [[VA_EXP_BASE + s * EXP_CORR_STRIDE + i * VB_RED
                    for i in range(VEC_ROW_BLOCKS)] for s in range(FIFO_SIZE)]
VA_AFTER_EXP = VA_EXP_BASE + FIFO_SIZE * EXP_CORR_STRIDE
VA7  = VA_AFTER_EXP                 # running_o [GU_ROWS, TD] FP32
VA8  = VA1                          # pv_vec [GU_ROWS, TD] FP32 — REUSE tmp_vec address
VA9  = VA7 + VB4                    # o_f16  [GU_ROWS, TD] FP16 — separate address
VA_END = VA9 + VB2                  # end of o_f16
assert VA_END <= 192 * 1024, f"VEC overflow: {VA_END} > {192*1024}"

event_ids_01 = (0, 1)
event_ids_23 = (2, 3)
vec_evids_01 = (4, 5)    # VEC section: backward sync event IDs for qk_vec/p_f16 double buffer

# Cross-core event IDs (0-10 available on Ascend NPU)
# Values can overlap with intra-core event_ids (different hardware namespace)
QK_READY_IDS = tuple(range(0, FIFO_SIZE))
P_READY_IDS  = tuple(range(FIFO_SIZE, 2 * FIFO_SIZE))
PV_READY_IDS = tuple(range(2 * FIFO_SIZE, 3 * FIFO_SIZE))
assert 3 * FIFO_SIZE <= 11, f"Too many cross-core event IDs: need {3*FIFO_SIZE}, max 11"
# max_event_id for codegen: only emit branches up to the highest ID used per type
QK_MAX_EID = FIFO_SIZE
P_MAX_EID  = 2 * FIFO_SIZE
PV_MAX_EID = 3 * FIFO_SIZE

# PV buffer: 2 Q-slots × FIFO_SIZE task-slots per core
PV_CORE_STRIDE = 2 * FIFO_SIZE * TS

Sq2      = pl.DynVar('Sq')
Skv2     = pl.DynVar('Skv')
D2       = pl.DynVar('D')


# ================================================================
def alloc_cube_buffer():
    # q_mat: M_BLOCKS(=2) 个 [baseM, TD] 子 tile per double-buffer slot, 手动展开避免 range()
    q_mat_type = plm.TileType(shape=[baseM, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
    q_mat_0_0 = plm.make_tile(q_mat_type, addr=MA0_0, size=Q_SUB_F16)
    q_mat_0_1 = plm.make_tile(q_mat_type, addr=MA0_1, size=Q_SUB_F16)
    q_mat_1_0 = plm.make_tile(q_mat_type, addr=MA0_PONG_0, size=Q_SUB_F16)
    q_mat_1_1 = plm.make_tile(q_mat_type, addr=MA0_PONG_1, size=Q_SUB_F16)

    k_mat_type = plm.TileType(shape=[TD, baseN], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat, blayout=1, slayout=2)
    k_mat_0 = plm.make_tile(k_mat_type, addr=MA1, size=KT_F16)
    k_mat_1 = plm.make_tile(k_mat_type, addr=MA1_PONG, size=KT_F16)

    p_mat_type = plm.TileType(shape=[baseM, baseK], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
    p_mat_0 = plm.make_tile(p_mat_type, addr=MA2, size=P_F16)
    p_mat_1 = plm.make_tile(p_mat_type, addr=MA2_PONG, size=P_F16)

    v_mat_type = plm.TileType(shape=[baseK, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
    v_mat_0 = plm.make_tile(v_mat_type, addr=MA3, size=V_F16)
    v_mat_1 = plm.make_tile(v_mat_type, addr=MA3_PONG, size=V_F16)

    left_0 = plm.make_tile(plm.TileType(shape=[baseM, baseK], dtype=pl.FP16, target_memory=pl.MemorySpace.Left), addr=LA0, size=LEFT_SIZE)
    left_1 = plm.make_tile(plm.TileType(shape=[baseM, baseK], dtype=pl.FP16, target_memory=pl.MemorySpace.Left), addr=LA1, size=LEFT_SIZE)
    right_0 = plm.make_tile(plm.TileType(shape=[baseN, baseK], dtype=pl.FP16, target_memory=pl.MemorySpace.Right), addr=RA0, size=RIGHT_SIZE)
    right_1 = plm.make_tile(plm.TileType(shape=[baseN, baseK], dtype=pl.FP16, target_memory=pl.MemorySpace.Right), addr=RA1, size=RIGHT_SIZE)
    acc_0 = plm.make_tile(plm.TileType(shape=[baseM, baseN], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc), addr=CA0, size=QK_HALF_F32)
    acc_1 = plm.make_tile(plm.TileType(shape=[baseM, baseN], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc), addr=CA1, size=PV_HALF_F32)

    return ((q_mat_0_0, q_mat_0_1, q_mat_1_0, q_mat_1_1),
            (k_mat_0, k_mat_1), (p_mat_0, p_mat_1),
            (v_mat_0, v_mat_1), (left_0, left_1), (right_0, right_1), (acc_0, acc_1))


# Write alloc_vec_state to a temp .py so auto-inline can read its source.
# All buffers are FLAT 1D tuples to avoid nested-tuple dynamic indexing in IR.
# Access pattern: buf[q_idx * STRIDE + ri]  instead of buf[q_idx][ri]
import tempfile as _tf, importlib.util as _ilu, os as _os
def _gen_alloc_vec_state():
    lines = ["import pypto.language as pl", "import pypto.language.op.manual as plm", ""]
    lines.append("def alloc_vec_state():")

    def _flat_tuple(names):
        return "(" + ", ".join(names) + ")"

    # --- global_max_rm_buf: flat (2 * VEC_ROW_BLOCKS) 个 [1, SOFTMAX_ROWS] ---
    # index: q_idx * VEC_ROW_BLOCKS + ri
    gmax_names = []
    for q in range(2):
        addrs = VA_GMAX0 if q == 0 else VA_GMAX1
        for ri in range(VEC_ROW_BLOCKS):
            n = f"gmax_{q}_{ri}"
            lines.append(f"    {n} = plm.make_tile(plm.TileType(shape=[1, {SOFTMAX_ROWS}], dtype=pl.FP32, "
                         f"target_memory=pl.MemorySpace.Vec), addr={addrs[ri]}, size={VB_RED})")
            gmax_names.append(n)

    # --- global_sum_buf: flat (2 * VEC_ROW_BLOCKS) 个 [SOFTMAX_ROWS, 1] ---
    gsum_names = []
    for q in range(2):
        addrs = VA_GSUM0 if q == 0 else VA_GSUM1
        for ri in range(VEC_ROW_BLOCKS):
            n = f"gsum_{q}_{ri}"
            lines.append(f"    {n} = plm.make_tile(plm.TileType(shape=[{SOFTMAX_ROWS}, 1], dtype=pl.FP32, "
                         f"target_memory=pl.MemorySpace.Vec, blayout=2), addr={addrs[ri]}, size={VB_RED})")
            gsum_names.append(n)

    # --- global_sum_rm_buf: flat (2 * VEC_ROW_BLOCKS) 个 [1, SOFTMAX_ROWS] ---
    gsum_rm_names = []
    for q in range(2):
        addrs = VA_GSUM0 if q == 0 else VA_GSUM1
        for ri in range(VEC_ROW_BLOCKS):
            n = f"gsum_rm_{q}_{ri}"
            lines.append(f"    {n} = plm.make_tile(plm.TileType(shape=[1, {SOFTMAX_ROWS}], dtype=pl.FP32, "
                         f"target_memory=pl.MemorySpace.Vec), addr={addrs[ri]}, size={VB_RED})")
            gsum_rm_names.append(n)

    # --- global_sum_buf_64: flat (2 * GU_BLOCKS) 个 [GU_ROWS, 1] ---
    # index: q_idx * GU_BLOCKS + gi
    gsum64_names = []
    softmax_per_gu = GU_ROWS // SOFTMAX_ROWS
    for q in range(2):
        addrs = VA_GSUM0 if q == 0 else VA_GSUM1
        for gi in range(GU_BLOCKS):
            n = f"gsum64_{q}_{gi}"
            lines.append(f"    {n} = plm.make_tile(plm.TileType(shape=[{GU_ROWS}, 1], dtype=pl.FP32, "
                         f"target_memory=pl.MemorySpace.Vec, blayout=2), addr={addrs[gi * softmax_per_gu]}, size={GU_ROWS * 4})")
            gsum64_names.append(n)

    # --- exp_corr_fifo: flat (FIFO_SIZE * VEC_ROW_BLOCKS) 个 [SOFTMAX_ROWS, 1] ---
    # index: slot * VEC_ROW_BLOCKS + ri
    ec_col_names = []
    ec_rm_names = []
    for s in range(FIFO_SIZE):
        for ri in range(VEC_ROW_BLOCKS):
            addr = EXP_CORR_ADDRS[s][ri]
            cn = f"ec{s}_{ri}"
            rn = f"ec{s}_{ri}_rm"
            lines.append(f"    {cn} = plm.make_tile(plm.TileType(shape=[{SOFTMAX_ROWS}, 1], dtype=pl.FP32, "
                         f"target_memory=pl.MemorySpace.Vec, blayout=2), addr={addr}, size={VB_RED})")
            lines.append(f"    {rn} = plm.make_tile(plm.TileType(shape=[1, {SOFTMAX_ROWS}], dtype=pl.FP32, "
                         f"target_memory=pl.MemorySpace.Vec), addr={addr}, size={VB_RED})")
            ec_col_names.append(cn); ec_rm_names.append(rn)

    # --- exp_corr_fifo_64: flat (FIFO_SIZE * GU_BLOCKS) 个 [GU_ROWS, 1] ---
    # index: slot * GU_BLOCKS + gi
    ec_gu_names = []
    for s in range(FIFO_SIZE):
        for gi in range(GU_BLOCKS):
            addr = EXP_CORR_ADDRS[s][gi * softmax_per_gu]
            gn = f"ec_gu{s}_{gi}"
            lines.append(f"    {gn} = plm.make_tile(plm.TileType(shape=[{GU_ROWS}, 1], dtype=pl.FP32, "
                         f"target_memory=pl.MemorySpace.Vec, blayout=2), addr={addr}, size={GU_ROWS * 4})")
            ec_gu_names.append(gn)

    lines.append(f"    return ({_flat_tuple(gmax_names)}, {_flat_tuple(gsum_names)}, "
                 f"{_flat_tuple(gsum_rm_names)}, {_flat_tuple(gsum64_names)}, "
                 f"{_flat_tuple(ec_col_names)}, {_flat_tuple(ec_rm_names)}, {_flat_tuple(ec_gu_names)})")
    src = "\n".join(lines) + "\n"
    tmp = _os.path.join(_tf.gettempdir(), "_alloc_vec_state.py")
    with open(tmp, "w") as f:
        f.write(src)
    spec = _ilu.spec_from_file_location("_alloc_vec_state", tmp)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.alloc_vec_state

alloc_vec_state = _gen_alloc_vec_state()


def compute_qk(ctx, state, const_info):
    """QK = Q * K^T with N/M loop blocking. N outer (K reuse), M inner.
    L0A/L0B optimization: Q moved to L0A only on first N iter, K moved to L0B once per N iter."""
    qk_fifo_slot = ctx.task_id % FIFO_SIZE
    skv_off = ctx.ki * TKV
    buf_idx = (ctx.q_count * ctx.skv_tiles + ctx.ki) % 2
    if ctx.ki == 0:
        for mi in pl.range(0, TS // baseM):
            plm.load(q_mat_buf[ctx.q_count % 2 * M_BLOCKS + mi], q, [ctx.sq_off + mi * baseM, 0])
    for ni in pl.range(0, TKV // baseN):
        k_buf_idx = ni % 2
        # Load K once per N iteration (reused across M iterations)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2,
                           event_id=event_ids_01[k_buf_idx])
        plm.load(k_mat_buf[k_buf_idx], k, [skv_off + ni * baseN, 0], layout="dn")
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        # Move Q to L0A only on first N iteration (L0A content persists across N)
        # Token NOT returned until compute_qk ends (Q occupies left_buf for all ni)
        if ni == 0:
            pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1,
                               event_id=event_ids_01[0])
            plm.move(left_buf[0], q_mat_buf[ctx.q_count % 2 * M_BLOCKS + 0])
            pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1,
                               event_id=event_ids_01[1])
            plm.move(left_buf[1], q_mat_buf[ctx.q_count % 2 * M_BLOCKS + 1])
        # Move K to L0B once per N iteration (uses event_id=2 to not conflict with Q)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1,
                           event_id=2)
        plm.move(right_buf[0], k_mat_buf[k_buf_idx])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        for mi in pl.range(0, TS // baseM):
            pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M,
                               event_id=event_ids_01[state.l0c_idx])
            plm.matmul(acc_buf[state.l0c_idx], left_buf[mi], right_buf[0])
            pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            plm.store(qk_buf, acc_buf[state.l0c_idx],
                          [const_info.core_id * FIFO_SIZE * TS + qk_fifo_slot * TS + mi * baseM,
                           ni * baseN])
            pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M,
                               event_id=event_ids_01[state.l0c_idx])
            state.l0c_idx = 1 - state.l0c_idx
        # M iterations done using right_buf[0]: release K's M→MTE1 token
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=2)
        # Also release k_mat_buf for next N load
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2,
                           event_id=event_ids_01[k_buf_idx])
    # N loop done: release Q's L0A tokens (event_id 0 and 1)
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[0])
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[1])
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_IDS[qk_fifo_slot],
                             max_event_id=QK_MAX_EID)
    return


def compute_pv(ctx, state, const_info):
    """PV = P * V with M/K loop blocking + matmul_acc accumulation."""
    pv_task_slot = ctx.task_id % FIFO_SIZE
    sv_off = ctx.ki * TKV
    pv_fifo_slot = ctx.task_id % FIFO_SIZE
    for mi in pl.range(0, TS // baseM):
        for ki_pv in pl.range(0, TKV // baseK):
            pv_buf_idx = ki_pv % 2
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2,
                               event_id=event_ids_23[pv_buf_idx])
            plm.load(v_mat_buf[pv_buf_idx], v, [sv_off + ki_pv * baseK, 0])
            if mi == 0:
                if ki_pv == 0:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2,
                                              event_id=P_READY_IDS[pv_fifo_slot],
                                              max_event_id=P_MAX_EID)
            plm.load(p_mat_buf[pv_buf_idx], p_buf,
                     [const_info.core_id * FIFO_SIZE * TS + pv_fifo_slot * TS + mi * baseM,
                      ki_pv * baseK])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1,
                               event_id=event_ids_01[state.l0ab_idx])
            plm.move(left_buf[state.l0ab_idx], p_mat_buf[pv_buf_idx])
            plm.move(right_buf[state.l0ab_idx], v_mat_buf[pv_buf_idx])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2,
                               event_id=event_ids_23[pv_buf_idx])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
            if ki_pv == 0:
                pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M,
                                   event_id=event_ids_01[state.l0c_idx])
                plm.matmul(acc_buf[state.l0c_idx],
                           left_buf[state.l0ab_idx], right_buf[state.l0ab_idx])
            if ki_pv > 0:
                plm.matmul_acc(acc_buf[state.l0c_idx], acc_buf[state.l0c_idx],
                               left_buf[state.l0ab_idx], right_buf[state.l0ab_idx])
            pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1,
                               event_id=event_ids_01[state.l0ab_idx])
            state.l0ab_idx = 1 - state.l0ab_idx
        plm.store(pv_buf, acc_buf[state.l0c_idx],
                      [const_info.core_id * PV_CORE_STRIDE +
                       (ctx.q_count % 2) * FIFO_SIZE * TS +
                       pv_task_slot * TS + mi * baseM, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M,
                           event_id=event_ids_01[state.l0c_idx])
        state.l0c_idx = 1 - state.l0c_idx
    pl.system.set_cross_core(pipe=pl.PipeType.FIX,
                             event_id=PV_READY_IDS[pv_task_slot],
                             max_event_id=PV_MAX_EID)
    return


@pl.inline
def softmax_body(ctx, row_off):
    """Softmax body with ri loop. Software-pipelined MTE2 loads + double-buffered qk_vec & p_f16.
    Opt 1: Removed barrier between muls(gmax_update) and row_expand_sub (independent buffers).
    Opt 2: Prologue starts first MTE2 load; each iteration prefetches next tile during V compute."""
    p_fifo_slot = ctx.task_id % FIFO_SIZE
    q_idx = ctx.q_count % 2
    N_RI = TS_HALF // SOFTMAX_ROWS  # 16
    # Software-pipeline prologue: start loading first qk_vec tile (buf[0])
    prologue_qk = qk_vec_buf[0]
    pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2,
                       event_id=vec_evids_01[0])
    plm.load(prologue_qk, qk_buf,
             [ctx.core_id * FIFO_SIZE * TS + p_fifo_slot * TS + row_off, 0])
    for ri in pl.range(0, N_RI):
        row_start = row_off + ri * SOFTMAX_ROWS
        buf_idx = ri % 2
        next_buf_idx = (ri + 1) % 2
        global_max_rm_cur = global_max_rm_buf[q_idx * VEC_ROW_BLOCKS + ri]
        global_sum_rm_cur = global_sum_rm_buf[q_idx * VEC_ROW_BLOCKS + ri]
        exp_corr_cur = exp_corr_rm_fifo[p_fifo_slot * VEC_ROW_BLOCKS + ri]
        qk_vec = qk_vec_buf[buf_idx]
        p_f16 = p_f16_buf[buf_idx]
        next_qk_vec = qk_vec_buf[next_buf_idx]
        # Wait for current tile's MTE2 load to complete
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        # Prefetch next tile (MTE2 runs in parallel with V compute below)
        if ri < N_RI - 1:
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2,
                               event_id=vec_evids_01[next_buf_idx])
            plm.load(next_qk_vec, qk_buf,
                     [ctx.core_id * FIFO_SIZE * TS + p_fifo_slot * TS +
                      row_off + (ri + 1) * SOFTMAX_ROWS, 0])
        # Backward sync: V waits for MTE3 to release p_f16[buf_idx] from 2 iters ago
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V,
                           event_id=vec_evids_01[buf_idx])
        if ctx.ki == 0:
            plm.row_max(reduce_dst, qk_vec, tmp_vec)
            pl.system.bar_v()
            plm.row_expand_sub(tmp_vec, qk_vec, reduce_dst)
            plm.muls(global_max_rm_cur, reduce_dst_rm, 1.0)
            plm.muls(tmp_vec, tmp_vec, SCALE)
            plm.exp(qk_vec, tmp_vec)
            pl.system.bar_v()
            plm.row_sum(reduce_dst, qk_vec, tmp_vec)
            pl.system.bar_v()
            plm.muls(global_sum_rm_cur, reduce_dst_rm, 1.0)
            plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
        if ctx.ki > 0:
            plm.row_max(reduce_dst, qk_vec, tmp_vec)
            pl.system.bar_v()
            plm.maximum(reduce_dst_rm, reduce_dst_rm, global_max_rm_cur)
            pl.system.bar_v()
            plm.sub(exp_corr_cur, global_max_rm_cur, reduce_dst_rm)
            pl.system.bar_v()
            plm.muls(global_max_rm_cur, reduce_dst_rm, 1.0)
            plm.row_expand_sub(tmp_vec, qk_vec, reduce_dst)
            plm.muls(exp_corr_cur, exp_corr_cur, SCALE)
            plm.muls(tmp_vec, tmp_vec, SCALE)
            plm.exp(exp_corr_cur, exp_corr_cur)
            plm.exp(qk_vec, tmp_vec)
            plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
            pl.system.bar_v()
            plm.mul(global_sum_rm_cur, global_sum_rm_cur, exp_corr_cur)
            plm.row_sum(reduce_dst, qk_vec, tmp_vec)
            pl.system.bar_v()
            plm.add(global_sum_rm_cur, global_sum_rm_cur, reduce_dst_rm)
        # Forward: V done reading qk_vec[buf_idx], release for MTE2
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2,
                           event_id=vec_evids_01[buf_idx])
        # Forward: V done writing p_f16[buf_idx], MTE3 can store
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        plm.store(p_buf, p_f16, [ctx.core_id * FIFO_SIZE * TS + p_fifo_slot * TS + row_start, 0])
        # Backward: MTE3 done reading p_f16[buf_idx], release for V
        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V,
                           event_id=vec_evids_01[buf_idx])
    return


@pl.inline
def compute_p(ctx, row_off):
    """Softmax on QK tile → P. Includes cross-core sync."""
    p_fifo_slot = ctx.task_id % FIFO_SIZE
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=QK_READY_IDS[p_fifo_slot], max_event_id=QK_MAX_EID)
    softmax_body(ctx, row_off)
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_IDS[p_fifo_slot], max_event_id=P_MAX_EID)
    return


def compute_gu(ctx, row_off):
    """GU: running output update with gi loop + ro_buf spill/reload."""
    pv_slot = ctx.task_id % FIFO_SIZE
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=PV_READY_IDS[pv_slot], max_event_id=PV_MAX_EID)
    for gi in pl.range(0, TS_HALF // GU_ROWS):
        row_start = row_off + gi * GU_ROWS
        exp_corr_gu_cur = exp_corr_fifo_64[pv_slot * GU_BLOCKS + gi]
        global_sum_gu_cur = global_sum_buf_64[ctx.q_count % 2 * GU_BLOCKS + gi]

        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=0)
        plm.load(pv_vec, pv_buf,
                    [ctx.core_id * PV_CORE_STRIDE +
                    (ctx.q_count % 2) * FIFO_SIZE * TS +
                    pv_slot * TS + row_start, 0])

        if ctx.ki != 0:
            plm.load(running_o, ro_buf,
                     [ctx.core_id * TS + row_start, 0])

        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        if ctx.ki == 0:
            plm.move(running_o, pv_vec)
        if ctx.ki != 0:
            plm.row_expand_mul(running_o, running_o, exp_corr_gu_cur)
            pl.system.bar_v()
            plm.add(running_o, running_o, pv_vec)
        if ctx.ki == ctx.skv_tiles - 1:
            pl.system.bar_v()
            plm.row_expand_div(running_o, running_o, global_sum_gu_cur)
            pl.system.bar_v()
            plm.cast(o_f16, running_o, target_type=pl.FP16, mode="round")
            # Forward: V done with running_o, release for MTE2 next iteration
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
            plm.store(o, o_f16, [ctx.sq_off + row_start, 0])
        if ctx.ki < ctx.skv_tiles - 1:
            # Forward: V done with running_o, release for MTE2 next iteration
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
            plm.store(ro_buf, running_o,
                      [ctx.core_id * TS + row_start, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE2, event_id=0)
    return


# ================================================================
#  Kernel
# ================================================================
@fe.kernel
def fa_perf_tkv_preload_kernel(
    q: pl.Tensor[[Sq2, D2], pl.FP16],
    k: pl.Tensor[[Skv2, D2], pl.FP16],
    v: pl.Tensor[[Skv2, D2], pl.FP16],
    o: pl.Tensor[[Sq2, D2], pl.FP16],
    qk_buf: pl.Tensor[[CUBE_CORE_NUM * FIFO_SIZE * TS, TKV], pl.FP32],  # per-core FIFO slots
    p_buf:  pl.Tensor[[CUBE_CORE_NUM * FIFO_SIZE * TS, TKV], pl.FP16],   # per-core FIFO slots
    pv_buf: pl.Tensor[[CUBE_CORE_NUM * PV_CORE_STRIDE, D2], pl.FP32],       # double-buffered per core
    ro_buf: pl.Tensor[[CUBE_CORE_NUM * TS, D2], pl.FP32],                   # running_o persistence
) -> pl.Tensor[[Sq2, D2], pl.FP16]:

    sq_dim = Sq2
    skv_dim = Skv2
    sq_tiles = (sq_dim + (TS - 1)) // TS
    skv_tiles = (skv_dim + (TKV - 1)) // TKV
    num_cores = pl.block.index_cast(pl.block.get_block_num())
    core_id = pl.block.index_cast(pl.block.get_block_idx())

    # =================== CUBE SECTION ===================
    with pl.section_cube():
        q_mat_buf, k_mat_buf, p_mat_buf, v_mat_buf, left_buf, right_buf, acc_buf = alloc_cube_buffer()

        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=2)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=1)
        
        task_id = 0
        q_count = 0
        const_info = pl.struct(core_id = core_id)
        state = pl.struct(l0ab_idx = 0, l0c_idx = 0)
        ctx_arr = pl.StructArray(3, sq_off=0, task_id=0, qi=0, ki=0, skv_tiles=0, q_count=0)
        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS
            # ---- Main loop: QK[ki+preload] ahead + PV[ki] current ----
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
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=1)

    # =================== VECTOR SECTION ===================
    # On a2/a3, two Vector sub-blocks run simultaneously, each handling half the
    # tile rows. We iterate over row halves (row_idx 0/1) so both sub-blocks
    # together cover all TS rows per Q tile.
    with pl.section_vector():
        # Double-buffered qk_vec (MTE2 writes, V reads)
        qk_vec_type = plm.TileType(shape=[SOFTMAX_ROWS, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        qk_vec_0   = plm.make_tile(qk_vec_type, addr=VA_QK0, size=VB4_KV)
        qk_vec_1   = plm.make_tile(qk_vec_type, addr=VA_QK1, size=VB4_KV)
        qk_vec_buf = (qk_vec_0, qk_vec_1)
        tmp_vec    = plm.make_tile(plm.TileType(shape=[SOFTMAX_ROWS, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA1, size=VB4_KV)
        # Double-buffered p_f16 (V writes, MTE3 reads)
        p_f16_type = plm.TileType(shape=[SOFTMAX_ROWS, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
        p_f16_0    = plm.make_tile(p_f16_type, addr=VA_P0, size=VB2_KV)
        p_f16_1    = plm.make_tile(p_f16_type, addr=VA_P1, size=VB2_KV)
        p_f16_buf  = (p_f16_0, p_f16_1)
        reduce_dst = plm.make_tile(plm.TileType(shape=[SOFTMAX_ROWS, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA3, size=VB_RED)
        reduce_dst_rm = plm.make_tile(plm.TileType(shape=[1, SOFTMAX_ROWS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA3, size=VB_RED)

        # All sub-tile buffers generated via temp .py to avoid range() in kernel
        (global_max_rm_buf, global_sum_buf, global_sum_rm_buf, global_sum_buf_64,
         exp_corr_fifo, exp_corr_rm_fifo, exp_corr_fifo_64) = alloc_vec_state()

        running_o = plm.make_tile(plm.TileType(shape=[GU_ROWS, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA7, size=VB4)
        pv_vec    = plm.make_tile(plm.TileType(shape=[GU_ROWS, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA8, size=VB4)
        o_f16     = plm.make_tile(plm.TileType(shape=[GU_ROWS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA9, size=VB2)
        task_id = 0
        q_count = 0
        sub_id = pl.block.index_cast(pl.block.get_subblock_idx())
        ctx_arr = pl.StructArray(3, sq_off=0, task_id=0, qi=0, ki=0, skv_tiles=0, q_count=0, core_id=0)

        row_off = sub_id * TS_HALF
        # Init backward sync counters for softmax double-buffered qk_vec (V→MTE2) and p_f16 (MTE3→V)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=vec_evids_01[0])
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=vec_evids_01[1])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=vec_evids_01[0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=vec_evids_01[1])

        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS

            # ---- Main loop: P[ki+preload] ahead + GU[ki] current ----
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
                    compute_p(ctx_arr[(task_id + 2) % 3], row_off)
                if task_id > 1:
                    compute_gu(ctx_arr[(task_id + 1) % 3], row_off)
                task_id = task_id + 1
            q_count = q_count + 1

        compute_p(ctx_arr[(task_id + 2) % 3], row_off)
        if task_id > 1:
            compute_gu(ctx_arr[(task_id + 1) % 3], row_off)
        task_id = task_id + 1
        compute_gu(ctx_arr[(task_id + 1) % 3], row_off)

        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=vec_evids_01[0])
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE2, event_id=vec_evids_01[1])
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=vec_evids_01[0])
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.V, event_id=vec_evids_01[1])
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
    row_sums = torch.sum(x_exp, dim=-1, keepdim=True)  # [sq, 1]
    attn = x_exp / row_sums
    return qk, x_exp, torch.matmul(attn, v.float()).half()


def test_fa_perf():
    compiled = fe.compile(fa_perf_tkv_preload_kernel, arch="a5", codegen_mode="cce")
    print("compiled:", compiled.lib_path)
    device = "npu:5"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    for sq, skv, d, num_cores in [
        # (128, 128, TD, 1),
        # (512, 512, TD, 4),
        (8192, 8192, TD, 24),
    ]:
        print(f"\nFA-Perf ({sq},{skv},{d}) cores={num_cores}  QK_PRELOAD={QK_PRELOAD}")
        q_t = torch.rand((sq, d), device=device, dtype=torch.float16)
        k_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        v_t = torch.rand((skv, d), device=device, dtype=torch.float16)
        o_t = torch.zeros((sq, d), device=device, dtype=torch.float16)
        qk_t = torch.zeros((CUBE_CORE_NUM * FIFO_SIZE * TS, TKV), device=device, dtype=torch.float32)
        p_t  = torch.zeros((CUBE_CORE_NUM * FIFO_SIZE * TS, TKV), device=device, dtype=torch.float16)
        pv_t = torch.zeros((CUBE_CORE_NUM * PV_CORE_STRIDE, d), device=device, dtype=torch.float32)
        ro_t = torch.zeros((CUBE_CORE_NUM * TS, d), device=device, dtype=torch.float32)
        for index in pl.range(0, 20):
            fe.launch(None, num_cores, compiled, q_t, k_t, v_t, o_t, qk_t, p_t, pv_t, ro_t)
            torch.npu.synchronize()
            qk_ref, x_exp_ref, o_ref = flash_attention_ref(q_t, k_t, v_t, d)
            # qk_diff = (qk_t[:sq, :skv] - qk_ref).abs().max().item()
            # print(f"  qk max|diff|={qk_diff:.6f}")
            # # Check PV
            # pv_ref = torch.matmul(x_exp_ref.half().float(), v_t.float())
            # pv_diff = (pv_t[:sq, :d] - pv_ref[:sq, :d]).abs().max().item()
            # print(f"  pv max|diff|={pv_diff:.6f}")
            diff = (o_t - o_ref).abs().max().item()
            print(f"  o  max|diff|={diff:.4f}")
        torch.testing.assert_close(o_t, o_ref, rtol=5e-3, atol=5e-3)
        print("  PASS")



if __name__ == "__main__":
    print(f"FA perf: double-buffer + QK pre-compute (QK_PRELOAD={QK_PRELOAD}, FIFO={FIFO_SIZE})")
    print("=" * 60)
    test_fa_perf()
    print("\nAll FlashAttention tests passed!")