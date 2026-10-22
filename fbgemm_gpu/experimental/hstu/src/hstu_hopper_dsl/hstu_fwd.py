# HSTU attention forward for SM90, CuTe DSL.

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32
import cutlass.utils.hopper_helpers as sm90_hh
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils.layout import LayoutEnum
from cutlass import const_expr
from cutlass.pipeline import CooperativeGroup, Agent, PipelineUserType
from .pipeline import PipelineTmaAsyncNoCluster, make_pipeline_state
from .seqlen_info import SeqlenInfo

from .block_info import BlockInfo  # noqa: F401
from .seqlen_info import SeqlenInfo  # noqa: F401
from .tile_scheduler import TileSchedulerArguments, SingleTileVarlenScheduler
from .hopper_helpers import partition_fragment_ABC, gemm_zero_init
from .hopper_helpers import partition_fragment_ABC, gemm_zero_init, gemm_w_idx
from .utils import transpose_view, convert_layout_acc_frgA, cvt_f16, tanhf
from .utils import (
    transpose_view, convert_layout_acc_frgA, cvt_f16, tanhf, get_smem_store_atom,
)
from .named_barrier import NamedBarrierFwd



class HSTUAttentionForwardSm90:
    arch = 90

    def __init__(
        self,
        head_dim: int,
        is_causal: bool = False,
        is_local: bool = False,
        is_context: bool = False,
        is_target: bool = False,
        target_group_size: int = 1,
        is_arbitrary: bool = False,
        func_num: int = 0,
        is_paged: bool = False,
        kBlockM: int = 128,
        kBlockN: int = 128,
        page_size: int = 64,
    ):
        self.head_dim = head_dim
        self.head_dim_padded = head_dim
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_context = is_context
        self.is_target = is_target
        self.target_group_size = target_group_size
        self.is_arbitrary = is_arbitrary
        self.func_num = func_num
        self.is_paged = is_paged
        self.kBlockM = kBlockM
        self.kBlockN = kBlockN
        self.page_size = page_size
        # MMA tile vs cache page. page_size==kBlockN → one TMA per tile (old path).
        if self.is_paged and self.page_size > 0:
            if self.kBlockN % self.page_size != 0:
                raise ValueError(
                    f"kBlockN={self.kBlockN} must be a multiple of page_size={self.page_size}"
                )
            self.pages_per_tile = self.kBlockN // self.page_size
        else:
            self.pages_per_tile = 1
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        
        self.dtype = cutlass.BFloat16
        self.overlap_qo = head_dim == 256
        self.num_stages = 2
        self.mma_pv_is_rs = True
        self.num_mma_warp_groups = self.kBlockM // 64
        self.threads_per_warp_group = 128
        self.threads_per_cta = (1 + self.num_mma_warp_groups) * 128  # 384
        self.buffer_align_bytes = 1024

    def _get_smem_layout_atom(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_hh.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.head_dim_padded
            ),
            self.dtype,
        )

        # Q/K/V/O are the same layout atom.
        sQ_layout_atom = atom
        sK_layout_atom = atom
        sV_layout_atom = atom
        sO_layout_atom = atom

        # PV is RS, so no sP.
        sP_layout_atom = None
        return (
            sQ_layout_atom,
            sK_layout_atom,
            sV_layout_atom,
            sO_layout_atom,
            sP_layout_atom,
        )
        
    def _get_tiled_mma(self):
        tiled_mma_qk = sm90_hh.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.kBlockM // 64, 1, 1),
            tiler_mn = (64, self.kBlockN),
        )

        tiled_mma_pv = sm90_hh.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(self.kBlockM // 64, 1, 1),
            tiler_mn = (64, self.head_dim_padded),
            a_source = warpgroup.OperandSource.RMEM,
        )
        return tiled_mma_qk, tiled_mma_pv

    def _setup_attributes(self):
        (
            sQ_layout_atom,
            sK_layout_atom,
            sV_layout_atom,
            sO_layout_atom,
            sP_layout_atom,
        ) = self._get_smem_layout_atom()

        self.sQ_layout = cute.tile_to_shape(sQ_layout_atom, (self.kBlockM, self.head_dim_padded), (0, 1))
        self.sK_layout = cute.tile_to_shape(sK_layout_atom, (self.kBlockN, self.head_dim_padded, self.num_stages), (0, 1, 2))
        self.sV_layout = cute.tile_to_shape(sV_layout_atom, (self.kBlockN, self.head_dim_padded, self.num_stages), (0, 1, 2))
        # Paged history TMA is one cache page, which may be a slice of the MMA tile.
        page_n = self.page_size if self.is_paged and self.page_size > 0 else self.kBlockN
        self.sK_page_layout = cute.tile_to_shape(
            sK_layout_atom, (page_n, self.head_dim_padded), (0, 1)
        )
        self.sV_page_layout = cute.tile_to_shape(
            sV_layout_atom, (page_n, self.head_dim_padded), (0, 1)
        )
        self.sO_layout = cute.tile_to_shape(sO_layout_atom, (self.kBlockM, self.head_dim_padded), (0, 1))
        self.sP_layout = None

        self.tiled_mma_qk, self.tiled_mma_pv = self._get_tiled_mma()

        self.tma_copy_q_bytes = cute.size_in_bytes(self.dtype, cute.select(self.sQ_layout, mode = [0, 1]))
        self.tma_copy_k_bytes = cute.size_in_bytes(self.dtype, cute.select(self.sK_layout, mode = [0, 1]))
        self.tma_copy_v_bytes = cute.size_in_bytes(self.dtype, cute.select(self.sV_layout, mode = [0, 1]))

        universal_copy_bits = 128
        async_copy_elems_O = universal_copy_bits // self.dtype.width
        tO_shape_dim_1 = sO_layout_atom.outer.shape[1] // async_copy_elems_O
        num_epi = self.num_mma_warp_groups * self.threads_per_warp_group
        tO_layout = cute.make_ordered_layout(
            (num_epi // tO_shape_dim_1, tO_shape_dim_1),
            order=(1, 0),
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tO_layout, cute.make_layout((1, async_copy_elems_O)),
        )

    def _get_shared_storage_cls(self):
        @cute.struct
        class SharedStorage:
            mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, 1 * 2]
            mbar_ptr_K: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mbar_ptr_V: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.sQ_layout)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.sK_layout)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.sV_layout)],
                self.buffer_align_bytes,
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.sO_layout)],
                self.buffer_align_bytes,
            ]

        @cute.struct
        class SharedStorageOverlapQO:
            mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, 1 * 2]
            mbar_ptr_K: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mbar_ptr_V: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.sQ_layout)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.sK_layout)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.sV_layout)],
                self.buffer_align_bytes,
            ]

        return SharedStorageOverlapQO if self.overlap_qo else SharedStorage

    @cute.jit
    def _kv_block_iter(
        self,
        seqlen: SeqlenInfo,
        m_block: Int32,
        window_size_left: Int32,
        window_size_right: Int32,
    ):
        """Ampere paged/dense KV visit set: causal n_block_max + target is_jump.

        Returns (n_block_min, n_block_hist, n_hist_count, n_tgt_begin, n_total,
        n_masking_lo, n_masking_hi). Tile i maps to n_block_min+i if
        i < n_hist_count else n_tgt_begin+(i-n_hist_count). Causal interior
        tiles in [n_masking_lo, n_masking_hi) still need the element mask.
        """
        seqlen_offset = seqlen.seqlen_k - seqlen.seqlen_q
        seqlen_h = seqlen.seqlen_h
        is_in_target = cutlass.Boolean(False)
        is_jump = cutlass.Boolean(False)
        if const_expr(self.is_target):
            is_in_target = (m_block + Int32(1)) * Int32(self.kBlockM) + seqlen_offset > seqlen_h
            is_jump = m_block * Int32(self.kBlockM) + seqlen_offset > seqlen_h

        n_block_hist = cute.ceil_div(seqlen_h, self.kBlockN)
        # Pad from seqlen_h to the end of the last history MMA tile. Equals
        # page_size-last_page_seqlen when kBlockN==page_size; extra dummy
        # pages in a kBlockN>page_size tile are included too.
        last_page_offset = Int32(0)
        if const_expr(self.is_paged and self.is_target):
            if is_in_target:
                last_page_offset = n_block_hist * Int32(self.kBlockN) - seqlen_h
        if const_expr(self.is_paged):
            n_block_max = n_block_hist + cute.ceil_div(seqlen.seqlen_t, self.kBlockN)
        else:
            n_block_max = cute.ceil_div(seqlen.seqlen_k, self.kBlockN)

        n_block_min = Int32(0)
        if const_expr(self.is_local):
            n_block_min = (
                m_block * Int32(self.kBlockM) + seqlen_offset - window_size_left
            ) // Int32(self.kBlockN)
            if n_block_min < Int32(0):
                n_block_min = Int32(0)

        if const_expr(self.is_causal or self.is_local):
            kv_limit = (
                (m_block + Int32(1)) * Int32(self.kBlockM)
                + seqlen_offset
                + window_size_right
            )
            if const_expr(self.is_paged):
                kv_limit = kv_limit + last_page_offset
            n_block_limit = cute.ceil_div(kv_limit, self.kBlockN)
            if n_block_limit < n_block_max:
                n_block_max = n_block_limit

        n_masking_steps = Int32(0)
        n_masking_block_min = Int32(0)
        n_masking_block_max = Int32(0)
        if const_expr(self.is_causal):
            n_masking_hi = seqlen.seqlen_k + last_page_offset
            n_masking_row = (
                (m_block + Int32(1)) * Int32(self.kBlockM)
                + seqlen_offset
                + last_page_offset
            )
            if n_masking_row < n_masking_hi:
                n_masking_hi = n_masking_row
            n_masking_block_max = cute.ceil_div(n_masking_hi, self.kBlockN)
            n_masking_block_min = (
                m_block * Int32(self.kBlockM) + seqlen_offset
            ) // Int32(self.kBlockN)
            if const_expr(self.is_target):
                if is_jump:
                    target_index = (
                        m_block * Int32(self.kBlockM) - seqlen_h
                    ) // Int32(self.target_group_size)
                    n_masking_block_min = (
                        seqlen_h
                        + seqlen_offset
                        + target_index * Int32(self.target_group_size)
                        + last_page_offset
                    ) // Int32(self.kBlockN)
            n_masking_steps = n_masking_block_max - n_masking_block_min

        n_hist_count = n_block_max - n_block_min
        n_tgt_begin = n_block_max
        if is_jump:
            if n_masking_steps > Int32(0):
                n_hist_count = n_block_hist - n_block_min
                n_tgt_begin = n_block_max - n_masking_steps
        if n_hist_count < Int32(0):
            n_hist_count = Int32(0)
        n_total = n_hist_count + (n_block_max - n_tgt_begin)
        if n_total < Int32(0):
            n_total = Int32(0)
        return (
            n_block_min,
            n_block_hist,
            n_hist_count,
            n_tgt_begin,
            n_total,
            n_masking_block_min,
            n_masking_block_max,
        )

    @cute.jit
    def _split_kv_range(self, n_total: Int32):
        """Evenly partition visited KV tiles across blockIdx.y (additive split-K)."""
        n_split = cute.arch.grid_dim()[1]
        split_id = cute.arch.block_idx()[1]
        chunk = cute.ceil_div(n_total, n_split)
        kv_start = split_id * chunk
        kv_end = kv_start + chunk
        if kv_end > n_total:
            kv_end = n_total
        n_this = kv_end - kv_start
        if n_this < Int32(0):
            n_this = Int32(0)
            kv_start = n_total
        return kv_start, n_this

    @cute.kernel
    def kernel(
        self,
            tma_tensor_Q: cute.Tensor,
            tma_tensor_K: cute.Tensor,
            tma_tensor_V: cute.Tensor,
            tma_tensor_O: cute.Tensor,
            mO: cute.Tensor,
            tma_atom_Q: cute.CopyAtom,
            tma_atom_K: cute.CopyAtom,
            tma_atom_V: cute.CopyAtom,
            tma_atom_O: cute.CopyAtom,
            tma_tensor_K_tgt: cute.Tensor,
            tma_tensor_V_tgt: cute.Tensor,
            tma_atom_K_tgt: cute.CopyAtom,
            tma_atom_V_tgt: cute.CopyAtom,
            sQ_layout: cute.ComposedLayout,
            sK_layout: cute.ComposedLayout,
            sV_layout: cute.ComposedLayout,
            sO_layout: cute.ComposedLayout,
            max_seqlen_q: Int32,
            max_seqlen_k: Int32,
            cu_seqlens_q: cute.Tensor,
            cu_seqlens_k: cute.Tensor,
            num_contexts: Optional[cute.Tensor],
            num_targets: Optional[cute.Tensor],
            page_ids: Optional[cute.Tensor],
            page_indptrs: Optional[cute.Tensor],
            last_page_seqlen: Optional[cute.Tensor],
            window_size_left: Int32,
            window_size_right: Int32,
            func: Optional[cute.Tensor],
            tile_sched_params,
            tiled_mma_qk: cute.TiledMma,
            tiled_mma_pv: cute.TiledMma,
            gmem_tiled_copy_O,
            alpha: Float32,
            SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_O)
            cpasync.prefetch_descriptor(tma_atom_K_tgt)
            cpasync.prefetch_descriptor(tma_atom_V_tgt)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        if const_expr(self.overlap_qo):
            sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        tma_warp = CooperativeGroup(Agent.Thread, 1)
        # consumer_release arrives once per MMA warp group (thread_idx % 128 == 0).
        # Arrive count must match, otherwise 2-stage KV deadlocks on wrap.
        mma_warp = CooperativeGroup(Agent.Thread, self.num_mma_warp_groups)

        pipeline_q = PipelineTmaAsyncNoCluster.create(
            barrier_storage = storage.mbar_ptr_Q.data_ptr(),
            num_stages = 1,
            producer_group = tma_warp,
            consumer_group = mma_warp,
            tx_count = self.tma_copy_q_bytes,
        )
        pipeline_k = PipelineTmaAsyncNoCluster.create(
            barrier_storage = storage.mbar_ptr_K.data_ptr(),
            num_stages = self.num_stages,
            producer_group = tma_warp,
            consumer_group = mma_warp,
            tx_count = self.tma_copy_k_bytes,
        )
        pipeline_v = PipelineTmaAsyncNoCluster.create(
            barrier_storage = storage.mbar_ptr_V.data_ptr(),
            num_stages = self.num_stages,
            producer_group = tma_warp,
            consumer_group = mma_warp,
            tx_count = self.tma_copy_v_bytes,
        )
        tidx = cute.arch.thread_idx()[0]
        if tidx < self.threads_per_warp_group:
            warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
            if warp_idx_in_wg == 0:
                tile_scheduler = SingleTileVarlenScheduler.create(tile_sched_params)
                work_tile = tile_scheduler.initial_work_tile_info()

                q_producer_state = make_pipeline_state(PipelineUserType.Producer, 1)
                kv_producer_state = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
                v_producer_state = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
                while work_tile.is_valid_tile:
                    m_block, head_idx, batch_idx = work_tile.tile_idx
                    seqlen = SeqlenInfo(
                        batch_idx, max_seqlen_q, max_seqlen_k,
                        cu_seqlens_q, cu_seqlens_k, num_contexts, num_targets,
                        page_indptrs, last_page_seqlen,
                    )
                    mQ_cur = cute.domain_offset((seqlen.offset_q, 0), tma_tensor_Q[None, None, head_idx])
                    if const_expr(self.is_paged):
                        mK_cur = tma_tensor_K[None, None, head_idx]
                        mV_cur = tma_tensor_V[None, None, head_idx]
                    else:
                        mK_cur = cute.domain_offset((seqlen.offset_k, 0), tma_tensor_K[None, None, head_idx])
                        mV_cur = cute.domain_offset((seqlen.offset_k, 0), tma_tensor_V[None, None, head_idx])

                    gQ = cute.local_tile(mQ_cur, (self.kBlockM, self.head_dim_padded), (m_block, 0))
                    if const_expr(self.is_paged):
                        gK = cute.local_tile(
                            mK_cur, (self.page_size, self.head_dim_padded), (None, 0)
                        )
                        gV = cute.local_tile(
                            mV_cur, (self.page_size, self.head_dim_padded), (None, 0)
                        )
                    else:
                        gK = cute.local_tile(
                            mK_cur, (self.kBlockN, self.head_dim_padded), (None, 0)
                        )
                        gV = cute.local_tile(
                            mV_cur, (self.kBlockN, self.head_dim_padded), (None, 0)
                        )

                    tQsQ, tQgQ = cpasync.tma_partition(
                        tma_atom_Q, 0, cute.make_layout(1),
                        cute.group_modes(sQ, 0, cute.rank(sQ)),
                        cute.group_modes(gQ, 0, cute.rank(gQ)),
                    )

                    if const_expr(not self.is_paged or self.pages_per_tile == 1):
                        tKsK, tKgK = cpasync.tma_partition(
                            tma_atom_K, 0, cute.make_layout(1),
                            cute.group_modes(sK, 0, cute.rank(sK) - 1),
                            cute.group_modes(gK, 0, cute.rank(gK) - 1),
                        )
                        tVsV, tVgV = cpasync.tma_partition(
                            tma_atom_V, 0, cute.make_layout(1),
                            cute.group_modes(sV, 0, cute.rank(sV) - 1),
                            cute.group_modes(gV, 0, cute.rank(gV) - 1),
                        )

                    if const_expr(self.is_paged):
                        offset_tgt = seqlen.offset_q + seqlen.seqlen_q - seqlen.seqlen_t
                        mK_tgt = cute.domain_offset(
                            (offset_tgt, 0), tma_tensor_K_tgt[None, None, head_idx]
                        )
                        mV_tgt = cute.domain_offset(
                            (offset_tgt, 0), tma_tensor_V_tgt[None, None, head_idx]
                        )
                        gK_tgt = cute.local_tile(mK_tgt, (self.kBlockN, self.head_dim_padded), (None, 0))
                        gV_tgt = cute.local_tile(mV_tgt, (self.kBlockN, self.head_dim_padded), (None, 0))
                        tKsK_tgt, tKgK_tgt = cpasync.tma_partition(
                            tma_atom_K_tgt, 0, cute.make_layout(1),
                            cute.group_modes(sK, 0, cute.rank(sK) - 1),
                            cute.group_modes(gK_tgt, 0, cute.rank(gK_tgt) - 1),
                        )
                        tVsV_tgt, tVgV_tgt = cpasync.tma_partition(
                            tma_atom_V_tgt, 0, cute.make_layout(1),
                            cute.group_modes(sV, 0, cute.rank(sV) - 1),
                            cute.group_modes(gV_tgt, 0, cute.rank(gV_tgt) - 1),
                        )
                    n_block_min, n_block_hist, n_hist_count, n_tgt_begin, n_total, n_masking_lo, n_masking_hi = self._kv_block_iter(
                        seqlen, m_block, window_size_left, window_size_right
                    )
                    kv_start, n_this = self._split_kv_range(n_total)
                    dummy_src = Int32(0)
                    if const_expr(self.is_paged and self.pages_per_tile > 1):
                        dummy_src = (
                            cute.size(mK_cur.shape[0]) // Int32(self.page_size)
                            - Int32(1)
                        )
                    if n_this > Int32(0):
                        pipeline_q.producer_acquire(q_producer_state)
                        cute.copy(tma_atom_Q, tQgQ, tQsQ, tma_bar_ptr = pipeline_q.producer_get_barrier(q_producer_state))
                        q_producer_state.advance()

                    for j in cutlass.range(n_this, unroll=1):
                        kv_i = kv_start + j
                        n_block = n_block_min + kv_i
                        if kv_i >= n_hist_count:
                            n_block = n_tgt_begin + (kv_i - n_hist_count)
                        pipeline_k.producer_acquire(kv_producer_state)
                        pipeline_v.producer_acquire(v_producer_state)
                        if const_expr(self.is_paged):
                            if n_block < n_block_hist:
                                if const_expr(self.pages_per_tile == 1):
                                    src_idx = page_ids[seqlen.page_ind + n_block]
                                    cute.copy(
                                        tma_atom_K, tKgK[None, src_idx],
                                        tKsK[None, kv_producer_state.index],
                                        tma_bar_ptr=pipeline_k.producer_get_barrier(kv_producer_state),
                                    )
                                    cute.copy(
                                        tma_atom_V, tVgV[None, src_idx],
                                        tVsV[None, v_producer_state.index],
                                        tma_bar_ptr=pipeline_v.producer_get_barrier(v_producer_state),
                                    )
                                else:
                                    for t in cutlass.range_constexpr(self.pages_per_tile):
                                        page_idx = (
                                            n_block * Int32(self.pages_per_tile)
                                            + Int32(t)
                                        )
                                        src_idx = dummy_src
                                        if page_idx < seqlen.n_pages:
                                            src_idx = page_ids[
                                                seqlen.page_ind + page_idx
                                            ]
                                        sK_t = cute.local_tile(
                                            sK,
                                            (
                                                self.page_size,
                                                self.head_dim_padded,
                                                self.num_stages,
                                            ),
                                            (t, 0, 0),
                                        )
                                        sV_t = cute.local_tile(
                                            sV,
                                            (
                                                self.page_size,
                                                self.head_dim_padded,
                                                self.num_stages,
                                            ),
                                            (t, 0, 0),
                                        )
                                        tKsK_t, tKgK_t = cpasync.tma_partition(
                                            tma_atom_K, 0, cute.make_layout(1),
                                            cute.group_modes(sK_t, 0, cute.rank(sK_t) - 1),
                                            cute.group_modes(gK, 0, cute.rank(gK) - 1),
                                        )
                                        tVsV_t, tVgV_t = cpasync.tma_partition(
                                            tma_atom_V, 0, cute.make_layout(1),
                                            cute.group_modes(sV_t, 0, cute.rank(sV_t) - 1),
                                            cute.group_modes(gV, 0, cute.rank(gV) - 1),
                                        )
                                        cute.copy(
                                            tma_atom_K, tKgK_t[None, src_idx],
                                            tKsK_t[None, kv_producer_state.index],
                                            tma_bar_ptr=pipeline_k.producer_get_barrier(
                                                kv_producer_state
                                            ),
                                        )
                                        cute.copy(
                                            tma_atom_V, tVgV_t[None, src_idx],
                                            tVsV_t[None, v_producer_state.index],
                                            tma_bar_ptr=pipeline_v.producer_get_barrier(
                                                v_producer_state
                                            ),
                                        )
                            else:
                                src_idx = n_block - n_block_hist
                                cute.copy(
                                    tma_atom_K_tgt, tKgK_tgt[None, src_idx],
                                    tKsK_tgt[None, kv_producer_state.index],
                                    tma_bar_ptr=pipeline_k.producer_get_barrier(kv_producer_state),
                                )
                                cute.copy(
                                    tma_atom_V_tgt, tVgV_tgt[None, src_idx],
                                    tVsV_tgt[None, v_producer_state.index],
                                    tma_bar_ptr=pipeline_v.producer_get_barrier(v_producer_state),
                                )
                        else:
                            cute.copy(
                                tma_atom_K, tKgK[None, n_block],
                                tKsK[None, kv_producer_state.index],
                                tma_bar_ptr=pipeline_k.producer_get_barrier(kv_producer_state),
                            )
                            cute.copy(
                                tma_atom_V, tVgV[None, n_block],
                                tVsV[None, v_producer_state.index],
                                tma_bar_ptr=pipeline_v.producer_get_barrier(v_producer_state),
                            )
                        kv_producer_state.advance()
                        v_producer_state.advance()

                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
        else:
            tidx = tidx - self.threads_per_warp_group
            warp_group_idx = cute.arch.make_warp_uniform(
                tidx // self.threads_per_warp_group
            )
            wg_layout = cute.make_layout(
                self.num_mma_warp_groups, stride=self.threads_per_warp_group
            )
            wg_mma_qk = tiled_mma_qk.get_slice(wg_layout(warp_group_idx))
            wg_mma_pv = tiled_mma_pv.get_slice(wg_layout(warp_group_idx))
            sVt = transpose_view(sV)
            acc_O, tOrP, tOrVt = partition_fragment_ABC(
                wg_mma_pv,
                (self.kBlockM, self.head_dim_padded, self.kBlockN),
                None,
                sVt,
            )
            _, tSrQ, tSrK = partition_fragment_ABC(
                wg_mma_qk,
                (self.kBlockM, self.kBlockN, self.head_dim_padded),
                sQ,
                sK,
            )
            q_consumer_state = make_pipeline_state(PipelineUserType.Consumer, 1)
            kv_consumer_state = make_pipeline_state(
                PipelineUserType.Consumer, self.num_stages
            )
            v_consumer_state = make_pipeline_state(
                PipelineUserType.Consumer, self.num_stages
            )
            tile_scheduler = SingleTileVarlenScheduler.create(tile_sched_params)
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                m_block, head_idx, batch_idx = work_tile.tile_idx
                seqlen = SeqlenInfo(
                    batch_idx, max_seqlen_q, max_seqlen_k,
                    cu_seqlens_q, cu_seqlens_k, num_contexts, num_targets,
                    page_indptrs, last_page_seqlen,
                )

                n_block_min, n_block_hist, n_hist_count, n_tgt_begin, n_total, n_masking_lo, n_masking_hi = self._kv_block_iter(
                    seqlen, m_block, window_size_left, window_size_right
                )
                kv_start, n_this = self._split_kv_range(n_total)
                is_first = True
                if n_this <= Int32(0):
                    for z in cutlass.range_constexpr(cute.size(acc_O)):
                        acc_O[z] = Float32(0)
                else:
                    pipeline_q.consumer_wait(q_consumer_state)
                for j in cutlass.range(n_this, unroll=1):
                    kv_i = kv_start + j
                    n_block = n_block_min + kv_i
                    if kv_i >= n_hist_count:
                        n_block = n_tgt_begin + (kv_i - n_hist_count)
                    pipeline_k.consumer_wait(kv_consumer_state)
                    acc_S = gemm_zero_init(
                        tiled_mma_qk,
                        (self.kBlockM, self.kBlockN),
                        tSrQ,
                        tSrK,
                        B_idx=kv_consumer_state.index,
                        wg_wait=0,
                    )
                    pipeline_k.consumer_release(kv_consumer_state)
                    kv_consumer_state.advance()

                    scale_half = alpha * Float32(0.5)
                    # Ampere: skip the element mask on fully-valid interior tiles.
                    # Always mask local/arbitrary, the last (partial) history page,
                    # and causal tiles in [n_masking_lo, n_masking_hi).
                    do_mask = cutlass.Boolean(False)
                    if const_expr(self.is_arbitrary or self.is_local):
                        do_mask = True
                    else:
                        if (n_block + Int32(1)) * Int32(self.kBlockN) > seqlen.seqlen_h:
                            do_mask = True
                        if const_expr(self.is_causal):
                            if n_block >= n_masking_lo and n_block < n_masking_hi:
                                do_mask = True
                    if do_mask:
                        cS = cute.make_identity_tensor((self.kBlockM, self.kBlockN))
                        tScS = tiled_mma_qk.get_slice(tidx).partition_C(cS)
                        seqlen_h = seqlen.seqlen_h
                        seqlen_offset = seqlen.seqlen_k - seqlen.seqlen_q
                        for i in cutlass.range_constexpr(cute.size(acc_S)):
                            row = m_block * self.kBlockM + tScS[i][0] + seqlen_offset
                            col = n_block * self.kBlockN + tScS[i][1]
                            last_page_offset = Int32(0)
                            if const_expr(self.is_paged and self.is_target):
                                last_page_offset = (
                                    n_block_hist * self.kBlockN - seqlen_h
                                )
                                if row >= seqlen_h:
                                    col = col - last_page_offset
                            if const_expr(self.is_arbitrary):
                                func_row = row + seqlen.offset_q - seqlen_offset
                                for j in cutlass.range_constexpr(self.func_num // 2):
                                    if col >= func[2 * j, func_row] and col < func[2 * j + 1, func_row]:
                                        acc_S[i] = Float32(0)
                                if col >= func[self.func_num - 1, func_row] or col >= seqlen.seqlen_k:
                                    acc_S[i] = Float32(0)
                            elif const_expr(not self.is_causal and not self.is_local):
                                if col >= seqlen.seqlen_k:
                                    acc_S[i] = Float32(0)
                            else:
                                col_limit_right = row + Int32(1) + window_size_right
                                if col_limit_right > seqlen.seqlen_k:
                                    col_limit_right = seqlen.seqlen_k
                                if col >= col_limit_right:
                                    acc_S[i] = Float32(0)
                                if const_expr(self.is_local):
                                    col_limit_left = row - window_size_left
                                    if col_limit_left < 0:
                                        col_limit_left = Int32(0)
                                    if col < col_limit_left:
                                        acc_S[i] = Float32(0)
                                if const_expr(self.is_target):
                                    if row >= seqlen_h and (col + last_page_offset) >= seqlen_h:
                                        target_index = (row - seqlen_h) // Int32(self.target_group_size)
                                        target_col_limit_left = seqlen_h + target_index * Int32(self.target_group_size)
                                        if col < target_col_limit_left:
                                            acc_S[i] = Float32(0)
                    for i in cutlass.range_constexpr(cute.size(acc_S)):
                        vh = acc_S[i] * scale_half
                        acc_S[i] = vh * tanhf(vh) + vh
                    tOrP_fp32 = cute.make_tensor(acc_S.iterator, convert_layout_acc_frgA(acc_S.layout))
                    cvt_f16(tOrP_fp32, tOrP)

                    pipeline_v.consumer_wait(v_consumer_state)
                    gemm_w_idx(
                        tiled_mma_pv, acc_O, tOrP, tOrVt,
                        zero_init=is_first,
                        B_idx=v_consumer_state.index,
                        wg_wait=0,
                    )
                    pipeline_v.consumer_release(v_consumer_state)
                    v_consumer_state.advance()
                    is_first = False
                # d=256: sO aliases sQ. Keep the Q stage until O is stored.
                if const_expr(not self.overlap_qo):
                    if n_this > Int32(0):
                        pipeline_q.consumer_release(q_consumer_state)
                        q_consumer_state.advance()

                rO = cute.make_fragment_like(acc_O, self.dtype)
                rO.store(acc_O.load().to(self.dtype))
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.Epilogue),
                    number_of_threads=self.num_mma_warp_groups * self.threads_per_warp_group,
                )
                smem_copy_atom_O = get_smem_store_atom(90, self.dtype)
                smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma_pv).get_slice(tidx)
                taccOrO = smem_thr_copy_O.retile(rO)
                taccOsO = smem_thr_copy_O.partition_D(sO)
                cute.copy(smem_copy_atom_O, taccOrO, taccOsO)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.Epilogue),
                    number_of_threads=self.num_mma_warp_groups * self.threads_per_warp_group,
                )
                n_split = cute.arch.grid_dim()[1]
                split_id = cute.arch.block_idx()[1]
                o_split_stride = cute.size(mO.shape[0]) // n_split
                offset_o = seqlen.offset_q + split_id * o_split_stride
                seqlen_remaining = seqlen.seqlen_q - m_block * self.kBlockM
                if seqlen_remaining >= Int32(self.kBlockM):
                    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
                    if warp_idx == 4:
                        mO_tma = cute.domain_offset(
                            (offset_o, 0), tma_tensor_O[None, None, head_idx]
                        )
                        gO_tma = cute.local_tile(
                            mO_tma, (self.kBlockM, self.head_dim_padded), (m_block, 0)
                        )
                        tOsO, tOgO = cpasync.tma_partition(
                            tma_atom_O, 0, cute.make_layout(1),
                            cute.group_modes(sO, 0, cute.rank(sO)),
                            cute.group_modes(gO_tma, 0, cute.rank(gO_tma)),
                        )
                        cute.copy(tma_atom_O, tOsO, tOgO)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                elif seqlen_remaining > 0:
                    # Packed varlen: a full TMA store would land in the next
                    # sequence. Only write rows that belong to this batch.
                    mO_cur = cute.domain_offset(
                        (offset_o, 0), mO[None, None, head_idx]
                    )
                    gO = cute.local_tile(
                        mO_cur, (self.kBlockM, self.head_dim_padded), (m_block, 0)
                    )
                    cO = cute.make_identity_tensor((self.kBlockM, self.head_dim_padded))
                    gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                    tOsO = gmem_thr_copy_O.partition_S(sO)
                    tOrO = cute.make_fragment_like(tOsO, self.dtype)
                    cute.autovec_copy(tOsO, tOrO)
                    tOgO = gmem_thr_copy_O.partition_D(gO)
                    tOcO = gmem_thr_copy_O.partition_S(cO)
                    t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                    for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                        if t0OcO[0, rest_m, 0][0] < seqlen_remaining - tOcO[0][0]:
                            cute.copy(
                                gmem_tiled_copy_O,
                                tOrO[None, rest_m, None],
                                tOgO[None, rest_m, None],
                            )

                if const_expr(self.overlap_qo):
                    cute.arch.barrier(
                        barrier_id=int(NamedBarrierFwd.Epilogue),
                        number_of_threads=self.num_mma_warp_groups * self.threads_per_warp_group,
                    )
                    if n_this > Int32(0):
                        pipeline_q.consumer_release(q_consumer_state)
                        q_consumer_state.advance()

                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        num_contexts: Optional[cute.Tensor],
        num_targets: Optional[cute.Tensor],
        alpha: Float32,
        stream: cuda.CUstream,
        window_size_left: Int32,
        window_size_right: Int32,
        func: Optional[cute.Tensor] = None,
        mPagedK: Optional[cute.Tensor] = None,
        mPagedV: Optional[cute.Tensor] = None,
        page_ids: Optional[cute.Tensor] = None,
        page_indptrs: Optional[cute.Tensor] = None,
        last_page_seqlen: Optional[cute.Tensor] = None,
        kv_split: Int32 = 1,
    ):
        self.dtype = mQ.element_type
        self.o_dtype = mO.element_type
        
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        mQ, mK, mV, mO = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            for t in (mQ, mK, mV, mO)
        ]
        
        trans = [0, 2, 1]
        mQ, mK, mV, mO = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=trans))
            for t in (mQ, mK, mV, mO)
        ]
        self._setup_attributes()
        if const_expr(self.is_paged):
            mPagedK = cute.make_tensor(mPagedK.iterator, cute.make_layout(mPagedK.shape, stride=new_stride(mPagedK)))
            mPagedV = cute.make_tensor(mPagedV.iterator, cute.make_layout(mPagedV.shape, stride=new_stride(mPagedV)))
            mPagedK = cute.make_tensor(mPagedK.iterator, cute.select(mPagedK.layout, mode=trans))
            mPagedV = cute.make_tensor(mPagedV.iterator, cute.select(mPagedV.layout, mode=trans))
        tma_op_load = cpasync.CopyBulkTensorTileG2SOp()
        tma_op_store = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            tma_op_load, mQ, self.sQ_layout, (self.kBlockM, self.head_dim_padded),
        )
        tma_atom_K_tgt, tma_tensor_K_tgt = cpasync.make_tiled_tma_atom(
            tma_op_load, mK, cute.select(self.sK_layout, mode=[0, 1]),
            (self.kBlockN, self.head_dim_padded),
        )
        tma_atom_V_tgt, tma_tensor_V_tgt = cpasync.make_tiled_tma_atom(
            tma_op_load, mV, cute.select(self.sV_layout, mode=[0, 1]),
            (self.kBlockN, self.head_dim_padded),
        )
        if const_expr(self.is_paged):
            if const_expr(self.pages_per_tile == 1):
                tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                    tma_op_load, mPagedK, cute.select(self.sK_layout, mode=[0, 1]),
                    (self.kBlockN, self.head_dim_padded),
                )
                tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                    tma_op_load, mPagedV, cute.select(self.sV_layout, mode=[0, 1]),
                    (self.kBlockN, self.head_dim_padded),
                )
            else:
                tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                    tma_op_load, mPagedK, self.sK_page_layout,
                    (self.page_size, self.head_dim_padded),
                )
                tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                    tma_op_load, mPagedV, self.sV_page_layout,
                    (self.page_size, self.head_dim_padded),
                )
        else:
            tma_atom_K, tma_tensor_K = tma_atom_K_tgt, tma_tensor_K_tgt
            tma_atom_V, tma_tensor_V = tma_atom_V_tgt, tma_tensor_V_tgt

        tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
            tma_op_store,
            mO,
            self.sO_layout,
            (self.kBlockM, self.head_dim_padded),
        )
        
        SharedStorage = self._get_shared_storage_cls()
        TileScheduler = SingleTileVarlenScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.kBlockM),
            cute.size(mQ.shape[2]),                 # heads
            cute.size(cu_seqlens_q.shape[0]) - 1,   # batch
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0]),
            tile_shape_mn=(self.kBlockM, self.kBlockN),
            cu_seqlens_q=cu_seqlens_q,
            element_size=self.dtype.width // 8,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_x, _, _ = TileScheduler.get_grid_shape(tile_sched_params)
        grid_dim = (grid_x, kv_split, Int32(1))
        self.kernel(
            tma_tensor_Q, tma_tensor_K, tma_tensor_V, tma_tensor_O, mO,
            tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O,
            tma_tensor_K_tgt, tma_tensor_V_tgt, tma_atom_K_tgt, tma_atom_V_tgt,
            self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout,
            max_seqlen_q, max_seqlen_k,
            cu_seqlens_q, cu_seqlens_k,
            num_contexts, num_targets,
            page_ids, page_indptrs, last_page_seqlen,
            window_size_left, window_size_right, func,
            tile_sched_params,
            self.tiled_mma_qk,
            self.tiled_mma_pv,
            self.gmem_tiled_copy_O,
            alpha,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )
