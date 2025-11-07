# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
from typing import Tuple, Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

from .seqlen_info import SeqlenInfo
import fbgemm_gpu.experimental.hstu.src.hstu_blackwell.utils as utils
from .named_barrier import NamedBarrierFwd


@dataclass(frozen=True)
class BlockInfo:
    kBlockM: cutlass.Constexpr[int]
    kBlockN: cutlass.Constexpr[int]
    is_causal: cutlass.Constexpr[bool] = False
    is_local: cutlass.Constexpr[bool] = False
    is_context: cutlass.Constexpr[bool] = False
    is_target: cutlass.Constexpr[bool] = False
    target_group_size: cutlass.Constexpr[int] = 1
    window_size_left: Optional[cutlass.Int32] = None
    window_size_right: Optional[cutlass.Int32] = None
    sn_valid_block_max: cute.Pointer = None
    sValidBlockIds: cute.Tensor = None # (MaxValidBlock,)
    sBlockBound: cute.Pointer = None
    func_num: cutlass.Constexpr[int] = 0
    func: cute.Tensor = None # (n_func, L_func)
    arbitrary_barrier: NamedBarrierFwd = None
    arbitrary_barrier_threads: cutlass.Constexpr[int] = 0

    @cute.jit
    def get_n_block_info(
        self, seqlen_info: SeqlenInfo, m_block: cutlass.Int32
    ) -> Tuple[cutlass.Int32, cutlass.Int32]:
        is_jump = self.is_target and m_block * self.kBlockM > seqlen_info.seqlen_h
        is_in_context = self.is_context and (m_block + 1) * self.kBlockM <= seqlen_info.seqlen_c
        is_in_mixed_context = self.is_context and (m_block + 1) * self.kBlockM > seqlen_info.seqlen_c and m_block * self.kBlockM < seqlen_info.seqlen_c

        n_block_history = cute.ceil_div(seqlen_info.seqlen_h, self.kBlockN)
        seqlen_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        target_index = (m_block * self.kBlockM - seqlen_info.seqlen_h) // self.target_group_size

        n_block_min = max(0, (m_block * self.kBlockM + seqlen_offset - self.window_size_left) // self.kBlockN) if self.is_local else 0
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.kBlockN)
        if self.is_causal or self.is_local:
            n_block_max = min(n_block_max, cute.ceil_div((m_block + 1) * self.kBlockM + seqlen_offset + self.window_size_right, self.kBlockN))
        if self.is_context:
            n_block_min = 0 if (is_in_context or is_in_mixed_context) else n_block_min
            n_block_max = max(cute.ceil_div(seqlen_info.seqlen_h, self.kBlockN), n_block_max) if (is_in_context or is_in_mixed_context) else n_block_max

        n_masking_block_max = cute.ceil_div(min(seqlen_info.seqlen_k, (m_block + 1) * self.kBlockM + seqlen_offset), self.kBlockN)
        n_masking_block_min = (m_block * self.kBlockM + seqlen_offset) // self.kBlockN
        if self.is_target:
            n_masking_block_min = (seqlen_info.seqlen_h + seqlen_offset + target_index * self.target_group_size) // self.kBlockN if is_jump else n_masking_block_min
        if self.is_context:
            n_masking_block_min = n_block_min if is_in_mixed_context else n_masking_block_min
            n_masking_block_max = n_block_max if is_in_mixed_context else n_masking_block_max

        n_masking_steps = 0 if (not self.is_causal or is_in_context) else n_masking_block_max - n_masking_block_min
        return n_block_max, n_block_min, n_masking_steps, is_jump, n_block_history


    @cute.jit
    def get_valid_block_ids(
        self, seqlen_info: SeqlenInfo, m_block: cutlass.Int32, n_block_max: cutlass.Int32, n_block_min: cutlass.Int32, is_calwarp: cutlass.Constexpr[bool]
    ):
        lane_id = cute.arch.lane_idx()
        sn_valid_block_max_tensor = cute.make_tensor(self.sn_valid_block_max, (1,))
        sBlockMin = cute.make_tensor(self.sBlockBound, (self.func_num // 2 + 1,))
        sBlockMax = cute.make_tensor(self.sBlockBound + (self.func_num // 2 + 1) * 4, (self.func_num // 2 + 1,))
        sValidBlockIds = self.sValidBlockIds
        int_max = (1 << 31) - 1
        int_min = -(1 << 31)
        if cutlass.const_expr(is_calwarp):
            sn_valid_block_max_tensor[0] = 0
            sBlockMin[0] = 0
            cute.arch.sync_warp()

            base_row = m_block * self.kBlockM + seqlen_info.offset_q
            f_min = int_max
            for i in cutlass.range(self.func_num // 2):
                for j in cutlass.range(lane_id, self.kBlockM, 32):
                    row = base_row + j
                    if row < seqlen_info.seqlen_q:
                        f_min = min(f_min, self.func[2 * i + 1, row])

                f_min = utils.warp_reduce(f_min, cutlass.min)
                if lane_id == 0:
                    sBlockMin[i + 1] = f_min
                f_min = int_max

            f_max = int_min
            for i in cutlass.range(self.func_num // 2 + 1):
                for j in cutlass.range(lane_id, self.kBlockM, 32):
                    row = base_row + j
                    if row < seqlen_info.seqlen_q:
                        f_max = max(f_max, self.func[2 * i, row])

                f_max = utils.warp_reduce(f_max, cutlass.max)
                if lane_id == 0:
                    sBlockMax[i] = f_max
                f_max = int_min

            if lane_id == 0:
                for n_block in cutlass.range(n_block_min, n_block_max):
                    b_max = (n_block + 1) * self.kBlockN
                    b_min = n_block * self.kBlockN
                    block_valid = False
                    for i in cutlass.range(self.func_num // 2 + 1):
                        f_min = sBlockMin[i]
                        f_max = sBlockMax[i]

                        case1 = (f_min <= b_min and f_max > b_min)
                        case2 = (f_min >= b_min and b_max > f_min)
                        case3 = (f_min >= b_min and f_max < b_max)

                        if case1 or case2 or case3:
                            block_valid = True

                    if block_valid:
                        sValidBlockIds[sn_valid_block_max_tensor[0]] = n_block
                        sn_valid_block_max_tensor[0] = sn_valid_block_max_tensor[0] + 1

        cute.arch.barrier(barrier_id=self.arbitrary_barrier, number_of_threads=self.arbitrary_barrier_threads)
        return sn_valid_block_max_tensor[0], 0