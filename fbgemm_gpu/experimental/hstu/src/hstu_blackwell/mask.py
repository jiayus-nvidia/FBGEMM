from typing import Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

import utils

from cutlass.cute.typing import Int32
from .utils import split_wg


@dataclass(frozen=True)
class AttentionMask:
    kBlockM: cutlass.Constexpr[int]
    kBlockN: cutlass.Constexpr[int]
    is_arbitrary: cutlass.Constexpr[bool]
    is_causal: cutlass.Constexpr[bool]
    is_local: cutlass.Constexpr[bool]
    is_context: cutlass.Constexpr[bool]
    is_target: cutlass.Constexpr[bool]
    target_group_size: cutlass.Constexpr[int]
    func_num: cutlass.Constexpr[int]
    window_size_left: cutlass.Constexpr[int]
    window_size_right: cutlass.Constexpr[int]
    offset_q: cutlass.Constexpr[int]
    seqlen_q: cutlass.Constexpr[int]
    seqlen_k: cutlass.Constexpr[int]
    seqlen_c: cutlass.Constexpr[int]
    seqlen_h: cutlass.Constexpr[int]
    func: Optional[cute.Tensor] # (n_func, L_func)
    swapAB: cutlass.Constexpr[bool]

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
    ) -> None:
        seqlen_offset = self.seqlen_k - self.seqlen_q
        cS = cute.make_identity_tensor((self.kBlockM, self.kBlockN))
        tScS = thr_mma.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        base_row = m_block * self.kBlockM + seqlen_offset
        base_col = n_block * self.kBlockN
        row_id, col_id = (1, 0) if cutlass.const_expr(self.swapAB) else (0, 1)

        col_limit_right = lambda row: min(self.seqlen_k, row + 1 + self.window_size_right)
        col_limit_left = lambda row: max(0, row - self.window_size_left)

        for i in cutlass.range(cute.size(acc_S), unroll_full=True):
            block_row = cute.get(tScS_t2r[i], mode=[row_id])
            row = block_row + base_row
            target_index = (row - self.seqlen_h) // self.target_group_size if cutlass.const_expr(self.is_target) else 0
            target_col_limit_left = self.seqlen_h + target_index * self.target_group_size if cutlass.const_expr(self.is_target) else 0

            block_col = cute.get(tScS_t2r[i], mode=[col_id])
            col = block_col + base_col

            if cutlass.const_expr(not self.is_causal and not self.is_local and not self.is_arbitrary):
                if col >= self.seqlen_k:
                    acc_S[i] = -cutlass.Float32.inf
            elif cutlass.const_expr(self.is_arbitrary):
                func_row = row + self.offset_q - seqlen_offset
                for j in cutlass.range(self.func_num // 2, unroll_full=True):
                    if col >= self.func[2 * j, func_row].value and col < self.func[2 * j + 1, func_row].value:
                        acc_S[i] = -cutlass.Float32.inf
                if col >= self.func[self.func_num - 1, func_row].value or col >= self.seqlen_k:
                    acc_S[i] = -cutlass.Float32.inf
            else:
                skip = False
                if cutlass.const_expr(self.is_context):
                    if row < self.seqlen_c and col < self.seqlen_h:
                        skip = True

                if not skip:
                    if col >= col_limit_right(row):
                        acc_S[i] = -cutlass.Float32.inf
                    if cutlass.const_expr(self.is_local):
                        if col < col_limit_left(row):
                            acc_S[i] = -cutlass.Float32.inf
                    if cutlass.const_expr(self.is_target):
                        if row >= self.seqlen_h and col >= self.seqlen_h and col < target_col_limit_left:
                            acc_S[i] = -cutlass.Float32.inf


    @cute.jit
    def apply_mask_swapAB(
        self,
        acc_S: cute.Tensor,
        wg_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
    ) -> None:
        seqlen_offset = self.seqlen_k - self.seqlen_q
        cS = cute.make_identity_tensor((self.kBlockM, self.kBlockN))
        tScS = thr_mma.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(cS)
        tScS_t2r = split_wg(tScS_t2r, 2, wg_idx)
        base_row = m_block * self.kBlockM + seqlen_offset
        base_col = n_block * self.kBlockN
        row_id, col_id = (1, 0) if cutlass.const_expr(self.swapAB) else (0, 1)

        col_limit_right = lambda row: min(self.seqlen_k, row + 1 + self.window_size_right)
        col_limit_left = lambda row: max(0, row - self.window_size_left)

        for i in cutlass.range(cute.size(acc_S), unroll_full=True):
            block_row = cute.get(tScS_t2r[i], mode=[row_id])
            row = block_row + base_row
            target_index = (row - self.seqlen_h) // self.target_group_size if cutlass.const_expr(self.is_target) else 0
            target_col_limit_left = self.seqlen_h + target_index * self.target_group_size if cutlass.const_expr(self.is_target) else 0
            block_col = cute.get(tScS_t2r[i], mode=[col_id])
            col = block_col + base_col
            if col >= self.seqlen_k or row >= self.seqlen_q + seqlen_offset:
                acc_S[i] = -cutlass.Float32.inf
            if cutlass.const_expr(self.is_arbitrary):
                func_row = row + self.offset_q - seqlen_offset
                for j in cutlass.range(self.func_num // 2, unroll_full=True):
                    if col >= self.func[2*j, func_row].value and col < self.func[2*j+1, func_row].value:
                        acc_S[i] = -cutlass.Float32.inf
                if col >= self.func[self.func_num - 1, func_row].value:
                    acc_S[i] = -cutlass.Float32.inf
            else:
                skip = False
                if cutlass.const_expr(self.is_context):
                    if row < self.seqlen_c and col < self.seqlen_h:
                        skip = True
                if not skip:
                    if col >= col_limit_right(row):
                        acc_S[i] = -cutlass.Float32.inf
                    if cutlass.const_expr(self.is_local):
                        if col < col_limit_left(row):
                            acc_S[i] = -cutlass.Float32.inf
                    if cutlass.const_expr(self.is_target):
                        if row >= self.seqlen_h and col >= self.seqlen_h and col < target_col_limit_left:
                            acc_S[i] = -cutlass.Float32.inf