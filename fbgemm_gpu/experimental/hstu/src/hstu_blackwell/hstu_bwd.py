# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import enum
import math
import random
import time
from typing import Type, Tuple, Union, Optional, Callable
from functools import partial

import torch
import torch.nn.functional as F
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Float32, Float8E4M3FN, Float16, BFloat16, Boolean

from .mask import AttentionMask
from .utils import split_wg, tanhf, mul_packed_f32x2, fma_packed_f32x2, sub_packed_f32x2, add_packed_f32x2
from .fast_math import FastSilU
from .block_info import BWDBlockInfo
from .seqlen_info import SeqlenInfo
from .named_barrier import NamedBarrierBwd

"""
A fused multi-head attention (FMHA) backward pass example for the NVIDIA Blackwell SM100 architecture using CUTE DSL

This example demonstrates an implementation of the backward pass of fused multi-head attention
using a TMA + Blackwell SM100 TensorCore warp-specialized kernel. The implementation fuses the computation of
dQ, dK, and dV into a single kernel, avoiding intermediate data movement between
global memory and shared memory, thus improving computational efficiency.

The kernel implements key optimizations including:
- Warp specialization for different computation phases (load, MMA, compute, reduce)
- Pipeline stages between different warps for overlapping computation and memory access
- Support causal masking
- Support for variable sequence lengths
- Support for sliding window attention

To run this example:

.. code-block:: bash

    python examples/blackwell/fmha_bwd.py \\
        --s_q_max 1024 --s_k_max 1024 \\
        --h_q 8 --h_k 8 --d 128 --b 1 \\
        --element_dtype float16 --acc_dtype float32 \\
        --mma_tiler_mn 128,128

The above example runs FMHA backward with max sequence length 1024 for Q and K,
batch size 1, 8 attention heads for Q and K, and head dimension 128.
The Blackwell tcgen05 MMA tile shape is (128, 128), and the kernel uses fp16 for input/output
with fp32 for accumulation.

Constraints for this example:
* Supported head dimensions: 64, and 128
* mma_tiler_mn must be 128,128
* For causal masking, use --is_causal
* For variable sequence lengths, use --varlen
* For sliding window attention, use --window_size x,y
"""

class MaskType(enum.Enum):
    NO_MASK = enum.auto()
    RESIDUAL_MASK_FOR_BACKWARD = enum.auto()
    CAUSAL_MASK_FOR_BACKWARD = enum.auto()

class HSTUAttentionBackwardSm100:

    def __init__(
        self,
        element_dtype: Type[cutlass.Numeric],
        head_dim: int,
        kBlockM: int,
        kBlockN: int,
        is_causal: bool = False,
        is_local: bool = False,
        is_context: bool = False,
        is_target: bool = False,
        target_group_size: int = 1,
        is_arbitrary: bool = False,
        func_num: int = 0,
    ):
        self.element_dtype = element_dtype
        self.acc_dtype = Float32
        self.kBlockM = kBlockM
        self.kBlockN = kBlockN
        self.cta_tiler = (
            kBlockM,
            kBlockN,
            head_dim,
        )
        self.tile_shape_Q = kBlockM
        self.tile_shape_K = kBlockN
        self.tile_shape_dQ_K = head_dim
        self.tile_shape_dV_dO = head_dim
        # For S
        self.KQ_mma_tiler = (
            kBlockN,
            kBlockM,
            head_dim,
        )
        # For dP
        self.VdO_mma_tiler = (
            kBlockN,
            kBlockM,
            head_dim,
        )
        # For dV
        self.PdO_mma_tiler = (
            kBlockN,
            head_dim,
            kBlockM,
        )
        # For dK
        self.dSQ_mma_tiler = (
            kBlockN,
            head_dim,
            kBlockM,
        )
        # For dQ
        self.dSK_mma_tiler = (
            kBlockM,
            head_dim,
            kBlockN,
        )
        self.cluster_shape_mn = (1, 1)
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_context = is_context
        self.is_target = is_target
        self.target_group_size = target_group_size
        self.is_arbitrary = is_arbitrary
        self.func_num = func_num
        assert not (self.is_arbitrary and (self.is_causal or self.is_local or self.is_context or self.is_target)), "a and b cannot both be True"

        self.reduce_warp_id = (0, 1, 2, 3)
        self.compute_warp_id = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.empty_warp_id = (14, 15)

        self.num_reduce_warps = 4
        self.num_compute_warps = 8

        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * (
            self.num_reduce_warps + self.num_compute_warps + 4
        )

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp,
        )
        self.compute_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.num_compute_warps * self.threads_per_warp,
        )
        self.epilogue_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.num_compute_warps * self.threads_per_warp,
        )
        self.reduce_sync_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=self.num_reduce_warps * self.threads_per_warp,
        )

        self.tmem_dK_offset = 0
        self.tmem_dV_offset = self.tmem_dK_offset + head_dim
        self.tmem_dQ_offset = self.tmem_dV_offset + head_dim
        self.tmem_dP_offset = self.tmem_dQ_offset
        self.tmem_S_offset = self.tmem_dQ_offset + max(kBlockM, head_dim)

        self.num_regs_reduce = 152
        self.num_regs_compute = 128
        self.num_regs_mma = 96
        self.num_regs_empty = 96
        self.num_regs_load = 96

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.load_mma_Q_stage = 2
        self.load_mma_dO_stage = 1
        self.mma_compute_S_stage = 1
        self.mma_compute_dP_stage = 1
        self.mma_reduce_dQ_stage = 1
        self.compute_mma_P_stage = 1
        self.compute_mma_dS_stage = 1
        self.mma_compute_dKdV_stage = 2
        self.reduce_tma_store_stage = 2

    @cute.jit
    def __call__(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        dQ: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        dO: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        num_contexts: Optional[cute.Tensor],
        num_targets: Optional[cute.Tensor],
        func: Optional[cute.Tensor],
        alpha: Float32,
        workspace: cute.Tensor,
        stream: cuda.CUstream,
    ):
        q_seq_max, k_seq_max, d, hb = problem_shape
        h, b = hb
        h_r, h_k = h
        # (s, d, h_r, h_k, b) -> (s, d, ((h_r, h_k), b))
        Q = cute.make_tensor(
            Q.iterator,
            cute.make_layout(
                (Q.shape[0], Q.shape[1], hb),
                stride=(
                    Q.stride[0],
                    Q.stride[1],
                    (
                        (Q.shape[0] * Q.shape[1], Q.shape[0] * Q.shape[1] * Q.shape[2]),
                        (
                            0
                        ),
                    ),
                ),
            ),
        )
        # (s, d, 1, h_k, b) -> (s, d, ((1, h_k), b))
        K = cute.make_tensor(
            K.iterator,
            cute.make_layout(
                (K.shape[0], K.shape[1], hb),
                stride=(
                    K.stride[0],
                    K.stride[1],
                    (
                        (0, K.shape[0] * K.shape[1]),
                        (
                            0
                        ),
                    ),
                ),
            ),
        )
        # (s, d, 1, h_k, b) -> (s, d, (（1, h_k), b))
        V = cute.make_tensor(
            V.iterator,
            cute.make_layout(
                (V.shape[0], V.shape[1], hb),
                stride=(
                    V.stride[0],
                    V.stride[1],
                    (
                        (0, V.shape[0] * V.shape[1]),
                        (
                            0
                        ),
                    ),
                ),
            ),
        )

        dQ = cute.make_tensor(dQ.iterator, Q.layout)
        dK = cute.make_tensor(dK.iterator, K.layout)
        dV = cute.make_tensor(dV.iterator, V.layout)
        dO = cute.make_tensor(dO.iterator, Q.layout)

        self.Q_major_mode = utils.LayoutEnum.from_tensor(Q).mma_major_mode()
        self.dQ_major_mode = utils.LayoutEnum.from_tensor(dQ).mma_major_mode()
        self.K_major_mode = utils.LayoutEnum.from_tensor(K).mma_major_mode()
        self.dK_major_mode = utils.LayoutEnum.from_tensor(dK).mma_major_mode()
        self.V_major_mode = utils.LayoutEnum.from_tensor(V).mma_major_mode()
        self.dV_major_mode = utils.LayoutEnum.from_tensor(dV).mma_major_mode()
        self.dO_major_mode = utils.LayoutEnum.from_tensor(dO).mma_major_mode()
        self.dQ_layout = utils.LayoutEnum.from_tensor(dQ)

        if cutlass.const_expr(self.Q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.dQ_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dq is not supported")
        if cutlass.const_expr(self.K_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.dK_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dk is not supported")
        if cutlass.const_expr(self.V_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of v is not supported")
        if cutlass.const_expr(self.dV_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dv is not supported")

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.ONE

        # compute S
        KQ_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.KQ_mma_tiler[:2],
        )
        # compute dP
        VdO_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.VdO_mma_tiler[:2],
        )
        # compute dV
        PdO_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.PdO_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )
        # compute dK
        dSQ_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.dSQ_mma_tiler[:2],
        )
        # compute dQ
        dSK_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.dSK_mma_tiler[:2],
        )

        self.cluster_shape_mn = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mn),
            (KQ_tiled_mma.thr_id.shape,),
        )

        K_smem_layout_staged = sm100_utils.make_smem_layout_a(
            KQ_tiled_mma,
            self.KQ_mma_tiler,
            self.element_dtype,
            1,
        )
        Q_smem_layout_staged = sm100_utils.make_smem_layout_b(
            KQ_tiled_mma,
            self.KQ_mma_tiler,
            self.element_dtype,
            self.load_mma_Q_stage,
        )
        V_smem_layout_staged = sm100_utils.make_smem_layout_a(
            VdO_tiled_mma,
            self.VdO_mma_tiler,
            self.element_dtype,
            1,
        )
        dO_smem_layout_staged = sm100_utils.make_smem_layout_b(
            VdO_tiled_mma,
            self.VdO_mma_tiler,
            self.element_dtype,
            self.load_mma_dO_stage,
        )
        dS_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dSK_tiled_mma,
            self.dSK_mma_tiler,
            self.element_dtype,
            self.compute_mma_dS_stage,
        )
        KT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dSK_tiled_mma,
            self.dSK_mma_tiler,
            self.element_dtype,
            1,
        )
        dST_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dSQ_tiled_mma,
            self.dSQ_mma_tiler,
            self.element_dtype,
            self.compute_mma_dS_stage,
        )
        QT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dSQ_tiled_mma,
            self.dSQ_mma_tiler,
            self.element_dtype,
            self.load_mma_Q_stage,
        )
        P_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            PdO_tiled_mma,
            self.PdO_mma_tiler,
            self.element_dtype,
            self.compute_mma_P_stage,
        )
        dOT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            PdO_tiled_mma,
            self.PdO_mma_tiler,
            self.element_dtype,
            self.load_mma_dO_stage,
        )

        dQ_smem_layout_atom = sm100_utils.make_smem_layout_atom(
            sm100_utils.get_smem_layout_atom_ab(
                tcgen05.OperandMajorMode.K,
                self.acc_dtype,
                (self.tile_shape_Q, 32),
            ),
            self.acc_dtype,
        )
        dQ_smem_layout_staged = cute.tile_to_shape(
            dQ_smem_layout_atom,
            (self.tile_shape_Q, 32, self.reduce_tma_store_stage),
            order=(1, 0, 2),
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_reduce_op = cpasync.CopyReduceBulkTensorTileS2GOp()

        K_smem_layout = cute.select(K_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            K,
            K_smem_layout,
            self.KQ_mma_tiler,
            KQ_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        V_smem_layout = cute.select(V_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            V,
            V_smem_layout,
            self.VdO_mma_tiler,
            VdO_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        Q_smem_layout = cute.select(Q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            Q,
            Q_smem_layout,
            self.KQ_mma_tiler,
            KQ_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        dO_smem_layout = cute.select(dO_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            dO,
            dO_smem_layout,
            self.VdO_mma_tiler,
            VdO_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        self.tma_copy_Q_bytes = cute.size_in_bytes(self.element_dtype, Q_smem_layout)
        self.tma_copy_K_bytes = cute.size_in_bytes(self.element_dtype, K_smem_layout)
        self.tma_copy_V_bytes = cute.size_in_bytes(self.element_dtype, V_smem_layout)
        self.tma_copy_dO_bytes = cute.size_in_bytes(self.element_dtype, dO_smem_layout)
        self.MaxValidBlock = 1024 // 4 if self.is_arbitrary else 1

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            load_mma_Q_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_mma_Q_stage * 2
            ]
            load_mma_dO_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_mma_dO_stage * 2
            ]
            mma_compute_S_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_compute_S_stage * 2
            ]
            mma_compute_dP_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_compute_dP_stage * 2
            ]
            mma_reduce_dQ_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_reduce_dQ_stage * 2
            ]
            compute_mma_P_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.compute_mma_P_stage * 2
            ]
            compute_mma_dS_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.compute_mma_dS_stage * 2
            ]
            mma_compute_dKdV_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_compute_dKdV_stage * 2
            ]
            tmem_holding_buf: cutlass.Int32
            # Smem tensors
            sK: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype, cute.cosize(K_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype, cute.cosize(V_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype, cute.cosize(Q_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype, cute.cosize(dO_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sdS: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype, cute.cosize(dS_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sdQ: cute.struct.Align[
                cute.struct.MemRange[
                    self.acc_dtype, cute.cosize(dQ_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sm_valid_block_max: cute.struct.MemRange[Int32, 1] #used for identification how much m_blocks is needed

            sValidBlockIds: cute.struct.Align[
                cute.struct.MemRange[Int32, self.MaxValidBlock],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        dQ_acc = self.get_workspace_tensor(
            problem_shape, workspace
        )

        dQ_smem_layout = cute.select(dQ_smem_layout_staged, mode=[0, 1])

        tma_atom_dQ_acc, tma_tensor_dQ_acc = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_reduce_op,
            dQ_acc,
            dQ_smem_layout,
            (self.tile_shape_Q, 32),
        )

        bwd_grid = self._compute_bwd_grid(problem_shape, self.cta_tiler[1])

        self.bwd(
            KQ_tiled_mma,
            VdO_tiled_mma,
            PdO_tiled_mma,
            dSQ_tiled_mma,
            dSK_tiled_mma,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_V,
            tma_tensor_V,
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_dO,
            tma_tensor_dO,
            tma_atom_dQ_acc,
            tma_tensor_dQ_acc,
            dK,
            dV,
            problem_shape,
            cu_seqlens_q,
            cu_seqlens_k,
            window_size_left,
            window_size_right,
            num_contexts,
            num_targets,
            func,
            alpha,
            K_smem_layout_staged,
            Q_smem_layout_staged,
            V_smem_layout_staged,
            dO_smem_layout_staged,
            dS_smem_layout_staged,
            KT_smem_layout_staged,
            dST_smem_layout_staged,
            QT_smem_layout_staged,
            dOT_smem_layout_staged,
            dQ_smem_layout_staged,
            P_tmem_layout_staged,
        ).launch(
            grid=bwd_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

        # =============================== Convert ===============================
        self.block_seq = 8
        self.num_threads_D_convert = 16
        self.num_threads_seq = 128 // self.num_threads_D_convert
        self.iter_seq = self.block_seq // self.num_threads_seq
        self.convert_elem_per_load = 4

        max_seq_in_qk = max(problem_shape[0], problem_shape[1])
        convert_grid_z = (max_seq_in_qk + self.block_seq - 1) // self.block_seq
        convert_grid = [
            cute.size(problem_shape[3][0]),
            cute.size(problem_shape[3][1]),
            convert_grid_z,
        ]
        convert_block = [self.num_threads_D_convert, self.num_threads_seq, 1]

        self.convert(
            dQ_acc,
            dQ,
            problem_shape[0],
            problem_shape[2],
            cu_seqlens_q,
        ).launch(
            grid=convert_grid,
            block=convert_block,
            cluster=[1, 1, 1],
            smem=0,
            stream=stream,
        )

    @cute.kernel
    def bwd(
        self,
        KQ_tiled_mma: cute.TiledMma,
        VdO_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        dSK_tiled_mma: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        K_in: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        V_in: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        Q_in: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        dO_in: cute.Tensor,
        tma_atom_dQ_acc: cute.CopyAtom,
        dQ_acc: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        cu_seqlens_q: Union[cute.Tensor, None],
        cu_seqlens_k: Union[cute.Tensor, None],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        num_contexts: Optional[cute.Tensor],
        num_targets: Optional[cute.Tensor],
        func: Optional[cute.Tensor],
        alpha: Float32,
        K_smem_layout_staged: cute.ComposedLayout,
        Q_smem_layout_staged: cute.ComposedLayout,
        V_smem_layout_staged: cute.ComposedLayout,
        dO_smem_layout_staged: cute.ComposedLayout,
        dS_smem_layout_staged: cute.ComposedLayout,
        KT_smem_layout_staged: cute.ComposedLayout,
        dST_smem_layout_staged: cute.ComposedLayout,
        QT_smem_layout_staged: cute.ComposedLayout,
        dOT_smem_layout_staged: cute.ComposedLayout,
        dQ_smem_layout_staged: cute.ComposedLayout,
        P_tmem_layout_staged: cute.ComposedLayout,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim_x, grid_dim_y, grid_dim_z = cute.arch.grid_dim()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_dO)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_mma_Q_pipeline = self.make_and_init_load_mma_Q_pipeline(
            storage.load_mma_Q_mbar_ptr.data_ptr()
        )
        load_mma_dO_pipeline = self.make_and_init_load_mma_dO_pipeline(
            storage.load_mma_dO_mbar_ptr.data_ptr()
        )
        mma_compute_S_pipeline = self.make_and_init_mma_compute_S_pipeline(
            storage.mma_compute_S_mbar_ptr.data_ptr()
        )
        mma_compute_dP_pipeline = self.make_and_init_mma_compute_dP_pipeline(
            storage.mma_compute_dP_mbar_ptr.data_ptr()
        )
        mma_reduce_dQ_pipeline = self.make_and_init_mma_reduce_dQ_pipeline(
            storage.mma_reduce_dQ_mbar_ptr.data_ptr()
        )
        compute_mma_P_pipeline = self.make_and_init_compute_mma_P_pipeline(
            storage.compute_mma_P_mbar_ptr.data_ptr()
        )
        compute_mma_dS_pipeline = self.make_and_init_compute_mma_dS_pipeline(
            storage.compute_mma_dS_mbar_ptr.data_ptr()
        )
        mma_compute_dKdV_pipeline = self.make_and_init_mma_compute_dKdV_pipeline(
            storage.mma_compute_dKdV_mbar_ptr.data_ptr()
        )
        reduce_tma_store_pipeline = self.make_and_init_reduce_tma_store_pipeline()

        self.cta_sync_barrier.arrive_and_wait()

        # setup mma
        sQ = storage.sQ.get_tensor(
            Q_smem_layout_staged.outer, swizzle=Q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            K_smem_layout_staged.outer, swizzle=K_smem_layout_staged.inner
        )
        sV = storage.sV.get_tensor(
            V_smem_layout_staged.outer, swizzle=V_smem_layout_staged.inner
        )
        sdO = storage.sdO.get_tensor(
            dO_smem_layout_staged.outer, swizzle=dO_smem_layout_staged.inner
        )
        sdQ = storage.sdQ.get_tensor(
            dQ_smem_layout_staged.outer, swizzle=dQ_smem_layout_staged.inner
        )
        tmem_holding_buf = storage.tmem_holding_buf

        sQT_ptr = cute.recast_ptr(sQ.iterator, QT_smem_layout_staged.inner)
        sQT = cute.make_tensor(sQT_ptr, QT_smem_layout_staged.outer)
        sKT_ptr = cute.recast_ptr(sK.iterator, KT_smem_layout_staged.inner)
        sKT = cute.make_tensor(sKT_ptr, KT_smem_layout_staged.outer)
        sdS = storage.sdS.get_tensor(
            dS_smem_layout_staged.outer, swizzle=dS_smem_layout_staged.inner
        )
        sdST_ptr = cute.recast_ptr(sdS.iterator, dST_smem_layout_staged.inner)
        sdST = cute.make_tensor(sdST_ptr, dST_smem_layout_staged.outer)
        tP_fake_ptr = cute.make_ptr(self.element_dtype, 0, cute.AddressSpace.tmem)
        tP = cute.make_tensor(tP_fake_ptr, P_tmem_layout_staged.outer)
        sdOT_ptr = cute.recast_ptr(sdO.iterator, dOT_smem_layout_staged.inner)
        sdOT = cute.make_tensor(sdOT_ptr, dOT_smem_layout_staged.outer)

        # (MMA, MMA_M, MMA_K, STAGE)
        tSTrK = KQ_tiled_mma.make_fragment_A(sK)
        # (MMA, MMA_N, MMA_K, STAGE)
        tSTrQ = KQ_tiled_mma.make_fragment_B(sQ)

        # (MMA, MMA_M, MMA_K, STAGE)
        tdPTrV = VdO_tiled_mma.make_fragment_A(sV)
        # (MMA, MMA_N, MMA_K, STAGE)
        tdPTrdO = VdO_tiled_mma.make_fragment_B(sdO)

        # (MMA, MMA_M, MMA_K, STAGE)
        tdQrdS = dSK_tiled_mma.make_fragment_A(sdS)
        # (MMA, MMA_N, MMA_K, STAGE)
        tdQrKT = dSK_tiled_mma.make_fragment_B(sKT)

        # (MMA, MMA_M, MMA_K, STAGE)
        tdKrdST = dSQ_tiled_mma.make_fragment_A(sdST)
        # (MMA, MMA_N, MMA_K, STAGE)
        tdKrQT = dSQ_tiled_mma.make_fragment_B(sQT)

        tSTtST_shape = KQ_tiled_mma.partition_shape_C(
            cute.select(self.KQ_mma_tiler, mode=[0, 1])
        )
        tSTtST = KQ_tiled_mma.make_fragment_C(tSTtST_shape)
        # (MMA, MMA_M, MMA_N)
        tSTtST = cute.make_tensor(tSTtST.iterator + self.tmem_S_offset, tSTtST.layout)

        # (MMA, MMA_M, MMA_K, STAGE)
        tdVrP = PdO_tiled_mma.make_fragment_A(tP)
        tdVrP = tdVrP[None, None, None, 0]
        tdVrP_iter = cute.recast_ptr(tSTtST.iterator, dtype=self.element_dtype)
        tdVrP = cute.make_tensor(tdVrP_iter, tdVrP.layout)
        # (MMA, MMA_N, MMA_K, STAGE)
        tdVrdOT = PdO_tiled_mma.make_fragment_B(sdOT)

        tdPTtdPT_shape = VdO_tiled_mma.partition_shape_C(
            cute.select(self.VdO_mma_tiler, mode=[0, 1])
        )
        tdPTtdPT = VdO_tiled_mma.make_fragment_C(tdPTtdPT_shape)
        # (MMA, MMA_M, MMA_N)
        tdPTtdPT = cute.make_tensor(
            tdPTtdPT.iterator + self.tmem_dP_offset, tdPTtdPT.layout
        )

        tdQtdQ_shape = dSK_tiled_mma.partition_shape_C(
            cute.select(self.dSK_mma_tiler, mode=[0, 1])
        )
        tdQtdQ = dSK_tiled_mma.make_fragment_C(tdQtdQ_shape)
        # (MMA, MMA_M, MMA_N)
        tdQtdQ = cute.make_tensor(tdQtdQ.iterator + self.tmem_dQ_offset, tdQtdQ.layout)

        tdKtdK_shape = dSQ_tiled_mma.partition_shape_C(
            cute.select(self.dSQ_mma_tiler, mode=[0, 1])
        )
        tdKtdK = dSQ_tiled_mma.make_fragment_C(tdKtdK_shape)
        # (MMA, MMA_M, MMA_N)
        tdKtdK = cute.make_tensor(tdKtdK.iterator + self.tmem_dK_offset, tdKtdK.layout)

        tdVtdV_shape = PdO_tiled_mma.partition_shape_C(
            cute.select(self.PdO_mma_tiler, mode=[0, 1])
        )
        tdVtdV = PdO_tiled_mma.make_fragment_C(tdVtdV_shape)
        # (MMA, MMA_M, MMA_N)
        tdVtdV = cute.make_tensor(tdVtdV.iterator + self.tmem_dV_offset, tdVtdV.layout)

        # get the current batch problem shape

        blk_coord = (Int32(0), bidx, Int32(0), ((Int32(0), bidy), bidz))
        max_seqlen_q = Float32(problem_shape[0])
        max_seqlen_k = Float32(problem_shape[1])
        Q_len_cur_batch = cu_seqlens_q[bidz + 1] - cu_seqlens_q[bidz]
        K_len_cur_batch = cu_seqlens_k[bidz + 1] - cu_seqlens_k[bidz]
        num_contexts_cur_batch = num_contexts[bidz] if num_contexts is not None else 0
        num_history_cur_batch = K_len_cur_batch - num_targets[bidz] if num_targets is not None else K_len_cur_batch
        func = func[0, None, None] if func is not None else None
        sValidBlockIdsTensor = cute.make_tensor(storage.sValidBlockIds.data_ptr(), self.MaxValidBlock)
        block_info = BWDBlockInfo(
            # This is cta_tiler, not mma_tiler_qk, since we move by block by (2 * mma_tiler[0], mma_tiler[1])
            self.kBlockM, self.kBlockN, self.cta_tiler, self.is_causal, self.is_local, 
            self.is_context, self.is_target, self.target_group_size,
            window_size_left, window_size_right, 
            storage.sm_valid_block_max.data_ptr(), sValidBlockIdsTensor,
            self.func_num, func,
            NamedBarrierBwd.Arbitrary, cute.arch.WARP_SIZE
                * len(
                    (
                        *self.reduce_warp_id,
                        *self.compute_warp_id,
                        self.mma_warp_id,
                        self.load_warp_id,
                    )
                )
        )
        SeqlenInfoCls = partial(
            SeqlenInfo,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            num_contexts=num_contexts, num_targets=num_targets,
        )
        problem_shape_cur_batch = (
            Q_len_cur_batch,
            K_len_cur_batch,
            problem_shape[2],
            problem_shape[3],
        )
        blk_offset = (
            cu_seqlens_q[bidz],
            cu_seqlens_k[bidz],
            Int32(0),
            ((Int32(0), Int32(0)), Int32(0)),
        )

        iter_start, iter_count = self.get_Q_block_min_max(
            problem_shape_cur_batch[0],
            problem_shape_cur_batch[1],
            blk_coord[1],
        )

        AttentionMaskCls = partial(
            AttentionMask, self.kBlockM, self.kBlockN, self.cta_tiler,
            self.is_arbitrary, self.is_causal, self.is_local, self.is_context, self.is_target,
            target_group_size=self.target_group_size, func_num=self.func_num,
            window_size_left=window_size_left, window_size_right=window_size_right,
            offset_dynamic=0,
            swapAB=True,
        )
        mask = AttentionMaskCls(offset_q=cu_seqlens_q[bidz], seqlen_q=Q_len_cur_batch, seqlen_k=K_len_cur_batch, seqlen_c=num_contexts_cur_batch, seqlen_h=num_history_cur_batch, func=func)

        if bidx * self.tile_shape_K < problem_shape_cur_batch[1]:
            iter_count -= iter_start
            if iter_count <= 0:
                self.epilogue_clear(
                    blk_coord,
                    blk_offset,
                    problem_shape_cur_batch,
                    dK,
                    dV,
                )
            else:
                # ///////////////////////////////////////////////////////////////////////////////
                #  LOAD
                # ///////////////////////////////////////////////////////////////////////////////
                if warp_idx == self.load_warp_id:
                    cute.arch.warpgroup_reg_dealloc(self.num_regs_load)

                    self.load(
                        K_in,
                        V_in,
                        Q_in,
                        dO_in,
                        sK,
                        sQ,
                        sV,
                        sdO,
                        KQ_tiled_mma,
                        VdO_tiled_mma,
                        tma_atom_K,
                        tma_atom_Q,
                        tma_atom_V,
                        tma_atom_dO,
                        blk_offset,
                        problem_shape_cur_batch,
                        # iter_count,
                        # iter_start,
                        (
                            load_mma_Q_pipeline,
                            load_mma_dO_pipeline,
                        ),
                        block_info,
                        SeqlenInfoCls,
                    )

                # ///////////////////////////////////////////////////////////////////////////////
                #  MMA
                # ///////////////////////////////////////////////////////////////////////////////
                elif warp_idx == self.mma_warp_id:
                    cute.arch.warpgroup_reg_dealloc(self.num_regs_mma)

                    self.mma(
                        KQ_tiled_mma,
                        VdO_tiled_mma,
                        PdO_tiled_mma,
                        dSK_tiled_mma,
                        dSQ_tiled_mma,
                        tSTtST,
                        tSTrQ,
                        tSTrK,
                        tdPTtdPT,
                        tdPTrV,
                        tdPTrdO,
                        tdVtdV,
                        tdVrP,
                        tdVrdOT,
                        tdQtdQ,
                        tdQrdS,
                        tdQrKT,
                        tdKrdST,
                        tdKtdK,
                        tdKrQT,
                        tmem_holding_buf,
                        # iter_count,
                        (
                            load_mma_Q_pipeline,
                            mma_compute_S_pipeline,
                            load_mma_dO_pipeline,
                            mma_compute_dP_pipeline,
                            mma_reduce_dQ_pipeline,
                            compute_mma_P_pipeline,
                            compute_mma_dS_pipeline,
                            mma_compute_dKdV_pipeline,
                        ),
                        sdOT,
                        block_info,
                        SeqlenInfoCls,
                        blk_coord
                    )
                # ///////////////////////////////////////////////////////////////////////////////
                #  Compute
                # ///////////////////////////////////////////////////////////////////////////////
                elif (
                    warp_idx >= self.compute_warp_id[0]
                    and warp_idx <= self.compute_warp_id[-1]
                ):
                    cute.arch.warpgroup_reg_alloc(self.num_regs_compute)

                    self.compute(
                        tSTtST,
                        tdPTtdPT,
                        tdVrP,
                        sdS,
                        dK,
                        dV,
                        tdKtdK,
                        tdVtdV,
                        KQ_tiled_mma.get_slice(0),
                        PdO_tiled_mma,
                        dSQ_tiled_mma,
                        blk_coord,
                        blk_offset,
                        problem_shape_cur_batch,
                        max_seqlen_q,
                        alpha,  
                        # iter_count,
                        # iter_start,
                        window_size_left,
                        window_size_right,
                        mask,
                        (
                            mma_compute_S_pipeline,
                            compute_mma_P_pipeline,
                            mma_compute_dP_pipeline,
                            compute_mma_dS_pipeline,
                            mma_compute_dKdV_pipeline,
                        ),
                        block_info,
                        SeqlenInfoCls,
                    )

                    self.epilogue_sync_barrier.arrive_and_wait()

                    if warp_idx % self.num_compute_warps == 0:
                        tmem_ptr = cute.arch.retrieve_tmem_ptr(
                            Float32,
                            alignment=16,
                            ptr_to_buffer_holding_addr=tmem_holding_buf,
                        )
                        cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)
                # ///////////////////////////////////////////////////////////////////////////////
                #  Reduce
                # ///////////////////////////////////////////////////////////////////////////////
                elif (
                    warp_idx >= self.reduce_warp_id[0]
                    and warp_idx <= self.reduce_warp_id[-1]
                ):
                    cute.arch.warpgroup_reg_alloc(self.num_regs_reduce)

                    self.reduce(
                        dSK_tiled_mma,
                        tdQtdQ,
                        tma_atom_dQ_acc,
                        dQ_acc,
                        sdQ,
                        blk_coord,
                        problem_shape_cur_batch,
                        max_seqlen_q,
                        # iter_count,
                        # iter_start,
                        (mma_reduce_dQ_pipeline, reduce_tma_store_pipeline),
                        block_info,
                        SeqlenInfoCls,
                    )

                else:
                    cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

    @cute.kernel
    def convert(
        self,
        dQ_acc: cute.Tensor,
        dQ: cute.Tensor,
        count: Int32,
        d_dim: Int32,
        cu_seqlens_q: Union[cute.Tensor, None],
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        seqlen = count

        offset = 0
        offset = cu_seqlens_q[bidy]
        seqlen = cu_seqlens_q[bidy + 1] - offset

        for idx_s_t in cutlass.range(tidy, self.block_seq, self.num_threads_seq):
            idx_s = idx_s_t + self.block_seq * bidz
            if idx_s < seqlen:
                dQ_acc_bhs = dQ_acc[idx_s, None, (bidx, bidy)]
                dQ_acc_bhs = cute.logical_divide(
                    dQ_acc_bhs, cute.make_layout(self.convert_elem_per_load)
                )
                dQ_bhs = dQ[idx_s + offset, None, (bidx, bidy)]
                dQ_bhs = cute.logical_divide(
                    dQ_bhs, cute.make_layout(self.convert_elem_per_load)
                )

                thr_start = tidx
                thr_step = self.num_threads_D_convert
                for idx_d in cutlass.range(
                    thr_start,
                    d_dim // self.convert_elem_per_load,
                    thr_step,
                ):
                    dQ_acc_frg = dQ_acc_bhs[None, idx_d].load()
                    dQ_bhs[None, idx_d].store(dQ_acc_frg.to(self.element_dtype))

    @cute.jit
    def get_Q_block_min_max(
        self,
        seq_Q: Int32,
        seq_K: Int32,
        blk_coord_k: Int32,
    ):
        Q_block_max = cute.ceil_div(seq_Q, self.tile_shape_Q)
        Q_block_min = cutlass.Int32(0)
        return Q_block_min, Q_block_max

    @cute.jit
    def load(
        self,
        K_in: cute.Tensor,
        V_in: cute.Tensor,
        Q_in: cute.Tensor,
        dO_in: cute.Tensor,
        sK: cute.Tensor,
        sQ: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        KQ_tiled_mma: cute.TiledMma,
        VdO_tiled_mma: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        blk_offset: cute.Shape,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        # iter_count: Int32,
        # iter_index: Int32,
        # (load_mma_Q_pipeline, load_mma_dO_pipeline)
        pipeline_args: tuple,
        block_info: BWDBlockInfo,
        SeqlenInfoCls: Callable,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        blk_coord_k, blk_coord_h_k, blk_coord_b = cute.arch.block_idx()
        seqlen_obj = SeqlenInfoCls(blk_coord_b)

        blk_coord_h_r = Int32(0)
        blk_coord_h = (blk_coord_h_r, blk_coord_h_k)
        seq_Q, seq_K, D, HB = problem_shape
        H, B = HB
        (
            load_mma_Q_pipeline,
            load_mma_dO_pipeline,
        ) = pipeline_args

        K = cute.domain_offset(cute.select(blk_offset, mode=[1, 2, 3]), K_in)
        V = cute.domain_offset(cute.select(blk_offset, mode=[1, 2, 3]), V_in)
        Q = cute.domain_offset(cute.select(blk_offset, mode=[0, 2, 3]), Q_in)
        dO = cute.domain_offset(cute.select(blk_offset, mode=[0, 2, 3]), dO_in)

        # (bM, bK, RestM, RestK, (H, B))
        gK = cute.local_tile(
            K, cute.select(self.KQ_mma_tiler, mode=[0, 2]), (None, None, None)
        )
        # (bN, bK, RestN, RestK, (H, B))
        gQ = cute.local_tile(
            Q, cute.select(self.KQ_mma_tiler, mode=[1, 2]), (None, None, None)
        )
        # (bM, bK, RestM, RestK, (H, B))
        gV = cute.local_tile(
            V, cute.select(self.VdO_mma_tiler, mode=[0, 2]), (None, None, None)
        )
        # (bN, bK, RestN, RestK, (H, B))
        gdO = cute.local_tile(
            dO, cute.select(self.VdO_mma_tiler, mode=[1, 2]), (None, None, None)
        )

        KQ_thr_mma = KQ_tiled_mma.get_slice(0)
        VdO_thr_mma = VdO_tiled_mma.get_slice(0)

        # (MMA, MMA_M, MMA_K, RestM, RestK, (H, B))
        tSTgK = KQ_thr_mma.partition_A(gK)
        # (MMA, MMA_N, MMA_K, RestN, RestK, (H, B))
        tSTgQ = KQ_thr_mma.partition_B(gQ)
        # (MMA, MMA_M, MMA_K, RestM, RestK, (H, B))
        tdPTgV = VdO_thr_mma.partition_A(gV)
        # (MMA, MMA_N, MMA_K, RestN, RestK, (H, B))
        tdPTgdO = VdO_thr_mma.partition_B(gdO)

        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, (H, B))
        tKsK, tKgK_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_K,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSTgK, 0, 3),
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, (H, B))
        tQsQ, tQgQ_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSTgQ, 0, 3),
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, (H, B))
        tVsV, tVgV_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_V,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tdPTgV, 0, 3),
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, (H, B))
        tdOsdO, tdOgdO_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dO,
            0,
            cute.make_layout(1),
            cute.group_modes(sdO, 0, 3),
            cute.group_modes(tdPTgdO, 0, 3),
        )

         #compute m_block info & init m_block
        n_block = blk_coord_k
        offset_dynamic = 0
        m_block_min, m_block_max, m_masking_steps, is_in_context, m_block_context = block_info.get_m_block_info(seqlen_obj, n_block, offset_dynamic)

        if cutlass.const_expr(self.is_arbitrary):
            m_block_max, m_block_min = block_info.get_bwd_valid_block_ids(seqlen_obj, n_block, m_block_min, m_block_max, is_calwarp=True)
        
        sValidBlockIds = block_info.sValidBlockIds

        m_block = 0 if is_in_context else m_block_min # we We start iterating from zero

        m_block = sValidBlockIds[m_block] if cutlass.const_expr(self.is_arbitrary) else m_block

        if m_block_min < m_block_max:
            load_mma_Q_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_mma_Q_stage
            )
            load_mma_dO_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_mma_dO_stage
            )
            load_mma_Q_pipeline.producer_acquire(load_mma_Q_producer_state)
            tma_barrier = load_mma_Q_pipeline.producer_get_barrier(
                load_mma_Q_producer_state
            )
            with cute.arch.elect_one():
                cute.arch.mbarrier_expect_tx(tma_barrier, self.tma_copy_K_bytes)

            # Load K
            cute.copy(
                tma_atom_K,
                tKgK_mkl[(None, blk_coord_k, 0, (blk_coord_h, blk_coord_b))],
                tKsK[None, 0],
                tma_bar_ptr=tma_barrier,
            )

            # Load Q
            cute.copy(
                tma_atom_Q,
                tQgQ_mkl[(None, m_block, 0, (blk_coord_h, blk_coord_b))], #iter_index
                tQsQ[None, load_mma_Q_producer_state.index],
                tma_bar_ptr=tma_barrier,
            )

            load_mma_Q_producer_state.advance()

            async_copy_num_elts = self.cta_tiler[0] // self.threads_per_warp
            atom_async_copy = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
                self.acc_dtype,
                num_bits_per_copy=self.acc_dtype.width,
            )

            load_mma_dO_pipeline.producer_acquire(load_mma_dO_producer_state)
            tma_barrier = load_mma_dO_pipeline.producer_get_barrier(
                load_mma_dO_producer_state
            )
            with cute.arch.elect_one():
                cute.arch.mbarrier_expect_tx(tma_barrier, self.tma_copy_V_bytes)

            # Load V
            cute.copy(
                tma_atom_V,
                tVgV_mkl[(None, blk_coord_k, 0, (blk_coord_h, blk_coord_b))],
                tVsV[(None, 0)],
                tma_bar_ptr=tma_barrier,
            )

            # Load dO
            cute.copy(
                tma_atom_dO,
                tdOgdO_mkl[(None, m_block, 0, (blk_coord_h, blk_coord_b))], #iter_index
                tdOsdO[(None, load_mma_dO_producer_state.index)],
                tma_bar_ptr=tma_barrier,
            )

            load_mma_dO_producer_state.advance()
            m_block = m_block_min if cutlass.const_expr(self.is_arbitrary) else m_block
            m_block += 1

            pipeline_do_q_args = (load_mma_dO_pipeline, load_mma_Q_pipeline)
            while m_block < m_block_max:
                m_block_valid = sValidBlockIds[m_block] if cutlass.const_expr(self.is_arbitrary) else m_block
                load_mma_dO_producer_state, load_mma_Q_producer_state = self.load_step(
                    m_block_valid,
                    tma_atom_dO,
                    tdOgdO_mkl,
                    tdOsdO,
                    tma_atom_Q,
                    tQgQ_mkl,
                    tQsQ,
                    blk_coord_h,
                    blk_coord_b,
                    pipeline_do_q_args,
                    load_mma_dO_producer_state,
                    load_mma_Q_producer_state
                )
                m_block += 1


    @cute.jit
    def load_step(
        self,
        m_block_valid: Int32,
        tma_atom_dO: cute.CopyAtom, #copy dO
        tdOgdO_mkl: cute.Tensor,
        tdOsdO: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tQgQ_mkl: cute.Tensor,
        tQsQ: cute.Tensor,
        blk_coord_h: tuple,
        blk_coord_b: Int32,
        pipeline_args: tuple,
        load_mma_dO_producer_state: cutlass.pipeline.PipelineState,
        load_mma_Q_producer_state: cutlass.pipeline.PipelineState,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        (load_mma_dO_pipeline, load_mma_Q_pipeline) = pipeline_args
        load_mma_Q_pipeline.producer_acquire(load_mma_Q_producer_state)
        tma_barrier = load_mma_Q_pipeline.producer_get_barrier(
                load_mma_Q_producer_state
        )

        # Load Q
        cute.copy(
            tma_atom_Q,
            tQgQ_mkl[(None, m_block_valid, 0, (blk_coord_h, blk_coord_b))],
            tQsQ[None, load_mma_Q_producer_state.index],
            tma_bar_ptr=tma_barrier,
        )

        load_mma_Q_producer_state.advance()

        load_mma_dO_pipeline.producer_acquire(load_mma_dO_producer_state)
        tma_barrier = load_mma_dO_pipeline.producer_get_barrier(
            load_mma_dO_producer_state
        )

        # Load dO
        cute.copy(
            tma_atom_dO,
            tdOgdO_mkl[(None, m_block_valid, 0, (blk_coord_h, blk_coord_b))],
            tdOsdO[None, load_mma_dO_producer_state.index],
            tma_bar_ptr=tma_barrier,
        )

        load_mma_dO_producer_state.advance()

        return load_mma_dO_producer_state, load_mma_Q_producer_state


    @cute.jit
    def mma(
        self,
        KQ_tiled_mma: cute.TiledMma,
        VdO_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        dSK_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        tSTtST: cute.Tensor,
        tSTrQ: cute.Tensor,
        tSTrK: cute.Tensor,
        tdPTtdPT: cute.Tensor,
        tdPTrV: cute.Tensor,
        tdPTrdO: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdVrP: cute.Tensor,
        tdVrdOT: cute.Tensor,
        tdQtdQ: cute.Tensor,
        tdQrdS: cute.Tensor,
        tdQrKT: cute.Tensor,
        tdKrdST: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdKrQT: cute.Tensor,
        tmem_holding_buf: Int32,
        # iter_count: Int32,
        # (load_mma_Q_pipeline, mma_compute_S_pipeline, load_mma_dO_pipeline, mma_compute_dP_pipeline, mma_reduce_dQ_pipeline, compute_mma_P_pipeline, compute_mma_dS_pipeline, mma_compute_dKdV_pipeline)
        pipeline_args: tuple,
        sdOT: cute.Tensor,
        block_info: BWDBlockInfo,
        SeqlenInfoCls: Callable,
        blk_coord: cute.Coord,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_batch = blk_coord
        blk_coord_h, blk_coord_b = blk_coord_batch
        seqlen_obj = SeqlenInfoCls(blk_coord_b)
        n_block = blk_coord_k

        offset_dynamic = 0
        m_block_min, m_block_max, m_masking_steps, is_in_context, m_block_context = block_info.get_m_block_info(seqlen_obj, n_block, offset_dynamic) #TODO m_masking_steps what is m_masking_steps here? and  do we need it? [cause load don't need it]
        if cutlass.const_expr(self.is_arbitrary):
            m_block_max, m_block_min = block_info.get_bwd_valid_block_ids(seqlen_obj, n_block, m_block_min, m_block_max, is_calwarp=False)

        m_block_nums = m_block_max - m_block_min

        (
            load_mma_Q_pipeline,
            mma_compute_S_pipeline,
            load_mma_dO_pipeline,
            mma_compute_dP_pipeline,
            mma_reduce_dQ_pipeline,
            compute_mma_P_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dKdV_pipeline,
        ) = pipeline_args
        # Alloc tmem buffer
        tmem_alloc_cols = cutlass.Int32(self.tmem_alloc_cols)
        cute.arch.alloc_tmem(tmem_alloc_cols, tmem_holding_buf)
        self.tmem_alloc_barrier.arrive_and_wait()
        load_mma_Q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_mma_Q_stage
        )
        load_mma_Q_release_state = load_mma_Q_consumer_state.clone()

        mma_compute_S_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_compute_S_stage
        )
        compute_mma_dS_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.compute_mma_dS_stage
        )
        mma_compute_dP_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_compute_dP_stage
        )
        mma_reduce_dQ_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_reduce_dQ_stage
        )
        load_mma_dO_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_mma_dO_stage
        )
        compute_mma_P_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.compute_mma_P_stage
        )
        mma_compute_dKdV_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_compute_dKdV_stage
        )


        if m_block_min < m_block_max:
            load_mma_Q_pipeline.consumer_wait(load_mma_Q_consumer_state)
            mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state)
            # S = K * Q
            KQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tSTrQ, mode=[2]), unroll_full=True):
                cute.gemm(
                    KQ_tiled_mma,
                    tSTtST,
                    tSTrK[None, None, k_block, 0],
                    tSTrQ[None, None, k_block, load_mma_Q_consumer_state.index],
                    tSTtST,
                )
                KQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            load_mma_Q_consumer_state.advance()
            mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
            mma_compute_S_producer_state.advance()

            load_mma_dO_pipeline.consumer_wait(load_mma_dO_consumer_state)

            mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)
            mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)

            # dP = V * dO
            VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdPTrV, mode=[2]), unroll_full=True):
                cute.gemm(
                    VdO_tiled_mma,
                    tdPTtdPT,
                    tdPTrV[None, None, k_block, 0],
                    tdPTrdO[None, None, k_block, load_mma_dO_consumer_state.index],
                    tdPTtdPT,
                )
                VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
            mma_compute_dP_producer_state.advance()

            compute_mma_P_pipeline.consumer_wait(compute_mma_P_consumer_state)

            # dV = P * dO
            for k_block in cutlass.range(0, cute.size(tdVrP, mode=[2]), unroll_full=True):
                cute.gemm(
                    PdO_tiled_mma,
                    tdVtdV,
                    tdVrP[None, None, k_block],
                    tdVrdOT[None, None, k_block, load_mma_dO_consumer_state.index],
                    tdVtdV,
                )
                PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            compute_mma_P_pipeline.consumer_release(compute_mma_P_consumer_state)
            compute_mma_P_consumer_state.advance()

            load_mma_dO_pipeline.consumer_release(load_mma_dO_consumer_state)
            load_mma_dO_consumer_state.advance()

            # iter_count -= 1
            m_block_nums -= 1

            # while iter_count > 0:
            while m_block_nums > 0:
                load_mma_Q_pipeline.consumer_wait(load_mma_Q_consumer_state)
                mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state)

                # S = K * Q
                KQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(
                    0, cute.size(tSTrQ, mode=[2]), unroll_full=True
                ):
                    cute.gemm(
                        KQ_tiled_mma,
                        tSTtST,
                        tSTrK[None, None, k_block, 0],
                        tSTrQ[None, None, k_block, load_mma_Q_consumer_state.index],
                        tSTtST,
                    )
                    KQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                load_mma_Q_consumer_state.advance()
                mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
                mma_compute_S_producer_state.advance()

                compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)

                # We need to acquire dP here, because tmem dQ == tmem dP
                mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)

                # dQ = dS * K
                dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(
                    0, cute.size(tdQrdS, mode=[2]), unroll_full=True
                ):
                    cute.gemm(
                        dSK_tiled_mma,
                        tdQtdQ,
                        tdQrdS[
                            None,
                            None,
                            k_block,
                            compute_mma_dS_consumer_state.index,
                        ],
                        tdQrKT[None, None, k_block, 0],
                        tdQtdQ,
                    )
                    dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                mma_reduce_dQ_pipeline.producer_commit(mma_reduce_dQ_producer_state)
                mma_reduce_dQ_producer_state.advance()

                # dK = dS * Q
                for k_block in cutlass.range(
                    0, cute.size(tdKrdST, mode=[2]), unroll_full=True
                ):
                    cute.gemm(
                        dSQ_tiled_mma,
                        tdKtdK,
                        tdKrdST[
                            None,
                            None,
                            k_block,
                            compute_mma_dS_consumer_state.index,
                        ],
                        tdKrQT[None, None, k_block, load_mma_Q_release_state.index],
                        tdKtdK,
                    )
                    dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
                load_mma_Q_release_state.advance()

                compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
                compute_mma_dS_consumer_state.advance()

                # We grab dQ here, because in tmem dQ == dP
                mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)
                load_mma_dO_pipeline.consumer_wait(load_mma_dO_consumer_state)

                # dP = V * dO
                VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(
                    0, cute.size(tdPTrV, mode=[2]), unroll_full=True
                ):
                    cute.gemm(
                        VdO_tiled_mma,
                        tdPTtdPT,
                        tdPTrV[None, None, k_block, 0],
                        tdPTrdO[None, None, k_block, load_mma_dO_consumer_state.index],
                        tdPTtdPT,
                    )
                    VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
                mma_compute_dP_producer_state.advance()

                compute_mma_P_pipeline.consumer_wait(compute_mma_P_consumer_state)

                # dV = P * dO
                for k_block in cutlass.range(
                    0, cute.size(tdVrP, mode=[2]), unroll_full=True
                ):
                    cute.gemm(
                        PdO_tiled_mma,
                        tdVtdV,
                        tdVrP[None, None, k_block],
                        tdVrdOT[None, None, k_block, load_mma_dO_consumer_state.index],
                        tdVtdV,
                    )
                    PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                compute_mma_P_pipeline.consumer_release(compute_mma_P_consumer_state)
                compute_mma_P_consumer_state.advance()

                load_mma_dO_pipeline.consumer_release(load_mma_dO_consumer_state)
                load_mma_dO_consumer_state.advance()

                # iter_count -= 1
                m_block_nums -= 1

            # Signal to the epilogue that dV is ready
            mma_compute_dKdV_pipeline.producer_acquire(mma_compute_dKdV_producer_state)
            mma_compute_dKdV_pipeline.producer_commit(mma_compute_dKdV_producer_state)
            mma_compute_dKdV_producer_state.advance()

            mma_compute_dKdV_pipeline.producer_acquire(mma_compute_dKdV_producer_state)

            compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)

            # dK = dS * Q
            for k_block in cutlass.range(0, cute.size(tdKrdST, mode=[2]), unroll_full=True):
                cute.gemm(
                    dSQ_tiled_mma,
                    tdKtdK,
                    tdKrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdKrQT[None, None, k_block, load_mma_Q_release_state.index],
                    tdKtdK,
                )
                dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # Signal to epilogue that dK is ready
            mma_compute_dKdV_pipeline.producer_commit(mma_compute_dKdV_producer_state)
            mma_compute_dKdV_producer_state.advance()

            # We've already acquired mma_reduce_dq in the loop

            # dQ = dS * K
            dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdQrdS, mode=[2]), unroll_full=True):
                cute.gemm(
                    dSK_tiled_mma,
                    tdQtdQ,
                    tdQrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdQrKT[None, None, k_block, 0],
                    tdQtdQ,
                )
                dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            mma_reduce_dQ_pipeline.producer_commit(mma_reduce_dQ_producer_state)
            mma_reduce_dQ_producer_state.advance()

            load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
            load_mma_Q_release_state.advance()

            compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
            compute_mma_dS_consumer_state.advance()

    @cute.jit
    def compute(
        self,
        tSTtST: cute.Tensor,
        tdPTtdPT: cute.Tensor,
        tdVrP: cute.Tensor,
        sdS: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        KQ_thr_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        max_seqlen_q: Float32,
        alpha: Float32,
        # iter_count: Int32,
        # iter_index: Int32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        mask: AttentionMask,
        # (mma_compute_S_pipeline, compute_mma_P_pipeline, mma_compute_dP_pipeline, compute_mma_dS_pipeline, mma_compute_dKdV_pipeline)
        pipeline_args: tuple,
        block_info: BWDBlockInfo,
        SeqlenInfoCls: Callable,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        Q, K, D, HB = problem_shape
        blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_batch = blk_coord
        blk_coord_h, blk_coord_b = blk_coord_batch
        seqlen_obj = SeqlenInfoCls(blk_coord_b)


        n_block = blk_coord_k
        offset_dynamic = 0
        m_block_min, m_block_max, m_masking_steps, is_in_context, m_block_context = block_info.get_m_block_info(seqlen_obj, n_block, offset_dynamic) #TODO m_masking_steps what is m_masking_steps here? and  do we need it? [cause load don't need it]
        
        if cutlass.const_expr(self.is_arbitrary):
            m_block_max, m_block_min = block_info.get_bwd_valid_block_ids(seqlen_obj, n_block, m_block_min, m_block_max, is_calwarp=False)
        #define m_block
        m_block = 0  if cutlass.const_expr(self.is_context) else m_block_min #so this m_block_min is 0 for is_arbitary
        sValidBlockIds = block_info.sValidBlockIds

        (
            mma_compute_S_pipeline,
            compute_mma_P_pipeline,
            mma_compute_dP_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dKdV_pipeline,
        ) = pipeline_args

        mma_compute_S_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_compute_S_stage
        )
        compute_mma_P_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.compute_mma_P_stage
        )
        mma_compute_dP_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_compute_dP_stage
        )
        compute_mma_dS_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.compute_mma_dS_stage
        )
        mma_compute_dKdV_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_compute_dKdV_stage
        )

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)),
            self.acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)),
            self.element_dtype,
        )

        tSTtST = tSTtST[(None, None), 0, 0]
        tdPTtdPT = tdPTtdPT[(None, None), 0, 0]

        cST = cute.make_identity_tensor(cute.select(self.KQ_mma_tiler, mode=[0, 1]))
        cdPT = cute.make_identity_tensor(cute.select(self.VdO_mma_tiler, mode=[0, 1]))

        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128
        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tSTtST)
        thr_t2r = tiled_t2r.get_slice(dp_idx)

        tTR_cST = thr_t2r.partition_D(cST)
        tTR_cST = split_wg(tTR_cST, num_warp_groups, wg_idx)
        tTR_rST = cute.make_fragment(tTR_cST.shape, self.acc_dtype)

        tTR_tST = thr_t2r.partition_S(tSTtST)
        tTR_tST = split_wg(tTR_tST, num_warp_groups, wg_idx)

        tTR_cdPT_p = thr_t2r.partition_D(cdPT)
        tTR_cdPT = split_wg(tTR_cdPT_p, num_warp_groups, wg_idx)
        tTR_rdPT = cute.make_fragment(tTR_cdPT.shape, self.acc_dtype)

        tTR_tdPT = thr_t2r.partition_S(tdPTtdPT)
        tTR_tdPT = split_wg(tTR_tdPT, num_warp_groups, wg_idx)

        tdVcST = PdO_tiled_mma.get_slice(0).partition_A(cST)

        tiled_r2t = tcgen05.make_tmem_copy(tmem_store_atom, tdVrP)
        thr_r2t = tiled_r2t.get_slice(dp_idx)

        tRT_tP = thr_r2t.partition_D(tdVrP)
        tRT_tP = split_wg(tRT_tP, num_warp_groups, wg_idx)

        tRT_cST = thr_r2t.partition_S(tdVcST)
        tRT_cST = split_wg(tRT_cST, num_warp_groups, wg_idx)

        is_residual_k = blk_coord_k * self.tile_shape_K + self.tile_shape_K >= K
        mask_fn = partial(
            mask.apply_mask_swapAB, n_block=blk_coord_k, wg_idx=wg_idx,
            thr_mma=KQ_thr_mma, thr_tmem_load=thr_t2r,
        )

        fastsilu = FastSilU(alpha)

        if m_block_min >= m_block_max:
            self.epilogue_clear(
                blk_coord,
                blk_offset,
                problem_shape,
                dK,
                dV,
            )
        else:
            ## 
            # s_p_pipline_args = (mma_compute_S_pipeline, compute_mma_P_pipeline)
            # ds_dp_pipeline_args = (mma_compute_dP_pipeline, compute_mma_dS_pipeline)
            # while m_block < m_block_max:
            #     m_block_valid = sValidBlockIds[m_block] if cutlass.const_expr(self.is_arbitrary) else m_block
            #     mma_compute_S_consumer_state, compute_mma_P_producer_state, mma_compute_dP_consumer_state, compute_mma_dS_producer_state = self.compute_step(
            #         m_block_valid,
            #         s_p_pipline_args,
            #         mma_compute_S_consumer_state,
            #         compute_mma_P_producer_state,
            #         tiled_t2r,
            #         tiled_r2t,
            #         tTR_tST,
            #         tTR_rST,
            #         tRT_cST,
            #         tRT_tP,
            #         ds_dp_pipeline_args,
            #         mma_compute_dP_consumer_state,
            #         compute_mma_dS_producer_state,
            #         tTR_tdPT,
            #         tTR_rdPT,
            #         sdS,
            #         dp_idx,
            #         tTR_cdPT_p,
            #         num_warp_groups,
            #         wg_idx,
            #         alpha,
            #         mask_fn,
            #     )
            #     m_block += 1

            s_p_pipline_args = (mma_compute_S_pipeline, compute_mma_P_pipeline)
            ds_dp_pipeline_args = (mma_compute_dP_pipeline, compute_mma_dS_pipeline)
            compute_mask_step = partial(
                self.compute_step,
                s_p_pipline_args = s_p_pipline_args,
                mma_compute_S_consumer_state = mma_compute_S_consumer_state,
                compute_mma_P_producer_state = compute_mma_P_producer_state,
                tiled_t2r = tiled_t2r,
                tiled_r2t = tiled_r2t,
                tTR_tST = tTR_tST,
                tTR_rST = tTR_rST,
                tRT_cST = tRT_cST,
                tRT_tP = tRT_tP,
                ds_dp_pipeline_args = ds_dp_pipeline_args,
                mma_compute_dP_consumer_state = mma_compute_dP_consumer_state,
                compute_mma_dS_producer_state = compute_mma_dS_producer_state,
                tTR_tdPT = tTR_tdPT,
                tTR_rdPT = tTR_rdPT,
                sdS = sdS,
                dp_idx = dp_idx,
                tTR_cdPT_p = tTR_cdPT_p,
                num_warp_groups = num_warp_groups,
                wg_idx = wg_idx,
                alpha = alpha,
            )

            masking_step = 0
            if cutlass.const_expr(self.is_local or self.is_arbitrary):
                while m_block < m_block_max:
                    m_block_valid = sValidBlockIds[m_block] if cutlass.const_expr(self.is_arbitrary) else m_block
                    mma_compute_S_consumer_state, compute_mma_P_producer_state, mma_compute_dP_consumer_state, compute_mma_dS_producer_state = compute_mask_step(
                    m_block_valid = m_block_valid,
                    mma_compute_S_consumer_state = mma_compute_S_consumer_state,
                    compute_mma_P_producer_state = compute_mma_P_producer_state,
                    mma_compute_dP_consumer_state = mma_compute_dP_consumer_state,
                    compute_mma_dS_producer_state = compute_mma_dS_producer_state,
                    mask_fn = partial(mask_fn, mask_casual=True, mask_target=False, mask_seqlen=True))

                    m_block += 1

            while m_block < m_block_max and masking_step < m_masking_steps:
                m_block_valid = sValidBlockIds[m_block] if cutlass.const_expr(self.is_arbitrary) else m_block
                if cutlass.const_expr(self.is_target) and (n_block + 1) * self.kBlockN > seqlen_obj.seqlen_h:
                    mma_compute_S_consumer_state, compute_mma_P_producer_state, mma_compute_dP_consumer_state, compute_mma_dS_producer_state = compute_mask_step(
                    m_block_valid = m_block_valid,
                    mma_compute_S_consumer_state = mma_compute_S_consumer_state,
                    compute_mma_P_producer_state = compute_mma_P_producer_state,
                    mma_compute_dP_consumer_state = mma_compute_dP_consumer_state,
                    compute_mma_dS_producer_state = compute_mma_dS_producer_state,
                    mask_fn = partial(mask_fn, mask_casual=True, mask_target=True, mask_seqlen=True))
                else:
                    mma_compute_S_consumer_state, compute_mma_P_producer_state, mma_compute_dP_consumer_state, compute_mma_dS_producer_state = compute_mask_step(
                    m_block_valid = m_block_valid,
                    mma_compute_S_consumer_state = mma_compute_S_consumer_state,
                    compute_mma_P_producer_state = compute_mma_P_producer_state,
                    mma_compute_dP_consumer_state = mma_compute_dP_consumer_state,
                    compute_mma_dS_producer_state = compute_mma_dS_producer_state,
                    mask_fn = partial(mask_fn, mask_casual=True, mask_target=False, mask_seqlen=True))

                masking_step += 1
                m_block += 1
            
            while m_block < m_block_max - 1 and (n_block + 1) * self.kBlockN <= seqlen_obj.seqlen_k:
                m_block_valid = sValidBlockIds[m_block] if cutlass.const_expr(self.is_arbitrary) else m_block
                mma_compute_S_consumer_state, compute_mma_P_producer_state, mma_compute_dP_consumer_state, compute_mma_dS_producer_state = compute_mask_step(
                    m_block_valid = m_block_valid,
                    mma_compute_S_consumer_state = mma_compute_S_consumer_state,
                    compute_mma_P_producer_state = compute_mma_P_producer_state,
                    mma_compute_dP_consumer_state = mma_compute_dP_consumer_state,
                    compute_mma_dS_producer_state = compute_mma_dS_producer_state,
                    mask_fn = None)
                
                m_block += 1

            while m_block < m_block_max:
                m_block_valid = sValidBlockIds[m_block] if cutlass.const_expr(self.is_arbitrary) else m_block
                mma_compute_S_consumer_state, compute_mma_P_producer_state, mma_compute_dP_consumer_state, compute_mma_dS_producer_state = compute_mask_step(
                    m_block_valid = m_block_valid,
                    mma_compute_S_consumer_state = mma_compute_S_consumer_state,
                    compute_mma_P_producer_state = compute_mma_P_producer_state,
                    mma_compute_dP_consumer_state = mma_compute_dP_consumer_state,
                    compute_mma_dS_producer_state = compute_mma_dS_producer_state,
                    mask_fn = partial(mask_fn, mask_casual=False, mask_target=False, mask_seqlen=True))
                
                m_block += 1


                    





            # if cutlass.const_expr(self.is_arbitrary or self.is_local):
            #     while m_block < m_block_max:
            #         m_block_valid = sValidBlockIds[m_block] if cutlass.const_expr(self.is_arbitrary) else m_block

            #         mma_compute_S_consumer_state, compute_mma_P_producer_state, mma_compute_dP_consumer_state, compute_mma_dS_producer_state = compute_mask_step(
            #         m_block_valid = m_block_valid,
            #         mma_compute_S_consumer_state = mma_compute_S_consumer_state,
            #         compute_mma_P_producer_state = compute_mma_P_producer_state,
            #         mma_compute_dP_consumer_state = mma_compute_dP_consumer_state,
            #         compute_mma_dS_producer_state = compute_mma_dS_producer_state,
            #         mask_fn = mask_fn)

            #         m_block += 1
            
            # while m_block < m_block_max and masking_step < m_masking_steps:
            #     m_block_valid = sValidBlockIds[m_block] if cutlass.const_expr(self.is_arbitrary) else m_block
            #     if self.is_target and (n_block + 1) * self.kBlockN > seqlen.seqlen_h:
            #         mma_compute_S_consumer_state, compute_mma_P_producer_state, mma_compute_dP_consumer_state, compute_mma_dS_producer_state = compute_mask_step(
            #         m_block_valid = m_block_valid,
            #         mma_compute_S_consumer_state = mma_compute_S_consumer_state,
            #         compute_mma_P_producer_state = compute_mma_P_producer_state,
            #         mma_compute_dP_consumer_state = mma_compute_dP_consumer_state,
            #         compute_mma_dS_producer_state = compute_mma_dS_producer_state,
            #         mask_fn = partial(mask_fn, mask_target=True))
            #     else:
            #         mma_compute_S_consumer_state, compute_mma_P_producer_state, mma_compute_dP_consumer_state, compute_mma_dS_producer_state = compute_mask_step(
            #         m_block_valid = m_block_valid,
            #         mma_compute_S_consumer_state = mma_compute_S_consumer_state,
            #         compute_mma_P_producer_state = compute_mma_P_producer_state,
            #         mma_compute_dP_consumer_state = mma_compute_dP_consumer_state,
            #         compute_mma_dS_producer_state = compute_mma_dS_producer_state,
            #         mask_fn = partial(mask_fn, mask_target=False))

            #         masking_step += 1
            #         m_block += 1
            
            # while m_block < m_block_max:
            #     m_block_valid = sValidBlockIds[m_block] if cutlass.const_expr(self.is_arbitrary) else m_block

            #     mma_compute_S_consumer_state, compute_mma_P_producer_state, mma_compute_dP_consumer_state, compute_mma_dS_producer_state = compute_mask_step(
            #         m_block_valid = m_block_valid,
            #         mma_compute_S_consumer_state = mma_compute_S_consumer_state,
            #         compute_mma_P_producer_state = compute_mma_P_producer_state,
            #         mma_compute_dP_consumer_state = mma_compute_dP_consumer_state,
            #         compute_mma_dS_producer_state = compute_mma_dS_producer_state,
            #         mask_fn = None) #don't need to mask, full attention
                
            #     m_block += 1

            # Epilogue
            self.epilogue(
                blk_coord,
                blk_offset,
                problem_shape,
                max_seqlen_q,
                dK,
                dV,
                tdKtdK,
                tdVtdV,
                (mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state),
            )

    @cute.jit
    def compute_step(
        self,
        m_block_valid: Int32,
        s_p_pipline_args: tuple,
        mma_compute_S_consumer_state: cutlass.pipeline.PipelineState,
        compute_mma_P_producer_state: cutlass.pipeline.PipelineState,
        tiled_t2r: cute.TiledCopy,
        tiled_r2t: cute.TiledCopy,
        tTR_tST: cute.Tensor,
        tTR_rST: cute.Tensor,
        tRT_cST: cute.Tensor,
        tRT_tP: cute.Tensor,
        ds_dp_pipeline_args: tuple,
        mma_compute_dP_consumer_state: cutlass.pipeline.PipelineState,
        compute_mma_dS_producer_state: cutlass.pipeline.PipelineState,
        tTR_tdPT: cute.Tensor,
        tTR_rdPT: cute.Tensor,
        sdS: cute.Tensor,
        dp_idx: Int32,
        tTR_cdPT_p: cute.Tensor,
        num_warp_groups: Int32,
        wg_idx: Int32,
        alpha: Float32,
        mask_fn: Optional[Callable] = None,
    ):
        (mma_compute_S_pipeline, compute_mma_P_pipeline) = s_p_pipline_args
        (mma_compute_dP_pipeline, compute_mma_dS_pipeline)= ds_dp_pipeline_args

        mma_compute_S_pipeline.consumer_wait(mma_compute_S_consumer_state)
        compute_mma_P_pipeline.producer_acquire(compute_mma_P_producer_state)

        # Compute P = silu(S)
        cute.copy(tiled_t2r, tTR_tST, tTR_rST)
        tTR_rST_preds = cute.make_rmem_tensor(tTR_rST.shape, cutlass.Boolean)
        for i in cutlass.range(0, cute.size(tTR_rST), unroll_full=True):
            tTR_rST_preds[i] = True

        if cutlass.const_expr(mask_fn is not None):
            mask_fn(tTR_rST_preds, m_block=m_block_valid)
        tTR_rST_silu = cute.make_fragment_like(tTR_rST)

        # fastsilu.dsilu_bwd_x2(tTR_rST, tTR_rST_silu, tTR_rST_preds, mask_fn)
        # for i in cutlass.range(0, cute.size(tTR_rST), unroll_full=True):
        #     tTR_rST[i] *= alpha
        #     sigmoid_v = 0.5 * cute.math.tanh(tTR_rST[i] * 0.5, fastmath=True) + 0.5
        #     out = tTR_rST[i] * sigmoid_v
        #     temp = sigmoid_v * (1 + tTR_rST[i] * (1 - sigmoid_v)) = silu*(1-sigmoid) + (1-sigmoid) #
        #     dsilu_temp = temp if tTR_rST_preds[i] else 0.0
        #     silu_out = out if tTR_rST_preds[i] else 0.0
        #     tTR_rST[i] = dsilu_temp
        #     tTR_rST_silu[i] = silu_out

        for i in cutlass.range_constexpr(0, cute.size(tTR_rST), 2):
            v0, v1 = mul_packed_f32x2((tTR_rST[i], tTR_rST[i + 1]), (alpha, alpha))
            tanh_in0, tanh_in1 = mul_packed_f32x2((v0, v1), (0.5, 0.5))
            tanh_v0 = tanhf(tanh_in0)
            tanh_v1 = tanhf(tanh_in1)
            sigmoid_v0, sigmoid_v1 = fma_packed_f32x2((0.5, 0.5), (tanh_v0, tanh_v1), (0.5, 0.5))
            sigmoid_v0 = sigmoid_v0 if tTR_rST_preds[i] else tTR_rST.element_type(0)
            sigmoid_v1 = sigmoid_v1 if tTR_rST_preds[i + 1] else tTR_rST.element_type(0)
            out_v0, out_v1 = mul_packed_f32x2((v0, v1), (sigmoid_v0, sigmoid_v1))
            one_minus_sig0, one_minus_sig1 = sub_packed_f32x2((1.0, 1.0), (sigmoid_v0, sigmoid_v1))
            inner0, inner1 = fma_packed_f32x2((v0, v1), (one_minus_sig0, one_minus_sig1), (1.0, 1.0))
            dsilu0, dsilu1 = mul_packed_f32x2((sigmoid_v0, sigmoid_v1), (inner0, inner1))
            tTR_rST[i] = dsilu0
            tTR_rST[i + 1] = dsilu1
            tTR_rST_silu[i] = out_v0
            tTR_rST_silu[i + 1] = out_v1

        # convert fp32 P to fp16 P which will be used in the PdO
        tRT_rST = self.quantize(tTR_rST_silu, 4)

        tRT_rST_reshaped = cute.make_tensor(
            tRT_rST.iterator, cute.make_layout(tRT_cST.shape)
        )

        cute.arch.fence_view_async_tmem_load()
        self.compute_sync_barrier.arrive_and_wait()

        cute.copy(tiled_r2t, tRT_rST_reshaped, tRT_tP)

        cute.arch.fence_view_async_tmem_store()

        # Notify for P
        compute_mma_P_pipeline.producer_commit(compute_mma_P_producer_state)
        compute_mma_P_producer_state.advance()

        # Release S
        mma_compute_S_pipeline.consumer_release(mma_compute_S_consumer_state)
        mma_compute_S_consumer_state.advance()

        # Wait for dP
        mma_compute_dP_pipeline.consumer_wait(mma_compute_dP_consumer_state)

        # Wait for dS
        compute_mma_dS_pipeline.producer_acquire(compute_mma_dS_producer_state)

        # Compute dS = dsilu(S, dP)
        cute.copy(tiled_t2r, tTR_tdPT, tTR_rdPT)

        for i in cutlass.range(0, cute.size(tTR_rdPT), unroll_full=True):
            tTR_rdPT[i] *= alpha
            tTR_rdPT[i] = tTR_rST[i] * tTR_rdPT[i]

        # for i in cutlass.range(0, cute.size(tTR_rdPT), unroll_full=True):
        #     tTR_rdPT[i] = tTR_rST[i] * tTR_rdPT[i]

        # convert fp32 dS to fp16 dS which will be used in the computation of dK and DQ
        tTR_rdST = self.quantize(tTR_rdPT, 4)

        # Release dP
        cute.arch.fence_view_async_tmem_load()
        mma_compute_dP_pipeline.consumer_release(mma_compute_dP_consumer_state)
        mma_compute_dP_consumer_state.advance()

        sdS_slice = sdS[None, None, None, compute_mma_dS_producer_state.index]

        thread_layout = cute.make_ordered_layout((128, 128), (1, 0))
        sdS_slice_tmp = cute.composition(sdS_slice, thread_layout)
        sdS_slice_p = cute.composition(
            sdS_slice_tmp[dp_idx, None], cute.make_layout(tTR_cdPT_p.shape)
        )
        sdS_slice = split_wg(sdS_slice_p, num_warp_groups, wg_idx)

        cute.autovec_copy(tTR_rdST, sdS_slice)

        # Notify for dS
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )
        compute_mma_dS_pipeline.producer_commit(compute_mma_dS_producer_state)
        compute_mma_dS_producer_state.advance()

        return mma_compute_S_consumer_state, compute_mma_P_producer_state, mma_compute_dP_consumer_state, compute_mma_dS_producer_state



    @cute.jit
    def reduce(
        self,
        dSK_tiled_mma: cute.TiledMma,
        tdQtdQ: cute.Tensor,
        tma_atom_dQ_acc: cute.CopyAtom,
        mdQ_acc: cute.Tensor,
        sdQ: cute.Tensor,
        blk_coord: cute.Coord,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        max_seqlen_q: Float32,
        # iter_count: Int32,
        # iter_index: Int32,
        # (mma_reduce_dQ_pipeline, reduce_tma_store_pipeline)
        pipeline_args: tuple,
        block_info: BWDBlockInfo,
        SeqlenInfoCls: Callable,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        Q, K, D, HB = problem_shape
        blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_batch = blk_coord

        blk_coord_h, blk_coord_b = blk_coord_batch
        blk_coord_h_r, blk_coord_h_k = blk_coord_h
        seqlen_obj = SeqlenInfoCls(blk_coord_b)

        n_block = blk_coord_k
        offset_dynamic = 0
        m_block_min, m_block_max, m_masking_steps, is_in_context, m_block_context = block_info.get_m_block_info(seqlen_obj, n_block, offset_dynamic) #TODO m_masking_steps what is m_masking_steps here? and  do we need it? [cause load don't need it]

        if cutlass.const_expr(self.is_arbitrary):
            m_block_max, m_block_min = block_info.get_bwd_valid_block_ids(seqlen_obj, n_block, m_block_min, m_block_max, is_calwarp=False)
        #define the init m_block id
        m_block = 0 if cutlass.const_expr(self.is_context) else m_block_min
        sValidBlockIds = block_info.sValidBlockIds

        mma_reduce_dQ_pipeline, reduce_tma_store_pipeline = pipeline_args

        mma_reduce_dQ_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_reduce_dQ_stage
        )
        reduce_tma_store_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.reduce_tma_store_stage
        )

        load_op = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.acc_dtype,
        )

        gdQ = cute.local_tile(mdQ_acc, (self.KQ_mma_tiler[1], 32), (None, None, None))

        cdQ = cute.make_identity_tensor((self.dSK_mma_tiler[0], self.dSK_mma_tiler[1]))

        thread_idx = tidx % (self.num_compute_warps * self.threads_per_warp)

        tdQtdQ = tdQtdQ[(None, None), 0, 0]

        tiled_t2r = tcgen05.make_tmem_copy(load_op, tdQtdQ)
        thr_t2r = tiled_t2r.get_slice(thread_idx)

        tTR_cdQ = thr_t2r.partition_D(cdQ)
        tTR_gdQ = thr_t2r.partition_D(gdQ)
        tTR_sdQ = thr_t2r.partition_D(sdQ)
        tTR_tdQ = thr_t2r.partition_S(tdQtdQ)

        tdQsdQ, tdQgdQ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dQ_acc,
            0,
            cute.make_layout(1),
            cute.group_modes(sdQ, 0, 2),
            cute.group_modes(gdQ, 0, 2),
        )

        inv_max_seq_len = 1.0 / max_seqlen_q

        if m_block_min < m_block_max:
            reduce_pipeline_args = (mma_reduce_dQ_pipeline, reduce_tma_store_pipeline)
            while m_block < m_block_max:
                m_block_valid = sValidBlockIds[m_block] if cutlass.const_expr(self.is_arbitrary) else m_block
                mma_reduce_dQ_consumer_state, reduce_tma_store_producer_state = self.store_dq_step(
                    m_block_valid,
                    reduce_pipeline_args,
                    mma_reduce_dQ_consumer_state,
                    tTR_cdQ,
                    tiled_t2r,
                    tTR_tdQ,
                    max_seqlen_q,
                    reduce_tma_store_producer_state,
                    tTR_sdQ,
                    tdQsdQ,
                    tdQgdQ,
                    tma_atom_dQ_acc,
                    blk_coord_batch,
                    warp_idx,
                    inv_max_seq_len
                )

                m_block += 1
            reduce_tma_store_pipeline.producer_tail()


    @cute.jit
    def store_dq_step(
        self,
        m_block_valid: Int32,
        pipeline_args: tuple,
        mma_reduce_dQ_consumer_state: cutlass.pipeline.PipelineState,
        tTR_cdQ: cute.Tensor,
        tiled_t2r: cute.TiledCopy,
        tTR_tdQ: cute.Tensor,
        max_seqlen_q: Int32,
        reduce_tma_store_producer_state: cutlass.pipeline.PipelineState,
        tTR_sdQ: cute.Tensor,
        tdQsdQ: cute.Tensor,
        tdQgdQ: cute.Tensor,
        tma_atom_dQ_acc: cute.CopyAtom,
        blk_coord_batch: Tuple[Int32, Int32],
        warp_idx: Int32,
        inv_max_seq_len: Float32
    ):
        (mma_reduce_dQ_pipeline, reduce_tma_store_pipeline) = pipeline_args
        mma_reduce_dQ_pipeline.consumer_wait(mma_reduce_dQ_consumer_state)

        tTR_rdQ = cute.make_fragment(tTR_cdQ.shape, self.acc_dtype)

        # Load dQ from tmem to rmem
        cute.copy(tiled_t2r, tTR_tdQ, tTR_rdQ)

        for i in cutlass.range(0, cute.size(tTR_rdQ), 2, unroll_full=True):
            tTR_rdQ[i], tTR_rdQ[i + 1] = mul_packed_f32x2((tTR_rdQ[i], tTR_rdQ[i + 1]), (inv_max_seq_len, inv_max_seq_len))

        cute.arch.fence_view_async_tmem_load()

        mma_reduce_dQ_pipeline.consumer_release(mma_reduce_dQ_consumer_state)
        mma_reduce_dQ_consumer_state.advance()

        # We don't have enough smem to dump it all to smem, so we do it in stages
        for i in cutlass.range(0, cute.size(tTR_cdQ, mode=[2]), unroll_full=True):
            if warp_idx == 0:
                reduce_tma_store_pipeline.producer_acquire()
            # Wait in all threads for the acquire to complete
            self.reduce_sync_barrier.arrive_and_wait()

            cute.autovec_copy(
                tTR_rdQ[None, None, i],
                tTR_sdQ[None, None, 0, reduce_tma_store_producer_state.index],
            )

            # Wait for the stores to all be visible to the TMA
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            self.reduce_sync_barrier.arrive_and_wait()

            if warp_idx == 0:
                cute.copy(
                    tma_atom_dQ_acc,
                    tdQsdQ[None, reduce_tma_store_producer_state.index],
                    tdQgdQ[None, m_block_valid, i, blk_coord_batch],
                )

                reduce_tma_store_pipeline.producer_commit()

            reduce_tma_store_producer_state.advance()

        return mma_reduce_dQ_consumer_state, reduce_tma_store_producer_state

    @cute.jit
    def quantize(
        self,
        input: cute.Tensor,
        frg_cnt: Int32,
    ) -> cute.Tensor:
        tidx, tidy, tidz = cute.arch.thread_idx()
        output = cute.make_fragment(input.shape, self.element_dtype)
        frg_tile = cute.size(input) // frg_cnt
        t_frg = cute.logical_divide(input, cute.make_layout(frg_cnt))
        output_frg = cute.make_tensor(output.iterator, t_frg.layout)
        for i in cutlass.range(frg_tile, unroll_full=True):
            frg_vec = t_frg[None, i].load()
            output_frg[None, i].store(frg_vec.to(self.element_dtype))
        return output

    @cute.jit
    def store(
        self,
        gmem: cute.Tensor,
        regs: cute.Tensor,
        coord: cute.Tensor,
        tensor_shape: cute.Shape,
    ):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.element_dtype,
            num_bits_per_copy=128,
        )
        copy_op = cute.make_cotiled_copy(
            copy_atom,
            cute.make_layout((1, 128 // self.element_dtype.width)),
            regs.layout,
        )
        thr_copy = copy_op.get_slice(0)

        tCg = thr_copy.partition_D(gmem)
        tCr = thr_copy.partition_S(self.quantize(regs, 4))
        tPc = thr_copy.partition_D(coord)

        # {$nv-internal-release begin}
        # FIXME cute.copy expects mode 0 (atom_v,rest_v) to be removed
        #       Fix this so that the predicate tensor can simply be congruent to
        #       the original partitioned tensor
        # {$nv-internal-release end}
        preds_shape = (tPc.shape[0][1], tPc.shape[1], tPc.shape[2], tPc.shape[3])
        preds = cute.make_fragment(preds_shape, Boolean)
        
        tPc_fake = cute.group_modes(tPc, 1, 4)
        preds_shape_fake = (tPc_fake.shape[0][1], cute.size(tPc_fake, mode=[1]))
        preds_fake = cute.make_tensor(preds.iterator, preds_shape_fake)

        for i in cutlass.range_constexpr(0, cute.size(preds_fake, mode=[0])):
            for j in cutlass.range_constexpr(0, cute.size(preds_fake, mode=[1])):
                lhs = tPc_fake[(0, i), j]
                val = cute.elem_less(lhs, tensor_shape)
                preds_fake[i, j] = val

        cute.copy(copy_atom, tCr, tCg, pred=preds)

    @cute.jit
    def epilogue_clear(
        self,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: cute.Shape,
        dK: cute.Tensor,
        dV: cute.Tensor,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        block_dim_x, block_dim_y, block_dim_z = cute.arch.block_dim()
        Q, K, D, HB = problem_shape
        blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_batch = blk_coord
        n_block = blk_coord_k

        mdK = cute.make_tensor(
            dK.iterator + blk_offset[1] * dK.stride[0],
            cute.make_layout((K, self.tile_shape_dQ_K, HB), stride=dK.stride),
        )
        gdK = cute.local_tile(
            mdK, (self.dSQ_mma_tiler[0], self.dSQ_mma_tiler[1]), (None, None, None)
        )
        gdK = gdK[None, None, blk_coord_k, 0, blk_coord_batch]

        cdK = cute.domain_offset(
            (blk_coord_k * self.tile_shape_K, 0),
            cute.make_identity_tensor((self.dSQ_mma_tiler[0], self.dSQ_mma_tiler[1])),
        )

        mdV = cute.make_tensor(
            dV.iterator + blk_offset[1] * dV.stride[0],
            cute.make_layout((K, self.tile_shape_dV_dO, HB), stride=dV.stride),
        )
        gdV = cute.local_tile(
            mdV, (self.PdO_mma_tiler[0], self.PdO_mma_tiler[1]), (None, None, None)
        )
        gdV = gdV[None, None, blk_coord_k, 0, blk_coord_batch]

        cdV = cute.domain_offset(
            (blk_coord_k * self.tile_shape_K, 0),
            cute.make_identity_tensor((self.PdO_mma_tiler[0], self.PdO_mma_tiler[1])),
        )

        for i in cutlass.range(tidx - 128, cute.size(gdK), 256):
            if cute.elem_less(cdK[i], cute.select(problem_shape, mode=[1, 2])):
                gdK[i] = self.element_dtype(0)
        for i in cutlass.range(tidx - 128, cute.size(gdV), 256):
            if cute.elem_less(cdV[i], cute.select(problem_shape, mode=[1, 2])):
                gdV[i] = self.element_dtype(0)

    @cute.jit
    def epilogue(
        self,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        max_seqlen_q: Float32,
        dK: cute.Tensor,
        dV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        # (mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state)
        pipeline_args: tuple,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        Q, K, D, HB = problem_shape
        blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_batch = blk_coord
        mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state = pipeline_args

        load_op = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)),
            self.acc_dtype,
        )

        tdKtdK = tdKtdK[(None, None), 0, 0]

        mdK = cute.make_tensor(
            dK.iterator + cute.assume(blk_offset[1] * dK.stride[0], divby=64),
            cute.make_layout((K, self.tile_shape_dQ_K, HB), stride=dK.stride),
        )
        gdK = cute.local_tile(
            mdK, (self.dSQ_mma_tiler[0], self.dSQ_mma_tiler[1]), (None, None, None)
        )
        gdK = gdK[None, None, blk_coord_k, 0, blk_coord_batch]

        cdK = cute.domain_offset(
            (blk_coord_k * self.tile_shape_K, 0),
            cute.make_identity_tensor((self.dSQ_mma_tiler[0], self.dSQ_mma_tiler[1])),
        )

        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128

        tiled_t2r_dK = tcgen05.make_tmem_copy(load_op, tdKtdK)
        thread_t2r_dK = tiled_t2r_dK.get_slice(dp_idx)

        tTR_cdK = thread_t2r_dK.partition_D(cdK)
        tTR_cdK = split_wg(tTR_cdK, num_warp_groups, wg_idx)
        tTR_gdK = thread_t2r_dK.partition_D(gdK)
        tTR_gdK = split_wg(tTR_gdK, num_warp_groups, wg_idx)
        tTR_rdK = cute.make_fragment(tTR_cdK.shape, self.acc_dtype)
        tTR_tdK = thread_t2r_dK.partition_S(tdKtdK)
        tTR_tdK = split_wg(tTR_tdK, num_warp_groups, wg_idx)

        mdV_in = cute.make_tensor(
            dV.iterator, cute.make_layout((K, self.cta_tiler[2], HB), stride=dV.stride)
        )
        mdV = cute.make_tensor(
            mdV_in.iterator + cute.assume(blk_offset[1] * mdV_in.stride[0], divby=64),
            mdV_in.layout,
        )
        gdV = cute.local_tile(
            mdV, (self.PdO_mma_tiler[0], self.PdO_mma_tiler[1]), (None, None, None)
        )
        gdV = gdV[None, None, blk_coord_k, 0, blk_coord_batch]

        cdV = cute.domain_offset(
            (blk_coord_k * self.cta_tiler[1], 0),
            cute.make_identity_tensor((self.PdO_mma_tiler[0], self.PdO_mma_tiler[1])),
        )

        tdVtdV = tdVtdV[(None, None), 0, 0]

        tiled_t2r_dV = tcgen05.make_tmem_copy(load_op, tdVtdV)
        thread_t2r_dV = tiled_t2r_dV.get_slice(dp_idx)

        tTR_cdV = thread_t2r_dV.partition_D(cdV)
        tTR_cdV = split_wg(tTR_cdV, num_warp_groups, wg_idx)
        tTR_gdV = thread_t2r_dV.partition_D(gdV)
        tTR_gdV = split_wg(tTR_gdV, num_warp_groups, wg_idx)
        tTR_rdV = cute.make_fragment(tTR_cdV.shape, self.acc_dtype)
        tTR_tdV = thread_t2r_dV.partition_S(tdVtdV)
        tTR_tdV = split_wg(tTR_tdV, num_warp_groups, wg_idx)

        inv_max_seq_len = 1.0 / max_seqlen_q

        mma_compute_dKdV_pipeline.consumer_wait(mma_compute_dKdV_consumer_state)
        

        # Load tdVtdV
        cute.copy(tiled_t2r_dV, tTR_tdV, tTR_rdV)

        for i in cutlass.range(0, cute.size(tTR_rdV), 2, unroll_full=True):
            tTR_rdV[i], tTR_rdV[i + 1] = mul_packed_f32x2((tTR_rdV[i], tTR_rdV[i + 1]), (inv_max_seq_len, inv_max_seq_len))

        # Store tdVgdV
        self.store(tTR_gdV, tTR_rdV, tTR_cdV, (K, D))

        cute.arch.fence_view_async_tmem_load()

        mma_compute_dKdV_pipeline.consumer_release(mma_compute_dKdV_consumer_state)
        mma_compute_dKdV_consumer_state.advance()

        mma_compute_dKdV_pipeline.consumer_wait(mma_compute_dKdV_consumer_state)

        # Load tdKtdK
        cute.copy(tiled_t2r_dK, tTR_tdK, tTR_rdK)

        for i in cutlass.range(0, cute.size(tTR_rdK), 2, unroll_full=True):
            tTR_rdK[i], tTR_rdK[i + 1] = mul_packed_f32x2((tTR_rdK[i], tTR_rdK[i + 1]), (inv_max_seq_len, inv_max_seq_len))

        # Store tdKgdK
        self.store(tTR_gdK, tTR_rdK, tTR_cdK, (K, D))

        cute.arch.fence_view_async_tmem_load()
        mma_compute_dKdV_pipeline.consumer_release(mma_compute_dKdV_consumer_state)
        mma_compute_dKdV_consumer_state.advance()

    def get_workspace_tensor(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        workspace: cute.Tensor,
    ) -> Tuple[cute.Tensor, cute.Tensor, cute.Tensor]:
        Q, D, HB = (
            problem_shape[0],
            problem_shape[2],
            problem_shape[3],
        )
        H, B = cute.size(problem_shape[3][0]), cute.size(problem_shape[3][1])
        H_r, H_k = problem_shape[3][0]
        D = cute.round_up(D, 8)
        Q = cute.round_up(Q, 8)

        acc_bytes = self.acc_dtype.width // 8
        dQ_acc_bytes = cute.assume(B * H * Q * D * acc_bytes, divby=acc_bytes)

        dQ_acc_iter = workspace.iterator

        dQ_acc_iter = cute.recast_ptr(dQ_acc_iter, dtype=self.acc_dtype)

        dQ_acc = cute.make_tensor(
            dQ_acc_iter,
            cute.make_layout(
                (Q, D, ((H_r, H_k), B)),
                stride=(D, 1, ((D * Q, D * Q * H_r), D * Q * H)),
            ),
        )

        return dQ_acc

    @staticmethod
    def _compute_bwd_grid(
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
        block_k: int,
    ) -> Tuple[int, int, int]:
        K = problem_shape[1]
        H_R, H_K = problem_shape[3][0]
        B = problem_shape[3][1]
        return (cute.ceil_div(K, block_k), cute.size(H_K), cute.size(B))

    @staticmethod
    def _get_workspace_size(
        q: int, k: int, d: int, h: int, b: int
    ):
        acc_dtype = Float32
        d = (d + 7) // 8 * 8  # round up to 8
        q = (q + 7) // 8 * 8  # round up to 8
        workspace_bytes = 0
        # FP32 versions of outputs that are churned (start off with Q only)
        workspace_bytes += b * h * q * d * acc_dtype.width // 8
        return workspace_bytes

    def make_and_init_load_mma_Q_pipeline(self, load_mma_Q_mbar_ptr):
        load_mma_Q_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_warp_id])
        )
        load_mma_Q_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_Q_mbar_ptr,
            num_stages=self.load_mma_Q_stage,
            producer_group=load_mma_Q_producer_group,
            consumer_group=load_mma_Q_consumer_group,
            tx_count=self.tma_copy_Q_bytes,
        )

    def make_and_init_load_mma_dO_pipeline(self, load_mma_dO_mbar_ptr):
        load_mma_dO_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_warp_id])
        )
        load_mma_dO_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_dO_mbar_ptr,
            num_stages=self.load_mma_dO_stage,
            producer_group=load_mma_dO_producer_group,
            consumer_group=load_mma_dO_consumer_group,
            tx_count=self.tma_copy_dO_bytes,
        )


    def make_and_init_mma_compute_S_pipeline(self, mma_compute_S_mbar_ptr):
        mma_compute_S_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_S_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
            self.num_compute_warps * self.threads_per_warp,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_S_mbar_ptr,
            num_stages=self.mma_compute_S_stage,
            producer_group=mma_compute_S_producer_group,
            consumer_group=mma_compute_S_consumer_group,
        )

    def make_and_init_mma_compute_dP_pipeline(self, mma_compute_dP_mbar_ptr):
        mma_compute_dP_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_dP_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
            self.num_compute_warps * self.threads_per_warp,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dP_mbar_ptr,
            num_stages=self.mma_compute_dP_stage,
            producer_group=mma_compute_dP_producer_group,
            consumer_group=mma_compute_dP_consumer_group,
        )

    def make_and_init_mma_reduce_dQ_pipeline(self, mma_reduce_dQ_mbar_ptr):
        mma_reduce_dQ_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_reduce_dQ_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_reduce_warps * self.threads_per_warp,
            self.num_reduce_warps * self.threads_per_warp,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_reduce_dQ_mbar_ptr,
            num_stages=self.mma_reduce_dQ_stage,
            producer_group=mma_reduce_dQ_producer_group,
            consumer_group=mma_reduce_dQ_consumer_group,
        )

    def make_and_init_compute_mma_P_pipeline(self, compute_mma_P_mbar_ptr):
        compute_mma_P_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
            self.num_compute_warps * self.threads_per_warp,
        )
        compute_mma_P_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=compute_mma_P_mbar_ptr,
            num_stages=self.compute_mma_P_stage,
            producer_group=compute_mma_P_producer_group,
            consumer_group=compute_mma_P_consumer_group,
        )

    def make_and_init_compute_mma_dS_pipeline(self, compute_mma_dS_mbar_ptr):
        compute_mma_dS_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
            self.num_compute_warps * self.threads_per_warp,
        )
        compute_mma_dS_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )

        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=compute_mma_dS_mbar_ptr,
            num_stages=self.compute_mma_dS_stage,
            producer_group=compute_mma_dS_producer_group,
            consumer_group=compute_mma_dS_consumer_group,
        )

    def make_and_init_mma_compute_dKdV_pipeline(self, mma_compute_dKdV_mbar_ptr):
        mma_compute_dKdV_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_dKdV_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
            self.num_compute_warps * self.threads_per_warp,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dKdV_mbar_ptr,
            num_stages=self.mma_compute_dKdV_stage,
            producer_group=mma_compute_dKdV_producer_group,
            consumer_group=mma_compute_dKdV_consumer_group,
        )

    def make_and_init_reduce_tma_store_pipeline(self):
        reduce_tma_store_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_reduce_warps * self.threads_per_warp,
            self.num_reduce_warps * self.threads_per_warp,
        )
        return pipeline.PipelineTmaStore.create(
            num_stages=self.reduce_tma_store_stage,
            producer_group=reduce_tma_store_producer_group,
        )


def run(
    s_q: int | Tuple[int, ...],
    s_k: int | Tuple[int, ...],
    h_q: int,
    h_k: int,
    d: int,
    b: int,
    is_causal: bool,
    element_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    mma_tiler_mn: Tuple[int, int],
    window_size: Tuple[int, int],
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool = False,
    **kwargs,
):
    print("Running Blackwell SM100 FMHA bwd test with:")
    print(f"  s_q: {s_q}")
    print(f"  s_k: {s_k}")
    print(f"  h_q: {h_q}")
    print(f"  h_k: {h_k}")
    print(f"  d: {d}")
    print(f"  b: {b}")
    print(f"  is_causal: {is_causal}")
    print(f"  element_dtype: {element_dtype}")
    print(f"  acc_dtype: {acc_dtype}")
    print(f"  mma_tiler_mn: {mma_tiler_mn}")
    print(f"  window_size: {window_size}")
    print(f"  warmup_iterations: {warmup_iterations}")
    print(f"  iterations: {iterations}")
    print(f"  skip_ref_check: {skip_ref_check}")

    if d not in {64, 128}:
        raise ValueError("head dimension must be 64, or 128")

    if h_q % h_k != 0:
        raise ValueError("h_q must be divisible by h_k")

    if element_dtype not in {Float8E4M3FN, Float16, BFloat16}:
        raise ValueError("in_dtype must be Float8E4M3FN or Float16 or BFloat16")

    if acc_dtype not in {Float32}:
        raise ValueError("acc_dtype must be Float32")

    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    if isinstance(s_q, tuple) and len(s_q) != b:
        raise ValueError("s_q must be a tuple of length b")

    window_size_left, window_size_right = window_size
    if window_size_left == -1:
        window_size_left = None
    if window_size_right == -1:
        window_size_right = None

    h_r = h_q // h_k
    orig_b = b

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    def create_and_permute_tensor(
        shape,
        permute_order,
        dtype,
        min_val=-1,
        max_val=1,
        is_dynamic_layout=True,
        zero_out=False,
    ):
        # (b, h_k, h_r, s, d) -> (s, d, h_r, h_k, b)
        ref_tensor = (
            torch.empty(*shape, dtype=torch.float32)
            .uniform_(min_val, max_val)
            .permute(permute_order)
        )
        print(ref_tensor.shape)
        print(ref_tensor.stride())
        if zero_out:
            ref_tensor.zero_()

        torch_dtype = cutlass_torch.dtype(dtype)

        dst_tensor = ref_tensor.to(dtype=torch_dtype).cuda()
        cute_tensor = from_dlpack(dst_tensor, assumed_align=16)
        cute_tensor.element_type = dtype
        if is_dynamic_layout:
            cute_tensor = cute_tensor.mark_layout_dynamic(
                leading_dim=1
            ).mark_compact_shape_dynamic(
                mode=1, stride_order=(4, 3, 2, 0, 1), divisibility=64
            )

        return ref_tensor, cute_tensor, dst_tensor

    # create sequence lengths for variable length inputs
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    varlen = False
    if isinstance(s_q, tuple):
        varlen = True
        for i in range(b):
            cu_seqlens_q.append(cu_seqlens_q[-1] + s_q[i])
            cu_seqlens_k.append(cu_seqlens_k[-1] + s_k[i])
        s_q_max = max(s_q)
        s_k_max = max(s_k)
        s_q = sum(s_q)
        s_k = sum(s_k)
        b = 1
    else:
        s_q_max = s_q
        s_k_max = s_k

    mask_type = MaskType.NO_MASK
    if varlen or s_q % mma_tiler_mn[0] != 0:
        mask_type = MaskType.RESIDUAL_MASK_FOR_BACKWARD
    if is_causal:
        mask_type = MaskType.CAUSAL_MASK_FOR_BACKWARD

    window_size_left, window_size_right = window_size
    if is_causal:
        window_size_right = 0
    if window_size_left >= s_k_max - 1:
        raise ValueError("window_size_left must be less than s_k_max - 1")
    if window_size_right >= s_q_max - 1:
        raise ValueError("window_size_right must be less than s_q_max - 1")

    if window_size_left > 0 or window_size_right > 0:
        if isinstance(s_q, tuple):
            for i in range(b):
                if s_q[i] != s_k[i]:
                    # TODO: support s_q != s_k for sliding window
                    raise ValueError("s_q and s_k must be the same for sliding window")

    problem_shape = (s_q_max, s_k_max, d, ((h_r, h_k), orig_b))
    cumulative_s_q_torch_tensor = (
        torch.tensor(cu_seqlens_q, dtype=torch.int32).cuda() if varlen else None
    )
    cumulative_s_k_torch_tensor = (
        torch.tensor(cu_seqlens_k, dtype=torch.int32).cuda() if varlen else None
    )
    cumulative_s_q_cute_tensor = (
        from_dlpack(cumulative_s_q_torch_tensor).mark_layout_dynamic()
        if varlen
        else None
    )
    cumulative_s_k_cute_tensor = (
        from_dlpack(cumulative_s_k_torch_tensor).mark_layout_dynamic()
        if varlen
        else None
    )

    q_ref, q_tensor, q_torch = create_and_permute_tensor(
        (b, h_k, h_r, s_q, d), (3, 4, 2, 1, 0), element_dtype, is_dynamic_layout=True
    )
    dq_ref, dq_tensor, dq_torch = create_and_permute_tensor(
        (b, h_k, h_r, s_q, d),
        (3, 4, 2, 1, 0),
        element_dtype,
        is_dynamic_layout=True,
        zero_out=True,
    )
    k_ref, k_tensor, k_torch = create_and_permute_tensor(
        (b, h_k, 1, s_k, d), (3, 4, 2, 1, 0), element_dtype, is_dynamic_layout=True
    )
    dk_ref, dk_tensor, dk_torch = create_and_permute_tensor(
        (b, h_k, 1, s_k, d),
        (3, 4, 2, 1, 0),
        element_dtype,
        is_dynamic_layout=True,
        zero_out=True,
    )
    v_ref, v_tensor, v_torch = create_and_permute_tensor(
        (b, h_k, 1, s_k, d), (3, 4, 2, 1, 0), element_dtype, is_dynamic_layout=True
    )
    dv_ref, dv_tensor, dv_torch = create_and_permute_tensor(
        (b, h_k, 1, s_k, d),
        (3, 4, 2, 1, 0),
        element_dtype,
        is_dynamic_layout=True,
        zero_out=True,
    )
    do_ref, do_tensor, do_torch = create_and_permute_tensor(
        (b, h_k, h_r, s_q, d), (3, 4, 2, 1, 0), element_dtype, is_dynamic_layout=True
    )

    mma_tiler = (*mma_tiler_mn, d)

    fmha_bwd = HSTUAttentionBackwardSm100(
        element_dtype, acc_dtype, mma_tiler, varlen, mask_type
    )

    workspace_size = HSTUAttentionBackwardSm100._get_workspace_size(
        s_q_max, s_k_max, d, h_q, orig_b
    )
    workspace_torch = torch.zeros(workspace_size, dtype=torch.uint8).cuda()
    workspace = from_dlpack(workspace_torch, assumed_align=16).mark_layout_dynamic()

    # Initialize Stream
    current_stream = cutlass_torch.default_stream()

    print("Compiling kernel with cute.compile ...")
    start_time = time.time()
    compiled_fmha_bwd = cute.compile(
        fmha_bwd,
        problem_shape,
        q_tensor,
        k_tensor,
        v_tensor,
        dq_tensor,
        dk_tensor,
        dv_tensor,
        do_tensor,
        cumulative_s_q_cute_tensor,
        cumulative_s_k_cute_tensor,
        window_size_left if window_size_left is None else Int32(window_size_left),
        window_size_right if window_size_right is None else Int32(window_size_right),
        workspace,
        current_stream,
    )
    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time:.4f} seconds")

    for _ in range(warmup_iterations):
        compiled_fmha_bwd(
            problem_shape,
            q_tensor,
            k_tensor,
            v_tensor,
            dq_tensor,
            dk_tensor,
            dv_tensor,
            do_tensor,
            cumulative_s_q_cute_tensor,
            cumulative_s_k_cute_tensor,
            window_size_left if window_size_left is None else Int32(window_size_left),
            window_size_right if window_size_right is None else Int32(window_size_right),
            workspace,
            current_stream,
        )

    for _ in range(iterations):
        compiled_fmha_bwd(
            problem_shape,
            q_tensor,
            k_tensor,
            v_tensor,
            dq_tensor,
            dk_tensor,
            dv_tensor,
            do_tensor,
            cumulative_s_q_cute_tensor,
            cumulative_s_k_cute_tensor,
            window_size_left if window_size_left is None else Int32(window_size_left),
            window_size_right if window_size_right is None else Int32(window_size_right),
            workspace,
            current_stream,
        )

    def get_tensor_for_reference(t, s, d, h_r, h_k, b, dtype):
        t = t.reshape((s, d, h_r * h_k * b))
        t = t.permute((2, 0, 1))  # (b*h, s, d)
        t = t.to(dtype=dtype)
        return t

    def get_result_from_fmha_bwd(t, s, d, h_r, h_k, b):
        t = t.reshape((s, d, h_r * h_k * b)).permute((2, 0, 1)).to(dtype=torch.float32)
        return t

    if not skip_ref_check:
        workspace_torch.fill_(0)
        compiled_fmha_bwd(
            problem_shape,
            q_tensor,
            k_tensor,
            v_tensor,
            dq_tensor,
            dk_tensor,
            dv_tensor,
            do_tensor,
            cumulative_s_q_cute_tensor,
            cumulative_s_k_cute_tensor,
            window_size_left if window_size_left is None else Int32(window_size_left),
            window_size_right if window_size_right is None else Int32(window_size_right),
            workspace,
            current_stream,
        )
        torch.cuda.synchronize()
        print("Verifying results...")

        q_ref = q_ref.cuda().to(cutlass.torch.dtype(element_dtype))
        k_ref = k_ref.cuda().to(cutlass.torch.dtype(element_dtype))
        v_ref = v_ref.cuda().to(cutlass.torch.dtype(element_dtype))
        do_ref = do_ref.cuda().to(cutlass.torch.dtype(element_dtype))
        dv = dv_torch.to(dtype=torch.float32)
        dk = dk_torch.to(dtype=torch.float32)
        dq = dq_torch.to(dtype=torch.float32)

        dv_ref, dk_ref, dq_ref = fmha_bwd_reference(
            problem_shape,
            q_ref,
            k_ref,
            v_ref,
            do_ref,
            cumulative_s_q_torch_tensor,
            cumulative_s_k_torch_tensor,
            is_causal,
            window_size_left,
            window_size_right,
        )
        dv_pt, dk_pt, dq_pt = fmha_bwd_reference(
            problem_shape,
            q_ref,
            k_ref,
            v_ref,
            do_ref,
            cumulative_s_q_torch_tensor,
            cumulative_s_k_torch_tensor,
            is_causal,
            window_size_left,
            window_size_right,
            upcast=False,
        )

        rtol = 2
        dv_atol = 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item()
        dk_atol = 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item()
        dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item()

        print(f"Pytorch dv max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dv max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"Pytorch dk max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dk max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"Pytorch dq max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dq max diff: {(dq - dq_ref).abs().max().item()}")

        assert (dv - dv_ref).abs().max().item() <= rtol * (
            dv_pt - dv_ref
        ).abs().max().item() + dv_atol
        assert (dk - dk_ref).abs().max().item() <= rtol * (
            dk_pt - dk_ref
        ).abs().max().item() + dk_atol
        assert (dq - dq_ref).abs().max().item() <= rtol * (
            dq_pt - dq_ref
        ).abs().max().item() + dq_atol

        print("Results verified successfully!")

    def generate_tensors():
        _, q_tensor_new, _ = create_and_permute_tensor(
            (b, h_k, h_r, s_q, d),
            (3, 4, 2, 1, 0),
            element_dtype,
            is_dynamic_layout=True,
        )
        _, dq_tensor_new, _ = create_and_permute_tensor(
            (b, h_k, h_r, s_q, d),
            (3, 4, 2, 1, 0),
            element_dtype,
            is_dynamic_layout=True,
        )
        _, k_tensor_new, _ = create_and_permute_tensor(
            (b, h_k, 1, s_k, d),
            (3, 4, 2, 1, 0),
            element_dtype,
            is_dynamic_layout=True,
        )
        _, dk_tensor_new, _ = create_and_permute_tensor(
            (b, h_k, 1, s_k, d), (3, 4, 2, 1, 0), element_dtype, is_dynamic_layout=True
        )
        _, v_tensor_new, _ = create_and_permute_tensor(
            (b, h_k, 1, s_k, d), (3, 4, 2, 1, 0), element_dtype, is_dynamic_layout=True
        )
        _, dv_tensor_new, _ = create_and_permute_tensor(
            (b, h_k, 1, s_k, d), (3, 4, 2, 1, 0), element_dtype, is_dynamic_layout=True
        )
        _, do_tensor_new, _ = create_and_permute_tensor(
            (b, h_k, h_r, s_q, d),
            (3, 4, 2, 1, 0),
            element_dtype,
            is_dynamic_layout=True,
        )

        return testing.JitArguments(
            problem_shape,
            q_tensor_new,
            k_tensor_new,
            v_tensor_new,
            dq_tensor_new,
            dk_tensor_new,
            dv_tensor_new,
            do_tensor_new,
            cumulative_s_q_cute_tensor,
            cumulative_s_k_cute_tensor,
            window_size_left if window_size_left is None else Int32(window_size_left),
            window_size_right if window_size_right is None else Int32(window_size_right),
            workspace,
            current_stream,
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            q_torch.numel() * q_torch.element_size()
            + dq_torch.numel() * dq_torch.element_size()
            + k_torch.numel() * k_torch.element_size()
            + dk_torch.numel() * dk_torch.element_size()
            + v_torch.numel() * v_torch.element_size()
            + dv_torch.numel() * dv_torch.element_size()
            + do_torch.numel() * do_torch.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = testing.benchmark(
        compiled_fmha_bwd,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    return exec_time  # Return execution time in microseconds


def fmha_bwd_reference(
    problem_shape: Tuple[int, int, int, Tuple[Tuple[int, int], int]],
    Q: torch.Tensor,  # [Q, D, H_R, H_K, B]
    K: torch.Tensor,  # [K, D, 1, H_K, B]
    V: torch.Tensor,  # [K, D, 1, H_K, B]
    dO: torch.Tensor,  # [Q, D, H_R, H_K, B]
    cu_seqlens_q: Union[torch.Tensor, None],
    cu_seqlens_k: Union[torch.Tensor, None],
    is_causal: bool,
    window_size_left=None,
    window_size_right=None,
    upcast=True,
):
    s_q_max, s_k_max, d, hb = problem_shape
    h, orig_b = hb
    h_r, h_k = h

    if upcast:
        Q = Q.to(dtype=torch.float32)
        K = K.to(dtype=torch.float32)
        V = V.to(dtype=torch.float32)
        dO = dO.to(dtype=torch.float32)

    dV = torch.zeros_like(V)
    dK = torch.zeros_like(K)
    dQ = torch.zeros_like(Q)

    def dsilu(dy, x):
        dy = dy.to(torch.float32)
        x = x.to(torch.float32)
        sigmoid = F.sigmoid(x)
        return dy * sigmoid * (1 + x * (1 - sigmoid))

    for b in range(orig_b):
        q_offset = cu_seqlens_q[b] if cu_seqlens_q is not None else 0
        k_offset = cu_seqlens_k[b] if cu_seqlens_k is not None else 0
        s_q = (
            cu_seqlens_q[b + 1] - cu_seqlens_q[b]
            if cu_seqlens_q is not None
            else s_q_max
        )
        s_k = (
            cu_seqlens_k[b + 1] - cu_seqlens_k[b]
            if cu_seqlens_k is not None
            else s_k_max
        )

        for h_k_idx in range(h_k):
            b_idx = b if cu_seqlens_k is None else 0
            cur_K = K[k_offset : k_offset + s_k, :, 0, h_k_idx, b_idx]
            cur_V = V[k_offset : k_offset + s_k, :, 0, h_k_idx, b_idx]
            for h_r_idx in range(h_r):
                cur_Q = Q[q_offset : q_offset + s_q, :, h_r_idx, h_k_idx, b_idx]
                cur_dO = dO[q_offset : q_offset + s_q, :, h_r_idx, h_k_idx, b_idx]
                cur_S = torch.einsum("qd,kd->qk", cur_Q, cur_K)
                cur_P = F.silu(cur_S)
                if is_causal:
                    window_size_right = 0
                mask = None
                if window_size_left != -1 or window_size_right != -1:
                    q_coords = torch.arange(0, s_q).cuda().view(-1, 1)
                    k_coords = torch.arange(0, s_k).cuda().view(1, -1)
                    offset = s_k - s_q
                    if window_size_left == -1:
                        mask = k_coords > q_coords + offset + window_size_right
                    elif window_size_right == -1:
                        mask = k_coords < q_coords + offset - window_size_left
                    else:
                        mask = (k_coords > q_coords + offset + window_size_right) | (
                            k_coords < q_coords + offset - window_size_left
                        )
                    cur_P = cur_P.masked_fill(mask, 0.0)

                cur_PT = cur_P.transpose(1, 0).to(dtype=Q.dtype)
                cur_dV = torch.einsum("kq,qd->kd", [cur_PT, cur_dO])

                cur_dP = torch.einsum("qd,kd->qk", cur_dO, cur_V)
                if mask is not None:
                    cur_dP = cur_dP.masked_fill(mask, 0.0)
                cur_dS = dsilu(cur_dP, cur_S).to(dtype=Q.dtype)
                cur_dST = cur_dS.transpose(1, 0)
                cur_dK = torch.einsum("kq,qd->kd", cur_dST, cur_Q)
                cur_dQ = torch.einsum("qk,kd->qd", cur_dS, cur_K)

                dQ[q_offset : q_offset + s_q, :, h_r_idx, h_k_idx, b_idx] = cur_dQ
                dV[k_offset : k_offset + s_k, :, 0, h_k_idx, b_idx] = cur_dV
                dK[k_offset : k_offset + s_k, :, 0, h_k_idx, b_idx] = cur_dK

    dV = dV.to(dtype=torch.float32) / problem_shape[0]
    dK = dK.to(dtype=torch.float32) / problem_shape[0]
    dQ = dQ.to(dtype=torch.float32) / problem_shape[0]

    return dV, dK, dQ


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...] | int:
        try:
            seqlen = tuple(int(x.strip()) for x in s.split(","))
            if len(seqlen) == 1:
                return seqlen[0]
            return seqlen
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser(description="Example of bwd FMHA on Blackwell.")

    parser.add_argument(
        "--element_dtype",
        type=cutlass.dtype,
        default=Float16,
        help="Input data type",
    )

    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        default=Float32,
        help="accumulator data type",
    )

    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="MMA tile shape (M, N)",
    )

    parser.add_argument(
        "--is_causal",
        action="store_true",
        help="Whether to use causal mask",
    )

    parser.add_argument(
        "--s_q",
        type=parse_comma_separated_ints,
        default=32,
        help="max sequence length of Q",
    )

    parser.add_argument(
        "--s_k",
        type=parse_comma_separated_ints,
        default=32,
        help="max sequence length of K",
    )

    parser.add_argument(
        "--d",
        type=int,
        default=64,
        help="head dimension",
    )

    parser.add_argument(
        "--h_q",
        type=int,
        default=2,
        help="number of heads of Q",
    )

    parser.add_argument(
        "--h_k",
        type=int,
        default=2,
        help="number of heads of K",
    )

    parser.add_argument(
        "--b",
        type=int,
        default=1,
        help="batch size",
    )

    parser.add_argument(
        "--window_size",
        type=parse_comma_separated_ints,
        default=(-1, -1),
        help="Sliding window size (left, right) for attention masking.",
    )

    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=0,
        help="Number of iterations for warmup",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations after warmup",
    )

    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        help="Skip reference check",
    )

    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    args = parser.parse_args()

    if args.mma_tiler_mn != (128, 128):
        parser.error("--mma_tiler_mn only supports (128, 128)")

    run(
        args.s_q,
        args.s_k,
        args.h_q,
        args.h_k,
        args.d,
        args.b,
        args.is_causal,
        args.element_dtype,
        args.acc_dtype,
        args.mma_tiler_mn,
        args.window_size,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )

    print("PASS")
