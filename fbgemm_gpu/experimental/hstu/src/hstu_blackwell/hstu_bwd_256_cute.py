# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026, NVIDIA Corporation & AFFILIATES.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dedicated two-kernel CuTe DSL backward for HSTU head dimension 256.

The kernel decomposition and Blackwell pipelines are derived from the
FlashAttention-4 SM100 head-dim-256 backward path.  The score transform is HSTU
SiLU rather than softmax; see the DQ and DK/DV kernel modules for the adapted
math.
"""

from __future__ import annotations

from typing import Optional

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32

from .hstu_bwd_256_cute_dkdv import (
    BlackwellFusedMultiHeadAttentionBackwardDKDVKernel,
)
from .hstu_bwd_256_cute_dq import (
    BlackwellFusedMultiHeadAttentionBackwardDQKernel,
)


def _as_bshkrd_tensor(
    tensor: cute.Tensor,
    h_k: Int32,
    h_r: Int32,
    varlen: bool,
) -> cute.Tensor:
    """Normalize (B,S,H,D)/(S,H,D) to a (B,S,H_k,H_r,D) view."""
    if cutlass.const_expr(cute.rank(tensor.layout) == 5):
        return tensor
    if cutlass.const_expr(cute.rank(tensor.layout) == 4):
        return cute.make_tensor(
            tensor.iterator,
            cute.make_layout(
                (tensor.shape[0], tensor.shape[1], h_k, h_r, tensor.shape[3]),
                stride=(
                    tensor.stride[0],
                    tensor.stride[1],
                    tensor.stride[2] * h_r,
                    tensor.stride[2],
                    tensor.stride[3],
                ),
            ),
        )
    assert cutlass.const_expr(cute.rank(tensor.layout) == 3)
    assert cutlass.const_expr(varlen)
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(
            (1, tensor.shape[0], h_k, h_r, tensor.shape[2]),
            stride=(
                0,
                tensor.stride[0],
                tensor.stride[1] * h_r,
                tensor.stride[1],
                tensor.stride[2],
            ),
        ),
    )


def _as_dummy_stats_tensor(
    tensor: cute.Tensor,
    sequence_extent: Int32,
    h_k: Int32,
    h_r: Int32,
    b: Int32,
) -> cute.Tensor:
    """Broadcast one unused scalar over the inherited FA4 stats layout."""
    assert cutlass.const_expr(cute.rank(tensor.layout) == 1)
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(
            (sequence_extent, ((h_r, h_k), b)),
            stride=(0, ((0, 0), 0)),
        ),
    )


class HSTUAttentionBackwardSm100D256:
    """Launch the dedicated DQ kernel followed by the dedicated DK/DV kernel."""

    def __init__(
        self,
        *,
        is_causal: bool,
        is_local: bool,
        is_target: bool,
        target_group_size: int,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
    ):
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_target = is_target
        self.target_group_size = target_group_size
        self.dq_kernel = BlackwellFusedMultiHeadAttentionBackwardDQKernel(
            cutlass.Float32,
            (128, 128, 256),
            is_causal,
            window_size_left,
            window_size_right,
            False,
            False,
            is_target=is_target,
            target_group_size=target_group_size,
            use_clc_scheduler=False,
        )
        self.dkdv_kernel = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel(
            cutlass.Float32,
            (128, 64, 256),
            is_causal,
            window_size_left,
            window_size_right,
            is_target=is_target,
            target_group_size=target_group_size,
            use_clc_scheduler=False,
        )

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        do: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        dummy_stats: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        num_targets: cute.Tensor | None,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        alpha: cutlass.Float32,
        normalization_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        varlen = True
        q_rank = cute.rank(q.layout)
        k_rank = cute.rank(k.layout)
        if cutlass.const_expr(q_rank == 5):
            h_q = q.shape[2] * q.shape[3]
        elif cutlass.const_expr(q_rank == 4):
            h_q = q.shape[2]
        else:
            h_q = q.shape[1]
        q_sequence_extent = q.shape[0] if q_rank == 3 else q.shape[1]
        if cutlass.const_expr(k_rank == 5):
            h_k = k.shape[2]
        elif cutlass.const_expr(k_rank == 4):
            h_k = k.shape[2]
        else:
            h_k = k.shape[1]
        h_r = h_q // h_k
        b = cu_seqlens_q.shape[0] - 1

        q = _as_bshkrd_tensor(q, h_k, h_r, varlen)
        k = _as_bshkrd_tensor(k, h_k, 1, varlen)
        v = _as_bshkrd_tensor(v, h_k, 1, varlen)
        do = _as_bshkrd_tensor(do, h_k, h_r, varlen)
        dq = _as_bshkrd_tensor(dq, h_k, h_r, varlen)
        dk = _as_bshkrd_tensor(dk, h_k, 1, varlen)
        dv = _as_bshkrd_tensor(dv, h_k, 1, varlen)
        stats = _as_dummy_stats_tensor(dummy_stats, q_sequence_extent, h_k, h_r, b)

        self.dq_kernel(
            q,
            k,
            v,
            dq,
            do,
            stats,
            stats,
            cu_seqlens_q,
            cu_seqlens_k,
            num_targets,
            max_seqlen_q,
            max_seqlen_k,
            alpha,
            alpha * normalization_scale,
            stream,
        )
        self.dkdv_kernel(
            q,
            k,
            v,
            dk,
            dv,
            do,
            stats,
            stats,
            cu_seqlens_q,
            cu_seqlens_k,
            num_targets,
            max_seqlen_q,
            max_seqlen_k,
            alpha,
            normalization_scale,
            stream,
        )


def _dynamic_tensor(tensor: torch.Tensor, leading_dim: int) -> cute.Tensor:
    return from_dlpack(
        tensor.detach(), assumed_align=16, enable_tvm_ffi=True
    ).mark_layout_dynamic(leading_dim=leading_dim)


def _dynamic_optional_tensor(
    tensor: Optional[torch.Tensor],
) -> Optional[cute.Tensor]:
    return None if tensor is None else _dynamic_tensor(tensor, tensor.ndim - 1)


def _copy_to_optional_output(
    work: torch.Tensor,
    output: Optional[torch.Tensor],
) -> torch.Tensor:
    if output is None:
        return work
    assert output.shape == work.shape
    assert output.dtype == work.dtype
    assert output.device == work.device
    if output.data_ptr() == work.data_ptr():
        return output
    output.copy_(work)
    return output


def _native_output_buffer(
    output: Optional[torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    if (
        output is not None
        and output.shape == reference.shape
        and output.dtype == reference.dtype
        and output.device == reference.device
        and output.is_contiguous()
    ):
        return output
    return torch.empty_like(reference)


def hstu_varlen_bwd_256_cute(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    window_size_left: int,
    window_size_right: int,
    alpha: float,
    *,
    num_targets: Optional[torch.Tensor] = None,
    target_group_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    """Compile and run the native CuTe DSL D=256 path.

    Native predicates cover unmasked, causal, local, and target attention.
    Arbitrary interval masks remain on the established Triton path.
    """
    assert q.ndim == k.ndim == v.ndim == do.ndim == 3
    assert q.is_cuda and k.is_cuda and v.is_cuda and do.is_cuda
    assert q.dtype in (torch.bfloat16, torch.float16)
    assert q.dtype == k.dtype == v.dtype == do.dtype
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == do.shape[-1] == 256
    assert q.shape[1] == k.shape[1] == v.shape[1] == do.shape[1]
    assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32
    assert cu_seqlens_q.numel() == cu_seqlens_k.numel()
    assert max_seqlen_q > 0 and max_seqlen_k > 0

    is_causal = window_size_left == max_seqlen_k and window_size_right == 0
    is_unmasked = window_size_left == max_seqlen_k and window_size_right == max_seqlen_k
    is_local = not (is_causal or is_unmasked)
    is_target = num_targets is not None
    assert not is_target or is_causal
    if is_target:
        assert target_group_size > 0
        assert num_targets.dtype == torch.int32
        assert num_targets.device == q.device
        assert num_targets.numel() == cu_seqlens_q.numel() - 1

    # The dedicated kernels require fully compact TMA operands.  Copies are
    # explicit here during bring-up; layout-preserving fast paths come after the
    # native kernels clear correctness and performance gates.
    q_work = q.contiguous()
    k_work = k.contiguous()
    v_work = v.contiguous()
    do_work = do.contiguous()
    dq_work = _native_output_buffer(dq, q_work)
    dk_work = _native_output_buffer(dk, k_work)
    dv_work = _native_output_buffer(dv, v_work)
    # Values are deliberately unused by the HSTU score transform.  The buffers
    # remain only because the inherited FA4 pipelines still issue their loads.
    dummy_stats = torch.empty((1,), dtype=torch.float32, device=q.device)

    compile_key = (
        q.dtype,
        q.shape[1],
        is_causal,
        is_local,
        is_target,
        target_group_size if is_target else 1,
        window_size_left if is_local else None,
        window_size_right if is_local else None,
    )
    normalization_scale = 1.0 / max_seqlen_q
    if compile_key not in hstu_varlen_bwd_256_cute.compile_cache:
        q_tensor, k_tensor, v_tensor, do_tensor = [
            _dynamic_tensor(tensor, tensor.ndim - 1)
            for tensor in (q_work, k_work, v_work, do_work)
        ]
        dq_tensor, dk_tensor, dv_tensor = [
            _dynamic_tensor(tensor, tensor.ndim - 1)
            for tensor in (dq_work, dk_work, dv_work)
        ]
        stats_tensor = _dynamic_tensor(dummy_stats, 0)
        cu_q_tensor, cu_k_tensor = [
            _dynamic_tensor(tensor, 0) for tensor in (cu_seqlens_q, cu_seqlens_k)
        ]
        num_targets_tensor = _dynamic_optional_tensor(num_targets)
        compile_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = HSTUAttentionBackwardSm100D256(
            is_causal=is_causal,
            is_local=is_local,
            is_target=is_target,
            target_group_size=target_group_size,
            window_size_left=window_size_left if is_local else None,
            window_size_right=window_size_right if is_local else None,
        )
        hstu_varlen_bwd_256_cute.compile_cache[compile_key] = cute.compile(
            kernel,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            dq_tensor,
            dk_tensor,
            dv_tensor,
            stats_tensor,
            cu_q_tensor,
            cu_k_tensor,
            num_targets_tensor,
            Int32(max_seqlen_q),
            Int32(max_seqlen_k),
            alpha,
            normalization_scale,
            compile_stream,
            options="--enable-tvm-ffi",
        )

    compiled = hstu_varlen_bwd_256_cute.compile_cache[compile_key]
    compiled(
        q_work,
        k_work,
        v_work,
        do_work,
        dq_work,
        dk_work,
        dv_work,
        dummy_stats,
        cu_seqlens_q,
        cu_seqlens_k,
        num_targets,
        Int32(max_seqlen_q),
        Int32(max_seqlen_k),
        alpha,
        normalization_scale,
    )
    return (
        _copy_to_optional_output(dq_work, dq),
        _copy_to_optional_output(dk_work, dk),
        _copy_to_optional_output(dv_work, dv),
        None,
    )


hstu_varlen_bwd_256_cute.compile_cache = {}
