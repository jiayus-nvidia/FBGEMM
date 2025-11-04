#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024, NVIDIA Corporation & AFFILIATES.

import torch
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32, Float16, BFloat16

from .hstu_fwd import HSTUAttentionForwardSm100
from .hstu_bwd import HSTUAttentionBackwardSm100

def hstu_varlen_fwd_100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    target_group_size: int,
    window_size_left: int,
    window_size_right: int,
    alpha: float,
    rab: torch.Tensor,
    func: torch.Tensor,
):
    head_dim = q.shape[2]
    head_dim_v = v.shape[2]
    assert head_dim == head_dim_v, "head_dim and head_dim_v must be equal"
    kBlockM = 128
    kBlockN = 128
    window_size_left = max_seqlen_k if window_size_left < 0 or window_size_left > max_seqlen_k else window_size_left
    window_size_right = max_seqlen_k if window_size_right < 0 or window_size_right > max_seqlen_k else window_size_right
    is_causal = window_size_left == max_seqlen_k and window_size_right == 0
    is_local = (window_size_left < max_seqlen_k or window_size_right < max_seqlen_k) and not is_causal
    is_context = num_contexts is not None
    is_target = num_targets is not None
    assert not (is_context and not is_causal), "Context mask is True, but causal mask is False, this is undefined behavior."
    assert not (is_target and not is_causal), "Target mask is True, but causal mask is False, this is undefined behavior."
    is_arbitrary = func is not None
    func_num = func.shape[-2] if func is not None else 0

    out = torch.empty_like(q)
    q_tensor, k_tensor, v_tensor, o_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (q, k, v, out)
    ]
    cu_seqlens_q_tensor, cu_seqlens_k_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (cu_seqlens_q, cu_seqlens_k)
    ]
    num_contexts_tensor, num_targets_tensor, func_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1) if t is not None else None
        for t in (num_contexts, num_targets, func)
    ]
    current_stream = cutlass_torch.default_stream()
    compile_key = (head_dim, kBlockM, kBlockN, is_causal, is_local, is_context, is_target, target_group_size, is_arbitrary, func_num)

    if compile_key not in hstu_varlen_fwd_100.compile_cache:
        hstu_fwd_sm100 = HSTUAttentionForwardSm100(
            head_dim=head_dim,
            is_causal=is_causal,
            is_local=is_local,
            is_context=is_context,
            is_target=is_target,
            target_group_size=target_group_size,
            is_arbitrary=is_arbitrary,
            func_num=func_num,
            kBlockM=kBlockM,
            kBlockN=kBlockN,
        )
        hstu_varlen_fwd_100.compile_cache[compile_key] = cute.compile(hstu_fwd_sm100, q_tensor, k_tensor, v_tensor, o_tensor, max_seqlen_q, max_seqlen_k, cu_seqlens_q_tensor, cu_seqlens_k_tensor, num_contexts_tensor, num_targets_tensor, alpha, current_stream, window_size_left, window_size_right, func_tensor)

    hstu_varlen_fwd_100.compile_cache[compile_key](q_tensor, k_tensor, v_tensor, o_tensor, max_seqlen_q, max_seqlen_k, cu_seqlens_q_tensor, cu_seqlens_k_tensor, num_contexts_tensor, num_targets_tensor, alpha, current_stream, window_size_left, window_size_right, func_tensor)

    return out, None

hstu_varlen_fwd_100.compile_cache = {}


def hstu_varlen_bwd_100(
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
    num_contexts: torch.Tensor,
    num_targets: torch.Tensor,
    target_group_size: int,
    window_size_left: int,
    window_size_right: int,
    alpha: float,
    rab: torch.Tensor,
    has_drab: bool,
    func: torch.Tensor,
    deterministic: bool,
):
    # asserts
    q_dtype = q.dtype
    assert q_dtype == torch.bfloat16 or q_dtype == torch.float16, "Only support bf16 and fp16"
    assert k.dtype == q_dtype, "k and q must have the same dtype"
    assert v.dtype == q_dtype, "v and q must have the same dtype"
    assert do.dtype == q_dtype, "do and q must have the same dtype"
    assert cu_seqlens_q.dtype == torch.int32, "cu_seqlens_q must have dtype int32"
    assert cu_seqlens_k.dtype == torch.int32, "cu_seqlens_k must have dtype int32"

    batch_size = cu_seqlens_q.shape[0] - 1
    total_q = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    total_k = k.shape[0]
    num_heads_k = k.shape[1]

    assert head_dim == 64 or head_dim == 128, "Only support head_dim 64 and 128"
    assert num_heads == num_heads_k, "Number of heads in key/value and query must be equal"

    kBlockM = 128
    kBlockN = 128
    window_size_left = max_seqlen_k if window_size_left < 0 or window_size_left > max_seqlen_k else window_size_left
    window_size_right = max_seqlen_k if window_size_right < 0 or window_size_right > max_seqlen_k else window_size_right
    is_causal = window_size_left == max_seqlen_k and window_size_right == 0
    is_local = (window_size_left < max_seqlen_k or window_size_right < max_seqlen_k) and not is_causal
    is_context = num_contexts is not None
    is_target = num_targets is not None
    assert not (is_context and not is_causal), "Context mask is True, but causal mask is False, this is undefined behavior."
    assert not (is_target and not is_causal), "Target mask is True, but causal mask is False, this is undefined behavior."
    is_arbitrary = func is not None
    func_num = func.shape[-2] if func is not None else 0

    if rab is not None:
        num_heads_rab = rab.shape[1]
        assert num_heads == num_heads_rab or num_heads_rab == 1, "Number of heads in rab must be 1 or equal to number of heads in query"

    assert not (has_drab and rab is None), "rab must be provided when has_drab is True"
    drab = torch.zeros_like(rab) if has_drab else None

    q = q.permute(1, 0, 2).contiguous().permute(1, 2, 0).unsqueeze(3).unsqueeze(2)
    k = k.permute(1, 0, 2).contiguous().permute(1, 2, 0).unsqueeze(3).unsqueeze(2)
    v = v.permute(1, 0, 2).contiguous().permute(1, 2, 0).unsqueeze(3).unsqueeze(2)
    do = do.permute(1, 0, 2).contiguous().permute(1, 2, 0).unsqueeze(3).unsqueeze(2)

    dq = torch.empty_like(q) if dq is None else dq
    dk = torch.empty_like(k) if dk is None else dk
    dv = torch.empty_like(v) if dv is None else dv

    q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1).mark_compact_shape_dynamic(mode=1, stride_order=(4, 3, 2, 0, 1), divisibility=64)
        for t in (q, k, v, do, dq, dk, dv)
    ]
    cu_seqlens_q_tensor, cu_seqlens_k_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (cu_seqlens_q, cu_seqlens_k)
    ]
    num_contexts_tensor, num_targets_tensor, func_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1) if t is not None else None
        for t in (num_contexts, num_targets, func)
    ]
    workspace_size = ((max_seqlen_q + 7) // 8 * 8) * num_heads * head_dim * batch_size * Float32.width // 8
    workspace_torch = torch.zeros(workspace_size, dtype=torch.uint8).cuda()
    workspace = from_dlpack(workspace_torch, assumed_align=16).mark_layout_dynamic()

    current_stream = cutlass_torch.default_stream()
    problem_shape = (Int32(max_seqlen_q), Int32(max_seqlen_k), Int32(head_dim), ((Int32(1), Int32(num_heads)), Int32(batch_size)))
    compile_key = (head_dim, kBlockM, kBlockN, is_causal, is_local, is_context, is_target, target_group_size, is_arbitrary, func_num)

    if compile_key not in hstu_varlen_bwd_100.compile_cache:
        hstu_bwd_sm100 = HSTUAttentionBackwardSm100(
            element_dtype=Float16 if q_dtype == torch.float16 else BFloat16,
            head_dim=head_dim,
            kBlockM=kBlockM,
            kBlockN=kBlockN,
            is_causal=is_causal,
            is_local=is_local,
            is_context=is_context,
            is_target=is_target,
            target_group_size=target_group_size,
            is_arbitrary=is_arbitrary,
            func_num=func_num,
        )
        hstu_varlen_bwd_100.compile_cache[compile_key] = cute.compile(hstu_bwd_sm100, problem_shape, q_tensor, k_tensor, v_tensor, dq_tensor, dk_tensor, dv_tensor, do_tensor, cu_seqlens_q_tensor, cu_seqlens_k_tensor, Int32(window_size_left), Int32(window_size_right), num_contexts_tensor, num_targets_tensor, func_tensor, workspace, current_stream)

    hstu_varlen_bwd_100.compile_cache[compile_key](
        problem_shape,
        q_tensor,
        k_tensor,
        v_tensor,
        dq_tensor,
        dk_tensor,
        dv_tensor,
        do_tensor,
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        Int32(window_size_left),
        Int32(window_size_right),
        num_contexts_tensor,
        num_targets_tensor,
        func_tensor,
        workspace,
        current_stream
    )

    dq = dq.squeeze(4).squeeze(2).permute(0, 2, 1)
    dk = dk.squeeze(4).squeeze(2).permute(0, 2, 1)
    dv = dv.squeeze(4).squeeze(2).permute(0, 2, 1)

    return dq, dk, dv, drab

hstu_varlen_bwd_100.compile_cache = {}