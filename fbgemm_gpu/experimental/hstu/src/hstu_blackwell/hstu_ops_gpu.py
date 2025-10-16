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
from cutlass.cute.typing import Float32

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
    num_contexts: torch.Tensor,
    num_targets: torch.Tensor,
    target_group_size: int,
    window_size_left: int,
    window_size_right: int,
    alpha: float,
    rab: torch.Tensor,
    func: torch.Tensor,
):
    # asserts

    head_dim = q.shape[2]
    kBlockM = 128 if head_dim <= 64 else 64
    kBlockN = 128 if head_dim <= 64 else 64
    out = torch.empty_like(q)

    q_tensor, k_tensor, v_tensor, o_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (q, k, v, out)
    ]
    cu_seqlens_q_tensor, cu_seqlens_k_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (cu_seqlens_q, cu_seqlens_k)
    ]
    current_stream = cutlass_torch.default_stream()
    compile_key = (head_dim, kBlockM, kBlockN)

    if compile_key not in hstu_varlen_fwd_100.compile_cache:
        hstu_fwd_sm100 = HSTUAttentionForwardSm100(
            head_dim=head_dim,
            kBlockM=kBlockM,
            kBlockN=kBlockN,
        )
        hstu_varlen_fwd_100.compile_cache[compile_key] = cute.compile(hstu_fwd_sm100, q_tensor, k_tensor, v_tensor, o_tensor, None, 1.0, current_stream, cu_seqlens_q_tensor, cu_seqlens_k_tensor)

    hstu_varlen_fwd_100.compile_cache[compile_key](
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        None,
        1.0,
        current_stream,
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
    )
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

    acc_dtype = Float32
    head_dim = q.shape[2]
    kBlockM = 128 if head_dim <= 64 else 64
    kBlockN = 128 if head_dim <= 64 else 64
    dq = torch.empty_like(q) if dq is None else dq
    dk = torch.empty_like(k) if dk is None else dk
    dv = torch.empty_like(v) if dv is None else dv

    q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (q, k, v, do, dq, dk, dv)
    ]
    current_stream = cutlass_torch.default_stream()
    compile_key = (head_dim, kBlockM, kBlockN)

    if compile_key not in hstu_varlen_bwd_100.compile_cache:
        hstu_bwd_sm100 = HSTUAttentionBackwardSm100(
            head_dim=head_dim,
            kBlockM=kBlockM,
            kBlockN=kBlockN,
        )
        workspace_size = HSTUAttentionBackwardSm100._get_workspace_size(
            max_seqlen_q, max_seqlen_k, head_dim, num_contexts, num_targets, acc_dtype
        )
        workspace_torch = torch.zeros(workspace_size, dtype=torch.uint8).cuda()
        workspace = from_dlpack(workspace_torch, assumed_align=16).mark_layout_dynamic()
        hstu_varlen_bwd_100.compile_cache[compile_key] = cute.compile(hstu_bwd_sm100, q_tensor, k_tensor, v_tensor, dq_tensor, dk_tensor, dv_tensor, do_tensor, cu_seqlens_q, cu_seqlens_k, window_size_left, window_size_right, workspace)

    hstu_varlen_bwd_100.compile_cache[compile_key](
        q_tensor,
        k_tensor,
        v_tensor,
        dq_tensor,
        dk_tensor,
        dv_tensor,
        do_tensor,
    )
    return dq, dk, dv, None

hstu_varlen_bwd_100.compile_cache = {}