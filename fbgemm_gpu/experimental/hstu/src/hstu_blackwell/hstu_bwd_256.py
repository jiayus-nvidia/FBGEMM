# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026, NVIDIA Corporation & AFFILIATES.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Head-dim 256 HSTU backward prototype for Blackwell.

The existing CuTe DSL backward keeps dQ, dK, dV, S, and dP live in a single
CTA's TMEM allocation.  That layout fits D=64/128 but cannot fit D=256.  This
module follows the decomposition used by FA4's dedicated D=256 path: dQ is
computed by one kernel and dK/dV by a second kernel.  This Triton implementation
is the compact correctness oracle, short-sequence path, and explicit fallback
for mask modes not promoted to the native 2-CTA CuTe DSL path.

Both kernels recompute score tiles and never materialize the full attention
matrix in global memory.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _valid_hstu_mask(
    rows,
    func_rows,
    cols,
    q_valid,
    k_valid,
    seqlen_k,
    num_targets,
    func,
    stride_func_n,
    stride_func_s,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    target_group_size: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    is_target: tl.constexpr,
    is_arbitrary: tl.constexpr,
    func_num: tl.constexpr,
):
    """Return the elementwise HSTU mask for one score tile."""
    valid = q_valid[:, None] & k_valid[None, :]

    if is_arbitrary:
        last = tl.load(
            func + (func_num - 1) * stride_func_n + func_rows * stride_func_s,
            mask=q_valid,
            other=0,
        )
        valid &= cols[None, :] < last[:, None]
        for interval_idx in tl.static_range(0, func_num // 2):
            interval_start = tl.load(
                func + (2 * interval_idx) * stride_func_n + func_rows * stride_func_s,
                mask=q_valid,
                other=0,
            )
            interval_end = tl.load(
                func
                + (2 * interval_idx + 1) * stride_func_n
                + func_rows * stride_func_s,
                mask=q_valid,
                other=0,
            )
            in_hole = (cols[None, :] >= interval_start[:, None]) & (
                cols[None, :] < interval_end[:, None]
            )
            valid &= ~in_hole
        return valid

    if is_causal or is_local or is_target:
        right_limit = tl.minimum(
            seqlen_k,
            rows[:, None] + 1 + window_size_right,
        )
        valid &= cols[None, :] < right_limit

    if is_local:
        left_limit = tl.maximum(0, rows[:, None] - window_size_left)
        valid &= cols[None, :] >= left_limit

    if is_target:
        seqlen_h = seqlen_k - num_targets
        target_index = (rows - seqlen_h) // target_group_size
        target_left = seqlen_h + target_index * target_group_size
        hides_previous_target_groups = (
            (rows[:, None] >= seqlen_h)
            & (cols[None, :] >= seqlen_h)
            & (cols[None, :] < target_left[:, None])
        )
        valid &= ~hides_previous_target_groups

    return valid


@triton.jit
def _hstu_bwd_dq_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    CU_Q,
    CU_K,
    NUM_TARGETS,
    FUNC,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_dot,
    stride_doh,
    stride_dod,
    stride_dqt,
    stride_dqh,
    stride_dqd,
    stride_func_n,
    stride_func_s,
    alpha,
    max_seqlen_q,
    max_seqlen_k,
    window_size_left,
    window_size_right,
    target_group_size: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    is_target: tl.constexpr,
    is_arbitrary: tl.constexpr,
    func_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_m = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    q_start = tl.load(CU_Q + batch_idx)
    q_end = tl.load(CU_Q + batch_idx + 1)
    k_start = tl.load(CU_K + batch_idx)
    k_end = tl.load(CU_K + batch_idx + 1)
    seqlen_q = q_end - q_start
    seqlen_k = k_end - k_start
    seqlen_offset = seqlen_k - seqlen_q
    target_count = tl.load(NUM_TARGETS + batch_idx) if is_target else 0

    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_valid = offs_m < seqlen_q

    q_ptrs = (
        Q
        + (q_start + offs_m)[:, None] * stride_qt
        + head_idx * stride_qh
        + offs_d[None, :] * stride_qd
    )
    do_ptrs = (
        DO
        + (q_start + offs_m)[:, None] * stride_dot
        + head_idx * stride_doh
        + offs_d[None, :] * stride_dod
    )
    q = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=q_valid[:, None], other=0.0)
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    score_rows = offs_m + seqlen_offset

    for start_n in tl.range(0, max_seqlen_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_valid = offs_n < seqlen_k
        k_ptrs = (
            K
            + (k_start + offs_n)[:, None] * stride_kt
            + head_idx * stride_kh
            + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            V
            + (k_start + offs_n)[:, None] * stride_vt
            + head_idx * stride_vh
            + offs_d[None, :] * stride_vd
        )
        k = tl.load(k_ptrs, mask=k_valid[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        valid = _valid_hstu_mask(
            score_rows,
            q_start + offs_m,
            offs_n,
            q_valid,
            k_valid,
            seqlen_k,
            target_count,
            FUNC,
            stride_func_n,
            stride_func_s,
            window_size_left,
            window_size_right,
            target_group_size,
            is_causal,
            is_local,
            is_target,
            is_arbitrary,
            func_num,
        )
        x = scores * alpha
        sigmoid = tl.sigmoid(x)
        dsilu = sigmoid * (1.0 + x * (1.0 - sigmoid))
        ds = tl.where(valid, alpha * dp * dsilu, 0.0).to(k.dtype)
        dq += tl.dot(ds, k, out_dtype=tl.float32)

    dq *= 1.0 / max_seqlen_q
    dq_ptrs = (
        DQ
        + (q_start + offs_m)[:, None] * stride_dqt
        + head_idx * stride_dqh
        + offs_d[None, :] * stride_dqd
    )
    tl.store(dq_ptrs, dq, mask=q_valid[:, None])


@triton.jit
def _hstu_bwd_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    DK,
    DV,
    CU_Q,
    CU_K,
    NUM_TARGETS,
    FUNC,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_dot,
    stride_doh,
    stride_dod,
    stride_dkt,
    stride_dkh,
    stride_dkd,
    stride_dvt,
    stride_dvh,
    stride_dvd,
    stride_func_n,
    stride_func_s,
    alpha,
    max_seqlen_q,
    max_seqlen_k,
    window_size_left,
    window_size_right,
    target_group_size: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    is_target: tl.constexpr,
    is_arbitrary: tl.constexpr,
    func_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_n = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    q_start = tl.load(CU_Q + batch_idx)
    q_end = tl.load(CU_Q + batch_idx + 1)
    k_start = tl.load(CU_K + batch_idx)
    k_end = tl.load(CU_K + batch_idx + 1)
    seqlen_q = q_end - q_start
    seqlen_k = k_end - k_start
    seqlen_offset = seqlen_k - seqlen_q
    target_count = tl.load(NUM_TARGETS + batch_idx) if is_target else 0

    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    k_valid = offs_n < seqlen_k
    k_ptrs = (
        K
        + (k_start + offs_n)[:, None] * stride_kt
        + head_idx * stride_kh
        + offs_d[None, :] * stride_kd
    )
    v_ptrs = (
        V
        + (k_start + offs_n)[:, None] * stride_vt
        + head_idx * stride_vh
        + offs_d[None, :] * stride_vd
    )
    k = tl.load(k_ptrs, mask=k_valid[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0)
    dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for start_m in tl.range(0, max_seqlen_q, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q_valid = offs_m < seqlen_q
        q_ptrs = (
            Q
            + (q_start + offs_m)[:, None] * stride_qt
            + head_idx * stride_qh
            + offs_d[None, :] * stride_qd
        )
        do_ptrs = (
            DO
            + (q_start + offs_m)[:, None] * stride_dot
            + head_idx * stride_doh
            + offs_d[None, :] * stride_dod
        )
        q = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=q_valid[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        score_rows = offs_m + seqlen_offset
        valid = _valid_hstu_mask(
            score_rows,
            q_start + offs_m,
            offs_n,
            q_valid,
            k_valid,
            seqlen_k,
            target_count,
            FUNC,
            stride_func_n,
            stride_func_s,
            window_size_left,
            window_size_right,
            target_group_size,
            is_causal,
            is_local,
            is_target,
            is_arbitrary,
            func_num,
        )
        x = scores * alpha
        sigmoid = tl.sigmoid(x)
        p = tl.where(valid, x * sigmoid, 0.0).to(do.dtype)
        dsilu = sigmoid * (1.0 + x * (1.0 - sigmoid))
        ds = tl.where(valid, alpha * dp * dsilu, 0.0).to(q.dtype)
        dk += tl.dot(tl.trans(ds), q, out_dtype=tl.float32)
        dv += tl.dot(tl.trans(p), do, out_dtype=tl.float32)

    scale = 1.0 / max_seqlen_q
    dk *= scale
    dv *= scale
    dk_ptrs = (
        DK
        + (k_start + offs_n)[:, None] * stride_dkt
        + head_idx * stride_dkh
        + offs_d[None, :] * stride_dkd
    )
    dv_ptrs = (
        DV
        + (k_start + offs_n)[:, None] * stride_dvt
        + head_idx * stride_dvh
        + offs_d[None, :] * stride_dvd
    )
    tl.store(dk_ptrs, dk, mask=k_valid[:, None])
    tl.store(dv_ptrs, dv, mask=k_valid[:, None])


def _output_or_strided_like(
    output: Optional[torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    if output is not None:
        assert output.shape == reference.shape
        assert output.dtype == reference.dtype
        assert output.device == reference.device
        return output
    return torch.empty_strided(
        reference.shape,
        reference.stride(),
        dtype=reference.dtype,
        device=reference.device,
    )


def hstu_varlen_bwd_256(
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
    num_targets: Optional[torch.Tensor],
    target_group_size: int,
    window_size_left: int,
    window_size_right: int,
    alpha: float,
    func: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    """Launch the dedicated two-kernel D=256 backward path."""
    assert q.ndim == k.ndim == v.ndim == do.ndim == 3
    assert q.is_cuda and k.is_cuda and v.is_cuda and do.is_cuda
    assert q.dtype in (torch.bfloat16, torch.float16)
    assert q.dtype == k.dtype == v.dtype == do.dtype
    assert q.device == k.device == v.device == do.device
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == do.shape[-1] == 256
    assert q.shape[1] == k.shape[1] == v.shape[1] == do.shape[1]
    assert q.shape[0] == do.shape[0]
    assert k.shape[0] == v.shape[0]
    assert max_seqlen_q > 0 and max_seqlen_k > 0
    assert cu_seqlens_q.device == q.device and cu_seqlens_k.device == q.device
    assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32
    assert cu_seqlens_q.numel() == cu_seqlens_k.numel()

    batch_size = cu_seqlens_q.numel() - 1
    num_heads = q.shape[1]
    is_causal = window_size_left == max_seqlen_k and window_size_right == 0
    is_local = (
        window_size_left < max_seqlen_k or window_size_right < max_seqlen_k
    ) and not is_causal
    is_target = num_targets is not None
    is_arbitrary = func is not None
    assert not (is_arbitrary and (is_causal or is_local or is_target))
    if num_targets is not None:
        assert target_group_size > 0
        assert num_targets.device == q.device
        assert num_targets.dtype == torch.int32
        assert num_targets.numel() == batch_size

    dq_out = _output_or_strided_like(dq, q)
    dk_out = _output_or_strided_like(dk, k)
    dv_out = _output_or_strided_like(dv, v)

    # Optional tensors are compile-time dead in variants that do not use them,
    # but Triton launch arguments still need a valid device pointer.
    num_targets_arg = num_targets if num_targets is not None else cu_seqlens_q
    func_arg = func if func is not None else cu_seqlens_q
    func_num = func.shape[-2] if func is not None else 1
    if func is not None:
        assert func.ndim == 3
        assert func.shape[0] == 1, "Blackwell arbitrary masks broadcast one func head"
        assert func.shape[-1] >= q.shape[0]
        assert func.device == q.device and func.dtype == torch.int32
        assert func_num > 0 and func_num % 2 == 1
        func_strides = func.stride()[1:]
    else:
        func_strides = (0, 0)

    # The two kernels use deliberately asymmetric tiles.  dQ benefits from a
    # wide K tile while dK/dV benefits from processing more Q rows per step.
    # These choices were swept independently on SM103; unlike the original
    # 32x32 prototype, they also stay below the 232448-byte shared-memory limit.
    dq_block_m, dq_block_n, dq_num_warps, dq_num_stages = 32, 128, 8, 2
    dkdv_block_m, dkdv_block_n = 64, 32
    dkdv_num_warps, dkdv_num_stages = 8, 3
    common_meta = dict(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        target_group_size=target_group_size,
        is_causal=is_causal,
        is_local=is_local,
        is_target=is_target,
        is_arbitrary=is_arbitrary,
        func_num=func_num,
        BLOCK_D=256,
    )
    common_args = (
        q,
        k,
        v,
        do,
    )
    input_strides = (
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *do.stride(),
    )

    dq_grid = (triton.cdiv(max_seqlen_q, dq_block_m), num_heads, batch_size)
    _hstu_bwd_dq_kernel[dq_grid](
        *common_args,
        dq_out,
        cu_seqlens_q,
        cu_seqlens_k,
        num_targets_arg,
        func_arg,
        *input_strides,
        *dq_out.stride(),
        *func_strides,
        alpha,
        BLOCK_M=dq_block_m,
        BLOCK_N=dq_block_n,
        num_warps=dq_num_warps,
        num_stages=dq_num_stages,
        **common_meta,
    )

    dkdv_grid = (triton.cdiv(max_seqlen_k, dkdv_block_n), num_heads, batch_size)
    _hstu_bwd_dkdv_kernel[dkdv_grid](
        *common_args,
        dk_out,
        dv_out,
        cu_seqlens_q,
        cu_seqlens_k,
        num_targets_arg,
        func_arg,
        *input_strides,
        *dk_out.stride(),
        *dv_out.stride(),
        *func_strides,
        alpha,
        BLOCK_M=dkdv_block_m,
        BLOCK_N=dkdv_block_n,
        num_warps=dkdv_num_warps,
        num_stages=dkdv_num_stages,
        **common_meta,
    )

    return dq_out, dk_out, dv_out, None
