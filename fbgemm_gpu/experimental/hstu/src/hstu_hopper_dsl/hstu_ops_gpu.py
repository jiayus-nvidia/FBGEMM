from __future__ import annotations

from typing import Optional

import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from .hstu_fwd import HSTUAttentionForwardSm90


def _mark_dynamic_tensor(tensor: torch.Tensor, leading_dim: int):
    t = from_dlpack(
        tensor.detach(), assumed_align=16, enable_tvm_ffi=True
    ).mark_layout_dynamic(leading_dim=leading_dim)
    # Dim 0 is tokens/pages. Without this, compile on S=128 freezes TMA shape
    # and S=200 last-page TMA goes out of bounds.
    stride_order = None
    if hasattr(tensor, "dim_order"):
        try:
            stride_order = tuple(tensor.dim_order())
        except Exception:
            stride_order = None
    return t.mark_compact_shape_dynamic(
        mode=0, stride_order=stride_order, divisibility=1
    )


def _mark_optional_tensor(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None
    return _mark_dynamic_tensor(tensor, tensor.ndim - 1)


def hstu_varlen_fwd_90_dsl(
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
    rab=None,
    func=None,
    paged_kv: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    page_indptrs: Optional[torch.Tensor] = None,
    last_page_lens: Optional[torch.Tensor] = None,
):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    q_tokens = q.shape[0]

    # Ampere C++ matrix: bf16/fp16 and head_dim 32/64/128/256.
    assert q.dtype in (torch.bfloat16, torch.float16), "Only support bf16 and fp16"
    assert k.dtype == q.dtype and v.dtype == q.dtype
    head_dim = q.shape[-1]
    assert head_dim in (32, 64, 128, 256), "Only support head_dim 32, 64, 128 and 256"
    assert rab is None, "rab is not supported in Hopper DSL"

    kBlockM = 128
    kBlockN = 64 if head_dim == 256 else 128
    window_size_left = (
        max_seqlen_k
        if window_size_left < 0 or window_size_left > max_seqlen_k
        else window_size_left
    )
    window_size_right = (
        max_seqlen_k
        if window_size_right < 0 or window_size_right > max_seqlen_k
        else window_size_right
    )
    is_causal = window_size_left == max_seqlen_k and window_size_right == 0
    is_local = (
        window_size_left < max_seqlen_k or window_size_right < max_seqlen_k
    ) and not is_causal
    is_context = num_contexts is not None
    assert not is_context, "HSTU-Hopper DSL does not support context mask (num_contexts)"
    is_target = num_targets is not None
    assert not (is_target and not is_causal), "Target mask is True, but causal mask is False, this is undefined behavior."
    is_arbitrary = func is not None
    func_num = func.shape[-2] if func is not None else 0
    is_paged = paged_kv is not None
    page_size = 0
    if is_paged:
        assert is_causal, "Paged KV is True, but causal mask is False, this is not supported."
        assert (not is_local) and (not is_context), "Paged KV is True, but local/context mask is True, this is not supported."
        assert not is_arbitrary, "Paged KV is True, but arbitrary mask is True, this is not supported."
        assert page_ids is not None and page_indptrs is not None, "Paged KV is True, but page metadata is missing."
        assert paged_kv.dim() == 5 and paged_kv.shape[1] == 2
        assert last_page_lens is not None, "Paged KV is True, but last_page_lens is missing."
        page_size = int(paged_kv.shape[2])
        assert page_size in (32, 64), "Only support page_size 32 and 64"
        kBlockN = page_size
        paged_k = paged_kv[:, 0].contiguous().reshape(-1, paged_kv.shape[-2], paged_kv.shape[-1])
        paged_v = paged_kv[:, 1].contiguous().reshape(-1, paged_kv.shape[-2], paged_kv.shape[-1])
    else:
        paged_k = paged_v = None

    # TMA Q loads a full kBlockM tile; pad so the last m-block stays in-bounds.
    q_pad = (kBlockM - q.shape[0] % kBlockM) % kBlockM
    if q_pad:
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, q_pad))

    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    compile_key = (
        q.dtype,
        head_dim,
        kBlockM,
        kBlockN,
        page_size,
        is_causal,
        is_local,
        is_context,
        is_target,
        target_group_size,
        is_arbitrary,
        func_num,
        is_paged,
    )

    if compile_key not in hstu_varlen_fwd_90_dsl.compile_cache:
        q_tensor, k_tensor, v_tensor, o_tensor = [
            _mark_dynamic_tensor(t, t.ndim - 1) for t in (q, k, v, out)
        ]
        cu_seqlens_q_tensor, cu_seqlens_k_tensor = [
            _mark_dynamic_tensor(t, t.ndim - 1) for t in (cu_seqlens_q, cu_seqlens_k)
        ]
        num_contexts_tensor, num_targets_tensor, func_tensor = [
            _mark_optional_tensor(t) for t in (num_contexts, num_targets, func)
        ]
        paged_k_tensor, paged_v_tensor, page_ids_tensor, page_indptrs_tensor, last_page_seqlen_tensor = [
            _mark_optional_tensor(t) for t in (paged_k, paged_v, page_ids, page_indptrs, last_page_lens)
        ]
        compile_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        hstu_fwd_sm90 = HSTUAttentionForwardSm90(
            head_dim=head_dim,
            is_causal=is_causal,
            is_local=is_local,
            is_context=is_context,
            is_target=is_target,
            target_group_size=target_group_size,
            is_arbitrary=is_arbitrary,
            func_num=func_num,
            is_paged=is_paged,
            kBlockM=kBlockM,
            kBlockN=kBlockN,
            page_size=page_size,
        )
        hstu_varlen_fwd_90_dsl.compile_cache[compile_key] = cute.compile(
            hstu_fwd_sm90,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            max_seqlen_q,
            max_seqlen_k,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            num_contexts_tensor,
            num_targets_tensor,
            alpha,
            compile_stream,
            window_size_left,
            window_size_right,
            func_tensor,
            paged_k_tensor,
            paged_v_tensor,
            page_ids_tensor,
            page_indptrs_tensor,
            last_page_seqlen_tensor,
            options="--enable-tvm-ffi",
        )

    compiled_fwd = hstu_varlen_fwd_90_dsl.compile_cache[compile_key]
    compiled_fwd(
        q,
        k,
        v,
        out,
        max_seqlen_q,
        max_seqlen_k,
        cu_seqlens_q,
        cu_seqlens_k,
        num_contexts,
        num_targets,
        alpha,
        window_size_left,
        window_size_right,
        func,
        paged_k,
        paged_v,
        page_ids,
        page_indptrs,
        last_page_lens,
    )
    
    return out[:q_tokens]


hstu_varlen_fwd_90_dsl.compile_cache = {}
