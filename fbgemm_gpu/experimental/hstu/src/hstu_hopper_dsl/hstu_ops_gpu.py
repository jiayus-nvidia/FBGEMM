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


def _choose_kv_split(
    max_seqlen_k: int,
    kBlockN: int,
    q_tokens: int,
    kBlockM: int,
    n_heads: int,
    n_batch: int,
) -> int:
    """Split KV tiles across blockIdx.y so short-Lq/long-hist fills the GPU.

    HSTU O is additive in N, so partials just sum. Official tests with large
    batch already launch enough CTAs and stay at n_split=1.
    """
    n_kv = max(1, (int(max_seqlen_k) + kBlockN - 1) // kBlockN)
    n_m = max(1, (int(q_tokens) + kBlockM - 1) // kBlockM)
    base = max(1, n_m * int(n_heads) * int(n_batch))
    try:
        sms = int(torch.cuda.get_device_properties(0).multi_processor_count)
    except Exception:
        return 1
    # Short KV loops lose to extra Q loads + partial-O reduction.
    # Threshold is in tokens so a larger kBlockN is not punished (old
    # `n_kv < 32` with kBlockN=32 meant <1024 tokens).
    if base >= sms or n_kv * int(kBlockN) < 1024:
        return 1
    n_split = min(n_kv, sms // base)
    while n_split > 1 and (n_kv + n_split - 1) // n_split < 4:
        n_split -= 1
    return max(1, int(n_split))


def _paged_smem_ok(head_dim: int, k_block_m: int, k_block_n: int, stages: int = 2) -> bool:
    """H100 227 KB smem. Occupancy stays 1; this is just a fit check."""
    elem = 2
    overlap = head_dim == 256
    q = k_block_m * head_dim * elem
    kv = 2 * k_block_n * head_dim * stages * elem
    o = 0 if overlap else q
    return q + kv + o + 4096 < 227 * 1024


def _choose_paged_kblock_n(
    head_dim: int, page_size: int, override: Optional[int] = None
) -> int:
    """MMA N tile. Independent of cache page_size; must be a multiple of it."""
    if override is not None:
        k = int(override)
        if k % page_size != 0:
            raise ValueError(f"kBlockN={k} must be a multiple of page_size={page_size}")
        if not _paged_smem_ok(head_dim, 128, k):
            raise ValueError(
                f"kBlockN={k} head_dim={head_dim} does not fit in Hopper smem"
            )
        return k
    if head_dim >= 128:
        for k in (128, 64, page_size):
            if k % page_size == 0 and _paged_smem_ok(head_dim, 128, k):
                return k
    return page_size


def _paged_kv_planes(
    paged_kv: torch.Tensor, extra_zero_page: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """K/V as (pages * page_size, heads, dim) for TMA. Reuse plane buffers.

    extra_zero_page: one trailing zero page so OOB slots of a kBlockN>page_size
    tile still complete the TMA transaction barrier.
    """
    n_pages, _, page_size, n_heads, head_dim = paged_kv.shape
    k_view = paged_kv.select(1, 0)
    v_view = paged_kv.select(1, 1)
    packed = n_pages + int(extra_zero_page)
    if (
        not extra_zero_page
        and k_view.is_contiguous()
        and v_view.is_contiguous()
    ):
        return (
            k_view.reshape(-1, n_heads, head_dim),
            v_view.reshape(-1, n_heads, head_dim),
        )
    plane = (packed, page_size, n_heads, head_dim)
    cache = hstu_varlen_fwd_90_dsl._paged_plane_cache
    buf = cache.get(plane)
    if (
        buf is None
        or buf[0].shape != plane
        or buf[0].dtype != paged_kv.dtype
        or buf[0].device != paged_kv.device
    ):
        buf = (
            torch.empty(plane, dtype=paged_kv.dtype, device=paged_kv.device),
            torch.empty(plane, dtype=paged_kv.dtype, device=paged_kv.device),
        )
        cache[plane] = buf
    buf[0][:n_pages].copy_(k_view)
    buf[1][:n_pages].copy_(v_view)
    if extra_zero_page:
        buf[0][n_pages:].zero_()
        buf[1][n_pages:].zero_()
    return buf[0].view(-1, n_heads, head_dim), buf[1].view(-1, n_heads, head_dim)


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
    k_block_n: Optional[int] = None,
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
        kBlockN = _choose_paged_kblock_n(head_dim, page_size, k_block_n)
        paged_k, paged_v = _paged_kv_planes(
            paged_kv, extra_zero_page=kBlockN > page_size
        )
    else:
        paged_k = paged_v = None

    # TMA Q loads a full kBlockM tile; pad so the last m-block stays in-bounds.
    q_pad = (kBlockM - q.shape[0] % kBlockM) % kBlockM
    if q_pad:
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, q_pad))

    n_batch = int(cu_seqlens_q.numel() - 1)
    kv_split = _choose_kv_split(
        max_seqlen_k, kBlockN, q.shape[0], kBlockM, q.shape[1], n_batch
    )
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if kv_split > 1:
        o_shape = (kv_split, q.shape[0], q.shape[1], q.shape[2])
        cache = hstu_varlen_fwd_90_dsl._o_work_cache
        o_work = cache.get(o_shape)
        if (
            o_work is None
            or o_work.shape != o_shape
            or o_work.dtype != q.dtype
            or o_work.device != q.device
        ):
            o_work = torch.empty(o_shape, dtype=q.dtype, device=q.device)
            cache[o_shape] = o_work
        o_launch = o_work.view(kv_split * q.shape[0], q.shape[1], q.shape[2])
    else:
        o_work = None
        o_launch = out
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
        kv_split,
    )

    if compile_key not in hstu_varlen_fwd_90_dsl.compile_cache:
        q_tensor, k_tensor, v_tensor, o_tensor = [
            _mark_dynamic_tensor(t, t.ndim - 1) for t in (q, k, v, o_launch)
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
            kv_split,
            options="--enable-tvm-ffi",
        )

    compiled_fwd = hstu_varlen_fwd_90_dsl.compile_cache[compile_key]
    compiled_fwd(
        q,
        k,
        v,
        o_launch,
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
        kv_split,
    )
    if kv_split > 1:
        out.copy_(o_work.float().sum(dim=0).to(out.dtype))
    return out[:q_tokens]


hstu_varlen_fwd_90_dsl.compile_cache = {}
hstu_varlen_fwd_90_dsl._o_work_cache = {}
hstu_varlen_fwd_90_dsl._paged_plane_cache = {}
