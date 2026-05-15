#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Callable, Optional, Tuple

import torch


def _round_up_to_multiple(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _int_list(t: torch.Tensor) -> list[int]:
    return [int(x) for x in t.detach().cpu().tolist()]


def _actual_varlen_lengths(
    cu_seqlens: torch.Tensor,
    seqused: Optional[torch.Tensor],
) -> list[int]:
    if seqused is not None:
        return _int_list(seqused)
    cu = _int_list(cu_seqlens)
    return [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]


def _pad_varlen_tensor_to_block(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seqused: Optional[torch.Tensor],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], bool]:
    actual_lengths = _actual_varlen_lengths(cu_seqlens, seqused)
    old_cu = _int_list(cu_seqlens)
    padded_lengths = [
        _round_up_to_multiple(length, block_size) for length in actual_lengths
    ]
    changed = any(
        (old_cu[i + 1] - old_cu[i]) != padded_lengths[i]
        for i in range(len(actual_lengths))
    )

    actual_tensor = torch.tensor(
        actual_lengths,
        dtype=torch.int32,
        device=cu_seqlens.device,
    )
    if not changed:
        return x, cu_seqlens, actual_tensor if seqused is not None else seqused, False

    new_cu_host = [0]
    for length in padded_lengths:
        new_cu_host.append(new_cu_host[-1] + length)
    new_cu = torch.tensor(new_cu_host, dtype=torch.int32, device=cu_seqlens.device)
    out = torch.zeros((new_cu_host[-1], *x.shape[1:]), dtype=x.dtype, device=x.device)
    for i, actual in enumerate(actual_lengths):
        if actual == 0:
            continue
        out[new_cu_host[i] : new_cu_host[i] + actual] = x[
            old_cu[i] : old_cu[i] + actual
        ]
    return out.contiguous(), new_cu, actual_tensor, True


def _pad_arbitrary_func_to_q_layout(
    func: Optional[torch.Tensor],
    old_cu_q: torch.Tensor,
    new_cu_q: torch.Tensor,
    actual_q: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if func is None or actual_q is None:
        return func
    old_cu = _int_list(old_cu_q)
    new_cu = _int_list(new_cu_q)
    actual = _int_list(actual_q)
    if old_cu == new_cu:
        return func

    tail = max(0, int(func.size(-1)) - old_cu[-1])
    padded = torch.zeros(
        (*func.shape[:-1], new_cu[-1] + tail),
        dtype=func.dtype,
        device=func.device,
    )
    for i, length in enumerate(actual):
        if length == 0:
            continue
        padded[..., new_cu[i] : new_cu[i] + length] = func[
            ..., old_cu[i] : old_cu[i] + length
        ]
    return padded.contiguous()


def _unpad_varlen_tensor_from_block(
    x: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    actual_lengths: torch.Tensor,
) -> torch.Tensor:
    cu = _int_list(cu_seqlens_padded)
    actual = _int_list(actual_lengths)
    pieces = [
        x[cu[i] : cu[i] + actual[i]]
        for i in range(len(actual))
        if actual[i] > 0
    ]
    if not pieces:
        return x[:0]
    return torch.cat(pieces, dim=0).contiguous()


def _pad_paged_kv_tail_to_page_layout(
    x: torch.Tensor,
    cu_seqlens_q_original: torch.Tensor,
    cu_seqlens_k_original: torch.Tensor,
    actual_q_lengths: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    last_page_lens: Optional[torch.Tensor],
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    old_offsets = _int_list(cu_seqlens_q_original)
    actual_q = (
        _int_list(actual_q_lengths)
        if actual_q_lengths is not None
        else [old_offsets[i + 1] - old_offsets[i] for i in range(len(old_offsets) - 1)]
    )
    actual_k = _actual_varlen_lengths(cu_seqlens_k_original, seqused_k)
    targets = (
        _int_list(num_targets)
        if num_targets is not None
        else [0 for _ in actual_k]
    )
    last_pages = (
        _int_list(last_page_lens)
        if last_page_lens is not None
        else [page_size for _ in actual_k]
    )

    new_offsets = [0]
    for b, actual_k_len in enumerate(actual_k):
        target_len = targets[b]
        actual_q_len = actual_q[b]
        if target_len > actual_q_len or target_len > actual_k_len:
            raise ValueError(
                f"num_targets[{b}]={target_len} exceeds q/k lengths "
                f"{actual_q_len}/{actual_k_len}"
            )
        cache_len = actual_k_len - target_len
        if target_len > 0:
            last_page = last_pages[b]
            if not (1 <= last_page <= page_size):
                raise ValueError(f"invalid last_page_lens[{b}]={last_page}")
            if (cache_len + (page_size - last_page)) % page_size != 0:
                raise ValueError(
                    "paged target tail must start on a page boundary for Blackwell RTX FP8 TMA"
                )
            physical_len = cache_len + (page_size - last_page) + target_len
        else:
            physical_len = max(actual_q_len, actual_k_len)
        new_offsets.append(new_offsets[-1] + _round_up_to_multiple(physical_len, page_size))

    new_cu = torch.tensor(
        new_offsets,
        dtype=torch.int32,
        device=cu_seqlens_q_original.device,
    )
    actual_k_tensor = torch.tensor(
        actual_k,
        dtype=torch.int32,
        device=cu_seqlens_q_original.device,
    )
    out = x.new_zeros((new_offsets[-1], *x.shape[1:]))

    for b, actual_q_len in enumerate(actual_q):
        old_start = old_offsets[b]
        new_start = new_offsets[b]
        target_len = targets[b]
        new_history_len = actual_q_len - target_len
        if new_history_len > 0:
            out[new_start : new_start + new_history_len] = x[
                old_start : old_start + new_history_len
            ]

        if target_len > 0:
            last_page = last_pages[b]
            cache_len = actual_k[b] - target_len
            target_start = new_start + cache_len + (page_size - last_page)
            target_end = target_start + target_len
            if target_end > new_offsets[b + 1]:
                raise ValueError(
                    "paged target tail does not fit in padded physical K/V layout"
                )
            src_start = old_start + new_history_len
            out[target_start:target_end] = x[src_start : src_start + target_len]

    return out.contiguous(), new_cu, actual_k_tensor


def _pad_rab_to_paged_kv_layout(
    rab: Optional[torch.Tensor],
    cu_seqlens_k_original: torch.Tensor,
    seqused_k_actual: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    last_page_lens: Optional[torch.Tensor],
    page_size: int,
) -> Optional[torch.Tensor]:
    if rab is None or num_targets is None or last_page_lens is None:
        return rab

    actual_k = _actual_varlen_lengths(cu_seqlens_k_original, seqused_k_actual)
    targets = _int_list(num_targets)
    last_pages = _int_list(last_page_lens)
    max_logical = int(rab.size(-1))
    gaps: list[int] = []
    max_physical = max_logical
    for b, actual_k_len in enumerate(actual_k):
        target_len = targets[b]
        gap = 0
        if target_len > 0:
            last_page = last_pages[b]
            if not (1 <= last_page <= page_size):
                raise ValueError(f"invalid last_page_lens[{b}]={last_page}")
            gap = page_size - last_page
        gaps.append(gap)
        max_physical = max(max_physical, actual_k_len + gap)

    if max_physical == max_logical and all(gap == 0 for gap in gaps):
        return rab

    out = rab.new_zeros((rab.size(0), rab.size(1), max_physical, max_physical))
    for b, actual_k_len in enumerate(actual_k):
        target_len = targets[b]
        gap = gaps[b]
        if target_len <= 0 or gap == 0:
            out[b, :, :max_logical, :max_logical] = rab[b]
            continue
        history_len = actual_k_len - target_len
        if history_len > 0:
            out[b, :, :max_logical, :history_len] = rab[b, :, :, :history_len]
        src_end = min(actual_k_len, max_logical)
        if src_end > history_len:
            dst_start = history_len + gap
            dst_end = dst_start + (src_end - history_len)
            out[b, :, :max_logical, dst_start:dst_end] = rab[
                b, :, :, history_len:src_end
            ]
    return out.contiguous()


def _pack_descale_to_e8m0x4_int32(descale: torch.Tensor) -> torch.Tensor:
    s = torch.clamp(descale.to(torch.float32), min=1e-10)
    exp_biased = torch.clamp(torch.ceil(torch.log2(s)) + 127.0, 0.0, 255.0).to(
        torch.uint8
    )
    if exp_biased.dim() == 3:
        if exp_biased.size(-1) > 4:
            raise ValueError(
                f"e8m0x4 packing supports at most 4 D chunks, got {exp_biased.size(-1)}"
            )
        padded = torch.zeros(
            (*exp_biased.shape[:-1], 4),
            dtype=torch.uint8,
            device=exp_biased.device,
        )
        padded[..., : exp_biased.size(-1)] = exp_biased
        word = padded.to(torch.int32)
        return (
            word[..., 0]
            | (word[..., 1] << 8)
            | (word[..., 2] << 16)
            | (word[..., 3] << 24)
        ).contiguous()
    word = exp_biased.to(torch.int32)
    return (word | (word << 8) | (word << 16) | (word << 24)).contiguous()


def quantize_paged_kv_cache_for_block_scale(
    kv_cache: torch.Tensor,
    block_size: int,
    fp8_type=torch.float8_e4m3fn,
    quantize_for_block_scale_fn: Optional[Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if kv_cache.dim() != 5:
        raise ValueError("kv_cache must have shape [pages, 2, page_size, heads, dim]")
    if kv_cache.size(1) != 2:
        raise ValueError("kv_cache second dimension must be 2")
    if kv_cache.size(2) != block_size:
        raise ValueError(f"kv_cache page_size must equal block_size={block_size}")
    if quantize_for_block_scale_fn is None:
        from hstu.cuda_hstu_attention import quantize_for_block_scale
        quantize_for_block_scale_fn = quantize_for_block_scale
    pages, _, page_size, heads, dim = kv_cache.shape

    with torch.no_grad():
        k_cache = kv_cache[:, 0].contiguous()
        v_cache = kv_cache[:, 1].contiguous()
        flat_offsets = torch.tensor(
            [0, pages * page_size],
            dtype=torch.int32,
            device=kv_cache.device,
        )

        k_fp8_flat, k_descale, _ = quantize_for_block_scale_fn(
            k_cache.view(pages * page_size, heads, dim),
            flat_offsets,
            fp8_type=fp8_type,
            scale_mode="token_dchunk",
            d_chunk_size=128,
            round_to_e8m0=True,
        )
        v_fp8_flat, v_descale, _ = quantize_for_block_scale_fn(
            v_cache.view(pages * page_size, heads, dim),
            flat_offsets,
            block_size=block_size,
            fp8_type=fp8_type,
            scale_mode="seq_block",
            round_to_e8m0=True,
        )

        kv_cache_fp8 = torch.empty_like(kv_cache, dtype=fp8_type)
        kv_cache_fp8[:, 0] = k_fp8_flat.view(pages, page_size, heads, dim)
        kv_cache_fp8[:, 1] = v_fp8_flat.view(pages, page_size, heads, dim)

    return (
        kv_cache_fp8.contiguous(),
        _pack_descale_to_e8m0x4_int32(k_descale),
        _pack_descale_to_e8m0x4_int32(v_descale).repeat_interleave(block_size, dim=1),
    )


def pack_descale_to_e8m0x4_int32(descale: torch.Tensor) -> torch.Tensor:
    return _pack_descale_to_e8m0x4_int32(descale)


def _blackwell_rtx_block_sizes(is_paged_kv: bool, kv_cache: Optional[torch.Tensor]) -> tuple[int, int]:
    if is_paged_kv:
        assert kv_cache is not None
        return 128, int(kv_cache.shape[2])
    return 128, 64


def run_blackwell_rtx_fp8_varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    scaling_seqlen: int,
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    target_group_size: int,
    window_size: Tuple[int, int],
    alpha: float,
    rab: Optional[torch.Tensor],
    func: Optional[torch.Tensor],
    kv_cache: Optional[torch.Tensor],
    page_offsets: Optional[torch.Tensor],
    page_ids: Optional[torch.Tensor],
    last_page_lens: Optional[torch.Tensor],
    quant_mode: Optional[int],
    quantize_for_block_scale_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if quant_mode != 2:
        raise ValueError("Blackwell RTX path only supports FP8 quant_mode=2 forward")

    is_paged_kv = (
        kv_cache is not None
        and page_offsets is not None
        and page_ids is not None
        and last_page_lens is not None
    )
    bm, bn = _blackwell_rtx_block_sizes(is_paged_kv, kv_cache)
    if is_paged_kv and (q.shape[-1] not in (32, 64, 128, 256) or bn != 64):
        raise ValueError(
            "Blackwell RTX FP8 paged KV requires headDim 32/64/128/256 and page_size=64"
        )

    blackwell_rtx_fp8_unpad_q: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    original_cu_seqlens_q = cu_seqlens_q
    q, cu_seqlens_q, seqused_q_actual, q_was_padded = _pad_varlen_tensor_to_block(
        q,
        cu_seqlens_q,
        seqused_q,
        bm,
    )
    if q_was_padded:
        func = _pad_arbitrary_func_to_q_layout(
            func,
            original_cu_seqlens_q,
            cu_seqlens_q,
            seqused_q_actual,
        )
        seqused_q = seqused_q_actual
        blackwell_rtx_fp8_unpad_q = (cu_seqlens_q, seqused_q_actual)

    if is_paged_kv:
        original_cu_seqlens_k = cu_seqlens_k
        k, cu_seqlens_k, seqused_k_actual = _pad_paged_kv_tail_to_page_layout(
            k,
            original_cu_seqlens_q,
            original_cu_seqlens_k,
            seqused_q_actual,
            seqused_k,
            num_targets,
            last_page_lens,
            bn,
        )
        v, _, _ = _pad_paged_kv_tail_to_page_layout(
            v,
            original_cu_seqlens_q,
            original_cu_seqlens_k,
            seqused_q_actual,
            seqused_k,
            num_targets,
            last_page_lens,
            bn,
        )
        rab = _pad_rab_to_paged_kv_layout(
            rab,
            original_cu_seqlens_k,
            seqused_k_actual,
            num_targets,
            last_page_lens,
            bn,
        )
        seqused_k = seqused_k_actual
        kv_quant_offsets = cu_seqlens_k
    else:
        original_cu_seqlens_k = cu_seqlens_k
        k, cu_seqlens_k, seqused_k_actual, k_was_padded = _pad_varlen_tensor_to_block(
            k,
            cu_seqlens_k,
            seqused_k,
            bn,
        )
        v, _, _, _ = _pad_varlen_tensor_to_block(
            v,
            original_cu_seqlens_k,
            seqused_k,
            bn,
        )
        if k_was_padded:
            seqused_k = seqused_k_actual
        kv_quant_offsets = cu_seqlens_k

    q, q_descale, cu_seqlens_q_block_descale = quantize_for_block_scale_fn(
        q,
        cu_seqlens_q,
        scale_mode="token_dchunk",
        d_chunk_size=128,
        round_to_e8m0=True,
    )
    k, k_descale, cu_seqlens_kv_block_descale = quantize_for_block_scale_fn(
        k,
        kv_quant_offsets,
        scale_mode="token_dchunk",
        d_chunk_size=128,
        round_to_e8m0=True,
    )
    v, v_descale, cu_seqlens_v_block_descale = quantize_for_block_scale_fn(
        v,
        kv_quant_offsets,
        block_size=bn,
        scale_mode="seq_block",
        round_to_e8m0=True,
    )
    sf_q_packed = _pack_descale_to_e8m0x4_int32(q_descale)
    sf_k_packed = _pack_descale_to_e8m0x4_int32(k_descale)
    sf_v_packed = _pack_descale_to_e8m0x4_int32(v_descale).repeat_interleave(
        bn,
        dim=1,
    )
    if is_paged_kv:
        assert kv_cache is not None
        kv_cache, sf_k_cache_packed, sf_v_cache_packed = quantize_paged_kv_cache_for_block_scale(
            kv_cache,
            block_size=bn,
            quantize_for_block_scale_fn=quantize_for_block_scale_fn,
        )
        sf_k_packed = torch.cat([sf_k_cache_packed, sf_k_packed], dim=1).contiguous()
        sf_v_packed = torch.cat([sf_v_cache_packed, sf_v_packed], dim=1).contiguous()

    rab_kernel = rab.to(torch.bfloat16) if rab is not None and rab.dtype == torch.float16 else rab
    out, rab_padded = torch.ops.fbgemm.hstu_varlen_fwd_120(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        scaling_seqlen,
        num_contexts,
        num_targets,
        target_group_size,
        window_size[0],
        window_size[1],
        alpha,
        rab_kernel,
        func,
        quant_mode if quant_mode is not None else -1,
        q_descale,
        k_descale,
        v_descale,
        sf_q_packed,
        sf_k_packed,
        sf_v_packed,
        cu_seqlens_q_block_descale,
        cu_seqlens_kv_block_descale,
        cu_seqlens_v_block_descale,
        kv_cache,
        page_offsets,
        page_ids,
        last_page_lens,
    )
    if blackwell_rtx_fp8_unpad_q is not None:
        out = _unpad_varlen_tensor_from_block(
            out,
            blackwell_rtx_fp8_unpad_q[0],
            blackwell_rtx_fp8_unpad_q[1],
        )
    return out, rab_padded
