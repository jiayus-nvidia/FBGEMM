# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.

import pytest
import torch
import torch.nn.functional as F

try:
    from hstu.hstu_blackwell import mxfp8_attention
    from hstu.hstu_blackwell.mxfp8_attention import (
        hstu_varlen_bwd_mxfp8_100,
        hstu_varlen_fwd_mxfp8_100,
    )
except (ImportError, OSError):
    from hstu_blackwell import mxfp8_attention
    from hstu_blackwell.mxfp8_attention import (
        hstu_varlen_bwd_mxfp8_100,
        hstu_varlen_fwd_mxfp8_100,
    )


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="SM100 MXFP8 requires a Blackwell datacenter GPU",
)

_E4M3_MAX = 448.0
_MXFP8_BLOCK_SIZE = 32
_TILE_SIZE = 128


def _offsets(lengths):
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return torch.tensor(result, dtype=torch.int32, device="cuda")


def _reference(
    q,
    k,
    v,
    q_lengths,
    k_lengths,
    max_seqlen_q,
    max_seqlen_k,
    alpha,
    window,
):
    outputs = []
    q_offset = 0
    k_offset = 0
    for query_length, key_length in zip(q_lengths, k_lengths):
        query = q[q_offset : q_offset + query_length]
        key = k[k_offset : k_offset + key_length]
        value = v[k_offset : k_offset + key_length]
        scores = torch.einsum("qhd,khd->hqk", query, key)

        query_idx = torch.arange(query_length, device=q.device)[:, None]
        key_idx = torch.arange(key_length, device=q.device)[None, :]
        aligned_query_idx = query_idx + key_length - query_length
        left = max_seqlen_k if window[0] < 0 else window[0]
        right = max_seqlen_k if window[1] < 0 else window[1]
        valid = (key_idx >= aligned_query_idx - left) & (
            key_idx < aligned_query_idx + 1 + right
        )

        probabilities = F.silu(scores * alpha) / max_seqlen_q
        probabilities = probabilities * valid.unsqueeze(0)
        outputs.append(torch.einsum("hqk,khd->qhd", probabilities, value))
        q_offset += query_length
        k_offset += key_length
    return torch.cat(outputs)


def _normalized_difference(actual, expected):
    actual = actual.double()
    expected = expected.double()
    denominator = (actual.square() + expected.square()).sum()
    if denominator == 0:
        return 0.0
    similarity = 2 * (actual * expected).sum() / denominator
    return (1 - similarity).item()


def _normalized_randn(shape, rms):
    values = torch.randn(shape, dtype=torch.float32, device="cuda")
    values = values * (
        rms / values.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1.0e-12)
    )
    return values.to(torch.bfloat16)


def _fake_quantize_mxfp8(values):
    values = values.float()
    rows, reduction = values.shape
    blocks = (reduction + _MXFP8_BLOCK_SIZE - 1) // _MXFP8_BLOCK_SIZE
    padded_reduction = blocks * _MXFP8_BLOCK_SIZE
    padded = F.pad(values, (0, padded_reduction - reduction))
    blocked = padded.view(rows, blocks, _MXFP8_BLOCK_SIZE)

    min_amax = 2.0**-126 * _E4M3_MAX
    amax = blocked.abs().amax(dim=-1, keepdim=True).clamp_min(min_amax)
    scale_exponent = torch.ceil(torch.log2(amax / _E4M3_MAX)).clamp(-126, 127)
    scale = torch.exp2(scale_exponent)
    quantized = (blocked / scale).to(torch.float8_e4m3fn).float()
    return (quantized * scale).reshape(rows, padded_reduction)[:, :reduction]


def _attention_mask(
    query_length,
    key_length,
    query_start,
    key_start,
    window_left,
    window_right,
    device,
):
    query_idx = torch.arange(query_start, query_start + _TILE_SIZE, device=device)[
        :, None
    ]
    key_idx = torch.arange(key_start, key_start + _TILE_SIZE, device=device)[None, :]
    aligned_query_idx = query_idx + key_length - query_length
    return (
        (query_idx < query_length)
        & (key_idx < key_length)
        & (key_idx >= torch.clamp(aligned_query_idx - window_left, min=0))
        & (key_idx < torch.clamp(aligned_query_idx + 1 + window_right, max=key_length))
    )


def _mxfp8_reference(
    q,
    k,
    v,
    dout,
    q_lengths,
    k_lengths,
    max_seqlen_q,
    max_seqlen_k,
    alpha,
    window,
):
    padded_q = ((max_seqlen_q + _TILE_SIZE - 1) // _TILE_SIZE) * _TILE_SIZE
    padded_k = ((max_seqlen_k + _TILE_SIZE - 1) // _TILE_SIZE) * _TILE_SIZE
    window_left = max_seqlen_k if window[0] < 0 else window[0]
    window_right = max_seqlen_k if window[1] < 0 else window[1]
    heads = q.shape[1]
    head_dim = q.shape[2]

    output = torch.empty_like(q)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    query_offset = 0
    key_offset = 0

    for query_length, key_length in zip(q_lengths, k_lengths):
        for head in range(heads):
            dense_q = torch.zeros(
                padded_q, head_dim, dtype=torch.float32, device=q.device
            )
            dense_k = torch.zeros(
                padded_k, head_dim, dtype=torch.float32, device=q.device
            )
            dense_v = torch.zeros_like(dense_k)
            dense_dout = torch.zeros_like(dense_q)
            dense_q[:query_length] = q[
                query_offset : query_offset + query_length, head
            ].float()
            dense_k[:key_length] = k[key_offset : key_offset + key_length, head].float()
            dense_v[:key_length] = v[key_offset : key_offset + key_length, head].float()
            dense_dout[:query_length] = dout[
                query_offset : query_offset + query_length, head
            ].float()

            q_tiles = list(dense_q.split(_TILE_SIZE))
            k_tiles = list(dense_k.split(_TILE_SIZE))
            v_tiles = list(dense_v.split(_TILE_SIZE))
            dout_tiles = list(dense_dout.split(_TILE_SIZE))

            k_mx_tiles = [_fake_quantize_mxfp8(tile) for tile in k_tiles]
            vt_mx_tiles = [
                _fake_quantize_mxfp8(tile.transpose(0, 1)) for tile in v_tiles
            ]
            dense_output = torch.empty_like(dense_q, dtype=torch.bfloat16)
            for query_tile_idx, query_tile in enumerate(q_tiles):
                query_start = query_tile_idx * _TILE_SIZE
                query_mx = _fake_quantize_mxfp8(query_tile)
                output_accumulator = torch.zeros_like(query_tile)
                for key_tile_idx, (key_mx, value_t_mx) in enumerate(
                    zip(k_mx_tiles, vt_mx_tiles)
                ):
                    key_start = key_tile_idx * _TILE_SIZE
                    scores = query_mx @ key_mx.transpose(0, 1)
                    valid = _attention_mask(
                        query_length,
                        key_length,
                        query_start,
                        key_start,
                        window_left,
                        window_right,
                        q.device,
                    )
                    probabilities = (F.silu(scores * alpha) / max_seqlen_q) * valid
                    output_accumulator += _fake_quantize_mxfp8(
                        probabilities
                    ) @ value_t_mx.transpose(0, 1)
                dense_output[query_start : query_start + _TILE_SIZE] = (
                    output_accumulator.to(torch.bfloat16)
                )

            q_mx_tiles = [_fake_quantize_mxfp8(tile) for tile in q_tiles]
            qt_mx_tiles = [
                _fake_quantize_mxfp8(tile.transpose(0, 1)) for tile in q_tiles
            ]
            kt_mx_tiles = [
                _fake_quantize_mxfp8(tile.transpose(0, 1)) for tile in k_tiles
            ]
            v_mx_tiles = [_fake_quantize_mxfp8(tile) for tile in v_tiles]
            dout_mx_tiles = [_fake_quantize_mxfp8(tile) for tile in dout_tiles]
            doutt_mx_tiles = [
                _fake_quantize_mxfp8(tile.transpose(0, 1)) for tile in dout_tiles
            ]

            dq_accumulators = [torch.zeros_like(tile) for tile in q_tiles]
            dense_dk = torch.empty_like(dense_k, dtype=torch.bfloat16)
            dense_dv = torch.empty_like(dense_v, dtype=torch.bfloat16)
            for key_tile_idx, (key_mx, key_t_mx, value_mx) in enumerate(
                zip(k_mx_tiles, kt_mx_tiles, v_mx_tiles)
            ):
                key_start = key_tile_idx * _TILE_SIZE
                dk_accumulator = torch.zeros_like(k_tiles[key_tile_idx])
                dv_accumulator = torch.zeros_like(v_tiles[key_tile_idx])
                for query_tile_idx, (
                    query_mx,
                    query_t_mx,
                    dout_mx,
                    dout_t_mx,
                ) in enumerate(
                    zip(
                        q_mx_tiles,
                        qt_mx_tiles,
                        dout_mx_tiles,
                        doutt_mx_tiles,
                    )
                ):
                    query_start = query_tile_idx * _TILE_SIZE
                    scores = query_mx @ key_mx.transpose(0, 1)
                    dprobabilities = dout_mx @ value_mx.transpose(0, 1)
                    valid = _attention_mask(
                        query_length,
                        key_length,
                        query_start,
                        key_start,
                        window_left,
                        window_right,
                        q.device,
                    )
                    scaled_scores = scores * alpha
                    sigmoid = torch.sigmoid(scaled_scores)
                    probabilities = (scaled_scores * sigmoid / max_seqlen_q) * valid
                    derivative = sigmoid * (1 + scaled_scores * (1 - sigmoid))
                    dscores = dprobabilities * derivative * alpha / max_seqlen_q * valid

                    dv_accumulator += _fake_quantize_mxfp8(
                        probabilities.transpose(0, 1)
                    ) @ dout_t_mx.transpose(0, 1)
                    dk_accumulator += _fake_quantize_mxfp8(
                        dscores.transpose(0, 1)
                    ) @ query_t_mx.transpose(0, 1)
                    dq_accumulators[query_tile_idx] += _fake_quantize_mxfp8(
                        dscores
                    ) @ key_t_mx.transpose(0, 1)

                dense_dk[key_start : key_start + _TILE_SIZE] = dk_accumulator.to(
                    torch.bfloat16
                )
                dense_dv[key_start : key_start + _TILE_SIZE] = dv_accumulator.to(
                    torch.bfloat16
                )

            dense_dq = torch.empty_like(dense_q, dtype=torch.bfloat16)
            for query_tile_idx, accumulator in enumerate(dq_accumulators):
                query_start = query_tile_idx * _TILE_SIZE
                dense_dq[query_start : query_start + _TILE_SIZE] = accumulator.to(
                    torch.bfloat16
                )

            output[query_offset : query_offset + query_length, head] = dense_output[
                :query_length
            ]
            dq[query_offset : query_offset + query_length, head] = dense_dq[
                :query_length
            ]
            dk[key_offset : key_offset + key_length, head] = dense_dk[:key_length]
            dv[key_offset : key_offset + key_length, head] = dense_dv[:key_length]

        query_offset += query_length
        key_offset += key_length

    return output, dq, dk, dv


@pytest.mark.parametrize(
    "q_lengths,k_lengths,heads,head_dim,window,alpha",
    [
        ([32], [32], 1, 64, (-1, -1), 1.0),
        ([17, 65], [31, 73], 2, 128, (-1, 0), 0.7),
        ([33, 96], [33, 96], 1, 64, (16, 4), 1.2),
        ([24], [24], 1, 256, (-1, -1), 0.5),
        ([129], [160], 1, 64, (-1, 0), 0.9),
        ([257] * 4, [385] * 4, 16, 64, (-1, 0), 0.9),
    ],
)
def test_hstu_mxfp8_forward_backward(
    q_lengths,
    k_lengths,
    heads,
    head_dim,
    window,
    alpha,
):
    torch.manual_seed(2026 + head_dim)
    total_q = sum(q_lengths)
    total_k = sum(k_lengths)
    max_q = max(q_lengths)
    max_k = max(k_lengths)
    q = _normalized_randn((total_q, heads, head_dim), 0.2)
    k = _normalized_randn((total_k, heads, head_dim), 0.2)
    v = _normalized_randn((total_k, heads, head_dim), 0.2)
    dout = _normalized_randn((total_q, heads, head_dim), 0.1)
    cu_q = _offsets(q_lengths)
    cu_k = _offsets(k_lengths)

    output, _ = hstu_varlen_fwd_mxfp8_100(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        None,
        None,
        1,
        window[0],
        window[1],
        alpha,
        None,
        None,
    )
    dq, dk, dv, _ = hstu_varlen_bwd_mxfp8_100(
        dout,
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        None,
        None,
        None,
        None,
        None,
        1,
        window[0],
        window[1],
        alpha,
        None,
        False,
        None,
        False,
    )

    q_ref = q.float().detach().requires_grad_()
    k_ref = k.float().detach().requires_grad_()
    v_ref = v.float().detach().requires_grad_()
    output_ref = _reference(
        q_ref,
        k_ref,
        v_ref,
        q_lengths,
        k_lengths,
        max_q,
        max_k,
        alpha,
        window,
    )
    output_ref.backward(dout.float())
    output_mx_ref, dq_mx_ref, dk_mx_ref, dv_mx_ref = _mxfp8_reference(
        q,
        k,
        v,
        dout,
        q_lengths,
        k_lengths,
        max_q,
        max_k,
        alpha,
        window,
    )

    assert output.dtype == torch.bfloat16
    assert dq.dtype == dk.dtype == dv.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert torch.isfinite(dq).all()
    assert torch.isfinite(dk).all()
    assert torch.isfinite(dv).all()
    assert _normalized_difference(output, output_ref) < 0.020
    assert (output.float() - output_ref.float()).abs().max().item() < 0.01
    assert _normalized_difference(dq, q_ref.grad) < 0.032
    assert _normalized_difference(dk, k_ref.grad) < 0.032
    assert _normalized_difference(dv, v_ref.grad) < 0.025
    assert _normalized_difference(output, output_mx_ref) < 5.0e-6
    assert _normalized_difference(dq, dq_mx_ref) < 5.0e-6
    assert _normalized_difference(dk, dk_mx_ref) < 5.0e-6
    assert _normalized_difference(dv, dv_mx_ref) < 5.0e-6


def test_hstu_mxfp8_has_no_quadratic_fp32_workspace(monkeypatch):
    allocations = []
    original_matrix = mxfp8_attention._matrix

    def recording_matrix(rows, columns, batches, dtype, device):
        allocations.append((rows, columns, batches, dtype))
        return original_matrix(rows, columns, batches, dtype, device)

    monkeypatch.setattr(mxfp8_attention, "_matrix", recording_matrix)
    query_length, key_length, heads, head_dim = 129, 160, 1, 64
    q = torch.randn(query_length, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(key_length, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    dout = torch.randn_like(q)
    cu_q = _offsets([query_length])
    cu_k = _offsets([key_length])

    hstu_varlen_fwd_mxfp8_100(
        q,
        k,
        v,
        cu_q,
        cu_k,
        query_length,
        key_length,
        None,
        None,
        1,
        -1,
        0,
        1.0,
        None,
        None,
    )
    hstu_varlen_bwd_mxfp8_100(
        dout,
        q,
        k,
        v,
        cu_q,
        cu_k,
        query_length,
        key_length,
        None,
        None,
        None,
        None,
        None,
        1,
        -1,
        0,
        1.0,
        None,
        False,
        None,
        False,
    )

    fp32_allocations = [shape for shape in allocations if shape[3] == torch.float32]
    assert fp32_allocations
    assert all(rows <= 128 for rows, _, _, _ in fp32_allocations)
    assert not any(
        rows > 128 and columns > 128 for rows, columns, _, _ in fp32_allocations
    )
    score_workspaces = [
        shape for shape in fp32_allocations if shape[0] == shape[1] == 128
    ]
    assert len(score_workspaces) == 3
