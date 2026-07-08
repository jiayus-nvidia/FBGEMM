# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.

from dataclasses import dataclass
from typing import Optional

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32

from .mxfp8_blockscaled_gemm import Sm100BlockScaledPersistentDenseGemmKernel
from .mxfp8_quantize import (
    E4M3_MAX,
    MXFP8_BLOCK_SIZE,
    MxFp8QuantizeSm100,
    scale_factor_storage_size,
)

MXFP8_QUANT_MODE = 6


class VarlenPackSm100:
    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        cu_seqlens: cute.Tensor,
        dense: cute.Tensor,
        num_heads: Int32,
        stream: cuda.CUstream,
    ):
        tile_count, tile_size, head_dim, batch_heads = dense.shape
        seqlen = tile_count * tile_size
        self.kernel(source, cu_seqlens, dense, num_heads).launch(
            grid=(cute.ceil_div(seqlen * head_dim, 256), batch_heads, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        cu_seqlens: cute.Tensor,
        dense: cute.Tensor,
        num_heads: Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, batch_head, _ = cute.arch.block_idx()
        tile_count, tile_size, head_dim, _ = dense.shape
        seqlen = tile_count * tile_size
        linear_idx = block * 256 + thread
        if linear_idx < seqlen * head_dim:
            row = linear_idx // head_dim
            dim = linear_idx - row * head_dim
            tile_idx = row // tile_size
            row_in_tile = row - tile_idx * tile_size
            batch = batch_head // num_heads
            head = batch_head - batch * num_heads
            offset = Int32(cu_seqlens[batch])
            length = Int32(cu_seqlens[batch + 1]) - offset
            value = dense.element_type(0)
            if row < length:
                value = source[offset + row, head, dim]
            dense[tile_idx, row_in_tile, dim, batch_head] = value


class VarlenUnpackSm100:
    @cute.jit
    def __call__(
        self,
        dense: cute.Tensor,
        cu_seqlens: cute.Tensor,
        output: cute.Tensor,
        num_heads: Int32,
        stream: cuda.CUstream,
    ):
        tile_count, tile_size, head_dim, batch_heads = dense.shape
        seqlen = tile_count * tile_size
        self.kernel(dense, cu_seqlens, output, num_heads).launch(
            grid=(cute.ceil_div(seqlen * head_dim, 256), batch_heads, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        dense: cute.Tensor,
        cu_seqlens: cute.Tensor,
        output: cute.Tensor,
        num_heads: Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, batch_head, _ = cute.arch.block_idx()
        tile_count, tile_size, head_dim, _ = dense.shape
        seqlen = tile_count * tile_size
        linear_idx = block * 256 + thread
        if linear_idx < seqlen * head_dim:
            row = linear_idx // head_dim
            dim = linear_idx - row * head_dim
            tile_idx = row // tile_size
            row_in_tile = row - tile_idx * tile_size
            batch = batch_head // num_heads
            head = batch_head - batch * num_heads
            offset = Int32(cu_seqlens[batch])
            length = Int32(cu_seqlens[batch + 1]) - offset
            if row < length:
                output[offset + row, head, dim] = dense[
                    tile_idx, row_in_tile, dim, batch_head
                ]


@cute.jit
def _valid_attention_position(
    query_idx: Int32,
    key_idx: Int32,
    query_length: Int32,
    key_length: Int32,
    window_left: Int32,
    window_right: Int32,
):
    aligned_query_idx = query_idx + key_length - query_length
    key_begin = cutlass.max(0, aligned_query_idx - window_left)
    key_end = cutlass.min(key_length, aligned_query_idx + 1 + window_right)
    return (
        query_idx < query_length
        and key_idx < key_length
        and key_idx >= key_begin
        and key_idx < key_end
    )


class SiluMaskQuantizeSm100:
    @cute.jit
    def __call__(
        self,
        scores: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        quantized: cute.Tensor,
        scale_storage: cute.Tensor,
        num_heads: Int32,
        normalization: Int32,
        alpha: Float32,
        window_left: Int32,
        window_right: Int32,
        query_start: Int32,
        key_start: Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(quantized.element_type != cutlass.Float8E4M3FN):
            raise TypeError("MXFP8 values must use Float8E4M3FN")
        if cutlass.const_expr(scale_storage.element_type != cutlass.Float8E8M0FNU):
            raise TypeError("MXFP8 scale factors must use Float8E8M0FNU")
        scale_layout = blockscaled_utils.tile_atom_to_shape_SF(
            scores.shape, MXFP8_BLOCK_SIZE
        )
        scales = cute.make_tensor(
            cute.recast_ptr(scale_storage.iterator, dtype=cutlass.Uint8),
            scale_layout,
        )
        query_len, key_len, batch_heads = scores.shape
        blocks_per_row = cute.ceil_div(key_len, MXFP8_BLOCK_SIZE)
        self.kernel(
            scores,
            cu_seqlens_q,
            cu_seqlens_k,
            quantized,
            scales,
            num_heads,
            normalization,
            alpha,
            window_left,
            window_right,
            query_start,
            key_start,
        ).launch(
            grid=(query_len * blocks_per_row, batch_heads, 1),
            block=(cute.arch.WARP_SIZE, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        scores: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        quantized: cute.Tensor,
        scales: cute.Tensor,
        num_heads: Int32,
        normalization: Int32,
        alpha: Float32,
        window_left: Int32,
        window_right: Int32,
        query_start: Int32,
        key_start: Int32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        block, batch_head, _ = cute.arch.block_idx()
        query_len, key_len, _ = scores.shape
        blocks_per_row = cute.ceil_div(key_len, MXFP8_BLOCK_SIZE)
        query_idx_local = block // blocks_per_row
        reduction_block = block - query_idx_local * blocks_per_row
        first_key_idx_local = reduction_block * MXFP8_BLOCK_SIZE + lane * 4
        query_idx = query_start + query_idx_local
        batch = batch_head // num_heads
        query_length = Int32(cu_seqlens_q[batch + 1]) - Int32(cu_seqlens_q[batch])
        key_length = Int32(cu_seqlens_k[batch + 1]) - Int32(cu_seqlens_k[batch])

        values = cute.make_rmem_tensor((4,), Float32)
        local_amax = Float32(0.0)
        for value_idx in cutlass.range_constexpr(4):
            key_idx_local = first_key_idx_local + value_idx
            key_idx = key_start + key_idx_local
            value = Float32(0.0)
            if lane < 8 and key_idx_local < key_len and _valid_attention_position(
                query_idx,
                key_idx,
                query_length,
                key_length,
                window_left,
                window_right,
            ):
                scaled_score = (
                    Float32(scores[query_idx_local, key_idx_local, batch_head]) * alpha
                )
                sigmoid = 1.0 / (1.0 + cute.math.exp(-scaled_score))
                value = scaled_score * sigmoid / Float32(normalization)
            values[value_idx] = value
            local_amax = cute.arch.fmax(local_amax, cute.arch.fmax(value, -value))

        block_amax = cute.arch.warp_reduction_max(local_amax)
        ratio = cute.arch.fmax(block_amax, 2.0**-126 * E4M3_MAX) / E4M3_MAX
        log_scale = cute.math.log2(ratio)
        scale_exponent = Int32(log_scale)
        if log_scale > Float32(scale_exponent):
            scale_exponent += 1
        scale_exponent = cutlass.max(-126, cutlass.min(scale_exponent, 127))
        scale = cute.math.exp2(Float32(scale_exponent))

        if lane == 0:
            scales[
                query_idx_local,
                reduction_block * MXFP8_BLOCK_SIZE,
                batch_head,
            ] = cutlass.Uint8(scale_exponent + 127)
        quantized_values = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
        quantized_values.store((values.load() / scale).to(cutlass.Float8E4M3FN))
        if lane < 8:
            for value_idx in cutlass.range_constexpr(4):
                key_idx_local = first_key_idx_local + value_idx
                if key_idx_local < key_len:
                    quantized[query_idx_local, key_idx_local, batch_head] = (
                        quantized_values[value_idx]
                    )


class SiluBackwardMaskQuantizeSm100:
    @cute.jit
    def __call__(
        self,
        scores: cute.Tensor,
        dprobabilities: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        quantized_dscores: cute.Tensor,
        dscore_scale_storage: cute.Tensor,
        quantized_probabilities_t: cute.Tensor,
        probability_t_scale_storage: cute.Tensor,
        quantized_dscores_t: cute.Tensor,
        dscore_t_scale_storage: cute.Tensor,
        num_heads: Int32,
        normalization: Int32,
        alpha: Float32,
        window_left: Int32,
        window_right: Int32,
        query_start: Int32,
        key_start: Int32,
        stream: cuda.CUstream,
    ):
        scale_layout = blockscaled_utils.tile_atom_to_shape_SF(
            scores.shape, MXFP8_BLOCK_SIZE
        )
        dscore_scales = cute.make_tensor(
            cute.recast_ptr(dscore_scale_storage.iterator, dtype=cutlass.Uint8),
            scale_layout,
        )
        query_len, key_len, batch_heads = scores.shape
        transpose_scale_layout = blockscaled_utils.tile_atom_to_shape_SF(
            (key_len, query_len, batch_heads), MXFP8_BLOCK_SIZE
        )
        probability_t_scales = cute.make_tensor(
            cute.recast_ptr(
                probability_t_scale_storage.iterator, dtype=cutlass.Uint8
            ),
            transpose_scale_layout,
        )
        dscore_t_scales = cute.make_tensor(
            cute.recast_ptr(dscore_t_scale_storage.iterator, dtype=cutlass.Uint8),
            transpose_scale_layout,
        )
        blocks_per_row = cute.ceil_div(key_len, MXFP8_BLOCK_SIZE)
        transpose_blocks_per_row = cute.ceil_div(query_len, MXFP8_BLOCK_SIZE)
        self.kernel(
            scores,
            dprobabilities,
            cu_seqlens_q,
            cu_seqlens_k,
            quantized_dscores,
            dscore_scales,
            quantized_probabilities_t,
            probability_t_scales,
            quantized_dscores_t,
            dscore_t_scales,
            num_heads,
            normalization,
            alpha,
            window_left,
            window_right,
            query_start,
            key_start,
        ).launch(
            grid=(
                query_len * blocks_per_row
                + key_len * transpose_blocks_per_row,
                batch_heads,
                1,
            ),
            block=(cute.arch.WARP_SIZE, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        scores: cute.Tensor,
        dprobabilities: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        quantized_dscores: cute.Tensor,
        dscore_scales: cute.Tensor,
        quantized_probabilities_t: cute.Tensor,
        probability_t_scales: cute.Tensor,
        quantized_dscores_t: cute.Tensor,
        dscore_t_scales: cute.Tensor,
        num_heads: Int32,
        normalization: Int32,
        alpha: Float32,
        window_left: Int32,
        window_right: Int32,
        query_start: Int32,
        key_start: Int32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        block, batch_head, _ = cute.arch.block_idx()
        query_len, key_len, _ = scores.shape
        blocks_per_row = cute.ceil_div(key_len, MXFP8_BLOCK_SIZE)
        batch = batch_head // num_heads
        query_length = Int32(cu_seqlens_q[batch + 1]) - Int32(cu_seqlens_q[batch])
        key_length = Int32(cu_seqlens_k[batch + 1]) - Int32(cu_seqlens_k[batch])
        row_block_count = query_len * blocks_per_row

        if block < row_block_count:
            query_idx_local = block // blocks_per_row
            reduction_block = block - query_idx_local * blocks_per_row
            first_key_idx_local = reduction_block * MXFP8_BLOCK_SIZE + lane * 4
            query_idx = query_start + query_idx_local
            dscore_values = cute.make_rmem_tensor((4,), Float32)
            local_amax = Float32(0.0)
            for value_idx in cutlass.range_constexpr(4):
                key_idx_local = first_key_idx_local + value_idx
                key_idx = key_start + key_idx_local
                dscore = Float32(0.0)
                if (
                    lane < 8
                    and key_idx_local < key_len
                    and _valid_attention_position(
                        query_idx,
                        key_idx,
                        query_length,
                        key_length,
                        window_left,
                        window_right,
                    )
                ):
                    scaled_score = (
                        Float32(scores[query_idx_local, key_idx_local, batch_head])
                        * alpha
                    )
                    sigmoid = 1.0 / (1.0 + cute.math.exp(-scaled_score))
                    derivative = sigmoid * (1.0 + scaled_score * (1.0 - sigmoid))
                    dscore = (
                        Float32(
                            dprobabilities[
                                query_idx_local, key_idx_local, batch_head
                            ]
                        )
                        * derivative
                        * alpha
                        / Float32(normalization)
                    )
                dscore_values[value_idx] = dscore
                local_amax = cute.arch.fmax(
                    local_amax, cute.arch.fmax(dscore, -dscore)
                )
            block_amax = cute.arch.warp_reduction_max(local_amax)
            ratio = cute.arch.fmax(block_amax, 2.0**-126 * E4M3_MAX) / E4M3_MAX
            log_scale = cute.math.log2(ratio)
            scale_exponent = Int32(log_scale)
            if log_scale > Float32(scale_exponent):
                scale_exponent += 1
            scale_exponent = cutlass.max(-126, cutlass.min(scale_exponent, 127))
            scale = cute.math.exp2(Float32(scale_exponent))

            if lane == 0:
                dscore_scales[
                    query_idx_local,
                    reduction_block * MXFP8_BLOCK_SIZE,
                    batch_head,
                ] = cutlass.Uint8(scale_exponent + 127)
            quantized_values = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
            quantized_values.store(
                (dscore_values.load() / scale).to(cutlass.Float8E4M3FN)
            )
            for value_idx in cutlass.range_constexpr(4):
                key_idx_local = first_key_idx_local + value_idx
                if lane < 8 and key_idx_local < key_len:
                    quantized_dscores[
                        query_idx_local, key_idx_local, batch_head
                    ] = quantized_values[value_idx]
        else:
            transpose_block = block - row_block_count
            transpose_blocks_per_row = cute.ceil_div(query_len, MXFP8_BLOCK_SIZE)
            key_idx_local = transpose_block // transpose_blocks_per_row
            reduction_block = (
                transpose_block - key_idx_local * transpose_blocks_per_row
            )
            first_query_idx_local = (
                reduction_block * MXFP8_BLOCK_SIZE + lane * 4
            )
            key_idx = key_start + key_idx_local
            probability_values = cute.make_rmem_tensor((4,), Float32)
            dscore_values = cute.make_rmem_tensor((4,), Float32)
            local_probability_amax = Float32(0.0)
            local_dscore_amax = Float32(0.0)
            for value_idx in cutlass.range_constexpr(4):
                query_idx_local = first_query_idx_local + value_idx
                query_idx = query_start + query_idx_local
                probability = Float32(0.0)
                dscore = Float32(0.0)
                if (
                    lane < 8
                    and query_idx_local < query_len
                    and _valid_attention_position(
                        query_idx,
                        key_idx,
                        query_length,
                        key_length,
                        window_left,
                        window_right,
                    )
                ):
                    scaled_score = (
                        Float32(scores[query_idx_local, key_idx_local, batch_head])
                        * alpha
                    )
                    sigmoid = 1.0 / (1.0 + cute.math.exp(-scaled_score))
                    probability = (
                        scaled_score * sigmoid / Float32(normalization)
                    )
                    derivative = sigmoid * (1.0 + scaled_score * (1.0 - sigmoid))
                    dscore = (
                        Float32(
                            dprobabilities[
                                query_idx_local, key_idx_local, batch_head
                            ]
                        )
                        * derivative
                        * alpha
                        / Float32(normalization)
                    )
                probability_values[value_idx] = probability
                dscore_values[value_idx] = dscore
                local_probability_amax = cute.arch.fmax(
                    local_probability_amax,
                    cute.arch.fmax(probability, -probability),
                )
                local_dscore_amax = cute.arch.fmax(
                    local_dscore_amax, cute.arch.fmax(dscore, -dscore)
                )

            probability_amax = cute.arch.warp_reduction_max(local_probability_amax)
            dscore_amax = cute.arch.warp_reduction_max(local_dscore_amax)
            min_amax = 2.0**-126 * E4M3_MAX
            probability_log_scale = cute.math.log2(
                cute.arch.fmax(probability_amax, min_amax) / E4M3_MAX
            )
            dscore_log_scale = cute.math.log2(
                cute.arch.fmax(dscore_amax, min_amax) / E4M3_MAX
            )
            probability_scale_exponent = Int32(probability_log_scale)
            dscore_scale_exponent = Int32(dscore_log_scale)
            if probability_log_scale > Float32(probability_scale_exponent):
                probability_scale_exponent += 1
            if dscore_log_scale > Float32(dscore_scale_exponent):
                dscore_scale_exponent += 1
            probability_scale_exponent = cutlass.max(
                -126, cutlass.min(probability_scale_exponent, 127)
            )
            dscore_scale_exponent = cutlass.max(
                -126, cutlass.min(dscore_scale_exponent, 127)
            )
            probability_scale = cute.math.exp2(Float32(probability_scale_exponent))
            dscore_scale = cute.math.exp2(Float32(dscore_scale_exponent))

            if lane == 0:
                scale_idx = reduction_block * MXFP8_BLOCK_SIZE
                probability_t_scales[
                    key_idx_local, scale_idx, batch_head
                ] = cutlass.Uint8(probability_scale_exponent + 127)
                dscore_t_scales[
                    key_idx_local, scale_idx, batch_head
                ] = cutlass.Uint8(dscore_scale_exponent + 127)

            quantized_probabilities = cute.make_rmem_tensor(
                (4,), cutlass.Float8E4M3FN
            )
            quantized_dscores_t_values = cute.make_rmem_tensor(
                (4,), cutlass.Float8E4M3FN
            )
            quantized_probabilities.store(
                (probability_values.load() / probability_scale).to(
                    cutlass.Float8E4M3FN
                )
            )
            quantized_dscores_t_values.store(
                (dscore_values.load() / dscore_scale).to(cutlass.Float8E4M3FN)
            )
            for value_idx in cutlass.range_constexpr(4):
                query_idx_local = first_query_idx_local + value_idx
                if lane < 8 and query_idx_local < query_len:
                    quantized_probabilities_t[
                        key_idx_local, query_idx_local, batch_head
                    ] = quantized_probabilities[value_idx]
                    quantized_dscores_t[
                        key_idx_local, query_idx_local, batch_head
                    ] = quantized_dscores_t_values[value_idx]


class MxFp8BackwardCoreSm100:
    def __init__(self):
        self.score_gemm = Sm100BlockScaledPersistentDenseGemmKernel(
            32, (128, 128), (1, 1)
        )
        self.dprobability_gemm = Sm100BlockScaledPersistentDenseGemmKernel(
            32, (128, 128), (1, 1)
        )
        self.silu_backward = SiluBackwardMaskQuantizeSm100()

    @cute.jit
    def __call__(
        self,
        query: cute.Tensor,
        key: cute.Tensor,
        query_scales: cute.Tensor,
        key_scales: cute.Tensor,
        dout: cute.Tensor,
        value: cute.Tensor,
        dout_scales: cute.Tensor,
        value_scales: cute.Tensor,
        scores: cute.Tensor,
        dprobabilities: cute.Tensor,
        quantized_dscores: cute.Tensor,
        dscore_scales: cute.Tensor,
        quantized_probabilities_t: cute.Tensor,
        probability_t_scales: cute.Tensor,
        quantized_dscores_t: cute.Tensor,
        dscore_t_scales: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        num_heads: Int32,
        normalization: Int32,
        alpha: Float32,
        window_left: Int32,
        window_right: Int32,
        query_start: Int32,
        key_start: Int32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.score_gemm(
            query,
            key,
            query_scales,
            key_scales,
            scores,
            scores,
            max_active_clusters,
            stream,
        )
        self.dprobability_gemm(
            dout,
            value,
            dout_scales,
            value_scales,
            dprobabilities,
            dprobabilities,
            max_active_clusters,
            stream,
        )
        self.silu_backward(
            scores,
            dprobabilities,
            cu_seqlens_q,
            cu_seqlens_k,
            quantized_dscores,
            dscore_scales,
            quantized_probabilities_t,
            probability_t_scales,
            quantized_dscores_t,
            dscore_t_scales,
            num_heads,
            normalization,
            alpha,
            window_left,
            window_right,
            query_start,
            key_start,
            stream,
        )


@dataclass
class _MxMatrix:
    values: torch.Tensor
    scales: torch.Tensor
    _cute_values: Optional[object] = None
    _cute_scales: Optional[object] = None


_compile_cache = {
    "pack": {},
    "unpack": {},
    "silu_quantize": {},
    "silu_bwd_quantize": {},
    "bwd_core": {},
    "quantize": {},
    "gemm": {},
}


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _leading_dim(tensor: torch.Tensor) -> int:
    if tensor.stride(1) == 1:
        return 1
    if tensor.stride(0) == 1:
        return 0
    raise ValueError(f"matrix must be major in M/N or K, got stride {tensor.stride()}")


def _cute_tensor(tensor: torch.Tensor, *, element_type=None):
    leading_dim = (
        _leading_dim(tensor)
        if tensor.ndim == 3
        else tuple(tensor.stride()).index(1)
    )
    result = from_dlpack(tensor.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=leading_dim
    )
    if element_type is not None:
        result.element_type = element_type
    return result


def _mark_matrix(tensor: cute.Tensor, leading_dim: int) -> cute.Tensor:
    tensor.mark_compact_shape_dynamic(
        mode=leading_dim,
        stride_order=(2, 0, 1) if leading_dim == 1 else (2, 1, 0),
        divisibility=16,
    )
    return tensor


def _mx_cute_tensors(matrix: _MxMatrix):
    if matrix._cute_values is None:
        leading_dim = _leading_dim(matrix.values)
        matrix._cute_values = _mark_matrix(
            _cute_tensor(matrix.values, element_type=cutlass.Float8E4M3FN),
            leading_dim,
        )
        matrix._cute_scales = _cute_tensor(
            matrix.scales, element_type=cutlass.Float8E8M0FNU
        )
    return matrix._cute_values, matrix._cute_scales


def _matrix(rows: int, columns: int, batches: int, dtype: torch.dtype, device):
    storage = torch.empty((batches, rows, columns), dtype=dtype, device=device)
    return storage.permute(1, 2, 0)


def _tiled_matrix(
    tile_count: int,
    tile_size: int,
    columns: int,
    batches: int,
    dtype: torch.dtype,
    device,
):
    storage = torch.empty(
        (tile_count, batches, tile_size, columns), dtype=dtype, device=device
    )
    return storage.permute(0, 2, 3, 1)


def _matrix_with_major(
    rows: int,
    columns: int,
    batches: int,
    dtype: torch.dtype,
    device,
    leading_dim: int,
):
    if leading_dim == 1:
        return _matrix(rows, columns, batches, dtype, device)
    storage = torch.empty((batches, columns, rows), dtype=dtype, device=device)
    return storage.permute(2, 1, 0)


def _mx_workspace(shape: tuple[int, int, int], device) -> _MxMatrix:
    values = _matrix(*shape, torch.uint8, device)
    scales = torch.empty(
        scale_factor_storage_size(*shape), dtype=torch.uint8, device=device
    )
    return _MxMatrix(values, scales)


def _validate_mx_workspace(workspace: _MxMatrix, shape, device) -> None:
    if (
        workspace.values.shape != shape
        or workspace.values.dtype != torch.uint8
        or workspace.values.device != device
        or workspace.scales.shape != (scale_factor_storage_size(*shape),)
        or workspace.scales.dtype != torch.uint8
        or workspace.scales.device != device
    ):
        raise ValueError(f"invalid MXFP8 workspace for shape {shape}")


def _transpose(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.permute(1, 0, 2)


def _pack(
    source: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padded_seqlen: int,
) -> torch.Tensor:
    batches = cu_seqlens.numel() - 1
    num_heads, head_dim = source.shape[1:]
    dense = _tiled_matrix(
        padded_seqlen // 128,
        128,
        head_dim,
        batches * num_heads,
        source.dtype,
        source.device,
    )
    source_tensor = from_dlpack(source.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=source.ndim - 1
    )
    offsets_tensor = from_dlpack(
        cu_seqlens.detach(), assumed_align=16
    ).mark_layout_dynamic(leading_dim=0)
    dense_tensor = _cute_tensor(dense)
    stream = cutlass_torch.default_stream()
    key = (source.dtype, padded_seqlen, head_dim, num_heads, batches)
    if key not in _compile_cache["pack"]:
        _compile_cache["pack"][key] = cute.compile(
            VarlenPackSm100(),
            source_tensor,
            offsets_tensor,
            dense_tensor,
            Int32(num_heads),
            stream,
        )
    _compile_cache["pack"][key](
        source_tensor,
        offsets_tensor,
        dense_tensor,
        Int32(num_heads),
        stream,
    )
    return dense


def _unpack(
    dense: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    num_heads: int,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    tile_count, tile_size, head_dim, _ = dense.shape
    if output is None:
        output = torch.empty(
            (total_tokens, num_heads, head_dim),
            dtype=dense.dtype,
            device=dense.device,
        )
    elif output.shape != (total_tokens, num_heads, head_dim):
        raise ValueError(
            f"invalid output buffer shape {output.shape}, expected "
            f"{(total_tokens, num_heads, head_dim)}"
        )
    dense_tensor = _cute_tensor(dense)
    offsets_tensor = from_dlpack(
        cu_seqlens.detach(), assumed_align=16
    ).mark_layout_dynamic(leading_dim=0)
    output_tensor = from_dlpack(output.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=output.ndim - 1
    )
    stream = cutlass_torch.default_stream()
    key = (
        dense.dtype,
        tile_count,
        tile_size,
        head_dim,
        num_heads,
        cu_seqlens.numel() - 1,
    )
    if key not in _compile_cache["unpack"]:
        _compile_cache["unpack"][key] = cute.compile(
            VarlenUnpackSm100(),
            dense_tensor,
            offsets_tensor,
            output_tensor,
            Int32(num_heads),
            stream,
        )
    _compile_cache["unpack"][key](
        dense_tensor,
        offsets_tensor,
        output_tensor,
        Int32(num_heads),
        stream,
    )
    return output


def _quantize(matrix: torch.Tensor) -> _MxMatrix:
    leading_dim = _leading_dim(matrix)
    values = _matrix_with_major(
        *matrix.shape,
        torch.uint8,
        matrix.device,
        leading_dim,
    )
    scales = torch.empty(
        scale_factor_storage_size(*matrix.shape),
        dtype=torch.uint8,
        device=matrix.device,
    )
    source_tensor = _cute_tensor(matrix)
    values_tensor = _mark_matrix(
        _cute_tensor(values, element_type=cutlass.Float8E4M3FN),
        leading_dim,
    )
    scales_tensor = _cute_tensor(scales, element_type=cutlass.Float8E8M0FNU)
    stream = cutlass_torch.default_stream()
    key = (
        matrix.dtype,
        tuple(matrix.shape),
        tuple(matrix.stride()),
        tuple(values.stride()),
    )
    if key not in _compile_cache["quantize"]:
        _compile_cache["quantize"][key] = cute.compile(
            MxFp8QuantizeSm100(),
            source_tensor,
            values_tensor,
            scales_tensor,
            stream,
        )
    _compile_cache["quantize"][key](source_tensor, values_tensor, scales_tensor, stream)
    return _MxMatrix(values, scales)


def _quantize_tiles(
    tiled_matrix: torch.Tensor, *, transpose: bool = False
) -> list[_MxMatrix]:
    if tiled_matrix.ndim != 4 or tiled_matrix.shape[1] != 128:
        raise ValueError(
            f"tile collection must have shape (tiles, 128, columns, batches), "
            f"got {tiled_matrix.shape}"
        )
    tile_count, tile_size, columns, batches = tiled_matrix.shape
    if transpose:
        matrix = tiled_matrix.permute(2, 1, 0, 3).reshape(
            columns, tile_size, tile_count * batches
        )
    else:
        matrix = tiled_matrix.permute(1, 2, 0, 3).reshape(
            tile_size, columns, tile_count * batches
        )

    combined = _quantize(matrix)
    scale_count = scale_factor_storage_size(
        matrix.shape[0], matrix.shape[1], batches
    )
    if combined.scales.numel() != tile_count * scale_count:
        raise RuntimeError("batched quantization produced an invalid scale layout")
    return [
        _MxMatrix(
            combined.values[:, :, tile_idx * batches : (tile_idx + 1) * batches],
            combined.scales[
                tile_idx * scale_count : (tile_idx + 1) * scale_count
            ],
        )
        for tile_idx in range(tile_count)
    ]


def _gemm(
    a: _MxMatrix,
    b: _MxMatrix,
    output_dtype: torch.dtype,
    output: Optional[torch.Tensor] = None,
    addend: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    m, k, batches = a.values.shape
    n, b_k, b_batches = b.values.shape
    if k != b_k or batches != b_batches:
        raise ValueError(
            f"incompatible MXFP8 GEMM shapes {a.values.shape} and {b.values.shape}"
        )
    expected_shape = (m, n, batches)
    if output is None:
        output = _matrix(m, n, batches, output_dtype, a.values.device)
    elif output.shape != expected_shape or output.dtype != output_dtype:
        raise ValueError(
            f"invalid GEMM output {output.shape}/{output.dtype}, expected "
            f"{expected_shape}/{output_dtype}"
        )
    accumulate_output = addend is not None
    if addend is None:
        addend = output
    elif addend.shape != expected_shape:
        raise ValueError(
            f"invalid GEMM addend shape {addend.shape}, expected {expected_shape}"
        )

    a_tensor, a_scales = _mx_cute_tensors(a)
    b_tensor, b_scales = _mx_cute_tensors(b)
    output_tensor = _mark_matrix(_cute_tensor(output), 1)
    addend_leading = _leading_dim(addend)
    addend_tensor = _mark_matrix(_cute_tensor(addend), addend_leading)
    stream = cutlass_torch.default_stream()
    mma_n = 64 if n <= 64 else 128
    key = (
        tuple(a.values.shape),
        tuple(a.values.stride()),
        tuple(b.values.shape),
        tuple(b.values.stride()),
        output_dtype,
        tuple(output.stride()),
        addend.dtype,
        tuple(addend.stride()),
        accumulate_output,
        mma_n,
    )
    if key not in _compile_cache["gemm"]:
        gemm = Sm100BlockScaledPersistentDenseGemmKernel(
            32,
            (128, mma_n),
            (1, 1),
            accumulate_output=accumulate_output,
        )
        max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
        _compile_cache["gemm"][key] = cute.compile(
            gemm,
            a_tensor,
            b_tensor,
            a_scales,
            b_scales,
            output_tensor,
            addend_tensor,
            max_active_clusters,
            stream,
        )
    _compile_cache["gemm"][key](
        a_tensor,
        b_tensor,
        a_scales,
        b_scales,
        output_tensor,
        addend_tensor,
        stream,
    )
    return output


def _tiles(matrix: torch.Tensor, tile_size: int = 128):
    if matrix.ndim == 4:
        if matrix.shape[1] != tile_size:
            raise ValueError(
                f"tile size {matrix.shape[1]} does not match expected {tile_size}"
            )
        return list(matrix.unbind(0))
    if matrix.shape[0] % tile_size != 0:
        raise ValueError(
            f"row count {matrix.shape[0]} must be divisible by {tile_size}"
        )
    return [
        matrix[start : start + tile_size]
        for start in range(0, matrix.shape[0], tile_size)
    ]


def _normalize_window(window: int, max_seqlen_k: int) -> int:
    return max_seqlen_k if window < 0 or window > max_seqlen_k else window


def _check_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
):
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("SM100 MXFP8 HSTU requires BF16 Q, K, and V")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Q, K, and V must have shape (total_tokens, heads, dim)")
    if q.shape[1:] != k.shape[1:] or q.shape[1:] != v.shape[1:]:
        raise ValueError(
            "SM100 MXFP8 HSTU currently requires equal Q/K/V heads and dims"
        )
    if q.shape[2] not in (64, 128, 256):
        raise ValueError("SM100 MXFP8 HSTU supports head dimensions 64, 128, and 256")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise TypeError("cu_seqlens_q and cu_seqlens_k must use int32")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("Q and K must have the same batch size")


def _run_silu(
    scores: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_heads: int,
    normalization: int,
    alpha: float,
    window_left: int,
    window_right: int,
    query_start: int = 0,
    key_start: int = 0,
    output: Optional[_MxMatrix] = None,
) -> _MxMatrix:
    if scores.shape[:2] != (128, 128) or scores.dtype != torch.float32:
        raise ValueError("fused SiLU quantization requires a 128x128 FP32 score tile")
    output = (
        _mx_workspace(tuple(scores.shape), scores.device)
        if output is None
        else output
    )
    _validate_mx_workspace(output, scores.shape, scores.device)
    scores_tensor = _cute_tensor(scores)
    values_tensor, scales_tensor = _mx_cute_tensors(output)
    offsets_q = from_dlpack(
        cu_seqlens_q.detach(), assumed_align=16
    ).mark_layout_dynamic(leading_dim=0)
    offsets_k = from_dlpack(
        cu_seqlens_k.detach(), assumed_align=16
    ).mark_layout_dynamic(leading_dim=0)
    stream = cutlass_torch.default_stream()
    key = (tuple(scores.shape), tuple(scores.stride()), num_heads)
    if key not in _compile_cache["silu_quantize"]:
        _compile_cache["silu_quantize"][key] = cute.compile(
            SiluMaskQuantizeSm100(),
            scores_tensor,
            offsets_q,
            offsets_k,
            values_tensor,
            scales_tensor,
            Int32(num_heads),
            Int32(normalization),
            Float32(alpha),
            Int32(window_left),
            Int32(window_right),
            Int32(query_start),
            Int32(key_start),
            stream,
        )
    _compile_cache["silu_quantize"][key](
        scores_tensor,
        offsets_q,
        offsets_k,
        values_tensor,
        scales_tensor,
        Int32(num_heads),
        Int32(normalization),
        Float32(alpha),
        Int32(window_left),
        Int32(window_right),
        Int32(query_start),
        Int32(key_start),
        stream,
    )
    return output


def _run_silu_backward(
    scores: torch.Tensor,
    dprobabilities: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_heads: int,
    normalization: int,
    alpha: float,
    window_left: int,
    window_right: int,
    query_start: int = 0,
    key_start: int = 0,
    outputs: Optional[tuple[_MxMatrix, _MxMatrix, _MxMatrix]] = None,
):
    if scores.shape[:2] != (128, 128):
        raise ValueError("fused SiLU backward quantization requires 128x128 tiles")
    shape = tuple(scores.shape)
    if outputs is None:
        outputs = tuple(_mx_workspace(shape, scores.device) for _ in range(3))
    elif len(outputs) != 3:
        raise ValueError("SiLU backward requires three MXFP8 workspaces")
    output_tensors = []
    for output in outputs:
        _validate_mx_workspace(output, scores.shape, scores.device)
        output_tensors.extend(_mx_cute_tensors(output))
    score_tensor = _cute_tensor(scores)
    dprobability_tensor = _cute_tensor(dprobabilities)
    offsets_q = from_dlpack(
        cu_seqlens_q.detach(), assumed_align=16
    ).mark_layout_dynamic(leading_dim=0)
    offsets_k = from_dlpack(
        cu_seqlens_k.detach(), assumed_align=16
    ).mark_layout_dynamic(leading_dim=0)
    stream = cutlass_torch.default_stream()
    key = (tuple(scores.shape), num_heads)
    if key not in _compile_cache["silu_bwd_quantize"]:
        _compile_cache["silu_bwd_quantize"][key] = cute.compile(
            SiluBackwardMaskQuantizeSm100(),
            score_tensor,
            dprobability_tensor,
            offsets_q,
            offsets_k,
            output_tensors[0],
            output_tensors[1],
            output_tensors[2],
            output_tensors[3],
            output_tensors[4],
            output_tensors[5],
            Int32(num_heads),
            Int32(normalization),
            Float32(alpha),
            Int32(window_left),
            Int32(window_right),
            Int32(query_start),
            Int32(key_start),
            stream,
        )
    _compile_cache["silu_bwd_quantize"][key](
        score_tensor,
        dprobability_tensor,
        offsets_q,
        offsets_k,
        output_tensors[0],
        output_tensors[1],
        output_tensors[2],
        output_tensors[3],
        output_tensors[4],
        output_tensors[5],
        Int32(num_heads),
        Int32(normalization),
        Float32(alpha),
        Int32(window_left),
        Int32(window_right),
        Int32(query_start),
        Int32(key_start),
        stream,
    )
    return outputs[0], outputs[1], outputs[2]


def _run_backward_core(
    query: _MxMatrix,
    key: _MxMatrix,
    dout: _MxMatrix,
    value: _MxMatrix,
    scores: torch.Tensor,
    dprobabilities: torch.Tensor,
    outputs: tuple[_MxMatrix, _MxMatrix, _MxMatrix],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_heads: int,
    normalization: int,
    alpha: float,
    window_left: int,
    window_right: int,
    query_start: int,
    key_start: int,
):
    if scores.shape[:2] != (128, 128) or scores.dtype != torch.float32:
        raise ValueError("backward core requires a 128x128 FP32 score workspace")
    if dprobabilities.shape != scores.shape or dprobabilities.dtype != torch.float32:
        raise ValueError("backward core requires a matching FP32 dP workspace")

    query_tensor, query_scales = _mx_cute_tensors(query)
    key_tensor, key_scales = _mx_cute_tensors(key)
    dout_tensor, dout_scales = _mx_cute_tensors(dout)
    value_tensor, value_scales = _mx_cute_tensors(value)
    score_tensor = _mark_matrix(_cute_tensor(scores), 1)
    dprobability_tensor = _mark_matrix(_cute_tensor(dprobabilities), 1)
    output_tensors = []
    for output in outputs:
        _validate_mx_workspace(output, scores.shape, scores.device)
        output_tensors.extend(_mx_cute_tensors(output))
    offsets_q = from_dlpack(
        cu_seqlens_q.detach(), assumed_align=16
    ).mark_layout_dynamic(leading_dim=0)
    offsets_k = from_dlpack(
        cu_seqlens_k.detach(), assumed_align=16
    ).mark_layout_dynamic(leading_dim=0)
    stream = cutlass_torch.default_stream()
    max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
    cache_key = (
        tuple(query.values.shape),
        tuple(query.values.stride()),
        tuple(key.values.stride()),
        tuple(dout.values.stride()),
        tuple(value.values.stride()),
        tuple(scores.stride()),
        num_heads,
    )
    if cache_key not in _compile_cache["bwd_core"]:
        _compile_cache["bwd_core"][cache_key] = cute.compile(
            MxFp8BackwardCoreSm100(),
            query_tensor,
            key_tensor,
            query_scales,
            key_scales,
            dout_tensor,
            value_tensor,
            dout_scales,
            value_scales,
            score_tensor,
            dprobability_tensor,
            output_tensors[0],
            output_tensors[1],
            output_tensors[2],
            output_tensors[3],
            output_tensors[4],
            output_tensors[5],
            offsets_q,
            offsets_k,
            Int32(num_heads),
            Int32(normalization),
            Float32(alpha),
            Int32(window_left),
            Int32(window_right),
            Int32(query_start),
            Int32(key_start),
            max_active_clusters,
            stream,
        )
    _compile_cache["bwd_core"][cache_key](
        query_tensor,
        key_tensor,
        query_scales,
        key_scales,
        dout_tensor,
        value_tensor,
        dout_scales,
        value_scales,
        score_tensor,
        dprobability_tensor,
        output_tensors[0],
        output_tensors[1],
        output_tensors[2],
        output_tensors[3],
        output_tensors[4],
        output_tensors[5],
        offsets_q,
        offsets_k,
        Int32(num_heads),
        Int32(normalization),
        Float32(alpha),
        Int32(window_left),
        Int32(window_right),
        Int32(query_start),
        Int32(key_start),
        stream,
    )
    return outputs


def hstu_varlen_fwd_mxfp8_100(
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
    rab: Optional[torch.Tensor],
    func: Optional[torch.Tensor],
    paged_kv: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    page_indptrs: Optional[torch.Tensor] = None,
):
    _check_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k)
    if any(
        x is not None
        for x in (
            num_contexts,
            num_targets,
            rab,
            func,
            paged_kv,
            page_ids,
            page_indptrs,
        )
    ):
        raise NotImplementedError(
            "SM100 MXFP8 HSTU currently supports causal, local, and full masks only"
        )
    q, k, v = (x.contiguous() for x in (q, k, v))
    num_heads = q.shape[1]
    padded_q = _round_up(max_seqlen_q, 128)
    padded_k = _round_up(max_seqlen_k, 128)
    dense_q = _pack(q, cu_seqlens_q, padded_q)
    dense_k = _pack(k, cu_seqlens_k, padded_k)
    dense_v = _pack(v, cu_seqlens_k, padded_k)

    window_size_left = _normalize_window(window_size_left, max_seqlen_k)
    window_size_right = _normalize_window(window_size_right, max_seqlen_k)
    batch_heads = (cu_seqlens_q.numel() - 1) * num_heads
    head_dim = q.shape[2]
    q_tiles = _tiles(dense_q)
    k_tiles = _tiles(dense_k)
    v_tiles = _tiles(dense_v)
    dense_output = _tiled_matrix(
        len(q_tiles), 128, head_dim, batch_heads, torch.bfloat16, q.device
    )
    q_mx_tiles = _quantize_tiles(dense_q)
    k_mx_tiles = _quantize_tiles(dense_k)
    v_mx_tiles = _quantize_tiles(dense_v, transpose=True)
    score_workspace = _matrix(128, 128, batch_heads, torch.float32, q.device)
    probability_workspace = _mx_workspace((128, 128, batch_heads), q.device)
    output_accumulator = (
        _matrix(128, head_dim, batch_heads, torch.float32, q.device)
        if len(k_mx_tiles) > 1
        else None
    )

    for query_tile_idx, query_mx in enumerate(q_mx_tiles):
        query_start = query_tile_idx * 128
        for key_tile_idx, (key_mx, value_mx) in enumerate(zip(k_mx_tiles, v_mx_tiles)):
            key_start = key_tile_idx * 128
            scores = _gemm(query_mx, key_mx, torch.float32, output=score_workspace)
            probabilities_mx = _run_silu(
                scores,
                cu_seqlens_q,
                cu_seqlens_k,
                num_heads,
                max_seqlen_q,
                alpha,
                window_size_left,
                window_size_right,
                query_start=query_start,
                key_start=key_start,
                output=probability_workspace,
            )
            is_last_key_tile = key_tile_idx + 1 == len(k_mx_tiles)
            output = (
                dense_output[query_tile_idx]
                if is_last_key_tile
                else output_accumulator
            )
            _gemm(
                probabilities_mx,
                value_mx,
                torch.bfloat16 if is_last_key_tile else torch.float32,
                output=output,
                addend=output_accumulator if key_tile_idx > 0 else None,
            )

    return _unpack(dense_output, cu_seqlens_q, q.shape[0], num_heads), None


def hstu_varlen_bwd_mxfp8_100(
    dout: torch.Tensor,
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
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    target_group_size: int,
    window_size_left: int,
    window_size_right: int,
    alpha: float,
    rab: Optional[torch.Tensor],
    has_drab: bool,
    func: Optional[torch.Tensor],
    deterministic: bool,
):
    _check_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k)
    if dout.dtype != torch.bfloat16 or dout.shape != q.shape:
        raise ValueError("dout must be BF16 and have the same shape as q")
    if has_drab or any(x is not None for x in (num_contexts, num_targets, rab, func)):
        raise NotImplementedError(
            "SM100 MXFP8 HSTU currently supports causal, local, and full masks only"
        )
    q, k, v, dout = (x.contiguous() for x in (q, k, v, dout))
    num_heads = q.shape[1]
    padded_q = _round_up(max_seqlen_q, 128)
    padded_k = _round_up(max_seqlen_k, 128)
    dense_q = _pack(q, cu_seqlens_q, padded_q)
    dense_k = _pack(k, cu_seqlens_k, padded_k)
    dense_v = _pack(v, cu_seqlens_k, padded_k)
    dense_dout = _pack(dout, cu_seqlens_q, padded_q)

    window_size_left = _normalize_window(window_size_left, max_seqlen_k)
    window_size_right = _normalize_window(window_size_right, max_seqlen_k)
    batch_heads = (cu_seqlens_q.numel() - 1) * num_heads
    head_dim = q.shape[2]
    q_tiles = _tiles(dense_q)
    k_tiles = _tiles(dense_k)
    v_tiles = _tiles(dense_v)
    dout_tiles = _tiles(dense_dout)

    q_mx_tiles = _quantize_tiles(dense_q)
    qt_mx_tiles = _quantize_tiles(dense_q, transpose=True)
    k_mx_tiles = _quantize_tiles(dense_k)
    kt_mx_tiles = _quantize_tiles(dense_k, transpose=True)
    v_mx_tiles = _quantize_tiles(dense_v)
    dout_mx_tiles = _quantize_tiles(dense_dout)
    doutt_mx_tiles = _quantize_tiles(dense_dout, transpose=True)

    dq_accumulators = (
        [
            _matrix(128, head_dim, batch_heads, torch.float32, q.device)
            for _ in q_tiles
        ]
        if len(k_mx_tiles) > 1
        else [None] * len(q_tiles)
    )
    dense_dq = _tiled_matrix(
        len(q_tiles), 128, head_dim, batch_heads, torch.bfloat16, q.device
    )
    dense_dk = _tiled_matrix(
        len(k_tiles), 128, head_dim, batch_heads, torch.bfloat16, q.device
    )
    dense_dv = _tiled_matrix(
        len(k_tiles), 128, head_dim, batch_heads, torch.bfloat16, q.device
    )
    score_workspace = _matrix(128, 128, batch_heads, torch.float32, q.device)
    dprobability_workspace = _matrix(
        128, 128, batch_heads, torch.float32, q.device
    )
    silu_workspaces = tuple(
        _mx_workspace((128, 128, batch_heads), q.device) for _ in range(3)
    )
    dk_accumulator = (
        _matrix(128, head_dim, batch_heads, torch.float32, q.device)
        if len(q_mx_tiles) > 1
        else None
    )
    dv_accumulator = (
        _matrix(128, head_dim, batch_heads, torch.float32, q.device)
        if len(q_mx_tiles) > 1
        else None
    )

    for key_tile_idx, (key_mx, keyt_mx, value_mx) in enumerate(
        zip(k_mx_tiles, kt_mx_tiles, v_mx_tiles)
    ):
        key_start = key_tile_idx * 128

        for query_tile_idx, (
            query_mx,
            queryt_mx,
            dout_mx,
            doutt_mx,
        ) in enumerate(zip(q_mx_tiles, qt_mx_tiles, dout_mx_tiles, doutt_mx_tiles)):
            query_start = query_tile_idx * 128
            dscores_mx, probabilities_t_mx, dscores_t_mx = _run_backward_core(
                query_mx,
                key_mx,
                dout_mx,
                value_mx,
                score_workspace,
                dprobability_workspace,
                silu_workspaces,
                cu_seqlens_q,
                cu_seqlens_k,
                num_heads,
                max_seqlen_q,
                alpha,
                window_size_left,
                window_size_right,
                query_start,
                key_start,
            )

            is_last_query_tile = query_tile_idx + 1 == len(q_mx_tiles)
            gradient_dtype = (
                torch.bfloat16 if is_last_query_tile else torch.float32
            )
            _gemm(
                probabilities_t_mx,
                doutt_mx,
                gradient_dtype,
                output=(
                    dense_dv[key_tile_idx]
                    if is_last_query_tile
                    else dv_accumulator
                ),
                addend=dv_accumulator if query_tile_idx > 0 else None,
            )
            _gemm(
                dscores_t_mx,
                queryt_mx,
                gradient_dtype,
                output=(
                    dense_dk[key_tile_idx]
                    if is_last_query_tile
                    else dk_accumulator
                ),
                addend=dk_accumulator if query_tile_idx > 0 else None,
            )
            is_last_key_tile = key_tile_idx + 1 == len(k_mx_tiles)
            _gemm(
                dscores_mx,
                keyt_mx,
                torch.bfloat16 if is_last_key_tile else torch.float32,
                output=(
                    dense_dq[query_tile_idx]
                    if is_last_key_tile
                    else dq_accumulators[query_tile_idx]
                ),
                addend=(
                    dq_accumulators[query_tile_idx]
                    if key_tile_idx > 0
                    else None
                ),
            )

    dq_result = _unpack(dense_dq, cu_seqlens_q, q.shape[0], num_heads, output=dq)
    dk_result = _unpack(dense_dk, cu_seqlens_k, k.shape[0], num_heads, output=dk)
    dv_result = _unpack(dense_dv, cu_seqlens_k, v.shape[0], num_heads, output=dv)
    return dq_result, dk_result, dv_result, None
