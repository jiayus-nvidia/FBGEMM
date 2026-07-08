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

from .hstu_fwd import HSTUAttentionForwardSm100
from .mxfp8_blockscaled_gemm import Sm100BlockScaledPersistentDenseGemmKernel
from .mxfp8_quantize import (
    E4M3_MAX,
    MXFP8_BLOCK_SIZE,
    MxFp8QuantizeSm100,
    initialize_scale_storage_kernel,
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


class VarlenPackQuantizeSm100:
    def __init__(self, transpose: bool, initialize_padding: bool):
        self.transpose = transpose
        self.initialize_padding = initialize_padding

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        cu_seqlens: cute.Tensor,
        quantized: cute.Tensor,
        scale_storage: cute.Tensor,
        num_heads: Int32,
        stream: cuda.CUstream,
    ):
        scale_layout = blockscaled_utils.tile_atom_to_shape_SF(
            quantized.shape, MXFP8_BLOCK_SIZE
        )
        scales = cute.make_tensor(
            cute.recast_ptr(scale_storage.iterator, dtype=cutlass.Uint8),
            scale_layout,
        )
        raw_scales = cute.make_tensor(
            cute.recast_ptr(scale_storage.iterator, dtype=cutlass.Uint8),
            scale_storage.layout,
        )
        rows, reduction, batches = quantized.shape
        blocks_per_row = cute.ceil_div(reduction, MXFP8_BLOCK_SIZE)
        if cutlass.const_expr(self.initialize_padding):
            initialize_scale_storage_kernel(raw_scales).launch(
                grid=(cute.ceil_div(cute.size(raw_scales), 256), 1, 1),
                block=(256, 1, 1),
                stream=stream,
            )
        self.kernel(
            source,
            cu_seqlens,
            quantized,
            scales,
            num_heads,
            rows,
            reduction,
        ).launch(
            grid=(rows * blocks_per_row, batches, 1),
            block=(cute.arch.WARP_SIZE, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        cu_seqlens: cute.Tensor,
        quantized: cute.Tensor,
        scales: cute.Tensor,
        num_heads: Int32,
        rows: Int32,
        reduction: Int32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        block, output_batch, _ = cute.arch.block_idx()
        blocks_per_row = cute.ceil_div(reduction, MXFP8_BLOCK_SIZE)
        row = block // blocks_per_row
        reduction_block = block - row * blocks_per_row
        first_reduction_idx = reduction_block * MXFP8_BLOCK_SIZE + lane * 4

        batch_heads = (cute.size(cu_seqlens) - 1) * num_heads
        tile_idx = output_batch // batch_heads
        batch_head = output_batch - tile_idx * batch_heads
        batch = batch_head // num_heads
        head = batch_head - batch * num_heads
        offset = Int32(cu_seqlens[batch])
        length = Int32(cu_seqlens[batch + 1]) - offset

        values = cute.make_rmem_tensor((4,), Float32)
        local_amax = Float32(0.0)
        for value_idx in cutlass.range_constexpr(4):
            reduction_idx = first_reduction_idx + value_idx
            token_idx = tile_idx * 128 + row
            dim = reduction_idx
            if cutlass.const_expr(self.transpose):
                token_idx = tile_idx * 128 + reduction_idx
                dim = row
            value = Float32(0.0)
            if lane < 8 and reduction_idx < reduction and token_idx < length:
                value = Float32(source[offset + token_idx, head, dim])
            values[value_idx] = value
            local_amax = cute.arch.fmax(
                local_amax, cute.arch.fmax(value, -value)
            )

        block_amax = cute.arch.warp_reduction_max(local_amax)
        min_amax = 2.0**-126 * E4M3_MAX
        ratio = cute.arch.fmax(block_amax, min_amax) / E4M3_MAX
        log_scale = cute.math.log2(ratio)
        scale_exponent = Int32(log_scale)
        if log_scale > Float32(scale_exponent):
            scale_exponent += 1
        scale_exponent = cutlass.max(-126, cutlass.min(scale_exponent, 127))
        scale = cute.math.exp2(Float32(scale_exponent))

        if lane == 0:
            scales[row, reduction_block * MXFP8_BLOCK_SIZE, output_batch] = (
                cutlass.Uint8(scale_exponent + 127)
            )
        quantized_values = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
        quantized_values.store(
            (values.load() / scale).to(cutlass.Float8E4M3FN)
        )
        if lane < 8:
            for value_idx in cutlass.range_constexpr(4):
                reduction_idx = first_reduction_idx + value_idx
                if reduction_idx < reduction:
                    quantized[row, reduction_idx, output_batch] = (
                        quantized_values[value_idx]
                    )


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
        query_starts: Optional[cute.Tensor],
        key_starts: Optional[cute.Tensor],
        quantized: cute.Tensor,
        scale_storage: cute.Tensor,
        batch_heads_per_pair: Int32,
        key_tiles_per_query: Int32,
        query_starts_offset: Int32,
        key_starts_offset: Int32,
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
        self.kernel(
            scores,
            cu_seqlens_q,
            cu_seqlens_k,
            query_starts,
            key_starts,
            quantized,
            scales,
            batch_heads_per_pair,
            key_tiles_per_query,
            query_starts_offset,
            key_starts_offset,
            num_heads,
            normalization,
            alpha,
            window_left,
            window_right,
            query_start,
            key_start,
        ).launch(
            grid=(query_len, batch_heads, 1),
            block=(cute.arch.WARP_SIZE, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        scores: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        query_starts: Optional[cute.Tensor],
        key_starts: Optional[cute.Tensor],
        quantized: cute.Tensor,
        scales: cute.Tensor,
        batch_heads_per_pair: Int32,
        key_tiles_per_query: Int32,
        query_starts_offset: Int32,
        key_starts_offset: Int32,
        num_heads: Int32,
        normalization: Int32,
        alpha: Float32,
        window_left: Int32,
        window_right: Int32,
        query_start: Int32,
        key_start: Int32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        block, output_batch, _ = cute.arch.block_idx()
        query_len, key_len, _ = scores.shape
        blocks_per_row = cute.ceil_div(key_len, MXFP8_BLOCK_SIZE)
        query_idx_local = block
        batch_head = output_batch
        query_tile_start = query_start
        if cutlass.const_expr(query_starts is not None):
            pair_idx = output_batch // batch_heads_per_pair
            batch_head = output_batch - pair_idx * batch_heads_per_pair
            query_idx_in_chunk = pair_idx
            key_idx_in_chunk = Int32(0)
            if cutlass.const_expr(key_starts is not None):
                query_idx_in_chunk = pair_idx // key_tiles_per_query
                key_idx_in_chunk = pair_idx - query_idx_in_chunk * key_tiles_per_query
                key_start = Int32(
                    key_starts[key_starts_offset + key_idx_in_chunk]
                )
            query_tile_start = Int32(
                query_starts[query_starts_offset + query_idx_in_chunk]
            )
        query_idx = query_tile_start + query_idx_local
        batch = batch_head // num_heads
        query_length = Int32(cu_seqlens_q[batch + 1]) - Int32(cu_seqlens_q[batch])
        key_length = Int32(cu_seqlens_k[batch + 1]) - Int32(cu_seqlens_k[batch])

        for reduction_block in cutlass.range(blocks_per_row, unroll=1):
            key_idx_local = reduction_block * MXFP8_BLOCK_SIZE + lane
            key_idx = key_start + key_idx_local
            value = Float32(0.0)
            if key_idx_local < key_len and _valid_attention_position(
                query_idx,
                key_idx,
                query_length,
                key_length,
                window_left,
                window_right,
            ):
                scaled_score = (
                    Float32(
                        scores[query_idx_local, key_idx_local, output_batch]
                    )
                    * alpha
                )
                sigmoid = 1.0 / (1.0 + cute.math.exp(-scaled_score))
                value = scaled_score * sigmoid / Float32(normalization)

            block_amax = cute.arch.warp_reduction_max(
                cute.arch.fmax(value, -value)
            )
            ratio = (
                cute.arch.fmax(block_amax, 2.0**-126 * E4M3_MAX) / E4M3_MAX
            )
            log_scale = cute.math.log2(ratio)
            scale_exponent = Int32(log_scale)
            if log_scale > Float32(scale_exponent):
                scale_exponent += 1
            scale_exponent = cutlass.max(
                -126, cutlass.min(scale_exponent, 127)
            )
            scale = cute.math.exp2(Float32(scale_exponent))

            if lane == 0:
                scales[
                    query_idx_local,
                    reduction_block * MXFP8_BLOCK_SIZE,
                    output_batch,
                ] = cutlass.Uint8(scale_exponent + 127)
            first_lane = (lane // 4) * 4
            packed_value_0 = cute.arch.shuffle_sync(value, first_lane)
            packed_value_1 = cute.arch.shuffle_sync(value, first_lane + 1)
            packed_value_2 = cute.arch.shuffle_sync(value, first_lane + 2)
            packed_value_3 = cute.arch.shuffle_sync(value, first_lane + 3)
            if lane % 4 == 0:
                packed_values = cute.make_rmem_tensor((4,), Float32)
                packed_values[0] = packed_value_0
                packed_values[1] = packed_value_1
                packed_values[2] = packed_value_2
                packed_values[3] = packed_value_3
                quantized_values = cute.make_rmem_tensor(
                    (4,), cutlass.Float8E4M3FN
                )
                quantized_values.store(
                    (packed_values.load() / scale).to(cutlass.Float8E4M3FN)
                )
                for value_idx in cutlass.range_constexpr(4):
                    output_key_idx_local = key_idx_local + value_idx
                    if output_key_idx_local < key_len:
                        quantized[
                            query_idx_local,
                            output_key_idx_local,
                            output_batch,
                        ] = quantized_values[value_idx]


class SiluBackwardMaskQuantizeSm100:
    @cute.jit
    def __call__(
        self,
        scores: cute.Tensor,
        dprobabilities: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        query_starts: Optional[cute.Tensor],
        key_starts: Optional[cute.Tensor],
        quantized_dscores: cute.Tensor,
        dscore_scale_storage: cute.Tensor,
        quantized_probabilities_t: cute.Tensor,
        probability_t_scale_storage: cute.Tensor,
        quantized_dscores_t: cute.Tensor,
        dscore_t_scale_storage: cute.Tensor,
        batch_heads_per_pair: Int32,
        key_tiles_per_query: Int32,
        query_starts_offset: Int32,
        key_starts_offset: Int32,
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
            query_starts,
            key_starts,
            quantized_dscores,
            dscore_scales,
            quantized_probabilities_t,
            probability_t_scales,
            quantized_dscores_t,
            dscore_t_scales,
            batch_heads_per_pair,
            key_tiles_per_query,
            query_starts_offset,
            key_starts_offset,
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
        query_starts: Optional[cute.Tensor],
        key_starts: Optional[cute.Tensor],
        quantized_dscores: cute.Tensor,
        dscore_scales: cute.Tensor,
        quantized_probabilities_t: cute.Tensor,
        probability_t_scales: cute.Tensor,
        quantized_dscores_t: cute.Tensor,
        dscore_t_scales: cute.Tensor,
        batch_heads_per_pair: Int32,
        key_tiles_per_query: Int32,
        query_starts_offset: Int32,
        key_starts_offset: Int32,
        num_heads: Int32,
        normalization: Int32,
        alpha: Float32,
        window_left: Int32,
        window_right: Int32,
        query_start: Int32,
        key_start: Int32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        block, output_batch, _ = cute.arch.block_idx()
        query_len, key_len, _ = scores.shape
        blocks_per_row = cute.ceil_div(key_len, MXFP8_BLOCK_SIZE)
        batch_head = output_batch
        query_tile_start = query_start
        if cutlass.const_expr(query_starts is not None):
            pair_idx = output_batch // batch_heads_per_pair
            batch_head = output_batch - pair_idx * batch_heads_per_pair
            query_idx_in_chunk = pair_idx
            key_idx_in_chunk = Int32(0)
            if cutlass.const_expr(key_starts is not None):
                query_idx_in_chunk = pair_idx // key_tiles_per_query
                key_idx_in_chunk = pair_idx - query_idx_in_chunk * key_tiles_per_query
                key_start = Int32(
                    key_starts[key_starts_offset + key_idx_in_chunk]
                )
            query_tile_start = Int32(
                query_starts[query_starts_offset + query_idx_in_chunk]
            )
        batch = batch_head // num_heads
        query_length = Int32(cu_seqlens_q[batch + 1]) - Int32(cu_seqlens_q[batch])
        key_length = Int32(cu_seqlens_k[batch + 1]) - Int32(cu_seqlens_k[batch])
        row_block_count = query_len * blocks_per_row

        if block < row_block_count:
            query_idx_local = block // blocks_per_row
            reduction_block = block - query_idx_local * blocks_per_row
            first_key_idx_local = reduction_block * MXFP8_BLOCK_SIZE + lane * 4
            query_idx = query_tile_start + query_idx_local
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
                        Float32(scores[query_idx_local, key_idx_local, output_batch])
                        * alpha
                    )
                    sigmoid = 1.0 / (1.0 + cute.math.exp(-scaled_score))
                    derivative = sigmoid * (1.0 + scaled_score * (1.0 - sigmoid))
                    dscore = (
                        Float32(
                            dprobabilities[
                                query_idx_local, key_idx_local, output_batch
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
                    output_batch,
                ] = cutlass.Uint8(scale_exponent + 127)
            quantized_values = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
            quantized_values.store(
                (dscore_values.load() / scale).to(cutlass.Float8E4M3FN)
            )
            for value_idx in cutlass.range_constexpr(4):
                key_idx_local = first_key_idx_local + value_idx
                if lane < 8 and key_idx_local < key_len:
                    quantized_dscores[
                        query_idx_local, key_idx_local, output_batch
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
                query_idx = query_tile_start + query_idx_local
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
                        Float32(scores[query_idx_local, key_idx_local, output_batch])
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
                                query_idx_local, key_idx_local, output_batch
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
                    key_idx_local, scale_idx, output_batch
                ] = cutlass.Uint8(probability_scale_exponent + 127)
                dscore_t_scales[
                    key_idx_local, scale_idx, output_batch
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
                        key_idx_local, query_idx_local, output_batch
                    ] = quantized_probabilities[value_idx]
                    quantized_dscores_t[
                        key_idx_local, query_idx_local, output_batch
                    ] = quantized_dscores_t_values[value_idx]


class PairOutputReduceSm100:
    @cute.jit
    def __call__(
        self,
        partials: cute.Tensor,
        addend: cute.Tensor,
        output: cute.Tensor,
        batch_heads: Int32,
        key_tiles: Int32,
        accumulate: Int32,
        stream: cuda.CUstream,
    ):
        rows, columns, _ = partials.shape
        output_batches = cute.size(output.shape[2])
        self.kernel(
            partials,
            addend,
            output,
            batch_heads,
            key_tiles,
            accumulate,
        ).launch(
            grid=(cute.ceil_div(rows * columns, 256), output_batches, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        partials: cute.Tensor,
        addend: cute.Tensor,
        output: cute.Tensor,
        batch_heads: Int32,
        key_tiles: Int32,
        accumulate: Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, output_batch, _ = cute.arch.block_idx()
        rows, columns, _ = partials.shape
        linear_idx = block * 256 + thread
        if linear_idx < rows * columns:
            row = linear_idx // columns
            column = linear_idx - row * columns
            value = Float32(0.0)
            if accumulate != 0:
                value = Float32(addend[row, column, output_batch])
            query_idx = output_batch // batch_heads
            batch_head = output_batch - query_idx * batch_heads
            pair_base = query_idx * key_tiles * batch_heads + batch_head
            for key_idx in cutlass.range(key_tiles, unroll=1):
                value += Float32(
                    partials[
                        row,
                        column,
                        pair_base + key_idx * batch_heads,
                    ]
                )
            output[row, column, output_batch] = value.to(output.element_type)


class PairLinearReduceSm100:
    @cute.jit
    def __call__(
        self,
        partials: cute.Tensor,
        addend: cute.Tensor,
        output: cute.Tensor,
        accumulate: Int32,
        stream: cuda.CUstream,
    ):
        rows, columns, pair_batches = partials.shape
        output_batches = cute.size(output.shape[2])
        reduction_count = pair_batches // output_batches
        self.kernel(
            partials,
            addend,
            output,
            output_batches,
            reduction_count,
            accumulate,
        ).launch(
            grid=(cute.ceil_div(rows * columns, 256), output_batches, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        partials: cute.Tensor,
        addend: cute.Tensor,
        output: cute.Tensor,
        output_batches: Int32,
        reduction_count: Int32,
        accumulate: Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, output_batch, _ = cute.arch.block_idx()
        rows, columns, _ = partials.shape
        linear_idx = block * 256 + thread
        if linear_idx < rows * columns:
            row = linear_idx // columns
            column = linear_idx - row * columns
            value = Float32(0.0)
            if accumulate != 0:
                value = Float32(addend[row, column, output_batch])
            for pair_idx in cutlass.range(reduction_count, unroll=1):
                value += Float32(
                    partials[
                        row,
                        column,
                        pair_idx * output_batches + output_batch,
                    ]
                )
            output[row, column, output_batch] = value.to(output.element_type)


class PairGradientReduceSm100:
    @cute.jit
    def __call__(
        self,
        dk_pairs: cute.Tensor,
        dv_pairs: cute.Tensor,
        dk_addend: cute.Tensor,
        dv_addend: cute.Tensor,
        dk_output: cute.Tensor,
        dv_output: cute.Tensor,
        batch_heads_per_pair: Int32,
        accumulate: Int32,
        stream: cuda.CUstream,
    ):
        rows, columns, pair_batches = dk_pairs.shape
        self.kernel(
            dk_pairs,
            dv_pairs,
            dk_addend,
            dv_addend,
            dk_output,
            dv_output,
            batch_heads_per_pair,
            accumulate,
        ).launch(
            grid=(
                cute.ceil_div(rows * columns, 256),
                batch_heads_per_pair,
                1,
            ),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        dk_pairs: cute.Tensor,
        dv_pairs: cute.Tensor,
        dk_addend: cute.Tensor,
        dv_addend: cute.Tensor,
        dk_output: cute.Tensor,
        dv_output: cute.Tensor,
        batch_heads_per_pair: Int32,
        accumulate: Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, batch_head, _ = cute.arch.block_idx()
        rows, columns, pair_batches = dk_pairs.shape
        linear_idx = block * 256 + thread
        if linear_idx < rows * columns:
            row = linear_idx // columns
            column = linear_idx - row * columns
            dk_value = Float32(0.0)
            dv_value = Float32(0.0)
            if accumulate != 0:
                dk_value = Float32(dk_addend[row, column, batch_head])
                dv_value = Float32(dv_addend[row, column, batch_head])
            pair_count = pair_batches // batch_heads_per_pair
            for pair_idx in cutlass.range(pair_count, unroll=1):
                pair_batch = pair_idx * batch_heads_per_pair + batch_head
                dk_value += Float32(dk_pairs[row, column, pair_batch])
                dv_value += Float32(dv_pairs[row, column, pair_batch])
            dk_output[row, column, batch_head] = dk_value.to(
                dk_output.element_type
            )
            dv_output[row, column, batch_head] = dv_value.to(
                dv_output.element_type
            )


@dataclass
class _MxMatrix:
    values: torch.Tensor
    scales: torch.Tensor
    _cute_values: Optional[object] = None
    _cute_scales: Optional[object] = None


_compile_cache = {
    "pack": {},
    "pack_quantize": {},
    "unpack": {},
    "silu_quantize": {},
    "silu_bwd_quantize": {},
    "pair_output_reduce": {},
    "pair_linear_reduce": {},
    "pair_gradient_reduce": {},
    "quantize": {},
    "gemm": {},
    "fused_fwd": {},
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


def _pack_quantize(
    source: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padded_seqlen: int,
    *,
    transpose: bool = False,
) -> tuple[_MxMatrix, list[_MxMatrix]]:
    batches = cu_seqlens.numel() - 1
    num_heads, head_dim = source.shape[1:]
    tile_count = padded_seqlen // 128
    batch_heads = batches * num_heads
    rows, columns = (head_dim, 128) if transpose else (128, head_dim)
    combined_batches = tile_count * batch_heads
    values = _matrix(rows, columns, combined_batches, torch.uint8, source.device)
    scales = torch.empty(
        scale_factor_storage_size(rows, columns, combined_batches),
        dtype=torch.uint8,
        device=source.device,
    )
    source_tensor = from_dlpack(source.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=source.ndim - 1
    )
    offsets_tensor = from_dlpack(
        cu_seqlens.detach(), assumed_align=16
    ).mark_layout_dynamic(leading_dim=0)
    values_tensor = _mark_matrix(
        _cute_tensor(values, element_type=cutlass.Float8E4M3FN), 1
    )
    scales_tensor = _cute_tensor(scales, element_type=cutlass.Float8E8M0FNU)
    stream = cutlass_torch.default_stream()
    blocks_per_row = _round_up(columns, MXFP8_BLOCK_SIZE) // MXFP8_BLOCK_SIZE
    initialize_padding = rows % 128 != 0 or blocks_per_row % 4 != 0
    key = (
        source.dtype,
        padded_seqlen,
        head_dim,
        num_heads,
        batches,
        transpose,
        initialize_padding,
    )
    if key not in _compile_cache["pack_quantize"]:
        _compile_cache["pack_quantize"][key] = cute.compile(
            VarlenPackQuantizeSm100(transpose, initialize_padding),
            source_tensor,
            offsets_tensor,
            values_tensor,
            scales_tensor,
            Int32(num_heads),
            stream,
        )
    _compile_cache["pack_quantize"][key](
        source_tensor,
        offsets_tensor,
        values_tensor,
        scales_tensor,
        Int32(num_heads),
        stream,
    )
    combined = _MxMatrix(values, scales)
    scale_count = scale_factor_storage_size(rows, columns, batch_heads)
    tiles = [
        _MxMatrix(
            values[:, :, tile_idx * batch_heads : (tile_idx + 1) * batch_heads],
            scales[tile_idx * scale_count : (tile_idx + 1) * scale_count],
        )
        for tile_idx in range(tile_count)
    ]
    return combined, tiles


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


def _quantize(
    matrix: torch.Tensor, *, output_leading_dim: Optional[int] = None
) -> _MxMatrix:
    leading_dim = (
        _leading_dim(matrix)
        if output_leading_dim is None
        else output_leading_dim
    )
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
    blocks_per_row = (
        _round_up(matrix.shape[1], MXFP8_BLOCK_SIZE) // MXFP8_BLOCK_SIZE
    )
    initialize_padding = matrix.shape[0] % 128 != 0 or blocks_per_row % 4 != 0
    key = (
        matrix.dtype,
        tuple(matrix.shape),
        tuple(matrix.stride()),
        tuple(values.stride()),
        initialize_padding,
    )
    if key not in _compile_cache["quantize"]:
        _compile_cache["quantize"][key] = cute.compile(
            MxFp8QuantizeSm100(initialize_padding),
            source_tensor,
            values_tensor,
            scales_tensor,
            stream,
        )
    _compile_cache["quantize"][key](source_tensor, values_tensor, scales_tensor, stream)
    return _MxMatrix(values, scales)


def _quantize_tile_collection(
    tiled_matrix: torch.Tensor, *, transpose: bool = False
) -> tuple[_MxMatrix, list[_MxMatrix]]:
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
    tiles = [
        _MxMatrix(
            combined.values[:, :, tile_idx * batches : (tile_idx + 1) * batches],
            combined.scales[
                tile_idx * scale_count : (tile_idx + 1) * scale_count
            ],
        )
        for tile_idx in range(tile_count)
    ]
    return combined, tiles


def _quantize_tiles(
    tiled_matrix: torch.Tensor, *, transpose: bool = False
) -> list[_MxMatrix]:
    return _quantize_tile_collection(tiled_matrix, transpose=transpose)[1]


def _slice_mx_batches(matrix: _MxMatrix, start: int, end: int) -> _MxMatrix:
    batches = matrix.values.shape[2]
    if start < 0 or end > batches or start >= end:
        raise ValueError(f"invalid MXFP8 batch slice [{start}, {end})/{batches}")
    scales_per_batch = matrix.scales.numel() // batches
    return _MxMatrix(
        matrix.values[:, :, start:end],
        matrix.scales[start * scales_per_batch : end * scales_per_batch],
    )


def _gemm(
    a: _MxMatrix,
    b: _MxMatrix,
    output_dtype: torch.dtype,
    output: Optional[torch.Tensor] = None,
    addend: Optional[torch.Tensor] = None,
    broadcast_a_batches: bool = False,
    broadcast_b_batches: bool = False,
    batch_heads: int = 0,
    broadcast_b_group_size: int = 0,
) -> torch.Tensor:
    m, k, batches = a.values.shape
    n, b_k, b_batches = b.values.shape
    output_batches = batches
    if broadcast_a_batches:
        if not broadcast_b_batches or batch_heads <= 0:
            raise ValueError("broadcasting A requires B broadcasting and batch_heads")
        if batches % batch_heads != 0 or b_batches % batch_heads != 0:
            raise ValueError("MXFP8 batch counts must be divisible by batch_heads")
        output_batches = batches * b_batches // batch_heads
        compatible_batches = True
    elif broadcast_b_group_size > 0:
        compatible_batches = (
            batch_heads > 0
            and broadcast_b_group_size % batch_heads == 0
            and batches % broadcast_b_group_size == 0
            and batches // broadcast_b_group_size * batch_heads == b_batches
        )
    else:
        compatible_batches = (
            batches % b_batches == 0 if broadcast_b_batches else batches == b_batches
        )
    if k != b_k or not compatible_batches:
        raise ValueError(
            f"incompatible MXFP8 GEMM shapes {a.values.shape} and {b.values.shape}"
        )
    expected_shape = (m, n, output_batches)
    if output is None:
        output = _matrix(m, n, output_batches, output_dtype, a.values.device)
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
        broadcast_a_batches,
        broadcast_b_batches,
        batch_heads,
        broadcast_b_group_size,
        mma_n,
    )
    if key not in _compile_cache["gemm"]:
        gemm = Sm100BlockScaledPersistentDenseGemmKernel(
            32,
            (128, mma_n),
            (1, 1),
            accumulate_output=accumulate_output,
            broadcast_a_batches=broadcast_a_batches,
            broadcast_b_batches=broadcast_b_batches,
            batch_heads=batch_heads,
            broadcast_b_group_size=broadcast_b_group_size,
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
    query_starts: Optional[torch.Tensor] = None,
    key_starts: Optional[torch.Tensor] = None,
    key_tiles_per_query: Optional[int] = None,
    query_starts_offset: int = 0,
    key_starts_offset: int = 0,
    batch_heads_per_pair: Optional[int] = None,
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
    if query_starts is None:
        if key_starts is not None:
            raise ValueError("key_starts requires query_starts")
        query_starts_tensor = None
        key_starts_tensor = None
        key_tiles_per_query = 1
        batch_heads_per_pair = scores.shape[2]
    else:
        if query_starts.dtype != torch.int32 or query_starts.ndim != 1:
            raise ValueError("query_starts must be a 1-D int32 tensor")
        if batch_heads_per_pair is None or batch_heads_per_pair <= 0:
            raise ValueError("batched SiLU requires batch_heads_per_pair")
        query_starts_tensor = from_dlpack(
            query_starts.detach(), assumed_align=16
        ).mark_layout_dynamic(leading_dim=0)
        if key_starts is None:
            key_starts_tensor = None
            key_tiles_per_query = 1
        else:
            if key_starts.dtype != torch.int32 or key_starts.ndim != 1:
                raise ValueError("key_starts must be a 1-D int32 tensor")
            key_starts_tensor = from_dlpack(
                key_starts.detach(), assumed_align=16
            ).mark_layout_dynamic(leading_dim=0)
            if key_tiles_per_query is None:
                key_tiles_per_query = key_starts.numel()
            elif key_tiles_per_query <= 0:
                raise ValueError("key_tiles_per_query must be positive")
    stream = cutlass_torch.default_stream()
    key = (
        tuple(scores.shape),
        tuple(scores.stride()),
        num_heads,
        query_starts is not None,
        key_starts is not None,
        batch_heads_per_pair,
        key_tiles_per_query,
    )
    if key not in _compile_cache["silu_quantize"]:
        _compile_cache["silu_quantize"][key] = cute.compile(
            SiluMaskQuantizeSm100(),
            scores_tensor,
            offsets_q,
            offsets_k,
            query_starts_tensor,
            key_starts_tensor,
            values_tensor,
            scales_tensor,
            Int32(batch_heads_per_pair),
            Int32(key_tiles_per_query),
            Int32(query_starts_offset),
            Int32(key_starts_offset),
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
        query_starts_tensor,
        key_starts_tensor,
        values_tensor,
        scales_tensor,
        Int32(batch_heads_per_pair),
        Int32(key_tiles_per_query),
        Int32(query_starts_offset),
        Int32(key_starts_offset),
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
    query_starts: Optional[torch.Tensor] = None,
    key_starts: Optional[torch.Tensor] = None,
    key_tiles_per_query: Optional[int] = None,
    query_starts_offset: int = 0,
    key_starts_offset: int = 0,
    batch_heads_per_pair: Optional[int] = None,
    outputs: Optional[tuple[_MxMatrix, _MxMatrix, _MxMatrix]] = None,
):
    if scores.shape[:2] != (128, 128) or scores.dtype != torch.float32:
        raise ValueError("fused SiLU backward quantization requires 128x128 tiles")
    if (
        dprobabilities.shape != scores.shape
        or dprobabilities.dtype != torch.float32
    ):
        raise ValueError("dprobabilities must match the FP32 score tile")
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
    if query_starts is None:
        if key_starts is not None:
            raise ValueError("key_starts requires query_starts")
        query_starts_tensor = None
        key_starts_tensor = None
        key_tiles_per_query = 1
        batch_heads_per_pair = scores.shape[2]
    else:
        if query_starts.dtype != torch.int32 or query_starts.ndim != 1:
            raise ValueError("query_starts must be a 1-D int32 tensor")
        if batch_heads_per_pair is None or batch_heads_per_pair <= 0:
            raise ValueError("batched SiLU backward requires batch_heads_per_pair")
        query_starts_tensor = from_dlpack(
            query_starts.detach(), assumed_align=16
        ).mark_layout_dynamic(leading_dim=0)
        if key_starts is None:
            key_starts_tensor = None
            key_tiles_per_query = 1
        else:
            if key_starts.dtype != torch.int32 or key_starts.ndim != 1:
                raise ValueError("key_starts must be a 1-D int32 tensor")
            key_starts_tensor = from_dlpack(
                key_starts.detach(), assumed_align=16
            ).mark_layout_dynamic(leading_dim=0)
            if key_tiles_per_query is None:
                key_tiles_per_query = key_starts.numel()
            elif key_tiles_per_query <= 0:
                raise ValueError("key_tiles_per_query must be positive")
    stream = cutlass_torch.default_stream()
    key = (
        tuple(scores.shape),
        tuple(scores.stride()),
        tuple(dprobabilities.stride()),
        num_heads,
        query_starts is not None,
        key_starts is not None,
        batch_heads_per_pair,
        key_tiles_per_query,
    )
    if key not in _compile_cache["silu_bwd_quantize"]:
        _compile_cache["silu_bwd_quantize"][key] = cute.compile(
            SiluBackwardMaskQuantizeSm100(),
            score_tensor,
            dprobability_tensor,
            offsets_q,
            offsets_k,
            query_starts_tensor,
            key_starts_tensor,
            output_tensors[0],
            output_tensors[1],
            output_tensors[2],
            output_tensors[3],
            output_tensors[4],
            output_tensors[5],
            Int32(batch_heads_per_pair),
            Int32(key_tiles_per_query),
            Int32(query_starts_offset),
            Int32(key_starts_offset),
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
        query_starts_tensor,
        key_starts_tensor,
        output_tensors[0],
        output_tensors[1],
        output_tensors[2],
        output_tensors[3],
        output_tensors[4],
        output_tensors[5],
        Int32(batch_heads_per_pair),
        Int32(key_tiles_per_query),
        Int32(query_starts_offset),
        Int32(key_starts_offset),
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


def _reduce_pair_outputs(
    partials: torch.Tensor,
    batch_heads: int,
    key_tiles: int,
    output: torch.Tensor,
    addend: Optional[torch.Tensor] = None,
) -> None:
    if partials.dtype != torch.float32 or partials.ndim != 3:
        raise ValueError("partial outputs must be a 3-D FP32 tensor")
    rows, columns, pair_batches = partials.shape
    if pair_batches % (batch_heads * key_tiles) != 0:
        raise ValueError("partial output batches must contain complete key chunks")
    output_batches = pair_batches // key_tiles
    expected_shape = (rows, columns, output_batches)
    if output.shape != expected_shape:
        raise ValueError(
            f"invalid reduced output shape {output.shape}, expected {expected_shape}"
        )
    accumulate = addend is not None
    if addend is None:
        addend = output
    elif addend.shape != expected_shape or addend.dtype != torch.float32:
        raise ValueError("partial output addend must be a matching FP32 tensor")

    partials_tensor = _mark_matrix(_cute_tensor(partials), 1)
    addend_tensor = _mark_matrix(_cute_tensor(addend), 1)
    output_tensor = _mark_matrix(_cute_tensor(output), 1)
    stream = cutlass_torch.default_stream()
    key = (
        tuple(partials.shape),
        tuple(partials.stride()),
        batch_heads,
        key_tiles,
        addend.dtype,
        tuple(addend.stride()),
        output.dtype,
        tuple(output.stride()),
        accumulate,
    )
    if key not in _compile_cache["pair_output_reduce"]:
        _compile_cache["pair_output_reduce"][key] = cute.compile(
            PairOutputReduceSm100(),
            partials_tensor,
            addend_tensor,
            output_tensor,
            Int32(batch_heads),
            Int32(key_tiles),
            Int32(accumulate),
            stream,
        )
    _compile_cache["pair_output_reduce"][key](
        partials_tensor,
        addend_tensor,
        output_tensor,
        Int32(batch_heads),
        Int32(key_tiles),
        Int32(accumulate),
        stream,
    )


def _reduce_linear_pairs(
    partials: torch.Tensor,
    output: torch.Tensor,
    addend: Optional[torch.Tensor] = None,
) -> None:
    if partials.dtype != torch.float32 or partials.ndim != 3:
        raise ValueError("linear pair partials must be a 3-D FP32 tensor")
    if output.ndim != 3 or partials.shape[:2] != output.shape[:2]:
        raise ValueError("linear pair output must match partial rows and columns")
    if partials.shape[2] % output.shape[2] != 0:
        raise ValueError("linear pair batches must be divisible by output batches")
    accumulate = addend is not None
    if addend is None:
        addend = output
    elif addend.shape != output.shape or addend.dtype != torch.float32:
        raise ValueError("linear pair addend must be a matching FP32 tensor")

    partials_tensor = _mark_matrix(_cute_tensor(partials), 1)
    addend_tensor = _mark_matrix(_cute_tensor(addend), 1)
    output_tensor = _mark_matrix(_cute_tensor(output), 1)
    stream = cutlass_torch.default_stream()
    key = (
        tuple(partials.shape),
        tuple(partials.stride()),
        addend.dtype,
        tuple(addend.stride()),
        output.dtype,
        tuple(output.stride()),
        accumulate,
    )
    if key not in _compile_cache["pair_linear_reduce"]:
        _compile_cache["pair_linear_reduce"][key] = cute.compile(
            PairLinearReduceSm100(),
            partials_tensor,
            addend_tensor,
            output_tensor,
            Int32(accumulate),
            stream,
        )
    _compile_cache["pair_linear_reduce"][key](
        partials_tensor,
        addend_tensor,
        output_tensor,
        Int32(accumulate),
        stream,
    )


def _reduce_pair_gradients(
    dk_pairs: torch.Tensor,
    dv_pairs: torch.Tensor,
    batch_heads_per_pair: int,
    dk_output: torch.Tensor,
    dv_output: torch.Tensor,
    dk_addend: Optional[torch.Tensor] = None,
    dv_addend: Optional[torch.Tensor] = None,
) -> None:
    if (
        dk_pairs.shape != dv_pairs.shape
        or dk_pairs.dtype != torch.float32
        or dv_pairs.dtype != torch.float32
    ):
        raise ValueError("pair gradients must have matching FP32 shapes")
    rows, columns, pair_batches = dk_pairs.shape
    if pair_batches % batch_heads_per_pair != 0:
        raise ValueError("pair gradient batches must contain complete pairs")
    expected_output_shape = (rows, columns, batch_heads_per_pair)
    if (
        dk_output.shape != expected_output_shape
        or dv_output.shape != expected_output_shape
        or dk_output.dtype != dv_output.dtype
    ):
        raise ValueError("pair gradient outputs have incompatible shapes or dtypes")
    accumulate = dk_addend is not None or dv_addend is not None
    if accumulate:
        if dk_addend is None or dv_addend is None:
            raise ValueError("both pair gradient addends are required")
        if (
            dk_addend.shape != expected_output_shape
            or dv_addend.shape != expected_output_shape
            or dk_addend.dtype != torch.float32
            or dv_addend.dtype != torch.float32
        ):
            raise ValueError("pair gradient addends must be matching FP32 tensors")
    else:
        dk_addend = dk_output
        dv_addend = dv_output

    dk_pairs_tensor = _cute_tensor(dk_pairs)
    dv_pairs_tensor = _cute_tensor(dv_pairs)
    dk_addend_tensor = _cute_tensor(dk_addend)
    dv_addend_tensor = _cute_tensor(dv_addend)
    dk_output_tensor = _cute_tensor(dk_output)
    dv_output_tensor = _cute_tensor(dv_output)
    stream = cutlass_torch.default_stream()
    key = (
        tuple(dk_pairs.shape),
        tuple(dk_pairs.stride()),
        tuple(dk_addend.stride()),
        dk_addend.dtype,
        tuple(dk_output.stride()),
        dk_output.dtype,
        batch_heads_per_pair,
        accumulate,
    )
    if key not in _compile_cache["pair_gradient_reduce"]:
        _compile_cache["pair_gradient_reduce"][key] = cute.compile(
            PairGradientReduceSm100(),
            dk_pairs_tensor,
            dv_pairs_tensor,
            dk_addend_tensor,
            dv_addend_tensor,
            dk_output_tensor,
            dv_output_tensor,
            Int32(batch_heads_per_pair),
            Int32(accumulate),
            stream,
        )
    _compile_cache["pair_gradient_reduce"][key](
        dk_pairs_tensor,
        dv_pairs_tensor,
        dk_addend_tensor,
        dv_addend_tensor,
        dk_output_tensor,
        dv_output_tensor,
        Int32(batch_heads_per_pair),
        Int32(accumulate),
        stream,
    )


def _can_use_fused_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> bool:
    batches = cu_seqlens_q.numel() - 1
    return (
        q.shape[2] == 128
        and k.shape == v.shape
        and q.shape[1:] == k.shape[1:]
        and max_seqlen_q % 128 == 0
        and max_seqlen_k % 128 == 0
        and q.shape[0] == batches * max_seqlen_q
        and k.shape[0] == batches * max_seqlen_k
        and cu_seqlens_k.numel() == cu_seqlens_q.numel()
    )


def _run_fused_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    window_size_left: int,
    window_size_right: int,
    alpha: float,
) -> torch.Tensor:
    q_mx = _quantize(q.permute(0, 2, 1))
    k_mx = _quantize(k.permute(0, 2, 1))
    v_mx = _quantize(v.permute(2, 0, 1), output_leading_dim=1)
    q_values = q_mx.values.permute(0, 2, 1)
    k_values = k_mx.values.permute(0, 2, 1)
    v_values = v_mx.values
    output = torch.empty_like(q)

    q_tensor, k_tensor, output_tensor = [
        from_dlpack(tensor.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=2
        )
        for tensor in (q_values, k_values, output)
    ]
    v_tensor = from_dlpack(
        v_values.detach(), assumed_align=16
    ).mark_layout_dynamic(leading_dim=1)
    q_tensor.element_type = cutlass.Float8E4M3FN
    k_tensor.element_type = cutlass.Float8E4M3FN
    v_tensor.element_type = cutlass.Float8E4M3FN
    q_scale_tensor = _cute_tensor(
        q_mx.scales, element_type=cutlass.Float8E8M0FNU
    )
    k_scale_tensor = _cute_tensor(
        k_mx.scales, element_type=cutlass.Float8E8M0FNU
    )
    v_scale_tensor = _cute_tensor(
        v_mx.scales, element_type=cutlass.Float8E8M0FNU
    )
    cu_q_tensor = _cute_tensor(cu_seqlens_q)
    cu_k_tensor = _cute_tensor(cu_seqlens_k)
    stream = cutlass_torch.default_stream()

    is_causal = window_size_left == max_seqlen_k and window_size_right == 0
    is_local = (
        window_size_left < max_seqlen_k or window_size_right < max_seqlen_k
    ) and not is_causal
    key = (
        tuple(q.shape),
        tuple(k.shape),
        tuple(q_values.stride()),
        tuple(k_values.stride()),
        tuple(v_values.stride()),
        is_causal,
        is_local,
    )
    args = (
        q_tensor,
        k_tensor,
        v_tensor,
        output_tensor,
        Int32(max_seqlen_q),
        Int32(max_seqlen_k),
        cu_q_tensor,
        cu_k_tensor,
        None,
        None,
        Float32(alpha),
        stream,
        Int32(window_size_left),
        Int32(window_size_right),
        None,
        None,
        None,
        None,
        None,
        q_scale_tensor,
        k_scale_tensor,
        v_scale_tensor,
    )
    if key not in _compile_cache["fused_fwd"]:
        kernel = HSTUAttentionForwardSm100(
            head_dim=128,
            is_causal=is_causal,
            is_local=is_local,
            is_mxfp8=True,
        )
        _compile_cache["fused_fwd"][key] = cute.compile(kernel, *args)
    _compile_cache["fused_fwd"][key](*args)
    return output


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

    window_size_left = _normalize_window(window_size_left, max_seqlen_k)
    window_size_right = _normalize_window(window_size_right, max_seqlen_k)
    if _can_use_fused_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    ):
        return (
            _run_fused_forward(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                window_size_left,
                window_size_right,
                alpha,
            ),
            None,
        )
    batch_heads = (cu_seqlens_q.numel() - 1) * num_heads
    head_dim = q.shape[2]
    q_tile_count = padded_q // 128
    dense_output = _tiled_matrix(
        q_tile_count, 128, head_dim, batch_heads, torch.bfloat16, q.device
    )
    q_mx_collection, _ = _pack_quantize(q, cu_seqlens_q, padded_q)
    k_mx_collection, k_mx_tiles = _pack_quantize(k, cu_seqlens_k, padded_k)
    v_mx_collection, v_mx_tiles = _pack_quantize(
        v, cu_seqlens_k, padded_k, transpose=True
    )
    combined_batches = q_tile_count * batch_heads
    dense_output_matrix = dense_output.permute(1, 2, 0, 3).reshape(
        128, head_dim, combined_batches
    )
    output_accumulator = (
        _matrix(128, head_dim, combined_batches, torch.float32, q.device)
        if len(k_mx_tiles) > 1
        else None
    )
    query_starts = torch.arange(
        0,
        q_tile_count * 128,
        128,
        dtype=torch.int32,
        device=q.device,
    )
    key_starts = torch.arange(
        0,
        len(k_mx_tiles) * 128,
        128,
        dtype=torch.int32,
        device=q.device,
    )
    max_workspace_batches = 512
    pair_capacity = max(1, max_workspace_batches // batch_heads)
    query_chunk_size = min(q_tile_count, 16, pair_capacity)
    key_chunk_size = min(
        len(k_mx_tiles), max(1, pair_capacity // query_chunk_size)
    )
    workspaces = {}

    for query_tile_idx in range(0, q_tile_count, query_chunk_size):
        query_tile_end = min(query_tile_idx + query_chunk_size, q_tile_count)
        query_count = query_tile_end - query_tile_idx
        batch_start = query_tile_idx * batch_heads
        batch_end = query_tile_end * batch_heads
        query_mx = _slice_mx_batches(q_mx_collection, batch_start, batch_end)
        for key_tile_idx in range(0, len(k_mx_tiles), key_chunk_size):
            key_tile_end = min(key_tile_idx + key_chunk_size, len(k_mx_tiles))
            key_count = key_tile_end - key_tile_idx
            pair_count = query_count * key_count
            pair_batches = pair_count * batch_heads
            key_batch_start = key_tile_idx * batch_heads
            key_batch_end = key_tile_end * batch_heads
            key_mx = _slice_mx_batches(
                k_mx_collection, key_batch_start, key_batch_end
            )
            value_mx = _slice_mx_batches(
                v_mx_collection, key_batch_start, key_batch_end
            )
            workspace_key = (query_count, key_count)
            if workspace_key not in workspaces:
                workspaces[workspace_key] = (
                    _matrix(128, 128, pair_batches, torch.float32, q.device),
                    _mx_workspace((128, 128, pair_batches), q.device),
                    (
                        _matrix(
                            128,
                            head_dim,
                            pair_batches,
                            torch.float32,
                            q.device,
                        )
                        if len(k_mx_tiles) > 1
                        else None
                    ),
                )
            score_workspace, probability_workspace, partial_output_workspace = (
                workspaces[workspace_key]
            )
            scores = _gemm(
                query_mx,
                key_mx,
                torch.float32,
                output=score_workspace,
                broadcast_a_batches=len(k_mx_tiles) > 1,
                broadcast_b_batches=query_count > 1 or len(k_mx_tiles) > 1,
                batch_heads=batch_heads,
            )
            probabilities_mx = _run_silu(
                scores,
                cu_seqlens_q,
                cu_seqlens_k,
                num_heads,
                max_seqlen_q,
                alpha,
                window_size_left,
                window_size_right,
                query_starts=query_starts,
                key_starts=key_starts,
                key_tiles_per_query=key_count,
                query_starts_offset=query_tile_idx,
                key_starts_offset=key_tile_idx,
                batch_heads_per_pair=batch_heads,
                output=probability_workspace,
            )
            if len(k_mx_tiles) == 1:
                _gemm(
                    probabilities_mx,
                    value_mx,
                    torch.bfloat16,
                    output=dense_output_matrix[:, :, batch_start:batch_end],
                    broadcast_b_batches=query_count > 1,
                )
                continue
            partial_outputs = _gemm(
                probabilities_mx,
                value_mx,
                torch.float32,
                output=partial_output_workspace,
                broadcast_b_batches=query_count > 1,
            )
            is_last_key_chunk = key_tile_end == len(k_mx_tiles)
            output = (
                dense_output_matrix[:, :, batch_start:batch_end]
                if is_last_key_chunk
                else output_accumulator[:, :, batch_start:batch_end]
            )
            addend = (
                output_accumulator[:, :, batch_start:batch_end]
                if key_tile_idx > 0
                else None
            )
            _reduce_pair_outputs(
                partial_outputs,
                batch_heads,
                key_count,
                output,
                addend=addend,
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

    window_size_left = _normalize_window(window_size_left, max_seqlen_k)
    window_size_right = _normalize_window(window_size_right, max_seqlen_k)
    batch_heads = (cu_seqlens_q.numel() - 1) * num_heads
    head_dim = q.shape[2]
    q_tile_count = padded_q // 128
    k_tile_count = padded_k // 128

    q_mx_collection, _ = _pack_quantize(q, cu_seqlens_q, padded_q)
    qt_mx_collection, _ = _pack_quantize(
        q, cu_seqlens_q, padded_q, transpose=True
    )
    k_mx_collection, _ = _pack_quantize(k, cu_seqlens_k, padded_k)
    kt_mx_collection, _ = _pack_quantize(
        k, cu_seqlens_k, padded_k, transpose=True
    )
    v_mx_collection, _ = _pack_quantize(v, cu_seqlens_k, padded_k)
    dout_mx_collection, _ = _pack_quantize(dout, cu_seqlens_q, padded_q)
    doutt_mx_collection, _ = _pack_quantize(
        dout, cu_seqlens_q, padded_q, transpose=True
    )

    combined_batches = q_tile_count * batch_heads
    dense_dq = _tiled_matrix(
        q_tile_count, 128, head_dim, batch_heads, torch.bfloat16, q.device
    )
    dense_dk = _tiled_matrix(
        k_tile_count, 128, head_dim, batch_heads, torch.bfloat16, q.device
    )
    dense_dv = _tiled_matrix(
        k_tile_count, 128, head_dim, batch_heads, torch.bfloat16, q.device
    )
    dense_dq_matrix = dense_dq.permute(1, 2, 0, 3).reshape(
        128, head_dim, combined_batches
    )
    dq_accumulator = (
        _matrix(128, head_dim, combined_batches, torch.float32, q.device)
        if k_tile_count > 1
        else None
    )
    query_starts = torch.arange(
        0,
        q_tile_count * 128,
        128,
        dtype=torch.int32,
        device=q.device,
    )
    key_starts = torch.arange(
        0,
        k_tile_count * 128,
        128,
        dtype=torch.int32,
        device=q.device,
    )
    max_workspace_batches = 512
    pair_capacity = max(1, max_workspace_batches // batch_heads)
    query_chunk_size = min(q_tile_count, 16, pair_capacity)
    key_chunk_size = min(
        k_tile_count, max(1, pair_capacity // query_chunk_size)
    )
    workspaces = {}

    for key_tile_idx in range(0, k_tile_count, key_chunk_size):
        key_tile_end = min(key_tile_idx + key_chunk_size, k_tile_count)
        key_count = key_tile_end - key_tile_idx
        key_batch_start = key_tile_idx * batch_heads
        key_batch_end = key_tile_end * batch_heads
        key_mx = _slice_mx_batches(
            k_mx_collection, key_batch_start, key_batch_end
        )
        keyt_mx = _slice_mx_batches(
            kt_mx_collection, key_batch_start, key_batch_end
        )
        value_mx = _slice_mx_batches(
            v_mx_collection, key_batch_start, key_batch_end
        )

        for query_tile_idx in range(0, q_tile_count, query_chunk_size):
            query_tile_end = min(
                query_tile_idx + query_chunk_size, q_tile_count
            )
            query_count = query_tile_end - query_tile_idx
            pair_count = query_count * key_count
            pair_batches = pair_count * batch_heads
            batch_start = query_tile_idx * batch_heads
            batch_end = query_tile_end * batch_heads
            workspace_key = (query_count, key_count)
            if workspace_key not in workspaces:
                workspaces[workspace_key] = (
                    _matrix(128, 128, pair_batches, torch.float32, q.device),
                    _matrix(128, 128, pair_batches, torch.float32, q.device),
                    tuple(
                        _mx_workspace((128, 128, pair_batches), q.device)
                        for _ in range(3)
                    ),
                    _matrix(
                        128, head_dim, pair_batches, torch.float32, q.device
                    ),
                    _matrix(
                        128,
                        head_dim,
                        key_count * batch_heads,
                        torch.float32,
                        q.device,
                    ),
                    _matrix(
                        128,
                        head_dim,
                        key_count * batch_heads,
                        torch.float32,
                        q.device,
                    ),
                )
            (
                score_workspace,
                dprobability_workspace,
                silu_workspaces,
                pair_gradient_workspace,
                dk_accumulator,
                dv_accumulator,
            ) = workspaces[workspace_key]
            query_mx = _slice_mx_batches(
                q_mx_collection, batch_start, batch_end
            )
            dout_mx = _slice_mx_batches(
                dout_mx_collection, batch_start, batch_end
            )
            scores = _gemm(
                query_mx,
                key_mx,
                torch.float32,
                output=score_workspace,
                broadcast_a_batches=True,
                broadcast_b_batches=True,
                batch_heads=batch_heads,
            )
            dprobabilities = _gemm(
                dout_mx,
                value_mx,
                torch.float32,
                output=dprobability_workspace,
                broadcast_a_batches=True,
                broadcast_b_batches=True,
                batch_heads=batch_heads,
            )
            dscores_mx, probabilities_t_mx, dscores_t_mx = _run_silu_backward(
                scores,
                dprobabilities,
                cu_seqlens_q,
                cu_seqlens_k,
                num_heads,
                max_seqlen_q,
                alpha,
                window_size_left,
                window_size_right,
                query_starts=query_starts,
                key_starts=key_starts,
                key_tiles_per_query=key_count,
                query_starts_offset=query_tile_idx,
                key_starts_offset=key_tile_idx,
                batch_heads_per_pair=batch_heads,
                outputs=silu_workspaces,
            )

            if q_tile_count == 1 and k_tile_count == 1:
                _gemm(
                    dscores_mx,
                    keyt_mx,
                    torch.bfloat16,
                    output=dense_dq_matrix,
                )
                _gemm(
                    probabilities_t_mx,
                    doutt_mx_collection,
                    torch.bfloat16,
                    output=dense_dv[0],
                )
                _gemm(
                    dscores_t_mx,
                    qt_mx_collection,
                    torch.bfloat16,
                    output=dense_dk[0],
                )
                continue

            _gemm(
                dscores_mx,
                keyt_mx,
                torch.float32,
                output=pair_gradient_workspace,
                broadcast_b_batches=query_count > 1,
            )
            is_last_key_chunk = key_tile_end == k_tile_count
            dq_output = (
                dense_dq_matrix[:, :, batch_start:batch_end]
                if is_last_key_chunk
                else dq_accumulator[:, :, batch_start:batch_end]
            )
            _reduce_pair_outputs(
                pair_gradient_workspace,
                batch_heads,
                key_count,
                dq_output,
                addend=(
                    dq_accumulator[:, :, batch_start:batch_end]
                    if key_tile_idx > 0
                    else None
                ),
            )

            queryt_chunk = _slice_mx_batches(
                qt_mx_collection, batch_start, batch_end
            )
            doutt_chunk = _slice_mx_batches(
                doutt_mx_collection, batch_start, batch_end
            )
            is_last_query_chunk = query_tile_end == q_tile_count
            dk_output = (
                dense_dk[key_tile_idx:key_tile_end]
                .permute(1, 2, 0, 3)
                .reshape(128, head_dim, key_count * batch_heads)
                if is_last_query_chunk
                else dk_accumulator
            )
            dv_output = (
                dense_dv[key_tile_idx:key_tile_end]
                .permute(1, 2, 0, 3)
                .reshape(128, head_dim, key_count * batch_heads)
                if is_last_query_chunk
                else dv_accumulator
            )
            _gemm(
                probabilities_t_mx,
                doutt_chunk,
                torch.float32,
                output=pair_gradient_workspace,
                batch_heads=batch_heads,
                broadcast_b_group_size=key_count * batch_heads,
            )
            _reduce_linear_pairs(
                pair_gradient_workspace,
                dv_output,
                addend=dv_accumulator if query_tile_idx > 0 else None,
            )
            _gemm(
                dscores_t_mx,
                queryt_chunk,
                torch.float32,
                output=pair_gradient_workspace,
                batch_heads=batch_heads,
                broadcast_b_group_size=key_count * batch_heads,
            )
            _reduce_linear_pairs(
                pair_gradient_workspace,
                dk_output,
                addend=dk_accumulator if query_tile_idx > 0 else None,
            )

    dq_result = _unpack(dense_dq, cu_seqlens_q, q.shape[0], num_heads, output=dq)
    dk_result = _unpack(dense_dk, cu_seqlens_k, k.shape[0], num_heads, output=dk)
    dv_result = _unpack(dense_dv, cu_seqlens_k, v.shape[0], num_heads, output=dv)
    return dq_result, dk_result, dv_result, None
