# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.

import math

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.typing import Float32, Int32

MXFP8_BLOCK_SIZE = 32
E4M3_MAX = 448.0


@cute.kernel
def initialize_scale_storage_kernel(raw_scales: cute.Tensor):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    idx = block * 256 + thread
    if idx < cute.size(raw_scales):
        # 0x01 is 2**-126 in E8M0 and remains finite if a padded scale is
        # speculatively loaded by a 128-row scale-factor tile.
        raw_scales[idx] = cutlass.Uint8(1)


def scale_factor_storage_size(m: int, k: int, batch: int) -> int:
    """Return the number of E8M0 bytes required by tcgen05's SF layout."""
    scale_k = (k + MXFP8_BLOCK_SIZE - 1) // MXFP8_BLOCK_SIZE
    return math.ceil(m / 128) * math.ceil(scale_k / 4) * 128 * 4 * batch


class MxFp8QuantizeSm100:
    """Quantize an (M, K, L) tensor to OCP MXFP8 along its K dimension."""

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        quantized: cute.Tensor,
        scale_storage: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(quantized.element_type != cutlass.Float8E4M3FN):
            raise TypeError("MXFP8 values must use Float8E4M3FN")
        if cutlass.const_expr(scale_storage.element_type != cutlass.Float8E8M0FNU):
            raise TypeError("MXFP8 scale factors must use Float8E8M0FNU")

        scale_layout = blockscaled_utils.tile_atom_to_shape_SF(
            source.shape, MXFP8_BLOCK_SIZE
        )
        scales = cute.make_tensor(
            cute.recast_ptr(scale_storage.iterator, dtype=cutlass.Uint8),
            scale_layout,
        )
        raw_scales = cute.make_tensor(
            cute.recast_ptr(scale_storage.iterator, dtype=cutlass.Uint8),
            scale_storage.layout,
        )
        rows, reduction, batches = source.shape
        blocks_per_row = cute.ceil_div(reduction, MXFP8_BLOCK_SIZE)

        initialize_scale_storage_kernel(raw_scales).launch(
            grid=(cute.ceil_div(cute.size(raw_scales), 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )
        self.kernel(source, quantized, scales, rows, reduction).launch(
            grid=(rows * blocks_per_row, batches, 1),
            block=(cute.arch.WARP_SIZE, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        quantized: cute.Tensor,
        scales: cute.Tensor,
        rows: Int32,
        reduction: Int32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        block, batch, _ = cute.arch.block_idx()
        blocks_per_row = cute.ceil_div(reduction, MXFP8_BLOCK_SIZE)
        row = block // blocks_per_row
        reduction_block = block - row * blocks_per_row
        first_reduction_idx = reduction_block * MXFP8_BLOCK_SIZE + lane * 4

        values = cute.make_rmem_tensor((4,), Float32)
        local_amax = Float32(0.0)
        for value_idx in cutlass.range_constexpr(4):
            reduction_idx = first_reduction_idx + value_idx
            value = Float32(0.0)
            if lane < 8 and reduction_idx < reduction:
                value = Float32(source[row, reduction_idx, batch])
            values[value_idx] = value
            local_amax = cute.arch.fmax(local_amax, cute.arch.fmax(value, -value))

        block_amax = cute.arch.warp_reduction_max(local_amax)

        # E8M0 stores a power-of-two dequantization scale. The lower clamp also
        # gives an all-zero block a finite representable scale.
        min_amax = 2.0**-126 * E4M3_MAX
        ratio = cute.arch.fmax(block_amax, min_amax) / E4M3_MAX
        log_scale = cute.math.log2(ratio)
        scale_exponent = Int32(log_scale)
        if log_scale > Float32(scale_exponent):
            scale_exponent += 1
        scale_exponent = cutlass.max(-126, cutlass.min(scale_exponent, 127))
        scale = cute.math.exp2(Float32(scale_exponent))

        if lane == 0:
            # E8M0 encodes 2**(byte - 127). Writing the encoding directly also
            # avoids the vector-only f32 -> E8M0 conversion instruction.
            scales[row, reduction_block * MXFP8_BLOCK_SIZE, batch] = cutlass.Uint8(
                scale_exponent + 127
            )
        quantized_values = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
        quantized_values.store((values.load() / scale).to(cutlass.Float8E4M3FN))
        if lane < 8:
            for value_idx in cutlass.range_constexpr(4):
                reduction_idx = first_reduction_idx + value_idx
                if reduction_idx < reduction:
                    quantized[row, reduction_idx, batch] = quantized_values[value_idx]


class MxFp8QuantizePairSm100:
    """Quantize two same-shaped matrices in one launch."""

    @cute.jit
    def __call__(
        self,
        source_a: cute.Tensor,
        source_b: cute.Tensor,
        quantized_a: cute.Tensor,
        quantized_b: cute.Tensor,
        scale_storage_a: cute.Tensor,
        scale_storage_b: cute.Tensor,
        stream: cuda.CUstream,
    ):
        scale_layout = blockscaled_utils.tile_atom_to_shape_SF(
            source_a.shape, MXFP8_BLOCK_SIZE
        )
        scales_a = cute.make_tensor(
            cute.recast_ptr(scale_storage_a.iterator, dtype=cutlass.Uint8),
            scale_layout,
        )
        scales_b = cute.make_tensor(
            cute.recast_ptr(scale_storage_b.iterator, dtype=cutlass.Uint8),
            scale_layout,
        )
        rows, reduction, batches = source_a.shape
        blocks_per_row = cute.ceil_div(reduction, MXFP8_BLOCK_SIZE)
        self.kernel(
            source_a,
            source_b,
            quantized_a,
            quantized_b,
            scales_a,
            scales_b,
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
        source_a: cute.Tensor,
        source_b: cute.Tensor,
        quantized_a: cute.Tensor,
        quantized_b: cute.Tensor,
        scales_a: cute.Tensor,
        scales_b: cute.Tensor,
        rows: Int32,
        reduction: Int32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        block, batch, _ = cute.arch.block_idx()
        blocks_per_row = cute.ceil_div(reduction, MXFP8_BLOCK_SIZE)
        row = block // blocks_per_row
        reduction_block = block - row * blocks_per_row
        first_reduction_idx = reduction_block * MXFP8_BLOCK_SIZE + lane * 4

        values_a = cute.make_rmem_tensor((4,), Float32)
        values_b = cute.make_rmem_tensor((4,), Float32)
        local_amax_a = Float32(0.0)
        local_amax_b = Float32(0.0)
        for value_idx in cutlass.range_constexpr(4):
            reduction_idx = first_reduction_idx + value_idx
            value_a = Float32(0.0)
            value_b = Float32(0.0)
            if lane < 8 and reduction_idx < reduction:
                value_a = Float32(source_a[row, reduction_idx, batch])
                value_b = Float32(source_b[row, reduction_idx, batch])
            values_a[value_idx] = value_a
            values_b[value_idx] = value_b
            local_amax_a = cute.arch.fmax(
                local_amax_a, cute.arch.fmax(value_a, -value_a)
            )
            local_amax_b = cute.arch.fmax(
                local_amax_b, cute.arch.fmax(value_b, -value_b)
            )

        block_amax_a = cute.arch.warp_reduction_max(local_amax_a)
        block_amax_b = cute.arch.warp_reduction_max(local_amax_b)
        min_amax = 2.0**-126 * E4M3_MAX
        log_scale_a = cute.math.log2(
            cute.arch.fmax(block_amax_a, min_amax) / E4M3_MAX
        )
        log_scale_b = cute.math.log2(
            cute.arch.fmax(block_amax_b, min_amax) / E4M3_MAX
        )
        scale_exponent_a = Int32(log_scale_a)
        scale_exponent_b = Int32(log_scale_b)
        if log_scale_a > Float32(scale_exponent_a):
            scale_exponent_a += 1
        if log_scale_b > Float32(scale_exponent_b):
            scale_exponent_b += 1
        scale_exponent_a = cutlass.max(-126, cutlass.min(scale_exponent_a, 127))
        scale_exponent_b = cutlass.max(-126, cutlass.min(scale_exponent_b, 127))
        scale_a = cute.math.exp2(Float32(scale_exponent_a))
        scale_b = cute.math.exp2(Float32(scale_exponent_b))

        if lane == 0:
            scale_idx = reduction_block * MXFP8_BLOCK_SIZE
            scales_a[row, scale_idx, batch] = cutlass.Uint8(scale_exponent_a + 127)
            scales_b[row, scale_idx, batch] = cutlass.Uint8(scale_exponent_b + 127)

        quantized_values_a = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
        quantized_values_b = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
        quantized_values_a.store(
            (values_a.load() / scale_a).to(cutlass.Float8E4M3FN)
        )
        quantized_values_b.store(
            (values_b.load() / scale_b).to(cutlass.Float8E4M3FN)
        )
        if lane < 8:
            for value_idx in cutlass.range_constexpr(4):
                reduction_idx = first_reduction_idx + value_idx
                if reduction_idx < reduction:
                    quantized_a[row, reduction_idx, batch] = quantized_values_a[
                        value_idx
                    ]
                    quantized_b[row, reduction_idx, batch] = quantized_values_b[
                        value_idx
                    ]


class MxFp8DequantizeSm100:
    """Dequantize an MXFP8 tensor, primarily for numerical validation."""

    @cute.jit
    def __call__(
        self,
        quantized: cute.Tensor,
        scale_storage: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        scale_layout = blockscaled_utils.tile_atom_to_shape_SF(
            quantized.shape, MXFP8_BLOCK_SIZE
        )
        scales = cute.make_tensor(
            cute.recast_ptr(scale_storage.iterator, dtype=cutlass.Uint8),
            scale_layout,
        )
        rows, reduction, batches = quantized.shape
        threads = 128
        self.kernel(quantized, scales, output, rows, reduction).launch(
            grid=(cute.ceil_div(rows * reduction, threads * 4), batches, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        quantized: cute.Tensor,
        scales: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        reduction: Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, batch, _ = cute.arch.block_idx()
        first_linear_idx = (block * 128 + thread) * 4
        if first_linear_idx < rows * reduction:
            row = first_linear_idx // reduction
            first_reduction_idx = first_linear_idx - row * reduction
            quantized_values = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
            for value_idx in cutlass.range_constexpr(4):
                quantized_values[value_idx] = quantized[
                    row, first_reduction_idx + value_idx, batch
                ]
            values = quantized_values.load().to(Float32)
            scale_idx = (first_reduction_idx // MXFP8_BLOCK_SIZE) * MXFP8_BLOCK_SIZE
            scale_exponent = Int32(scales[row, scale_idx, batch]) - 127
            scale = cute.math.exp2(Float32(scale_exponent))
            for value_idx in cutlass.range_constexpr(4):
                output[row, first_reduction_idx + value_idx, batch] = (
                    values[value_idx] * scale
                )
