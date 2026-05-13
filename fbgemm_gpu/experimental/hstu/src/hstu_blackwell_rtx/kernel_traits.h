/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Blackwell RTX kernel traits for HSTU forward pass.
// Uses per-warp mma.sync (same model as Ampere), NOT warpgroup MMA (Hopper).
// Provides BF16 and FP8 MMA atom selection.

#pragma once

#include <cutlass/numeric_types.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"

#include "cute/atom/mma_traits_sm90_gmma.hpp"  // for GMMA::Layout_K_SW128_Atom / Layout_MN_SW128_Atom

using namespace cute;

// Base kernel traits for Blackwell RTX.
// For BF16: uses SM80 mma.sync atom (available on Blackwell RTX via backward compat).
// For FP8: uses Blackwell RTX f8f6f4 mma.sync atom when CUTE_ARCH_F8F6F4_MMA_ENABLED.
template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kNWarps_,
    typename elem_type = cutlass::bfloat16_t>
struct Flash_kernel_traits_blackwell_rtx {
  using Element = elem_type;
  using ElementAccum = float;
  using index_t = int64_t;

  using MMA_Atom_Arch = std::conditional_t<
      std::is_same_v<elem_type, cutlass::half_t>,
      MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
};

// Blackwell RTX BF16 forward kernel traits — identical structure to Ampere.
// Compiled with -gencode arch=compute_120,code=sm_120 for native Blackwell execution.
template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kNWarps_,
    bool Is_causal_,
    bool Is_target_,
    bool Is_context_,
    bool Is_local_,
    bool Is_arbitrary_,
    int kNFunc_,
    bool Has_rab_,
    bool Is_Q_in_regs_ = false,
    bool Share_Q_K_smem_ = false,
    typename elem_type = cutlass::bfloat16_t,
    typename Base =
        Flash_kernel_traits_blackwell_rtx<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type>>
struct Hstu_fwd_kernel_traits_blackwell_rtx : public Base {
  static constexpr bool Is_causal = Is_causal_;
  static constexpr bool Is_target = Is_target_;
  static constexpr bool Is_context = Is_context_;
  static constexpr bool Is_local = Is_local_;
  static constexpr bool Is_arbitrary = Is_arbitrary_;
  static constexpr int kNFunc = Is_arbitrary_ ? kNFunc_ : 0;
  static constexpr bool Has_rab = Has_rab_;
  static constexpr bool Paged_KV = false;  // Not supported in first Blackwell impl
  static constexpr bool Is_fp8 = false;

  using Element = typename Base::Element;
  // For BF16 path: ElementSmem == Element (no conversion needed)
  using ElementSmem = Element;
  // For BF16 path: OutputType == Element (output same dtype as input)
  using OutputType = Element;
  using ElementAccum = typename Base::ElementAccum;
  using index_t = typename Base::index_t;
  using SmemCopyAtom = typename Base::SmemCopyAtom;
  using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

  static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
  static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static_assert(kHeadDim % 32 == 0);
  static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
  static constexpr int kBlockKSmemRab = kBlockN % 64 == 0 ? 64 : 32;
  static constexpr int kBlockKGmem =
      kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
  static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
  static constexpr int kSwizzleRab = kBlockKSmemRab == 32 ? 2 : 3;
  static constexpr int kStages = 1;

  using TiledMma = TiledMMA<
      typename Base::MMA_Atom_Arch,
      Layout<Shape<Int<kNWarps>, _1, _1>>,
      Tile<Int<16 * kNWarps>, _16, _16>>;
  static_assert(16 * kNWarps <= kBlockM);

  using SmemLayoutAtomQ = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));

  using SmemLayoutAtomRab = decltype(composition(
      Swizzle<kSwizzleRab, 3, 3>{},
      Layout<Shape<_8, Int<kBlockKSmemRab>>, Stride<Int<kBlockKSmemRab>, _1>>{}));

  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtomQ{},
      Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutRab = decltype(tile_to_shape(
      SmemLayoutAtomRab{},
      Shape<Int<kBlockM>, Int<kBlockN>, Int<kStages>>{}));
  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtomQ{},
      Shape<Int<kBlockN>, Int<kHeadDim>, Int<kStages>>{}));

  using SmemLayoutVtransposed = decltype(composition(
      SmemLayoutKV{},
      make_layout(Shape<Int<kHeadDim>, Int<kBlockN>, Int<kStages>>{},
          Stride<Int<kBlockN>, _1, Int<kHeadDim * kBlockN>>{})));
  using SmemLayoutVtransposedNoSwizzle =
      decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

  using SmemLayoutAtomO = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      Layout<Shape<Int<8>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutO = decltype(tile_to_shape(
      SmemLayoutAtomO{},
      Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopy, Element>;

  static constexpr int MaxSeqLenK = 64 * 1024;
  static constexpr int MaxValidBlock = MaxSeqLenK / kBlockN;
  using SmemLayoutValidBlockIds = Layout<Shape<Int<MaxValidBlock>>, Stride<_1>>;
  using SmemLayoutMaxFunc = Layout<Shape<Int<kNFunc/2 + 1>>, Stride<_1>>;
  using SmemLayoutMinFunc = Layout<Shape<Int<kNFunc/2 + 1>>, Stride<_1>>;

  static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
  static constexpr int kSmemRabSize =
      Has_rab ? size(SmemLayoutRab{}) * sizeof(Element) : 0;
  static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
  static constexpr int kSmemSizeQKV = Share_Q_K_smem
      ? std::max(kSmemQSize, kSmemKVSize)
      : kSmemQSize + kSmemKVSize;
  static constexpr int kSmemSizeQKVRab = kSmemSizeQKV + kSmemRabSize;
  static constexpr int kSmemSizeQKVRabValidBlockIds = kSmemSizeQKVRab +
      (Is_arbitrary ? size(SmemLayoutValidBlockIds{}) * sizeof(int) : 0);
  static constexpr int kSmemSize = kSmemSizeQKVRabValidBlockIds +
      (Is_arbitrary ? (size(SmemLayoutMaxFunc{}) + size(SmemLayoutMinFunc{}) + 1) * sizeof(int) : 0);

  static constexpr int kGmemElemsPerLoad =
      sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(kHeadDim % kGmemElemsPerLoad == 0);
  static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
  static constexpr int kGmemThreadsPerRowRab = kBlockKSmemRab / kGmemElemsPerLoad;
  static_assert(kNThreads % kGmemThreadsPerRow == 0);
  static_assert(kNThreads % kGmemThreadsPerRowRab == 0);

  using GmemLayoutAtom = Layout<
      Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemLayoutAtomRab = Layout<
      Shape<Int<kNThreads / kGmemThreadsPerRowRab>, Int<kGmemThreadsPerRowRab>>,
      Stride<Int<kGmemThreadsPerRowRab>, _1>>;

  using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using GmemTiledCopyQKV = decltype(make_tiled_copy(
      Copy_Atom<Gmem_copy_struct, Element>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, _8>>{}));

  // rab_row_size: M-values per thread per atom call; must cover exactly kBlockM rows total.
  static constexpr int rab_row_size = Has_rab
      ? kBlockM * kGmemThreadsPerRowRab / kNThreads
      : 1;
  using GmemTiledCopyRab = decltype(make_tiled_copy(
      Copy_Atom<Gmem_copy_struct, Element>{},
      GmemLayoutAtomRab{},
      Layout<Shape<Int<rab_row_size>, _8>, Stride<_8, _1>>{}));

  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, Element>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, _8>>{}));
};

// Common Blackwell RTX FP8 forward kernel traits.
// Q, K, V are FP8 (e4m3) in both GMEM and SMEM. The WS TMA kernel consumes
// these layouts and applies block-scale factors in Blackwell RTX QMMA.
template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kNWarps_,
    bool Is_causal_,
    bool Is_target_,
    bool Is_context_,
    bool Is_local_,
    bool Is_arbitrary_,
    int kNFunc_,
    bool Has_rab_,
    bool Paged_KV_ = false,
    bool Is_Q_in_regs_ = false,
    bool Share_Q_K_smem_ = false,
    typename out_type = cutlass::bfloat16_t>
struct Hstu_fwd_kernel_traits_blackwell_rtx_fp8 {
  // Common Blackwell RTX FP8 metadata used by the WS TMA kernel.
  // All Blackwell RTX FP8 forward dispatch routes through Hstu_fwd_kernel_traits_blackwell_rtx_fp8_ws.
  // The legacy non-WS FP8 launch path has been removed.
  using Element = cutlass::float_e4m3_t;
  using ElementSmem = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using OutputType = out_type;
  using index_t = int64_t;

  static constexpr bool Is_causal = Is_causal_;
  static constexpr bool Is_target = Is_target_;
  static constexpr bool Is_context = Is_context_;
  static constexpr bool Is_local = Is_local_;
  static constexpr bool Is_arbitrary = Is_arbitrary_;
  static constexpr int kNFunc = Is_arbitrary_ ? kNFunc_ : 0;
  static constexpr bool Has_rab = Has_rab_;
  static constexpr bool Paged_KV = Paged_KV_;
  static constexpr bool Is_fp8 = true;

  // Note: MMA_Atom_Arch / TiledMma / SmemCopyAtom are intentionally absent here.
  // The WS kernel uses BlackwellRtxQmmaBuilder (BS1/BS2) for MMA and s2r copies directly.

  static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
  static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static_assert(kHeadDim % 32 == 0);
  // kBlockKSmem=32: one 16×32 FP8 atom aligns with Blackwell RTX QMMA K=32 per step.
  static constexpr int kBlockKSmem = 32;
  static constexpr int kBlockKSmemRab = kBlockN % 64 == 0 ? 64 : 32;
  static constexpr int kBlockKGmem =
      kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
  static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
  // RAB is always BF16: use 64-BF16 strip with Swizzle<3,3,3> (8 BF16 contiguous = 128 bits)
  static constexpr int kSwizzleRab = kBlockKSmemRab == 32 ? 2 : 3;
  static constexpr int kStages = 1;

  // SMEM layout for Q/K/V: flat 16×32 atom (kBlockKSmem=32).
  // 16-row × 32-col FP8 tile = 512 bytes; aligns with Blackwell RTX QMMA K=32 per step.
  using SmemLayoutAtomQ =
      Layout<Shape<_16, _32>, Stride<_32, _1>>;

  using SmemLayoutAtomRab = decltype(composition(
      Swizzle<kSwizzleRab, 3, 3>{},
      Layout<Shape<_8, Int<kBlockKSmemRab>>, Stride<Int<kBlockKSmemRab>, _1>>{}));

  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtomQ{},
      Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutRab = decltype(tile_to_shape(
      SmemLayoutAtomRab{},
      Shape<Int<kBlockM>, Int<kBlockN>, Int<kStages>>{}));
  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtomQ{},
      Shape<Int<kBlockN>, Int<kHeadDim>, Int<kStages>>{}));

  using SmemLayoutVtransposed = decltype(composition(
      SmemLayoutKV{},
      make_layout(Shape<Int<kHeadDim>, Int<kBlockN>, Int<kStages>>{},
          Stride<Int<kBlockN>, _1, Int<kHeadDim * kBlockN>>{})));
  using SmemLayoutVtransposedNoSwizzle =
      decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

  // WS TMA layouts: choose the smallest GMMA K-swizzle atom that matches the
  // contiguous head-dim tile.
  using SmemLayoutAtomTMA = std::conditional_t<
      (kHeadDim == 32),
      GMMA::Layout_K_SW32_Atom<Element>,
      std::conditional_t<
          (kHeadDim == 64),
          GMMA::Layout_K_SW64_Atom<Element>,
          GMMA::Layout_K_SW128_Atom<Element>>>;
  using SmemLayoutQ_TMA  = decltype(tile_to_shape(SmemLayoutAtomTMA{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutK_TMA  = decltype(tile_to_shape(SmemLayoutAtomTMA{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
  // V row-major [kBlockN, kHeadDim] in SMEM, loaded via TMA from row-major V (d stride-1).
  // K_SW128_Atom: dim1 = kHeadDim (d axis) is the SMEM fast axis; dim0 = kBlockN (n_k, token).
  // Physical byte: physical(n_k, d) = n_k * kHeadDim + (d ^ ((n_k & 7) << 4)).
  // LDSM_T address for n_k row, d-group: addr = v_base + n_k * kHeadDim + (d_start ^ ((n_k & 7) << 4)). ✓
  using SmemLayoutVt_TMA = decltype(tile_to_shape(
      SmemLayoutAtomTMA{},
      Shape<Int<kBlockN>, Int<kHeadDim>>{}));

  // Output layout: BF16 written to sO (reuses smem_ base), then copied to GMEM.
  // Blackwell RTX QMMA output uses PermMmaTileN = Layout<_8,_4,_4>, Stride<_1,_32,_8>, which
  // permutes the N-axis of the C fragment. The interleaved atom below matches this permutation:
  // within each 8-element group (cols 0..7, 8..15, etc.), elements are consecutive in physical
  // SMEM (stride-1 within group, stride-64 between groups). This enables AutoVectorizingCopy
  // to perform 8-element (128-bit) stores correctly.
  // Reference: Blackwell block-scaled MoE GEMM SmemAtomLayoutO.
  // NOTE: kBlockKSmem=32 is for FP8 INPUT; output is BF16 with 64-wide SW128 atom.
  using SmemLayoutAtomO = std::conditional_t<
      (kHeadDim == 32),
      Layout<Shape<_8, _32>, Stride<_32, _1>>,
      decltype(composition(
          Swizzle<3, 3, 3>{},
          Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, Int<64>>>>{}))>;
  using SmemLayoutO = decltype(tile_to_shape(
      SmemLayoutAtomO{},
      Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopy, OutputType>;

  static constexpr int MaxSeqLenK = 64 * 1024;
  static constexpr int MaxValidBlock = MaxSeqLenK / kBlockN;
  using SmemLayoutValidBlockIds = Layout<Shape<Int<MaxValidBlock>>, Stride<_1>>;
  using SmemLayoutMaxFunc = Layout<Shape<Int<kNFunc/2 + 1>>, Stride<_1>>;
  using SmemLayoutMinFunc = Layout<Shape<Int<kNFunc/2 + 1>>, Stride<_1>>;

  // SMEM sizes: FP8 is 1 byte each (half of BF16), allowing kNWarps=8
  static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
  static constexpr int kSmemRabSize =
      Has_rab ? size(SmemLayoutRab{}) * sizeof(cutlass::bfloat16_t) : 0;  // RAB is always BF16
  static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
  // FP8 epilogue writes BF16 output into sO which reuses smem_ base.
  // sO needs size(SmemLayoutO) * sizeof(OutputType) bytes (BF16 = 2x FP8).
  // kSmemSizeQKV must be >= kSmemOSize to avoid OOB SMEM access in epilogue.
  static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(OutputType);
  // Phase 3 single-buffer V layout (when Share_Q_K_smem=true):
  //   [0, kSmemQSize)       Q = K  (shared, single buffer)
  //   [kSmemQSize, 2x)      V (single buffer)
  // kSmemOSize (BF16 epilogue) = kBlockM*kHeadDim*2 = 128*128*2 = 32768 = 2*kSmemQSize,
  // so output can reuse smem_q safely.
  static constexpr int kSmemSizeQKV = Share_Q_K_smem
      ? std::max(2 * kSmemQSize, kSmemOSize)   // Q=K(0) + V(1), single-buf V
      : std::max(kSmemQSize + kSmemKVSize, kSmemOSize);
  static constexpr int kSmemSizeQKVRab = kSmemSizeQKV + kSmemRabSize;
  static constexpr int kSmemSizeQKVRabValidBlockIds = kSmemSizeQKVRab +
      (Is_arbitrary ? size(SmemLayoutValidBlockIds{}) * sizeof(int) : 0);
  // Extra SMEM for block-scale SF: SFA (kBlockM int32 = 512B) + SFB (kBlockN int32 = 512B)
  static constexpr int kSmemSFSize = 1024;
  // Extra SMEM for TMA barrier(s). The base FP8 layout keeps one barrier slot;
  // WS traits define their own producer/consumer mbarrier area below.
  static constexpr int kSmemMbarSize = 8;
  static constexpr int kSmemSize = kSmemSizeQKVRabValidBlockIds +
      (Is_arbitrary ? (size(SmemLayoutMaxFunc{}) + size(SmemLayoutMinFunc{}) + 1) * sizeof(int) : 0) +
      kSmemSFSize + kSmemMbarSize;

  // GMEM copy: cp.async FP8 elements directly into flat SMEM (no conversion).
  // Uses SM80_CP_ASYNC_CACHEGLOBAL<uint128_t> (16 bytes = 16 FP8 per thread per load).
  // Flat SMEM layout has stride-1 in K → 16 consecutive FP8 are physically contiguous. ✓
  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);  // = 16 for FP8
  static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;  // = 32/16 = 2 for FP8
  // RAB is always BF16: 8 BF16 per uint128_t (16 bytes)
  static constexpr int kGmemElemsPerLoadRab = sizeof(cute::uint128_t) / sizeof(cutlass::bfloat16_t);
  static constexpr int kGmemThreadsPerRowRab = kBlockKSmemRab / kGmemElemsPerLoadRab;
  static_assert(kHeadDim % kGmemElemsPerLoad == 0);
  static_assert(kNThreads % kGmemThreadsPerRow == 0);
  static_assert(kNThreads % kGmemThreadsPerRowRab == 0);

  using GmemLayoutAtom = Layout<
      Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemLayoutAtomRab = Layout<
      Shape<Int<kNThreads / kGmemThreadsPerRowRab>, Int<kGmemThreadsPerRowRab>>,
      Stride<Int<kGmemThreadsPerRowRab>, _1>>;

  // cp.async for FP8 Q/K/V: loads 16 FP8 (128-bit) directly into flat FP8 SMEM per thread.
  // CACHEGLOBAL<uint128_t> issues `cp.async.cg` (L1-bypass), accepts 16-byte transfers.
  using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using GmemTiledCopyQKV = decltype(make_tiled_copy(
      Copy_Atom<Gmem_copy_struct, Element>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // 16 FP8 per thread

  // RAB is BF16 (not quantized)
  using Gmem_copy_struct_rab = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  // rab_row_size: M-values per thread per atom call in the RAB copy.
  // Must satisfy: (kNThreads / kGmemThreadsPerRowRab) * rab_row_size == kBlockM
  // so the tiled copy covers exactly [kBlockM, kBlockN] BF16 elements without
  // threads writing beyond kBlockM rows (which would corrupt SMEM past sRab).
  // Formula: kBlockM * kGmemThreadsPerRowRab / kNThreads
  // For kBlockM=128, kNThreads=256, kGmemThreadsPerRowRab=8: 128*8/256 = 4. ✓
  static constexpr int rab_row_size = Has_rab
      ? kBlockM * kGmemThreadsPerRowRab / kNThreads
      : 1;
  using GmemTiledCopyRab = decltype(make_tiled_copy(
      Copy_Atom<Gmem_copy_struct_rab, cutlass::bfloat16_t>{},
      GmemLayoutAtomRab{},
      Layout<Shape<Int<rab_row_size>, _8>, Stride<_8, _1>>{}));

  // Output copy (BF16 or FP16): uses OutputType-sized loads (not FP8-sized).
  // SmemLayoutAtomO has 64-wide BF16 row (128-byte SW128 atom).
  // kGmemThreadsPerRowO must be 64/8=8 for BF16 output — NOT kBlockKSmem=32 (FP8 input).
  static constexpr int kGmemElemsPerLoadO =
      sizeof(cute::uint128_t) / sizeof(OutputType);  // = 8 for BF16
  static constexpr int kGmemThreadsPerRowO = 64 / kGmemElemsPerLoadO;  // = 8
  using GmemLayoutAtomO = Layout<
      Shape<Int<kNThreads / kGmemThreadsPerRowO>, Int<kGmemThreadsPerRowO>>,
      Stride<Int<kGmemThreadsPerRowO>, _1>>;
  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, OutputType>{},
      GmemLayoutAtomO{},
      Layout<Shape<_1, Int<kGmemElemsPerLoadO>>>{}));
};

template <int kHeadDim_, int kBlockM_, typename OutType_>
struct Hstu_blackwell_rtx_ws_o_tma_layout {
  using type = Layout<
      Shape<Int<kBlockM_>, Int<kHeadDim_>>,
      Stride<Int<kHeadDim_>, _1>>;
};

template <int kBlockM_, typename OutType_>
struct Hstu_blackwell_rtx_ws_o_tma_layout<32, kBlockM_, OutType_> {
  using type = decltype(tile_to_shape(
      GMMA::Layout_K_SW32_Atom<OutType_>{},
      Shape<Int<kBlockM_>, Int<32>>{}));
};

template <int kBlockM_, typename OutType_>
struct Hstu_blackwell_rtx_ws_o_tma_layout<64, kBlockM_, OutType_> {
  using type = decltype(tile_to_shape(
      GMMA::Layout_K_SW64_Atom<OutType_>{},
      Shape<Int<kBlockM_>, Int<64>>{}));
};

template <int kBlockM_, typename OutType_>
struct Hstu_blackwell_rtx_ws_o_tma_layout<128, kBlockM_, OutType_> {
  using type = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<OutType_>{},
      Shape<Int<kBlockM_>, Int<128>>{}));
};

template <int kBlockM_, typename OutType_>
struct Hstu_blackwell_rtx_ws_o_tma_layout<256, kBlockM_, OutType_> {
  using type = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<OutType_>{},
      Shape<Int<kBlockM_>, Int<256>>{}));
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Warp-specialized FP8 kernel traits.
// Extends the common FP8 metadata with dedicated load/store warps and 8 math warps.
// kNThreads = 288 (9 warps × 32), kNMathThreads = 256 (8 math warps × 32).
// Math warps use tidx_math = tidx - 32 ∈ [0, 255].
// All GMEM/SMEM layout arithmetic (SmemLayoutQ, GmemTiledCopyQKV, etc.) is inherited from
// the base FP8 traits where kNThreads=256, so those layouts remain correct for 256 math threads.
template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kNWarps_,   // number of MATH warps (= 8 for kBlockM=128)
    bool Is_causal_,
    bool Is_target_,
    bool Is_context_,
    bool Is_local_,
    bool Is_arbitrary_,
    int kNFunc_,
    bool Has_rab_,
    bool Paged_KV_ = false,
    bool Is_Q_in_regs_ = false,
    bool Share_Q_K_smem_ = false,
    typename out_type = cutlass::bfloat16_t>
struct Hstu_fwd_kernel_traits_blackwell_rtx_fp8_ws
    : public Hstu_fwd_kernel_traits_blackwell_rtx_fp8<
          kHeadDim_, kBlockM_, kBlockN_, kNWarps_,
          Is_causal_, Is_target_, Is_context_, Is_local_, Is_arbitrary_, kNFunc_, Has_rab_,
          Paged_KV_, Is_Q_in_regs_, Share_Q_K_smem_, out_type> {
  using Base = Hstu_fwd_kernel_traits_blackwell_rtx_fp8<
      kHeadDim_, kBlockM_, kBlockN_, kNWarps_,
      Is_causal_, Is_target_, Is_context_, Is_local_, Is_arbitrary_, kNFunc_, Has_rab_,
      Paged_KV_, Is_Q_in_regs_, Share_Q_K_smem_, out_type>;

  // Warp roles:
  //   warps 0..kNWarps_-1  = math warps  (WG0: warps 0-3, WG1: warps 4-7)
  //   warps kNWarps_..+3   = load warpgroup WG2 (warps 8-11, 4 warps = 1 complete warpgroup)
  //     warp kNWarps_      = Q/SFA TMA load
  //     warp kNWarps_+1    = K/SFB TMA load
  //     warp kNWarps_+2    = V/SFV TMA load
  //     warp kNWarps_+3    = O TMA store
  // Having a complete WG2 allows setmaxnreg WARPSYNC.ALL retry loop to function correctly.
  static constexpr int kNMathWarps   = kNWarps_;     // 8 math warps
  static constexpr int kNLoadWarps   = 4;            // Q, K, V, and O-specialized load/store warps
  static constexpr int kLoadWarpIdx  = kNWarps_;     // warp 8 starts the load warpgroup

  // kNThreads overrides Base::kNThreads: total = 8 math + 4 load = 12 warps × 32 = 384.
  static constexpr int kNThreads     = (kNMathWarps + kNLoadWarps) * cutlass::NumThreadsPerWarp;  // 384
  // kNMathThreads: math-warp-only thread count used for per-warp layout arithmetic.
  static constexpr int kNMathThreads = kNMathWarps * cutlass::NumThreadsPerWarp;  // 256

  // Double-buffer KV SMEM layout.
  // Each K or Vt tile is kBlockN * kHeadDim FP8 bytes (1 byte each).
  static constexpr int kSmemKVBytes =
      Base::kBlockN * Base::kHeadDim * (int)sizeof(typename Base::Element);
  // hdim256 arbitrary needs ValidBlockIds SMEM; two KV stages would exceed Blackwell RTX's
  // 101376B opt-in limit.  RAB-in-SMEM also needs a 128x64 BF16 tile; for D256
  // RAB and D128 arbitrary+RAB we release one KV stage to stay under the limit.
  static constexpr bool kUseRabSmem = Has_rab_;
  static constexpr bool kUseSingleKVStage =
      (Is_arbitrary_ && kHeadDim_ > 128) ||
      (kUseRabSmem && (kHeadDim_ > 128 || (Is_arbitrary_ && kHeadDim_ == 128)));
  static constexpr int kSmemWsKVStages = kUseSingleKVStage ? 1 : 2;
  static constexpr int kSmemWsRabStages = kUseRabSmem ? 1 : 0;
  static constexpr int kSmemWsRabLayoutStages = 1;

  // Producer/consumer mbarriers: ready barriers model cnt=1, empty barriers model cnt=0.
  //   k_ready[0/1]: K/SFK TMA completion per stage
  //   v_ready[0/1]: V/SFV TMA completion per stage
  //   k_empty[0/1]: K/SFK consumed into registers per stage
  //   v_empty[0/1]: V/SFV consumed into registers per stage
  //   q_ready: Q/SFA TMA completion
  //   q_empty: Q/SFA consumed into registers
  //   o_ready[0]: O SMEM ready for TMA store
  //   o_empty[0]: independent O SMEM buffer is free for math epilogue writes
  // Each barrier is 8 bytes.  RAB-in-SMEM adds one ready/empty pair.
  static constexpr int kSmemMbarSize = kUseRabSmem ? 128 : 112;
  // kSmemWsKVStages × (K + Vt).
  static constexpr int kSmemWsKVTotalBytes = 2 * kSmemWsKVStages * kSmemKVBytes;

  // WS SMEM region offsets (double-buffer layout):
  //   [0 .. kSmemWsKVTotalBytes)          : K/Vt stages (one stage for hdim256 arbitrary, otherwise two)
  //   [kSmemWsKVTotalBytes .. +ValidBl)   : ValidBlockIds (Is_arbitrary only)
  //   [.. + func region)                  : func arrays (Is_arbitrary only)
  //   [padded to 8B)                      : SFA(512B) + SFB(512B) = 1024B
  //   [last kSmemMbarSize bytes)          : 14 or 16 mbarriers × 8B
  static constexpr int kSmemWsValidBlockIdsOffset = kSmemWsKVTotalBytes;
  static constexpr int kSmemWsFuncOffset = kSmemWsValidBlockIdsOffset +
      (Is_arbitrary_ ? (int)(size(typename Base::SmemLayoutValidBlockIds{}) * sizeof(int)) : 0);
  // Arbitrary func data: sn_valid_block_max (1 int) + sFunc_min (kNFunc/2+1 ints) + sFunc_max (kNFunc/2+1 ints)
  static constexpr int kSmemWsFuncEnd = kSmemWsFuncOffset +
      (Is_arbitrary_ ? (int)((Base::kNFunc/2 + 1 + Base::kNFunc/2 + 1 + 1) * (int)sizeof(int)) : 0);

  // Persistent Q tile: math warp TMA writes Q directly here; LDSM reads it every GEMM1 iteration.
  // Aligned to the SW128 absolute-swizzle period (2048B) so TMA's absolute-swizzle write addresses
  // match the PI-swizzled LDSM read addresses for both Is_causal and Is_arbitrary.
  static constexpr int kSmemQPersistBytes =
      kBlockM_ * kHeadDim_ * (int)sizeof(typename Base::Element);
  static constexpr int kSmemWsQPersistOffset = ((kSmemWsFuncEnd + 2047) / 2048) * 2048;
  static constexpr int kSmemWsAfterQPersist = kSmemWsQPersistOffset + kSmemQPersistBytes;
  static constexpr int kSmemWsAfterQPersistPadded = ((kSmemWsAfterQPersist + 127) / 128) * 128;
  using SmemLayoutWsO_TMA =
      typename Hstu_blackwell_rtx_ws_o_tma_layout<kHeadDim_, kBlockM_, out_type>::type;
  static constexpr int kSmemWsOStoreBytes =
      (int)size(SmemLayoutWsO_TMA{}) * (int)sizeof(out_type);
  // D32/D64/D128 can afford a dedicated swizzled O buffer. D256 no-RAB keeps
  // two KV stages, whose combined 64KB region can be safely reused after math
  // consumes all K/V tiles and before the next tile's K/V loads.
  static constexpr bool kUseIndependentOBuffer = kHeadDim_ <= 128;
  static constexpr bool kUseAliasedOBuffer =
      kHeadDim_ > 128 && !kUseSingleKVStage;
  static constexpr bool kUseTmaOStore =
      kUseIndependentOBuffer || kUseAliasedOBuffer;
  static constexpr int kSmemWsOOffset = kSmemWsAfterQPersistPadded;
  static constexpr int kSmemWsOBytes =
      kUseIndependentOBuffer
          ? kSmemWsOStoreBytes
          : 0;
  static constexpr int kSmemWsAfterO = kSmemWsOOffset + kSmemWsOBytes;
  static constexpr int kSmemWsRabOffset = kSmemWsAfterO;
  static constexpr int kSmemWsRabBytes =
      kUseRabSmem ? kBlockM_ * kBlockN_ * (int)sizeof(cutlass::bfloat16_t) : 0;
  static constexpr int kSmemWsAfterRab = kSmemWsRabOffset + kSmemWsRabBytes;
  // WS data region padded to 128B; SF (TMA targets) starts at this offset from smem_.
  static constexpr int kSmemWsDataSizePadded = ((kSmemWsAfterRab + 127) / 128) * 128;
  // WS SF SMEM: SFA(512B) + SFP unit(512B) + SFB[0]+[1] + SFV[0]+[1] (each 512B @ 128) = 3072B.
  // SFP is separate so real SFA SMEM is never clobbered — enables per-tile s2r of SFA for GEMM1.
  static constexpr int kSmemWsSFSize =
      (kBlockM_ * 2 + kBlockN_ * 4) * (int)sizeof(int32_t);
  static constexpr int kSmemSize = kSmemWsDataSizePadded + kSmemWsSFSize + kSmemMbarSize;

  // RAB is loaded as a dense row-major BF16 [M,N] tile.  Math warps read it
  // scalar from SMEM after GEMM1; no QMMA/LDSM layout is required.
  using SmemLayoutRab_TMA = Layout<
      Shape<Int<kBlockM_>, Int<kBlockN_>, _1>,
      Stride<Int<kBlockN_>, _1, Int<kBlockM_ * kBlockN_>>>;

  // Invariant checks
  static_assert(kNMathWarps * 16 == Base::kBlockM,
      "kNMathWarps * 16 must equal kBlockM (8 warps × 16 rows = 128)");
  static_assert(kNThreads == (kNMathWarps + kNLoadWarps) * 32,
      "kNThreads == (kNMathWarps + kNLoadWarps) * 32");  // 384
  static_assert(kSmemMbarSize == (kUseRabSmem ? 128 : 112),
      "kSmemMbarSize must be 112 without RAB SMEM or 128 with RAB SMEM");
};
