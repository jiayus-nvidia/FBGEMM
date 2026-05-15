/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// HSTU-local minimal subset of the Blackwell RTX block-scaled GEMM builder.
//
// Extracted from:
//   6KD_fp8_block_scale/kernels/include/sm120_blockscaled_gemm/sm120_blockscaled_utils.cuh
//
// Only the members consumed by hstu_fwd_kernel_fp8_ws.h (WS FP8 path) are included:
//   - TiledMma / MmaAtom (Blackwell RTX block-scaled QMMA)
//   - SmemCopyAtomA/B/SF + SmemLayoutAtomA/B + SmemLayoutSFA/SFB
//   - partition_fragment_SFA/B, get_layoutSFA/B_TV, transform_fragment_for_qmma
//
// Intentionally omitted: TMA descriptors, TMA transaction bytes, SharedStorage,
// BarrierStorage, SmemLayoutA/B (multi-stage), scheduler helpers, padded-offset utils.

#pragma once

#include "cute/atom/mma_atom.hpp"
#include <cute/atom/copy_traits_sm75.hpp>     // SM75_U32x4_LDSM_N
#include <cute/atom/mma_traits_sm90_gmma.hpp> // GMMA::Layout_K_SW128_Atom
#include <cute/atom/mma_traits_sm120.hpp>     // SM120::BLOCKSCALED::SM120_16x8x32_TN_VS
#include <cute/config.hpp>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>            // float_e4m3_t, float_ue8m0_t
#include <type_traits>

namespace hstu {

using namespace cute;

// Minimal HSTU-local subset of the Blackwell RTX block-scaled GEMM builder.
//
// Template parameters match the original to preserve call-site compatibility:
//   TileM_  = kBlockM  (128 in HSTU)
//   TileN_  = kBlockN  (128 for GEMM1) or kHeadDim (128 for GEMM2)
//   Stages_ = pipeline stages (fixed at 4 in WS kernel)
template <int TileM_ = 32, int TileN_ = 128, int Stages_ = 4>
struct BlackwellRtxQmmaBuilder {
  using ElementA         = cutlass::float_e4m3_t;
  using ElementB         = cutlass::float_e4m3_t;
  using ElementSFLoad    = int32_t;                // 4 e8m0 values packed per int32
  using ElementSFCompute = cutlass::float_ue8m0_t; // scale type consumed by QMMA atom
  using ElementAccum     = float;

  static constexpr int AB_Stages      = Stages_;
  static constexpr int SF_Stages      = 1;
  static constexpr int kTileM         = TileM_;
  static constexpr int kTileN         = TileN_;
  static constexpr int kTileSF        = 1;
  static constexpr int kTileK         = 128;
  static constexpr int kNumTileKPerSF = 512 / kTileK;
  static constexpr int kNumStagePerSF = kNumTileKPerSF / AB_Stages;
  static_assert(
      kNumStagePerSF > 0 && kNumStagePerSF <= 2,
      "kNumStagePerSF must be 1 or 2");

  using TileShape      = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using ScaleTileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileSF>>;

  // ====== MMA atom and tiled MMA ======
  // Replicates the reference block-scaled GEMM builder's TiledMma exactly.
  // AtomLayout <_2,_4,_1>: 2×4 warp grid (8 warps in the math warpgroup).
  // PermMmaTileN swizzles 128-wide N-tiles for Blackwell RTX QMMA optimal bank layout.
  // The 32/64-wide cases use a linear N tile; callers with hand-written
  // fragment placement must branch on kTileN because the N-atom order changes.
  using PermMmaTileM = Int<32>;
  using PermMmaTileN = std::conditional_t<
      (kTileN == 32),
      Int<32>,
      std::conditional_t<
          (kTileN == 64),
          Int<64>,
          Layout<Shape<_8, _4, _4>, Stride<_1, _32, _8>>>>;
  using PermMmaTileK = Underscore;

  using MmaAtom = MMA_Atom<SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
      ElementA,
      ElementB,
      ElementAccum,
      ElementSFCompute,
      32>>;

  using TiledMma = TiledMMA<
      MmaAtom,
      Layout<Shape<_8, _1, _1>, Stride<_1, _0, _0>>,
      Tile<PermMmaTileM, PermMmaTileN, PermMmaTileK>>;

  static_assert(
      kTileM % cute::size(PermMmaTileM{}) == 0,
      "TileM must be divisible by PermMmaTileM (32)");
  static_assert(
      kTileN % cute::size(PermMmaTileN{}) == 0,
      "TileN must be divisible by PermMmaTileN");

  static constexpr int kNumMathThreads = size(typename TiledMma::ThrLayoutVMNK{});
  static constexpr int kNumMathWarps   = kNumMathThreads / 32;

  // ====== SMEM → RF copy atoms ======
  // SM75_U32x4_LDSM_N: ldmatrix.x4 in K-inner layout (required by TiledMma).
  using SmemCopyAtomA  = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
  using SmemCopyAtomB  = Copy_Atom<SM75_U32x4_LDSM_N, ElementB>;
  // AutoVectorizingCopy for scale factors (int32, 4-byte aligned LDS).
  using SmemCopyAtomSF = Copy_Atom<AutoVectorizingCopy, ElementSFLoad>;

  // ====== SMEM layout atoms ======
  // K_SW128: 8-row × 128-element swizzled atom, K-inner.
  // Required for SM75_U32x4_LDSM_N (ldmatrix.x4 requires K-inner layout).
  using SmemLayoutAtomA = GMMA::Layout_K_SW128_Atom<ElementA>;
  using SmemLayoutAtomB = GMMA::Layout_K_SW128_Atom<ElementB>;

  // ====== Scale factor SMEM layouts ======
  // SFA: [kTileM, kTileSF=1] single-stage, used for Q-side (GEMM1) and P-side (GEMM2) scales.
  // SFB: [kTileN, kTileSF=1] single-stage, used for K-side (GEMM1) and V-side (GEMM2) scales.
  using SmemLayoutAtomSFA = decltype(
      make_ordered_layout(select<0, 2>(ScaleTileShape{}), Step<_1, _2>{}));

  using SmemLayoutAtomSFB = decltype(
      make_ordered_layout(select<1, 2>(ScaleTileShape{}), Step<_1, _2>{}));

  using SmemLayoutSFA = decltype(tile_to_shape(
      SmemLayoutAtomSFA{},
      make_shape(
          shape<0>(ScaleTileShape{}),
          shape<2>(ScaleTileShape{}),
          Int<SF_Stages>{}),
      Step<_1, _2, _3>{}));

  using SmemLayoutSFB = decltype(tile_to_shape(
      SmemLayoutAtomSFB{},
      make_shape(
          shape<1>(ScaleTileShape{}),
          shape<2>(ScaleTileShape{}),
          Int<SF_Stages>{}),
      Step<_1, _2, _3>{}));

  // ====== Scale factor partition helpers ======

  // Internal: partition SFA tensor into (ThrV, (ThrM, ThrK)) x (FrgV, (RestM, RestK)).
  // AtomLayoutSFA_TV encodes per-atom thread<->value mapping; stride _0 in second dim
  // means 16 effective threads (from 2x2x8=32 logical) due to mode collapse.
  template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE static constexpr auto
  thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
    CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

    auto permutation_mnk = TiledPerm{};
    auto t_tile          = make_tile(get<0>(permutation_mnk), _1{});
    auto tiled_sfa       = logical_divide(sfatensor, t_tile);

    using AtomShape_MNK = typename Atom::Shape_MNK;
    auto atom_tile      = make_tile(
        make_layout(size<0>(AtomShape_MNK{})),
        make_layout(_1{}));
    auto tiled_atom_sfa = zipped_divide(tiled_sfa, atom_tile);

    using AtomLayoutSFA_TV =
        Layout<Shape<Shape<_2, _2, _8>, _1>, Stride<Stride<_8, _0, _1>, _16>>;
    auto tv_atom_sfa = tiled_atom_sfa.compose(AtomLayoutSFA_TV{}, _);

    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto thr_tile        = make_tile(
        _,
        make_tile(
            make_layout(size<1>(thr_layout_vmnk)),
            make_layout(size<3>(thr_layout_vmnk))));
    return zipped_divide(tv_atom_sfa, thr_tile);
  }

  // Returns a register fragment for SFA matching the per-thread MMA partition.
  template <class SFATensor, class ThrMma>
  CUTE_HOST_DEVICE static constexpr auto
  partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma) {
    auto thr_tensor = make_tensor(
        static_cast<SFATensor&&>(sfatensor).data(),
        thrfrg_SFA(sfatensor.layout(), thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vmk  = make_coord(
        get<0>(thr_vmnk),
        make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFA =
        thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
    return make_fragment_like<ElementSFLoad>(partition_SFA);
  }

  // Returns the SFA thread<->value layout for use with make_tiled_copy_impl.
  template <class TiledMmaType>
  CUTE_HOST_DEVICE static constexpr auto
  get_layoutSFA_TV(TiledMmaType& mma) {
    auto tile_shape_mnk  = tile_shape(mma);
    auto ref_A           = make_layout(make_shape(size<0>(tile_shape_mnk), _1{}));
    auto thr_tensor      = thrfrg_SFA(ref_A, mma);
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    // SFA strides: M-direction contributes (Int<1>{}), K-direction is collapsed (Int<0>{}).
    auto atile = make_tile(
        _,
        make_tile(
            make_layout(
                make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                make_stride(Int<1>{}, Int<0>{})),
            _));
    auto tv_sfa         = thr_tensor.compose(atile, _);
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    return tv_sfa.compose(thridx_2_thrid, _);
  }

  // Internal: partition SFB tensor into (ThrV, (ThrN, ThrK)) x (FrgV, (RestN, RestK)).
  // AtomLayoutSFB_TV: 4x8=32 logical threads, 8 effective (stride _0 collapses first dim).
  template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE static constexpr auto
  thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
    CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

    auto permutation_mnk = TiledPerm{};
    auto t_tile          = make_tile(get<1>(permutation_mnk), _1{});
    auto tiled_sfb       = logical_divide(sfbtensor, t_tile);

    using AtomShape_MNK = typename Atom::Shape_MNK;
    auto atom_tile      = make_tile(
        make_layout(size<1>(AtomShape_MNK{})),
        make_layout(_1{}));
    auto tiled_atom_sfb = zipped_divide(tiled_sfb, atom_tile);

    using AtomLayoutSFB_TV =
        Layout<Shape<Shape<_4, _8>, _1>, Stride<Stride<_0, _1>, _8>>;
    auto tv_atom_sfb = tiled_atom_sfb.compose(AtomLayoutSFB_TV{}, _);

    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto thr_tile        = make_tile(
        _,
        make_tile(
            make_layout(size<2>(thr_layout_vmnk)),
            make_layout(size<3>(thr_layout_vmnk))));
    return zipped_divide(tv_atom_sfb, thr_tile);
  }

  // Returns a register fragment for SFB matching the per-thread MMA partition.
  template <class SFBTensor, class ThrMma>
  CUTE_HOST_DEVICE static constexpr auto
  partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
    auto thr_tensor = make_tensor(
        static_cast<SFBTensor&&>(sfbtensor).data(),
        thrfrg_SFB(sfbtensor.layout(), thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    // SFB is indexed by ThrN (get<2>) and ThrK (get<3>), not ThrM (get<1>).
    auto thr_vnk = make_coord(
        get<0>(thr_vmnk),
        make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFB =
        thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
    return make_fragment_like<ElementSFLoad>(partition_SFB);
  }

  // Returns the SFB thread<->value layout for use with make_tiled_copy_impl.
  template <class TiledMmaType>
  CUTE_HOST_DEVICE static constexpr auto
  get_layoutSFB_TV(TiledMmaType& mma) {
    auto tile_shape_mnk  = tile_shape(mma);
    auto ref_B           = make_layout(make_shape(size<1>(tile_shape_mnk), _1{}));
    auto thr_tensor      = thrfrg_SFB(ref_B, mma);
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    // SFB strides: N-direction is collapsed (Int<0>{}), K-direction contributes (Int<1>{}).
    auto btile = make_tile(
        _,
        make_tile(
            make_layout(
                make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                make_stride(Int<0>{}, Int<1>{})),
            _));
    auto tv_sfb         = thr_tensor.compose(btile, _);
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    return tv_sfb.compose(thridx_2_thrid, _);
  }

  // Reinterprets an int32 SF fragment as float_ue8m0_t for Blackwell RTX QMMA consumption.
  // Output layout: (32:0, num_mn:4, 4:0, 4:1) matches Blackwell RTX QMMA scale operand format.
  template <class Tensor>
  CUTE_HOST_DEVICE static constexpr auto
  transform_fragment_for_qmma(Tensor&& tensor) {
    CUTE_STATIC_ASSERT_V(rank(tensor) == Int<3>{});
    auto old_ptr = tensor.data();
    auto new_ptr = recast_ptr<ElementSFCompute>(old_ptr);
    auto num_mn  = size<1>(shape(tensor.layout()));
    CUTE_STATIC_ASSERT_V(size<2>(shape(tensor.layout())) == Int<1>{});
    auto new_layout = make_layout(
        make_shape(_32{}, num_mn, _4{}, _4{}),
        make_stride(_0{}, _4{}, _0{}, _1{}));
    return make_tensor(new_ptr, new_layout);
  }
};

} // namespace hstu
