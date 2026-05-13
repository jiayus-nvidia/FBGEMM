// Phase 6 warp-specialized FP8 kernel body — Q, K, and V^T via TMA.
// Included inside namespace flash from hstu_fwd_kernel.h.
// Do not include directly; use hstu_fwd_kernel.h.
//
// Warp layout (12 warps = 3 complete warpgroups, kNThreads=384):
//   WG0: warps 0-3  (math)
//   WG1: warps 4-7  (math)
//   WG2: warps 8-11 (load warpgroup)
//     warp 8  = Q/SFA TMA load
//     warp 9  = K/SFB TMA load
//     warp 10 = V/SFV TMA load
//     warp 11 = O TMA store
// Complete WG2 ensures setmaxnreg TRY_ALLOC WARPSYNC.ALL retry loop works correctly.
// With __launch_bounds__(384,1): compiler budget = 65536/384 ≈ 168 regs → TRY_ALLOC(inc) succeeds
// on first attempt (already at budget), no retry needed.
//
// CTA-level __syncthreads__ map (must match between branches):
//   S_arb : Is_arbitrary only — math warp 1 writes sValidBlockIds; load warp just syncs
//   S1    : after load warp inits producer/consumer mbarriers + fence; persistent full/causal does this
//           once before the static scheduler loop, non-persistent paths keep the per-tile S1
//   S2→   : replaced by wait_mbar_parity(q_ready_mbar_ptr, parity) for all 256 math threads
//   S3    : removed (was a no-op rendezvous with no real data dependency)
//   S5    : partial-tile-only math bar.sync; full O-store handoff uses o_ready/o_empty.

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ uint32_t mbar_smem_addr(uint64_t* mbar) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
}

__device__ __forceinline__ void arrive_mbar(uint32_t maddr) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" : : "r"(maddr));
}

__device__ __forceinline__ void arrive_mbar(uint64_t* mbar) {
  arrive_mbar(mbar_smem_addr(mbar));
}

__device__ __forceinline__ void arrive_expect_tx_mbar(
    uint32_t maddr,
    uint32_t expected_tx_bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               : : "r"(maddr), "r"(expected_tx_bytes));
}

__device__ __forceinline__ void arrive_expect_tx_mbar(
    uint64_t* mbar,
    uint32_t expected_tx_bytes) {
  arrive_expect_tx_mbar(mbar_smem_addr(mbar), expected_tx_bytes);
}

// Fast path with one non-suspending probe; long waits then use try_wait so the
// warp can yield while TMA or another producer completes the phase.
__device__ inline void wait_mbar_parity(uint32_t maddr, uint32_t parity) {
  uint32_t done = 0;
  asm volatile(
      "{.reg .pred p;\n"
      "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      "selp.u32 %0, 1, 0, p;}\n"
      : "=r"(done) : "r"(maddr), "r"(parity) : "memory");
  if (done) return;

  constexpr uint32_t kTryWaitSuspendHint = 0x989680u;
  do {
    asm volatile(
        "{.reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n"
        "selp.u32 %0, 1, 0, p;}\n"
        : "=r"(done) : "r"(maddr), "r"(parity), "r"(kTryWaitSuspendHint) : "memory");
  } while (!done);
}

__device__ inline void wait_mbar_parity(uint64_t* mbar, uint32_t parity) {
  wait_mbar_parity(mbar_smem_addr(mbar), parity);
}

struct HstuWsTileCoord {
  int bidb;
  int bidh;
  int m_block;
};

__device__ __forceinline__ HstuWsTileCoord hstu_ws_decode_tile(
    const int tile,
    const int num_m_block,
    const int num_heads,
    const bool head_shared_rab) {
  if (head_shared_rab) {
    const int bidh = tile % num_heads;
    const int bm = tile / num_heads;
    const int m_linear = bm % num_m_block;
    return HstuWsTileCoord{
        bm / num_m_block,
        bidh,
        num_m_block - 1 - m_linear};
  } else {
    const int m_linear = tile % num_m_block;
    const int bh = tile / num_m_block;
    return HstuWsTileCoord{
        bh / num_heads,
        bh % num_heads,
        num_m_block - 1 - m_linear};
  }
}

// Pack GEMM1 C-fragment (acc_s) to FP8 and rearrange it to GEMM2 A-fragment
// (tCrP) layout using warp shuffle only.  BN64 and BN128 use different
// C-fragment N atom order.
//
// With AtomLayout <_8,_1,_1> (1 N-warp), each warp holds all 128 N-values of P in its
// own registers after GEMM1. No cross-warp communication is needed: the entire D→A
// rearrangement is intra-warp, performed by SHFL.IDX within each 4-thread quad.
//
// QMMA.SF.16832 coordinate mappings (SPA ISA):
//   C-fragment: packed N-atom nr has bytes {mr=0,i=0},{mr=0,i=1},{mr=1,i=0},{mr=1,i=1}
//     where N_base(nr) = (nr%4)*32 + (nr/4)*8  (from PermMmaTileN strides (1,32,8))
//     and   col = N_base(nr) + (lane&3)*2 + i
//   A-fragment: tCrP[4*kb + 2*c + mr] at row=8*mr+(lane>>2), col=32*kb+16*c+(lane&3)*4+{0..3}
//
// For each (kb,c), the 4 K-values needed by one thread span two consecutive C-fragment
// atoms: atom_lo covers col[0..1] (from src_lane0) and atom_hi covers col[2..3] (from src_lane1).
//   src_lane0 = (lane & ~3u) | ((lane&1u) << 1)  — lower pair of the quad
//   src_lane1 = src_lane0 + 1                     — upper pair of the quad
// kAtomLut[kb][c][h]: C-fragment atom index for K-half h (h = (lane&3)>>1).
//   kAtomLut[kb][c][h] = kb + 4*(2*c + h)
// Byte assembly:
//   mr=0 (row=quad):   __byte_perm(a, b, 0x5410) = {a.b0,a.b1,b.b0,b.b1}
//   mr=1 (row=8+quad): __byte_perm(a, b, 0x7632) = {a.b2,a.b3,b.b2,b.b3}
//
// Cost for BN128: 16 FP8 pair converts + 32 SHFL + 16 BYTE_PERM per thread.
// Savings: eliminates 2 × bar.sync (256-thread) + 16KB SMEM write + LDSM load-back.
template <int kBlockN, int kHeadDim, typename PFragment, typename AccFragment>
__device__ __forceinline__ void permute_acc_s_packed_to_tCrP(
    PFragment& tCrP,
    AccFragment const& acc_s,
    int tidx_math) {
  static_assert(kBlockN == 64 || kBlockN == 128, "GEMM2 P permutation supports BN64/BN128");
  constexpr int kPackedElems = kBlockN / 8;
  constexpr int kAccSElems = 4 * kPackedElems;
  uint32_t acc_s_packed[kPackedElems];

  CUTE_UNROLL
  for (int flat = 0; flat < kAccSElems; flat += 4) {
    uint32_t out;
    asm volatile(
        "{\n"
        ".reg .b16 lo, hi;\n"
        "cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n"
        "cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
        "mov.b32 %0, {lo, hi};\n"
        "}\n"
        : "=r"(out)
        : "f"(static_cast<float>(acc_s(flat + 0))),
          "f"(static_cast<float>(acc_s(flat + 1))),
          "f"(static_cast<float>(acc_s(flat + 2))),
          "f"(static_cast<float>(acc_s(flat + 3))));
    acc_s_packed[flat / 4] = out;
  }

  auto tXrP = recast<uint32_t>(tCrP);
  const unsigned lane      = (unsigned)tidx_math & 31u;
  const unsigned tiq       = lane & 3u;
  const unsigned quad_base = lane & ~3u;
  const unsigned src_lane0 = quad_base | ((tiq & 1u) << 1u);
  const unsigned src_lane1 = src_lane0 + 1u;

  if constexpr (kBlockN == 64) {
    CUTE_UNROLL
    for (int kb = 0; kb < 2; ++kb) {
      CUTE_UNROLL
      for (int c = 0; c < 2; ++c) {
        const uint32_t pk0 = acc_s_packed[4 * kb + 2 * c + 0];
        const uint32_t pk1 = acc_s_packed[4 * kb + 2 * c + 1];
        const uint32_t a0  = __shfl_sync(0xFFFFFFFFu, pk0, src_lane0);
        const uint32_t b0  = __shfl_sync(0xFFFFFFFFu, pk0, src_lane1);
        const uint32_t a1  = __shfl_sync(0xFFFFFFFFu, pk1, src_lane0);
        const uint32_t b1  = __shfl_sync(0xFFFFFFFFu, pk1, src_lane1);
        const uint32_t a   = (tiq >> 1u) ? a1 : a0;
        const uint32_t b   = (tiq >> 1u) ? b1 : b0;
        tXrP(4 * kb + 2 * c + 0) = __byte_perm(a, b, 0x5410u);
        tXrP(4 * kb + 2 * c + 1) = __byte_perm(a, b, 0x7632u);
      }
    }
    if constexpr (kHeadDim == 256) {
      CUTE_UNROLL
      for (int i = 8; i < size(tXrP); ++i) {
        tXrP(i) = tXrP(i & 7);
      }
    }
  } else {
    constexpr uint8_t kLut0[4][2] = {{ 0, 8}, { 1, 9}, { 2, 10}, { 3, 11}};
    constexpr uint8_t kLut1[4][2] = {{ 4,12}, { 5,13}, { 6, 14}, { 7, 15}};
    CUTE_UNROLL
    for (int kb = 0; kb < 4; ++kb) {
      CUTE_UNROLL
      for (int c = 0; c < 2; ++c) {
        const uint32_t pk0 = acc_s_packed[kLut0[kb][c]];
        const uint32_t pk1 = acc_s_packed[kLut1[kb][c]];
        const uint32_t a0  = __shfl_sync(0xFFFFFFFFu, pk0, src_lane0);
        const uint32_t b0  = __shfl_sync(0xFFFFFFFFu, pk0, src_lane1);
        const uint32_t a1  = __shfl_sync(0xFFFFFFFFu, pk1, src_lane0);
        const uint32_t b1  = __shfl_sync(0xFFFFFFFFu, pk1, src_lane1);
        const uint32_t a   = (tiq >> 1u) ? a1 : a0;
        const uint32_t b   = (tiq >> 1u) ? b1 : b0;
        tXrP(4 * kb + 2 * c + 0) = __byte_perm(a, b, 0x5410u);
        tXrP(4 * kb + 2 * c + 1) = __byte_perm(a, b, 0x7632u);
      }
    }
  }
}

template <
    typename Kernel_traits,
    int kBlockN,
    int kHeadDim,
    int kHeadDimGemm2,
    typename FP8Elem,
    typename Fragment>
__device__ __forceinline__ void hstu_ws_load_v_ldsm_t(
    FP8Elem* sVt_cur,
    Fragment& tCrV,
    int tidx_math) {
  auto tXrV = recast<uint32_t>(tCrV);
  if constexpr (kHeadDim < kHeadDimGemm2) {
    clear(tCrV);
  }
  const uint32_t v_smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(sVt_cur));
  typename Kernel_traits::SmemLayoutVt_TMA smem_layout_vt;
  const int lane = tidx_math & 31;
  const int n_off = lane & 15;
  const int d_mat = (lane >> 4) << 4;

  CUTE_UNROLL
  for (int dg = 0; dg < kHeadDim / 32; ++dg) {
    CUTE_UNROLL
    for (int ni = 0; ni < kBlockN / 16; ++ni) {
      const int n_k_row = ni * 16 + n_off;
      const int d_start = dg * 32 + d_mat;
      const uint32_t addr_slab =
          v_smem_base + (uint32_t)smem_layout_vt(n_k_row, d_start);

      if constexpr (kHeadDim <= 64) {
        const int base = 16 * (ni >> 1) + 8 * dg + (ni & 1);
        asm volatile(
            "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];\n"
            : "=r"(tXrV(base + 0)), "=r"(tXrV(base + 2)),
              "=r"(tXrV(base + 4)), "=r"(tXrV(base + 6))
            : "r"(addr_slab));
      } else {
        const int dg_slab = dg >> 2;
        const int dg_in_slab = dg & 3;
        const int base =
            (kHeadDim / 4) * (ni >> 1) + 32 * dg_slab + 2 * dg_in_slab + (ni & 1);
        asm volatile(
            "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];\n"
            : "=r"(tXrV(base + 0)), "=r"(tXrV(base + 8)),
              "=r"(tXrV(base + 16)), "=r"(tXrV(base + 24))
            : "r"(addr_slab));
      }
    }
  }
}

__device__ __forceinline__ int32_t hstu_ws_replicate_e8m0_lane(
    int32_t packed,
    int lane) {
  const uint32_t byte = (static_cast<uint32_t>(packed) >> (8 * lane)) & 0xffu;
  return static_cast<int32_t>(byte | (byte << 8) | (byte << 16) | (byte << 24));
}

template <
    typename Kernel_traits,
    bool Target_group_size_1,
    typename Params,
    typename ThrMmaG1,
    typename AccTensor,
    typename MinFuncTensor,
    typename MaxFuncTensor>
__device__ __forceinline__ void hstu_ws_apply_mask_bs(
    AccTensor& tSrS,
    int nb,
    const Params& params,
    ThrMmaG1& thr_mma_g1,
    MinFuncTensor& gMinFunc,
    MaxFuncTensor& gMaxFunc,
    int m_block,
    int actual_seqlen_k,
    int actual_seqlen_h,
    int actual_seqlen_c,
    int actual_seqlen_offset,
    int last_page_offset) {
  constexpr bool Is_causal    = Kernel_traits::Is_causal;
  constexpr bool Is_target    = Kernel_traits::Is_target;
  constexpr bool Is_context   = Kernel_traits::Is_context;
  constexpr bool Is_arbitrary = Kernel_traits::Is_arbitrary;
  constexpr bool Is_local     = Kernel_traits::Is_local;
  constexpr bool Paged_KV     = Kernel_traits::Paged_KV;
  constexpr int kBlockM       = Kernel_traits::kBlockM;
  constexpr int kBlockN       = Kernel_traits::kBlockN;
  static constexpr int Row = 0, Col = 1;

  Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
  Tensor tScS = thr_mma_g1.partition_C(cS);
  const int base_row = m_block * kBlockM + actual_seqlen_offset;
  const int base_col = nb * kBlockN;
  Tensor col_min = make_tensor<int>(make_shape(size<0>(gMinFunc)));
  Tensor col_max = make_tensor<int>(make_shape(size<0>(gMaxFunc)));
  int prev_block_row = -1;
  int row = 0;
  [[maybe_unused]] int tgt_col_lft = 0;

#pragma unroll
  for (int flat = 0; flat < size(tSrS); ++flat) {
    const auto coord = tScS(flat);
    const int block_row = int(get<Row>(coord));
    if (block_row != prev_block_row) {
      row = block_row + base_row;
      prev_block_row = block_row;
      if constexpr (Is_target) {
        if constexpr (Target_group_size_1) {
          tgt_col_lft = row;
        } else {
          const int tgt_idx = (row - actual_seqlen_h) / params.target_group_size;
          tgt_col_lft = actual_seqlen_h + tgt_idx * params.target_group_size;
        }
      }
      if constexpr (Is_arbitrary) {
        col_max(0) = gMaxFunc(0, block_row);
#pragma unroll
        for (int j = 0; j < size<0>(gMinFunc); ++j) {
          col_min(j) = gMinFunc(j, block_row);
          col_max(j + 1) = gMaxFunc(j + 1, block_row);
        }
      }
    }

    const int block_col = int(get<Col>(coord));
    int col = block_col + base_col;
    if (Paged_KV && row >= actual_seqlen_h) {
      col -= last_page_offset;
    }
    if constexpr (!Is_causal && !Is_local && !Is_arbitrary) {
      if (col >= actual_seqlen_k) {
        tSrS(flat) = -INFINITY;
        continue;
      }
    } else {
      if constexpr (Is_context) {
        if (row < actual_seqlen_c && col < actual_seqlen_h) continue;
      }
      if (col >= std::min(actual_seqlen_k, row + 1 + params.window_size_right)) {
        tSrS(flat) = -INFINITY;
        continue;
      }
      if constexpr (Is_local) {
        if (col < std::max(0, row - params.window_size_left)) {
          tSrS(flat) = -INFINITY;
          continue;
        }
      }
      if constexpr (Is_target) {
        if (row >= actual_seqlen_h &&
            (col + (Paged_KV ? last_page_offset : 0)) >= actual_seqlen_h &&
            col < tgt_col_lft) {
          tSrS(flat) = -INFINITY;
        }
      }
      if constexpr (Is_arbitrary) {
        bool non_mask = (0 <= col) && (col < col_max(0));
        if (non_mask) continue;
#pragma unroll
        for (int j = 0; j < size<0>(gMinFunc); ++j) {
          non_mask = (col_min(j) <= col) && (col < col_max(j + 1));
          if (non_mask) break;
        }
        if (!non_mask) tSrS(flat) = -INFINITY;
      }
    }
  }
}

template <
    typename Kernel_traits,
    typename AccTensor,
    typename ThrMmaG1,
    typename RabTensor>
__device__ __forceinline__ void hstu_ws_add_rab_smem_bs(
    AccTensor& tSrS,
    int nb,
    bool skip_masked,
    int stage,
    ThrMmaG1& thr_mma_g1,
    RabTensor& sRab,
    int m_block,
    int actual_seqlen_q,
    int actual_seqlen_k,
    int actual_seqlen_h,
    int n_block_paged,
    int last_page_offset) {
  if constexpr (Kernel_traits::Has_rab) {
    constexpr bool Is_target = Kernel_traits::Is_target;
    constexpr bool Paged_KV  = Kernel_traits::Paged_KV;
    constexpr int kBlockM    = Kernel_traits::kBlockM;
    constexpr int kBlockN    = Kernel_traits::kBlockN;
    static constexpr int Row = 0, Col = 1;

    Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thr_mma_g1.partition_C(cS);
    const int base_col = nb * kBlockN;

    CUTE_UNROLL
    for (int flat = 0; flat < size(tSrS); ++flat) {
      if (skip_masked && tSrS(flat) == -INFINITY) {
        continue;
      }
      const auto coord = tScS(flat);
      const int block_row = int(get<Row>(coord));
      const int q_idx = m_block * kBlockM + block_row;
      if (q_idx >= actual_seqlen_q) {
        continue;
      }
      const int block_col = int(get<Col>(coord));
      int col = block_col + base_col;
      if constexpr (Paged_KV && Is_target) {
        if (nb >= n_block_paged) {
          col -= last_page_offset;
        }
        if (nb < n_block_paged && col >= actual_seqlen_h) {
          continue;
        }
      }
      if (0 <= col && col < actual_seqlen_k) {
        tSrS(flat) += static_cast<float>(sRab(block_row, block_col, stage));
      }
    }
  }
}

template <
    typename Kernel_traits,
    typename AccTensor,
    typename ThrMmaG1,
    typename RabTensor>
__device__ __forceinline__ void hstu_ws_consume_rab_bs(
    AccTensor& tSrS,
    int nb,
    bool skip_masked,
    int& rab_ready_wait_parity0,
    uint32_t smem_base32,
    int tidx_math,
    ThrMmaG1& thr_mma_g1,
    RabTensor& sRab,
    int m_block,
    int actual_seqlen_q,
    int actual_seqlen_k,
    int actual_seqlen_h,
    int n_block_paged,
  int last_page_offset) {
  if constexpr (Kernel_traits::Has_rab) {
    static constexpr int kSmemMbar0Offset =
        Kernel_traits::kSmemSize - Kernel_traits::kSmemMbarSize;
    wait_mbar_parity(
        smem_base32 + (uint32_t)kSmemMbar0Offset + 112u,
        (uint32_t)rab_ready_wait_parity0);
    rab_ready_wait_parity0 ^= 1;
    asm volatile("" ::: "memory");
    hstu_ws_add_rab_smem_bs<Kernel_traits>(
        tSrS,
        nb,
        skip_masked,
        0,
        thr_mma_g1,
        sRab,
        m_block,
        actual_seqlen_q,
        actual_seqlen_k,
        actual_seqlen_h,
        n_block_paged,
        last_page_offset);
    if ((tidx_math & 31) == 0) {
      arrive_mbar(smem_base32 + (uint32_t)kSmemMbar0Offset + 120u);
    }
  }
}

template <
    typename Kernel_traits,
    typename AccTensor,
    typename ThrMmaG1,
    typename RabTensor>
__device__ __forceinline__ void hstu_ws_add_rab_smem_paged_target_bs(
    AccTensor& tSrS,
    ThrMmaG1& thr_mma_g1,
    RabTensor& sRab) {
  if constexpr (Kernel_traits::Has_rab) {
    static_assert(Kernel_traits::Paged_KV && Kernel_traits::Is_target);
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    static constexpr int Row = 0, Col = 1;

    Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thr_mma_g1.partition_C(cS);

    CUTE_UNROLL
    for (int flat = 0; flat < size(tSrS); ++flat) {
      const auto coord = tScS(flat);
      const int block_row = int(get<Row>(coord));
      const int block_col = int(get<Col>(coord));
      tSrS(flat) += static_cast<float>(sRab(block_row, block_col, _0{}));
    }
  }
}

template <
    typename Kernel_traits,
    typename AccTensor,
    typename ThrMmaG1,
    typename RabTensor>
__device__ __forceinline__ void hstu_ws_consume_rab_paged_target_bs(
    AccTensor& tSrS,
    int& rab_ready_wait_parity0,
    uint32_t smem_base32,
    int tidx_math,
    ThrMmaG1& thr_mma_g1,
    RabTensor& sRab) {
  if constexpr (Kernel_traits::Has_rab) {
    static_assert(Kernel_traits::Paged_KV && Kernel_traits::Is_target);
    static constexpr int kSmemMbar0Offset =
        Kernel_traits::kSmemSize - Kernel_traits::kSmemMbarSize;
    wait_mbar_parity(
        smem_base32 + (uint32_t)kSmemMbar0Offset + 112u,
        (uint32_t)rab_ready_wait_parity0);
    rab_ready_wait_parity0 ^= 1;
    asm volatile("" ::: "memory");
    hstu_ws_add_rab_smem_paged_target_bs<Kernel_traits>(
        tSrS, thr_mma_g1, sRab);
    if ((tidx_math & 31) == 0) {
      arrive_mbar(smem_base32 + (uint32_t)kSmemMbar0Offset + 120u);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename Kernel_traits,
    bool Use_full_persistent = false,
    bool Use_paired_persistent = false,
    bool Target_group_size_1 = false,
    typename Params>
inline __device__ void hstu_compute_attn_1rowblock_blackwell_rtx_fp8_ws(
    const Params& params,
    const int bidb_arg,
    const int bidh_arg,
    int m_block_arg) {

  static_assert(Kernel_traits::Is_fp8, "Phase 6 WS: FP8 path only");
  static_assert(
      !(Use_full_persistent && Use_paired_persistent),
      "Only one FP8 WS persistent scheduler can be enabled");
  constexpr bool Use_persistent = Use_full_persistent || Use_paired_persistent;

  using BS1 = hstu::BlackwellRtxQmmaBuilder<
      Kernel_traits::kBlockM, Kernel_traits::kBlockN, 4>;
  static constexpr int kHeadDimGemm2 =
      Kernel_traits::kHeadDim < 64 ? 64 : Kernel_traits::kHeadDim;
  using BS2 = hstu::BlackwellRtxQmmaBuilder<
      Kernel_traits::kBlockM, kHeadDimGemm2, 4>;
  using FP8Elem = typename Kernel_traits::Element;

  extern __shared__ char smem_[];
  static constexpr int kSmemMbar0Offset =
      Kernel_traits::kSmemSize - Kernel_traits::kSmemMbarSize;

  const int tidx = threadIdx.x;
  constexpr int kNMathThreads = Kernel_traits::kNMathThreads;  // 256
  const bool is_load_store_warp = (tidx >= kNMathThreads);

  constexpr int kBlockM = Kernel_traits::kBlockM;
  const int num_m_block_capacity = (params.seqlen_q + kBlockM - 1) / kBlockM;
  int num_m_block_scheduler = num_m_block_capacity;
  constexpr bool Use_actual_target_scheduler =
      Use_persistent &&
      Kernel_traits::Is_target &&
      !Kernel_traits::Paged_KV &&
      !Kernel_traits::Has_rab;
  const int uniform_actual_seqlen_q =
      (params.b > 0 && params.total_q % params.b == 0)
      ? params.total_q / params.b
      : 0;
  if constexpr (Use_actual_target_scheduler) {
    if (uniform_actual_seqlen_q > 0 &&
        uniform_actual_seqlen_q < params.seqlen_q) {
      num_m_block_scheduler =
          (uniform_actual_seqlen_q + kBlockM - 1) / kBlockM;
    }
    const int total_tiles_scheduler =
        num_m_block_scheduler * params.h * params.b;
    int work_units_scheduler = total_tiles_scheduler;
    if constexpr (Use_paired_persistent) {
      work_units_scheduler =
          ((num_m_block_scheduler + 1) / 2) * params.h * params.b;
    }
    if (int(blockIdx.x) >= work_units_scheduler) return;
  }

  // CTA-level sync before setmaxnreg: all warps start from clean state.
  __syncthreads();

  // ============================================================
  // LOAD WARPGROUP PATH  (warps 8-11, threads 256-383)
  //   warp 8  (tidx 256-287): Q/SFA load
  //   warp 9  (tidx 288-319): K/SFB load
  //   warp 10 (tidx 320-351): V/SFV load
  //   warp 11 (tidx 352-383): O store
  // ============================================================
  if (is_load_store_warp) {
    // Four load warps are role-specialized:
    //   warp 8  : Q + SFA TMA load
    //   warp 9  : K + SFB TMA load
    //   warp 10 : V + SFV TMA load
    //   warp 11 : O TMA store
    const int load_warp_id = (tidx - kNMathThreads) >> 5;
    const bool is_q_load_warp = load_warp_id == 0;
    const bool is_k_load_warp = load_warp_id == 1;
    const bool is_v_load_warp = load_warp_id == 2;

    if constexpr (
        Kernel_traits::kHeadDim > 128 &&
        Kernel_traits::Paged_KV &&
        Kernel_traits::Has_rab) {
      if (is_k_load_warp || is_v_load_warp) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" : : "n"(56));
      } else {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" : : "n"(40));
      }
    } else if constexpr (Kernel_traits::kHeadDim > 128) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" : : "n"(40));
    } else {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" : : "n"(56));
    }

    constexpr bool Is_causal    = Kernel_traits::Is_causal;
    constexpr bool Is_target    = Kernel_traits::Is_target;
    constexpr bool Is_context   = Kernel_traits::Is_context;
    constexpr bool Is_arbitrary = Kernel_traits::Is_arbitrary;
    constexpr int  kNFunc       = Kernel_traits::kNFunc;
    constexpr bool Is_local     = Kernel_traits::Is_local;
    constexpr bool Has_rab      = Kernel_traits::Has_rab;
    constexpr bool Use_rab_smem = Kernel_traits::kUseRabSmem;
    constexpr bool Paged_KV     = Kernel_traits::Paged_KV;
    constexpr int  kBlockM      = Kernel_traits::kBlockM;
    constexpr int  kBlockN      = Kernel_traits::kBlockN;
    constexpr int  kHeadDim     = Kernel_traits::kHeadDim;
    using OutElement = typename Kernel_traits::OutputType;

    uint64_t* k_ready_mbar_ptr0 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset);
    uint64_t* k_ready_mbar_ptr1 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 8);
    uint64_t* v_ready_mbar_ptr0 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 16);
    uint64_t* v_ready_mbar_ptr1 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 24);
    uint64_t* k_empty_mbar_ptr0 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 32);
    uint64_t* k_empty_mbar_ptr1 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 40);
    uint64_t* v_empty_mbar_ptr0 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 48);
    uint64_t* v_empty_mbar_ptr1 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 56);
    uint64_t* q_ready_mbar_ptr  = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 64);
    uint64_t* q_empty_mbar_ptr  = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 72);
    uint64_t* o_ready_mbar_ptr0 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 80);
    uint64_t* o_ready_mbar_ptr1 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 88);
    uint64_t* o_empty_mbar_ptr0 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 96);
    uint64_t* o_empty_mbar_ptr1 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 104);
    uint64_t* rab_ready_mbar_ptr0 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 112);
    uint64_t* rab_empty_mbar_ptr0 = reinterpret_cast<uint64_t*>(smem_ + kSmemMbar0Offset + 120);

    auto init_ws_mbarriers = [&]() {
      if (tidx == kNMathThreads) {
        uint32_t kr0 = static_cast<uint32_t>(__cvta_generic_to_shared(k_ready_mbar_ptr0));
        uint32_t kr1 = static_cast<uint32_t>(__cvta_generic_to_shared(k_ready_mbar_ptr1));
        uint32_t vr0 = static_cast<uint32_t>(__cvta_generic_to_shared(v_ready_mbar_ptr0));
        uint32_t vr1 = static_cast<uint32_t>(__cvta_generic_to_shared(v_ready_mbar_ptr1));
        uint32_t ke0 = static_cast<uint32_t>(__cvta_generic_to_shared(k_empty_mbar_ptr0));
        uint32_t ke1 = static_cast<uint32_t>(__cvta_generic_to_shared(k_empty_mbar_ptr1));
        uint32_t ve0 = static_cast<uint32_t>(__cvta_generic_to_shared(v_empty_mbar_ptr0));
        uint32_t ve1 = static_cast<uint32_t>(__cvta_generic_to_shared(v_empty_mbar_ptr1));
        uint32_t qr  = static_cast<uint32_t>(__cvta_generic_to_shared(q_ready_mbar_ptr));
        uint32_t qe  = static_cast<uint32_t>(__cvta_generic_to_shared(q_empty_mbar_ptr));
        uint32_t or0 = static_cast<uint32_t>(__cvta_generic_to_shared(o_ready_mbar_ptr0));
        uint32_t or1 = static_cast<uint32_t>(__cvta_generic_to_shared(o_ready_mbar_ptr1));
        uint32_t oe0 = static_cast<uint32_t>(__cvta_generic_to_shared(o_empty_mbar_ptr0));
        uint32_t oe1 = static_cast<uint32_t>(__cvta_generic_to_shared(o_empty_mbar_ptr1));
        uint32_t rr0 = static_cast<uint32_t>(__cvta_generic_to_shared(rab_ready_mbar_ptr0));
        uint32_t re0 = static_cast<uint32_t>(__cvta_generic_to_shared(rab_empty_mbar_ptr0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(kr0), "r"(1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(kr1), "r"(1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(vr0), "r"(1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(vr1), "r"(1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(ke0), "r"(8));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(ke1), "r"(8));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(ve0), "r"(8));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(ve1), "r"(8));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(qr), "r"(1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(qe), "r"(8));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(or0), "r"(8));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(or1), "r"(8));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(oe0), "r"(1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(oe1), "r"(1));
        if constexpr (Use_rab_smem) {
          asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(rr0), "r"(1));
          asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(re0), "r"(8));
        }
        for (int i = 0; i < 8; i++) {
          arrive_mbar(ke0);
          arrive_mbar(ke1);
          arrive_mbar(ve0);
          arrive_mbar(ve1);
          arrive_mbar(qe);
          if constexpr (Use_rab_smem) {
            arrive_mbar(re0);
          }
        }
        // Independent O buffer starts empty; math epilogue waits on this phase before
        // writing O, and the O-store warp releases the next phase after TMA store wait.
        arrive_mbar(oe0);
      }
      asm volatile("fence.proxy.async.shared::cta;\n" : : : "memory");
    };

    int k_empty_wait_parity0 = 0;
    int k_empty_wait_parity1 = 0;
    int v_empty_wait_parity0 = 0;
    int v_empty_wait_parity1 = 0;
    int q_empty_wait_parity = 0;
    int o_ready_wait_parity0 = 0;
    int o_empty_load_wait_parity0 = 0;
    int rab_empty_wait_parity0 = 0;
    int kv_load_stage = 0;
    if constexpr (Use_persistent) {
      init_ws_mbarriers();
      __syncthreads();  // One-time S1 for persistent: WS mbarriers initialized before scheduler loop.
    }

    auto run_load_store_tile = [&](auto load_role, const int bidb, const int bidh, int m_block) {
      constexpr int LoadRole = decltype(load_role)::value;
      constexpr bool IsQLoadRole = LoadRole == 0;
      constexpr bool IsKLoadRole = LoadRole == 1;
      constexpr bool IsVLoadRole = LoadRole == 2;
      constexpr bool IsOStoreRole = LoadRole == 3;
      const HstuBlockInfo<Kernel_traits, Params> binfo(params, bidb);
      // Early exit 1: before any sync — both branches exit simultaneously.
      if (m_block * kBlockM >= binfo.actual_seqlen_q_padded) return;

      char* smem_q    = reinterpret_cast<char*>(smem_);
      char* smem_func = reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsFuncOffset;
      int* sn_valid_block_max = reinterpret_cast<int*>(smem_func);

      const int actual_seqlen_q        = binfo.actual_seqlen_q;
      const int actual_seqlen_k        = binfo.actual_seqlen_k;
      const int actual_seqlen_t        = Is_target  ? binfo.actual_seqlen_t : 0;
      const int actual_seqlen_c        = Is_context ? binfo.actual_seqlen_c : 0;
      const int actual_seqlen_h        = Is_target  ? actual_seqlen_k - actual_seqlen_t : actual_seqlen_k;
      const int actual_seqlen_offset   = actual_seqlen_k - actual_seqlen_q;
      const int last_page_seqlen       = Paged_KV ? binfo.last_page_seqlen : kBlockN;
      const int page_offset            = Paged_KV ? binfo.sum_s_page : 0;

      const bool is_jump             = Is_target && m_block * kBlockM + actual_seqlen_offset > actual_seqlen_h;
      const bool is_in_target        = Is_target && (m_block + 1) * kBlockM + actual_seqlen_offset > actual_seqlen_h;
      const bool is_in_context       = Is_context && (m_block + 1) * kBlockM <= actual_seqlen_c;
      const bool is_in_mixed_context = Is_context &&
          (m_block + 1) * kBlockM > actual_seqlen_c && m_block * kBlockM < actual_seqlen_c;
      const bool is_in_paged_target  = is_in_target && Paged_KV;
      const int last_page_offset     = is_in_paged_target ? kBlockN - last_page_seqlen : 0;

      const int n_block_history = cute::ceil_div(actual_seqlen_h, kBlockN);
      const int n_block_paged   = Paged_KV ? n_block_history : 0;
      const int n_block_target  = cute::ceil_div(actual_seqlen_t, kBlockN);

      int n_block_min = !Is_local ? 0
          : std::max(0, (m_block * kBlockM + actual_seqlen_offset - params.window_size_left) / kBlockN);
      int n_block_max = Paged_KV ? n_block_history + n_block_target : cute::ceil_div(actual_seqlen_k, kBlockN);
      if constexpr (Is_causal || Is_local) {
        int offset = (m_block + 1) * kBlockM + actual_seqlen_offset + params.window_size_right;
        if (is_in_paged_target) offset += last_page_offset;
        n_block_max = std::min(n_block_max, cute::ceil_div(offset, kBlockN));
      }
      if constexpr (Is_context) {
        n_block_min = (is_in_context || is_in_mixed_context) ? 0 : n_block_min;
        n_block_max = (is_in_context || is_in_mixed_context)
            ? std::max(n_block_history, n_block_max) : n_block_max;
      }

      int n_masking_block_max = cute::ceil_div(
          std::min(actual_seqlen_k + last_page_offset,
                   (m_block + 1) * kBlockM + actual_seqlen_offset + last_page_offset),
          kBlockN);
      int n_masking_block_min = (m_block * kBlockM + actual_seqlen_offset) / kBlockN;
      if constexpr (Is_target) {
        if constexpr (Target_group_size_1) {
          n_masking_block_min = is_jump
              ? (m_block * kBlockM + actual_seqlen_offset + last_page_offset) / kBlockN
              : n_masking_block_min;
        } else {
          const int target_index =
              (m_block * kBlockM - actual_seqlen_h) / params.target_group_size;
          n_masking_block_min = is_jump
              ? (actual_seqlen_h + actual_seqlen_offset +
                 target_index * params.target_group_size + last_page_offset) / kBlockN
              : n_masking_block_min;
        }
      }
      if constexpr (Is_context) {
        n_masking_block_min = is_in_mixed_context ? n_block_min : n_masking_block_min;
        n_masking_block_max = is_in_mixed_context ? n_block_max : n_masking_block_max;
      }
      const int n_masking_steps = (!Is_causal || is_in_context)
          ? 0 : n_masking_block_max - n_masking_block_min;

      // Is_arbitrary: load warp only participates in __syncthreads__; math warp 1 does the work.
      if constexpr (Is_arbitrary) {
        __syncthreads();  // S_arb: wait for math warp 1 to write sValidBlockIds
        n_block_max = *sn_valid_block_max;
        n_block_min = 0;
      }

      // Early exit 2 (after Is_arbitrary): non-persistent paths still consume the per-tile S1.
      if (((Is_causal || Is_local || Is_arbitrary) && n_block_max <= n_block_min) ||
          m_block * kBlockM >= actual_seqlen_q) {
        if constexpr (!Use_persistent) {
          __syncthreads();  // S1
        }
        return;
      }

      // SMEM layout types for TMA partition_D.
      using SmemLayoutK_SW128  = typename Kernel_traits::SmemLayoutK_TMA;
      using SmemLayoutVt_SW128 = typename Kernel_traits::SmemLayoutVt_TMA;
      using SmemLayoutQ_SW128  = typename Kernel_traits::SmemLayoutQ_TMA;
      using SmemLayoutRab_TMA  = typename Kernel_traits::SmemLayoutRab_TMA;
      using RabElement = cutlass::bfloat16_t;

      constexpr int kSmemKVElems = kBlockN * kHeadDim;
      FP8Elem* const sK_base[2] = {
          reinterpret_cast<FP8Elem*>(smem_q),
          Kernel_traits::kUseSingleKVStage
              ? reinterpret_cast<FP8Elem*>(smem_q)
              : reinterpret_cast<FP8Elem*>(smem_q) + 2 * kSmemKVElems
      };
      FP8Elem* const sVt_base[2] = {
          reinterpret_cast<FP8Elem*>(smem_q) + kSmemKVElems,
          Kernel_traits::kUseSingleKVStage
              ? reinterpret_cast<FP8Elem*>(smem_q) + kSmemKVElems
              : reinterpret_cast<FP8Elem*>(smem_q) + 3 * kSmemKVElems
      };

      // q_tma_mbar_ptr is initialized with the other WS mbarriers above.

      // SF SMEM pointers.
      static constexpr int kSmemSFOffset_WS = Kernel_traits::kSmemWsDataSizePadded;
      int32_t* smem_sfa_ptr = reinterpret_cast<int32_t*>(smem_ + kSmemSFOffset_WS);
      int32_t* smem_sfp_ptr = smem_sfa_ptr + kBlockM;
      int32_t* const smem_sfb_ptr[2] = {
          smem_sfa_ptr + 2 * kBlockM,
          smem_sfa_ptr + 2 * kBlockM + kBlockN
      };
      int32_t* const smem_sfv_ptr[2] = {
          smem_sfa_ptr + 2 * kBlockM + 2 * kBlockN,
          smem_sfa_ptr + 2 * kBlockM + 3 * kBlockN
      };

      if constexpr (!Use_persistent) {
        init_ws_mbarriers();
        __syncthreads();  // S1: WS mbarriers visible to all warps.
      }

      // TMA tensor setup for K, V^T, SFB, SFV (load warps only).
      const int bidh_kv = bidh / params.h_h_k_ratio;
      constexpr int kSmemKBytes    = kBlockN * kHeadDim * (int)sizeof(FP8Elem);
      constexpr int kSmemVtBytes   = kHeadDim * kBlockN * (int)sizeof(FP8Elem);
      constexpr int kSmemSFBBytes  = kBlockN * (int)sizeof(int32_t);
      constexpr int kSmemSFVBytes  = kBlockN * (int)sizeof(int32_t);
      constexpr uint32_t kSmemKSFBBytes =
          (uint32_t)(kSmemKBytes + kSmemSFBBytes);
      constexpr uint32_t kSmemVtSFVBytes =
          (uint32_t)(kSmemVtBytes + kSmemSFVBytes);
      constexpr uint32_t kSmemRabBytes =
          (uint32_t)(kBlockM * kBlockN * (int)sizeof(RabElement));

      Tensor sValidBlockIds = make_tensor(
          make_smem_ptr(reinterpret_cast<int*>(smem_ + Kernel_traits::kSmemWsValidBlockIdsOffset)),
          typename Kernel_traits::SmemLayoutValidBlockIds{});

      // ===== ROLE-SPECIALIZED LOAD WARPS =====
      // Q/SFA warp issues the per-row-tile Q preamble. K/SFB and V/SFV warps independently
      // feed the double-buffered N loop. O-store warp owns the final TMA store.  Q and O do
      // not get extra buffers.  Producer/consumer mbarriers replace the old tile-end load
      // rendezvous: load warps wait empty, math waits ready, and O-store waits O ready.
      if constexpr (IsQLoadRole) {
        wait_mbar_parity(q_empty_mbar_ptr, (uint32_t)q_empty_wait_parity);
        q_empty_wait_parity ^= 1;
        if (tidx == kNMathThreads) {
          constexpr uint32_t kSmemQBytes   = kBlockM * kHeadDim * (uint32_t)sizeof(FP8Elem);
          constexpr uint32_t kSmemSFABytes = kBlockM * (uint32_t)sizeof(int32_t);
          using SmemLayoutSFA_TMA_t = cute::Layout<cute::Shape<cute::Int<kBlockM>, cute::Int<1>>,
                                                   cute::Stride<cute::_1, cute::Int<kBlockM>>>;
          auto mQ_tma   = params.tma_q.get_tma_tensor(make_shape(params.total_q, params.d, params.h));
          auto gQ_head  = mQ_tma(_, _, bidh);
          auto gQ_tiles = local_tile(gQ_head, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(_, _));
          auto tma_slice_Q = params.tma_q.get_slice(0);
          auto sQ_buf      = make_tensor(
              make_smem_ptr(reinterpret_cast<FP8Elem*>(
                  reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsQPersistOffset)),
              SmemLayoutQ_SW128{});
          auto tQsQ_d      = tma_slice_Q.partition_D(sQ_buf);
          auto tQgQ_tma    = tma_slice_Q.partition_S(gQ_tiles(_, _, _, Int<0>{}));
          auto mSFA_tma    = params.tma_sfa.get_tma_tensor(
              make_shape((int64_t)params.q_block_descale_head_stride, cute::Int<1>{}, params.h));
          auto gSFA_head   = mSFA_tma(_, _, bidh);
          auto gSFA_tiles  = local_tile(gSFA_head, Shape<Int<kBlockM>, Int<1>>{}, make_coord(_, _));
          auto tma_slice_SFA = params.tma_sfa.get_slice(0);
          auto tSFAsSFA_d    = tma_slice_SFA.partition_D(
              make_tensor(make_smem_ptr(smem_sfa_ptr), SmemLayoutSFA_TMA_t{}));
          auto tSFAgSFA_tma  = tma_slice_SFA.partition_S(gSFA_tiles(_, _, _, Int<0>{}));
          arrive_expect_tx_mbar(q_ready_mbar_ptr, kSmemQBytes + kSmemSFABytes);
          const int m_abs = binfo.sum_s_q / kBlockM + m_block;
          cute::copy(params.tma_q.with(*q_ready_mbar_ptr),   tQgQ_tma(_, _, _, m_abs),     tQsQ_d);
          cute::copy(params.tma_sfa.with(*q_ready_mbar_ptr), tSFAgSFA_tma(_, _, _, m_abs), tSFAsSFA_d);
        }
      }

      if constexpr (Kernel_traits::kUseAliasedOBuffer && (IsKLoadRole || IsVLoadRole)) {
        wait_mbar_parity(o_empty_mbar_ptr0, (uint32_t)o_empty_load_wait_parity0);
        o_empty_load_wait_parity0 ^= 1;
      }

      if constexpr (IsKLoadRole) {
        using SmemLayoutSFB_TMA_t = cute::Layout<cute::Shape<cute::Int<kBlockN>, cute::Int<1>>,
                                                 cute::Stride<cute::_1, cute::Int<kBlockN>>>;

        auto load_rab_tma = [&](int nb) {
          if constexpr (Has_rab) {
            wait_mbar_parity(rab_empty_mbar_ptr0, (uint32_t)rab_empty_wait_parity0);
            rab_empty_wait_parity0 ^= 1;
            if (tidx == kNMathThreads + 32) {
              RabElement* smem_rab = reinterpret_cast<RabElement*>(
                  reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsRabOffset);
              Tensor sRab_tma = make_tensor(make_smem_ptr(smem_rab), SmemLayoutRab_TMA{});
              const int bidh_rab = Has_rab && params.h_rab > 1 ? bidh : 0;
              auto mRab_tma = params.tma_rab.get_tma_tensor(
                  make_shape(params.seqlen_k_rounded, params.seqlen_k_rounded,
                             Has_rab ? params.h_rab : 1, params.b))(_, _, bidh_rab, bidb);
              auto gRab_tiles = local_tile(
                  domain_offset(make_coord(actual_seqlen_offset, _0{}), mRab_tma),
                  Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block, _));
              auto tma_slice_Rab = params.tma_rab.get_slice(0);
              auto tRabgRab_tma = group_modes<0, 3>(tma_slice_Rab.partition_S(gRab_tiles));
              auto tRabsRab_d = group_modes<0, 3>(tma_slice_Rab.partition_D(sRab_tma));
              arrive_expect_tx_mbar(rab_ready_mbar_ptr0, kSmemRabBytes);
              cute::copy(params.tma_rab.with(*rab_ready_mbar_ptr0),
                         tRabgRab_tma(_, nb), tRabsRab_d(_, _0{}));
            }
          }
        };

        if constexpr (Paged_KV) {
          for (int n_valid = n_block_max - 1, masking_step_load = 0; n_valid >= n_block_min;
               ++masking_step_load, --n_valid) {
            const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;

            uint64_t* k_empty_mbar_ptr = kv_load_stage ? k_empty_mbar_ptr1 : k_empty_mbar_ptr0;
            int& k_empty_wait_parity = kv_load_stage ? k_empty_wait_parity1 : k_empty_wait_parity0;
            wait_mbar_parity(k_empty_mbar_ptr, (uint32_t)k_empty_wait_parity);
            k_empty_wait_parity ^= 1;

            uint64_t* k_ready_mbar_ptr = kv_load_stage ? k_ready_mbar_ptr1 : k_ready_mbar_ptr0;
            if (tidx == kNMathThreads + 32) {
              arrive_expect_tx_mbar(k_ready_mbar_ptr, kSmemKSFBBytes);
              auto mSFB_tma = params.tma_sfb.get_tma_tensor(
                  make_shape((int64_t)params.kv_block_descale_head_stride, cute::Int<1>{}, params.h_k));
              auto gSFB_head_k = mSFB_tma(_, _, bidh_kv);
              auto gSFB_tiles_k = local_tile(gSFB_head_k, Shape<Int<kBlockN>, Int<1>>{}, make_coord(_, _));
              auto tma_slice_SFB = params.tma_sfb.get_slice(0);
              auto tSFBgSFB_tma = tma_slice_SFB.partition_S(gSFB_tiles_k(_, _, _, Int<0>{}));
              auto tSFBsSFB_d = tma_slice_SFB.partition_D(make_tensor(
                  make_smem_ptr(kv_load_stage ? smem_sfb_ptr[1] : smem_sfb_ptr[0]),
                  SmemLayoutSFB_TMA_t{}));
              if (nb < n_block_paged) {
                const int page_id = params.page_ids[page_offset + nb];
                const int sf_page_block = (page_id * params.page_size) / kBlockN;
                auto mK_page_tma = params.tma_k_page.get_tma_tensor(
                    make_shape(params.page_size, params.d, params.h_k, params.total_pages));
                auto gK_page_head = mK_page_tma(_, _, bidh_kv, _);
                auto gK_page_tiles = local_tile(
                    gK_page_head, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _));
                auto tma_slice_K_page = params.tma_k_page.get_slice(0);
                auto tKPgK_tma = tma_slice_K_page.partition_S(
                    gK_page_tiles(_, _, _, Int<0>{}, _));
                auto tKPsK_d = tma_slice_K_page.partition_D(make_tensor(
                    make_smem_ptr(kv_load_stage ? sK_base[1] : sK_base[0]),
                    SmemLayoutK_SW128{}));
                cute::copy(params.tma_k_page.with(*k_ready_mbar_ptr),
                           tKPgK_tma(_, _, _, Int<0>{}, page_id), tKPsK_d);
                cute::copy(params.tma_sfb.with(*k_ready_mbar_ptr),
                           tSFBgSFB_tma(_, _, _, sf_page_block), tSFBsSFB_d);
              } else {
                // Wrapper pads paged target K/V and packed SF to kBlockN-aligned
                // physical blocks, so target tail uses the same TMA path as history.
                const int target_block = nb - n_block_paged;
                const int target_start = binfo.sum_s_k + actual_seqlen_k - actual_seqlen_t
                    + last_page_offset + target_block * kBlockN;
                const int nb_abs = target_start / kBlockN;
                const int sf_abs = (params.total_pages * params.page_size + target_start) / kBlockN;
                auto mK_tma = params.tma_k.get_tma_tensor(
                    make_shape(params.total_k, params.d, params.h_k));
                auto gK_head = mK_tma(_, _, bidh_kv);
                auto gK_tiles = local_tile(
                    gK_head, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _));
                auto tma_slice_K = params.tma_k.get_slice(0);
                auto tKgK_tma = tma_slice_K.partition_S(gK_tiles(_, _, _, Int<0>{}));
                auto tKsK_d = tma_slice_K.partition_D(make_tensor(
                    make_smem_ptr(kv_load_stage ? sK_base[1] : sK_base[0]),
                    SmemLayoutK_SW128{}));
                cute::copy(params.tma_k.with(*k_ready_mbar_ptr),
                           tKgK_tma(_, _, _, nb_abs), tKsK_d);
                cute::copy(params.tma_sfb.with(*k_ready_mbar_ptr),
                           tSFBgSFB_tma(_, _, _, sf_abs), tSFBsSFB_d);
              }
            }
            load_rab_tma(nb);

            if (is_jump && masking_step_load == n_masking_steps - 1)
              n_valid = std::min(n_valid, n_block_history);

            if constexpr (!Kernel_traits::kUseSingleKVStage) {
              kv_load_stage ^= 1;
            }
          }
        } else {
          for (int n_valid = n_block_max - 1, masking_step_load = 0; n_valid >= n_block_min;
               ++masking_step_load, --n_valid) {
            const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;

            uint64_t* k_empty_mbar_ptr = kv_load_stage ? k_empty_mbar_ptr1 : k_empty_mbar_ptr0;
            int& k_empty_wait_parity = kv_load_stage ? k_empty_wait_parity1 : k_empty_wait_parity0;
            wait_mbar_parity(k_empty_mbar_ptr, (uint32_t)k_empty_wait_parity);
            k_empty_wait_parity ^= 1;

            if (tidx == kNMathThreads + 32) {
              const int nb_abs = binfo.sum_s_k / kBlockN + nb;
              uint64_t* k_ready_mbar_ptr = kv_load_stage ? k_ready_mbar_ptr1 : k_ready_mbar_ptr0;
              auto mK_tma = params.tma_k.get_tma_tensor(make_shape(params.total_k, params.d, params.h_k));
              auto gK_head = mK_tma(_, _, bidh_kv);
              auto gK_tiles = local_tile(gK_head, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _));
              auto tma_slice_K = params.tma_k.get_slice(0);
              auto tKgK_tma = tma_slice_K.partition_S(gK_tiles(_, _, _, Int<0>{}));
              auto tKsK_d = tma_slice_K.partition_D(make_tensor(
                  make_smem_ptr(kv_load_stage ? sK_base[1] : sK_base[0]),
                  SmemLayoutK_SW128{}));
              auto mSFB_tma = params.tma_sfb.get_tma_tensor(
                  make_shape((int64_t)params.kv_block_descale_head_stride, cute::Int<1>{}, params.h_k));
              auto gSFB_head_k = mSFB_tma(_, _, bidh_kv);
              auto gSFB_tiles_k = local_tile(gSFB_head_k, Shape<Int<kBlockN>, Int<1>>{}, make_coord(_, _));
              auto tma_slice_SFB = params.tma_sfb.get_slice(0);
              auto tSFBgSFB_tma = tma_slice_SFB.partition_S(gSFB_tiles_k(_, _, _, Int<0>{}));
              auto tSFBsSFB_d = tma_slice_SFB.partition_D(make_tensor(
                  make_smem_ptr(kv_load_stage ? smem_sfb_ptr[1] : smem_sfb_ptr[0]),
                  SmemLayoutSFB_TMA_t{}));
              arrive_expect_tx_mbar(k_ready_mbar_ptr, kSmemKSFBBytes);
              cute::copy(params.tma_k.with(*k_ready_mbar_ptr), tKgK_tma(_, _, _, nb_abs), tKsK_d);
              cute::copy(params.tma_sfb.with(*k_ready_mbar_ptr), tSFBgSFB_tma(_, _, _, nb_abs), tSFBsSFB_d);
            }
            load_rab_tma(nb);

            if (is_jump && masking_step_load == n_masking_steps - 1)
              n_valid = std::min(n_valid, n_block_history);

            if constexpr (!Kernel_traits::kUseSingleKVStage) {
              kv_load_stage ^= 1;
            }
          }
        }
      }

      if constexpr (IsVLoadRole) {
        using SmemLayoutSFV_TMA_t = cute::Layout<cute::Shape<cute::Int<kBlockN>, cute::Int<1>>,
                                                 cute::Stride<cute::_1, cute::Int<kBlockN>>>;

        if constexpr (Paged_KV) {
          for (int n_valid = n_block_max - 1, masking_step_load = 0; n_valid >= n_block_min;
               ++masking_step_load, --n_valid) {
            const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;

            uint64_t* v_empty_mbar_ptr = kv_load_stage ? v_empty_mbar_ptr1 : v_empty_mbar_ptr0;
            int& v_empty_wait_parity = kv_load_stage ? v_empty_wait_parity1 : v_empty_wait_parity0;
            wait_mbar_parity(v_empty_mbar_ptr, (uint32_t)v_empty_wait_parity);
            v_empty_wait_parity ^= 1;

            if (tidx == kNMathThreads + 64) {
              uint64_t* v_ready_mbar_ptr = kv_load_stage ? v_ready_mbar_ptr1 : v_ready_mbar_ptr0;
              arrive_expect_tx_mbar(v_ready_mbar_ptr, kSmemVtSFVBytes);
              auto mSFV_tma = params.tma_sfv.get_tma_tensor(
                  make_shape((int64_t)params.v_block_descale_head_stride, cute::Int<1>{}, params.h_k));
              auto gSFV_head_k = mSFV_tma(_, _, bidh_kv);
              auto gSFV_tiles_k = local_tile(gSFV_head_k, Shape<Int<kBlockN>, Int<1>>{}, make_coord(_, _));
              auto tma_slice_SFV = params.tma_sfv.get_slice(0);
              auto tSFVgSFV_tma = tma_slice_SFV.partition_S(gSFV_tiles_k(_, _, _, Int<0>{}));
              auto tSFVsSFV_d = tma_slice_SFV.partition_D(make_tensor(
                  make_smem_ptr(kv_load_stage ? smem_sfv_ptr[1] : smem_sfv_ptr[0]),
                  SmemLayoutSFV_TMA_t{}));
              if (nb < n_block_paged) {
                const int page_id = params.page_ids[page_offset + nb];
                const int sf_page_block = (page_id * params.page_size) / kBlockN;
                auto mVt_page_tma = params.tma_vt_page.get_tma_tensor(
                    make_shape(params.page_size, params.d, params.h_k, params.total_pages));
                auto gVt_page_head = mVt_page_tma(_, _, bidh_kv, _);
                auto gVt_page_tiles = local_tile(
                    gVt_page_head, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _));
                auto tma_slice_Vt_page = params.tma_vt_page.get_slice(0);
                auto tVPgVt_tma = tma_slice_Vt_page.partition_S(
                    gVt_page_tiles(_, _, _, Int<0>{}, _));
                auto tVPsVt_d = tma_slice_Vt_page.partition_D(make_tensor(
                    make_smem_ptr(kv_load_stage ? sVt_base[1] : sVt_base[0]),
                    SmemLayoutVt_SW128{}));
                cute::copy(params.tma_vt_page.with(*v_ready_mbar_ptr),
                           tVPgVt_tma(_, _, _, Int<0>{}, page_id), tVPsVt_d);
                cute::copy(params.tma_sfv.with(*v_ready_mbar_ptr),
                           tSFVgSFV_tma(_, _, _, sf_page_block), tSFVsSFV_d);
              } else {
                // Wrapper pads paged target K/V and packed SF to kBlockN-aligned
                // physical blocks, so target tail uses the same TMA path as history.
                const int target_block = nb - n_block_paged;
                const int target_start = binfo.sum_s_k + actual_seqlen_k - actual_seqlen_t
                    + last_page_offset + target_block * kBlockN;
                const int nb_abs = target_start / kBlockN;
                const int sf_abs = (params.total_pages * params.page_size + target_start) / kBlockN;
                auto mVt_tma = params.tma_vt.get_tma_tensor(
                    make_shape(params.total_k, params.d, params.h_k));
                auto gVt_head = mVt_tma(_, _, bidh_kv);
                auto gVt_tiles = local_tile(
                    gVt_head, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _));
                auto tma_slice_Vt = params.tma_vt.get_slice(0);
                auto tVtgVt_tma = tma_slice_Vt.partition_S(gVt_tiles(_, _, _, Int<0>{}));
                auto tVtsVt_d = tma_slice_Vt.partition_D(make_tensor(
                    make_smem_ptr(kv_load_stage ? sVt_base[1] : sVt_base[0]),
                    SmemLayoutVt_SW128{}));
                cute::copy(params.tma_vt.with(*v_ready_mbar_ptr),
                           tVtgVt_tma(_, _, _, nb_abs), tVtsVt_d);
                cute::copy(params.tma_sfv.with(*v_ready_mbar_ptr),
                           tSFVgSFV_tma(_, _, _, sf_abs), tSFVsSFV_d);
              }
            }

            if (is_jump && masking_step_load == n_masking_steps - 1)
              n_valid = std::min(n_valid, n_block_history);

            if constexpr (!Kernel_traits::kUseSingleKVStage) {
              kv_load_stage ^= 1;
            }
          }
        } else {
          for (int n_valid = n_block_max - 1, masking_step_load = 0; n_valid >= n_block_min;
               ++masking_step_load, --n_valid) {
            const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;

            uint64_t* v_empty_mbar_ptr = kv_load_stage ? v_empty_mbar_ptr1 : v_empty_mbar_ptr0;
            int& v_empty_wait_parity = kv_load_stage ? v_empty_wait_parity1 : v_empty_wait_parity0;
            wait_mbar_parity(v_empty_mbar_ptr, (uint32_t)v_empty_wait_parity);
            v_empty_wait_parity ^= 1;

            if (tidx == kNMathThreads + 64) {
              const int nb_abs = binfo.sum_s_k / kBlockN + nb;
              uint64_t* v_ready_mbar_ptr = kv_load_stage ? v_ready_mbar_ptr1 : v_ready_mbar_ptr0;
              auto mVt_tma = params.tma_vt.get_tma_tensor(make_shape(params.total_k, params.d, params.h_k));
              auto gVt_head = mVt_tma(_, _, bidh_kv);
              auto gVt_tiles = local_tile(gVt_head, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _));
              auto tma_slice_Vt = params.tma_vt.get_slice(0);
              auto tVtgVt_tma = tma_slice_Vt.partition_S(gVt_tiles(_, _, _, Int<0>{}));
              auto tVtsVt_d = tma_slice_Vt.partition_D(make_tensor(
                  make_smem_ptr(kv_load_stage ? sVt_base[1] : sVt_base[0]),
                  SmemLayoutVt_SW128{}));
              auto mSFV_tma = params.tma_sfv.get_tma_tensor(
                  make_shape((int64_t)params.v_block_descale_head_stride, cute::Int<1>{}, params.h_k));
              auto gSFV_head_k = mSFV_tma(_, _, bidh_kv);
              auto gSFV_tiles_k = local_tile(gSFV_head_k, Shape<Int<kBlockN>, Int<1>>{}, make_coord(_, _));
              auto tma_slice_SFV = params.tma_sfv.get_slice(0);
              auto tSFVgSFV_tma = tma_slice_SFV.partition_S(gSFV_tiles_k(_, _, _, Int<0>{}));
              auto tSFVsSFV_d = tma_slice_SFV.partition_D(make_tensor(
                  make_smem_ptr(kv_load_stage ? smem_sfv_ptr[1] : smem_sfv_ptr[0]),
                  SmemLayoutSFV_TMA_t{}));
              arrive_expect_tx_mbar(v_ready_mbar_ptr, kSmemVtSFVBytes);
              cute::copy(params.tma_vt.with(*v_ready_mbar_ptr), tVtgVt_tma(_, _, _, nb_abs), tVtsVt_d);
              cute::copy(params.tma_sfv.with(*v_ready_mbar_ptr), tSFVgSFV_tma(_, _, _, nb_abs), tSFVsSFV_d);
            }

            if (is_jump && masking_step_load == n_masking_steps - 1)
              n_valid = std::min(n_valid, n_block_history);

            if constexpr (!Kernel_traits::kUseSingleKVStage) {
              kv_load_stage ^= 1;
            }
          }
        }
      }

      if constexpr (IsOStoreRole) {
        if constexpr (Kernel_traits::kUseTmaOStore) {
        // Math warps own compute/softmax and write O to independent SMEM; O-store waits
        // for o_ready[0], then releases o_empty[0] after the TMA store completes.
        wait_mbar_parity(o_ready_mbar_ptr0, (uint32_t)o_ready_wait_parity0);
        o_ready_wait_parity0 ^= 1;
        if (tidx == kNMathThreads + 96) {
          asm volatile("fence.proxy.async.shared::cta;\n" : : : "memory");
          static_assert(
              Kernel_traits::kSmemWsOStoreBytes >= kBlockM * kHeadDim * (int)sizeof(OutElement),
              "O SMEM buffer is too small.");
          char* smem_o = reinterpret_cast<char*>(smem_) +
              (Kernel_traits::kUseAliasedOBuffer ? 0 : Kernel_traits::kSmemWsOOffset);
          Tensor sO_tma = make_tensor(
              make_smem_ptr(reinterpret_cast<OutElement*>(smem_o)),
              typename Kernel_traits::SmemLayoutWsO_TMA{});
          auto mO_tma   = params.tma_o.get_tma_tensor(make_shape(params.total_q, params.d, params.h));
          auto gO_head  = mO_tma(_, _, bidh);
          auto gO_tiles = local_tile(gO_head, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(_, _));
          const int m_abs = binfo.sum_s_q / kBlockM + m_block;
          auto tma_slice_O = params.tma_o.get_slice(0);
          Tensor tOsO     = tma_slice_O.partition_S(sO_tma);
          Tensor tOgO_all = tma_slice_O.partition_D(gO_tiles(_, _, _, Int<0>{}));
          cute::copy(params.tma_o, tOsO, tOgO_all(_, _, _, m_abs));
          cute::tma_store_arrive();
          cute::tma_store_wait<0>();
          arrive_mbar(o_empty_mbar_ptr0);
        }
        }
      }
    };

    // Static tile-id broadcast: load and math paths run the same deterministic
    // scheduler and decode the same tile ids locally.  There is no dynamic work
    // queue, atomic counter, or shared tile-id sync on this path.
    const bool head_shared_rab = Has_rab && params.h_rab == 1 && params.h > 1;
    auto run_load_store_scheduler = [&](auto load_role) {
      if constexpr (Use_paired_persistent) {
        const int num_m_block_persistent = Use_actual_target_scheduler
            ? num_m_block_scheduler
            : num_m_block_capacity;
        if constexpr (Use_actual_target_scheduler) {
          const int pairs_per_bh = (num_m_block_persistent + 1) / 2;
          const int total_tile_pairs_persistent =
              pairs_per_bh * params.h * params.b;
          #pragma unroll 1
          for (int tile_pair = int(blockIdx.x); tile_pair < total_tile_pairs_persistent; tile_pair += int(gridDim.x)) {
            const int pair_idx = tile_pair % pairs_per_bh;
            const int bh = tile_pair / pairs_per_bh;
            const int bidb = bh / params.h;
            const int bidh = bh - bidb * params.h;
            const int m_hi = num_m_block_persistent - 1 - pair_idx;
            const int m_lo = pair_idx;

            run_load_store_tile(load_role, bidb, bidh, m_hi);
            if (m_lo != m_hi) {
              run_load_store_tile(load_role, bidb, bidh, m_lo);
            }
          }
        } else {
          const int total_tiles_persistent = num_m_block_persistent * params.h * params.b;
          const int total_tile_pairs_persistent = (total_tiles_persistent + 1) / 2;
          #pragma unroll 1
          for (int tile_pair = int(blockIdx.x); tile_pair < total_tile_pairs_persistent; tile_pair += int(gridDim.x)) {
            const int paired_tile = total_tiles_persistent - 1 - tile_pair;
            const int tiles_this_pair = paired_tile == tile_pair ? 1 : 2;

            #pragma unroll 1
            for (int pair_slot = 0; pair_slot < tiles_this_pair; ++pair_slot) {
              const int tile = pair_slot == 0 ? tile_pair : paired_tile;
              const HstuWsTileCoord coord =
                  hstu_ws_decode_tile(tile, num_m_block_persistent, params.h, head_shared_rab);
              run_load_store_tile(load_role, coord.bidb, coord.bidh, coord.m_block);
            }
          }
        }
      } else if constexpr (Use_full_persistent) {
        const int num_m_block_persistent = Use_actual_target_scheduler
            ? num_m_block_scheduler
            : num_m_block_capacity;
        const int total_tiles_persistent = num_m_block_persistent * params.h * params.b;
        #pragma unroll 1
        for (int tile = int(blockIdx.x); tile < total_tiles_persistent; tile += int(gridDim.x)) {
          const HstuWsTileCoord coord =
              hstu_ws_decode_tile(tile, num_m_block_persistent, params.h, head_shared_rab);
          run_load_store_tile(load_role, coord.bidb, coord.bidh, coord.m_block);
        }
      } else {
        run_load_store_tile(load_role, bidb_arg, bidh_arg, m_block_arg);
      }
    };

    if (is_q_load_warp) {
      run_load_store_scheduler(std::integral_constant<int, 0>{});
    } else if (is_k_load_warp) {
      run_load_store_scheduler(std::integral_constant<int, 1>{});
    } else if (is_v_load_warp) {
      run_load_store_scheduler(std::integral_constant<int, 2>{});
    } else {
      run_load_store_scheduler(std::integral_constant<int, 3>{});
    }
    // Load warp path exits here.  Active load warp has also completed O TMA-store.

  // ============================================================
  } else {
  // MATH WARP PATH  (warps 0-7, threads 0-255)
  // ============================================================
    if constexpr (
        Kernel_traits::kHeadDim > 128 &&
        Kernel_traits::Paged_KV &&
        Kernel_traits::Has_rab) {
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" : : "n"(216));
    } else if constexpr (Kernel_traits::kHeadDim > 128) {
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" : : "n"(232));
    } else if constexpr (Kernel_traits::Paged_KV) {
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" : : "n"(216));
    } else {
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" : : "n"(224));
    }

    constexpr bool Is_causal    = Kernel_traits::Is_causal;
    constexpr bool Is_target    = Kernel_traits::Is_target;
    constexpr bool Is_context   = Kernel_traits::Is_context;
    constexpr bool Is_arbitrary = Kernel_traits::Is_arbitrary;
    constexpr int  kNFunc       = Kernel_traits::kNFunc;
    constexpr bool Is_local     = Kernel_traits::Is_local;
    constexpr bool Has_rab      = Kernel_traits::Has_rab;
    constexpr bool Paged_KV     = Kernel_traits::Paged_KV;
    constexpr int  kBlockM      = Kernel_traits::kBlockM;
    constexpr int  kBlockN      = Kernel_traits::kBlockN;
    constexpr int  kHeadDim     = Kernel_traits::kHeadDim;
    using RabElement = cutlass::bfloat16_t;

    const int tidx_math = tidx;  // math warps: tidx_math == tidx (0-255)
    int tma_parity0 = 0;  // Persistent K/V mbarrier parity for stage 0.
    int tma_parity1 = 0;  // Persistent K/V mbarrier parity for stage 1.
    int q_tma_parity = 0;
    int o_empty_wait_parity = 0;
    int rab_ready_wait_parity0 = 0;
    int math_stage  = 0;

    if constexpr (Use_persistent) {
      __syncthreads();  // One-time S1: load warp has initialized K/V mbarriers.
    }

    auto run_math_tile = [&](const int bidb, const int bidh, int m_block) {
      const HstuBlockInfo<Kernel_traits, Params> binfo(params, bidb);
      // Early exit 1: before any sync — both branches exit simultaneously.
      if (m_block * kBlockM >= binfo.actual_seqlen_q_padded) return;

      char* smem_q    = reinterpret_cast<char*>(smem_);
      char* smem_func = reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsFuncOffset;
      int* sn_valid_block_max = reinterpret_cast<int*>(smem_func);
      int* sf_min_ptr = reinterpret_cast<int*>(sn_valid_block_max) + 1;
      int* sf_max_ptr = sf_min_ptr + (kNFunc/2 + 1);

      const int actual_seqlen_q        = binfo.actual_seqlen_q;
      const int actual_seqlen_k        = binfo.actual_seqlen_k;
      const int actual_seqlen_q_padded = binfo.actual_seqlen_q_padded;
      const int actual_seqlen_t        = Is_target  ? binfo.actual_seqlen_t : 0;
      const int actual_seqlen_c        = Is_context ? binfo.actual_seqlen_c : 0;
      const int actual_seqlen_h        = Is_target  ? actual_seqlen_k - actual_seqlen_t : actual_seqlen_k;
      const int actual_seqlen_offset   = actual_seqlen_k - actual_seqlen_q;
      const int last_page_seqlen       = Paged_KV ? binfo.last_page_seqlen : kBlockN;

      const bool is_jump             = Is_target && m_block * kBlockM + actual_seqlen_offset > actual_seqlen_h;
      const bool is_in_target        = Is_target && (m_block + 1) * kBlockM + actual_seqlen_offset > actual_seqlen_h;
      const bool is_in_context       = Is_context && (m_block + 1) * kBlockM <= actual_seqlen_c;
      const bool is_in_mixed_context = Is_context &&
          (m_block + 1) * kBlockM > actual_seqlen_c && m_block * kBlockM < actual_seqlen_c;
      const bool is_in_paged_target  = is_in_target && Paged_KV;
      const int last_page_offset     = is_in_paged_target ? kBlockN - last_page_seqlen : 0;

      const int n_block_history = cute::ceil_div(actual_seqlen_h, kBlockN);
      const int n_block_paged   = Paged_KV ? n_block_history : 0;
      const int n_block_target  = cute::ceil_div(actual_seqlen_t, kBlockN);

      int n_block_min = !Is_local ? 0
          : std::max(0, (m_block * kBlockM + actual_seqlen_offset - params.window_size_left) / kBlockN);
      int n_block_max = Paged_KV ? n_block_history + n_block_target : cute::ceil_div(actual_seqlen_k, kBlockN);
      if constexpr (Is_causal || Is_local) {
        int offset = (m_block + 1) * kBlockM + actual_seqlen_offset + params.window_size_right;
        if (is_in_paged_target) offset += last_page_offset;
        n_block_max = std::min(n_block_max, cute::ceil_div(offset, kBlockN));
      }
      if constexpr (Is_context) {
        n_block_min = (is_in_context || is_in_mixed_context) ? 0 : n_block_min;
        n_block_max = (is_in_context || is_in_mixed_context)
            ? std::max(n_block_history, n_block_max) : n_block_max;
      }

      int n_masking_block_max = cute::ceil_div(
          std::min(actual_seqlen_k + last_page_offset,
                   (m_block + 1) * kBlockM + actual_seqlen_offset + last_page_offset),
          kBlockN);
      int n_masking_block_min = (m_block * kBlockM + actual_seqlen_offset) / kBlockN;
      if constexpr (Is_target) {
        if constexpr (Target_group_size_1) {
          n_masking_block_min = is_jump
              ? (m_block * kBlockM + actual_seqlen_offset + last_page_offset) / kBlockN
              : n_masking_block_min;
        } else {
          const int target_index =
              (m_block * kBlockM - actual_seqlen_h) / params.target_group_size;
          n_masking_block_min = is_jump
              ? (actual_seqlen_h + actual_seqlen_offset +
                 target_index * params.target_group_size + last_page_offset) / kBlockN
              : n_masking_block_min;
        }
      }
      if constexpr (Is_context) {
        n_masking_block_min = is_in_mixed_context ? n_block_min : n_masking_block_min;
        n_masking_block_max = is_in_mixed_context ? n_block_max : n_masking_block_max;
      }
      const int n_masking_steps = (!Is_causal || is_in_context)
          ? 0 : n_masking_block_max - n_masking_block_min;

      // Arbitrary func GMEM tensors (only needed when Is_arbitrary, but declared always for lambda).
      Tensor mMaxFunc = make_tensor(
          make_gmem_ptr(reinterpret_cast<int*>(params.func_ptr) + binfo.sum_s_q),
          make_shape(Int<1>{}, Int<kNFunc/2 + 1>{}, actual_seqlen_q),
          make_stride(params.func_head_stride, 2 * params.func_ids_stride, _1{}));
      Tensor mMinFunc = make_tensor(
          make_gmem_ptr(reinterpret_cast<int*>(params.func_ptr) + binfo.sum_s_q + params.func_ids_stride),
          make_shape(Int<1>{}, Int<kNFunc/2>{}, actual_seqlen_q),
          make_stride(params.func_head_stride, 2 * params.func_ids_stride, _1{}));
      Tensor gMaxFunc = local_tile(mMaxFunc(Int<0>{}, _, _),
          make_shape(Int<kNFunc/2 + 1>{}, Int<kBlockM>{}), make_coord(Int<0>{}, m_block));
      Tensor gMinFunc = local_tile(mMinFunc(Int<0>{}, _, _),
          make_shape(Int<kNFunc/2>{}, Int<kBlockM>{}), make_coord(Int<0>{}, m_block));

      // SMEM tensors for Is_arbitrary.
      Tensor sValidBlockIds = make_tensor(
          make_smem_ptr(reinterpret_cast<int*>(smem_ + Kernel_traits::kSmemWsValidBlockIdsOffset)),
          typename Kernel_traits::SmemLayoutValidBlockIds{});
      Tensor sFunc_min = make_tensor(make_smem_ptr(sf_min_ptr), typename Kernel_traits::SmemLayoutMinFunc{});
      Tensor sFunc_max = make_tensor(make_smem_ptr(sf_max_ptr), typename Kernel_traits::SmemLayoutMaxFunc{});

      // Is_arbitrary setup: warp 1 (threads 32-63) does the computation; load warp just syncs.
      if constexpr (Is_arbitrary) {
        const int lane_id = cutlass::canonical_lane_idx();
        const int warp_id = cutlass::canonical_warp_idx_sync();
        if (warp_id == 1) {
          *sn_valid_block_max = 0;
          sFunc_min[0] = 0;
          __syncwarp();
          int f_min = INT_MAX, f_max = INT_MIN;
          const int base_row = m_block * kBlockM;
          for (int i = 0; i < size<0>(gMinFunc); i++) {
            for (int j = lane_id; j < size<1>(gMinFunc); j += 32) {
              const int row = base_row + j;
              if (row < actual_seqlen_q) if (f_min > gMinFunc(i, j)) f_min = gMinFunc(i, j);
            }
            warpReduce(f_min, MinOp<int>());
            if (lane_id == 0) sFunc_min[i+1] = f_min;
            f_min = INT_MAX;
          }
          for (int i = 0; i < size<0>(gMaxFunc); i++) {
            for (int j = lane_id; j < size<1>(gMaxFunc); j += 32) {
              const int row = base_row + j;
              if (row < actual_seqlen_q) if (f_max < gMaxFunc(i, j)) f_max = gMaxFunc(i, j);
            }
            warpReduce(f_max, MaxOp<int>());
            if (lane_id == 0) sFunc_max[i] = f_max;
            f_max = INT_MIN;
          }
          if (lane_id == 0) {
            for (int n_block = n_block_min; n_block < n_block_max; n_block++) {
              int b_max = (n_block + 1) * kBlockN, b_min = n_block * kBlockN;
              for (int i = 0; i < (kNFunc + 1)/2; i++) {
                int fmin = sFunc_min[i], fmax = sFunc_max[i];
                if (fmax <= fmin) continue;
                bool c1 = fmin <= b_min && fmax > b_min;
                bool c2 = fmin >= b_min && b_max > fmin;
                bool c3 = fmin >= b_min && fmax < b_max;
                if (c1 || c2 || c3) {
                  sValidBlockIds[*sn_valid_block_max] = n_block;
                  (*sn_valid_block_max)++;
                  break;
                }
              }
            }
          }
        }
        __syncthreads();  // S_arb: sValidBlockIds written by warp 1, visible to all (incl. load warp).
        n_block_max = *sn_valid_block_max;
        n_block_min = 0;
      }

      // Early exit 2 (after Is_arbitrary): math warps write zeros; non-persistent paths still
      // consume the per-tile S1.
      if (((Is_causal || Is_local || Is_arbitrary) && n_block_max <= n_block_min) ||
          m_block * kBlockM >= actual_seqlen_q) {
        using OutElement = typename Kernel_traits::OutputType;
        Tensor mO = make_tensor(
            make_gmem_ptr(reinterpret_cast<OutElement*>(params.o_ptr) + binfo.q_offset(params.o_row_stride)),
            make_shape(actual_seqlen_q, params.h, params.d),
            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));
        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx_math);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<OutElement>(shape(tOgO));
        clear(tOrO);
        Tensor cO_zr = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO_zr);
        flash::copy<false, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO,
            actual_seqlen_q_padded - m_block * kBlockM);
        if constexpr (!Use_persistent) {
          __syncthreads();  // S1
        }
        return;
      }

      // WS SMEM layouts. Kernel_traits selects SW32/SW64/SW128 by headDim.
      using SmemLayoutQ_SW128  = typename Kernel_traits::SmemLayoutQ_TMA;
      using SmemLayoutK_SW128  = typename Kernel_traits::SmemLayoutK_TMA;

      constexpr int kSmemKVElems = kBlockN * kHeadDim;

      // smem_base32: shared-memory base address as uint32 for on-demand barrier/KV
      // pointer computation in the main loop — replaces 6 persistent pointer arrays.
      const uint32_t smem_base32 =
          static_cast<uint32_t>(__cvta_generic_to_shared(smem_));
      uint64_t* q_ready_mbar_ptr = reinterpret_cast<uint64_t*>(
          reinterpret_cast<char*>(smem_) + kSmemMbar0Offset + 64);
      uint64_t* q_empty_mbar_ptr = reinterpret_cast<uint64_t*>(
          reinterpret_cast<char*>(smem_) + kSmemMbar0Offset + 72);

      // SF SMEM pointers.
      static constexpr int kSmemSFOffset_WS = Kernel_traits::kSmemWsDataSizePadded;
      int32_t* smem_sfa_ptr = reinterpret_cast<int32_t*>(smem_ + kSmemSFOffset_WS);
      int32_t* smem_sfp_ptr = smem_sfa_ptr + kBlockM;

      using SmemLayoutSFA = typename BS1::SmemLayoutSFA;
      using SmemLayoutSFB = typename BS1::SmemLayoutSFB;
      using SmemLayoutSFV = typename BS2::SmemLayoutSFB;
      Tensor sSFA_ = make_tensor(make_smem_ptr(smem_sfa_ptr), SmemLayoutSFA{});
      auto sSFA = as_position_independent_swizzle_tensor(sSFA_);
      Tensor sSFP_ = make_tensor(make_smem_ptr(smem_sfp_ptr), SmemLayoutSFA{});
      auto sSFP = as_position_independent_swizzle_tensor(sSFP_);
      using SmemLayoutRab_TMA = typename Kernel_traits::SmemLayoutRab_TMA;
      RabElement* smem_rab = reinterpret_cast<RabElement*>(
          reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsRabOffset);
      Tensor sRab = make_tensor(make_smem_ptr(smem_rab), SmemLayoutRab_TMA{});

      // MMA tiled objects (use tidx_math = tidx for math warps).
      typename BS1::TiledMma tiled_mma_g1;
      auto thr_mma_g1 = tiled_mma_g1.get_thread_slice(tidx_math);
      typename BS2::TiledMma tiled_mma_g2;
      auto thr_mma_g2 = tiled_mma_g2.get_thread_slice(tidx_math);

      Tensor acc_o = partition_fragment_C(tiled_mma_g2, Shape<Int<kBlockM>, Int<kHeadDimGemm2>>{});
      clear(acc_o);

      // s2r copy atoms. Q/P use explicit CuTe LDSM_N tiling below; K uses the
      // standard B tiled copy, and V^T still uses LDSM_T inline PTX.
      // SFB/SFV: get_layoutSFB_TV degenerates under AtomLayout <_8,_1,_1> (ThrN=1 → all
      // threads map to SMEM position 0). Use direct SMEM reads at the call site instead.
      auto s2r_copy_SFP = make_tiled_copy_impl(typename BS2::SmemCopyAtomSF{},
          BS2::get_layoutSFA_TV(tiled_mma_g2), make_shape(size<0>(tile_shape(tiled_mma_g2)), _1{}));
      auto s2r_thr_SFP  = s2r_copy_SFP.get_thread_slice(tidx_math);
      auto s2r_copy_B_g1 = make_tiled_copy_B(typename BS1::SmemCopyAtomB{}, tiled_mma_g1);
      auto s2r_thr_B_g1  = s2r_copy_B_g1.get_thread_slice(tidx_math);

      if constexpr (!Use_persistent) {
        __syncthreads();  // S1: tma_mbar0/1 and math_mbar0/1 visible to all warps.
      }

      FP8Elem* q_persist_base = reinterpret_cast<FP8Elem*>(
          reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsQPersistOffset);
      Tensor sQ_persist = make_tensor(
          make_smem_ptr(q_persist_base),
          SmemLayoutQ_SW128{});
      auto sQ_persist_pi = as_position_independent_swizzle_tensor(sQ_persist);

      // Q+SFA TMA is issued by load warp 8 into sQ_persist and smem_sfa_ptr.
      // Math waits q_ready, loads Q/SFA into registers, then releases q_empty.
      wait_mbar_parity(q_ready_mbar_ptr, (uint32_t)q_tma_parity);
      q_tma_parity ^= 1;

	      // Mask and RAB consumption are force-inlined helpers.  Their call sites
	      // below now show the GEMM1 -> mask -> RAB -> activation handoff directly.

      // SFP (unit scale for P) — separate buffer so real SFA SMEM stays valid for per-tile GEMM1 s2r.
      for (int i = tidx_math; i < kBlockM; i += kNMathThreads)
        smem_sfp_ptr[i] = 0x7f7f7f7f;

      // ===== MATH WARP DOUBLE-BUFFER QMMA CONSUMER LOOP =====
      // Phase 11: s2r V^T uses ldmatrix.m16n16.x2.trans.b8 (LDSM_T) directly from MN_SW128 SMEM.
      // Under AtomLayout <_8,_1,_1>, all 8 warps load all 4 N-slabs (nw=0..3 outer loop);
      // 16B alignment: d_start = nw*32+d_mat ∈ multiples of 16, XOR swizzle_xor also multiple of 16.
      static_assert(kHeadDim % 32 == 0 && kBlockN % 16 == 0,
          "LDSM_T requires kHeadDim divisible by 32 and kBlockN divisible by 16.");
      // ── N-loop tile-invariants (Opt A): hoisted from per-tile load ───────────────────────
      // Q (sQ_persist) and SFA are written by TMA before the loop (guaranteed visible after
      // wait_mbar_parity at S2) and never change.  Hoisting eliminates 4 ldmatrix + 1 LDS
      // per N-tile.
      // NOTE: tCrSFP is NOT hoisted — smem_sfp_ptr is written by distributed thread writes
      // (not TMA) and requires the long per-tile gap (mbarrier wait + GEMM1, ~500 cycles)
      // to guarantee visibility.  Hoisting would introduce a race with no explicit barrier.
      Tensor tCrSFA = BS1::partition_fragment_SFA(sSFA(_,_,_0{}), thr_mma_g1);
      const int warp_m  = tidx_math / 32;
      const int lane    = tidx_math & 31;
      const int q_mat_num = lane >> 3;
      const int q_mat_row = lane & 7;
      const int q_m_abs = warp_m * 16 + ((q_mat_num & 1) << 3) + q_mat_row;
      const int q_k_half = q_mat_num >> 1;
      const int sfa_row = warp_m * 16 + 8 * (lane & 1) + (lane >> 2);
      tCrSFA(0, 0, 0)  = smem_sfa_ptr[sfa_row];
      auto tCrSFA_frg = BS1::transform_fragment_for_qmma(tCrSFA);
      auto q_ldsm_atom = Copy_Atom<SM75_U32x4_LDSM_N, uint32_t>{};
      auto sQ_persist_u128 = recast<uint128_t>(sQ_persist);

      Tensor tCrQ = thr_mma_g1.partition_fragment_A(sQ_persist_pi);
      if constexpr (kHeadDim <= 128) {
        clear(tCrQ);
        constexpr int kQBlockCount = kHeadDim / 32;
        auto tXrQ = recast<uint32_t>(tCrQ);
        const int q_src_offset = sQ_persist_u128.layout()(q_m_abs, q_k_half);
        Tensor tXsQ = make_tensor(
            sQ_persist_u128.data() + q_src_offset,
            cute::Layout<Shape<_1, Int<kQBlockCount>>, Stride<_0, _2>>{});
        Tensor tXrQ_copy = make_tensor(
            tXrQ.data(),
            cute::Layout<Shape<_4, Int<kQBlockCount>>, Stride<_1, _4>>{});
        cute::copy(q_ldsm_atom, tXsQ, tXrQ_copy);
      }
      if constexpr (kHeadDim <= 128) {
        if ((tidx_math & 31) == 0) {
          arrive_mbar(q_empty_mbar_ptr);
        }
      }

      // Per-tile lambda (Opt C): instantiated with kIsMasking=true (Phase 1, causal diagonal)
      // and kIsMasking=false (Phase 2, steady-state unmasked) to allow compile-time dead-code
      // elimination of apply_mask_bs in the hot path.  n_valid_ref is modified in-place by
      // the is_jump adjustment; math_stage persists across tiles and flips after every N tile.
      constexpr bool kStoreOInMainloop =
          kHeadDim <= 128 &&
          Is_causal && !Is_target && !Is_context && !Is_arbitrary && !Is_local;
      bool o_epilogue_done = false;
      auto store_o_epilogue = [&]() {
        for (int i = 0; i < size(acc_o); ++i) acc_o(i) /= params.scaling_seqlen;
        using OutElement = typename Kernel_traits::OutputType;
        Tensor rO = make_tensor_like<OutElement>(acc_o);
        flash::convert_type_safe(acc_o, rO);

        if constexpr (!Kernel_traits::kUseTmaOStore) {
          Tensor cO_id = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimGemm2>>{});
          Tensor tOcO = thr_mma_g2.partition_C(cO_id);
          OutElement* gO_ptr = reinterpret_cast<OutElement*>(params.o_ptr)
              + binfo.q_offset(params.o_row_stride)
              + bidh * params.o_head_stride;
          const int valid_rows = actual_seqlen_q - m_block * kBlockM;
          CUTE_UNROLL
          for (int flat = 0; flat < size(rO); ++flat) {
            const auto coord = tOcO(flat);
            const int m_rel = int(get<0>(coord));
            const int n_pos = int(get<1>(coord));
            if (m_rel < valid_rows && n_pos < kHeadDim) {
              gO_ptr[(m_block * kBlockM + m_rel) * params.o_row_stride + n_pos] = rO(flat);
            }
          }
        } else {
          static_assert(
              Kernel_traits::kSmemWsOStoreBytes >= kBlockM * kHeadDim * (int)sizeof(OutElement),
              "O SMEM buffer is too small.");
          if constexpr (Kernel_traits::kUseAliasedOBuffer) {
            // The aliased D256 path writes O over the KV staging region.  All math
            // warps must finish consuming V before any warp starts STSM into it.
            asm volatile("bar.sync 1, 256;\n" : : : "memory");
          }
          char* smem_o = reinterpret_cast<char*>(smem_) +
              (Kernel_traits::kUseAliasedOBuffer ? 0 : Kernel_traits::kSmemWsOOffset);

          // Wait until the O buffer is no longer owned by the previous TMA store.
          wait_mbar_parity(
              smem_base32 + (uint32_t)kSmemMbar0Offset + 96u,
              (uint32_t)o_empty_wait_parity);
          o_empty_wait_parity ^= 1;
          Tensor sO_tma = make_tensor(
              make_smem_ptr(reinterpret_cast<OutElement*>(smem_o)),
              typename Kernel_traits::SmemLayoutWsO_TMA{});
          auto sO_swz = as_position_independent_swizzle_tensor(sO_tma);
          {
            // STSM writes the same swizzled SMEM layout consumed by O TMA store.
            static_assert(sizeof(OutElement) == 2, "stmatrix.m8n8.b16 requires 2-byte elements");
            static_assert(kHeadDim <= 256, "O STSM path currently supports D32/D64/D128/D256");

            auto rO_u32 = recast<uint32_t>(rO);
            auto sO_u128 = recast<uint128_t>(sO_swz);
            auto stsm_atom = Copy_Atom<SM90_U32x4_STSM_N, uint32_t>{};
            constexpr int kOStsmNGroups = kHeadDim / 32;

            const int lane        = tidx_math & 31;
            const int lq          = lane >> 2;   // lane-quad (0..7): selects data row within 8x8 matrix
            const int lqt         = lane & 3;    // lane-quad-thread (0..3): selects data col-group
            const int warp_m      = tidx_math >> 5;  // M-warp index (0..7)
            const int addr_row    = ((lq & 1) << 2) | lqt;  // STSM address-row within matrix (0..7)
            const int mat_in_lane = lq >> 1;                 // which of 4 matrices this thread addresses

            CUTE_UNROLL
            for (int m_grp = 0; m_grp < 2; ++m_grp) {
              const int row = warp_m * 16 + m_grp * 8 + addr_row;

              CUTE_UNROLL
              for (int n_grp = 0; n_grp < kOStsmNGroups; ++n_grp) {
                // This thread addresses row addr_row of matrix mat_in_lane,
                // starting at N-column (n_grp*32 + mat_in_lane*8).
                const int col = n_grp * 32 + mat_in_lane * 8;

                int j0, j1, j2, j3;
                if constexpr (kHeadDimGemm2 == 64) {
                  // BS2 D32/D64 uses linear 64-wide N atom order: 0,8,16,...,56.
                  j0 = 4 * n_grp + 0;
                  j1 = 4 * n_grp + 1;
                  j2 = 4 * n_grp + 2;
                  j3 = 4 * n_grp + 3;
                } else if constexpr (kHeadDimGemm2 == 128) {
                  // BS2 D128 uses PermMmaTileN <_8,_4,_4>:<_1,_32,_8>.
                  j0 = n_grp;
                  j1 = n_grp + kOStsmNGroups;
                  j2 = n_grp + 2 * kOStsmNGroups;
                  j3 = n_grp + 3 * kOStsmNGroups;
                } else {
                  // D256 is two D128 slabs in C-fragment order.
                  const int slab = n_grp >> 2;
                  const int grp = n_grp & 3;
                  const int base = slab * 16 + grp;
                  j0 = base;
                  j1 = base + 4;
                  j2 = base + 8;
                  j3 = base + 12;
                }
                Tensor tXsO = make_tensor(
                    sO_u128.data() + sO_u128.layout()(row, col / 8),
                    Layout<_1>{});
                if constexpr (kHeadDimGemm2 == 64) {
                  Tensor tXrO = make_tensor(
                      rO_u32.data() + 2 * j0 + m_grp,
                      Layout<Shape<_4>, Stride<_2>>{});
                  cute::copy(stsm_atom, tXrO, tXsO);
                } else {
                  static_assert(
                      2 * (4) == 8,
                      "D128/D256 STSM source order expects 8-u32 stride.");
                  Tensor tXrO = make_tensor(
                      rO_u32.data() + 2 * j0 + m_grp,
                      Layout<Shape<_4>, Stride<_8>>{});
                  cute::copy(stsm_atom, tXrO, tXsO);
                }
              }
            }
          }

          // For partial tiles (last tile of a varlen sequence): zero SMEM rows [valid_rows, kBlockM)
          // so TMA bulk store does not write garbage to GMEM beyond actual_seqlen_q.
          const int valid_rows = actual_seqlen_q - m_block * kBlockM;
          if (valid_rows < kBlockM) {
            // Ensure all STSM stores are complete before any warp zeros OOB rows.
            // Full tiles skip this: o_ready is initialized with count=8, so the O-store warp
            // cannot proceed until each math warp has arrived after its own STSM stores.
            asm volatile("bar.sync 1, 256;\n" : : : "memory");
            const int oob_elems = (kBlockM - valid_rows) * kHeadDim;
            for (int i = tidx_math; i < oob_elems; i += kNMathThreads) {
              const int row_offset = i / kHeadDim;
              const int row = valid_rows + row_offset;
              const int col = i - row_offset * kHeadDim;
              sO_swz(row, col) = OutElement(0);
            }
            asm volatile("bar.sync 1, 256;\n" : : : "memory");  // OOB zeros visible before TMA.
          }

          // Produce o_ready[0]. O-store warp owns the TMA store and releases
          // o_empty[0] after the store completes.
          if ((tidx_math & 31) == 0) {
            const uint32_t o_ready_addr =
                smem_base32 + (uint32_t)kSmemMbar0Offset + 80u;
            arrive_mbar(o_ready_addr);
          }
        }
      };

      auto run_n_tile = [&](int nb, int n_valid_ref, int masking_step, bool do_mask_runtime,
                            bool store_o_after, auto kMaskMode_c) {
        constexpr int kMaskMode = decltype(kMaskMode_c)::value;
        constexpr bool kIsMasking = kMaskMode == 1;
        constexpr bool kRuntimeMasking = kMaskMode == 2;

        // Stage-dependent SMEM pointers — pure pointer arithmetic, no TMA dependency.
        int32_t* smem_sfb_cur = smem_sfa_ptr + 2 * kBlockM + math_stage * kBlockN;
        FP8Elem* sK_cur  = reinterpret_cast<FP8Elem*>(smem_q) + math_stage * 2 * kSmemKVElems;
        FP8Elem* sVt_cur = reinterpret_cast<FP8Elem*>(smem_q) + kSmemKVElems + math_stage * 2 * kSmemKVElems;

        // GEMM1 only needs K/SFB. Wait V/SFV later, right before GEMM2.
        {
          const int cur_parity = math_stage ? tma_parity1 : tma_parity0;
          uint32_t kaddr = smem_base32 + (uint32_t)kSmemMbar0Offset + (uint32_t)(math_stage * 8);
          wait_mbar_parity(kaddr, (uint32_t)cur_parity);
        }
        // mbarrier.test_wait already guarantees K/SFB data visibility in SMEM.
        // K and V have separate empty counters: K is released after K/SFB s2r,
        // V is waited/released later around V/SFV s2r.
        asm volatile("" ::: "memory");

        Tensor sSFB_ = make_tensor(make_smem_ptr(smem_sfb_cur), SmemLayoutSFB{});
        auto sSFB = as_position_independent_swizzle_tensor(sSFB_);

        // GEMM1: acc_s += Q × K^T (block-scaled QMMA).
        // Full-K=128 copy: make_tiled_copy_A/B expects the full TiledMMA tile;
        // cute::gemm internally loops 4×K=32 when the fragment covers K=128.
        Tensor acc_s = partition_fragment_C(tiled_mma_g1, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);

        Tensor tCrSFB = BS1::partition_fragment_SFB(sSFB(_,_,_0{}), thr_mma_g1);
        {
          // get_layoutSFB_TV degenerates under <_8,_1,_1>: ThrN=1 causes all threads to map
          // to SMEM position 0 via the stride-0 btile, leaving tCrSFB[1..15] uninitialized.
          // Direct load from smem_sfb_ptr (LINEAR N-row order [0..kBlockN-1]).
          // partition_fragment_SFB->make_fragment_like compacts the (4,4) N-rep sub-modes in
          // column-major order (first dim fastest), giving fragment nr -> N_base = 8*nr (LINEAR).
          // n_row_sfb: this thread's N-offset within the 8-wide N-atom (lane >> 2 = 0..7).
          const int n_row_sfb = (tidx_math & 31) >> 2;
          constexpr int kNAtomsSFB = kBlockN / 8;
          CUTE_UNROLL
          for (int nr = 0; nr < kNAtomsSFB; ++nr) {
            const int N_base = nr * 8;  // LINEAR: 0, 8, 16, ..., 120
            tCrSFB(0, nr, 0) = smem_sfb_cur[N_base + n_row_sfb];
          }
        }
        if constexpr (kHeadDim <= 128) {
          auto sK_cur_pi = as_position_independent_swizzle_tensor(
              make_tensor(make_smem_ptr(sK_cur), SmemLayoutK_SW128{}));
          auto tCrSFB_frg = BS1::transform_fragment_for_qmma(tCrSFB);
          Tensor tCrK = thr_mma_g1.partition_fragment_B(sK_cur_pi);
          clear(tCrK);
          // GEMM1: tCrQ and tCrSFA_frg are N-loop invariants hoisted above the loop.
          auto tXsK = s2r_thr_B_g1.partition_S(sK_cur_pi);
          auto tXrK = s2r_thr_B_g1.retile_D(tCrK);
          cute::copy(s2r_copy_B_g1, tXsK, tXrK);
          if ((tidx_math & 31) == 0) {
            uint32_t k_empty_addr =
                smem_base32 + (uint32_t)kSmemMbar0Offset + 32u + (uint32_t)(math_stage * 8);
            arrive_mbar(k_empty_addr);
          }
          cute::gemm(tiled_mma_g1,
              make_zip_tensor(tCrQ, tCrSFA_frg(_,_,_,_0{})),
              make_zip_tensor(tCrK, tCrSFB_frg(_,_,_,_0{})),
              acc_s);
        } else if constexpr (kHeadDim == 256) {
          using SmemLayoutQChunk_SW128 = decltype(tile_to_shape(
              GMMA::Layout_K_SW128_Atom<FP8Elem>{},
              Shape<Int<kBlockM>, Int<128>>{}));
          FP8Elem* q_persist_base = reinterpret_cast<FP8Elem*>(
              reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemWsQPersistOffset);

          {
            Tensor tCrSFA0 = make_tensor_like<int32_t>(tCrSFA);
            Tensor tCrSFB0 = make_tensor_like<int32_t>(tCrSFB);
            CUTE_UNROLL
            for (int i = 0; i < size(tCrSFA); ++i) {
              tCrSFA0(i) = hstu_ws_replicate_e8m0_lane(tCrSFA(i), 0);
            }
            CUTE_UNROLL
            for (int i = 0; i < size(tCrSFB); ++i) {
              tCrSFB0(i) = hstu_ws_replicate_e8m0_lane(tCrSFB(i), 0);
            }
            auto tCrSFA0_frg = BS1::transform_fragment_for_qmma(tCrSFA0);
            auto tCrSFB0_frg = BS1::transform_fragment_for_qmma(tCrSFB0);

            Tensor sQ_chunk0 = make_tensor(
                make_smem_ptr(q_persist_base),
                SmemLayoutQChunk_SW128{});
            auto sQ_chunk0_pi = as_position_independent_swizzle_tensor(sQ_chunk0);
            Tensor tCrQ0 = thr_mma_g1.partition_fragment_A(sQ_chunk0_pi);
            clear(tCrQ0);
            auto tXrQ0 = recast<uint32_t>(tCrQ0);
            const int q_src_offset0 = sQ_persist_u128.layout()(q_m_abs, q_k_half);
            Tensor tXsQ0 = make_tensor(
                sQ_persist_u128.data() + q_src_offset0,
                cute::Layout<Shape<_1, _4>, Stride<_0, _2>>{});
            Tensor tXrQ0_copy = make_tensor(
                tXrQ0.data(),
                cute::Layout<Shape<_4, _4>, Stride<_1, _4>>{});
            cute::copy(q_ldsm_atom, tXsQ0, tXrQ0_copy);

            Tensor sK_full = make_tensor(
                make_smem_ptr(sK_cur),
                SmemLayoutK_SW128{});
            Tensor sK_chunk0 = local_tile(
                sK_full,
                Shape<Int<kBlockN>, Int<128>>{},
                make_coord(0, 0));
            auto sK_chunk0_pi = as_position_independent_swizzle_tensor(sK_chunk0);
            Tensor tCrK0 = thr_mma_g1.partition_fragment_B(sK_chunk0_pi);
            clear(tCrK0);
            auto tXsK0 = s2r_thr_B_g1.partition_S(sK_chunk0_pi);
            auto tXrK0 = s2r_thr_B_g1.retile_D(tCrK0);
            cute::copy(s2r_copy_B_g1, tXsK0, tXrK0);

            cute::gemm(tiled_mma_g1,
                make_zip_tensor(tCrQ0, tCrSFA0_frg(_,_,_,_0{})),
                make_zip_tensor(tCrK0, tCrSFB0_frg(_,_,_,_0{})),
                acc_s);
          }

          {
            Tensor tCrSFA1 = make_tensor_like<int32_t>(tCrSFA);
            Tensor tCrSFB1 = make_tensor_like<int32_t>(tCrSFB);
            CUTE_UNROLL
            for (int i = 0; i < size(tCrSFA); ++i) {
              tCrSFA1(i) = hstu_ws_replicate_e8m0_lane(tCrSFA(i), 1);
            }
            CUTE_UNROLL
            for (int i = 0; i < size(tCrSFB); ++i) {
              tCrSFB1(i) = hstu_ws_replicate_e8m0_lane(tCrSFB(i), 1);
            }
            auto tCrSFA1_frg = BS1::transform_fragment_for_qmma(tCrSFA1);
            auto tCrSFB1_frg = BS1::transform_fragment_for_qmma(tCrSFB1);

            Tensor sQ_chunk1 = make_tensor(
                make_smem_ptr(q_persist_base),
                SmemLayoutQChunk_SW128{});
            auto sQ_chunk1_pi = as_position_independent_swizzle_tensor(sQ_chunk1);
            Tensor tCrQ1 = thr_mma_g1.partition_fragment_A(sQ_chunk1_pi);
            clear(tCrQ1);
            auto tXrQ1 = recast<uint32_t>(tCrQ1);
            const int q_src_offset1 = sQ_persist_u128.layout()(q_m_abs, 8 + q_k_half);
            Tensor tXsQ1 = make_tensor(
                sQ_persist_u128.data() + q_src_offset1,
                cute::Layout<Shape<_1, _4>, Stride<_0, _2>>{});
            Tensor tXrQ1_copy = make_tensor(
                tXrQ1.data(),
                cute::Layout<Shape<_4, _4>, Stride<_1, _4>>{});
            cute::copy(q_ldsm_atom, tXsQ1, tXrQ1_copy);

            Tensor sK_full = make_tensor(
                make_smem_ptr(sK_cur),
                SmemLayoutK_SW128{});
            Tensor sK_chunk1 = local_tile(
                sK_full,
                Shape<Int<kBlockN>, Int<128>>{},
                make_coord(0, 1));
            auto sK_chunk1_pi = as_position_independent_swizzle_tensor(sK_chunk1);
            Tensor tCrK1 = thr_mma_g1.partition_fragment_B(sK_chunk1_pi);
            clear(tCrK1);
            auto tXsK1 = s2r_thr_B_g1.partition_S(sK_chunk1_pi);
            auto tXrK1 = s2r_thr_B_g1.retile_D(tCrK1);
            cute::copy(s2r_copy_B_g1, tXsK1, tXrK1);

            cute::gemm(tiled_mma_g1,
                make_zip_tensor(tCrQ1, tCrSFA1_frg(_,_,_,_0{})),
                make_zip_tensor(tCrK1, tCrSFB1_frg(_,_,_,_0{})),
                acc_s);
          }

          if ((tidx_math & 31) == 0) {
            uint32_t k_empty_addr =
                smem_base32 + (uint32_t)kSmemMbar0Offset + 32u + (uint32_t)(math_stage * 8);
            arrive_mbar(k_empty_addr);
          }
        }
        constexpr bool kUseRabSkipMasked =
            (Is_causal && !Is_context && !Is_target && !Is_local && !Is_arbitrary) ||
            Is_local || Is_arbitrary || (Is_context && !Paged_KV) || (Is_target && Paged_KV);
        if constexpr (!(Has_rab && kUseRabSkipMasked)) {
          hstu_ws_consume_rab_bs<Kernel_traits>(
              acc_s, nb, false, rab_ready_wait_parity0, smem_base32, tidx_math,
              thr_mma_g1, sRab, m_block, actual_seqlen_q, actual_seqlen_k,
              actual_seqlen_h, n_block_paged, last_page_offset);
        } else if constexpr (Has_rab && Paged_KV && Is_target) {
          hstu_ws_consume_rab_paged_target_bs<Kernel_traits>(
              acc_s, rab_ready_wait_parity0, smem_base32, tidx_math,
              thr_mma_g1, sRab);
        }
        // Masking (Opt C: compile-time specialized).  Most masked RAB paths
        // apply the mask first so the SMEM add can skip -inf entries.  Paged
        // target consumes RAB before masking to keep the hot path spill-free;
        // the target/causal mask overwrites invalid entries below.
        // kIsMasking=true  (Phase 1): always apply_mask_bs (causal diagonal or Is_arbitrary/Is_local).
        // kIsMasking=false (Phase 2): steady-state; skip diagonal masking; varlen-end check only.
        bool mask_applied = false;
        if constexpr (Is_arbitrary || Is_local || kIsMasking) {
          hstu_ws_apply_mask_bs<Kernel_traits, Target_group_size_1>(
              acc_s, nb, params, thr_mma_g1, gMinFunc, gMaxFunc, m_block,
              actual_seqlen_k, actual_seqlen_h, actual_seqlen_c,
              actual_seqlen_offset, last_page_offset);
          mask_applied = true;
        } else if constexpr (kRuntimeMasking) {
          if (do_mask_runtime) {
            hstu_ws_apply_mask_bs<Kernel_traits, Target_group_size_1>(
                acc_s, nb, params, thr_mma_g1, gMinFunc, gMaxFunc, m_block,
                actual_seqlen_k, actual_seqlen_h, actual_seqlen_c,
                actual_seqlen_offset, last_page_offset);
            mask_applied = true;
          } else {
            if ((nb + 1) * kBlockN > actual_seqlen_h) {
              hstu_ws_apply_mask_bs<Kernel_traits, Target_group_size_1>(
                  acc_s, nb, params, thr_mma_g1, gMinFunc, gMaxFunc, m_block,
                  actual_seqlen_k, actual_seqlen_h, actual_seqlen_c,
                  actual_seqlen_offset, last_page_offset);
              mask_applied = true;
            }
          }
        } else {
          if ((nb + 1) * kBlockN > actual_seqlen_h) {
            hstu_ws_apply_mask_bs<Kernel_traits, Target_group_size_1>(
                acc_s, nb, params, thr_mma_g1, gMinFunc, gMaxFunc, m_block,
                actual_seqlen_k, actual_seqlen_h, actual_seqlen_c,
                actual_seqlen_offset, last_page_offset);
            mask_applied = true;
          }
        }
        if constexpr (Has_rab && kUseRabSkipMasked) {
          if constexpr (Paged_KV && Is_target) {
            // Paged target consumed RAB before masking; invalid entries are
            // overwritten by the target/causal mask.
          } else {
            hstu_ws_consume_rab_bs<Kernel_traits>(
                acc_s, nb, mask_applied, rab_ready_wait_parity0, smem_base32, tidx_math,
                thr_mma_g1, sRab, m_block, actual_seqlen_q, actual_seqlen_k,
                actual_seqlen_h, n_block_paged, last_page_offset);
          }
        }
        for (int i = 0; i < size(acc_s); ++i) acc_s(i) *= params.alpha;
        fast_silu(acc_s);

        // ── P pre-conversion: F32 → packed FP8 ───────────────────────────────
        // Problem: during the P-write loop below, both acc_s (64 F32 = 64 regs)
        // and acc_o (64 F32 = 64 regs) must be live, and the SW128 swizzle address
        // LOP3 computation needs ~20 temp regs.  These overlap with acc_o's lower
        // register range (R4–R23), causing the compiler to spill acc_o (10×STL.64 +
        // 10×LDL.LU.64 = 20 local-memory accesses per loop iteration).
        //
        // Fix: pre-convert acc_s (64 F32) → acc_s_packed (16 uint32, 4 FP8/reg)
        // BEFORE the bar.sync + P-write loop.  After this block acc_s is dead
        // (last read is here), so its 64 registers are freed and available to the
        // LOP3 address computation, eliminating the acc_o spill.
        //
        // Build GEMM2 A-fragment P directly in registers via warp shuffle.
        // With AtomLayout <_8,_1,_1> (1 N-warp), all N-columns of P live in each
        // warp's own registers — no cross-warp SMEM staging or bar.sync is needed.
        // permute_acc_s_packed_to_tCrP packs acc_s in flat C-fragment order,
        // then performs the C-fragment → A-fragment register permutation.
        // sPbuf_pi is defined for partition_fragment_A shape inference only; sK_cur is
        // not written and remains available for the load warp's next TMA fill.
        auto sPbuf_pi = [&]() {
          if constexpr (kBlockN == 64) {
            using SmemLayoutP_SW64 = decltype(tile_to_shape(
                GMMA::Layout_K_SW64_Atom<FP8Elem>{},
                Shape<Int<kBlockM>, Int<kBlockN>>{}));
            Tensor sPbuf = make_tensor(make_smem_ptr(sK_cur), SmemLayoutP_SW64{});
            return as_position_independent_swizzle_tensor(sPbuf);
          } else {
            using SmemLayoutP_SW128 = decltype(tile_to_shape(
                typename BS2::SmemLayoutAtomA{},
                Shape<Int<kBlockM>, Int<kBlockN>>{}));
            Tensor sPbuf = make_tensor(make_smem_ptr(sK_cur), SmemLayoutP_SW128{});
            return as_position_independent_swizzle_tensor(sPbuf);
          }
        }();
        Tensor tCrP = thr_mma_g2.partition_fragment_A(sPbuf_pi);
        permute_acc_s_packed_to_tCrP<kBlockN, kHeadDim>(tCrP, acc_s, tidx_math);
        // acc_s is now DEAD.

        { // tCrV, tCrSFV, tCrSFP scoped here: compiler can reuse registers freed by tCrK/tCrSFB.
          {
            const int cur_parity = math_stage ? tma_parity1 : tma_parity0;
            uint32_t vaddr =
                smem_base32 + (uint32_t)kSmemMbar0Offset + 16u + (uint32_t)(math_stage * 8);
            wait_mbar_parity(vaddr, (uint32_t)cur_parity);
          }
          if (math_stage) { tma_parity1 ^= 1; } else { tma_parity0 ^= 1; }
          asm volatile("" ::: "memory");

          // s2r V row-major — LDSM_T from K_SW128-swizzled SMEM [kBlockN, kHeadDim].
          //
          // SmemLayoutVt_SW128 = K_SW128 on [kBlockN, kHeadDim]:
          //   physical_byte(n_k, d) = n_k * kHeadDim + (d ^ ((n_k & 7) << 4)).
          //   TMA loads row-major V directly into this layout.
          //
          // LDSM_T (ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8): transposes [16 n_k × 32 d]
          // from SMEM into registers. Lane t provides address for n_k row (ni*16 + t%16),
          // d_start = dg*32 + (t>>4)*16 (matrix 0 or 1). After transpose, thread t receives
          // 16 n_k values at its specific d-column, split into 4 registers r0..r3.
          //
          // Fragment placement: PermMmaTileN = Layout<Shape<_8,_4,_4>, Stride<_1,_32,_8>>
          //   maps d-group dg (d ∈ [dg*32,(dg+1)*32)) to N-atoms nr ∈ {dg, dg+4, dg+8, dg+12}.
          //   ni maps to K-block kb=ni>>1, K-half k_h=ni&1.
          //   base = 32*(ni>>1) + 2*dg + (ni&1):
          //     r0 → slot base+0  (nr=dg,    k_h)
          //     r1 → slot base+8  (nr=dg+4,  k_h)
          //     r2 → slot base+16 (nr=dg+8,  k_h)
          //     r3 → slot base+24 (nr=dg+12, k_h)
          //   Covers all 128 uint32 slots (16 N-atoms × 4 K-blocks × 2 regs). ✓
          auto sVt_ns = [&]() {
            if constexpr (kBlockN == 64) {
              using SmemLayoutVt_SW64 = decltype(tile_to_shape(
                  GMMA::Layout_K_SW64_Atom<FP8Elem>{},
                  Shape<Int<kHeadDimGemm2>, Int<kBlockN>>{}));
              return make_tensor(make_smem_ptr(sVt_cur), SmemLayoutVt_SW64{});
            } else {
              using SmemLayoutVt_SW128 = decltype(tile_to_shape(
                  typename BS2::SmemLayoutAtomB{},
                  Shape<Int<kHeadDim>, Int<kBlockN>>{}));
              return make_tensor(make_smem_ptr(sVt_cur), SmemLayoutVt_SW128{});
            }
          }();
          Tensor tCrV = thr_mma_g2.partition_fragment_B(sVt_ns);
          hstu_ws_load_v_ldsm_t<Kernel_traits, kBlockN, kHeadDim, kHeadDimGemm2>(
              sVt_cur, tCrV, tidx_math);

          // s2r SFP (unit scale) and SFV.
          // tCrSFP is per-tile: smem_sfp_ptr is thread-written (not TMA); the mbarrier wait +
          // GEMM1 above (~500 cycles) ensure all write-thread SFP commits are visible here.
          Tensor tCrSFP = BS2::partition_fragment_SFA(sSFP(_,_,_0{}), thr_mma_g2);
          {
            auto tXsSFP = s2r_thr_SFP.partition_S(sSFP);
            auto tXrSFP = s2r_thr_SFP.retile_D(tCrSFP);
            cute::copy(s2r_copy_SFP, tXsSFP(_,_,_,_0{}), tXrSFP);
          }
          auto tCrSFP_frg = BS2::transform_fragment_for_qmma(tCrSFP);
          int32_t* smem_sfv_cur = smem_sfa_ptr + 2 * kBlockM + 2 * kBlockN + math_stage * kBlockN;
          Tensor sSFV_ = make_tensor(make_smem_ptr(smem_sfv_cur), SmemLayoutSFV{});
          auto sSFV = as_position_independent_swizzle_tensor(sSFV_);
          Tensor tCrSFV = BS2::partition_fragment_SFB(sSFV(_,_,_0{}), thr_mma_g2);
          {
            // SFV TMA loads one scale vector per K/V tile stage (kBlockN entries).
            // GEMM2's B operand is V^T, so BS2 expects SFV across the head-dim N atoms.
            // The HSTU FP8 path stores the same V scale across the tile entries; for BN64,
            // reading nr*8 would run past the 64-entry stage. Broadcast a valid entry.
            // AtomLayoutSFB_TV = (4,8):(0,1): lane>>2 indexes the within-atom N-column offset.
            const int n_row_sfv = (tidx_math & 31) >> 2;
            const int sfv = smem_sfv_cur[n_row_sfv];
            constexpr int kNAtomsSFV = kHeadDimGemm2 / 8;
            CUTE_UNROLL
            for (int nr = 0; nr < kNAtomsSFV; ++nr) {
              tCrSFV(0, nr, 0) = sfv;
            }
          }
          auto tCrSFV_frg = BS2::transform_fragment_for_qmma(tCrSFV);

          // Signal load warp: sVt[math_stage]/SFV consumed; V load warp may overwrite.
          if ((tidx_math & 31) == 0) {
            uint32_t v_empty_addr =
                smem_base32 + (uint32_t)kSmemMbar0Offset + 48u + (uint32_t)(math_stage * 8);
            arrive_mbar(v_empty_addr);
          }

          // GEMM2: acc_o += P × V^T (block-scaled, tCrSFP_frg hoisted).
          cute::gemm(tiled_mma_g2,
              make_zip_tensor(tCrP, tCrSFP_frg(_,_,_,_0{})),
              make_zip_tensor(tCrV, tCrSFV_frg(_,_,_,_0{})),
              acc_o);
        } // tCrV, tCrSFV freed here.

        // End-of-tile cleanup: is_jump adjustment and double-buffer flip.
        if (is_jump && masking_step == n_masking_steps - 1)
          n_valid_ref = std::min(n_valid_ref, n_block_history);
        if constexpr (!Kernel_traits::kUseSingleKVStage) {
          math_stage ^= 1;
        }
        if constexpr (kStoreOInMainloop) {
          if (store_o_after) {
            store_o_epilogue();
            o_epilogue_done = true;
          }
        }
        return n_valid_ref;
      };  // end run_n_tile lambda

      // N-loop (Opt C): for causal (not arbitrary, not local), split into masked Phase 1
      // (n_masking_steps tiles) and unmasked Phase 2 (steady-state) to allow compile-time
      // dead-code elimination of apply_mask_bs in the hot path.
      if constexpr (Is_causal && !Is_arbitrary && !Is_local) {
        if constexpr (kHeadDim == 256 && !Is_target && !Is_context) {
          for (int n_valid = n_block_max - 1, masking_step = 0; n_valid >= n_block_min;
               ++masking_step, --n_valid) {
            n_valid = run_n_tile(n_valid, n_valid, masking_step,
                masking_step < n_masking_steps,
                false,
                std::integral_constant<int, 2>{});
          }
        } else {
          int n_valid    = n_block_max - 1;
          int masking_step = 0;
          // Phase 1: causal diagonal tiles — apply_mask_bs compiled in (kIsMasking=true).
          for (; n_valid >= n_block_min && masking_step < n_masking_steps; ++masking_step, --n_valid)
            n_valid = run_n_tile(n_valid, n_valid, masking_step,
                false,
                (!Is_target && !Is_context && n_valid == n_block_min),
                std::integral_constant<int, 1>{});
          // Phase 2: steady-state tiles — apply_mask_bs compile-time eliminated (kIsMasking=false).
          for (; n_valid >= n_block_min; ++masking_step, --n_valid)
            n_valid = run_n_tile(n_valid, n_valid, masking_step,
                false,
                (!Is_target && !Is_context && n_valid == n_block_min),
                std::integral_constant<int, 0>{});
        }
      } else if constexpr (Is_arbitrary || Is_local) {
        // Every tile needs masking: pass true_type so apply_mask_bs is always compiled in.
        for (int n_valid = n_block_max - 1, masking_step = 0; n_valid >= n_block_min;
             ++masking_step, --n_valid) {
          const int nb = Is_arbitrary ? int(sValidBlockIds[n_valid]) : n_valid;
          n_valid = run_n_tile(nb, n_valid, masking_step,
              false, false, std::integral_constant<int, 1>{});
        }
      } else {
        // Full attention (!Is_causal, !Is_arbitrary, !Is_local): varlen-end check only.
        for (int n_valid = n_block_max - 1, masking_step = 0; n_valid >= n_block_min;
             ++masking_step, --n_valid)
          n_valid = run_n_tile(n_valid, n_valid, masking_step,
              false, false, std::integral_constant<int, 0>{});
      }

      if constexpr (kHeadDim > 128) {
        if ((tidx_math & 31) == 0) {
          arrive_mbar(q_empty_mbar_ptr);
        }
      }

      // Complex/full paths keep the post-loop epilogue fallback.
      if constexpr (kStoreOInMainloop) {
        if (!o_epilogue_done) {
          store_o_epilogue();
        }
      } else {
        store_o_epilogue();
      }
    };

    // Static tile-id broadcast mirrors the load path exactly.
    const bool head_shared_rab = Has_rab && params.h_rab == 1 && params.h > 1;
    if constexpr (Use_paired_persistent) {
      const int num_m_block_persistent = Use_actual_target_scheduler
          ? num_m_block_scheduler
          : num_m_block_capacity;
      if constexpr (Use_actual_target_scheduler) {
        const int pairs_per_bh = (num_m_block_persistent + 1) / 2;
        const int total_tile_pairs_persistent =
            pairs_per_bh * params.h * params.b;
        #pragma unroll 1
        for (int tile_pair = int(blockIdx.x); tile_pair < total_tile_pairs_persistent; tile_pair += int(gridDim.x)) {
          const int pair_idx = tile_pair % pairs_per_bh;
          const int bh = tile_pair / pairs_per_bh;
          const int bidb = bh / params.h;
          const int bidh = bh - bidb * params.h;
          const int m_hi = num_m_block_persistent - 1 - pair_idx;
          const int m_lo = pair_idx;

          run_math_tile(bidb, bidh, m_hi);
          if (m_lo != m_hi) {
            run_math_tile(bidb, bidh, m_lo);
          }
        }
      } else {
        const int total_tiles_persistent = num_m_block_persistent * params.h * params.b;
        const int total_tile_pairs_persistent = (total_tiles_persistent + 1) / 2;
        #pragma unroll 1
        for (int tile_pair = int(blockIdx.x); tile_pair < total_tile_pairs_persistent; tile_pair += int(gridDim.x)) {
          const int paired_tile = total_tiles_persistent - 1 - tile_pair;
          const int tiles_this_pair = paired_tile == tile_pair ? 1 : 2;

          #pragma unroll 1
          for (int pair_slot = 0; pair_slot < tiles_this_pair; ++pair_slot) {
            const int tile = pair_slot == 0 ? tile_pair : paired_tile;
            const HstuWsTileCoord coord =
                hstu_ws_decode_tile(tile, num_m_block_persistent, params.h, head_shared_rab);
            run_math_tile(coord.bidb, coord.bidh, coord.m_block);
          }
        }
      }
    } else if constexpr (Use_full_persistent) {
      const int num_m_block_persistent = Use_actual_target_scheduler
          ? num_m_block_scheduler
          : num_m_block_capacity;
      const int total_tiles_persistent = num_m_block_persistent * params.h * params.b;
      #pragma unroll 1
      for (int tile = int(blockIdx.x); tile < total_tiles_persistent; tile += int(gridDim.x)) {
        const HstuWsTileCoord coord =
            hstu_ws_decode_tile(tile, num_m_block_persistent, params.h, head_shared_rab);
        run_math_tile(coord.bidb, coord.bidh, coord.m_block);
      }
    } else {
      run_math_tile(bidb_arg, bidh_arg, m_block_arg);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, typename Params, bool Target_group_size_1 = false>
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1)
hstu_fwd_kernel_blackwell_rtx_fp8_ws_tma(
    __grid_constant__ Params const params) {
  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr bool Use_full_persistent =
      !Kernel_traits::Is_causal &&
      !Kernel_traits::Is_target &&
      !Kernel_traits::Is_context &&
      !Kernel_traits::Is_local &&
      !Kernel_traits::Is_arbitrary;
  constexpr bool Use_paired_persistent =
      Kernel_traits::Is_causal &&
      !Kernel_traits::Is_context &&
      !Kernel_traits::Is_local &&
      !Kernel_traits::Is_arbitrary &&
      (!Kernel_traits::Is_target || Kernel_traits::Paged_KV) &&
      (Kernel_traits::kHeadDim <= 128 || (Kernel_traits::Paged_KV && !Kernel_traits::Has_rab));
  constexpr bool Use_persistent = Use_full_persistent || Use_paired_persistent;

  if constexpr (!Use_persistent) {
    int m_block;
    int bidh;
    int bidb = blockIdx.z;
    if constexpr (Kernel_traits::Has_rab) {
      if (params.h_rab == 1 && params.h > 1) {
        m_block = gridDim.y - blockIdx.y - 1;
        bidh    = blockIdx.x;
      } else {
        m_block = gridDim.x - blockIdx.x - 1;
        bidh    = blockIdx.y;
      }
    } else {
      m_block = gridDim.x - blockIdx.x - 1;
      bidh    = blockIdx.y;
    }
    hstu_compute_attn_1rowblock_blackwell_rtx_fp8_ws<
        Kernel_traits,
        false,
        false,
        Target_group_size_1>(params, bidb, bidh, m_block);
    return;
  }

  hstu_compute_attn_1rowblock_blackwell_rtx_fp8_ws<
      Kernel_traits,
      Use_full_persistent,
      Use_paired_persistent,
      Target_group_size_1>(
      params,
      0,
      0,
      0);
}

template <typename Kernel_traits, typename Params, bool Target_group_size_1 = false>
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1)
hstu_fwd_kernel_blackwell_rtx_fp8_ws_tma_target_persistent(
    __grid_constant__ Params const params) {
  static_assert(Kernel_traits::Is_target, "target persistent entry is target-only");
  static_assert(!Kernel_traits::Paged_KV, "target persistent entry is non-paged-only");
  static_assert(!Kernel_traits::Has_rab, "target persistent entry is no-RAB-only");

  hstu_compute_attn_1rowblock_blackwell_rtx_fp8_ws<
      Kernel_traits,
      false,
      true,
      Target_group_size_1>(params, 0, 0, 0);
}
