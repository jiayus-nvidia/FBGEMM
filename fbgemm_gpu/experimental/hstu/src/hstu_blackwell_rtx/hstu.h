/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Blackwell RTX HSTU forward parameters.
// Extends Ampere params with FP8 quantization fields (matching Hopper layout
// so the same Python-level quantization code can be reused).

#pragma once

#include <cuda.h>
#include <cstdint>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Hstu_params {
  using index_t = int64_t;
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;

  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;
  int h, h_k, h_rab;
  int h_h_k_ratio;

  bool is_balance_fwd;
  bool is_balance_bwd;
  int arch;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Hstu_fwd_params : public Hstu_params {
  void* __restrict__ o_ptr;
  index_t o_row_stride;
  index_t o_head_stride;

  int* __restrict__ cu_seqlens_q;
  int* __restrict__ cu_seqlens_k;
  int* __restrict__ seqused_q;
  int* __restrict__ seqused_k;

  int* __restrict__ num_targets;
  int* __restrict__ num_contexts;

  void* __restrict__ func_ptr;
  index_t func_ids_stride;
  index_t func_head_stride;
  int func_batch;
  int n_func;

  void* __restrict__ rab_ptr;
  index_t rab_seqlen_qk_stride;
  index_t rab_seqlen_q_stride;
  index_t rab_seqlen_k_stride;

  // Paged KV cache for Blackwell RTX FP8 forward.
  void* __restrict__ kv_cache_ptr = nullptr;
  index_t kv_cache_row_stride = 0;
  index_t kv_cache_head_stride = 0;
  index_t kv_cache_page_stride = 0;
  index_t kv_cache_kvtensor_stride = 0;
  int page_size = 0;
  int total_pages = 0;
  int*  __restrict__ page_offsets = nullptr;
  int*  __restrict__ page_ids = nullptr;
  int*  __restrict__ last_page_lens = nullptr;

  int b, seqlen_q, seqlen_k, d, seqlen_q_rounded, seqlen_k_rounded;
  int scaling_seqlen;
  float alpha;
  int target_group_size;

  int window_size_left;
  int window_size_right;

  bool has_rab;
  bool is_bf16;
  bool is_causal;
  bool is_local;
  bool is_target;
  bool is_context;
  bool is_paged_kv = false;
  bool is_arbitrary_mask;

  // FP8 quantization fields (quant_mode >= 0 means FP8)
  // quant_mode: -1=BF16/FP16, 0=per-tensor FP8, 1-5=other FP8 modes
  int quant_mode = -1;
  int output_dtype = 0;  // 0=BF16, 1=FP16

  // Per-tensor descale scalars (float*)
  float* descale_q_ptr = nullptr;
  float* descale_k_ptr = nullptr;
  float* descale_v_ptr = nullptr;
  float* descale_vt_ptr = nullptr;

  index_t descale_q_head_stride = 0;
  index_t descale_k_head_stride = 0;
  index_t descale_v_head_stride = 0;
  index_t descale_vt_head_stride = 0;
  index_t descale_vt_row_stride = 0;

  // Block-wise descale (quant_mode 2+); not used for quant_mode=0
  int* cu_seqlens_vt_descale = nullptr;
  int* cu_seqlens_q_block_descale = nullptr;
  int* cu_seqlens_kv_block_descale = nullptr;
  int* cu_seqlens_v_block_descale = nullptr;  // separate from K (V has different block granularity)
  index_t q_block_descale_head_stride = 0;
  index_t kv_block_descale_head_stride = 0;
  index_t v_block_descale_head_stride = 0;   // stride for sf_v_packed (= descale_v.stride(0))
  // Optional pre-packed e8m0x4 scales for blockwise FP8 (quant_mode=2).
  // Layout is [head, total_blocks], each int32 packs 4 identical e8m0 bytes.
  int32_t* sf_q_packed_ptr = nullptr;
  int32_t* sf_k_packed_ptr = nullptr;
  int32_t* sf_v_packed_ptr = nullptr;

  // V-transposed pointer for quant_mode=1
  void* __restrict__ vt_ptr = nullptr;
  index_t vt_row_stride = 0;
  index_t vt_head_stride = 0;

  bool is_e4m3 = false;
  bool is_e5m2 = false;

  // Semaphore for dynamic tile scheduling (optional)
  int* __restrict__ tile_count_semaphore = nullptr;
  int num_sm = 0;
  int total_q = 0;
  int total_k = 0;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Forward declaration (defined in generated .cu instantiation files compiled
// in global namespace; matching the Hopper pattern in hstu_hopper/hstu.h).
template <int Arch, typename T, int Headdim, bool Has_rab, bool Is_local,
          bool Is_causal, bool Is_context, bool Is_target, bool Is_arbitrary, int kNFunc>
void run_hstu_fwd_(Hstu_fwd_params& params, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////////////////////////
