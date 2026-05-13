/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Blackwell RTX HSTU forward attention ops.
// Registered as the hstu_varlen_fwd_120 operator.
// Supports Blackwell RTX FP8 (quant_mode >= 0) forward pass.

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/nn/functional.h>

#include <optional>

#include "hstu.h"
#include "static_switch.h"

namespace fbgemm_gpu::hstu {

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                           \
  TORCH_CHECK(                                        \
      x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
      #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

////////////////////////////////////////////////////////////////////////////////////////////////////

void set_params_fprop_blackwell_rtx(
    Hstu_fwd_params* params,
    const size_t b,
    const size_t seqlen_q,
    const size_t seqlen_k,
    const size_t scaling_seqlen,
    const size_t target_group_size,
    const size_t seqlen_q_rounded,
    const size_t seqlen_k_rounded,
    const size_t h,
    const size_t h_k,
    const size_t h_rab,
    const size_t d,
    const float alpha,
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    const at::Tensor rab,
    const std::optional<at::Tensor>& kv_cache,
    at::Tensor out,
    void* num_contexts_d,
    void* cu_seqlens_q_d,
    void* cu_seqlens_k_d,
    void* seqused_q_d,
    void* seqused_k_d,
    void* page_offsets,
    void* page_ids,
    void* last_page_lens,
    void* num_targets_d,
    bool has_rab,
    bool is_paged_kv,
    int quant_mode,
    const std::optional<at::Tensor>& func,
    int window_size_left,
    int window_size_right,
    // FP8 descale factors (optional, required when quant_mode >= 0)
    const std::optional<at::Tensor>& descale_q,
    const std::optional<at::Tensor>& descale_k,
    const std::optional<at::Tensor>& descale_v) {
  *params = {};

  params->arch = at::cuda::getCurrentDeviceProperties()->major * 10 +
      at::cuda::getCurrentDeviceProperties()->minor;

  params->q_ptr = q.data_ptr();
  params->k_ptr = k.data_ptr();
  params->v_ptr = v.data_ptr();
  params->q_row_stride = q.stride(-3);
  params->k_row_stride = k.stride(-3);
  params->v_row_stride = v.stride(-3);
  params->q_head_stride = q.stride(-2);
  params->k_head_stride = k.stride(-2);
  params->v_head_stride = v.stride(-2);

  if (out.numel() > 0) {
    params->o_ptr = out.data_ptr();
    params->o_row_stride = out.stride(-3);
    params->o_head_stride = out.stride(-2);
  }

  params->has_rab = has_rab;
  if (has_rab) {
    params->rab_ptr = rab.data_ptr();
    params->rab_seqlen_qk_stride = rab.stride(-4);
    params->rab_seqlen_q_stride = rab.stride(-3);
    params->rab_seqlen_k_stride = rab.stride(-2);
    params->h_rab = h_rab;
  } else {
    params->rab_ptr = nullptr;
    params->rab_seqlen_qk_stride = 0;
    params->rab_seqlen_q_stride = 0;
    params->rab_seqlen_k_stride = 0;
    params->h_rab = 0;
  }

  params->num_contexts = static_cast<int*>(num_contexts_d);
  params->cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params->cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);
  params->num_targets = static_cast<int*>(num_targets_d);
  params->seqused_q = static_cast<int*>(seqused_q_d);
  params->seqused_k = static_cast<int*>(seqused_k_d);

  params->quant_mode = quant_mode;
  params->is_bf16 = true;  // FP8 forward uses the existing half-output metadata path.
  params->is_e4m3 = true;
  params->is_e5m2 = false;
  params->output_dtype = 0;  // 0 = BF16

  // FP8 descale factors (per-tensor, quant_mode=0).
  if (descale_q.has_value()) {
    params->descale_q_ptr = static_cast<float*>(descale_q.value().data_ptr());
    params->descale_q_head_stride = descale_q.value().numel() > 1 ? 1 : 0;
  }
  if (descale_k.has_value()) {
    params->descale_k_ptr = static_cast<float*>(descale_k.value().data_ptr());
    params->descale_k_head_stride = descale_k.value().numel() > 1 ? 1 : 0;
  }
  if (descale_v.has_value()) {
    params->descale_v_ptr = static_cast<float*>(descale_v.value().data_ptr());
    params->descale_v_head_stride = descale_v.value().numel() > 1 ? 1 : 0;
  }

  params->b = b;
  params->h = h;
  params->h_k = h_k;
  params->h_h_k_ratio = h / h_k;
  params->seqlen_q = seqlen_q;
  params->seqlen_k = seqlen_k;
  params->seqlen_q_rounded = seqlen_q_rounded;
  params->seqlen_k_rounded = seqlen_k_rounded;
  params->scaling_seqlen = (scaling_seqlen == 0 || scaling_seqlen == (size_t)-1) ? seqlen_q : scaling_seqlen;
  params->d = d;
  params->alpha = alpha;
  params->is_target = num_targets_d != nullptr;
  params->target_group_size = target_group_size;
  if (params->is_target) {
    TORCH_CHECK(target_group_size > 0, "target_group_size must be positive");
  }
  params->is_context = num_contexts_d != nullptr;

  if (window_size_left < 0 || window_size_left > (int)seqlen_k) window_size_left = seqlen_k;
  if (window_size_right < 0 || window_size_right > (int)seqlen_k) window_size_right = seqlen_k;
  params->window_size_left = window_size_left;
  params->window_size_right = window_size_right;
  params->is_causal = params->window_size_left == (int)seqlen_k && params->window_size_right == 0;
  params->is_local = (window_size_left < (int)seqlen_k || window_size_right < (int)seqlen_k) &&
      !params->is_causal;

  params->is_arbitrary_mask = func.has_value() && func.value().defined();
  if (params->is_arbitrary_mask) {
    TORCH_CHECK(func.value().dtype() == torch::kInt32, "func must have dtype int32");
    CHECK_DEVICE(func.value());
    params->func_ptr = func.value().data_ptr();
    params->func_ids_stride = func.value().stride(-2);
    params->func_head_stride = func.value().stride(-3);
    params->n_func = func.value().size(-2);
    TORCH_CHECK(params->n_func == HSTU_ARBITRARY_NFUNC, "n_func mismatch");
  }

  params->is_paged_kv = is_paged_kv;
  if (is_paged_kv) {
    const at::Tensor& kv = kv_cache.value();
    params->kv_cache_ptr = kv.data_ptr();
    params->kv_cache_row_stride = kv.stride(-2);
    params->kv_cache_head_stride = kv.stride(-3);
    params->kv_cache_page_stride = kv.stride(-4);
    params->kv_cache_kvtensor_stride = kv.stride(-5);
    params->page_size = kv.size(-3);
    params->total_pages = kv.size(-5);
  } else {
    params->kv_cache_ptr = nullptr;
    params->page_size = 0;
    params->total_pages = 0;
  }
  params->page_offsets = static_cast<int*>(page_offsets);
  params->page_ids = static_cast<int*>(page_ids);
  params->last_page_lens = static_cast<int*>(last_page_lens);

}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype, bool Has_rab, bool Is_local, bool Is_causal,
          bool Is_context, bool Is_target, bool Is_arbitrary, int kNFunc>
void run_hstu_fwd_headdim_blackwell_rtx(Hstu_fwd_params& params, cudaStream_t stream) {
  constexpr int Arch = 120;
#ifndef HSTU_DISABLE_HDIM64
  if (params.d == 64) {
    run_hstu_fwd_<Arch, Dtype, 64, Has_rab, Is_local, Is_causal,
        Is_context, Is_target, Is_arbitrary, kNFunc>(params, stream);
    return;
  }
#endif
#ifndef HSTU_DISABLE_HDIM32
  if (params.d == 32) {
    run_hstu_fwd_<Arch, Dtype, 32, Has_rab, Is_local, Is_causal,
        Is_context, Is_target, Is_arbitrary, kNFunc>(params, stream);
    return;
  }
#endif
#ifndef HSTU_DISABLE_HDIM128
  if (params.d == 128) {
    run_hstu_fwd_<Arch, Dtype, 128, Has_rab, Is_local, Is_causal,
        Is_context, Is_target, Is_arbitrary, kNFunc>(params, stream);
    return;
  }
#endif
#ifndef HSTU_DISABLE_HDIM256
  if (params.d == 256) {
    run_hstu_fwd_<Arch, Dtype, 256, Has_rab, Is_local, Is_causal,
        Is_context, Is_target, Is_arbitrary, kNFunc>(params, stream);
    return;
  }
#endif
  TORCH_CHECK(
      false,
      "Unsupported head dim: ",
      params.d,
      " (Blackwell RTX FP8 supports 32/64/128/256)");
}

void run_hstu_fwd_blackwell_rtx(Hstu_fwd_params& params, cudaStream_t stream) {
  TORCH_CHECK(params.quant_mode >= 0, "Blackwell RTX build only supports FP8 forward");
  RAB_SWITCH(params.has_rab, Has_rab, [&] {
    TORCH_CHECK(
        params.d == 32 || params.d == 64 || params.d == 128 || params.d == 256,
        "Blackwell RTX FP8 WS-only path supports headDim32/64/128/256, got headDim=",
        params.d);
    using FP8Type = cutlass::float_e4m3_t;
#ifndef HSTU_DISABLE_ARBITRARY
    if (params.is_arbitrary_mask) {
      run_hstu_fwd_headdim_blackwell_rtx<FP8Type, Has_rab, false, false, false, false, true, HSTU_ARBITRARY_NFUNC>(params, stream);
      return;
    }
#endif
#ifndef HSTU_DISABLE_LOCAL
    if (params.is_local) {
      run_hstu_fwd_headdim_blackwell_rtx<FP8Type, Has_rab, true, false, false, false, false, 0>(params, stream);
      return;
    }
#endif
    if (!params.is_causal) {
      run_hstu_fwd_headdim_blackwell_rtx<FP8Type, Has_rab, false, false, false, false, false, 0>(params, stream);
      return;
    }
#ifndef HSTU_DISABLE_CAUSAL
    CONTEXT_SWITCH(params.is_context, Is_context, [&] {
      TARGET_SWITCH(params.is_target, Is_target, [&] {
        run_hstu_fwd_headdim_blackwell_rtx<FP8Type, Has_rab, false, true, Is_context, Is_target, false, 0>(params, stream);
      });
    });
#endif
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Main entry point: varlen forward for Blackwell RTX FP8.
std::tuple<at::Tensor, at::Tensor> hstu_varlen_fwd_120(
    const at::Tensor& q,   // total_q x num_heads x head_size
    const at::Tensor& k,   // total_k x num_heads_k x head_size
    const at::Tensor& v,   // total_k x num_heads_k x head_size
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const std::optional<at::Tensor>& seqused_q,
    const std::optional<at::Tensor>& seqused_k,
    const int64_t max_seqlen_q,
    const int64_t max_seqlen_k,
    const int64_t scaling_seqlen,
    const std::optional<at::Tensor>& num_contexts,
    const std::optional<at::Tensor>& num_targets,
    const int64_t target_group_size,
    int64_t window_size_left,
    int64_t window_size_right,
    const double alpha,
    std::optional<at::Tensor> rab,
    const std::optional<at::Tensor>& func,
    // FP8 arguments.
    int64_t quant_mode,  // 0=per-tensor FP8, 2=blockwise FP8
    const std::optional<at::Tensor>& descale_q,
    const std::optional<at::Tensor>& descale_k,
    const std::optional<at::Tensor>& descale_v,
    const std::optional<at::Tensor>& sf_q_packed,
    const std::optional<at::Tensor>& sf_k_packed,
    const std::optional<at::Tensor>& sf_v_packed,
    // Block-wise descale cumulative seqlens (quant_mode=2 only)
    const std::optional<at::Tensor>& cu_seqlens_q_block_descale,
    const std::optional<at::Tensor>& cu_seqlens_kv_block_descale,
    const std::optional<at::Tensor>& cu_seqlens_v_block_descale,
    const std::optional<at::Tensor>& kv_cache,
    const std::optional<at::Tensor>& page_offsets,
    const std::optional<at::Tensor>& page_ids,
    const std::optional<at::Tensor>& last_page_lens) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  const int arch = dprops->major * 10 + dprops->minor;
  TORCH_CHECK(arch >= 120, "hstu_varlen_fwd_120 requires Blackwell RTX GPU, got SM", arch);

  auto q_dtype = q.dtype();
  TORCH_CHECK(quant_mode >= 0, "Blackwell RTX build only supports FP8 forward");
  TORCH_CHECK(q_dtype == at::kFloat8_e4m3fn,
      "FP8 mode requires float8_e4m3fn input, got ", q_dtype);
  TORCH_CHECK(k.dtype() == at::kFloat8_e4m3fn, "k must be float8_e4m3fn in FP8 mode");
  TORCH_CHECK(v.dtype() == at::kFloat8_e4m3fn, "v must be float8_e4m3fn in FP8 mode");
  TORCH_CHECK(quant_mode == 0 || quant_mode == 2,
      "Blackwell RTX supports quant_mode=0 (per-tensor FP8) or quant_mode=2 (blockwise FP8), got ", quant_mode);
  TORCH_CHECK(descale_q.has_value(), "descale_q required for FP8 mode");
  TORCH_CHECK(descale_k.has_value(), "descale_k required for FP8 mode");
  TORCH_CHECK(descale_v.has_value(), "descale_v required for FP8 mode");
  if (quant_mode == 2) {
    if (sf_q_packed.has_value()) {
      TORCH_CHECK(sf_q_packed.value().dtype() == at::kInt, "sf_q_packed must be int32");
    }
    if (sf_k_packed.has_value()) {
      TORCH_CHECK(sf_k_packed.value().dtype() == at::kInt, "sf_k_packed must be int32");
    }
    if (sf_v_packed.has_value()) {
      TORCH_CHECK(sf_v_packed.value().dtype() == at::kInt, "sf_v_packed must be int32");
    }
  }

  TORCH_CHECK(cu_seqlens_q.dtype() == at::kInt, "cu_seqlens_q must be int32");
  TORCH_CHECK(cu_seqlens_k.dtype() == at::kInt, "cu_seqlens_k must be int32");

  CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
  CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);
  CHECK_CONTIGUOUS(cu_seqlens_q); CHECK_CONTIGUOUS(cu_seqlens_k);

  const int batch_size = cu_seqlens_q.numel() - 1;
  const int num_heads = q.size(1);
  const int head_size = q.size(2);
  const int total_k = k.size(0);
  const int num_heads_k = k.size(1);
  const bool is_paged_kv = kv_cache.has_value() && page_offsets.has_value() &&
      page_ids.has_value() && last_page_lens.has_value();
  bool has_rab = rab.has_value();

  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
  CHECK_SHAPE(k, total_k, num_heads_k, head_size);
  CHECK_SHAPE(v, total_k, num_heads_k, head_size);

  TORCH_CHECK(batch_size > 0, "batch_size must be positive");
  TORCH_CHECK(num_heads == num_heads_k, "num_heads_k must equal num_heads");
  TORCH_CHECK(
      head_size == 32 || head_size == 64 || head_size == 128 || head_size == 256,
      "Blackwell RTX FP8 supports head_size 32, 64, 128, or 256, got ",
      head_size);
  TORCH_CHECK(q.stride(-1) == 1, "q must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "k must have contiguous last dimension");
  // V must be row-major (d-stride=1). WS TMA path uses row-major V with LDSM_T transpose.
  TORCH_CHECK(v.stride(-1) == 1,
      "v must have contiguous last dimension (row-major)");

  if (is_paged_kv) {
    TORCH_CHECK(quant_mode == 2, "Blackwell RTX paged KV initial path requires quant_mode=2");
    TORCH_CHECK(
        head_size == 32 || head_size == 64 || head_size == 128 || head_size == 256,
        "Blackwell RTX FP8 paged KV path requires head_size=32, 64, 128, or 256");
    const bool is_no_target_paged_kv = !num_targets.has_value();
    const bool is_causal_or_target_paged_kv =
        num_targets.has_value() && window_size_left < 0 && window_size_right == 0;
    TORCH_CHECK(
        is_no_target_paged_kv || is_causal_or_target_paged_kv,
        "Blackwell RTX FP8 paged KV supports no-target full/local/arbitrary windows, "
        "or causal/target mode with num_targets and window_size_right=0");
    const at::Tensor& kv = kv_cache.value();
    CHECK_DEVICE(kv);
    CHECK_CONTIGUOUS(kv);
    TORCH_CHECK(kv.dtype() == at::kFloat8_e4m3fn,
        "Blackwell RTX FP8 paged KV requires kv_cache dtype float8_e4m3fn");
    TORCH_CHECK(kv.dim() == 5,
        "kv_cache must have shape [total_pages, 2, page_size, num_heads_k, head_size]");
    TORCH_CHECK(kv.size(1) == 2, "kv_cache second dimension must be 2 (K,V)");
    TORCH_CHECK(kv.size(2) == 64,
        "Blackwell RTX FP8 paged KV initial path requires page_size=64");
    TORCH_CHECK(kv.size(3) == num_heads_k, "kv_cache num_heads_k mismatch");
    TORCH_CHECK(kv.size(4) == head_size, "kv_cache head_size mismatch");
    TORCH_CHECK(kv.stride(-1) == 1, "kv_cache must have contiguous last dimension");
    CHECK_DEVICE(page_offsets.value());
    CHECK_DEVICE(page_ids.value());
    CHECK_DEVICE(last_page_lens.value());
    CHECK_CONTIGUOUS(page_offsets.value());
    CHECK_CONTIGUOUS(page_ids.value());
    CHECK_CONTIGUOUS(last_page_lens.value());
    TORCH_CHECK(page_offsets.value().dtype() == at::kInt, "page_offsets must be int32");
    TORCH_CHECK(page_ids.value().dtype() == at::kInt, "page_ids must be int32");
    TORCH_CHECK(last_page_lens.value().dtype() == at::kInt, "last_page_lens must be int32");
    CHECK_SHAPE(page_offsets.value(), batch_size + 1);
    CHECK_SHAPE(last_page_lens.value(), batch_size);
    TORCH_CHECK(sf_k_packed.has_value() && sf_v_packed.has_value(),
        "Blackwell RTX FP8 paged KV requires combined sf_k_packed and sf_v_packed");
  } else {
    TORCH_CHECK(!kv_cache.has_value() && !page_offsets.has_value() &&
            !page_ids.has_value() && !last_page_lens.has_value(),
        "kv_cache/page_offsets/page_ids/last_page_lens must either all be provided or all be None");
  }

  at::Tensor out = torch::empty({q.size(0), num_heads, head_size},
      q.options().dtype(at::kHalf));

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  // RAB is read by kBlockM x kBlockN TMA tiles in the RTX FP8 WS path, so pad
  // both RAB dimensions to 128. Non-RAB only needs the legacy 16B alignment.
  const int seqlen_q_rounded = round_multiple(max_seqlen_q, 16);
  int rab_extent = max_seqlen_k;
  if (has_rab) {
    rab_extent = std::max(rab_extent, (int)rab.value().size(-2));
    rab_extent = std::max(rab_extent, (int)rab.value().size(-1));
  }
  const int seqlen_k_rounded = round_multiple(rab_extent, has_rab ? 128 : 16);

  int num_heads_rab = num_heads;
  if (has_rab) {
    num_heads_rab = rab.value().size(1);
    CHECK_DEVICE(rab.value());
    TORCH_CHECK(rab.value().stride(-1) == 1, "rab must have contiguous last dimension");
    TORCH_CHECK(num_heads == num_heads_rab || num_heads_rab == 1, "rab num_heads mismatch");
    TORCH_CHECK(rab.value().size(0) == batch_size, "rab batch size mismatch");
    TORCH_CHECK(rab.value().size(2) >= max_seqlen_k,
        "rab seqlen_q dimension must cover max_seqlen_k");
    TORCH_CHECK(rab.value().size(3) >= max_seqlen_k,
        "rab seqlen_k dimension must cover max_seqlen_k");
    const int rab_q_pad = seqlen_k_rounded - rab.value().size(2);
    const int rab_k_pad = seqlen_k_rounded - rab.value().size(3);
    if (rab_q_pad != 0 || rab_k_pad != 0) {
      rab = torch::nn::functional::pad(
          rab.value(),
          torch::nn::functional::PadFuncOptions({
              0, rab_k_pad,
              0, rab_q_pad}));
    }
  }

  if (seqused_q.has_value()) {
    CHECK_DEVICE(seqused_q.value());
    CHECK_CONTIGUOUS(seqused_q.value());
    CHECK_SHAPE(seqused_q.value(), batch_size);
  }
  if (seqused_k.has_value()) {
    CHECK_DEVICE(seqused_k.value());
    CHECK_CONTIGUOUS(seqused_k.value());
    CHECK_SHAPE(seqused_k.value(), batch_size);
  }

  Hstu_fwd_params params;
  set_params_fprop_blackwell_rtx(
      &params,
      batch_size, max_seqlen_q, max_seqlen_k,
      scaling_seqlen, target_group_size,
      seqlen_q_rounded, seqlen_k_rounded,
      num_heads, num_heads_k, num_heads_rab,
      head_size, static_cast<float>(alpha),
      q, k, v,
      has_rab ? rab.value() : at::Tensor(),
      kv_cache,
      out,
      num_contexts.has_value() ? num_contexts.value().data_ptr() : nullptr,
      cu_seqlens_q.data_ptr(),
      cu_seqlens_k.data_ptr(),
      seqused_q.has_value() ? seqused_q.value().data_ptr() : nullptr,
      seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
      page_offsets.has_value() ? page_offsets.value().data_ptr() : nullptr,
      page_ids.has_value() ? page_ids.value().data_ptr() : nullptr,
      last_page_lens.has_value() ? last_page_lens.value().data_ptr() : nullptr,
      num_targets.has_value() ? num_targets.value().data_ptr() : nullptr,
      has_rab,
      is_paged_kv,
      static_cast<int>(quant_mode),
      func,
      static_cast<int>(window_size_left),
      static_cast<int>(window_size_right),
      descale_q, descale_k, descale_v);

  // Block-wise descale parameters (quant_mode=2 only)
  if (quant_mode == 2) {
    if (cu_seqlens_q_block_descale.has_value()) {
      params.cu_seqlens_q_block_descale =
          static_cast<int*>(cu_seqlens_q_block_descale.value().data_ptr());
    }
    if (cu_seqlens_kv_block_descale.has_value()) {
      params.cu_seqlens_kv_block_descale =
          static_cast<int*>(cu_seqlens_kv_block_descale.value().data_ptr());
    }
    // Head strides: descale_q/k/v are [head, total_blocks], stride(0) = total_blocks
    if (descale_q.has_value() && descale_q.value().dim() >= 2) {
      params.q_block_descale_head_stride = descale_q.value().stride(0);
    }
    if (descale_k.has_value() && descale_k.value().dim() >= 2) {
      params.kv_block_descale_head_stride = descale_k.value().stride(0);
    }
    if (descale_v.has_value() && descale_v.value().dim() >= 2) {
      params.v_block_descale_head_stride = descale_v.value().stride(0);
    }
    if (cu_seqlens_v_block_descale.has_value()) {
      params.cu_seqlens_v_block_descale =
          static_cast<int*>(cu_seqlens_v_block_descale.value().data_ptr());
    }
    if (sf_q_packed.has_value()) {
      params.sf_q_packed_ptr = static_cast<int32_t*>(sf_q_packed.value().data_ptr());
      // For TMA: use sf_q_packed.size(1) as head stride (PyTorch sets stride(0)=1 for H=1,
      // which would make TMA globalDim[0]=1 < boxDim=kBlockM → fail).
      if (sf_q_packed.value().dim() >= 2) {
        params.q_block_descale_head_stride = sf_q_packed.value().size(1);
      }
    }
    if (sf_k_packed.has_value()) {
      params.sf_k_packed_ptr = static_cast<int32_t*>(sf_k_packed.value().data_ptr());
      if (sf_k_packed.value().dim() >= 2) {
        params.kv_block_descale_head_stride = sf_k_packed.value().size(1);
      }
    }
    if (sf_v_packed.has_value()) {
      params.sf_v_packed_ptr = static_cast<int32_t*>(sf_v_packed.value().data_ptr());
      if (sf_v_packed.value().dim() >= 2) {
        params.v_block_descale_head_stride = sf_v_packed.value().size(1);
      }
    }
  }

  params.total_k = total_k;  // Phase 5: needed for TMA descriptor covering full concat K/V
  params.total_q = q.size(0);  // Phase 6 WS TMA: needed for TMA Q descriptor

  if (total_k > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_hstu_fwd_blackwell_rtx(params, stream);
  } else {
    out.zero_();
  }

  return {out, has_rab ? rab.value() : at::Tensor()};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "hstu_varlen_fwd_120("
      "Tensor q, Tensor k, Tensor v, "
      "Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
      "Tensor? seqused_q, Tensor? seqused_k, "
      "int max_seqlen_q, int max_seqlen_k, int scaling_seqlen, "
      "Tensor? num_contexts, Tensor? num_targets, int target_group_size, "
      "int window_size_left, int window_size_right, float alpha, "
      "Tensor? rab, Tensor? func, "
      "int quant_mode, "
      "Tensor? descale_q, Tensor? descale_k, Tensor? descale_v, "
      "Tensor? sf_q_packed, Tensor? sf_k_packed, Tensor? sf_v_packed, "
      "Tensor? cu_seqlens_q_block_descale, Tensor? cu_seqlens_kv_block_descale, "
      "Tensor? cu_seqlens_v_block_descale, "
      "Tensor? kv_cache=None, Tensor? page_offsets=None, Tensor? page_ids=None, "
      "Tensor? last_page_lens=None"
      ") -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("hstu_varlen_fwd_120", hstu_varlen_fwd_120);
}

} // namespace fbgemm_gpu::hstu
