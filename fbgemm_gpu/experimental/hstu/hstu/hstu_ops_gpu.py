# Fake (abstract) implementations for fbgemm::hstu_varlen_fwd_* ops.
# These are required by torch.export / FakeTensor tracing.
# Registered via set_python_module("hstu.hstu_ops_gpu") in the C++ TORCH_LIBRARY_FRAGMENT.

import torch


@torch.library.register_fake("fbgemm::hstu_varlen_fwd_80")
def _hstu_varlen_fwd_80_fake(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    seqused_q, seqused_k,
    max_seqlen_q, max_seqlen_k,
    scaling_seqlen=-1,
    num_contexts=None, num_targets=None,
    target_group_size=1,
    window_size_left=-1, window_size_right=-1,
    alpha=1.0,
    rab=None, func=None,
    kv_cache=None, page_offsets=None, page_ids=None, last_page_lens=None,
):
    # Output is the attention-weighted sum of v: same shape and dtype as v.
    # Second return value (softmax_lse) is unused by callers; return empty tensor.
    return torch.empty_like(v), torch.empty(0, device=q.device, dtype=torch.float32)


@torch.library.register_fake("fbgemm::hstu_varlen_fwd_90")
def _hstu_varlen_fwd_90_fake(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    seqused_q, seqused_k,
    max_seqlen_q, max_seqlen_k,
    scaling_seqlen=-1,
    num_contexts=None, num_targets=None,
    target_group_size=1,
    window_size_left=-1, window_size_right=-1,
    alpha=1.0,
    rab=None, func=None,
    quant_mode=-1, output_dtype=-1,
    vt=None, cu_seqlens_vt_descale=None,
    q_descale=None, k_descale=None, v_descale=None, vt_descale=None,
    cu_seqlens_q_block_descale=None, cu_seqlens_kv_block_descale=None,
):
    # Output dtype may differ when output_dtype is specified (0=bfloat16, 1=float16).
    if output_dtype == 0:
        out_dtype = torch.bfloat16
    elif output_dtype == 1:
        out_dtype = torch.float16
    else:
        out_dtype = v.dtype
    return v.new_empty(v.shape, dtype=out_dtype), torch.empty(0, device=q.device, dtype=torch.float32)
