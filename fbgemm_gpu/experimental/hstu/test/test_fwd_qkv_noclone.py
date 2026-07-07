"""Forward q/k/v no-clone fast path: large-scale correctness on non-contiguous qkv.

The forward wrapper skips contiguous() when q/k/v have a unit-stride last dim and
128-bit aligned token/head strides (feeds original layout to the kernel via
mark_layout_dynamic). This validates, across dtype x head_dim x mask:
  - PACKED qkv (non-contiguous views, the case contiguous() would copy): the
    no-clone fast path is bit-identical to the layout fallback path;
  - a NON-aligned input correctly falls back to contiguous() and stays correct;
  - already-contiguous qkv is unchanged.

Run: PYTHONPATH=src python -m pytest test/test_fwd_qkv_noclone.py -v -s
"""
import pytest
import torch

try:
    from hstu_blackwell.hstu_ops_gpu import hstu_varlen_fwd_100, _supports_bwd_original_qkv_layout
except ImportError:
    from hstu.hstu_blackwell.hstu_ops_gpu import hstu_varlen_fwd_100, _supports_bwd_original_qkv_layout

B, H, S = 2, 4, 2048
_MASK = {"causal": (-1, 0), "local": (256, 0), "target": (-1, 0)}


def _fwd(q, k, v, mask, num_targets=None):
    cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device="cuda")
    wl, wr = _MASK[mask]
    out = hstu_varlen_fwd_100(
        q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=S, max_seqlen_k=S, num_contexts=None, num_targets=num_targets,
        target_group_size=(4 if mask == "target" else 1),
        window_size_left=wl, window_size_right=wr,
        alpha=1.0, rab=None, func=None,
    )
    return (out[0] if isinstance(out, tuple) else out).clone()


def _non_fast_path_view(t):
    """Same logical values, but last-dim stride != 1 so the wrapper must clone."""
    out = t.transpose(1, 2).contiguous().transpose(1, 2)
    assert out.shape == t.shape
    assert torch.equal(out, t)
    assert not _supports_bwd_original_qkv_layout(out)
    return out


def _targets(mask):
    if mask != "target":
        return None
    return torch.full((B,), 128, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hdim", [64, 128, 256])
@pytest.mark.parametrize("mask", ["causal", "local", "target"])
def test_packed_noncontiguous_qkv_agrees(dtype, hdim, mask):
    torch.manual_seed(0)
    packed = torch.randn(B * S, H, 3, hdim, dtype=dtype, device="cuda")
    q, k, v = packed[:, :, 0, :], packed[:, :, 1, :], packed[:, :, 2, :]
    assert not q.is_contiguous() and q.stride(2) == 1
    assert _supports_bwd_original_qkv_layout(q), "fast path should fire on packed qkv"
    nt = _targets(mask)
    out_fast = _fwd(q, k, v, mask, num_targets=nt)
    out_ref = _fwd(
        _non_fast_path_view(q),
        _non_fast_path_view(k),
        _non_fast_path_view(v),
        mask,
        num_targets=nt,
    )
    torch.testing.assert_close(out_fast, out_ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hdim", [64, 128])
def test_nonaligned_falls_back_and_matches_clean(dtype, hdim):
    """Last dim contiguous but head stride not 8-aligned -> must fall back to
    contiguous() and match a cleanly-allocated contiguous reference."""
    torch.manual_seed(0)
    padded = torch.randn(B * S, H, hdim + 1, dtype=dtype, device="cuda")
    q = padded[:, :, :hdim]  # stride(1) = hdim+1 -> not 8-aligned for these hdims
    k = padded.clone()[:, :, :hdim]
    v = padded.clone()[:, :, :hdim]
    assert q.stride(2) == 1 and not _supports_bwd_original_qkv_layout(q), "should NOT take fast path"
    out_fb = _fwd(q, k, v, "causal")
    # clean contiguous reference computed independently
    out_ref = _fwd(q.contiguous(), k.contiguous(), v.contiguous(), "causal")
    torch.testing.assert_close(out_fb, out_ref, rtol=0, atol=0)


@pytest.mark.parametrize("hdim", [64, 128])
def test_contiguous_qkv_unchanged(hdim):
    torch.manual_seed(0)
    q = torch.randn(B * S, H, hdim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B * S, H, hdim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B * S, H, hdim, dtype=torch.bfloat16, device="cuda")
    out_fast = _fwd(q, k, v, "causal")
    out_ref = _fwd(
        _non_fast_path_view(q),
        _non_fast_path_view(k),
        _non_fast_path_view(v),
        "causal",
    )
    torch.testing.assert_close(out_fast, out_ref, rtol=0, atol=0)


def test_output_is_contiguous_and_viewable():
    """The public output must support the common (T, H * D) view directly."""
    hdim = 64
    q = torch.randn(B * S, H, hdim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = _fwd(q, k, v, "causal")

    assert out.is_contiguous()
    assert out.view(B * S, H * hdim).shape == (B * S, H * hdim)
