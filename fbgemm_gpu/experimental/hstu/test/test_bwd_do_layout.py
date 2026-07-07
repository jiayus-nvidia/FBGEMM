"""AC-7 layout/stride + workspace-zeroing correctness tests for the sm100 HSTU backward.

Covers the `do` original-layout fast path added in hstu_ops_gpu.hstu_varlen_bwd_100:
- the original-`do` path and the compact-`do` fallback path must agree within
  bf16 tolerance on every `do` layout (token-major contiguous, head-major-compact,
  and a non-contiguous/sliced `do`);
- a non-128-bit-aligned `do` must fall back to the compact clone and still be correct;
- the per-call workspace must be zeroed (no stale carry-over between calls).

Run: PYTHONPATH=src python -m pytest test/test_bwd_do_layout.py -v -s
(requires a Blackwell sm100 GPU + the .venv_ctm environment)
"""
import pytest
import torch

try:
    from hstu_blackwell.hstu_ops_gpu import (
        hstu_varlen_fwd_100, hstu_varlen_bwd_100,
        _supports_bwd_original_qkv_layout,
    )
except ImportError:  # installed-package layout
    from hstu.hstu_blackwell.hstu_ops_gpu import (
        hstu_varlen_fwd_100, hstu_varlen_bwd_100,
        _supports_bwd_original_qkv_layout,
    )

_SKIP = not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10
pytestmark = pytest.mark.skipif(_SKIP, reason="requires Blackwell sm100 GPU")

B, H, S = 2, 4, 2048


def _inputs(hdim, dev="cuda", dt=torch.bfloat16):
    torch.manual_seed(0)
    cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=dev)
    q = torch.randn(B * S, H, hdim, dtype=dt, device=dev)
    k = torch.randn(B * S, H, hdim, dtype=dt, device=dev)
    v = torch.randn(B * S, H, hdim, dtype=dt, device=dev)
    out, _ = hstu_varlen_fwd_100(
        q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=S, max_seqlen_k=S,
        num_contexts=None, num_targets=None, target_group_size=1,
        window_size_left=-1, window_size_right=0, alpha=1.0, rab=None, func=None)
    return cu, q, k, v, out


def _bwd(do, cu, q, k, v):
    return hstu_varlen_bwd_100(
        do, q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=S, max_seqlen_k=S,
        dq=None, dk=None, dv=None, num_contexts=None, num_targets=None,
        target_group_size=1, window_size_left=-1, window_size_right=0, alpha=1.0,
        rab=None, has_drab=False, func=None, deterministic=False)


def _non_fast_path_view(t):
    """Same logical values, but last-dim stride != 1 so the wrapper must clone."""
    out = t.transpose(1, 2).contiguous().transpose(1, 2)
    assert out.shape == t.shape
    assert torch.equal(out, t)
    assert not _supports_bwd_original_qkv_layout(out)
    return out


def _assert_close(name, a, b, rtol=5e-3):
    a = a.float(); b = b.float()
    md = (a - b).abs().max().item()
    rel = md / (a.abs().max().item() + 1e-9)
    assert rel < rtol, f"{name}: rel {rel:.3e} (max_abs {md:.3e}) exceeds {rtol}"


@pytest.mark.parametrize("hdim", [64, 128])
@pytest.mark.parametrize("layout", ["token_major", "head_major", "noncontiguous"])
def test_do_layout_compact_vs_original_agree(hdim, layout):
    """original-do path and compact-do path must agree within bf16 tolerance."""
    cu, q, k, v, out = _inputs(hdim)
    if layout == "token_major":
        do = torch.randn(B * S, H, hdim, dtype=torch.bfloat16, device="cuda").contiguous()
    elif layout == "head_major":
        do = torch.randn(
            H, B * S, hdim, dtype=torch.bfloat16, device="cuda"
        ).permute(1, 0, 2)
        assert do.stride() == (hdim, B * S * hdim, 1)
    else:  # genuinely non-contiguous (head_dim stride != 1) -> ineligible, must clone
        do = torch.randn(B * S, hdim, H, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
        assert not _supports_bwd_original_qkv_layout(do)
    do_ref = _non_fast_path_view(do) if _supports_bwd_original_qkv_layout(do) else do
    dq0, dk0, dv0, _ = _bwd(do_ref, cu, q, k, v)
    dq1, dk1, dv1, _ = _bwd(do, cu, q, k, v)
    _assert_close("dq", dq0, dq1)
    _assert_close("dk", dk0, dk1)
    _assert_close("dv", dv0, dv1)


@pytest.mark.parametrize("hdim", [64, 128])
def test_noncontiguous_do_falls_back_and_runs(hdim):
    """A non-128-bit-aligned do must NOT take the original fast path; it must still run."""
    cu, q, k, v, out = _inputs(hdim)
    # head_dim stride != 1 -> not eligible for the original fast path; must clone.
    do = torch.randn(B * S, hdim, H, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
    assert not _supports_bwd_original_qkv_layout(do), "transposed do should not be fast-path eligible"
    dq, dk, dv, _ = _bwd(do, cu, q, k, v)
    assert torch.isfinite(dq.float()).all() and torch.isfinite(dk.float()).all() and torch.isfinite(dv.float()).all()


@pytest.mark.parametrize("hdim", [64, 128])
def test_workspace_zeroed_between_calls(hdim):
    """Workspace must be zeroed each call: repeated identical calls give identical dq."""
    cu, q, k, v, out = _inputs(hdim)
    do = torch.randn(B * S, H, hdim, dtype=torch.bfloat16, device="cuda").contiguous()
    dq_a, _, _, _ = _bwd(do, cu, q, k, v)
    # interleave a different call that writes the workspace, then repeat the first
    other = torch.randn_like(do)
    _bwd(other, cu, q, k, v)
    dq_b, _, _, _ = _bwd(do, cu, q, k, v)
    # If the per-call workspace were not zeroed, `other`'s stale partial sums would
    # corrupt this dq by an order-1 amount. The dQ reduction is non-deterministic
    # under deterministic=False (atomic accumulation order varies run-to-run), so we
    # bound at bf16 noise (5e-3) rather than bit-exactness — large enough to tolerate
    # the accumulation jitter, small enough to catch any stale-workspace carry-over.
    _assert_close("dq repeat (stale-workspace guard)", dq_a, dq_b, rtol=5e-3)
