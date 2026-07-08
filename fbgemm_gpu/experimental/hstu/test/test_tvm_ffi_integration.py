"""TVM-FFI integration tests for the Blackwell HSTU Python wrappers."""

import pytest
import torch

try:
    from hstu_blackwell import hstu_ops_gpu as ops
except ImportError:
    from hstu.hstu_blackwell import hstu_ops_gpu as ops


_SKIP = not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10
pytestmark = pytest.mark.skipif(
    _SKIP,
    reason="requires a Blackwell sm100 GPU",
)

B, H, S, D = 1, 4, 128, 64


def _inputs(dtype=torch.bfloat16):
    torch.manual_seed(0)
    shape = (B * S, H, D)
    q = torch.randn(shape, dtype=dtype, device="cuda")
    k = torch.randn(shape, dtype=dtype, device="cuda")
    v = torch.randn(shape, dtype=dtype, device="cuda")
    do = torch.randn(shape, dtype=dtype, device="cuda")
    cu = torch.tensor([0, S], dtype=torch.int32, device="cuda")
    return q, k, v, do, cu


def _fwd(q, k, v, cu):
    return ops.hstu_varlen_fwd_100(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=S,
        max_seqlen_k=S,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size_left=-1,
        window_size_right=0,
        alpha=1.0,
        rab=None,
        func=None,
    )[0]


def _bwd(do, q, k, v, cu):
    return ops.hstu_varlen_bwd_100(
        do,
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=S,
        max_seqlen_k=S,
        dq=None,
        dk=None,
        dv=None,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size_left=-1,
        window_size_right=0,
        alpha=1.0,
        rab=None,
        has_drab=False,
        func=None,
        deterministic=False,
    )[:3]


def test_cached_tvm_ffi_call_does_not_rebuild_dlpack(monkeypatch):
    """DLPack objects are compile-time specs, not per-call runtime adapters."""
    ops.hstu_varlen_fwd_100.compile_cache.clear()
    ops.hstu_varlen_bwd_100.compile_cache.clear()
    q, k, v, do, cu = _inputs()
    # The real custom-autograd entry point forwards saved tensors that may
    # require gradients. TVM FFI must accept those tensors directly at runtime.
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    _fwd(q, k, v, cu)
    _bwd(do, q, k, v, cu)
    torch.cuda.synchronize()

    def fail_from_dlpack(*args, **kwargs):
        raise AssertionError("cache-hit TVM-FFI call rebuilt a DLPack adapter")

    monkeypatch.setattr(ops, "from_dlpack", fail_from_dlpack)
    out = _fwd(q, k, v, cu)
    grads = _bwd(do, q, k, v, cu)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert all(torch.isfinite(grad.float()).all() for grad in grads)


def test_tvm_ffi_uses_current_stream_and_is_graph_capturable():
    ops.hstu_varlen_fwd_100.compile_cache.clear()
    ops.hstu_varlen_bwd_100.compile_cache.clear()
    q, k, v, do, cu = _inputs()
    # Compile outside capture.
    _fwd(q, k, v, cu)
    _bwd(do, q, k, v, cu)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = _fwd(q, k, v, cu)
        captured_grads = _bwd(do, q, k, v, cu)
    # Capture records work but does not populate the captured outputs. Replay
    # once to produce the reference values.
    graph.replay()
    torch.cuda.synchronize()
    expected_out = captured_out.clone()
    expected_grads = tuple(grad.clone() for grad in captured_grads)

    captured_out.zero_()
    for grad in captured_grads:
        grad.zero_()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured_out, expected_out, rtol=0, atol=0)
    for actual, expected in zip(captured_grads, expected_grads):
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)


def test_compile_cache_separates_bf16_and_fp16():
    ops.hstu_varlen_fwd_100.compile_cache.clear()
    ops.hstu_varlen_bwd_100.compile_cache.clear()
    for dtype in (torch.bfloat16, torch.float16):
        q, k, v, do, cu = _inputs(dtype)
        out = _fwd(q, k, v, cu)
        grads = _bwd(do, q, k, v, cu)
        torch.cuda.synchronize()
        assert out.dtype == dtype
        assert all(grad.dtype == dtype for grad in grads)

    assert len(ops.hstu_varlen_fwd_100.compile_cache) == 2
    assert len(ops.hstu_varlen_bwd_100.compile_cache) == 2
