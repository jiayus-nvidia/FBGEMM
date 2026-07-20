"""Correctness coverage for the Blackwell HSTU D=256 backward path.

Run with:
    PYTHONPATH=src python -m pytest test/test_bwd_dim256.py -v -s
"""

from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

try:
    from hstu_blackwell.hstu_ops_gpu import hstu_varlen_bwd_100
    from hstu_blackwell.hstu_bwd_256 import hstu_varlen_bwd_256
    from hstu_blackwell.hstu_bwd_256_cute import hstu_varlen_bwd_256_cute
except ImportError:  # installed-package layout
    from hstu.hstu_blackwell.hstu_ops_gpu import hstu_varlen_bwd_100
    from hstu.hstu_blackwell.hstu_bwd_256 import hstu_varlen_bwd_256
    from hstu.hstu_blackwell.hstu_bwd_256_cute import (
        hstu_varlen_bwd_256_cute,
    )


_SKIP = not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10
pytestmark = pytest.mark.skipif(_SKIP, reason="requires an SM100-family GPU")


@dataclass
class MaskInputs:
    mask: torch.Tensor
    window_left: int
    window_right: int
    num_targets: Optional[torch.Tensor] = None
    target_group_size: int = 1
    func: Optional[torch.Tensor] = None


def _make_mask(case: str, seqlen: int, device: torch.device) -> MaskInputs:
    rows = torch.arange(seqlen, device=device)[:, None]
    cols = torch.arange(seqlen, device=device)[None, :]
    if case == "none":
        return MaskInputs(
            mask=torch.ones((seqlen, seqlen), dtype=torch.bool, device=device),
            window_left=-1,
            window_right=-1,
        )
    if case == "causal":
        return MaskInputs(
            mask=cols <= rows,
            window_left=-1,
            window_right=0,
        )
    if case == "local":
        left, right = 17, 3
        return MaskInputs(
            mask=(cols >= torch.clamp(rows - left, min=0))
            & (cols < torch.clamp(rows + 1 + right, max=seqlen)),
            window_left=left,
            window_right=right,
        )
    if case == "target":
        num_targets = 16
        group_size = 4
        history = seqlen - num_targets
        target_index = torch.div(rows - history, group_size, rounding_mode="floor")
        target_left = history + target_index * group_size
        target_hole = (rows >= history) & (cols >= history) & (cols < target_left)
        return MaskInputs(
            mask=(cols <= rows) & ~target_hole,
            window_left=-1,
            window_right=0,
            num_targets=torch.tensor([num_targets], dtype=torch.int32, device=device),
            target_group_size=group_size,
        )
    if case == "arbitrary":
        func = torch.empty((1, 3, seqlen + 256), dtype=torch.int32, device=device)
        func0 = torch.clamp(torch.arange(seqlen, device=device) - 7, min=0)
        func1 = torch.clamp(torch.arange(seqlen, device=device) - 2, min=0)
        func2 = torch.clamp(torch.arange(seqlen, device=device) + 5, max=seqlen)
        func[0, 0, :seqlen] = func0
        func[0, 1, :seqlen] = func1
        func[0, 2, :seqlen] = func2
        func[:, :, seqlen:] = seqlen
        return MaskInputs(
            mask=(cols < func2[:, None])
            & ~((cols >= func0[:, None]) & (cols < func1[:, None])),
            window_left=-1,
            window_right=-1,
            func=func,
        )
    raise AssertionError(f"unknown mask case: {case}")


def _torch_grads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
    scaling_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q.detach().clone().requires_grad_()
    k = k.detach().clone().requires_grad_()
    v = v.detach().clone().requires_grad_()
    scores = torch.einsum("ihd,jhd->hij", q, k)
    p = torch.where(mask[None, :, :], F.silu(alpha * scores), 0.0)
    out = torch.einsum("hij,jhd->ihd", p, v) / scaling_seqlen
    return torch.autograd.grad(out, (q, k, v), do)


def _assert_error_tracks_low_precision(
    name: str,
    got: torch.Tensor,
    low_precision: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    got_error = (got.float() - reference).abs().max().item()
    low_error = (low_precision.float() - reference).abs().max().item()
    limit = max(5.0 * low_error, 2.0e-3)
    assert got_error <= limit, (
        f"{name}: candidate max error {got_error:.4e} exceeds {limit:.4e}; "
        f"low-precision PyTorch error is {low_error:.4e}"
    )
    assert torch.isfinite(got.float()).all(), f"{name} contains NaN/Inf"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("window_left", "window_right"),
    ((65, 65), (65, 0), (17, 3)),
    ids=("none", "causal", "local"),
)
def test_dim256_native_cute_matches_triton(
    dtype,
    window_left,
    window_right,
):
    """Keep the native kernels covered even below the auto-dispatch crossover."""
    torch.manual_seed(1122)
    seqlen, heads, head_dim = 65, 2, 256
    device = torch.device("cuda")
    cu = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    q, k, v, do = [
        torch.empty((seqlen, heads, head_dim), device=device).uniform_(-1, 1).to(dtype)
        for _ in range(4)
    ]
    native = hstu_varlen_bwd_256_cute(
        do,
        q,
        k,
        v,
        cu,
        cu,
        seqlen,
        seqlen,
        None,
        None,
        None,
        window_left,
        window_right,
        0.7,
    )[:3]
    triton_ref = hstu_varlen_bwd_256(
        do,
        q,
        k,
        v,
        cu,
        cu,
        seqlen,
        seqlen,
        None,
        None,
        None,
        None,
        1,
        window_left,
        window_right,
        0.7,
        None,
    )[:3]
    eps = torch.finfo(dtype).eps
    for name, actual, expected in zip(("dq", "dk", "dv"), native, triton_ref):
        error = (actual.float() - expected.float()).abs().max().item()
        limit = 4 * eps * max(1.0, expected.float().abs().max().item())
        assert error <= limit, f"{name} native/Triton max error {error} > {limit}"
        assert torch.isfinite(actual.float()).all()


def test_dim256_native_cute_varlen_delta_matches_triton():
    torch.manual_seed(2233)
    q_lens, k_lens = (33, 65), (65, 97)
    max_q, max_k = max(q_lens), max(k_lens)
    heads, head_dim = 1, 256
    device = torch.device("cuda")
    cu_q = torch.tensor([0, q_lens[0], sum(q_lens)], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, k_lens[0], sum(k_lens)], dtype=torch.int32, device=device)
    q = (
        torch.empty((sum(q_lens), heads, head_dim), device=device)
        .uniform_(-0.5, 0.5)
        .bfloat16()
    )
    k = (
        torch.empty((sum(k_lens), heads, head_dim), device=device)
        .uniform_(-0.5, 0.5)
        .bfloat16()
    )
    v = torch.empty_like(k).uniform_(-0.5, 0.5)
    do = torch.empty_like(q).uniform_(-0.5, 0.5)
    common = (do, q, k, v, cu_q, cu_k, max_q, max_k)
    native = hstu_varlen_bwd_256_cute(*common, None, None, None, max_k, 0, 0.6)[:3]
    triton_ref = hstu_varlen_bwd_256(
        *common, None, None, None, None, 1, max_k, 0, 0.6, None
    )[:3]
    for name, actual, expected in zip(("dq", "dk", "dv"), native, triton_ref):
        error = (actual.float() - expected.float()).abs().max().item()
        assert error <= 0.01, f"{name} native/Triton max error is {error}"


def test_dim256_native_cute_target_delta_matches_triton():
    torch.manual_seed(3344)
    q_lens, k_lens = (33, 65), (65, 97)
    max_q, max_k = max(q_lens), max(k_lens)
    heads, head_dim = 1, 256
    device = torch.device("cuda")
    cu_q = torch.tensor([0, q_lens[0], sum(q_lens)], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, k_lens[0], sum(k_lens)], dtype=torch.int32, device=device)
    num_targets = torch.tensor([8, 16], dtype=torch.int32, device=device)
    q = (
        torch.empty((sum(q_lens), heads, head_dim), device=device)
        .uniform_(-0.5, 0.5)
        .bfloat16()
    )
    k = (
        torch.empty((sum(k_lens), heads, head_dim), device=device)
        .uniform_(-0.5, 0.5)
        .bfloat16()
    )
    v = torch.empty_like(k).uniform_(-0.5, 0.5)
    do = torch.empty_like(q).uniform_(-0.5, 0.5)
    common = (do, q, k, v, cu_q, cu_k, max_q, max_k)
    native = hstu_varlen_bwd_256_cute(
        *common,
        None,
        None,
        None,
        max_k,
        0,
        0.6,
        num_targets=num_targets,
        target_group_size=4,
    )[:3]
    triton_ref = hstu_varlen_bwd_256(
        *common, None, None, None, num_targets, 4, max_k, 0, 0.6, None
    )[:3]
    for name, actual, expected in zip(("dq", "dk", "dv"), native, triton_ref):
        error = (actual.float() - expected.float()).abs().max().item()
        assert error <= 0.01, f"{name} native/Triton max error is {error}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "mask_case", ["none", "causal", "local", "target", "arbitrary"]
)
def test_dim256_gradients_match_reference(dtype, mask_case):
    torch.manual_seed(1234)
    seqlen, heads, head_dim = 65, 2, 256
    alpha = 0.7
    device = torch.device("cuda")
    cu = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    q_ref, k_ref, v_ref, do_ref = [
        torch.empty((seqlen, heads, head_dim), device=device).uniform_(-1, 1)
        for _ in range(4)
    ]
    q, k, v, do = [x.to(dtype) for x in (q_ref, k_ref, v_ref, do_ref)]
    mask_inputs = _make_mask(mask_case, seqlen, device)

    dq, dk, dv, _ = hstu_varlen_bwd_100(
        do,
        q,
        k,
        v,
        cu,
        cu,
        seqlen,
        seqlen,
        None,
        None,
        None,
        None,
        mask_inputs.num_targets,
        mask_inputs.target_group_size,
        mask_inputs.window_left,
        mask_inputs.window_right,
        alpha,
        None,
        False,
        mask_inputs.func,
        False,
    )
    ref_grads = _torch_grads(
        q_ref, k_ref, v_ref, do_ref, mask_inputs.mask, alpha, seqlen
    )
    low_grads = _torch_grads(q, k, v, do, mask_inputs.mask, alpha, seqlen)

    for name, got, low, ref in zip(
        ("dq", "dk", "dv"), (dq, dk, dv), low_grads, ref_grads
    ):
        _assert_error_tracks_low_precision(name, got, low, ref)


def test_dim256_varlen_delta_q_and_noncontiguous_do():
    torch.manual_seed(5678)
    heads, head_dim = 2, 256
    q_lens = (33, 65)
    k_lens = (65, 97)
    max_q, max_k = max(q_lens), max(k_lens)
    device = torch.device("cuda")
    cu_q = torch.tensor((0, q_lens[0], sum(q_lens)), dtype=torch.int32, device=device)
    cu_k = torch.tensor((0, k_lens[0], sum(k_lens)), dtype=torch.int32, device=device)
    q = torch.empty((sum(q_lens), heads, head_dim), device=device).uniform_(-1, 1)
    k = torch.empty((sum(k_lens), heads, head_dim), device=device).uniform_(-1, 1)
    v = torch.empty_like(k).uniform_(-1, 1)
    # Last-dimension stride != 1 exercises the fallback input layout.
    do = (
        torch.empty((sum(q_lens), head_dim, heads), device=device)
        .uniform_(-1, 1)
        .transpose(1, 2)
    )
    q_bf16, k_bf16, v_bf16, do_bf16 = [x.to(torch.bfloat16) for x in (q, k, v, do)]

    got = hstu_varlen_bwd_100(
        do_bf16,
        q_bf16,
        k_bf16,
        v_bf16,
        cu_q,
        cu_k,
        max_q,
        max_k,
        None,
        None,
        None,
        None,
        None,
        1,
        -1,
        0,
        0.6,
        None,
        False,
        None,
        False,
    )[:3]

    refs = (torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v))
    q_offset = k_offset = 0
    for q_len, k_len in zip(q_lens, k_lens):
        rows = torch.arange(q_len, device=device)[:, None] + k_len - q_len
        cols = torch.arange(k_len, device=device)[None, :]
        grads = _torch_grads(
            q[q_offset : q_offset + q_len],
            k[k_offset : k_offset + k_len],
            v[k_offset : k_offset + k_len],
            do[q_offset : q_offset + q_len],
            cols <= rows,
            0.6,
            max_q,
        )
        refs[0][q_offset : q_offset + q_len] = grads[0]
        refs[1][k_offset : k_offset + k_len] = grads[1]
        refs[2][k_offset : k_offset + k_len] = grads[2]
        q_offset += q_len
        k_offset += k_len

    for name, candidate, reference in zip(("dq", "dk", "dv"), got, refs):
        max_error = (candidate.float() - reference).abs().max().item()
        relative = max_error / (reference.abs().max().item() + 1.0e-12)
        assert relative < 1.0e-2, f"{name} relative max error is {relative:.4e}"
        assert torch.isfinite(candidate.float()).all()


def test_dim256_delta_q_arbitrary_mask_uses_packed_query_rows():
    """Arbitrary-mask rows are indexed by packed Q, not shifted score rows."""
    torch.manual_seed(6789)
    heads, head_dim = 1, 256
    q_lens = (31, 67)
    k_lens = (63, 99)
    max_q, max_k = max(q_lens), max(k_lens)
    device = torch.device("cuda")
    cu_q = torch.tensor((0, q_lens[0], sum(q_lens)), dtype=torch.int32, device=device)
    cu_k = torch.tensor((0, k_lens[0], sum(k_lens)), dtype=torch.int32, device=device)
    q, k, v, do = [
        torch.empty((tokens, heads, head_dim), device=device).uniform_(-1, 1)
        for tokens in (sum(q_lens), sum(k_lens), sum(k_lens), sum(q_lens))
    ]
    q_bf16, k_bf16, v_bf16, do_bf16 = [
        tensor.to(torch.bfloat16) for tensor in (q, k, v, do)
    ]
    func = torch.zeros((1, 3, sum(q_lens) + 256), dtype=torch.int32, device=device)

    refs = (torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v))
    q_offset = k_offset = 0
    for q_len, k_len in zip(q_lens, k_lens):
        query_rows = torch.arange(q_len, device=device)
        score_rows = query_rows + k_len - q_len
        hole_start = torch.clamp(score_rows - 7, min=0)
        hole_end = torch.clamp(score_rows - 2, min=0)
        last = torch.clamp(score_rows + 5, max=k_len)
        func[0, 0, q_offset : q_offset + q_len] = hole_start
        func[0, 1, q_offset : q_offset + q_len] = hole_end
        func[0, 2, q_offset : q_offset + q_len] = last
        cols = torch.arange(k_len, device=device)[None, :]
        mask = (cols < last[:, None]) & ~(
            (cols >= hole_start[:, None]) & (cols < hole_end[:, None])
        )
        grads = _torch_grads(
            q[q_offset : q_offset + q_len],
            k[k_offset : k_offset + k_len],
            v[k_offset : k_offset + k_len],
            do[q_offset : q_offset + q_len],
            mask,
            0.6,
            max_q,
        )
        refs[0][q_offset : q_offset + q_len] = grads[0]
        refs[1][k_offset : k_offset + k_len] = grads[1]
        refs[2][k_offset : k_offset + k_len] = grads[2]
        q_offset += q_len
        k_offset += k_len

    got = hstu_varlen_bwd_100(
        do_bf16,
        q_bf16,
        k_bf16,
        v_bf16,
        cu_q,
        cu_k,
        max_q,
        max_k,
        None,
        None,
        None,
        None,
        None,
        1,
        -1,
        -1,
        0.6,
        None,
        False,
        func,
        False,
    )[:3]
    for name, candidate, reference in zip(("dq", "dk", "dv"), got, refs):
        max_error = (candidate.float() - reference).abs().max().item()
        relative = max_error / (reference.abs().max().item() + 1.0e-12)
        assert relative < 1.0e-2, f"{name} relative max error is {relative:.4e}"


def test_dim256_causal_boundary_lengths():
    torch.manual_seed(7890)
    lengths = (1, 63, 64, 127, 128, 129, 255, 256)
    max_seqlen = max(lengths)
    heads, head_dim = 1, 256
    device = torch.device("cuda")
    cu = torch.tensor(
        (0, *torch.tensor(lengths).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    total = sum(lengths)
    q, k, v, do = [
        torch.empty((total, heads, head_dim), device=device).uniform_(-0.5, 0.5)
        for _ in range(4)
    ]
    candidate = hstu_varlen_bwd_100(
        do.to(torch.bfloat16),
        q.to(torch.bfloat16),
        k.to(torch.bfloat16),
        v.to(torch.bfloat16),
        cu,
        cu,
        max_seqlen,
        max_seqlen,
        None,
        None,
        None,
        None,
        None,
        1,
        -1,
        0,
        0.7,
        None,
        False,
        None,
        False,
    )[:3]
    reference = (torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v))
    offset = 0
    for length in lengths:
        mask = torch.ones((length, length), dtype=torch.bool, device=device).tril()
        grads = _torch_grads(
            q[offset : offset + length],
            k[offset : offset + length],
            v[offset : offset + length],
            do[offset : offset + length],
            mask,
            0.7,
            max_seqlen,
        )
        for output, grad in zip(reference, grads):
            output[offset : offset + length] = grad
        offset += length

    for name, actual, expected in zip(("dq", "dk", "dv"), candidate, reference):
        max_error = (actual.float() - expected).abs().max().item()
        relative = max_error / (expected.abs().max().item() + 1.0e-12)
        assert relative < 1.0e-2, f"{name} relative max error is {relative:.4e}"


def test_dim256_long_causal_matches_reference():
    torch.manual_seed(8901)
    seqlen, heads, head_dim = 2048, 1, 256
    device = torch.device("cuda")
    cu = torch.tensor((0, seqlen), dtype=torch.int32, device=device)
    q, k, v, do = [
        torch.empty((seqlen, heads, head_dim), device=device).uniform_(-0.25, 0.25)
        for _ in range(4)
    ]
    candidate = hstu_varlen_bwd_100(
        do.to(torch.bfloat16),
        q.to(torch.bfloat16),
        k.to(torch.bfloat16),
        v.to(torch.bfloat16),
        cu,
        cu,
        seqlen,
        seqlen,
        None,
        None,
        None,
        None,
        None,
        1,
        -1,
        0,
        0.7,
        None,
        False,
        None,
        False,
    )[:3]
    rows = torch.arange(seqlen, device=device)[:, None]
    cols = torch.arange(seqlen, device=device)[None, :]
    reference = _torch_grads(q, k, v, do, cols <= rows, 0.7, seqlen)
    for name, actual, expected in zip(("dq", "dk", "dv"), candidate, reference):
        max_error = (actual.float() - expected).abs().max().item()
        relative = max_error / (expected.abs().max().item() + 1.0e-12)
        assert relative < 1.0e-2, f"{name} relative max error is {relative:.4e}"


def test_dim256_writes_packed_gradient_buffers():
    torch.manual_seed(9012)
    batch, heads, seqlen, head_dim = 2, 4, 64, 256
    device = torch.device("cuda")
    cu = torch.arange(0, (batch + 1) * seqlen, seqlen, dtype=torch.int32, device=device)
    qkv = torch.randn(
        (batch * seqlen, 3, heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    q, k, v = qkv.unbind(1)
    do = torch.randn_like(q)
    expected = hstu_varlen_bwd_100(
        do,
        q,
        k,
        v,
        cu,
        cu,
        seqlen,
        seqlen,
        None,
        None,
        None,
        None,
        None,
        1,
        -1,
        0,
        1.0,
        None,
        False,
        None,
        False,
    )[:3]
    dqkv = torch.full_like(qkv, float("nan"))
    returned = hstu_varlen_bwd_100(
        do,
        q,
        k,
        v,
        cu,
        cu,
        seqlen,
        seqlen,
        dqkv[:, 0],
        dqkv[:, 1],
        dqkv[:, 2],
        None,
        None,
        1,
        -1,
        0,
        1.0,
        None,
        False,
        None,
        False,
    )[:3]

    assert torch.isfinite(dqkv.float()).all()
    for idx, (actual, reference) in enumerate(zip(returned, expected)):
        assert actual.data_ptr() == dqkv[:, idx].data_ptr()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_dim256_backward_is_cuda_graph_capturable():
    torch.manual_seed(2345)
    seqlen, heads, head_dim = 64, 2, 256
    device = torch.device("cuda")
    cu = torch.tensor((0, seqlen), dtype=torch.int32, device=device)
    q, k, v, do = [
        torch.randn((seqlen, heads, head_dim), dtype=torch.bfloat16, device=device)
        for _ in range(4)
    ]
    outputs = [torch.empty_like(q) for _ in range(3)]

    def launch():
        return hstu_varlen_bwd_100(
            do,
            q,
            k,
            v,
            cu,
            cu,
            seqlen,
            seqlen,
            *outputs,
            None,
            None,
            1,
            -1,
            0,
            0.7,
            None,
            False,
            None,
            False,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        launch()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = launch()
    graph.replay()

    for name, grad in zip(("dq", "dk", "dv"), captured[:3]):
        assert torch.isfinite(grad.float()).all(), f"{name} contains NaN/Inf"


def test_dim256_end_to_end_autograd_dispatch():
    from hstu.cuda_hstu_attention import hstu_attn_varlen_func

    torch.manual_seed(3456)
    batch, heads, seqlen, head_dim = 2, 2, 65, 256
    device = torch.device("cuda")
    cu = torch.arange(0, (batch + 1) * seqlen, seqlen, dtype=torch.int32, device=device)
    q, k, v = [
        torch.randn(
            (batch * seqlen, heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    out = hstu_attn_varlen_func(
        q,
        k,
        v,
        cu,
        cu,
        None,
        None,
        seqlen,
        seqlen,
        -1,
        None,
        None,
        1,
        (-1, 0),
        0.7,
        None,
        False,
        None,
    )
    out.backward(torch.randn_like(out))

    assert out.shape == q.shape
    assert torch.isfinite(out.float()).all()
    for name, grad in (("dq", q.grad), ("dk", k.grad), ("dv", v.grad)):
        assert grad is not None, f"{name} was not produced by autograd"
        assert torch.isfinite(grad.float()).all(), f"{name} contains NaN/Inf"
