#!/usr/bin/env python3

"""Benchmark public forward plus autograd backward for Blackwell D=256 HSTU.

Example:
    PYTHONPATH=src:. python benchmarks/benchmark_e2e_dim256.py
"""

import argparse

import torch

from benchmark_bwd_dim256 import _csv_ints, _do_bench, _shapes
from hstu.cuda_hstu_attention import hstu_attn_varlen_func


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", type=_shapes, default=((1, 2), (8, 8)))
    parser.add_argument("--seqlens", type=_csv_ints, default=(128, 512, 2048, 8192))
    parser.add_argument(
        "--masks",
        type=lambda value: tuple(value.split(",")),
        default=("none", "causal"),
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("this benchmark requires an SM100-family GPU")

    torch.manual_seed(args.seed)
    print(
        "implementation,mask,batch,heads,seqlen,"
        "eager_forward_p20_ms,eager_forward_median_ms,eager_forward_p80_ms,"
        "eager_e2e_p20_ms,eager_e2e_median_ms,eager_e2e_p80_ms,"
        "graph_forward_p20_ms,graph_forward_median_ms,graph_forward_p80_ms,"
        "graph_e2e_p20_ms,graph_e2e_median_ms,graph_e2e_p80_ms,"
        "graph_backward_share_pct,eager_over_graph",
        flush=True,
    )
    for mask in args.masks:
        if mask not in ("none", "causal"):
            raise ValueError(f"unsupported benchmark mask: {mask}")
        for batch, heads in args.shapes:
            for seqlen in args.seqlens:
                total_tokens, head_dim = batch * seqlen, 256
                q, k, v = [
                    torch.randn(
                        (total_tokens, heads, head_dim),
                        device="cuda",
                        dtype=torch.bfloat16,
                        requires_grad=True,
                    )
                    for _ in range(3)
                ]
                do = torch.randn_like(q)
                cu_seqlens = torch.arange(
                    0,
                    total_tokens + 1,
                    seqlen,
                    device="cuda",
                    dtype=torch.int32,
                )
                window_size = (-1, 0) if mask == "causal" else (-1, -1)

                def forward(
                    q_input: torch.Tensor,
                    k_input: torch.Tensor,
                    v_input: torch.Tensor,
                ) -> torch.Tensor:
                    return hstu_attn_varlen_func(
                        q_input,
                        k_input,
                        v_input,
                        cu_seqlens,
                        cu_seqlens,
                        None,
                        None,
                        seqlen,
                        seqlen,
                        -1,
                        None,
                        None,
                        1,
                        window_size,
                        1.0,
                        None,
                        False,
                        None,
                    )

                def launch_forward() -> None:
                    forward(q, k, v)

                def launch_e2e() -> None:
                    out = forward(q, k, v)
                    torch.autograd.grad(out, (q, k, v), do)

                eager_forward = _do_bench(
                    launch_forward,
                    warmup_ms=args.warmup_ms,
                    rep_ms=args.rep_ms,
                )
                eager_e2e = _do_bench(
                    launch_e2e,
                    warmup_ms=args.warmup_ms,
                    rep_ms=args.rep_ms,
                )

                # Use fresh leaves for capture so their AccumulateGrad nodes are
                # created on the graph stream rather than an eager warmup stream.
                q_graph, k_graph, v_graph = [
                    tensor.detach().clone().requires_grad_() for tensor in (q, k, v)
                ]
                do_graph = do.detach().clone()
                q_forward, k_forward, v_forward = [
                    tensor.detach().clone() for tensor in (q, k, v)
                ]
                torch.cuda.synchronize()

                forward_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(forward_graph):
                    captured_forward = forward(q_forward, k_forward, v_forward)

                e2e_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(e2e_graph):
                    captured_out = forward(q_graph, k_graph, v_graph)
                    captured_grads = torch.autograd.grad(
                        captured_out,
                        (q_graph, k_graph, v_graph),
                        do_graph,
                    )

                graph_forward = _do_bench(
                    forward_graph.replay,
                    warmup_ms=args.warmup_ms,
                    rep_ms=args.rep_ms,
                )
                graph_e2e = _do_bench(
                    e2e_graph.replay,
                    warmup_ms=args.warmup_ms,
                    rep_ms=args.rep_ms,
                )
                backward_share = max(
                    0.0,
                    100.0 * (graph_e2e[1] - graph_forward[1]) / graph_e2e[1],
                )
                eager_over_graph = eager_e2e[1] / graph_e2e[1]
                print(
                    f"cute,{mask},{batch},{heads},{seqlen},"
                    f"{eager_forward[0]:.6f},{eager_forward[1]:.6f},"
                    f"{eager_forward[2]:.6f},"
                    f"{eager_e2e[0]:.6f},{eager_e2e[1]:.6f},{eager_e2e[2]:.6f},"
                    f"{graph_forward[0]:.6f},{graph_forward[1]:.6f},"
                    f"{graph_forward[2]:.6f},"
                    f"{graph_e2e[0]:.6f},{graph_e2e[1]:.6f},"
                    f"{graph_e2e[2]:.6f},{backward_share:.1f},"
                    f"{eager_over_graph:.2f}",
                    flush=True,
                )

                # Keep captured tensors alive until both measurements finish.
                del captured_forward, captured_out, captured_grads


if __name__ == "__main__":
    main()
