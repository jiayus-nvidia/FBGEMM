#!/usr/bin/env python3

"""Benchmark the native CuTe DSL D=256 Blackwell backward path.

Example:
    PYTHONPATH=src python benchmarks/benchmark_bwd_dim256.py
"""

import argparse
import math

import torch

from hstu_blackwell.hstu_bwd_256_cute import hstu_varlen_bwd_256_cute


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def _shapes(value: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        tuple(int(item) for item in shape.lower().split("x"))
        for shape in value.split(",")
    )


def _do_bench(
    launch,
    *,
    warmup_ms: int,
    rep_ms: int,
) -> tuple[float, float, float]:
    """Return CUDA-event p20/p50/p80 without a Triton benchmark dependency."""
    for _ in range(5):
        launch()
    torch.cuda.synchronize()
    probe_start = torch.cuda.Event(enable_timing=True)
    probe_end = torch.cuda.Event(enable_timing=True)
    probe_start.record()
    for _ in range(10):
        launch()
    probe_end.record()
    probe_end.synchronize()
    estimate_ms = max(probe_start.elapsed_time(probe_end) / 10.0, 1.0e-3)
    warmup_iters = max(5, math.ceil(warmup_ms / estimate_ms))
    for _ in range(warmup_iters):
        launch()
    torch.cuda.synchronize()

    rep_iters = max(20, math.ceil(rep_ms / estimate_ms))
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep_iters)]
    for start, end in zip(starts, ends):
        start.record()
        launch()
        end.record()
    ends[-1].synchronize()
    samples = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(starts, ends)],
        dtype=torch.float64,
    )
    p20, median, p80 = torch.quantile(
        samples, samples.new_tensor([0.2, 0.5, 0.8])
    )
    return float(p20), float(median), float(p80)


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
        "p20_ms,median_ms,p80_ms,effective_tflops",
        flush=True,
    )
    for mask in args.masks:
        if mask not in ("none", "causal"):
            raise ValueError(f"unsupported benchmark mask: {mask}")
        for batch, heads in args.shapes:
            for seqlen in args.seqlens:
                total_tokens, head_dim = batch * seqlen, 256
                q = torch.randn(
                    (total_tokens, heads, head_dim),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                k = torch.randn_like(q)
                v = torch.randn_like(q)
                do = torch.randn_like(q)
                cu_seqlens = torch.arange(
                    0,
                    total_tokens + 1,
                    seqlen,
                    device="cuda",
                    dtype=torch.int32,
                )
                outputs = tuple(torch.empty_like(q) for _ in range(3))
                window_left, window_right = (
                    (seqlen, 0) if mask == "causal" else (seqlen, seqlen)
                )

                def launch() -> None:
                    hstu_varlen_bwd_256_cute(
                        do,
                        q,
                        k,
                        v,
                        cu_seqlens,
                        cu_seqlens,
                        seqlen,
                        seqlen,
                        *outputs,
                        window_left,
                        window_right,
                        1.0,
                    )

                p20, median, p80 = _do_bench(
                    launch,
                    warmup_ms=args.warmup_ms,
                    rep_ms=args.rep_ms,
                )
                dense_flops = 14.0 * batch * heads * seqlen * seqlen * head_dim
                effective_tflops = dense_flops / (float(median) * 1.0e9)
                print(
                    f"cute,{mask},{batch},{heads},{seqlen},"
                    f"{float(p20):.6f},{float(median):.6f},{float(p80):.6f},"
                    f"{effective_tflops:.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
