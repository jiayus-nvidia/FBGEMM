#!/usr/bin/env python3

"""Compare Triton and native CuTe D=256 Blackwell backward paths.

Example:
    PYTHONPATH=src python benchmarks/benchmark_bwd_dim256.py
"""

import argparse

import torch
import triton

from hstu_blackwell.hstu_bwd_256 import hstu_varlen_bwd_256
from hstu_blackwell.hstu_bwd_256_cute import hstu_varlen_bwd_256_cute


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def _shapes(value: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        tuple(int(item) for item in shape.lower().split("x"))
        for shape in value.split(",")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", type=_shapes, default=((1, 2), (8, 8)))
    parser.add_argument("--seqlens", type=_csv_ints, default=(128, 512, 2048, 8192))
    parser.add_argument(
        "--masks",
        type=lambda value: tuple(value.split(",")),
        default=("none", "causal"),
    )
    parser.add_argument(
        "--implementations",
        type=lambda value: tuple(value.split(",")),
        default=("triton", "cute"),
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
        for implementation in args.implementations:
            if implementation not in ("triton", "cute"):
                raise ValueError(f"unsupported implementation: {implementation}")
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

                    if implementation == "triton":

                        def launch() -> None:
                            hstu_varlen_bwd_256(
                                do,
                                q,
                                k,
                                v,
                                cu_seqlens,
                                cu_seqlens,
                                seqlen,
                                seqlen,
                                *outputs,
                                None,
                                1,
                                window_left,
                                window_right,
                                1.0,
                                None,
                            )
                    else:

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

                    p20, median, p80 = triton.testing.do_bench(
                        launch,
                        warmup=args.warmup_ms,
                        rep=args.rep_ms,
                        quantiles=(0.2, 0.5, 0.8),
                    )
                    dense_flops = 14.0 * batch * heads * seqlen * seqlen * head_dim
                    effective_tflops = dense_flops / (float(median) * 1.0e9)
                    print(
                        f"{implementation},{mask},{batch},{heads},{seqlen},"
                        f"{float(p20):.6f},{float(median):.6f},{float(p80):.6f},"
                        f"{effective_tflops:.3f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
