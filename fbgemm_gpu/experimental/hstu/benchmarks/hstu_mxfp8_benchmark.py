# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from hstu_blackwell import (
    hstu_varlen_bwd_100,
    hstu_varlen_bwd_mxfp8_100,
    hstu_varlen_fwd_100,
    hstu_varlen_fwd_mxfp8_100,
)


MODES = ("bf16", "mxfp8")
PHASES = ("fwd", "bwd", "e2e")


def _parse_configs(value):
    configs = []
    for item in value.split(","):
        batch_size, seqlen = item.lower().split("x", maxsplit=1)
        configs.append((int(batch_size), int(seqlen)))
    return configs


def _parse_list(value, choices):
    selected = tuple(dict.fromkeys(item.strip().lower() for item in value.split(",")))
    if not selected or any(item not in choices for item in selected):
        raise argparse.ArgumentTypeError(f"expected a comma-separated subset of {choices}")
    return selected


def _inputs(batch_size, seqlen, heads, head_dim):
    shape = (batch_size * seqlen, heads, head_dim)
    device = torch.device("cuda")
    offsets = torch.arange(
        0,
        (batch_size + 1) * seqlen,
        seqlen,
        dtype=torch.int32,
        device=device,
    )
    return {
        "q": torch.randn(shape, dtype=torch.bfloat16, device=device),
        "k": torch.randn(shape, dtype=torch.bfloat16, device=device),
        "v": torch.randn(shape, dtype=torch.bfloat16, device=device),
        "dout": torch.randn(shape, dtype=torch.bfloat16, device=device),
        "offsets": offsets,
    }


def _forward(mode, tensors, seqlen, alpha):
    function = hstu_varlen_fwd_mxfp8_100 if mode == "mxfp8" else hstu_varlen_fwd_100
    return function(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["offsets"],
        tensors["offsets"],
        seqlen,
        seqlen,
        None,
        None,
        1,
        -1,
        0,
        alpha,
        None,
        None,
    )[0]


def _backward(mode, tensors, seqlen, alpha):
    function = hstu_varlen_bwd_mxfp8_100 if mode == "mxfp8" else hstu_varlen_bwd_100
    return function(
        tensors["dout"],
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["offsets"],
        tensors["offsets"],
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
        alpha,
        None,
        False,
        None,
        False,
    )[:3]


def _run(mode, phase, tensors, seqlen, alpha):
    if phase == "fwd":
        return _forward(mode, tensors, seqlen, alpha)
    if phase == "bwd":
        return _backward(mode, tensors, seqlen, alpha)
    output = _forward(mode, tensors, seqlen, alpha)
    gradients = _backward(mode, tensors, seqlen, alpha)
    return output, gradients


def _measure(function, warmup_iters, bench_iters):
    for _ in range(warmup_iters):
        function()
    torch.cuda.synchronize()

    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(bench_iters)
    ]
    for start, end in events:
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in events)


def _effective_flops(batch_size, seqlen, heads, head_dim, phase):
    causal_pairs = seqlen * (seqlen + 1) // 2
    forward_flops = 4 * batch_size * heads * head_dim * causal_pairs
    multiplier = {"fwd": 1.0, "bwd": 2.5, "e2e": 3.5}[phase]
    return forward_flops * multiplier


def _profile(args):
    batch_size, seqlen = _parse_configs(args.profile_config)[0]
    tensors = _inputs(batch_size, seqlen, args.heads, args.head_dim)
    alpha = 1.0 / math.sqrt(args.head_dim)
    function = lambda: _run(
        args.profile_mode,
        args.profile_phase,
        tensors,
        seqlen,
        alpha,
    )
    for _ in range(args.warmup_iters):
        function()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"{args.profile_mode}_{args.profile_phase}")
    for _ in range(args.profile_replays):
        function()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(
        f"Profiled {args.profile_mode} {args.profile_phase}: "
        f"B={batch_size}, S={seqlen}, H={args.heads}, D={args.head_dim}, "
        f"replays={args.profile_replays}"
    )


def _sweep(args):
    configs = _parse_configs(args.configs)
    alpha = 1.0 / math.sqrt(args.head_dim)
    records = []
    measurements = {}

    for batch_size, seqlen in configs:
        tensors = _inputs(batch_size, seqlen, args.heads, args.head_dim)
        for mode in args.modes:
            for phase in args.phases:
                elapsed_ms = _measure(
                    lambda mode=mode, phase=phase: _run(
                        mode,
                        phase,
                        tensors,
                        seqlen,
                        alpha,
                    ),
                    args.warmup_iters,
                    args.bench_iters,
                )
                effective_tflops = (
                    _effective_flops(
                        batch_size,
                        seqlen,
                        args.heads,
                        args.head_dim,
                        phase,
                    )
                    / (elapsed_ms * 1.0e-3)
                    / 1.0e12
                )
                record = {
                    "mode": mode,
                    "phase": phase,
                    "batch_size": batch_size,
                    "seqlen": seqlen,
                    "heads": args.heads,
                    "head_dim": args.head_dim,
                    "elapsed_ms": elapsed_ms,
                    "effective_tflops": effective_tflops,
                }
                records.append(record)
                measurements[(batch_size, seqlen, mode, phase)] = elapsed_ms

        for phase in args.phases:
            bf16_ms = measurements[(batch_size, seqlen, "bf16", phase)]
            mxfp8_ms = measurements[(batch_size, seqlen, "mxfp8", phase)]
            print(
                f"B={batch_size:>3} S={seqlen:>5} {phase:>3}: "
                f"BF16={bf16_ms:>9.3f} ms  MXFP8={mxfp8_ms:>9.3f} ms  "
                f"speedup={bf16_ms / mxfp8_ms:>6.3f}x"
            )

    output = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "warmup_iters": args.warmup_iters,
        "bench_iters": args.bench_iters,
        "records": records,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        default="64x128,16x512,4x2048,1x8192",
    )
    parser.add_argument(
        "--modes",
        type=lambda value: _parse_list(value, MODES),
        default=MODES,
    )
    parser.add_argument(
        "--phases",
        type=lambda value: _parse_list(value, PHASES),
        default=PHASES,
    )
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--bench-iters", type=int, default=10)
    parser.add_argument("--output-json", default="mxfp8_bf16_results.json")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-mode", choices=MODES, default="mxfp8")
    parser.add_argument("--profile-phase", choices=PHASES, default="fwd")
    parser.add_argument("--profile-config", default="16x512")
    parser.add_argument("--profile-replays", type=int, default=1)
    args = parser.parse_args()

    if args.profile:
        _profile(args)
    else:
        if set(args.modes) != set(MODES):
            parser.error("sweep mode requires both bf16 and mxfp8")
        _sweep(args)


if __name__ == "__main__":
    main()
