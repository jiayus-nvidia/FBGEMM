# HSTU Blackwell D=256 results

## Status

D=256 backward is now CuTe-DSL-only. The public Blackwell entry always launches
the native dQ and dK/dV kernels, independent of sequence length or mask type.
Unmasked, causal, local, target-group, arbitrary interval, packed-varlen, and
delta-Q cases are native. The former D=256 Triton source, length crossover, and
experiment environment override have been removed.

The implementation uses the FA4 head-dim-256 two-kernel decomposition. Both
kernels recompute score tiles with HSTU's
`P = silu(alpha*S)` and `dS = alpha*dP*silu'(alpha*S)` math; no score matrix or
softmax-only statistics are materialized in global memory.

## Environment

- GPU: NVIDIA B300 SXM6 AC, SM103, 275040 MiB
- Driver: 595.58.03
- PyTorch: 2.12.1+cu130
- CUDA runtime reported by PyTorch: 13.0
- NVIDIA CUTLASS DSL: 4.4.0

## Correctness and integration

| Gate | Result |
|---|---:|
| D=256 targeted and public-entry suite | 36 passed |
| Public forward+backward reference matrix | 10 passed |
| D=64/128/256 layout and TVM-FFI regressions | 18 passed |
| Full-tile S=128 fast-path reference check | passed |
| Backward-only and full-E2E CUDA Graph capture with 10 replays | passed |
| Ruff, `compileall`, and `git diff --check` | passed |

The D=256 matrix compares both public forward outputs and dQ/dK/dV against
PyTorch for BF16 and FP16 across unmasked, causal, local, target-group, and
arbitrary interval attention. It also covers packed
variable-length and unequal Q/K batches; non-contiguous dO; preallocated strided
gradient buffers; boundary lengths 1, 63, 64, 127, 128, 129, 255, and 256; a
2048-token PyTorch reference row; and the public autograd entry. PyTorch is the
numerical oracle; no D=256 test imports Triton.

## Retained short-sequence changes

- Aligned, full-length unmasked batches skip residual and semantic predicates
  that are known true at compile time.
- The final per-kernel synchronization is CTA-local. `TmemAllocator.free`
  retains the required cross-CTA deallocation handshake.
- Residual masking is applied only to actual partial Q/K tiles.

Several larger structural changes were rejected on correctness or performance:
compile-time length buckets did not reduce the launch floor, a one-CTA dQ kernel
did not launch, and a 64-row dQ tile was faster but numerically invalid.

## Performance

The final dependency-free benchmark uses CUDA Events, preallocated outputs,
BF16, 100 ms warmup, and 500 ms sampling. Full p20/median/p80 data is in
`benchmark_bwd_dim256_cute_only.csv`.

| Mask | B,H | S=128 | S=512 | S=2048 | S=8192 |
|---|---:|---:|---:|---:|---:|
| none | 1,2 | 0.087168 | 0.093504 | 0.130144 | 0.435360 |
| none | 8,8 | 0.090240 | 0.285824 | 1.465616 | 13.480000 |
| causal | 1,2 | 0.087168 | 0.099456 | 0.144384 | 0.373920 |
| causal | 8,8 | 0.093280 | 0.294016 | 1.312768 | 9.247744 |

Values are median milliseconds on B300. To compare against the frozen baseline
without mixing timer implementations, the final CuTe code was also measured
once with the baseline's timer. The conservative same-timer short results are:

| Mask | B,H,S | Old CuTe (ms) | Final CuTe (ms) | CuTe gain | Final / historical Triton latency |
|---|---:|---:|---:|---:|---:|
| none | 1,2,128 | 0.093184 | 0.087104 | 1.070x | 3.038x |
| none | 1,2,512 | 0.103424 | 0.097248 | 1.064x | 2.109x |
| none | 8,8,128 | 0.099264 | 0.093216 | 1.065x | 2.460x |
| none | 8,8,512 | 0.306176 | 0.293888 | 1.042x | 0.993x |
| causal | 1,2,128 | 0.091168 | 0.089120 | 1.023x | 3.220x |
| causal | 1,2,512 | 0.103424 | 0.101408 | 1.020x | 2.199x |
| causal | 8,8,128 | 0.097312 | 0.097216 | 1.001x | 2.566x |
| causal | 8,8,512 | 0.302112 | 0.300064 | 1.007x | 1.014x |

Across the eight short rows, the conservative equal-weight geometric-mean gain
over the old CuTe code is 1.036x. Across the eight S=2048/8192 rows it is
1.026x, so the short optimization did not trade away long-sequence performance.
The final long-row geometric-mean speedup over the historical Triton prototype
is 3.01x.

Small-batch short rows remain limited by two approximately 40--43 us,
warp-specialized 2-CTA kernels. S=128 remains about 3.0--3.2x slower and S=512
about 2.1--2.2x slower than the deleted lightweight prototype at B=1,H=2. At
B=8,H=8,S=512, CuTe is at parity. Closing the remaining small-batch gap would
require a separate lightweight CuTe kernel rather than another scheduler flag.

## Forward plus backward E2E

The public autograd entry was measured in eager mode and as a captured CUDA
Graph. CUDA Graph medians below represent steady-state GPU execution of the
forward kernel followed by dQ and dK/dV; full distributions are in
`benchmark_e2e_dim256_cute_only.csv`.

| Mask | B,H | S=128 | S=512 | S=2048 | S=8192 |
|---|---:|---:|---:|---:|---:|
| none | 1,2 | 0.093152 | 0.105440 | 0.154592 | 0.534432 |
| none | 8,8 | 0.099072 | 0.315552 | 1.691712 | 16.520384 |
| causal | 1,2 | 0.095328 | 0.109664 | 0.167040 | 0.452736 |
| causal | 8,8 | 0.101472 | 0.318624 | 1.437824 | 10.955664 |

Values are median milliseconds on B300. Backward accounts for 80.6%--91.9% of
captured E2E GPU time. For small rows, ordinary eager execution is 1.3x--10.1x
slower than Graph replay because Python/autograd launch latency dominates; once
the B=8,H=8 workload reaches S=2048, eager and Graph times converge.
