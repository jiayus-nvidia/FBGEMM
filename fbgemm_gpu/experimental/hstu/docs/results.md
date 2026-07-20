# HSTU Blackwell D=256 backward results

## Status

The branch now contains a native CuTe DSL D=256 backward implementation for
Blackwell. It uses the FA4 head-dim-256 decomposition: one 2-CTA kernel produces
dQ and a second 2-CTA kernel produces dK/dV. Both kernels recompute score tiles
and use HSTU's
`P = silu(alpha*S)` / `dS = alpha*dP*silu'(alpha*S)` math; no score matrix is
materialized in global memory.

Native predicates cover unmasked, causal, local, target-group, packed-varlen,
and delta-Q cases. Arbitrary interval masks retain an explicit Triton fallback.
The public selector uses CuTe when `max(max_seqlen_q, max_seqlen_k) >= 2048`
and Triton below that measured crossover. Set `FBGEMM_HSTU_D256_CUTE=force`
to force CuTe, or `=off` to disable it.

## Environment

- FBGEMM baseline: `2c31d908acfbe0210500876be7c0406e64033dd3`
- FA4 reference PR head: `c1e5e8c`
- GPU: NVIDIA B300 SXM6 AC, SM103, 275040 MiB
- Driver: 595.58.03
- PyTorch: 2.12.1+cu130
- CUDA runtime reported by PyTorch: 13.0
- NVIDIA CUTLASS DSL: 4.4.0
- Triton: 3.7.1

## Correctness and integration

| Gate | Result |
|---|---:|
| D=256 targeted suite | 25 passed |
| Direct CuTe/Triton BF16+FP16, no/causal/local comparison | 6 passed |
| Direct CuTe/Triton delta-Q and target-delta comparison | 2 passed |
| D=64/128/256 layout regressions + TVM-FFI integration | 18 passed |
| Native CuTe CUDA Graph capture (`force`) | passed |
| Ruff, `compileall`, and `git diff --check` | passed |

The matrix covers BF16 and FP16; unmasked, causal, local, target, and arbitrary
masks; packed variable-length and unequal Q/K batches; non-contiguous dO;
preallocated strided gradient buffers; boundary lengths 1, 63, 64, 127, 128,
129, 255, and 256; a 2048-token PyTorch reference row; and the public autograd
entry. Arbitrary-mask correctness is provided by the retained Triton fallback.

Primary commands:

```bash
TRITON_CACHE_DIR=/tmp/hstu-bwd-d256-triton-cache PYTHONPATH=src \
  python -m pytest test/test_bwd_dim256.py -q -s
TRITON_CACHE_DIR=/tmp/hstu-bwd-d256-triton-cache PYTHONPATH=src \
  python -m pytest test/test_bwd_do_layout.py test/test_tvm_ffi_integration.py -q -s
FBGEMM_HSTU_D256_CUTE=force PYTHONPATH=src \
  python -m pytest test/test_bwd_dim256.py -k cuda_graph -q -s
```

## Native implementation notes

- DQ uses a `(128,128,256)` 2-CTA tile; DK/DV uses `(128,64,256)`.
- Packed-varlen launch grids use maximum per-sequence length, not total packed
  tokens. This removes an approximately batch-size-fold CTA over-launch present
  in a direct application of the reference wrapper.
- HSTU's `1/max_seqlen_q` normalization is fused into the native epilogues.
- The inherited LSE/dPsum transport accepts one zero-stride dummy scalar; no
  per-token softmax statistics are allocated or consumed.
- D=64 and D=128 code and compile-cache keys are unchanged.

## Performance

Matched public wrappers used preallocated outputs, BF16, p20/median/p80 timing,
and the same B300. Effective TFLOP/s uses `14*B*H*S^2*D` as a nominal dense-work
metric. Full distributions are in `benchmark_bwd_dim256.csv`.

| Mask | B,H,S | Triton median (ms) | CuTe median (ms) | CuTe speedup |
|---|---:|---:|---:|---:|
| none | 1,2,2048 | 0.150528 | 0.150592 | 1.00x |
| none | 1,2,8192 | 2.157552 | 0.478208 | 4.51x |
| none | 8,8,2048 | 3.942336 | 1.539104 | 2.56x |
| none | 8,8,8192 | 59.131920 | 13.810624 | 4.28x |
| causal | 1,2,2048 | 0.150528 | 0.154592 | 0.97x |
| causal | 1,2,8192 | 2.165680 | 0.381952 | 5.67x |
| causal | 8,8,2048 | 3.946496 | 1.340416 | 2.94x |
| causal | 8,8,8192 | 59.219984 | 9.182272 | 6.45x |

At S=128/512, Triton's lighter launch footprint wins, which is why those rows
remain on Triton in auto mode. The benchmark is reproducible with:

```bash
PYTHONPATH=src python benchmarks/benchmark_bwd_dim256.py
```

## Remaining work

- Add native arbitrary-interval predicates if that workload benefits from the
  2-CTA path; correctness currently remains on Triton.
- Measure and retune the crossover on SM100/B200 in addition to SM103/B300.
- Remove the inherited, value-dead stats transport pipelines as a follow-up
  compile-time/shared-memory cleanup.
