# Native CuTe D=256 backward plan

## Frozen inputs

- FBGEMM baseline: `2c31d908acfbe0210500876be7c0406e64033dd3`
- FA4 reference: PR 2412 head `c1e5e8c`
- Hardware/toolchain: B300 SM103, PyTorch 2.12.1+cu130, CUDA 13.0,
  NVIDIA CUTLASS DSL 4.4.0
- Frozen comparison data: `benchmark_bwd_dim256_stats_cleanup.csv`

## Final contract

- Keep the public `hstu_varlen_bwd_100` interface and output-layout behavior.
- Implement every supported D=256 backward case in CuTe DSL: all sequence
  lengths, BF16/FP16, unmasked, causal, local, target-group, arbitrary interval,
  packed variable-length, and delta-Q inputs.
- Do not retain a D=256 Triton implementation, runtime selector, crossover,
  experiment environment override, or benchmark dependency.
- Leave the D=64/128 implementation and compile-cache behavior unchanged.

## Completed work

1. [x] Port the FA4 D=256 two-kernel decomposition: a 2-CTA dQ kernel followed
   by a 2-CTA dK/dV kernel.
2. [x] Replace softmax/LSE math with HSTU SiLU and dSiLU recomputation and fuse
   `1/max_seqlen_q` normalization into the gradient path.
3. [x] Adapt the public packed layouts, cu-seqlens, output buffers, current
   stream, and TVM-FFI compile cache.
4. [x] Implement native causal, local, target-group, delta-Q, and arbitrary
   interval predicates.
5. [x] Remove inherited softmax-only statistics, dead pipelines, dummy buffers,
   and unreachable scheduler scaffolding.
6. [x] Profile the S=128 fixed-cost floor and test short-sequence candidates.
7. [x] Retain only candidates that pass correctness and improve measured time:
   skip redundant predicates for aligned full-tile unmasked inputs, and use a
   CTA-local completion barrier before the allocator's cross-CTA TMEM release.
8. [x] Remove `hstu_bwd_256.py`, the length crossover, the
   `FBGEMM_HSTU_D256_CUTE` override, direct CuTe/Triton test branches, and the
   Triton benchmark import.
9. [x] Run the 36-test D=256 suite, public forward+backward reference checks,
   full-E2E CUDA Graph replay, D=64/128 layout and TVM-FFI regressions, static
   checks, and the final backward and E2E benchmarks.

## Rejected short-sequence candidates

- Compile-time S=128/512 buckets did not change the approximately 42 us kernel
  floor.
- A one-CTA dQ configuration failed at launch with `cudaErrorInvalidValue`.
- A 64-row dQ tile reduced time by about 22%, but failed dQ correctness; the
  split-D repair produced NaNs.
- Tighter causal trip bounds and relaxed cluster initialization regressed or
  produced unstable results.

All rejected experiments were reverted; no selector or dormant branch remains.

## Validation

```bash
PYTHONPATH=src:. python -m pytest test/test_bwd_dim256.py -q -s
PYTHONPATH=src python -m pytest \
  test/test_bwd_do_layout.py test/test_tvm_ffi_integration.py -q -s
PYTHONPATH=src python benchmarks/benchmark_bwd_dim256.py
PYTHONPATH=src:. python benchmarks/benchmark_e2e_dim256.py
python -m ruff check src/hstu_blackwell test/test_bwd_dim256.py \
  benchmarks/benchmark_bwd_dim256.py benchmarks/benchmark_e2e_dim256.py
python -m compileall -q src/hstu_blackwell test/test_bwd_dim256.py \
  benchmarks/benchmark_bwd_dim256.py benchmarks/benchmark_e2e_dim256.py
git diff --check
```

The final benchmarks record backward-only and public forward+backward
p20/median/p80 for B=1,H=2 and B=8,H=8 at S=128/512/2048/8192 with unmasked
and causal attention. The E2E benchmark reports both eager latency and CUDA
Graph steady-state GPU time. The historical Triton prototype is comparison data
only and is not part of the final source or runtime.
