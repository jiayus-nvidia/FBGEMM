# HSTU Blackwell backward head-dim 256 design record

## Task contract

- Add native CuTe DSL backward for BF16/FP16 `head_dim == 256` while preserving
  the `hstu_varlen_bwd_100` interface.
- Support all existing Blackwell mask and packed-layout cases without a D=256
  Triton runtime implementation.
- Preserve D=64/128 behavior, caller-provided output buffers, current-stream
  execution, and CUDA Graph capture.

## Why D=256 needs a dedicated path

The existing fused one-CTA kernel stores dK, dV, dQ/dP, and S in one TMEM map.
At D=256 the first S column would be 768, beyond the 512-column allocation, so
relaxing the old host assertion is invalid. FA4 solves the same constraint with
one 2-CTA kernel for dQ and a second 2-CTA kernel for dK/dV. This branch follows
that decomposition while replacing softmax with HSTU SiLU math and masks.

## Final design

- dQ tile: `(128,128,256)`, two CTAs.
- dK/dV tile: `(128,64,256)`, two CTAs.
- Scores and HSTU activation derivatives are recomputed in each kernel; no
  score matrix, LSE, or `sum(O*dO)` buffer is written.
- Packed variable-length launch grids use the maximum per-sequence length rather
  than total packed tokens.
- Unmasked, causal, local, target-group, delta-Q, and arbitrary interval
  predicates are native.
- The public D=256 dispatch has no length selector or environment override.

## Short-sequence investigation

The frozen B300 profile showed a fixed floor of roughly 44.3 us for dQ plus
41.9 us for dK/dV at B=1,H=2,S=128. The earlier lightweight prototype needed
roughly 6.2 us plus 9.2 us, so ordinary loop-bound tuning could not erase the
whole gap.

Retained changes:

- Skip known-true mask predicates for aligned full-tile unmasked batches.
- Replace the extra cluster-wide completion barrier with a CTA-local barrier;
  the allocator still coordinates the two CTAs during TMEM release.
- Restrict residual predicates to actual partial Q/K tiles.

Rejected and reverted changes:

- Compile-time S=128/512 specialization: no measurable reduction in the kernel
  floor.
- One-CTA dQ: invalid launch configuration.
- 64-row dQ: approximately 22% faster but incorrect dQ; split-D repair produced
  NaNs.
- Tighter causal trip bounds and relaxed cluster initialization: regressions or
  unstable results.

The retained changes improve the conservative same-timer short-row geometric
mean by 3.6% and long rows by 2.6%. The remaining small-batch gap requires a
separate lightweight CuTe design if it becomes a product priority.

## Promotion gates

- PyTorch-reference forward outputs and gradients for BF16/FP16 and every
  supported mask.
- Boundary lengths around both 64- and 128-row tiles, packed varlen, delta-Q,
  non-contiguous dO, and strided output buffers.
- Public autograd dispatch, backward-only and full-E2E CUDA Graph capture plus
  repeated replay, and D=64/128 layout/TVM-FFI regressions.
- Dependency-free p20/median/p80 benchmark at B=1/H=2 and B=8/H=8 for
  S=128/512/2048/8192, causal and unmasked.
- Static checks and a source scan proving no D=256 Triton import, experiment
  switch, or stale fallback branch remains.
