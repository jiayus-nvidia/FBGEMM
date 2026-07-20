# HSTU Blackwell backward head-dim 256 draft

## Task contract

- Objective: add a native CuTe DSL backward path for HSTU Blackwell when
  `q/k/v/dO` use BF16 or FP16 with `head_dim == 256`.
- Public interface: keep `hstu_varlen_bwd_100` inputs and return values unchanged.
- Baseline: `jiayus-nvidia/FBGEMM` `dev` at
  `2c31d908acfbe0210500876be7c0406e64033dd3`.
- Hardware available for development: NVIDIA B300 SXM6 AC, compute capability
  10.3, driver 595.58.03.
- Existing behavior that must not regress: head dimensions 64 and 128, original
  and compact Q/K/V/dO layouts, workspace zeroing, and all existing Blackwell
  masks.
- Initial D=256 promotion scope: equal Q/K/V head dimensions, equal Q and KV
  head counts, BF16 and FP16, packed variable-length batches, no RAB/dRAB,
  non-deterministic backward. Unmasked, causal, local, and target-group masks
  are native correctness gates; arbitrary masks must retain explicit fallback
  routing until native predicates are validated.

## Frozen baseline and blocker

- Forward already accepts `(64, 128, 256)` and uses `q_stage = 1` at D=256.
- Backward currently accepts only `(64, 128)`.
- The existing one-kernel/one-CTA TMEM map is:
  `dK [0,D)`, `dV [D,2D)`, `dQ/dP [2D,2D+max(128,D))`, and
  `S [2D+max(128,D), ...)`. At D=256 the first S column is 768 and the
  layout cannot fit the 512-column TMEM allocation. Merely relaxing the host
  assertion is therefore invalid.
- FA4's merged D=256 implementation uses a dedicated path: one 2-CTA kernel for
  dQ and one 2-CTA kernel for dK/dV. This is the primary reference design.

## Candidate directions

1. Preferred: add a dedicated HSTU D=256 wrapper and two 2-CTA kernels, using
   FA4's D=256 scheduling/TMEM/data-movement structure while retaining HSTU's
   `P = silu(alpha * S)` and `dS = alpha * dP * silu'(alpha * S)` math and HSTU
   mask semantics.
2. Fallback: split dQ, dK, and dV into three one-CTA kernels so each persistent
   accumulator plus S/dP fits TMEM. Consider only if the 2-CTA port is blocked;
   it adds launches and recomputation and requires benchmark justification.
3. Rejected: enable D=256 in the existing fused kernel without changing its
   TMEM layout.

## Current branch milestone

- Native CuTe DSL DQ and DK/DV 2-CTA kernels now provide D=256 backward for
  unmasked, causal, local, and target-group attention.
- The decomposition and recomputation structure match FA4: dQ is produced
  first, then dK/dV, without materializing the score or activation matrix.
- The Triton prototype remains the short-sequence and arbitrary-mask fallback.
  Auto dispatch selects CuTe at the measured S>=2048 crossover.

## Native CuTe phase contract

- The requested deliverable is now an in-repository CuTe DSL implementation,
  not a Triton-only D=256 path.
- Reference source is FlashAttention PR 2412 head `c1e5e8c`, specifically its
  dedicated D=256 2-CTA dQ and dK/dV kernels.
- The Triton implementation is frozen as the numerical oracle and safe fallback
  until the CuTe path passes the same public-wrapper tests.
- The default D=256 dispatch may switch to CuTe only after BF16/FP16 causal and
  unmasked rows pass. Local and target now pass; arbitrary remains explicit
  Triton fallback.
- No per-token softmax-only state (`LSE` or `sum(O*dO)`) is allocated or
  consumed. A single broadcast dummy scalar temporarily satisfies inherited,
  value-dead transport pipelines while the kernel directly recomputes
  `P = silu(alpha*S)` and `dS = alpha*dP*silu'(alpha*S)`.

## Correctness gates

- Compare D=256 forward and gradients against the existing PyTorch HSTU
  reference for short boundary-heavy rows and representative long rows.
- Cover sequence lengths around tile boundaries: 1, 63, 64, 127, 128, 129,
  255, 256, and at least 2048.
- Cover BF16 and FP16; contiguous, packed-QKV, head-major, and non-contiguous
  dO layouts.
- Check finite outputs, untouched/out-of-range rows, repeated-call workspace
  clearing, and D=64/D=128 regression tests.
- Use the existing test policy: each HSTU max-absolute gradient error must be at
  most 5x the corresponding low-precision PyTorch error against the upcast
  reference. Add direct finite/relative checks where that denominator is zero.

## Validation and evaluation

- Syntax/static: `python -m compileall src/hstu_blackwell test`.
- Targeted wrapper/layout tests:
  `PYTHONPATH=src python -m pytest test/test_bwd_do_layout.py -v -s`.
- End-to-end correctness: a deterministic D=256 test matrix derived from
  `test/hstu_test.py`, followed by the existing HSTU16 test with SM100/SM103
  D=256 enabled.
- Performance rows after correctness: B=1/8, H=2/8, S=128/512/2048/8192,
  causal and unmasked, BF16. Use matched wrappers, warmup, interleaved A/B runs,
  and report distribution statistics. The comparison baseline is the best
  correct available D=256 implementation (PyTorch/Hopper fallback only as
  explicitly labeled; never compare asymmetric wrapper costs).

## Promotion criteria

- No unsupported D=256 call is silently dispatched to an invalid kernel.
- All scoped D=256 correctness rows pass on SM103 and D=64/D=128 regressions
  remain green.
- Kernel launch/build is stable across repeated fresh-process compilations.
- Performance results and any unsupported mask cases are recorded before the
  branch is declared complete.
