# Native CuTe D=256 backward plan

## Frozen inputs

- FBGEMM baseline: `2c31d908acfbe0210500876be7c0406e64033dd3`
- FA4 reference: PR 2412 head `c1e5e8c`
- Correctness oracle: `hstu_bwd_256.py` through the existing
  `hstu_varlen_bwd_100` interface
- Hardware/toolchain: B300 SM103, PyTorch 2.12.1+cu130, CUDA 13.0,
  nvidia-cutlass-dsl 4.4.0

## Source mapping

| HSTU responsibility | FA4 D=256 source |
|---|---|
| dQ score/dP recomputation and 2-CTA scheduling | `sm100_hd256_2cta_fmha_backward_dqkernel.py` |
| dK/dV recomputation and 2-CTA scheduling | `sm100_hd256_2cta_fmha_backward_dkdvkernel.py` |
| HSTU activation derivative | existing `hstu_bwd.py::compute_mask_step` / `FastSilU` |
| HSTU mask semantics | existing `mask.py`, `block_info.py`, and Triton oracle |
| Public layouts and compile cache | `hstu_ops_gpu.py` |

## Build order

1. [x] Bring the two dedicated FA4 kernels into the HSTU package with local-only
   imports and a non-persistent static scheduler.
2. [x] Add an HSTU CuTe wrapper using the same `(T,H,D)` tensors, cu-seqlens,
   caller-provided gradient buffers, alpha, and current stream as the public
   wrapper.
3. [x] Replace the softmax/LSE transformation in dQ with HSTU SiLU and dSiLU.
4. [x] Replace the dK/dV softmax/LSE transformation with the same HSTU math.
5. [x] Bring up BF16/FP16 unmasked, causal, local, target, delta-Q, and packed
   variable-length batches. Route arbitrary interval masks explicitly to Triton.
6. [x] Enable evidence-based hybrid dispatch, run the D=256 and D=64/128
   regression suites, CUDA Graph capture, and matched performance measurements.

## Failure boundaries

- A compile failure is recorded with the failing kernel and CuTe diagnostic;
  it does not justify weakening tests or silently routing all calls back to
  Triton.
- Arbitrary interval masks remain a tested, explicit Triton fallback. Local and
  target-group masks are native.
- The existing D=64/128 class and compile keys must remain unchanged.

## Validation

```bash
TRITON_CACHE_DIR=/tmp/hstu-bwd-d256-triton-cache PYTHONPATH=src \
  python -m pytest test/test_bwd_dim256.py -v -s
PYTHONPATH=src python -m pytest test/test_bwd_do_layout.py -v -s
```

Direct bring-up tests compare native CuTe outputs with the frozen Triton path.
The final targeted results are 25 D=256 tests and 18 layout/TVM-FFI regression
tests; forced-native CUDA Graph capture also passes.

## Evaluation

Use preallocated outputs and matching public wrappers for CuTe/Triton A/B
timing. Record p20/median/p80 for B=1,H=2 and B=8,H=8 at S=128,512,2048,8192.
Auto mode keeps short rows on Triton and selects CuTe at S>=2048, where it is at
parity or faster in the measured rows. `FBGEMM_HSTU_D256_CUTE` provides forced
CuTe/off modes for testing and retuning.
