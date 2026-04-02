# HSTU Blackwell (sm100) — Development Makefile
#
# Usage:
#   make setup                - Install HSTU package (Blackwell-only)
#   make tt                   - Quick correctness test (single case)
#   make vt                   - Full hypothesis test suite
#   make bb                   - Performance benchmark (fwd + bwd)
#   make profile              - Single run for ncu profiling
#   make bm                   - ncu full profile (NVTX-filtered)
#   make bm-cli               - ncu register/spill/smem analysis
#   make clean                - Clear caches and build artifacts
#   make help                 - Show all targets

SHELL := /bin/bash
# Add src/ to PYTHONPATH so hstu_blackwell is importable without pip install
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)
PYTHON := python
PYTEST := python -m pytest
TEST_DIR := test
TEST_FILE := $(TEST_DIR)/hstu_test.py
PROFILE_FILE := $(TEST_DIR)/profile_blackwell.py

# HDIM: 64 or 128 (default: 128)
HDIM ?= 128
# BATCH: batch size for benchmark/profile
BATCH ?= 8
# HEADS: number of attention heads
HEADS ?= 4
# SEQLEN: sequence length for benchmark/profile
SEQLEN ?= 8192
# DTYPE: bf16 or fp16
DTYPE ?= bf16
# NWARMUP: warmup iterations for benchmark
NWARMUP ?= 5
# NITER: benchmark iterations
NITER ?= 20
# MAX_EXAMPLES: max hypothesis examples for vt
MAX_EXAMPLES ?= 100
# MODE: fwd, bwd, or both
MODE ?= both
# MASK: causal, local, or target
MASK ?= causal
# TARGET_LEN: target length (0 = no target)
TARGET_LEN ?= 0
# TARGET_GROUP_SIZE: target group size (must be >= 1, 0 causes compile-time div-by-zero)
TARGET_GROUP_SIZE ?= 1

.PHONY: setup tt vt bb profile bm bm-cli clean help

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup:
	@echo "=== Installing HSTU (Blackwell sm100 only) ==="
	HSTU_ARCH_LIST="10.0" pip install -e .

# ---------------------------------------------------------------------------
# Quick correctness test — single deterministic case
# ---------------------------------------------------------------------------
tt:
	@echo "=== Quick correctness test (Blackwell) ==="
	$(PYTHON) -u -c "\
from test.hstu_test import HSTU16Test; \
import torch; \
assert torch.cuda.get_device_capability()[0] >= 10, 'Requires Blackwell GPU (sm100)'; \
HSTU16Test().test_hstu_attn.hypothesis.inner_test( \
    HSTU16Test(), \
    batch_size=$(BATCH), heads=$(HEADS), max_context_len=0, \
    attn_hidden_dims=($(HDIM), $(HDIM)), alpha=1.0, \
    rab_params=(False, False, None), \
    seq_len_params=($(SEQLEN), $(SEQLEN)), \
    target_params=($(TARGET_LEN), (-1, 0), $(TARGET_GROUP_SIZE), False), \
    dtype=torch.bfloat16, full_batch=True); \
print('PASSED')"

# ---------------------------------------------------------------------------
# Full hypothesis test suite via pytest
# ---------------------------------------------------------------------------
vt:
	HSTU_MAX_EXAMPLES=$(MAX_EXAMPLES) \
	$(PYTEST) $(TEST_FILE)::HSTU16Test -v -x -s

# ---------------------------------------------------------------------------
# Profile script (generates profile_blackwell.py on the fly)
# ---------------------------------------------------------------------------
define PROFILE_SCRIPT
import sys
import time
import torch
try:
    from hstu.hstu_blackwell.hstu_ops_gpu import hstu_varlen_fwd_100, hstu_varlen_bwd_100
except ImportError:
    from hstu_blackwell.hstu_ops_gpu import hstu_varlen_fwd_100, hstu_varlen_bwd_100

mode = "$(MODE)"
batch_size = $(BATCH)
num_heads = $(HEADS)
head_dim = $(HDIM)
seqlen = $(SEQLEN)
dtype = torch.bfloat16 if "$(DTYPE)" == "bf16" else torch.float16
n_warmup = $(NWARMUP)
n_iter = $(NITER)
mask_type = "$(MASK)"
target_len = $(TARGET_LEN)
target_group_size = $(TARGET_GROUP_SIZE)

is_benchmark = len(sys.argv) > 1 and sys.argv[1] == "benchmark"
is_profile = len(sys.argv) > 1 and sys.argv[1] == "profile"

assert torch.cuda.get_device_capability()[0] >= 10, "Requires Blackwell GPU (sm100)"

# --- build inputs ---
cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, seqlen, dtype=torch.int32, device="cuda")
total_tokens = batch_size * seqlen

q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
k = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
v = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")

if mask_type == "causal":
    wl, wr = -1, 0
elif mask_type == "local":
    wl, wr = min(256, seqlen), 0
else:
    wl, wr = -1, 0

num_targets = None
if target_len > 0:
    num_targets = torch.full((batch_size,), target_len, dtype=torch.int32, device="cuda")

def run_fwd():
    with torch.cuda.nvtx.range("hstu_blackwell_fwd"):
        return hstu_varlen_fwd_100(
            q, k, v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seqlen, max_seqlen_k=seqlen,
            num_contexts=None, num_targets=num_targets,
            target_group_size=target_group_size,
            window_size_left=wl, window_size_right=wr,
            alpha=1.0, rab=None, func=None,
        )

def run_bwd(out):
    do = torch.randn_like(out)
    with torch.cuda.nvtx.range("hstu_blackwell_bwd"):
        return hstu_varlen_bwd_100(
            do, q, k, v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seqlen, max_seqlen_k=seqlen,
            dq=None, dk=None, dv=None,
            num_contexts=None, num_targets=num_targets,
            target_group_size=target_group_size,
            window_size_left=wl, window_size_right=wr,
            alpha=1.0, rab=None, has_drab=False, func=None,
            deterministic=False,
        )

print(f"Config: B={batch_size}, H={num_heads}, D={head_dim}, S={seqlen}, "
      f"dtype={dtype}, mask={mask_type}, target_len={target_len}")

# --- warmup (also triggers compilation) ---
print("Warming up (includes JIT compile)...")
for _ in range(n_warmup):
    out, _ = run_fwd()
    if mode in ("bwd", "both"):
        run_bwd(out)
torch.cuda.synchronize()

if is_profile:
    # single run for ncu capture
    print("Running single iteration for profiling...")
    torch.cuda.cudart().cudaProfilerStart()
    if mode in ("fwd", "both"):
        out, _ = run_fwd()
    if mode in ("bwd", "both"):
        run_bwd(out)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("Profile run done.")

elif is_benchmark:
    # fwd benchmark
    if mode in ("fwd", "both"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            out, _ = run_fwd()
        end.record()
        torch.cuda.synchronize()
        fwd_ms = start.elapsed_time(end) / n_iter
        flops_fwd = 2 * batch_size * num_heads * seqlen * seqlen * head_dim
        print(f"FWD: {fwd_ms:.3f} ms | {flops_fwd / fwd_ms / 1e9:.1f} TFLOPS")

    # bwd benchmark
    if mode in ("bwd", "both"):
        out, _ = run_fwd()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            run_bwd(out)
        end.record()
        torch.cuda.synchronize()
        bwd_ms = start.elapsed_time(end) / n_iter
        flops_bwd = 4 * batch_size * num_heads * seqlen * seqlen * head_dim
        print(f"BWD: {bwd_ms:.3f} ms | {flops_bwd / bwd_ms / 1e9:.1f} TFLOPS")

else:
    # just run once and verify output
    out, _ = run_fwd()
    print(f"FWD output shape: {out.shape}, max={out.abs().max().item():.4f}")
    if mode in ("bwd", "both"):
        dq, dk, dv, _ = run_bwd(out)
        print(f"BWD dq shape: {dq.shape}, max={dq.abs().max().item():.4f}")
    print("OK")
endef
export PROFILE_SCRIPT

_gen_profile_script:
	@echo "$$PROFILE_SCRIPT" > $(PROFILE_FILE)

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
bb: _gen_profile_script
	@echo "=== HSTU Blackwell Benchmark ==="
	$(PYTHON) -u $(PROFILE_FILE) benchmark

# ---------------------------------------------------------------------------
# Profile — single run for ncu capture
# ---------------------------------------------------------------------------
profile: _gen_profile_script
	@echo "=== HSTU Blackwell Profile Run ==="
	$(PYTHON) -u $(PROFILE_FILE) profile

# ---------------------------------------------------------------------------
# ncu full profile (NVTX-filtered)
# ---------------------------------------------------------------------------
NVTX_FWD := hstu_blackwell_fwd/hstu_varlen_fwd_kernel/
NVTX_BWD := hstu_blackwell_bwd/hstu_varlen_bwd_kernel/
NVTX_FILTER := --nvtx --nvtx-include "$(NVTX_FWD)" --nvtx-include "$(NVTX_BWD)"

bm: _gen_profile_script
	@mkdir -p profile
	ncu --set full $(NVTX_FILTER) \
		-f -o profile/hstu_blackwell.%p \
		$(PYTHON) -u $(PROFILE_FILE) profile

NCU_METRICS := launch__registers_per_thread,sm__cycles_elapsed.avg,sm__cycles_elapsed.max,sm__cycles_active.avg,sm__cycles_elapsed.avg.per_second,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__inst_executed.sum,sm__inst_executed_pipe_lsu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_xu.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum

# ---------------------------------------------------------------------------
# ncu register/spill/smem CLI analysis
# ---------------------------------------------------------------------------
bm-cli: _gen_profile_script
	@echo "=== ncu register/local-spill/smem analysis (HSTU Blackwell) ==="
	ncu $(NVTX_FILTER) \
		--metrics $(NCU_METRICS) \
		--target-processes all \
		$(PYTHON) -u $(PROFILE_FILE) profile; true
	@echo ""
	@echo "--- Legend ---"
	@echo "SOL:          sm__throughput = compute SOL%, gpu__compute_memory_throughput = memory SOL%"
	@echo "Duration:     gpu__time_duration (kernel wall time)"
	@echo "Cycles:       sm__cycles_elapsed (avg/max across SMs), sm__cycles_active (active only)"
	@echo "Occupancy:    sm__warps_active = achieved occupancy %"
	@echo "Pipe SOL:     pipe_fma / pipe_shared / pipe_tensor (UTCMMA) = per-pipe utilization %"
	@echo "Instructions: sm__inst_executed (total), pipe_lsu/fma/xu (breakdown)"
	@echo "Registers:    launch__registers_per_thread"
	@echo "Local spills: local_op_ld/st sectors = 0 → no spills"
	@echo "Smem banks:   bank_conflicts = 0 → no conflicts"

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf profile/
	rm -f $(PROFILE_FILE)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .hypothesis -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo "HSTU Blackwell (sm100) — Development Targets"
	@echo ""
	@echo "  make setup                          Install HSTU package (Blackwell-only)"
	@echo "  make tt   [BATCH= HEADS= HDIM= SEQLEN=]  Quick single-case correctness test"
	@echo "  make vt   [MAX_EXAMPLES=200]        Full hypothesis test suite"
	@echo "  make bb   [BATCH= HEADS= HDIM= SEQLEN= MODE=fwd|bwd|both]  Benchmark"
	@echo "  make profile                        Single run for ncu profiling"
	@echo "  make bm                             ncu full profile (NVTX-filtered, saves .ncu-rep)"
	@echo "  make bm-cli                         ncu register/spill/smem analysis"
	@echo "  make clean                          Clear build artifacts and caches"
	@echo ""
	@echo "Variables (override on command line):"
	@echo "  HDIM=64|128       Head dimension       (default: 128)"
	@echo "  BATCH=N           Batch size            (default: 32)"
	@echo "  HEADS=N           Number of heads       (default: 2)"
	@echo "  SEQLEN=N          Sequence length       (default: 1024)"
	@echo "  DTYPE=bf16|fp16   Data type             (default: bf16)"
	@echo "  MODE=fwd|bwd|both Forward/backward      (default: both)"
	@echo "  MASK=causal|local Mask type             (default: causal)"
	@echo "  TARGET_LEN=N      Target length         (default: 0)"
	@echo "  NWARMUP=N         Warmup iterations     (default: 10)"
	@echo "  NITER=N           Benchmark iterations  (default: 100)"
	@echo "  MAX_EXAMPLES=N    Hypothesis examples   (default: 200)"
