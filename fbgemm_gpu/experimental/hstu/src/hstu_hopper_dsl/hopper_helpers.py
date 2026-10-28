# Hopper WGMMA issue helpers for HSTU CuTe DSL.
#
# v1: MMA atoms and smem layouts live in hstu_fwd.py and call
# cutlass.utils.hopper_helpers directly (same as gr_decode SM90 flash_fwd.py).
# This file only keeps the tile table and the WGMMA fence/commit/wait loop,
# which CUTLASS hopper_helpers does not provide.

from __future__ import annotations

from typing import Optional, Union

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute.nvgpu import warpgroup


WGMMA_ATOM_M = 64
THREADS_PER_WARPGROUP = 128
KV_STAGES = 2


def get_tile_size_fwd(
    head_dim: int,
    has_rab: bool = False,
    is_fp8: bool = False,
) -> tuple[int, int, int]:
    """Return (kBlockM, kBlockN, kNWarps), same table as C++ get_tile_size_fwd."""
    if is_fp8:
        if has_rab:
            return (128, 128, 12) if head_dim == 64 else (128, 64, 12)
        if head_dim == 64:
            return (128, 128, 12)
        if head_dim == 128:
            return (128, 128, 12)
        return (128, 64, 12)
    if has_rab:
        if head_dim == 32:
            return (192, 128, 16)
        if head_dim == 64:
            return (128, 128, 12)
        return (128, 64, 12)
    if head_dim <= 64:
        return (192, 128, 16)
    if head_dim == 128:
        return (128, 128, 12)
    return (128, 64, 12)


def get_num_mma_warp_groups(kBlockM: int) -> int:
    """MMA warpgroups = kBlockM / 64. CTA threads = (1 + this) * 128."""
    if kBlockM % WGMMA_ATOM_M != 0:
        raise ValueError(f"kBlockM={kBlockM} must be a multiple of {WGMMA_ATOM_M}")
    return kBlockM // WGMMA_ATOM_M


def partition_fragment_ABC(
    thr_mma: cute.ThrMma,
    shape_mnk: cute.Shape,
    sA: Optional[cute.Tensor],
    sB: Optional[cute.Tensor],
):
    is_rs = thr_mma.op.a_src == warpgroup.OperandSource.RMEM
    acc = cute.make_rmem_tensor(thr_mma.partition_shape_C(shape_mnk[:2]), Float32)
    if const_expr(not is_rs):
        assert sA is not None
        tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
    else:
        tCrA = thr_mma.make_fragment_A(
            thr_mma.partition_shape_A((shape_mnk[0], shape_mnk[2]))
        )
    assert sB is not None
    tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB))
    return acc, tCrA, tCrB


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: bool | Boolean = False,
    wg_wait: cutlass.Constexpr[int] = 0,
) -> None:
    """Issue one WGMMA along K. Mirrors C++ flash::gemm (hstu_hopper/utils.h)."""
    warpgroup.fence()
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
        mma_atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()
    if const_expr(wg_wait >= 0):
        warpgroup.wait_group(wg_wait)


def gemm_zero_init(
    tiled_mma: cute.TiledMma,
    shape: cute.Shape,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
) -> cute.Tensor:
    acc = cute.make_rmem_tensor(tiled_mma.partition_shape_C(shape), Float32)
    rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    gemm(tiled_mma, acc, rA, rB, zero_init=True, wg_wait=wg_wait)
    return acc


def gemm_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: Union[bool, Boolean] = False,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
) -> None:
    rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    gemm(tiled_mma, acc, rA, rB, zero_init=zero_init, wg_wait=wg_wait)
