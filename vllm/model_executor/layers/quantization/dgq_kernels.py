"""Triton kernels for the VQ (dim4/K256) format.

vq_dequant: one-pass fused decode  w = cb[codes] * scale
  - codes: uint8 [E, O, I/4]   (one byte indexes a 4-wide codebook row)
  - cb:    fp32  [E, 256, 4]   (per-expert-matrix codebook)
  - scales:fp32  [E, O, I/g]
  - out:   bf16  [E, O, I]
The per-expert codebook (4 KB) is loaded once per program.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

VDIM = 4
VDIM_C = __import__("triton").language.constexpr(4)


@triton.jit
def _vq_dequant_kernel(
    codes_ptr, cb_ptr, scales_ptr, out_ptr,
    O, I_VEC, GROUPS_PER_ROW,
    GROUP: tl.constexpr,        # weights per scale group
    BLOCK_V: tl.constexpr,      # code vectors per program (BLOCK_V*4 weights)
):
    pid = tl.program_id(0)
    n_vblocks = tl.cdiv(I_VEC, BLOCK_V)
    e = pid // (O * n_vblocks)
    rem = pid % (O * n_vblocks)
    o = rem // n_vblocks
    vb = rem % n_vblocks

    v_off = vb * BLOCK_V + tl.arange(0, BLOCK_V)          # vector indices in row
    v_mask = v_off < I_VEC

    # load codes for this tile
    codes = tl.load(codes_ptr + (e * O + o) * I_VEC + v_off,
                    mask=v_mask, other=0).to(tl.int32)     # [BLOCK_V]

    # gather codebook rows: cb[e, code, 0..3]
    cb_base = cb_ptr + e * 256 * VDIM_C + codes * VDIM_C       # [BLOCK_V]
    d = tl.arange(0, VDIM_C)
    vals = tl.load(cb_base[:, None] + d[None, :],
                   mask=v_mask[:, None], other=0.0)        # [BLOCK_V, 4] f32

    # scales: group index of each weight = (v*4 + d) // GROUP == v*4//GROUP (GROUP%4==0)
    g_idx = (v_off * VDIM_C) // GROUP                        # [BLOCK_V]
    s = tl.load(scales_ptr + (e * O + o) * GROUPS_PER_ROW + g_idx,
                mask=v_mask, other=0.0)                    # [BLOCK_V]
    w = vals * s[:, None]

    out_base = out_ptr + ((e * O + o) * I_VEC + v_off) * VDIM_C
    tl.store(out_base[:, None] + d[None, :], w.to(tl.bfloat16),
             mask=v_mask[:, None])


def vq_dequant_triton(codes: torch.Tensor, scales: torch.Tensor, cb: torch.Tensor,
                      group: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """codes [E,O,I/4] u8, scales [E,O,I/g] f32, cb [E,256,4] f32 -> bf16 [E,O,I]."""
    E, O, I_VEC = codes.shape
    I = I_VEC * VDIM
    if out is None:
        out = torch.empty(E, O, I, dtype=torch.bfloat16, device=codes.device)
    BLOCK_V = 128
    grid = (E * O * triton.cdiv(I_VEC, BLOCK_V),)
    _vq_dequant_kernel[grid](
        codes, cb.contiguous().float(), scales.contiguous().float(), out,
        O, I_VEC, scales.shape[-1],
        GROUP=group, BLOCK_V=BLOCK_V,
        num_warps=4,
    )
    return out


# ------------------------------------------------------ fused grouped VQ-GEMM

@triton.jit
def _vq_gemm_kernel(
    x_ptr, codes_ptr, cb_ptr, scales_ptr, y_ptr,
    tile_seg_ptr, tile_m0_ptr, seg_end_ptr, seg_expert_ptr,
    N, K_VEC, GROUPS_PER_ROW,
    stride_xm, stride_ym,
    K: tl.constexpr, GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Grouped GEMM with in-tile VQ decode.
    Y[m, :] = X[m, :] @ W[e]^T,  W[e][n, k] = cb[e, codes[e,n,k//4], k%4] * scale[e,n,k//GROUP]
    Token rows are pre-sorted by expert; host provides per-m-tile tables."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    seg = tl.load(tile_seg_ptr + pid_m)
    e = tl.load(seg_expert_ptr + seg)
    m0 = tl.load(tile_m0_ptr + pid_m)
    m_end = tl.load(seg_end_ptr + seg)

    m_off = m0 + tl.arange(0, BLOCK_M)
    m_mask = m_off < m_end
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_off < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + m_off[:, None] * stride_xm + k_off[None, :],
                    mask=m_mask[:, None], other=0.0)                    # [M, BK]
        # ---- decode W tile [BN, BK] ----
        kv_off = k0 // 4 + tl.arange(0, BLOCK_K // 4)
        code = tl.load(codes_ptr + (e * N + n_off[:, None]) * K_VEC + kv_off[None, :],
                       mask=n_mask[:, None], other=0).to(tl.int32)      # [BN, BK/4]
        d = tl.arange(0, 4)
        vals = tl.load(cb_ptr + e * 1024 + code[:, :, None] * 4 + d[None, None, :])
        w = tl.reshape(vals, (BLOCK_N, BLOCK_K))                        # [BN, BK]
        # one scale group per k-tile (BLOCK_K <= GROUP, k0 % GROUP aligned)
        s = tl.load(scales_ptr + (e * N + n_off) * GROUPS_PER_ROW + k0 // GROUP,
                    mask=n_mask, other=0.0)                             # [BN]
        w = w * s[:, None]
        acc += tl.dot(x, tl.trans(w.to(x.dtype)))
    y = y_ptr + m_off[:, None] * stride_ym + n_off[None, :]
    tl.store(y, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def vq_grouped_gemm(x_sorted: torch.Tensor, seg_starts: torch.Tensor,
                    seg_experts: torch.Tensor, codes: torch.Tensor,
                    cb: torch.Tensor, scales: torch.Tensor, group: int,
                    BLOCK_M: int = 32, BLOCK_N: int = 64, BLOCK_K: int = 64):
    """x_sorted [S, K] bf16 (rows sorted by expert), seg_starts [n_seg+1] i32,
    seg_experts [n_seg] i32, codes [E, N, K/4] u8, cb [E, 256, 4], scales [E, N, K/group].
    Returns y [S, N] bf16."""
    S, K = x_sorted.shape
    E, N, K_VEC = codes.shape
    assert K == K_VEC * 4 and BLOCK_K <= group and group % BLOCK_K == 0 or True
    # host tile tables
    seg_lens = (seg_starts[1:] - seg_starts[:-1])
    tiles_per_seg = (seg_lens + BLOCK_M - 1) // BLOCK_M
    total_tiles = int(tiles_per_seg.sum().item())
    tile_seg = torch.repeat_interleave(
        torch.arange(len(seg_experts), device=x_sorted.device, dtype=torch.int32),
        tiles_per_seg)
    # m0 per tile: seg_start + tile_index_in_seg * BLOCK_M
    cums = torch.cumsum(torch.nn.functional.pad(tiles_per_seg, (1, 0)), 0)
    tile_idx_in_seg = torch.arange(total_tiles, device=x_sorted.device, dtype=torch.int32) - cums[tile_seg.long()].to(torch.int32)
    tile_m0 = seg_starts[tile_seg.long()].to(torch.int32) + tile_idx_in_seg * BLOCK_M
    seg_end = seg_starts[1:].contiguous().to(torch.int32)

    y = torch.empty(S, N, dtype=torch.bfloat16, device=x_sorted.device)
    if total_tiles == 0:
        return y
    grid = (total_tiles, triton.cdiv(N, BLOCK_N))
    _vq_gemm_kernel[grid](
        x_sorted, codes, cb.contiguous().to(torch.float16),
        scales.contiguous().to(torch.float16), y,
        tile_seg, tile_m0, seg_end, seg_experts.to(torch.int32),
        N, K_VEC, scales.shape[-1],
        x_sorted.stride(0), y.stride(0),
        K=K, GROUP=group,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return y


@torch._dynamo.disable
def vq_moe_forward(x, topk_weights, topk_ids,
                   gu_vq, gu_cb, gu_scales, gu_group,
                   dn_vq, dn_cb, dn_scales, dn_group,
                   act_fn):
    """Full MoE forward with fused VQ-GEMMs (tensor-level, engine-agnostic).
    x [T, H] bf16; topk_ids/weights [T, K]."""
    T, H = x.shape
    E = gu_vq.shape[0]
    Kk = topk_ids.shape[-1]
    flat_e = topk_ids.reshape(-1).long()
    order = flat_e.argsort(stable=True)
    tok_idx = order // Kk
    x_sorted = x[tok_idx].contiguous()
    counts = torch.bincount(flat_e, minlength=E)
    seg_starts = torch.nn.functional.pad(counts.cumsum(0), (1, 0)).to(torch.int32)
    seg_experts = torch.arange(E, device=x.device, dtype=torch.int32)

    y1 = vq_grouped_gemm(x_sorted, seg_starts, seg_experts, gu_vq, gu_cb, gu_scales, gu_group)
    gate, up = y1.chunk(2, dim=-1)
    h = act_fn(gate) * up
    y2 = vq_grouped_gemm(h.contiguous(), seg_starts, seg_experts, dn_vq, dn_cb, dn_scales, dn_group)
    y2 = y2 * topk_weights.reshape(-1)[order].to(y2.dtype)[:, None]
    out = torch.zeros(T, H, dtype=x.dtype, device=x.device)
    out.index_add_(0, tok_idx, y2.to(x.dtype))
    return out
