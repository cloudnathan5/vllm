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


# -------------------------------------- capture-safe fused VQ path (no host syncs)

@triton.jit
def _vq_gemm_cs_kernel(
    x_ptr, codes_ptr, cb_ptr, scales_ptr, y_ptr,
    tile_seg_ptr, tile_m0_ptr, seg_end_ptr, n_tiles_ptr,
    N, K_VEC, GROUPS_PER_ROW,
    stride_xm, stride_ym,
    K: tl.constexpr, GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Same math as _vq_gemm_kernel, launched on a static worst-case grid.
    Tiles past the live count (device scalar) exit immediately, so the whole
    forward is CUDA-graph capturable. Segment id == expert id."""
    pid_m = tl.program_id(0)
    if pid_m >= tl.load(n_tiles_ptr):
        return
    pid_n = tl.program_id(1)
    e = tl.load(tile_seg_ptr + pid_m)
    m0 = tl.load(tile_m0_ptr + pid_m)
    m_end = tl.load(seg_end_ptr + e)

    m_off = m0 + tl.arange(0, BLOCK_M)
    m_mask = m_off < m_end
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_off < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + m_off[:, None] * stride_xm + k_off[None, :],
                    mask=m_mask[:, None], other=0.0)
        kv_off = k0 // 4 + tl.arange(0, BLOCK_K // 4)
        code = tl.load(codes_ptr + (e * N + n_off[:, None]) * K_VEC + kv_off[None, :],
                       mask=n_mask[:, None], other=0).to(tl.int32)
        d = tl.arange(0, 4)
        vals = tl.load(cb_ptr + e * 1024 + code[:, :, None] * 4 + d[None, None, :])
        w = tl.reshape(vals, (BLOCK_N, BLOCK_K))
        s = tl.load(scales_ptr + (e * N + n_off) * GROUPS_PER_ROW + k0 // GROUP,
                    mask=n_mask, other=0.0)
        w = w * s[:, None]
        acc += tl.dot(x, tl.trans(w.to(x.dtype)))
    y = y_ptr + m_off[:, None] * stride_ym + n_off[None, :]
    tl.store(y, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def build_vq_tile_tables(seg_starts: torch.Tensor, S: int, E: int, BLOCK_M: int = 32):
    """Device-built m-tile tables with static shapes (graph-capturable).
    seg_starts [E+1] i32. Worst case sum(ceil(len_e/BM)) <= ceil(S/BM) + E.
    Returns (tile_seg [MAX] i32, tile_m0 [MAX] i32, n_tiles [1] i32, MAX)."""
    seg_lens = (seg_starts[1:] - seg_starts[:-1]).to(torch.int64)
    tiles_per_seg = (seg_lens + (BLOCK_M - 1)) // BLOCK_M
    tile_cums = tiles_per_seg.cumsum(0)
    n_tiles = tile_cums[-1:].to(torch.int32)
    MAX_TILES = (S + BLOCK_M - 1) // BLOCK_M + E
    t = torch.arange(MAX_TILES, device=seg_starts.device)
    seg = torch.searchsorted(tile_cums, t, right=True).clamp_(max=E - 1)
    prev = tile_cums[seg] - tiles_per_seg[seg]
    tile_m0 = (seg_starts[seg].to(torch.int64) + (t - prev) * BLOCK_M).to(torch.int32)
    return seg.to(torch.int32), tile_m0, n_tiles, MAX_TILES


def vq_grouped_gemm_cs(x_sorted, tables, seg_end, codes, cb, scales, group,
                       BLOCK_M: int = 32, BLOCK_N: int = 64, BLOCK_K: int = 64):
    """Capture-safe grouped VQ-GEMM. cb/scales should be contiguous fp16
    (pre-cast once at setup; the .to() below is a no-op then)."""
    tile_seg, tile_m0, n_tiles, MAX_TILES = tables
    S, K = x_sorted.shape
    E, N, K_VEC = codes.shape
    y = torch.empty(S, N, dtype=torch.bfloat16, device=x_sorted.device)
    grid = (MAX_TILES, triton.cdiv(N, BLOCK_N))
    _vq_gemm_cs_kernel[grid](
        x_sorted, codes, cb.contiguous().to(torch.float16),
        scales.contiguous().to(torch.float16), y,
        tile_seg, tile_m0, seg_end, n_tiles,
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
    x [T, H] bf16; topk_ids/weights [T, K]. No host syncs: bincount/repeat_
    interleave replaced with scatter_add/searchsorted so CUDA graphs capture it."""
    T, H = x.shape
    E = gu_vq.shape[0]
    Kk = topk_ids.shape[-1]
    flat_e = topk_ids.reshape(-1).long()
    order = flat_e.argsort(stable=True)
    tok_idx = order // Kk
    x_sorted = x[tok_idx].contiguous()
    counts = torch.zeros(E, device=x.device, dtype=torch.long)
    counts.scatter_add_(0, flat_e, torch.ones_like(flat_e))
    seg_starts = torch.nn.functional.pad(counts.cumsum(0), (1, 0)).to(torch.int32)
    seg_end = seg_starts[1:].contiguous()
    tables = build_vq_tile_tables(seg_starts, T * Kk, E)

    y1 = vq_grouped_gemm_cs(x_sorted, tables, seg_end, gu_vq, gu_cb, gu_scales, gu_group)
    gate, up = y1.chunk(2, dim=-1)
    h = act_fn(gate) * up
    y2 = vq_grouped_gemm_cs(h.contiguous(), tables, seg_end, dn_vq, dn_cb, dn_scales, dn_group)
    y2 = y2 * topk_weights.reshape(-1)[order].to(y2.dtype)[:, None]
    out = torch.zeros(T, H, dtype=x.dtype, device=x.device)
    out.index_add_(0, tok_idx, y2.to(x.dtype))
    return out


# torch custom op wrapper: lets vLLM's torch.compile + piecewise CUDA graphs
# treat the fused MoE as a single opaque node instead of graph-breaking into
# eager python every forward (only sound because the path has no host syncs).
try:
    from torch.library import custom_op as _custom_op, register_fake as _register_fake

    @_custom_op("dgq::vq_moe", mutates_args=())
    def _vq_moe_op(x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                   gu_vq: torch.Tensor, gu_cb: torch.Tensor, gu_scales: torch.Tensor,
                   gu_group: int,
                   dn_vq: torch.Tensor, dn_cb: torch.Tensor, dn_scales: torch.Tensor,
                   dn_group: int) -> torch.Tensor:
        return vq_moe_forward(
            x, topk_weights, topk_ids,
            gu_vq, gu_cb, gu_scales, gu_group,
            dn_vq, dn_cb, dn_scales, dn_group,
            lambda t: torch.nn.functional.gelu(t, approximate="tanh"))

    @_register_fake("dgq::vq_moe")
    def _vq_moe_fake(x, topk_weights, topk_ids,
                     gu_vq, gu_cb, gu_scales, gu_group,
                     dn_vq, dn_cb, dn_scales, dn_group):
        return torch.empty_like(x)
except (ImportError, AttributeError):  # very old torch: transformers path only
    pass


# ---------------------------------------------------------- MXFP4 (v3) path

_E2M1_VALS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _e2m1_encode(t: torch.Tensor) -> torch.Tensor:
    """bf16/f32 values already ON the e2m1 grid -> 4-bit codes (sign|idx)."""
    vals = torch.tensor(_E2M1_VALS, device=t.device)
    idx = (t.abs().unsqueeze(-1) - vals).abs().argmin(-1).to(torch.uint8)
    return idx | (t < 0).to(torch.uint8) * 8


def pack_codebook_fp4(cb: torch.Tensor) -> torch.Tensor:
    """cb [E,256,4] on-grid -> packed nibbles [E,256,2] u8 (lo nibble = even elem)."""
    n = _e2m1_encode(cb)                                   # [E,256,4]
    return (n[..., 0::2] | (n[..., 1::2] << 4)).contiguous()


def quantize_act_mx(x: torch.Tensor):
    """x [S,K] bf16 -> (packed e2m1 [S,K/2] u8, e8m0 scales [S,K/32] u8).
    Per-block two-candidate scale (fit vs half-with-clip), min-MSE."""
    S, K = x.shape
    xb = x.float().view(S, K // 32, 32)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    vals = torch.tensor(_E2M1_VALS, device=x.device)
    def round_e2m1_idx(xn):
        # arithmetic nearest-e2m1 for |xn| in [0, 6]: grid 0,.5,1,1.5,2,3,4,6
        a = xn.abs()
        # region boundaries (midpoints): .25,.75,1.25,1.75,2.5,3.5,5
        idx = (a >= 0.25).to(torch.uint8) + (a >= 0.75).to(torch.uint8) \
            + (a >= 1.25).to(torch.uint8) + (a >= 1.75).to(torch.uint8) \
            + (a >= 2.5).to(torch.uint8) + (a >= 3.5).to(torch.uint8) \
            + (a >= 5.0).to(torch.uint8)
        return idx

    e0 = torch.ceil(torch.log2(amax / 6.0))
    best_err, best_e, best_idx, best_sign = None, None, None, None
    for de in (0.0, -1.0):
        e = e0 + de
        xn = (xb / torch.exp2(e)).clamp(-6.0, 6.0)
        idx = round_e2m1_idx(xn)
        deq = vals[idx.long()] * torch.sign(xn)
        err = (deq * torch.exp2(e) - xb).pow(2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_e, best_idx, best_sign = err, e, idx, (xn < 0)
        else:
            better = err < best_err
            best_e = torch.where(better, e, best_e)
            best_idx = torch.where(better, idx, best_idx)
            best_sign = torch.where(better, xn < 0, best_sign)
            best_err = torch.minimum(err, best_err)
    n = (best_idx.to(torch.uint8) | best_sign.to(torch.uint8) * 8).reshape(S, K)
    packed = (n[:, 0::2] | (n[:, 1::2] << 4)).contiguous()
    return packed, (best_e.squeeze(-1) + 127).to(torch.uint8).contiguous()


# dynamic: continuous batching varies the row count nearly every forward;
# a static compile re-specializes per shape (recompile storm at c>1)
quantize_act_mx_c = torch.compile(quantize_act_mx, dynamic=True)


@triton.jit
def _act_mx_kernel(x_ptr, xq_ptr, xs_ptr, S, K: tl.constexpr, BLOCK_G: tl.constexpr):
    """One-kernel port of quantize_act_mx: per-32 two-candidate e8m0 scale
    (fit vs half-with-clip, min group MSE) + arithmetic e2m1 rounding + pack."""
    NCH: tl.constexpr = (K // 32 + BLOCK_G - 1) // BLOCK_G
    pid = tl.program_id(0)
    row = pid // NCH
    gch = pid % NCH
    if row >= S:
        return
    g = tl.arange(0, BLOCK_G)
    o = tl.arange(0, 32)
    gmask = gch * BLOCK_G + g < K // 32
    k0 = gch * (32 * BLOCK_G)
    x = tl.load(x_ptr + row * K + k0 + g[:, None] * 32 + o[None, :],
                mask=gmask[:, None], other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)
    l2 = tl.log2(amax / 6.0)
    fl = tl.floor(l2)
    e0 = tl.where(l2 > fl, fl + 1.0, fl)  # ceil

    best_err = tl.full((BLOCK_G,), float("inf"), tl.float32)
    best_e = tl.zeros((BLOCK_G,), tl.float32)
    best_idx = tl.zeros((BLOCK_G, 32), tl.int32)
    best_neg = tl.zeros((BLOCK_G, 32), tl.int32)
    for de in tl.static_range(2):
        e = e0 - de
        sc = tl.exp2(e)
        xn = tl.minimum(tl.maximum(x / sc[:, None], -6.0), 6.0)
        a = tl.abs(xn)
        idx = ((a >= 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32)
               + (a >= 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
               + (a >= 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32)
               + (a >= 5.0).to(tl.int32))
        deq = tl.where(idx == 7, 6.0, tl.where(idx == 6, 4.0,
              tl.where(idx == 5, 3.0, idx.to(tl.float32) * 0.5)))
        sgn = tl.where(xn < 0, -1.0, 1.0)
        err = tl.sum((deq * sgn * sc[:, None] - x) * (deq * sgn * sc[:, None] - x), axis=1)
        better = err < best_err
        best_err = tl.where(better, err, best_err)
        best_e = tl.where(better, e, best_e)
        best_idx = tl.where(better[:, None], idx, best_idx)
        best_neg = tl.where(better[:, None], (xn < 0).to(tl.int32), best_neg)

    nib = best_idx | (best_neg << 3)
    r = tl.reshape(nib, (BLOCK_G, 16, 2))
    lo, hi = tl.split(r)
    packed = (lo | (hi << 4)).to(tl.uint8)
    ev = tl.arange(0, 16)
    tl.store(xq_ptr + row * (K // 2) + k0 // 2 + g[:, None] * 16 + ev[None, :], packed,
             mask=gmask[:, None])
    tl.store(xs_ptr + row * (K // 32) + gch * BLOCK_G + g,
             (best_e + 127.0).to(tl.uint8), mask=gmask)


def quantize_act_mx_triton(x: torch.Tensor):
    """Triton act quant; bit-compatible with quantize_act_mx."""
    S, K = x.shape
    xq = torch.empty(S, K // 2, dtype=torch.uint8, device=x.device)
    xs = torch.empty(S, K // 32, dtype=torch.uint8, device=x.device)
    BLOCK_G = 8
    grid = (S * triton.cdiv(K // 32, BLOCK_G),)
    _act_mx_kernel[grid](x, xq, xs, S, K=K, BLOCK_G=BLOCK_G, num_warps=4)
    return xq, xs


@triton.jit
def _mx_gemm_kernel(
    xq_ptr, xs_ptr, codes_ptr, cbp_ptr, wexp_ptr, row_ptr, y_ptr,
    tile_seg_ptr, tile_m0_ptr, seg_end_ptr, seg_expert_ptr,
    N, K_VEC, K_G32,
    K: tl.constexpr, GROUP: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    seg = tl.load(tile_seg_ptr + pid_m)
    e = tl.load(seg_expert_ptr + seg)
    m0 = tl.load(tile_m0_ptr + pid_m)
    m_end = tl.load(seg_end_ptr + seg)
    m_off = m0 + tl.arange(0, BM)
    m_mask = m_off < m_end
    n_off = pid_n * BN + tl.arange(0, BN)
    n_mask = n_off < N

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        # activations: packed fp4 [BM, BK/2] + e8m0 [BM, BK/32]
        ka = k0 // 2 + tl.arange(0, BK // 2)
        xq = tl.load(xq_ptr + m_off[:, None] * (K // 2) + ka[None, :],
                     mask=m_mask[:, None], other=0)
        ks = k0 // 32 + tl.arange(0, BK // 32)
        xs = tl.load(xs_ptr + m_off[:, None] * K_G32 + ks[None, :],
                     mask=m_mask[:, None], other=127)
        # weights: gather pre-packed codeword nibble-pairs -> [BN, BK/2]
        kv = k0 // 4 + tl.arange(0, BK // 4)
        code = tl.load(codes_ptr + (e * N + n_off[:, None]) * K_VEC + kv[None, :],
                       mask=n_mask[:, None], other=0).to(tl.int32)
        d2 = tl.arange(0, 2)
        wq = tl.load(cbp_ptr + e * 512 + code[:, :, None] * 2 + d2[None, None, :])
        wq = tl.reshape(wq, (BN, BK // 2))
        # weight e8m0: per-GROUP exp broadcast to per-32 (GROUP % 32 == 0)
        we = tl.load(wexp_ptr + (e * N + n_off[:, None]) * (K // GROUP) + (k0 // GROUP),
                     mask=n_mask[:, None], other=127)
        ws = tl.broadcast_to(we, (BN, BK // 32))
        acc = tl.dot_scaled(xq, xs, "e2m1", tl.trans(wq), ws, "e2m1", acc)
    row = tl.load(row_ptr + e * N + n_off, mask=n_mask, other=0.0)
    y = y_ptr + m_off[:, None] * N + n_off[None, :]
    tl.store(y, (acc * row[None, :]).to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


def mx_grouped_gemm(xq, xs, seg_starts, seg_experts, codes, cb_packed,
                    wexp_biased, row, group, BLOCK_M=32, BLOCK_N=64, BLOCK_K=64):
    """Fused MXFP4 grouped GEMM. xq [S,K/2] u8, xs [S,K/32] u8 (e8m0 biased),
    codes [E,N,K/4] u8, cb_packed [E,256,2] u8, wexp_biased [E,N,K/group] u8,
    row [E,N] f32 -> y [S,N] bf16."""
    S = xq.shape[0]
    E, N, K_VEC = codes.shape
    K = K_VEC * 4
    seg_lens = (seg_starts[1:] - seg_starts[:-1])
    tiles_per_seg = (seg_lens + BLOCK_M - 1) // BLOCK_M
    total_tiles = int(tiles_per_seg.sum().item())
    y = torch.empty(S, N, dtype=torch.bfloat16, device=xq.device)
    if total_tiles == 0:
        return y
    tile_seg = torch.repeat_interleave(
        torch.arange(len(seg_experts), device=xq.device, dtype=torch.int32), tiles_per_seg)
    cums = torch.cumsum(torch.nn.functional.pad(tiles_per_seg, (1, 0)), 0)
    tidx = torch.arange(total_tiles, device=xq.device, dtype=torch.int32) - cums[tile_seg.long()].to(torch.int32)
    tile_m0 = seg_starts[tile_seg.long()].to(torch.int32) + tidx * BLOCK_M
    grid = (total_tiles, triton.cdiv(N, BLOCK_N))
    _mx_gemm_kernel[grid](
        xq, xs, codes, cb_packed, wexp_biased, row.contiguous().float(), y,
        tile_seg, tile_m0, seg_starts[1:].contiguous().to(torch.int32),
        seg_experts.to(torch.int32),
        N, K_VEC, K // 32,
        K=K, GROUP=group, BM=BLOCK_M, BN=BLOCK_N, BK=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return y


@triton.jit
def _mx_gemm_cs_kernel(
    xq_ptr, xs_ptr, codes_ptr, cbp_ptr, wexp_ptr, row_ptr, y_ptr,
    tile_seg_ptr, tile_m0_ptr, seg_end_ptr, n_tiles_ptr,
    N, K_VEC, K_G32,
    K: tl.constexpr, GROUP: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """Capture-safe _mx_gemm_kernel: static worst-case grid + live-count exit."""
    pid_m = tl.program_id(0)
    if pid_m >= tl.load(n_tiles_ptr):
        return
    pid_n = tl.program_id(1)
    e = tl.load(tile_seg_ptr + pid_m)
    m0 = tl.load(tile_m0_ptr + pid_m)
    m_end = tl.load(seg_end_ptr + e)
    m_off = m0 + tl.arange(0, BM)
    m_mask = m_off < m_end
    n_off = pid_n * BN + tl.arange(0, BN)
    n_mask = n_off < N

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ka = k0 // 2 + tl.arange(0, BK // 2)
        xq = tl.load(xq_ptr + m_off[:, None] * (K // 2) + ka[None, :],
                     mask=m_mask[:, None], other=0)
        ks = k0 // 32 + tl.arange(0, BK // 32)
        xs = tl.load(xs_ptr + m_off[:, None] * K_G32 + ks[None, :],
                     mask=m_mask[:, None], other=127)
        kv = k0 // 4 + tl.arange(0, BK // 4)
        code = tl.load(codes_ptr + (e * N + n_off[:, None]) * K_VEC + kv[None, :],
                       mask=n_mask[:, None], other=0).to(tl.int32)
        d2 = tl.arange(0, 2)
        wq = tl.load(cbp_ptr + e * 512 + code[:, :, None] * 2 + d2[None, None, :])
        wq = tl.reshape(wq, (BN, BK // 2))
        we = tl.load(wexp_ptr + (e * N + n_off[:, None]) * (K // GROUP) + (k0 // GROUP),
                     mask=n_mask[:, None], other=127)
        ws = tl.broadcast_to(we, (BN, BK // 32))
        acc = tl.dot_scaled(xq, xs, "e2m1", tl.trans(wq), ws, "e2m1", acc)
    row = tl.load(row_ptr + e * N + n_off, mask=n_mask, other=0.0)
    y = y_ptr + m_off[:, None] * N + n_off[None, :]
    tl.store(y, (acc * row[None, :]).to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


def mx_grouped_gemm_cs(xq, xs, tables, seg_end, codes, cb_packed,
                       wexp_biased, row_f32, group,
                       BLOCK_M=32, BLOCK_N=128, BLOCK_K=64,
                       num_warps=4, num_stages=3):
    """Capture-safe fused MXFP4 grouped GEMM (row_f32 must be contiguous f32).
    Constraints: K % BLOCK_K == 0, BLOCK_K <= group, group % BLOCK_K == 0."""
    tile_seg, tile_m0, n_tiles, MAX_TILES = tables
    S = xq.shape[0]
    E, N, K_VEC = codes.shape
    K = K_VEC * 4
    y = torch.empty(S, N, dtype=torch.bfloat16, device=xq.device)
    grid = (MAX_TILES, triton.cdiv(N, BLOCK_N))
    _mx_gemm_cs_kernel[grid](
        xq, xs, codes, cb_packed, wexp_biased, row_f32, y,
        tile_seg, tile_m0, seg_end, n_tiles,
        N, K_VEC, K // 32,
        K=K, GROUP=group, BM=BLOCK_M, BN=BLOCK_N, BK=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return y


def _mx_moe_impl(x, topk_weights, topk_ids,
                 gu_vq, gu_cbp, gu_expb, gu_row, gu_group,
                 dn_vq, dn_cbp, dn_expb, dn_row, dn_group):
    """MXFP4 MoE forward, tensor-args form (capture-safe: no host syncs).
    gu_cbp/dn_cbp packed fp4 nibbles [E,256,2] u8; gu_expb/dn_expb e8m0
    biased u8; gu_row/dn_row [E,N] f32 contiguous."""
    T, H = x.shape
    E = gu_vq.shape[0]
    Kk = topk_ids.shape[-1]
    flat_e = topk_ids.reshape(-1).long()
    order = flat_e.argsort(stable=True)
    tok_idx = order // Kk
    x_sorted = x[tok_idx].contiguous()
    counts = torch.zeros(E, device=x.device, dtype=torch.long)
    counts.scatter_add_(0, flat_e, torch.ones_like(flat_e))
    seg_starts = torch.nn.functional.pad(counts.cumsum(0), (1, 0)).to(torch.int32)
    seg_end = seg_starts[1:].contiguous()
    tables = build_vq_tile_tables(seg_starts, T * Kk, E)

    xq, xs = quantize_act_mx_triton(x_sorted.to(torch.bfloat16))
    # tuned on SM120 @ T=256: gu (32,128,128), dn (32,256,64)
    y1 = mx_grouped_gemm_cs(xq, xs, tables, seg_end, gu_vq, gu_cbp, gu_expb,
                            gu_row, gu_group, 32, 128, min(gu_group, 128))
    gate, up = y1.chunk(2, dim=-1)
    h = torch.nn.functional.gelu(gate, approximate="tanh") * up
    hq, hs = quantize_act_mx_triton(h.contiguous())
    y2 = mx_grouped_gemm_cs(hq, hs, tables, seg_end, dn_vq, dn_cbp, dn_expb,
                            dn_row, dn_group, 32, 256, min(dn_group, 64),
                            num_warps=4, num_stages=2)
    y2 = y2 * topk_weights.reshape(-1)[order].to(y2.dtype)[:, None]
    out = torch.zeros(T, H, dtype=x.dtype, device=x.device)
    out.index_add_(0, tok_idx, y2.to(x.dtype))
    return out


try:
    @_custom_op("dgq::mx_moe", mutates_args=())
    def _mx_moe_op(x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                   gu_vq: torch.Tensor, gu_cbp: torch.Tensor, gu_expb: torch.Tensor,
                   gu_row: torch.Tensor, gu_group: int,
                   dn_vq: torch.Tensor, dn_cbp: torch.Tensor, dn_expb: torch.Tensor,
                   dn_row: torch.Tensor, dn_group: int) -> torch.Tensor:
        return _mx_moe_impl(x, topk_weights, topk_ids,
                            gu_vq, gu_cbp, gu_expb, gu_row, gu_group,
                            dn_vq, dn_cbp, dn_expb, dn_row, dn_group)

    @_register_fake("dgq::mx_moe")
    def _mx_moe_fake(x, topk_weights, topk_ids,
                     gu_vq, gu_cbp, gu_expb, gu_row, gu_group,
                     dn_vq, dn_cbp, dn_expb, dn_row, dn_group):
        return torch.empty_like(x)
except NameError:  # _custom_op unavailable (very old torch)
    pass


@torch._dynamo.disable
def mx_moe_forward(x, topk_weights, topk_ids, m):
    """Module-form wrapper (transformers path, m = MXFP4Experts)."""
    if not hasattr(m, "_fp4_cache"):
        m._fp4_cache = (
            pack_codebook_fp4(m.gu_cb.data), (m.gu_exp.to(torch.int16) + 127).to(torch.uint8).contiguous(),
            m.gu_row.squeeze(-1).float().contiguous(),
            pack_codebook_fp4(m.dn_cb.data), (m.dn_exp.to(torch.int16) + 127).to(torch.uint8).contiguous(),
            m.dn_row.squeeze(-1).float().contiguous(),
        )
    gu_cbp, gu_e, gu_row, dn_cbp, dn_e, dn_row = m._fp4_cache
    return _mx_moe_impl(x, topk_weights, topk_ids,
                        m.gu_vq, gu_cbp, gu_e, gu_row, m.GU_GROUP,
                        m.dn_vq, dn_cbp, dn_e, dn_row, m.DN_GROUP)
