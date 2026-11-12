"""Forward-only Triton attention kernel shared by Part 03 exercises/projects."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    seqlen_q,
    seqlen_k,
    head_dim,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Compute one tile of dense attention for one batch-head pair."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = q_ptr + (pid_bh * seqlen_q + offs_m[:, None]) * head_dim + offs_d[None, :]
    q_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    row_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for block_n in range(0, tl.cdiv(seqlen_k, BLOCK_N)):
        current_n = block_n * BLOCK_N + offs_n
        kv_mask = (current_n[:, None] < seqlen_k) & (offs_d[None, :] < head_dim)
        k_ptrs = k_ptr + (pid_bh * seqlen_k + current_n[:, None]) * head_dim + offs_d[None, :]
        v_ptrs = v_ptr + (pid_bh * seqlen_k + current_n[:, None]) * head_dim + offs_d[None, :]
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k)) * sm_scale
        valid = current_n[None, :] < seqlen_k
        if IS_CAUSAL:
            valid = valid & (current_n[None, :] <= offs_m[:, None])
        scores = tl.where(valid, scores, float("-inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, block_max)
        weights = tl.exp(scores - new_max[:, None])
        correction = tl.exp(row_max - new_max)
        new_sum = row_sum * correction + tl.sum(weights, axis=1)
        acc = acc * correction[:, None] + tl.dot(weights.to(v.dtype), v)
        row_max = new_max
        row_sum = new_sum

    result = acc / row_sum[:, None]
    out_ptrs = out_ptr + (pid_bh * seqlen_q + offs_m[:, None]) * head_dim + offs_d[None, :]
    tl.store(out_ptrs, result, mask=q_mask)


def triton_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_causal: bool = False) -> torch.Tensor:
    """Apply forward-only attention to contiguous [batch, heads, sequence, dim] tensors.

    This teaching kernel supports FP16/BF16, head_dim up to 128, dense attention,
    and optional causal self-attention. It does not implement autograd, dropout,
    grouped-query attention, or arbitrary tensor strides.
    """
    if not isinstance(is_causal, bool):
        raise TypeError("is_causal must be bool")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q/K/V must have shape [batch, heads, sequence, head_dim]")
    if any(size == 0 for tensor in (q, k, v) for size in tensor.shape):
        raise ValueError("Q/K/V dimensions must be nonempty")
    if q.shape[:2] != k.shape[:2] or k.shape[:2] != v.shape[:2]:
        raise ValueError("Q/K/V batch and head dimensions must match")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError("Q/K/V head_dim must match")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("K/V sequence lengths must match")
    if not q.is_cuda or q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must be on the same CUDA device")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Q/K/V must share FP16 or BF16 dtype")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("Q/K/V must be contiguous")
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise ValueError("This kernel is forward-only and does not support autograd")

    batch, heads, seqlen_q, head_dim = q.shape
    seqlen_k = k.shape[-2]
    if not 0 < head_dim <= 128:
        raise ValueError("head_dim must be in [1, 128]")
    if is_causal and seqlen_q != seqlen_k:
        raise ValueError("Causal self-attention requires equal Q/K sequence lengths")

    block_m, block_n = 32, 64
    block_dmodel = max(16, triton.next_power_of_2(head_dim))
    output = torch.empty_like(q)
    grid = (triton.cdiv(seqlen_q, block_m), batch * heads)
    with torch.cuda.device(q.device):
        flash_attn_fwd_kernel[grid](
            q,
            k,
            v,
            output,
            seqlen_q,
            seqlen_k,
            head_dim,
            1.0 / math.sqrt(head_dim),
            block_m,
            block_n,
            block_dmodel,
            is_causal,
            num_warps=4,
        )
    return output
