"""P0 diagnostic reference for one-token grouped-query decode attention.

IMPORTANT:
- Quantizer outputs LOGICAL signed 4-bit codes held in int8 tensors. They are
  NOT packed INT4 storage and have no claimed resident-memory advantage.
- This intentionally materializes score/probability tensors for observation.
- This is not a fused serving kernel or the actual OFD quantization contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class GroupwiseQuantized:
    codes: torch.Tensor
    scales: torch.Tensor
    decoded: torch.Tensor
    group_size: int


@dataclass(frozen=True)
class AttentionTrace:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    z: torch.Tensor
    p: torch.Tensor
    y: torch.Tensor
    w: torch.Tensor | None = None


def symmetric_int4_groupwise(x: torch.Tensor, group_size: int = 32) -> GroupwiseQuantized:
    """Illustrative symmetric [-7,7] round-to-nearest groupwise quantizer.

    Returns FP32 scales and FP32 decoded values. Storage-scale dtype and
    rounding must be replaced for an actual compressed-cache implementation.
    """
    if x.ndim < 1 or not x.is_floating_point():
        raise ValueError("x must be a floating tensor")
    if group_size < 1 or x.shape[-1] % group_size:
        raise ValueError("group_size must be positive and divide last dimension")
    x32 = x.to(torch.float32)
    if not bool(torch.isfinite(x32).all().item()):
        raise ValueError("nonfinite input is not supported by this illustrative quantizer")
    groups = x32.reshape(*x32.shape[:-1], x32.shape[-1] // group_size, group_size)
    magnitudes = groups.abs().amax(dim=-1)
    scales = torch.where(magnitudes > 0, magnitudes / 7.0, torch.ones_like(magnitudes))
    codes = torch.round(groups / scales.unsqueeze(-1)).clamp(-7, 7).to(torch.int8)
    decoded = (codes.to(torch.float32) * scales.unsqueeze(-1)).reshape(x32.shape)
    return GroupwiseQuantized(codes.reshape(x.shape), scales, decoded, group_size)


def reference_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    v_scales: torch.Tensor | None = None,
) -> AttentionTrace:
    """Single-token decode: Q [B,Hq,1,D], K/V [B,Hkv,L,D].

    Returns Z/P/Y in FP32. No explicit mask is needed for a final-position
    decode query when K/V contain only the currently visible tokens.
    Caller is responsible for model-specific RoPE and attention bias.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected 4D Q,K,V")
    batch, hq, lq, dim = q.shape
    bk, hkv, length, dk = k.shape
    if lq != 1 or length < 1:
        raise ValueError("this reference supports one query and at least one KV token")
    if bk != batch or v.shape != k.shape or dim != dk or hkv < 1 or hq % hkv:
        raise ValueError("incompatible Q,K,V shapes or GQA head counts")
    if q.device != k.device or k.device != v.device:
        raise ValueError("all tensors must be on the same device")
    if not (q.is_floating_point() and k.is_floating_point() and v.is_floating_point()):
        raise ValueError("Q,K,V must be floating point tensors")

    repetition = hq // hkv
    q32 = q.to(torch.float32)
    k32 = k.to(torch.float32).repeat_interleave(repetition, dim=1)
    v32 = v.to(torch.float32).repeat_interleave(repetition, dim=1)
    z = torch.matmul(q32, k32.transpose(-2, -1)) * (1.0 / math.sqrt(dim))
    p = torch.softmax(z, dim=-1)
    y = torch.matmul(p, v32)

    w = None
    if v_scales is not None:
        if v_scales.ndim != 4 or v_scales.shape[:3] != (batch, hkv, length):
            raise ValueError("v_scales must have shape [B,Hkv,L,G]")
        if v_scales.device != q.device:
            raise ValueError("v_scales device must match Q")
        w = p.squeeze(-2).unsqueeze(-1) * v_scales.to(torch.float32).repeat_interleave(repetition, dim=1)
    return AttentionTrace(q=q32, k=k.to(torch.float32), v=v.to(torch.float32), z=z, p=p, y=y, w=w)
