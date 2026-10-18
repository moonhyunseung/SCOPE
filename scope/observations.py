"""Bounded-sample summaries. P0 only: not a real-model distribution analyzer."""
from __future__ import annotations

import torch

from scope.reference import AttentionTrace, GroupwiseQuantized


def tensor_summary(x: torch.Tensor, *, max_samples: int = 131072) -> dict:
    """Compute descriptive statistics on a deterministic stride sample."""
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    flat = x.detach().reshape(-1)
    count = flat.numel()
    if not count:
        raise ValueError("empty tensors are outside this P0 schema")
    stride = max(1, (count + max_samples - 1) // max_samples)
    subset = flat[::stride][:max_samples].to(torch.float32)
    valid = subset[torch.isfinite(subset)]
    result = {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "total_elements": count,
        "sample_elements": int(subset.numel()),
        "sampling_stride": stride,
        "sample_nonfinite_count": int((~torch.isfinite(subset)).sum().item()),
    }
    if not valid.numel():
        result["finite_stats"] = None
        return result
    abs_values = valid.abs()
    abs_quantiles = torch.quantile(abs_values, torch.tensor([0.50, 0.99], device=x.device))
    result["finite_stats"] = {
        "min": float(valid.min().item()),
        "max": float(valid.max().item()),
        "mean": float(valid.mean().item()),
        "std_population": float(valid.std(unbiased=False).item()),
        "rms": float(valid.square().mean().sqrt().item()),
        "abs_p50": float(abs_quantiles[0].item()),
        "abs_p99": float(abs_quantiles[1].item()),
        "zero_fraction": float((valid == 0).to(torch.float32).mean().item()),
    }
    return result


def code_histogram(codes: torch.Tensor) -> dict:
    """Exact histogram for signed INT4 codes in [-8,7]."""
    if codes.dtype != torch.int8:
        raise ValueError("logical codes must be int8")
    values = codes.reshape(-1).to(torch.int16)
    if bool(((values < -8) | (values > 7)).any().item()):
        raise ValueError("code outside signed INT4 alphabet")
    counts = torch.bincount((values + 8).to(torch.int64).cpu(), minlength=16)
    return {str(code): int(counts[code + 8].item()) for code in range(-8, 8)}


def probability_summary(p: torch.Tensor) -> dict:
    """P last axis is the token axis; summarize rows independently then average."""
    if p.ndim != 4:
        raise ValueError("expected [B,Hq,1,L]")
    p32 = p.to(torch.float32)
    entropy = -(p32 * torch.log(p32.clamp_min(1e-30))).sum(dim=-1)
    top_k_mass = p32.topk(min(8, p32.shape[-1]), dim=-1).values.sum(dim=-1)
    return {
        "mean_entropy_nats": float(entropy.mean().item()),
        "mean_top1_mass": float(p32.max(dim=-1).values.mean().item()),
        "mean_top8_mass": float(top_k_mass.mean().item()),
        "max_row_sum_error": float((p32.sum(dim=-1) - 1).abs().max().item()),
    }


def trace_summary(trace: AttentionTrace, kq: GroupwiseQuantized, vq: GroupwiseQuantized) -> dict:
    tensors = {
        "Q": trace.q, "K_decoded": trace.k, "V_decoded": trace.v,
        "Z": trace.z, "P": trace.p, "Y": trace.y,
        "S_K": kq.scales, "S_V": vq.scales,
    }
    if trace.w is not None:
        tensors["W=P_times_S_V"] = trace.w
    return {
        "tensor_statistics": {name: tensor_summary(x) for name, x in tensors.items()},
        "code_histograms": {
            "C_K": code_histogram(kq.codes),
            "C_V": code_histogram(vq.codes),
        },
        "probability_statistics": probability_summary(trace.p),
    }
