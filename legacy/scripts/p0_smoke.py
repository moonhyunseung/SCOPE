#!/usr/bin/env python3
"""Synthetic P0 correctness/observability smoke; NOT a real-model trace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from scope.reference import reference_decode_attention, symmetric_int4_groupwise
from scope.observations import trace_summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--hq", type=int, default=24)
    ap.add_argument("--hkv", type=int, default=4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA was requested but torch.cuda.is_available() is false")
    if min(args.length, args.batch, args.hq, args.hkv, args.dim, args.group) < 1:
        ap.error("dimensions must be positive")
    if args.hq % args.hkv or args.dim % args.group:
        ap.error("require hq divisible by hkv and dim divisible by group")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    # Device-independent seeded inputs; synthetic only, NOT a model sample.
    q = torch.randn((args.batch, args.hq, 1, args.dim), generator=generator).to(args.device)
    k = torch.randn((args.batch, args.hkv, args.length, args.dim), generator=generator).to(args.device)
    v = torch.randn((args.batch, args.hkv, args.length, args.dim), generator=generator).to(args.device)
    with torch.inference_mode():
        kq = symmetric_int4_groupwise(k, args.group)
        vq = symmetric_int4_groupwise(v, args.group)
        trace = reference_decode_attention(q, kq.decoded, vq.decoded, v_scales=vq.scales)
        repetition = args.hq // args.hkv
        sdpa = F.scaled_dot_product_attention(
            q,
            kq.decoded.repeat_interleave(repetition, dim=1),
            vq.decoded.repeat_interleave(repetition, dim=1),
            is_causal=False,
        )
        difference = (trace.y - sdpa).abs()
        abs_max = float(difference.max().item())
        rel_l2 = float((trace.y - sdpa).norm().div(sdpa.norm().clamp_min(1e-12)).item())
        # This is a consistency sanity check, not an end-to-end model-fidelity gate.
        passed = bool(torch.allclose(trace.y, sdpa, rtol=5e-4, atol=5e-4))
        report = {
            "schema_version": "scope-p0-0.1",
            "status": "PASS" if passed else "FAIL",
            "input_kind": "synthetic",
            "is_real_model_trace": False,
            "is_performance_benchmark": False,
            "model_id": None,
            "model_revision": None,
            "dataset_id": None,
            "git_sha": None,
            "seed": args.seed,
            "device": args.device,
            "gpu_name": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "reference": "single-query decode / FP32 QK-softmax-PV / contiguous GQA / no masking needed at final visible token",
            "quantizer": "illustrative symmetric signed [-7,7], FP32 group scales, int8 container (NOT packed INT4)",
            "dimensions": {"B": args.batch, "Hq": args.hq, "Hkv": args.hkv, "L": args.length, "D": args.dim, "G": args.dim // args.group},
            "sdpa_consistency": {"max_abs_error": abs_max, "relative_l2": rel_l2, "atol": 5e-4, "rtol": 5e-4, "passed": passed},
            "observations": trace_summary(trace, kq, vq),
        }
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"{report['status']} | synthetic P0 | written to {args.output}")
    else:
        print(encoded)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
