#!/usr/bin/env python3
"""Benchmark the feature-only VGGT-Omega backbone over a view-count ladder."""

import argparse
import json
import time
from pathlib import Path

import torch

from src.checkpoint_utils import load_vggt_omega_aggregator
from src.model.encoder.backbone.backbone_vggt_omega import (
    BackboneVggtOmega,
    BackboneVggtOmegaCfg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", default="/home/v-hanhaoxuan/vggt-omega")
    parser.add_argument("--checkpoint", default="pretrained_weights/vggt_omega_1b_512.pt")
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--views", default="1,2,6,12,24,50")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output", type=Path, default=Path("outputs/vggt_omega_benchmark.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    cfg = BackboneVggtOmegaCfg(
        name="vggt_omega",
        source_path=args.source_path,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        frozen=True,
    )
    backbone = BackboneVggtOmega(cfg, 3).cuda().eval()
    load_vggt_omega_aggregator(backbone, Path(args.checkpoint), args.checkpoint_sha256)

    results = []
    for views in (int(value) for value in args.views.split(",")):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        images = torch.rand(1, views, 3, args.height, args.width, device="cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                output = backbone(images)
            torch.cuda.synchronize()
            result = {
                "views": views,
                "success": True,
                "seconds": time.perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "dtype": str(output.tokens.dtype),
                "token_shape": list(output.tokens.shape),
                "finite": bool(torch.isfinite(output.tokens).all()),
            }
        except torch.cuda.OutOfMemoryError as exc:
            result = {"views": views, "success": False, "error": str(exc)}
            results.append(result)
            break
        results.append(result)
        print(json.dumps(result))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
