#!/usr/bin/env python3
"""Render every lightweight training checkpoint as a GT/normal/RGB sheet."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def latest_run_dir(exp_root: Path) -> Path | None:
    candidates = [path for path in exp_root.iterdir() if path.is_dir()] if exp_root.is_dir() else []
    candidates = [path for path in candidates if (path / "checkpoints").is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def available_steps(checkpoint_dir: Path) -> list[int]:
    steps: set[int] = set()
    if (checkpoint_dir / "render_initial.ckpt").is_file():
        steps.add(0)
    for path in checkpoint_dir.glob("render_step_*.ckpt"):
        try:
            steps.add(int(path.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return sorted(steps)


def render_step(
    repository_root: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    metrics_path: Path,
    data_root: Path,
    experiment: str,
    step: int,
    device: str,
    num_context_views: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/plot_and_render_checkpoints.py",
        "--metrics",
        str(metrics_path),
        "--out",
        str(output_dir),
        "--ckpt-dir",
        str(checkpoint_dir),
        "--data-root",
        str(data_root),
        "--experiment",
        experiment,
        "--step",
        str(step),
        "--num-context-views",
        str(num_context_views),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = device
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository_root), str(repository_root / "submodules" / "diff-triangle-rasterization"), environment.get("PYTHONPATH", "")]
    )
    print(f"rendering step={step} output={output_dir}", flush=True)
    subprocess.run(command, check=True, cwd=repository_root, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DL3DV_ROOT", "./data/dl3dv")))
    parser.add_argument("--experiment", default="trisplat_dl3dv_tsdpt_5k_noschedule")
    parser.add_argument("--num-context-views", type=int, default=6)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    exp_root = repository_root / "outputs" / f"exp_{args.run_name}"
    metrics_path = repository_root / "outputs" / args.run_name / "metrics" / "metrics.jsonl"
    rendered: set[int] = set()

    while True:
        run_dir = latest_run_dir(exp_root)
        if run_dir is not None:
            checkpoint_dir = run_dir / "checkpoints"
            output_dir = run_dir / "analysis"
            for step in available_steps(checkpoint_dir):
                output_path = output_dir / f"render_grid_step_{step:04d}.jpg"
                if step in rendered or output_path.is_file():
                    rendered.add(step)
                    continue
                if not metrics_path.is_file():
                    continue
                try:
                    render_step(
                        repository_root,
                        checkpoint_dir,
                        output_dir,
                        metrics_path,
                        args.data_root,
                        args.experiment,
                        step,
                        args.device,
                        args.num_context_views,
                    )
                except Exception as error:  # Keep training alive if one export fails.
                    print(f"render failed step={step}: {error}", file=sys.stderr, flush=True)
                    break
                rendered.add(step)
        time.sleep(max(args.poll_seconds, 1.0))


if __name__ == "__main__":
    main()
