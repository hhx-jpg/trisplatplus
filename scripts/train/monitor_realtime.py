#!/usr/bin/env python3
"""Continuously export training curves and the newest validation render."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


CURVES = (
    ("loss/total", "Total loss"),
    ("loss/mse", "MSE"),
    ("loss/lpips", "LPIPS"),
    ("train/psnr_probabilistic", "Train PSNR"),
    ("val/psnr", "Validation PSNR"),
)


def read_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "step" in row:
                rows.append(row)
    return rows


def export_metrics_csv(rows: list[dict], output_dir: Path) -> None:
    """Publish the same live metrics as a readable, atomically replaced CSV."""
    fields = (
        "step",
        "loss_total",
        "loss_mse",
        "loss_lpips",
        "mse_black_hole_fraction",
        "skipped",
        "global_step",
        "train_psnr_probabilistic",
        "ratio_opacity_lt_0_01",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "training_log.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            # Weight-only continuation starts Lightning's logger step at zero,
            # while the model records the actual continued step separately.
            # Publish the latter so curves remain continuous across resumes.
            plot_step = row.get("info/global_step", row.get("step", ""))
            writer.writerow(
                {
                    "step": int(plot_step),
                    "loss_total": row.get("loss/total", ""),
                    "loss_mse": row.get("loss/mse", ""),
                    "loss_lpips": row.get("loss/lpips", ""),
                    # Current LossMse logs this as loss/mse_black_hole_fraction;
                    # accept the dotted spelling from older metric files too.
                    "mse_black_hole_fraction": row.get(
                        "loss/mse_black_hole_fraction",
                        row.get("loss/mse.mse_black_hole_fraction", ""),
                    ),
                    "skipped": bool(row.get("info/skipped_batch", 0.0)),
                    "global_step": row.get("info/global_step", ""),
                    "train_psnr_probabilistic": row.get("train/psnr_probabilistic", ""),
                    "ratio_opacity_lt_0_01": row.get("info/ratio_opacity<0.01", ""),
                }
            )
    temporary.replace(output_dir / "training_log.csv")


def latest_render(render_dir: Path) -> Path | None:
    paths = list(render_dir.glob("*.png")) if render_dir.exists() else []
    if not paths:
        return None

    def step_key(path: Path) -> tuple[int, str]:
        try:
            return (int(path.stem.rsplit("_", 1)[-1]), path.name)
        except ValueError:
            return (-1, path.name)

    return max(paths, key=step_key)


def export_dashboard(rows: list[dict], render_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_metrics_csv(rows, output_dir)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    axes_flat = axes.flat
    for index, (key, title) in enumerate(CURVES):
        axis = next(axes_flat)
        points = [
            (
                float(row.get("info/global_step", row["step"])),
                float(row[key]),
            )
            for row in rows
            if key in row
        ]
        if points:
            steps, values = zip(*points)
            axis.plot(steps, values, linewidth=1.5, marker=".", markersize=2)
            axis.set_xlim(left=0)
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.set_ylabel(key)
        axis.grid(True, alpha=0.3)
    image_axis = next(axes_flat)
    image_axis.axis("off")
    render = latest_render(render_dir)
    if render is not None:
        try:
            with Image.open(render) as image:
                image_axis.imshow(image)
            image_axis.set_title(f"Latest render: {render.stem}")
            shutil.copyfile(render, output_dir / "latest_render.png")
        except (OSError, ValueError):
            pass
    else:
        image_axis.text(0.5, 0.5, "Waiting for validation render", ha="center", va="center")
    fig.savefig(output_dir / "training_dashboard.png", dpi=140)
    fig.savefig(output_dir / "training_curves.png", dpi=140)
    # Keep the filename used by the log-based plotter live as well.
    fig.savefig(output_dir / "training_log_curves.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("render_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        export_dashboard(read_metrics(args.metrics), args.render_dir, args.output_dir)
        if args.once:
            return
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    main()
