"""Compare two native TriSplat test-render directories for one scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def alpha_stats(path: Path) -> dict[str, float]:
    alpha = np.load(path).astype(np.float32)
    return {
        "alpha_mean": float(alpha.mean()),
        "alpha_gt_001": float((alpha > 1e-2).mean()),
        "alpha_gt_01": float((alpha > 1e-1).mean()),
        "alpha_gt_05": float((alpha > 5e-1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline_metrics = json.loads((args.baseline / "metrics.json").read_text())
    candidate_metrics = json.loads((args.candidate / "metrics.json").read_text())
    baseline_by_index = {int(row["index"]): row for row in baseline_metrics["frames"]}
    candidate_by_index = {int(row["index"]): row for row in candidate_metrics["frames"]}
    indices = [int(index) for index in baseline_metrics["target_indices"]]
    if indices != [int(index) for index in candidate_metrics["target_indices"]]:
        raise ValueError("The two runs do not contain the same target indices")

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    panels: list[tuple[int, Image.Image, Image.Image, Image.Image, Image.Image]] = []
    for index in indices:
        baseline_path = args.baseline / "color" / f"{index:06d}.png"
        candidate_path = args.candidate / "color" / f"{index:06d}.png"
        gt_path = args.baseline / "gt" / f"{index:06d}.png"
        gt = load_rgb(gt_path)
        baseline = load_rgb(baseline_path)
        candidate = load_rgb(candidate_path)
        diff = np.abs(candidate - baseline)
        diff = np.clip(diff * 3.0, 0.0, 1.0)
        to_image = lambda array: Image.fromarray((array * 255.0).round().astype(np.uint8))
        panels.append((index, to_image(gt), to_image(baseline), to_image(candidate), to_image(diff)))

        row = {
            "index": index,
            "baseline": baseline_by_index[index],
            "candidate": candidate_by_index[index],
            "candidate_minus_baseline_psnr_db": float(
                candidate_by_index[index]["psnr"] - baseline_by_index[index]["psnr"]
            ),
            "rgb_diff_mae": float(np.abs(candidate - baseline).mean()),
            "rgb_diff_rms": float(np.sqrt(np.square(candidate - baseline).mean())),
            "rgb_diff_gt_0_1_fraction": float((np.abs(candidate - baseline).mean(axis=-1) > 0.1).mean()),
            "baseline_alpha": alpha_stats(args.baseline / "raw_alpha" / "target" / f"{index:06d}.npy"),
            "candidate_alpha": alpha_stats(args.candidate / "raw_alpha" / "target" / f"{index:06d}.npy"),
        }
        rows.append(row)

    panel_w, panel_h = panels[0][1].size
    header_h, row_gap = 28, 4
    sheet = Image.new("RGB", (panel_w * 4, header_h + len(panels) * (panel_h + row_gap)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, title in enumerate(("GT", "TriSplat baseline", "TSDPT 1k", "|TSDPT-baseline| x3")):
        draw.text((col * panel_w + 6, 7), title, fill="black")
    for row_idx, (index, gt, baseline, candidate, diff) in enumerate(panels):
        y = header_h + row_idx * (panel_h + row_gap)
        for col, image in enumerate((gt, baseline, candidate, diff)):
            sheet.paste(image, (col * panel_w, y))
        draw.text((panel_w * 3 - 90, y + 6), f"target {index}", fill="white")
    sheet.save(args.out / "comparison_grid.png")

    summary = {
        "scene": baseline_metrics["scene"],
        "context_indices": baseline_metrics["context_indices"],
        "target_indices": indices,
        "baseline_summary": {key: baseline_metrics[f"{key}_mean"] for key in ("psnr", "ssim", "lpips")},
        "candidate_summary": {key: candidate_metrics[f"{key}_mean"] for key in ("psnr", "ssim", "lpips")},
        "delta_candidate_minus_baseline": {
            key: candidate_metrics[f"{key}_mean"] - baseline_metrics[f"{key}_mean"]
            for key in ("psnr", "ssim", "lpips")
        },
        "frames": rows,
    }
    (args.out / "metrics_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
