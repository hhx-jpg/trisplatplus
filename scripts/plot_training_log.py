"""Extract a compact training curve from the rank-zero/main training log."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


TRAIN_RE = re.compile(
    r"train step (?P<step>\d+);.*?loss = (?P<loss>[-+\deE.]+); "
    r"loss_breakdown = mse=(?P<mse>[-+\deE.]+), lpips=(?P<lpips>[-+\deE.]+)"
    r"(?P<tail>.*)$"
)


def parse_log(path: Path) -> list[dict[str, float | int | bool]]:
    rows: list[dict[str, float | int | bool]] = []
    seen: set[tuple[int, float, float, float, bool]] = set()
    for line in path.read_text(errors="replace").splitlines():
        match = TRAIN_RE.search(line)
        if match is None:
            continue
        tail = match.group("tail")
        row: dict[str, float | int | bool] = {
            "step": int(match.group("step")),
            "loss_total": float(match.group("loss")),
            "loss_mse": float(match.group("mse")),
            "loss_lpips": float(match.group("lpips")),
            "skipped": "skip_reasons" in tail,
        }
        key = (
            int(row["step"]),
            float(row["loss_total"]),
            float(row["loss_mse"]),
            float(row["loss_lpips"]),
            bool(row["skipped"]),
        )
        if key not in seen:
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: (int(row["step"]), bool(row["skipped"])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = parse_log(args.log)
    if not rows:
        raise SystemExit(f"No training-step records found in {args.log}")

    with (args.out_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    steps = [int(row["step"]) for row in rows]
    skipped = [bool(row["skipped"]) for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    series = (
        ("loss_total", "Total loss", "tab:blue"),
        ("loss_mse", "MSE", "tab:orange"),
        ("loss_lpips", "LPIPS", "tab:green"),
    )
    for axis, (key, title, color) in zip(axes, series):
        values = [float(row[key]) for row in rows]
        axis.plot(steps, values, ".-", color=color, linewidth=0.8, markersize=2, label=title)
        if any(skipped):
            skip_steps = [step for step, flag in zip(steps, skipped) if flag]
            skip_values = [value for value, flag in zip(values, skipped) if flag]
            axis.scatter(skip_steps, skip_values, color="crimson", s=12, label="skipped")
        axis.set_ylabel(title)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Optimizer step (repeated values are skipped batches)")
    fig.suptitle("TSDPT training curve from main.log")
    fig.savefig(args.out_dir / "training_log_curves.png", dpi=160)
    plt.close(fig)
    print(f"wrote {len(rows)} records to {args.out_dir}")


if __name__ == "__main__":
    main()
