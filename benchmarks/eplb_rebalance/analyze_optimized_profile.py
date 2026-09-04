# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize trace-based scheduler and decode-heavy serving profiles."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_summary", type=Path)
    parser.add_argument("serving_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def load_trace(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            implementation, operation, migrations = row["event"].split(".")
            key = f"{implementation}_{operation}"
            values = rows.setdefault(int(migrations), {})
            values[f"{key}_p50_ms"] = float(row["p50_ms"])
            values[f"{key}_p99_ms"] = float(row["p99_ms"])
    return rows


def nic_percentiles(path: Path) -> tuple[float, float]:
    samples = np.loadtxt(path)
    seconds = np.diff(samples[:, 0])
    transferred_bytes = np.diff(samples[:, 1] + samples[:, 2])
    rates = transferred_bytes[seconds > 0] / seconds[seconds > 0] / 1e6
    return tuple(float(value) for value in np.percentile(rates, (50, 99)))


def load_serving(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workload in ("random", "phased"):
        for batching, tag in (("off", "00_async_off"), ("on", "01_async_on")):
            case = root / workload / tag
            result = json.loads((case / "bench.json").read_text())
            nic_p50, nic_p99 = nic_percentiles(case / "nic.tsv")
            rows.append(
                {
                    "workload": workload,
                    "batching": batching,
                    "completed": result["completed"],
                    "duration_s": result["duration"],
                    "output_tok_s": result["output_throughput"],
                    "ttft_p50_ms": result["median_ttft_ms"],
                    "ttft_p99_ms": result["p99_ttft_ms"],
                    "tpot_p50_ms": result["median_tpot_ms"],
                    "tpot_p99_ms": result["p99_tpot_ms"],
                    "e2el_p50_ms": result["median_e2el_ms"],
                    "e2el_p99_ms": result["p99_e2el_ms"],
                    "nic_p50_mb_s": nic_p50,
                    "nic_p99_mb_s": nic_p99,
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def improvement(off: dict[str, Any], on: dict[str, Any], metric: str) -> float:
    if metric == "output_tok_s":
        return (on[metric] - off[metric]) / off[metric] * 100
    return (off[metric] - on[metric]) / off[metric] * 100


def plot(
    trace: dict[int, dict[str, float]],
    serving: list[dict[str, Any]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.5))

    migrations = np.asarray(sorted(trace))
    scheduler = axes[0]
    for key, label, style in (
        ("before_build_p50_ms", "Previous build P50", "o-"),
        ("after_build_p50_ms", "Optimized build P50", "o-"),
        ("after_greedy_p50_ms", "Greedy grouping P50", "o--"),
    ):
        scheduler.plot(
            migrations,
            [trace[count][key] for count in migrations],
            style,
            label=label,
        )
    scheduler.set_xlabel("Expert migrations per scheduler call")
    scheduler.set_ylabel("PyTorch trace time (ms per call)")
    scheduler.set_title("Scheduler CPU cost")
    scheduler.grid(True, alpha=0.25)
    scheduler.legend(fontsize=8)

    by_workload = {
        workload: {
            row["batching"]: row for row in serving if row["workload"] == workload
        }
        for workload in ("random", "phased")
    }
    metrics = (
        ("output_tok_s", "Throughput"),
        ("ttft_p50_ms", "TTFT P50"),
        ("ttft_p99_ms", "TTFT P99"),
        ("tpot_p50_ms", "TPOT P50"),
        ("tpot_p99_ms", "TPOT P99"),
        ("e2el_p50_ms", "E2EL P50"),
        ("e2el_p99_ms", "E2EL P99"),
        ("nic_p99_mb_s", "NIC P99"),
    )
    positions = np.arange(len(metrics))
    width = 0.38
    serving_axis = axes[1]
    for offset, workload in ((-width / 2, "random"), (width / 2, "phased")):
        pair = by_workload[workload]
        values = [improvement(pair["off"], pair["on"], metric) for metric, _ in metrics]
        bars = serving_axis.bar(
            positions + offset,
            values,
            width,
            label=workload.title(),
        )
        for bar, value in zip(bars, values):
            serving_axis.annotate(
                f"{value:+.1f}%",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3 if value >= 0 else -3),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7,
                rotation=90,
            )
    serving_axis.axhline(0, color="black", linewidth=0.8)
    serving_axis.set_xticks(positions)
    serving_axis.set_xticklabels(
        [label for _, label in metrics], rotation=35, ha="right"
    )
    serving_axis.set_ylabel("Improvement with batching (%)")
    serving_axis.set_title("Decode-heavy async serving")
    serving_axis.grid(True, axis="y", alpha=0.25)
    serving_axis.legend(fontsize=8)
    serving_axis.margins(y=0.28)

    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = load_trace(args.trace_summary)
    serving = load_serving(args.serving_dir)
    write_csv(serving, args.output_dir / "serving_summary.csv")
    plot(trace, serving, args.output_dir / "optimized_scheduler_and_serving.png")


if __name__ == "__main__":
    main()
