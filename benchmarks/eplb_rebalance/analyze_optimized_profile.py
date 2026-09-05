# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize trace-based scheduler and serving profiles."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import regex as re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_summary", type=Path)
    parser.add_argument("serving_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--migration-log-dir", type=Path, required=True)
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
        for mode, batching, tag in (
            ("sync", "off", "00_sync_off"),
            ("sync", "on", "01_sync_on"),
            ("async", "off", "00_async_off"),
            ("async", "on", "01_async_on"),
        ):
            case = root / workload / tag
            result = json.loads((case / "bench.json").read_text())
            nic_p50, nic_p99 = nic_percentiles(case / "nic.tsv")
            rows.append(
                {
                    "workload": workload,
                    "mode": mode,
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


def formal_run_log(log: str, warmup_requests: int = 8) -> str:
    completed = 0
    lines = log.splitlines(keepends=True)
    for offset, line in enumerate(lines):
        if 'POST /v1/completions HTTP/1.1" 200 OK' in line:
            completed += 1
            if completed == warmup_requests:
                return "".join(lines[offset + 1 :])
    raise RuntimeError("Could not find the start of the formal benchmark")


def calculate_cost_benefit(
    trace: dict[int, dict[str, float]],
    serving: list[dict[str, Any]],
    migration_log_dir: Path,
) -> list[dict[str, Any]]:
    trace_migrations = np.asarray(sorted(trace))
    trace_p50_ms = np.asarray(
        [trace[count]["fused_total_p50_ms"] for count in trace_migrations]
    )
    rows: list[dict[str, Any]] = []
    for workload in ("random", "phased"):
        log = (migration_log_dir / workload / "01_async_on" / "server.log").read_text(
            errors="replace"
        )
        migrations = [
            int(value)
            for value in re.findall(
                r"EPLB migration stats:.*?transfers=(\d+)", formal_run_log(log)
            )
        ]
        pair = {
            row["batching"]: row
            for row in serving
            if row["workload"] == workload and row["mode"] == "async"
        }
        saved_ms = (pair["off"]["duration_s"] - pair["on"]["duration_s"]) * 1000
        cost_p50 = float(np.interp(migrations, trace_migrations, trace_p50_ms).sum())
        rows.append(
            {
                "workload": workload,
                "scheduler_calls": len(migrations),
                "maximum_migrations_per_call": max(migrations),
                "scheduler_cost_p50_ms": cost_p50,
                "serving_time_saved_ms": saved_ms,
                "cost_saved_p50_pct": cost_p50 / saved_ms * 100,
            }
        )
    return rows


def plot(
    trace: dict[int, dict[str, float]],
    cost_bounds: list[dict[str, Any]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.5))

    migrations = np.asarray(sorted(trace))
    scheduler = axes[0]
    scheduler.plot(
        migrations,
        [trace[count]["fused_total_p50_ms"] for count in migrations],
        "o-",
        label="Flow scheduling P50",
    )
    scheduler.set_xlabel("Expert migrations per scheduler call")
    scheduler.set_ylabel("PyTorch trace time (ms per call)")
    scheduler.set_title("Scheduler CPU cost")
    scheduler.grid(True, alpha=0.25)
    scheduler.legend(fontsize=8)

    positions = np.arange(len(cost_bounds))
    width = 0.34
    serving_axis = axes[1]
    cost_bars = serving_axis.bar(
        positions - width / 2,
        [row["scheduler_cost_p50_ms"] for row in cost_bounds],
        width,
        label="Trace-derived scheduler cost P50",
    )
    saving_bars = serving_axis.bar(
        positions + width / 2,
        [row["serving_time_saved_ms"] for row in cost_bounds],
        width,
        label="Async serving time saved",
    )
    serving_axis.set_xticks(positions)
    serving_axis.set_xticklabels(
        [
            f"{row['workload'].title()}\ncost/saved: {row['cost_saved_p50_pct']:.2f}%"
            for row in cost_bounds
        ]
    )
    serving_axis.set_ylabel("Time over full benchmark (ms)")
    serving_axis.set_title("Scheduler cost versus serving time saved")
    serving_axis.grid(True, axis="y", alpha=0.25)
    serving_axis.legend(fontsize=8)
    for bars in (cost_bars, saving_bars):
        for bar in bars:
            height = bar.get_height()
            serving_axis.annotate(
                f"{height:,.0f} ms",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    serving_axis.margins(y=0.35)

    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = load_trace(args.trace_summary)
    serving = load_serving(args.serving_dir)
    cost_bounds = calculate_cost_benefit(trace, serving, args.migration_log_dir)
    write_csv(serving, args.output_dir / "serving_summary.csv")
    write_csv(cost_bounds, args.output_dir / "cost_benefit.csv")
    plot(
        trace,
        cost_bounds,
        args.output_dir / "scheduler_and_serving.png",
    )


if __name__ == "__main__":
    main()
