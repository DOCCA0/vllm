# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze EPLB scheduler scaling and compare it with serving time saved."""

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
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--serving-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def summarize_profile(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    build_ms: list[float] = []
    greedy_ms: list[float] = []
    for rank in result["scheduler_per_rank"]:
        build_ms.extend(value / 1e6 for value in rank["build_ns"])
        greedy_ms.extend(value / 1e6 for value in rank["schedule_ns"])
    total_ms = [build + greedy for build, greedy in zip(build_ms, greedy_ms)]
    return {
        "migrations": result["num_instructions"],
        "flows": result["num_flows"],
        "batches": result["num_batches"],
        "build_p50_ms": percentile(build_ms, 50),
        "build_p99_ms": percentile(build_ms, 99),
        "greedy_p50_ms": percentile(greedy_ms, 50),
        "greedy_p99_ms": percentile(greedy_ms, 99),
        "total_p50_ms": percentile(total_ms, 50),
        "total_p99_ms": percentile(total_ms, 99),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def measure_cost_benefit(serving_dir: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    pattern = re.compile(
        r"EPLB migration stats:.*?transfers=(\d+).*?"
        r"build_ns=(\d+) greedy_ns=(\d+)"
    )
    for workload in ("random", "phased"):
        workload_dir = serving_dir / workload
        off = json.loads(
            (workload_dir / "00_async_off_clean" / "bench.json").read_text()
        )
        on_dir = next(
            path
            for path in (
                workload_dir / "01_async_batching_on",
                workload_dir / "02_async_batching_on",
            )
            if path.exists()
        )
        on = json.loads((on_dir / "bench.json").read_text())
        log = (on_dir / "server.log").read_text(errors="replace")
        timings = [tuple(map(int, match)) for match in pattern.findall(log)]
        migrations = [timing[0] for timing in timings]
        build_ms = [timing[1] / 1e6 for timing in timings]
        greedy_ms = [timing[2] / 1e6 for timing in timings]
        total_ms = [build + greedy for build, greedy in zip(build_ms, greedy_ms)]
        saved_ms = (off["duration"] - on["duration"]) * 1000
        scheduler_total_ms = sum(total_ms)
        output.append(
            {
                "workload": workload,
                "layer_migrations": len(migrations),
                "expert_migrations": sum(migrations),
                "build_total_ms": sum(build_ms),
                "greedy_total_ms": sum(greedy_ms),
                "scheduler_total_ms": scheduler_total_ms,
                "scheduler_per_layer_p50_ms": percentile(total_ms, 50),
                "scheduler_per_layer_p99_ms": percentile(total_ms, 99),
                "async_off_duration_s": off["duration"],
                "async_on_duration_s": on["duration"],
                "serving_time_saved_ms": saved_ms,
                "cost_to_saving_ratio_pct": (
                    scheduler_total_ms / saved_ms * 100 if saved_ms > 0 else None
                ),
            }
        )
    return output


def plot(
    rows: list[dict[str, Any]], cost_benefit: list[dict[str, Any]], path: Path
) -> None:
    migrations = np.asarray([row["migrations"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))

    scheduler = axes[0]
    scheduler.plot(
        migrations,
        [row["build_p50_ms"] for row in rows],
        "o-",
        label="Instruction build P50",
    )
    scheduler.plot(
        migrations,
        [row["greedy_p50_ms"] for row in rows],
        "o-",
        label="Greedy grouping P50",
    )
    scheduler.plot(
        migrations,
        [row["total_p50_ms"] for row in rows],
        "o--",
        label="Build + greedy grouping P50",
    )
    scheduler.set_xlabel("Expert migrations per layer")
    scheduler.set_ylabel("Python scheduling time (ms)")
    scheduler.set_title("Scheduler scaling (9 MiB experts)")
    scheduler.grid(True, alpha=0.25)
    scheduler.legend(fontsize=8)

    benefit = axes[1]
    positions = np.arange(len(cost_benefit))
    width = 0.34
    scheduler_bars = benefit.bar(
        positions - width / 2,
        [row["scheduler_total_ms"] for row in cost_benefit],
        width,
        label="Measured build + greedy grouping",
    )
    saving_bars = benefit.bar(
        positions + width / 2,
        [row["serving_time_saved_ms"] for row in cost_benefit],
        width,
        label="Async serving time saved",
    )
    benefit.set_xticks(positions)
    benefit.set_xticklabels([row["workload"].title() for row in cost_benefit])
    benefit.set_ylabel("Time over the full benchmark (ms)")
    benefit.set_title("Actual scheduling cost versus serving change")
    benefit.grid(True, axis="y", alpha=0.25)
    benefit.legend(fontsize=8)
    for bars in (scheduler_bars, saving_bars):
        for bar in bars:
            height = bar.get_height()
            benefit.annotate(
                f"{height:.0f} ms",
                (bar.get_x() + bar.get_width() / 2, max(height, 0)),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    benefit.margins(y=0.12)

    figure.suptitle("EPLB batching scheduler overhead")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    serving_dir = args.serving_dir or (
        args.result_dir.parent / "async_profile_20260904"
    )
    output_dir = args.output_dir or args.result_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        (summarize_profile(path) for path in (args.result_dir / "raw").glob("*.json")),
        key=lambda row: row["migrations"],
    )
    if not rows:
        raise RuntimeError(f"No JSON files found in {args.result_dir / 'raw'}")
    write_csv(rows, output_dir / "scheduler_scaling.csv")
    cost_benefit = measure_cost_benefit(serving_dir)
    write_csv(cost_benefit, output_dir / "cost_benefit.csv")
    plot(rows, cost_benefit, output_dir / "scheduler_cost_benefit.png")


if __name__ == "__main__":
    main()
