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


def estimate_cost_benefit(
    rows: list[dict[str, Any]], result_dir: Path
) -> list[dict[str, Any]]:
    serving_dir = result_dir.parent / "nixl_cache_on_20260901"
    x = np.asarray([row["migrations"] for row in rows])
    p50 = np.asarray([row["total_p50_ms"] for row in rows])
    p99 = np.asarray([row["total_p99_ms"] for row in rows])
    output: list[dict[str, Any]] = []
    pattern = re.compile(r"EPLB migration stats:.*?transfers=(\d+)")
    for workload in ("random", "phased"):
        workload_dir = serving_dir / workload
        off = json.loads((workload_dir / "03_async_off" / "bench.json").read_text())
        on = json.loads(
            (workload_dir / "04_async_batching_on" / "bench.json").read_text()
        )
        log = (workload_dir / "04_async_batching_on" / "server.log").read_text(
            errors="replace"
        )
        migrations = [int(value) for value in pattern.findall(log)]
        saved_ms = (off["duration"] - on["duration"]) * 1000
        scheduler_p50_ms = float(sum(np.interp(migrations, x, p50)))
        scheduler_p99_ms = float(sum(np.interp(migrations, x, p99)))
        output.append(
            {
                "workload": workload,
                "layer_migrations": len(migrations),
                "expert_migrations": sum(migrations),
                "min_migrations_per_layer": min(migrations),
                "max_migrations_per_layer": max(migrations),
                "serving_time_saved_ms": saved_ms,
                "scheduler_p50_ms": scheduler_p50_ms,
                "scheduler_p99_ms": scheduler_p99_ms,
                "overhead_ratio_p50_pct": scheduler_p50_ms / saved_ms * 100,
                "overhead_ratio_p99_pct": scheduler_p99_ms / saved_ms * 100,
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
        [row["total_p99_ms"] for row in rows],
        "o--",
        label="Total scheduler P99",
    )
    scheduler.set_xlabel("Expert migrations per layer")
    scheduler.set_ylabel("Python scheduling time (ms)")
    scheduler.set_title("Scheduler scaling (9 MiB experts)")
    scheduler.grid(True, alpha=0.25)
    scheduler.legend(fontsize=8)

    benefit = axes[1]
    positions = np.arange(len(cost_benefit))
    width = 0.25
    benefit.bar(
        positions - width,
        [row["scheduler_p50_ms"] for row in cost_benefit],
        width,
        label="Scheduler estimate P50",
    )
    benefit.bar(
        positions,
        [row["scheduler_p99_ms"] for row in cost_benefit],
        width,
        label="Scheduler estimate P99",
    )
    benefit.bar(
        positions + width,
        [row["serving_time_saved_ms"] for row in cost_benefit],
        width,
        label="Async serving time saved",
    )
    benefit.set_xticks(positions)
    benefit.set_xticklabels([row["workload"].title() for row in cost_benefit])
    benefit.set_ylabel("Time over the full benchmark (ms)")
    benefit.set_title("Scheduling cost versus serving benefit")
    benefit.grid(True, axis="y", alpha=0.25)
    benefit.legend(fontsize=8)
    for index, row in enumerate(cost_benefit):
        benefit.text(
            index + width,
            row["serving_time_saved_ms"],
            f"  {row['overhead_ratio_p50_pct']:.1f}% P50 ratio",
            rotation=90,
            va="top",
            ha="center",
            fontsize=8,
            color="white",
        )

    figure.suptitle("EPLB batching scheduler overhead")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    rows = sorted(
        (summarize_profile(path) for path in (args.result_dir / "raw").glob("*.json")),
        key=lambda row: row["migrations"],
    )
    if not rows:
        raise RuntimeError(f"No JSON files found in {args.result_dir / 'raw'}")
    write_csv(rows, args.result_dir / "scheduler_scaling.csv")
    cost_benefit = estimate_cost_benefit(rows, args.result_dir)
    write_csv(cost_benefit, args.result_dir / "cost_benefit.csv")
    plot(rows, cost_benefit, args.result_dir / "scheduler_cost_benefit.png")


if __name__ == "__main__":
    main()
