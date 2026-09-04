# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize and plot the four-node NIXL migration profile."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    return parser.parse_args()


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values), value))


def summarize(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    build_ms: list[float] = []
    greedy_ms: list[float] = []
    for rank in result["scheduler_per_rank"]:
        build_ms.extend(value / 1e6 for value in rank["build_ns"])
        greedy_ms.extend(value / 1e6 for value in rank["schedule_ns"])

    totals: dict[bool, dict[int, float]] = {False: {}, True: {}}
    execute_totals: dict[bool, list[float]] = {False: [], True: []}
    for iteration in result["iterations"]:
        totals[iteration["batching"]][iteration["repeat"]] = (
            max(rank["total_ns"] for rank in iteration["per_rank"]) / 1e6
        )
        execute_totals[iteration["batching"]].append(
            max(sum(rank["execute_ns"]) for rank in iteration["per_rank"]) / 1e6
        )
    repeat_ids = sorted(totals[False])
    off_ms = [totals[False][repeat] for repeat in repeat_ids]
    on_ms = [totals[True][repeat] for repeat in repeat_ids]
    slowdown_pct = [
        (on_value - off_value) / off_value * 100
        for off_value, on_value in zip(off_ms, on_ms)
    ]
    scheduler_ms = [
        build_value + greedy_value
        for build_value, greedy_value in zip(build_ms, greedy_ms)
    ]
    off_p50 = percentile(off_ms, 50)
    return {
        "expert_mib": result["expert_bytes"] / 2**20,
        "instructions": result["num_instructions"],
        "flows": result["num_flows"],
        "batches": result["num_batches"],
        "build_p50_ms": percentile(build_ms, 50),
        "build_p99_ms": percentile(build_ms, 99),
        "greedy_p50_ms": percentile(greedy_ms, 50),
        "greedy_p99_ms": percentile(greedy_ms, 99),
        "scheduler_p50_ms": percentile(scheduler_ms, 50),
        "scheduler_p99_ms": percentile(scheduler_ms, 99),
        "scheduler_share_pct": percentile(scheduler_ms, 50) / off_p50 * 100,
        "off_p50_ms": off_p50,
        "off_p99_ms": percentile(off_ms, 99),
        "on_p50_ms": percentile(on_ms, 50),
        "on_p99_ms": percentile(on_ms, 99),
        "off_execute_p50_ms": percentile(execute_totals[False], 50),
        "on_execute_p50_ms": percentile(execute_totals[True], 50),
        "slowdown_p50_pct": percentile(slowdown_pct, 50),
        "slowdown_p99_pct": percentile(slowdown_pct, 99),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def serving_improvements(result_dir: Path) -> dict[str, list[float]]:
    serving_dir = result_dir.parent / "nixl_cache_on_20260901"
    metric_keys = [
        "output_throughput",
        "p50_ttft_ms",
        "p99_ttft_ms",
        "p50_tpot_ms",
        "p99_tpot_ms",
        "p50_e2el_ms",
        "p99_e2el_ms",
    ]
    improvements: dict[str, list[float]] = {}
    for workload in ("random", "phased"):
        off = json.loads(
            (serving_dir / workload / "03_async_off" / "bench.json").read_text()
        )
        on = json.loads(
            (serving_dir / workload / "04_async_batching_on" / "bench.json").read_text()
        )
        values = [(on[metric_keys[0]] / off[metric_keys[0]] - 1) * 100]
        values.extend((off[key] - on[key]) / off[key] * 100 for key in metric_keys[1:])
        improvements[workload] = values
    return improvements


def plot(rows: list[dict[str, Any]], result_dir: Path, path: Path) -> None:
    sizes = np.asarray([row["expert_mib"] for row in rows])
    build_p50 = np.asarray([row["build_p50_ms"] for row in rows])
    greedy_p50 = np.asarray([row["greedy_p50_ms"] for row in rows])
    scheduler_p99 = np.asarray([row["scheduler_p99_ms"] for row in rows])
    serving = serving_improvements(result_dir)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    scheduler = axes[0]
    scheduler.plot(sizes, build_p50, "o-", label="Instruction build P50")
    scheduler.plot(sizes, greedy_p50, "o-", label="Greedy grouping P50")
    scheduler.plot(sizes, scheduler_p99, "o--", label="Total scheduler P99")
    scheduler.set_xscale("log", base=2)
    scheduler.set_yscale("log")
    scheduler.set_xlabel("Payload per expert (MiB)")
    scheduler.set_ylabel("Python time (ms)")
    scheduler.set_title("Scheduling stays near 1.2 ms per layer")
    scheduler.grid(True, which="both", alpha=0.25)
    scheduler.legend(fontsize=8)

    serving_axis = axes[1]
    labels = [
        "Throughput",
        "TTFT P50",
        "TTFT P99",
        "TPOT P50",
        "TPOT P99",
        "E2EL P50",
        "E2EL P99",
    ]
    positions = np.arange(len(labels))
    width = 0.36
    serving_axis.bar(
        positions - width / 2,
        serving["random"],
        width,
        label="Random",
    )
    serving_axis.bar(
        positions + width / 2,
        serving["phased"],
        width,
        label="Phased English",
    )
    serving_axis.axhline(0, color="black", linewidth=0.8)
    serving_axis.set_xticks(positions)
    serving_axis.set_xticklabels(labels, rotation=35, ha="right")
    serving_axis.set_ylabel("Improvement with batching (%)")
    serving_axis.set_title("Measured async serving result")
    serving_axis.grid(True, axis="y", alpha=0.25)
    serving_axis.legend(fontsize=8)

    size_labels = [
        "4 KiB",
        "16 KiB",
        "64 KiB",
        "256 KiB",
        "1 MiB",
        "4 MiB",
        "9 MiB",
        "16 MiB",
        "64 MiB",
    ]
    scheduler.set_xticks(sizes)
    scheduler.set_xticklabels(size_labels, rotation=35, ha="right")
    scheduler.axvline(9, color="black", linestyle=":", alpha=0.7)
    qwen_index = list(sizes).index(9)
    scheduler.annotate(
        "Qwen3-30B: 1.202/1.329 ms P50/P99",
        xy=(9, build_p50[qwen_index]),
        xytext=(20, -55),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )
    figure.suptitle("EPLB scheduling overhead and async serving impact")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    raw_dir = args.result_dir / "raw"
    rows = sorted(
        (summarize(path) for path in raw_dir.glob("*.json")),
        key=lambda row: row["expert_mib"],
    )
    if not rows:
        raise RuntimeError(f"No profile JSON files found in {raw_dir}")
    write_csv(rows, args.result_dir / "summary.csv")
    plot(rows, args.result_dir, args.result_dir / "migration_profile.png")


if __name__ == "__main__":
    main()
