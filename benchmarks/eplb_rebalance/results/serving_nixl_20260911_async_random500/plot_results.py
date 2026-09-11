# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
CASES = {
    "Before: batching off": ROOT / "random_async_off",
    "After: batching on": ROOT / "random_async_on",
}


def load_bench(case: Path) -> dict[str, float]:
    with (case / "bench.json").open() as file:
        return json.load(file)


def load_nic(case: Path) -> tuple[np.ndarray, np.ndarray]:
    samples = np.loadtxt(case / "nic.tsv", dtype=np.float64)
    elapsed = samples[1:, 0] - samples[1, 0]
    combined_bytes = samples[:, 1] + samples[:, 2]
    rate = np.diff(combined_bytes) / np.diff(samples[:, 0]) / 1_000_000
    return elapsed, rate


off = load_bench(CASES["Before: batching off"])
on = load_bench(CASES["After: batching on"])

metrics = [
    ("Output\nthroughput", "output_throughput", True),
    ("TTFT P50", "p50_ttft_ms", False),
    ("TTFT P99", "p99_ttft_ms", False),
    ("TPOT P50", "p50_tpot_ms", False),
    ("TPOT P99", "p99_tpot_ms", False),
    ("E2EL P50", "p50_e2el_ms", False),
    ("E2EL P99", "p99_e2el_ms", False),
]
improvements = [
    (on[key] / off[key] - 1) * 100
    if higher_is_better
    else (off[key] - on[key]) / off[key] * 100
    for _, key, higher_is_better in metrics
]

fig, ax = plt.subplots(figsize=(11.5, 4.8))
colors = ["#2f6fed"] + ["#34a853"] * 6
bars = ax.bar([label for label, _, _ in metrics], improvements, color=colors)
ax.axhline(0, color="#333333", linewidth=0.8)
ax.set_ylabel("Improvement with batching (%)")
ax.set_title("Async serving: batching on versus off")
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)
for bar, value in zip(bars, improvements):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.07,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )
fig.tight_layout()
fig.savefig(ROOT / "e2e_improvement.png", dpi=180)
plt.close(fig)

nic_data = {label: load_nic(case) for label, case in CASES.items()}
max_x = max(values[0][-1] for values in nic_data.values())
max_y = max(np.max(values[1]) for values in nic_data.values())
fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, sharey=True)
for ax, (label, (elapsed, rate)) in zip(axes, nic_data.items()):
    p99 = float(np.percentile(rate, 99))
    color = "#7f8c8d" if label.startswith("Before") else "#2f6fed"
    ax.plot(elapsed, rate, color=color, linewidth=0.55, alpha=0.85)
    ax.axhline(p99, color="#d93025", linestyle="--", linewidth=1.2)
    ax.text(
        max_x * 0.99,
        p99 + max_y * 0.025,
        f"P99 {p99:.2f} MB/s",
        ha="right",
        va="bottom",
        color="#b3261e",
    )
    ax.set_title(label, loc="left", fontweight="bold")
    ax.set_ylabel("NIC RX + TX (MB/s)")
    ax.grid(alpha=0.2)
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y * 1.05)
axes[-1].set_xlabel("Elapsed benchmark time (s)")
fig.suptitle("Head-node NIC traffic, one-second samples", y=0.995)
fig.tight_layout()
fig.savefig(ROOT / "nic_timeseries.png", dpi=180)
plt.close(fig)

with (ROOT / "e2e_summary.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["metric", "batching_off", "batching_on", "improvement_percent"])
    for (label, key, _), improvement in zip(metrics, improvements):
        writer.writerow([label.replace("\n", " "), off[key], on[key], improvement])

with (ROOT / "nic_summary.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["case", "p99_mb_s"])
    for label, (_, rate) in nic_data.items():
        writer.writerow([label, float(np.percentile(rate, 99))])
