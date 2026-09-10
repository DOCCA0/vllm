# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot measured cycle costs and their amortized per-layer means."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent / "results"
SOURCE = ROOT / "scheduler_exploration_20260910/dense_comparison"
DEST = ROOT / "serving_trace_20260910"
rows = list(csv.DictReader((SOURCE / "bulk_serving_comparison.csv").open()))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
colors = ["#65758b", "#168878", "#e19a33"]
for i, (label, group) in enumerate(
    [
        ("Per-layer scheduling", rows[:4]),
        ("Bulk run 1", rows[4:8]),
        ("Bulk run 2", rows[8:]),
    ]
):
    x = np.arange(4) + (i - 1) * 0.26
    values = [float(r["normalized_ms_per_layer"]) for r in group]
    bars = axes[0].bar(x, values, 0.25, label=label, color=colors[i])
    axes[0].bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
for i, group in enumerate([rows[4:8], rows[8:]]):
    values = [float(r["total_ms"]) / int(r["nvtx_calls"]) for r in group]
    bars = axes[1].bar(
        np.arange(4) + (i - 0.5) * 0.35,
        values,
        0.34,
        label=f"Bulk run {i + 1}",
        color=colors[i + 1],
    )
    axes[1].bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
for ax in axes:
    ax.set_xticks(range(4), [f"Rank {i}" for i in range(4)])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("Mean elapsed scheduling time (ms / built layer)")
axes[0].set_title("Amortized cost — not per-call P50")
axes[1].set_ylabel("Mean elapsed time (ms / 48-layer call)")
axes[1].set_title("Full batch cost remains visible")
fig.savefig(DEST / "bulk_scheduler.png", dpi=180)
