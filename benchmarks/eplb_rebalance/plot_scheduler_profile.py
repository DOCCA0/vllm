# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot cycle-wide scheduling cost and its serving cost/benefit."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent / "results"
RESULTS = ROOT / "serving_nixl_20260910_cycle_precompute"

comparison = list(csv.DictReader((RESULTS / "profile_comparison.csv").open()))
ranks = np.arange(4)
per_layer_x48 = [float(row["per_layer_times_48_ms"]) for row in comparison]
cycle_wide = [float(row["cycle_wide_call_mean_ms"]) for row in comparison]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
left_width = 0.36
old_bars = axes[0].bar(
    ranks - left_width / 2,
    per_layer_x48,
    left_width,
    color="#65758b",
    label="48 separate per-layer calls",
)
new_bars = axes[0].bar(
    ranks + left_width / 2,
    cycle_wide,
    left_width,
    color="#168878",
    label="One cycle-wide 48-layer call",
)
axes[0].bar_label(old_bars, fmt="%.2f", padding=3, fontsize=8)
axes[0].bar_label(new_bars, fmt="%.2f", padding=3, fontsize=8)
axes[0].set_xticks(ranks, [f"Rank {rank}" for rank in ranks])
axes[0].set_ylabel("Elapsed scheduling time per 48 layers (ms)")
axes[0].set_title("Same scheduler, different call placement")
axes[0].legend(fontsize=8)

workloads = ["Random", "Phased English"]
scheduler_ms = [130.215731, 146.525013]
saved_ms = [19747.495267, 17436.749478]
right_width = 0.36
scheduler_bars = axes[1].bar(
    ranks[:2] - right_width / 2,
    scheduler_ms,
    right_width,
    color="#e19a33",
    label="Measured scheduler cost (max rank)",
)
saved_bars = axes[1].bar(
    ranks[:2] + right_width / 2,
    saved_ms,
    right_width,
    color="#168878",
    label="Measured serving time saved",
)
axes[1].bar_label(scheduler_bars, fmt="%.2f ms", padding=3, fontsize=8)
axes[1].bar_label(saved_bars, fmt="%.2f ms", padding=3, fontsize=8)
for index, (cost, saved) in enumerate(zip(scheduler_ms, saved_ms, strict=True)):
    axes[1].text(
        index,
        saved * 1.08,
        f"cost / saved = {cost / saved * 100:.3f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )
axes[1].set_xticks(ranks[:2], workloads)
axes[1].set_ylabel("Elapsed time (ms)")
axes[1].set_title("Actual calls during full async serving")
axes[1].legend(fontsize=8)

for axis in axes:
    axis.set_ylim(0, axis.get_ylim()[1] * 1.18)

fig.savefig(RESULTS / "scheduler_profile.png", dpi=180)
