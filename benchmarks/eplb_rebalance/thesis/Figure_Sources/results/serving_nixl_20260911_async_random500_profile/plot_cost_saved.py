# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot measured scheduler cost against serving time saved."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

PROFILE_DIR = Path(__file__).parent
E2E_DIR = PROFILE_DIR.parent / "serving_nixl_20260911_async_random500"

with (PROFILE_DIR / "profile_summary.csv").open() as profile_file:
    scheduler_ms = max(
        float(row["cumulative_ms"]) for row in csv.DictReader(profile_file)
    )

with (E2E_DIR / "random_async_off" / "bench.json").open() as result_file:
    duration_off_s = float(json.load(result_file)["duration"])
with (E2E_DIR / "random_async_on" / "bench.json").open() as result_file:
    duration_on_s = float(json.load(result_file)["duration"])

saved_ms = (duration_off_s - duration_on_s) * 1000
ratio = scheduler_ms / saved_ms * 100

fig, axis = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
bars = axis.bar(
    ["Added scheduler cost", "Serving time saved"],
    [scheduler_ms, saved_ms],
    color=["#e19a33", "#168878"],
    width=0.58,
)
axis.set_ylabel("Time (ms)")
axis.set_title("Scheduler cost versus serving time saved")
axis.set_ylim(0, saved_ms * 1.18)
axis.grid(axis="y", alpha=0.22)

axis.annotate(
    f"{scheduler_ms:,.3f} ms",
    xy=(bars[0].get_x() + bars[0].get_width() / 2, scheduler_ms),
    xytext=(0, 14),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=10,
    arrowprops={"arrowstyle": "-", "color": "#555555"},
)
axis.text(
    bars[1].get_x() + bars[1].get_width() / 2,
    saved_ms + saved_ms * 0.025,
    f"{saved_ms:,.0f} ms",
    ha="center",
    va="bottom",
    fontsize=10,
)
axis.text(
    0.5,
    0.92,
    f"Added cost / saved time = {ratio:.3f}%",
    transform=axis.transAxes,
    ha="center",
    fontsize=11,
)

fig.savefig(PROFILE_DIR / "scheduler_cost_vs_saved.png", dpi=180)
