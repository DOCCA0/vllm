# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create a PyTorch trace comparing EPLB scheduler implementations."""

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from vllm.distributed.eplb.migration_scheduler import (
    MigrationInstruction,
    build_migration_instructions,
    schedule_migration_batches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("placements_dir", type=Path)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def build_reference(
    num_local_experts: int,
    old_indices: np.ndarray,
    new_indices: np.ndarray,
) -> list[MigrationInstruction]:
    """Previous per-expert scan implementation used as the trace baseline."""
    old_experts = old_indices[old_indices != -1]
    new_experts = new_indices[new_indices != -1]
    if old_experts.size == 0 and new_experts.size == 0:
        return []

    expert_ids = np.unique(np.concatenate((old_experts, new_experts)))
    instructions: list[MigrationInstruction] = []
    for expert_id in expert_ids.tolist():
        send_positions = np.flatnonzero(old_indices == expert_id)
        send_ranks = sorted(
            {int(position // num_local_experts) for position in send_positions}
        )
        recv_positions = np.flatnonzero(new_indices == expert_id)
        recv_ranks = [
            rank
            for rank in sorted(
                {int(position // num_local_experts) for position in recv_positions}
            )
            if rank not in send_ranks
        ]
        if not send_ranks or not recv_ranks:
            continue

        num_per_sender, remainder = divmod(len(recv_ranks), len(send_ranks))
        remainder_start = len(send_ranks) * num_per_sender
        for sender_idx, src_rank in enumerate(send_ranks):
            start = sender_idx * num_per_sender
            assigned_ranks = recv_ranks[start : start + num_per_sender]
            if sender_idx < remainder:
                assigned_ranks.append(recv_ranks[remainder_start + sender_idx])
            instructions.extend(
                MigrationInstruction(src_rank, dst_rank, int(expert_id))
                for dst_rank in assigned_ranks
            )
    return instructions


def load_placements(path: Path) -> tuple[int, int, np.ndarray, np.ndarray]:
    result = json.loads(path.read_text())
    return (
        result["num_instructions"],
        result["num_local_experts"],
        np.asarray(result["old_indices"], dtype=np.int64),
        np.asarray(result["new_indices"], dtype=np.int64),
    )


def run_build(
    build: Callable[[int, np.ndarray, np.ndarray], list[MigrationInstruction]],
    num_local_experts: int,
    old_indices: np.ndarray,
    new_indices: np.ndarray,
) -> list[MigrationInstruction]:
    return build(num_local_experts, old_indices, new_indices)


def main() -> None:
    args = parse_args()
    placements = sorted(
        (load_placements(path) for path in args.placements_dir.glob("*.json")),
        key=lambda item: item[0],
    )
    if not placements:
        raise RuntimeError(f"No placement JSON found in {args.placements_dir}")

    for _, num_local, old_indices, new_indices in placements:
        for _ in range(args.warmup):
            before = run_build(build_reference, num_local, old_indices, new_indices)
            after = run_build(
                build_migration_instructions, num_local, old_indices, new_indices
            )
            assert before == after
            schedule_migration_batches(after)

    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(activities=activities) as profiler:
        for migrations, num_local, old_indices, new_indices in placements:
            for _ in range(args.repeats):
                with torch.profiler.record_function(f"before.build.{migrations}"):
                    before = run_build(
                        build_reference, num_local, old_indices, new_indices
                    )
                with torch.profiler.record_function(f"after.build.{migrations}"):
                    after = run_build(
                        build_migration_instructions,
                        num_local,
                        old_indices,
                        new_indices,
                    )
                assert before == after
                with torch.profiler.record_function(f"after.greedy.{migrations}"):
                    schedule_migration_batches(after)

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(args.trace))

    durations: dict[str, list[float]] = {}
    for event in profiler.events():
        if event.name.startswith(("before.", "after.")):
            durations.setdefault(event.name, []).append(event.cpu_time_total / 1000)

    with args.summary.open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("event", "calls", "p50_ms", "p99_ms"))
        for name, values in sorted(durations.items()):
            writer.writerow(
                (
                    name,
                    len(values),
                    float(np.percentile(values, 50)),
                    float(np.percentile(values, 99)),
                )
            )


if __name__ == "__main__":
    main()
