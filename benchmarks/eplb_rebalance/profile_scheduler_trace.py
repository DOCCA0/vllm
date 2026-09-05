# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create a PyTorch trace of EPLB migration flow scheduling."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vllm.distributed.eplb.migration_scheduler import (
    MigrationFlow,
    schedule_migration_batches,
)


@dataclass(frozen=True)
class ReferenceInstruction:
    src_rank: int
    dst_rank: int
    expert_id: int


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
) -> list[ReferenceInstruction]:
    """Build individual transfers before grouping them by rank pair."""
    old_experts = old_indices[old_indices != -1]
    new_experts = new_indices[new_indices != -1]
    if old_experts.size == 0 and new_experts.size == 0:
        return []

    expert_ids = np.unique(np.concatenate((old_experts, new_experts)))
    instructions: list[ReferenceInstruction] = []
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
                ReferenceInstruction(src_rank, dst_rank, int(expert_id))
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


def schedule_reference(
    instructions: list[ReferenceInstruction],
) -> list[list[ReferenceInstruction]]:
    flows_by_pair: dict[tuple[int, int], list[ReferenceInstruction]] = {}
    for instruction in instructions:
        key = (instruction.src_rank, instruction.dst_rank)
        flows_by_pair.setdefault(key, []).append(instruction)

    batches: list[list[ReferenceInstruction]] = []
    endpoints_used: list[set[int]] = []
    for flow in flows_by_pair.values():
        src_rank = flow[0].src_rank
        dst_rank = flow[0].dst_rank
        for batch, used in zip(batches, endpoints_used):
            if src_rank not in used and dst_rank not in used:
                batch.extend(flow)
                used.update((src_rank, dst_rank))
                break
        else:
            batches.append(list(flow))
            endpoints_used.append({src_rank, dst_rank})
    return batches


def flatten_flows(batches: list[list[MigrationFlow]]) -> list[tuple[int, int, int]]:
    return [
        (flow.src_rank, flow.dst_rank, expert_id)
        for batch in batches
        for flow in batch
        for expert_id in flow.expert_ids
    ]


def flatten_reference(
    batches: list[list[ReferenceInstruction]],
) -> list[tuple[int, int, int]]:
    return [
        (item.src_rank, item.dst_rank, item.expert_id)
        for batch in batches
        for item in batch
    ]


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
            reference = schedule_reference(
                build_reference(num_local, old_indices, new_indices)
            )
            fused = schedule_migration_batches(num_local, old_indices, new_indices)
            assert flatten_reference(reference) == flatten_flows(fused)

    activities = [torch.profiler.ProfilerActivity.CPU]
    expected = {
        migrations: flatten_reference(
            schedule_reference(build_reference(num_local, old_indices, new_indices))
        )
        for migrations, num_local, old_indices, new_indices in placements
    }
    with torch.profiler.profile(activities=activities) as profiler:
        for migrations, num_local, old_indices, new_indices in placements:
            for _ in range(args.repeats):
                with torch.profiler.record_function(f"fused.total.{migrations}"):
                    fused = schedule_migration_batches(
                        num_local, old_indices, new_indices
                    )
                assert expected[migrations] == flatten_flows(fused)

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(args.trace))

    durations: dict[str, list[float]] = {}
    for event in profiler.events():
        if event.name.startswith("fused."):
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
