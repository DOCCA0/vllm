# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile contention-aware EPLB batching with synthetic NIXL transfers."""

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.eplb.eplb_communicator import NixlEplbCommunicator
from vllm.distributed.eplb.migration_scheduler import (
    build_migration_instructions,
    schedule_migration_batches,
)
from vllm.distributed.eplb.rebalance_execute import move_to_buffer
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
)


class TimedCommunicator:
    """Record each execute call while forwarding the communicator API."""

    def __init__(self, communicator: NixlEplbCommunicator) -> None:
        self.communicator = communicator
        self.execute_ns: list[int] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.communicator, name)

    def execute(self) -> None:
        start_ns = time.perf_counter_ns()
        self.communicator.execute()
        self.execute_ns.append(time.perf_counter_ns() - start_ns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-bytes", type=int, required=True)
    parser.add_argument("--num-local-experts", type=int, default=32)
    parser.add_argument("--num-tensors", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--scheduler-repeats", type=int, default=500)
    parser.add_argument(
        "--migrations-per-flow",
        type=int,
        help=(
            "Move this many experts on each of the 12 directed rank-pair "
            "flows. If omitted, use a seeded random placement."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_placement(
    world_size: int,
    num_local_experts: int,
    seed: int,
    migrations_per_flow: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    num_experts = world_size * num_local_experts
    old_indices = np.arange(num_experts, dtype=np.int64)
    if migrations_per_flow is not None:
        if migrations_per_flow < 1:
            raise ValueError("migrations-per-flow must be positive")
        if migrations_per_flow * (world_size - 1) > num_local_experts:
            raise ValueError("migrations-per-flow exceeds the available slots")
        new_indices = old_indices.copy()
        next_slot = [0] * world_size
        for src_rank in range(world_size):
            for dst_rank in range(src_rank + 1, world_size):
                src_start = src_rank * num_local_experts + next_slot[src_rank]
                dst_start = dst_rank * num_local_experts + next_slot[dst_rank]
                src_slice = slice(src_start, src_start + migrations_per_flow)
                dst_slice = slice(dst_start, dst_start + migrations_per_flow)
                new_indices[src_slice] = old_indices[dst_slice]
                new_indices[dst_slice] = old_indices[src_slice]
                next_slot[src_rank] += migrations_per_flow
                next_slot[dst_rank] += migrations_per_flow
        return old_indices, new_indices

    for candidate_seed in range(seed, seed + 10_000):
        new_indices = np.random.default_rng(candidate_seed).permutation(old_indices)
        instructions = build_migration_instructions(
            num_local_experts, old_indices, new_indices
        )
        flows = {(item.src_rank, item.dst_rank) for item in instructions}
        batches = schedule_migration_batches(instructions)
        if len(flows) == world_size * (world_size - 1) and len(batches) == 6:
            return old_indices, new_indices
    raise RuntimeError("Could not construct a 12-flow, 6-batch placement")


def allocate_expert_tensors(
    num_local_experts: int,
    expert_bytes: int,
    num_tensors: int,
    device: torch.device,
    rank: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if expert_bytes < num_tensors:
        raise ValueError("expert-bytes must be at least num-tensors")
    widths = [expert_bytes // num_tensors] * num_tensors
    for index in range(expert_bytes % num_tensors):
        widths[index] += 1
    weights = [
        torch.full(
            (num_local_experts, width),
            rank,
            dtype=torch.uint8,
            device=device,
        )
        for width in widths
    ]
    buffers = [torch.empty_like(weight) for weight in weights]
    return weights, buffers


def profile_scheduler(
    num_local_experts: int,
    old_indices: np.ndarray,
    new_indices: np.ndarray,
    repeats: int,
) -> dict[str, list[int]]:
    build_ns: list[int] = []
    schedule_ns: list[int] = []
    for _ in range(repeats):
        start_ns = time.perf_counter_ns()
        instructions = build_migration_instructions(
            num_local_experts, old_indices, new_indices
        )
        middle_ns = time.perf_counter_ns()
        schedule_migration_batches(instructions)
        end_ns = time.perf_counter_ns()
        build_ns.append(middle_ns - start_ns)
        schedule_ns.append(end_ns - middle_ns)
    return {"build_ns": build_ns, "schedule_ns": schedule_ns}


def run_transfer(
    *,
    batching: bool,
    num_local_experts: int,
    old_indices: np.ndarray,
    new_indices: np.ndarray,
    rank: int,
    weights: list[torch.Tensor],
    buffers: list[torch.Tensor],
    communicator: TimedCommunicator,
    cpu_group: torch.distributed.ProcessGroup,
) -> list[dict[str, Any]]:
    torch.distributed.barrier(group=cpu_group)
    communicator.execute_ns.clear()
    start_ns = time.perf_counter_ns()
    move_to_buffer(
        num_local_experts=num_local_experts,
        old_indices=old_indices,
        new_indices=new_indices,
        expert_weights=weights,
        expert_weights_buffers=buffers,
        cuda_stream=None,
        ep_rank=rank,
        communicator=communicator,  # type: ignore[arg-type]
        layer_idx=0,
        enable_migration_batching=batching,
    )
    torch.accelerator.synchronize()
    total_ns = time.perf_counter_ns() - start_ns
    local_result = {
        "rank": rank,
        "total_ns": total_ns,
        "execute_ns": communicator.execute_ns.copy(),
    }
    gathered: list[dict[str, Any] | None] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local_result, group=cpu_group)
    return [result for result in gathered if result is not None]


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise ValueError("This profile expects exactly four ranks")

    device = torch.device("cuda", local_rank)
    torch.accelerator.set_device_index(device)
    config = VllmConfig()
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        cpu_group = get_world_group().cpu_group

        old_indices, new_indices = make_placement(
            world_size,
            args.num_local_experts,
            args.seed,
            args.migrations_per_flow,
        )
        instructions = build_migration_instructions(
            args.num_local_experts, old_indices, new_indices
        )
        batches = schedule_migration_batches(instructions)
        scheduler_profile = profile_scheduler(
            args.num_local_experts,
            old_indices,
            new_indices,
            args.scheduler_repeats,
        )
        gathered_scheduler: list[dict[str, list[int]] | None] = [None] * world_size
        torch.distributed.all_gather_object(
            gathered_scheduler, scheduler_profile, group=cpu_group
        )

        weights, buffers = allocate_expert_tensors(
            args.num_local_experts,
            args.expert_bytes,
            args.num_tensors,
            device,
            rank,
        )
        communicator_impl = NixlEplbCommunicator(
            cpu_group,
            all_expert_weights=[weights],
            expert_buffer=buffers,
        )
        communicator = TimedCommunicator(communicator_impl)

        for _ in range(args.warmup):
            run_transfer(
                batching=False,
                num_local_experts=args.num_local_experts,
                old_indices=old_indices,
                new_indices=new_indices,
                rank=rank,
                weights=weights,
                buffers=buffers,
                communicator=communicator,
                cpu_group=cpu_group,
            )
            run_transfer(
                batching=True,
                num_local_experts=args.num_local_experts,
                old_indices=old_indices,
                new_indices=new_indices,
                rank=rank,
                weights=weights,
                buffers=buffers,
                communicator=communicator,
                cpu_group=cpu_group,
            )

        iterations: list[dict[str, Any]] = []
        for repeat in range(args.repeats):
            modes = (False, True) if repeat % 2 == 0 else (True, False)
            for batching in modes:
                per_rank = run_transfer(
                    batching=batching,
                    num_local_experts=args.num_local_experts,
                    old_indices=old_indices,
                    new_indices=new_indices,
                    rank=rank,
                    weights=weights,
                    buffers=buffers,
                    communicator=communicator,
                    cpu_group=cpu_group,
                )
                if rank == 0:
                    iterations.append(
                        {
                            "repeat": repeat,
                            "batching": batching,
                            "per_rank": per_rank,
                        }
                    )

        if rank == 0:
            result = {
                "schema_version": 1,
                "git_revision": git_revision(),
                "world_size": world_size,
                "expert_bytes": args.expert_bytes,
                "num_local_experts": args.num_local_experts,
                "num_tensors": args.num_tensors,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "scheduler_repeats": args.scheduler_repeats,
                "migrations_per_flow": args.migrations_per_flow,
                "num_instructions": len(instructions),
                "num_flows": len(
                    {(item.src_rank, item.dst_rank) for item in instructions}
                ),
                "num_batches": len(batches),
                "old_indices": old_indices.tolist(),
                "new_indices": new_indices.tolist(),
                "scheduler_per_rank": gathered_scheduler,
                "iterations": iterations,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")

        torch.distributed.barrier(group=cpu_group)
        del communicator
        del communicator_impl
        del weights
        del buffers
        gc.collect()
        torch.accelerator.empty_cache()
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
