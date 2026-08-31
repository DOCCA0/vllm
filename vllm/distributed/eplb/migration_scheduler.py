# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy scheduling for expert migration instructions during EPLB rebalancing.

The scheduler takes a set of remote expert weight transfers (directed edges
``src_rank -> dst_rank``) and groups them into batches where no rank participates
in more than one rank-pair flow at a time. Transfers between the same source and
destination are coalesced into one flow so that multiple experts can share one
communicator execution. This reduces per-rank NIC/RDMA hot spots without
serializing every expert tensor individually.

The algorithms are intentionally simple (online greedy) so that they can run on
every EPLB rank using only the replicated global expert indices. The offline
ILP solver in ``ILP/eplb_migration_ilp.py`` can be used as a baseline to
quantify how far the greedy schedule is from the optimum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MigrationInstruction:
    """One remote expert weight transfer during EPLB rebalancing."""

    src_rank: int
    dst_rank: int
    expert: int
    layer: int | None = None


def _ranks_with_expert(
    indices: np.ndarray, expert: int, num_local_experts: int
) -> list[int]:
    """Return sorted ranks that hold ``expert`` in the given global index map."""
    positions = np.where(indices == expert)[0]
    return sorted({int(pos // num_local_experts) for pos in positions})


def build_migration_instructions(
    num_local_experts: int,
    old_indices: np.ndarray,
    new_indices: np.ndarray,
    layer: int | None = None,
) -> list[MigrationInstruction]:
    """Build all remote migration edges implied by an old/new expert mapping.

    The output is deterministic and identical on every rank because both
    ``old_indices`` and ``new_indices`` are global arrays replicated across the
    EP group. Each returned instruction represents a single P2P transfer from a
    rank that already holds the expert to a rank that needs it in the new
    mapping. Ranks that already hold the expert locally are excluded from the
    receive side.

    Args:
        num_local_experts: Number of local experts per EP rank.
        old_indices: ``(num_ranks * num_local_experts,)`` global old mapping.
        new_indices: ``(num_ranks * num_local_experts,)`` global new mapping.
        layer: Optional layer index for debugging/telemetry.

    Returns:
        A deterministic list of remote migration instructions.
    """
    assert old_indices.shape == new_indices.shape
    old_flat = old_indices[old_indices != -1]
    new_flat = new_indices[new_indices != -1]
    if old_flat.size == 0 and new_flat.size == 0:
        return []

    experts = np.unique(np.concatenate([old_flat, new_flat]))
    instructions: list[MigrationInstruction] = []

    for expert in sorted(experts.tolist()):
        send_ranks = _ranks_with_expert(old_indices, int(expert), num_local_experts)
        recv_all = _ranks_with_expert(new_indices, int(expert), num_local_experts)
        # Ranks that already hold this expert do not need a remote receive.
        recv_ranks = [r for r in recv_all if r not in send_ranks]
        if not send_ranks or not recv_ranks:
            continue

        n_send = len(send_ranks)
        n_recv = len(recv_ranks)
        num_per = n_recv // n_send
        remainder = n_recv % n_send
        remainder_start = n_send * num_per

        for sender_idx, src in enumerate(send_ranks):
            start = sender_idx * num_per
            end = start + num_per
            assigned = recv_ranks[start:end]
            if sender_idx < remainder:
                assigned.append(recv_ranks[remainder_start + sender_idx])
            for dst in assigned:
                instructions.append(
                    MigrationInstruction(int(src), int(dst), int(expert), layer)
                )

    return instructions


def per_rank_transfer_counts(
    instructions: list[MigrationInstruction],
    world_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Count per-rank send and receive transfers.

    Returns:
        ``(send_counts, recv_counts)``, each of shape ``(world_size,)``.
    """
    send = np.zeros(world_size, dtype=np.int64)
    recv = np.zeros(world_size, dtype=np.int64)
    for inst in instructions:
        send[inst.src_rank] += 1
        recv[inst.dst_rank] += 1
    return send, recv


def schedule_migration_batches(
    instructions: list[MigrationInstruction],
) -> list[list[MigrationInstruction]]:
    """Group migration instructions into conflict-free batches.

    Instructions with the same directed ``(src_rank, dst_rank)`` pair are first
    coalesced into one flow. A batch is conflict-free when no rank appears in
    more than one distinct flow. All instructions in a flow remain in the same
    batch and are submitted in one communicator execution. Scheduling the
    coalesced flows is equivalent to a greedy edge coloring of the rank-pair
    graph.

    Coalesced flows keep their deterministic input order and are placed into
    the first batch without an endpoint conflict.

    Args:
        instructions: Remote migration instructions to schedule. Instructions
            for the same directed rank pair are kept together.

    Returns:
        A list of batches. Each batch is a list of instructions that can be
        executed concurrently without endpoint conflicts.
    """
    if not instructions:
        return []

    flows_by_pair: dict[tuple[int, int], list[MigrationInstruction]] = {}
    for instruction in instructions:
        flows_by_pair.setdefault(
            (instruction.src_rank, instruction.dst_rank), []
        ).append(instruction)
    flows = list(flows_by_pair.values())

    batches: list[list[MigrationInstruction]] = []
    endpoints_used: list[set[int]] = []

    for flow in flows:
        src_rank = flow[0].src_rank
        dst_rank = flow[0].dst_rank
        placed = False
        for batch_idx, used in enumerate(endpoints_used):
            if src_rank not in used and dst_rank not in used:
                batches[batch_idx].extend(flow)
                used.add(src_rank)
                used.add(dst_rank)
                placed = True
                break
        if not placed:
            batches.append(list(flow))
            endpoints_used.append({src_rank, dst_rank})

    return batches


__all__ = [
    "MigrationInstruction",
    "build_migration_instructions",
    "schedule_migration_batches",
]
