# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy scheduling for expert migration instructions during EPLB rebalancing.

The scheduler takes a set of remote expert weight transfers (directed edges
``src_rank -> dst_rank``) and groups them into batches where no rank participates
in more than one transfer at a time. This reduces per-rank NIC/RDMA hot spots
without changing the final expert placement.

The algorithms are intentionally simple (online greedy) so that they can run on
every EPLB rank using only the replicated global expert indices. The offline
ILP solver in ``ILP/eplb_migration_ilp.py`` can be used as a baseline to
quantify how far the greedy schedule is from the optimum.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Literal

import numpy as np

MigrationOrder = Literal["first_fit", "degree_desc"]


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


def schedule_migrations_greedy(
    instructions: list[MigrationInstruction],
    order: MigrationOrder = "degree_desc",
) -> list[list[MigrationInstruction]]:
    """Group migration instructions into conflict-free batches.

    A batch is conflict-free when no rank appears as the source or destination
    of more than one instruction. This is equivalent to a greedy edge coloring
    of the migration multigraph.

    Two ordering policies are supported:

    - ``first_fit``: keep the input order. This matches the example from
      Omni-infer and is easy to reason about.
    - ``degree_desc``: process edges incident to the highest-degree endpoints
      first. This usually yields fewer batches and is the default.

    Args:
        instructions: Remote migration instructions to schedule.
        order: Greedy ordering policy.

    Returns:
        A list of batches. Each batch is a list of instructions that can be
        executed concurrently without endpoint conflicts.
    """
    if not instructions:
        return []

    if order == "degree_desc":
        degree = collections.Counter[int]()
        for inst in instructions:
            degree[inst.src_rank] += 1
            degree[inst.dst_rank] += 1

        ordered = sorted(
            instructions,
            key=lambda inst: (
                -degree[inst.src_rank] - degree[inst.dst_rank],
                -degree[inst.src_rank],
                -degree[inst.dst_rank],
                inst.src_rank,
                inst.dst_rank,
                inst.expert,
            ),
        )
    else:
        ordered = list(instructions)

    batches: list[list[MigrationInstruction]] = []
    endpoints_used: list[set[int]] = []

    for inst in ordered:
        placed = False
        for batch_idx, used in enumerate(endpoints_used):
            if inst.src_rank not in used and inst.dst_rank not in used:
                batches[batch_idx].append(inst)
                used.add(inst.src_rank)
                used.add(inst.dst_rank)
                placed = True
                break
        if not placed:
            batches.append([inst])
            endpoints_used.append({inst.src_rank, inst.dst_rank})

    return batches


__all__ = [
    "MigrationInstruction",
    "MigrationOrder",
    "build_migration_instructions",
    "schedule_migrations_greedy",
]
