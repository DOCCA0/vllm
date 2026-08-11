# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.distributed.eplb.eplb_communicator import EplbCommunicator
from vllm.distributed.eplb.migration_scheduler import (
    MigrationInstruction,
    build_migration_instructions,
    per_rank_transfer_counts,
    schedule_migrations_greedy,
)
from vllm.distributed.eplb.rebalance_execute import move_to_buffer


class _MockEplbCommunicator(EplbCommunicator):
    """Minimal communicator that records P2P calls and counts execute waves."""

    def __init__(self):
        self.send_calls: list[tuple[int, torch.Tensor]] = []
        self.recv_calls: list[tuple[int, torch.Tensor]] = []
        self.execute_count = 0

    def add_send(self, tensor: torch.Tensor, dst_rank: int) -> None:
        self.send_calls.append((dst_rank, tensor))

    def add_recv(self, tensor: torch.Tensor, src_rank: int) -> None:
        self.recv_calls.append((src_rank, tensor))

    def execute(self) -> None:
        self.execute_count += 1


def test_migration_batching_first_fit_order():
    instructions = [
        MigrationInstruction(1, 3, expert=0),
        MigrationInstruction(2, 4, expert=1),
        MigrationInstruction(0, 1, expert=2),
        MigrationInstruction(0, 2, expert=3),
        MigrationInstruction(0, 3, expert=4),
        MigrationInstruction(0, 4, expert=5),
    ]

    first_fit_batches = schedule_migrations_greedy(instructions)
    # first_fit starts with the two disjoint edges, leaving only one slot per
    # batch for a rank-0 edge, so the (0,4) edge ends up in a 5th batch.
    assert len(first_fit_batches) == 5
    assert _as_endpoints(first_fit_batches[0]) == [(1, 3), (2, 4)]
    assert _as_endpoints(first_fit_batches[1]) == [(0, 1)]
    assert _as_endpoints(first_fit_batches[2]) == [(0, 2)]
    assert _as_endpoints(first_fit_batches[3]) == [(0, 3)]
    assert _as_endpoints(first_fit_batches[4]) == [(0, 4)]
    for batch in first_fit_batches:
        _assert_no_endpoint_conflict(batch)


def test_migration_batching_coalesces_same_rank_pair():
    instructions = [
        MigrationInstruction(0, 1, expert=0),
        MigrationInstruction(0, 1, expert=1),
        MigrationInstruction(2, 3, expert=2),
    ]

    batches = schedule_migrations_greedy(instructions)
    assert len(batches) == 1
    assert batches[0] == instructions


def test_build_migration_instructions_excludes_local_copies():
    # 2 ranks, 2 local experts. Old: rank0 [0,1], rank1 [1,2].
    # New:  rank0 [0,1], rank1 [0,2]. Only rank1 needs expert 0 and rank0 can send it.
    old = np.array([0, 1, 1, 2], dtype=np.int64)
    new = np.array([0, 1, 0, 2], dtype=np.int64)
    instructions = build_migration_instructions(2, old, new)
    assert len(instructions) == 1
    assert instructions[0] == MigrationInstruction(0, 1, expert=0)


def test_build_migration_instructions_with_remote_moves():
    # 2 ranks, 2 local experts. Old: rank0 [0,1], rank1 [2,3].
    # New:  rank0 [0,2], rank1 [1,3]. Two remote swaps.
    old = np.array([0, 1, 2, 3], dtype=np.int64)
    new = np.array([0, 2, 1, 3], dtype=np.int64)
    instructions = build_migration_instructions(2, old, new)
    assert set(_as_endpoints(instructions)) == {(0, 1), (1, 0)}


def test_greedy_batches_no_conflicts_and_cover_all():
    np.random.seed(42)
    for _ in range(20):
        instructions = [
            MigrationInstruction(
                int(np.random.randint(0, 8)),
                int(np.random.randint(0, 8)),
                expert=i,
            )
            for i in range(40)
            if np.random.randint(0, 8) != np.random.randint(0, 8)
        ]
        # Remove possible self-loops deterministically by re-generation.
        instructions = [inst for inst in instructions if inst.src_rank != inst.dst_rank]
        if not instructions:
            continue

        batches = schedule_migrations_greedy(instructions)
        scheduled = [inst for batch in batches for inst in batch]
        assert len(scheduled) == len(instructions)
        assert set(scheduled) == set(instructions)
        for batch in batches:
            _assert_no_endpoint_conflict(batch)


def test_build_migration_instructions_empty():
    old = np.array([-1, -1, -1, -1], dtype=np.int64)
    new = np.array([-1, -1, -1, -1], dtype=np.int64)
    assert build_migration_instructions(2, old, new) == []


def test_per_rank_transfer_counts():
    instructions = [
        MigrationInstruction(0, 1, expert=0),
        MigrationInstruction(0, 2, expert=1),
        MigrationInstruction(0, 3, expert=2),
        MigrationInstruction(1, 2, expert=3),
    ]
    send, recv = per_rank_transfer_counts(instructions, world_size=4)
    assert send.tolist() == [3, 1, 0, 0]
    assert recv.tolist() == [0, 1, 2, 1]
    total = send + recv
    # Rank 0 participates in 3 of 4 transfers: clear hot spot.
    assert total.max() == 3
    assert total.sum() == 2 * len(instructions)


def test_per_rank_transfer_counts_empty():
    send, recv = per_rank_transfer_counts([], world_size=3)
    assert send.tolist() == [0, 0, 0]
    assert recv.tolist() == [0, 0, 0]


def test_move_to_buffer_batched():
    # 4 ranks, 1 local expert each. The new mapping rotates the experts so
    # every rank both sends and receives. The cycle needs 2 conflict-free
    # batches, so communicator.execute() should be called twice.
    old = np.array([0, 1, 2, 3], dtype=np.int64)
    new = np.array([1, 2, 3, 0], dtype=np.int64)
    ep_rank = 0
    weight = torch.zeros(1, 1)
    buffer = torch.zeros(1, 1)
    communicator = _MockEplbCommunicator()

    move_to_buffer(
        num_local_experts=1,
        old_indices=old,
        new_indices=new,
        expert_weights=[weight],
        expert_weights_buffers=[buffer],
        cuda_stream=None,
        ep_rank=ep_rank,
        communicator=communicator,
        migration_batching=True,
    )

    assert communicator.execute_count == 2
    # Rank 0 sends expert 0 to rank 3 and receives expert 1 from rank 1.
    assert len(communicator.send_calls) == 1
    assert communicator.send_calls[0][0] == 3
    assert len(communicator.recv_calls) == 1
    assert communicator.recv_calls[0][0] == 1


def _as_endpoints(instructions):
    return [(inst.src_rank, inst.dst_rank) for inst in instructions]


def _assert_no_endpoint_conflict(batch):
    used = set()
    pairs = {(inst.src_rank, inst.dst_rank) for inst in batch}
    for src_rank, dst_rank in pairs:
        assert src_rank not in used, f"src {src_rank} reused"
        assert dst_rank not in used, f"dst {dst_rank} reused"
        used.add(src_rank)
        used.add(dst_rank)


def test_move_to_buffer_max_batches_warning():
    # 4 ranks, 1 local expert each. rank 0 holds expert 0 and all ranks want it,
    # so the remote sends from rank 0 need 3 conflict-free batches.
    # With max_batches=2 this triggers the soft-limit warning.
    old = np.array([0, 1, 2, 3], dtype=np.int64)
    new = np.array([0, 0, 0, 0], dtype=np.int64)
    ep_rank = 0
    weight = torch.zeros(1, 1)
    buffer = torch.zeros(1, 1)
    communicator = _MockEplbCommunicator()

    with patch(
        "vllm.distributed.eplb.rebalance_execute.logger.warning"
    ) as mock_warning:
        move_to_buffer(
            num_local_experts=1,
            old_indices=old,
            new_indices=new,
            expert_weights=[weight],
            expert_weights_buffers=[buffer],
            cuda_stream=None,
            ep_rank=ep_rank,
            communicator=communicator,
            migration_batching=True,
            migration_batching_max_batches=2,
        )

    assert communicator.execute_count == 3
    assert len(communicator.send_calls) == 3
    assert set(dst for dst, _ in communicator.send_calls) == {1, 2, 3}
    mock_warning.assert_called_once()
    args, _ = mock_warning.call_args
    assert "exceeding the soft limit" in args[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
