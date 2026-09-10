# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare experimental schedulers against the current exact schedule."""

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from vllm.distributed.eplb.migration_scheduler import (
    MigrationFlow,
    schedule_migration_batches,
)


@lru_cache(maxsize=4096)
def assignment(source_mask, destination_mask):
    sources = [r for r in range(source_mask.bit_length()) if source_mask >> r & 1]
    destinations = [
        r for r in range(destination_mask.bit_length()) if destination_mask >> r & 1
    ]
    if not sources:
        return ()
    count, remainder = divmod(len(destinations), len(sources))
    pairs = []
    for index, source in enumerate(sources):
        for destination in destinations[index * count : (index + 1) * count]:
            pairs.append((source, destination))
        if index < remainder:
            pairs.append((source, destinations[len(sources) * count + index]))
    return tuple(pairs)


@lru_cache(maxsize=4096)
def grouping(pairs):
    masks, groups = [], []
    for index, (source, destination) in enumerate(pairs):
        mask = (1 << source) | (1 << destination)
        for batch, used in enumerate(masks):
            if not used & mask:
                masks[batch] |= mask
                groups[batch].append(index)
                break
        else:
            masks.append(mask)
            groups.append([index])
    return tuple(tuple(group) for group in groups)


@lru_cache(maxsize=32)
def fixed_pair_batches(ranks):
    """Map every directed rank pair to a precomputed conflict-free batch."""
    pairs = tuple(
        (src, dst) for src in range(ranks) for dst in range(ranks) if src != dst
    )
    groups = grouping(pairs)
    return {
        pairs[index]: batch for batch, group in enumerate(groups) for index in group
    }


def masked(num_local, old, new, cache_groups=True):
    source_masks, destination_masks = {}, {}
    for rank, (old_local, new_local) in enumerate(
        zip(old.reshape(-1, num_local), new.reshape(-1, num_local))
    ):
        bit = 1 << rank
        old_set = set(old_local.tolist()) - {-1}
        for expert in old_set:
            source_masks[expert] = source_masks.get(expert, 0) | bit
        for expert in set(new_local.tolist()) - old_set - {-1}:
            destination_masks[expert] = destination_masks.get(expert, 0) | bit
    flows = {}
    for expert in sorted(destination_masks):
        for pair in assignment(source_masks.get(expert, 0), destination_masks[expert]):
            flows.setdefault(pair, []).append(expert)
    pairs = tuple(flows)
    groups = grouping(pairs) if cache_groups else grouping.__wrapped__(pairs)
    values = [MigrationFlow(*pair, tuple(experts)) for pair, experts in flows.items()]
    return [[values[index] for index in group] for group in groups]


def masked_fixed(num_local, old, new):
    """Use rank masks and a static pair-to-batch mapping instead of greedy grouping."""
    source_masks, destination_masks = {}, {}
    ranks = old.size // num_local
    for rank, (old_local, new_local) in enumerate(
        zip(old.reshape(-1, num_local), new.reshape(-1, num_local))
    ):
        bit = 1 << rank
        old_set = set(old_local.tolist()) - {-1}
        for expert in old_set:
            source_masks[expert] = source_masks.get(expert, 0) | bit
        for expert in set(new_local.tolist()) - old_set - {-1}:
            destination_masks[expert] = destination_masks.get(expert, 0) | bit
    flows = {}
    for expert in sorted(destination_masks):
        for pair in assignment(source_masks.get(expert, 0), destination_masks[expert]):
            flows.setdefault(pair, []).append(expert)
    pair_batches = fixed_pair_batches(ranks)
    batches = [[] for _ in range(max(pair_batches.values(), default=-1) + 1)]
    for pair, experts in flows.items():
        batches[pair_batches[pair]].append(MigrationFlow(*pair, tuple(experts)))
    return [batch for batch in batches if batch]


def dense(num_local, old, new):
    """Vectorize membership; use compact IDs only, otherwise fall back."""
    ranks = old.size // num_local
    largest = int(max(old.max(), new.max()))
    if ranks > 63 or largest > 65536:
        return masked(num_local, old, new)
    old_masks = np.zeros(largest + 2, dtype=np.uint64)
    new_masks = np.zeros_like(old_masks)
    bits = np.repeat(
        np.left_shift(np.uint64(1), np.arange(ranks, dtype=np.uint64)), num_local
    )
    # -1 is a sentinel slot, removed below. OR handles duplicate replicas.
    np.bitwise_or.at(old_masks, old, bits)
    np.bitwise_or.at(new_masks, new, bits)
    needed = new_masks & ~old_masks
    needed[-1] = 0
    experts = np.flatnonzero(needed)
    flows = {}
    for expert, source, destination in zip(
        experts.tolist(), old_masks[experts].tolist(), needed[experts].tolist()
    ):
        for pair in assignment(source, destination):
            flows.setdefault(pair, []).append(expert)
    groups = grouping(tuple(flows))
    values = [MigrationFlow(*pair, tuple(ids)) for pair, ids in flows.items()]
    return [[values[index] for index in group] for group in groups]


def _batches_from_masks(old_masks, needed_masks):
    experts = np.flatnonzero(needed_masks)
    flows = {}
    for expert, source, destination in zip(
        experts.tolist(),
        old_masks[experts].tolist(),
        needed_masks[experts].tolist(),
    ):
        for pair in assignment(source, destination):
            flows.setdefault(pair, []).append(expert)
    groups = grouping(tuple(flows))
    values = [MigrationFlow(*pair, tuple(ids)) for pair, ids in flows.items()]
    return [[values[index] for index in group] for group in groups]


def dense_all(num_local, old, new):
    """Vectorize rank-membership construction across all MoE layers."""
    layers, positions = old.shape
    ranks = positions // num_local
    largest = int(max(old.max(), new.max()))
    if ranks > 63 or largest > 65536:
        return [masked(num_local, a, b) for a, b in zip(old, new)]
    old_masks = np.zeros((layers, largest + 2), dtype=np.uint64)
    new_masks = np.zeros_like(old_masks)
    layer_ids = np.repeat(np.arange(layers), positions)
    rank_bits = np.repeat(
        np.left_shift(np.uint64(1), np.arange(ranks, dtype=np.uint64)), num_local
    )
    bits = np.tile(rank_bits, layers)
    np.bitwise_or.at(old_masks, (layer_ids, old.ravel()), bits)
    np.bitwise_or.at(new_masks, (layer_ids, new.ravel()), bits)
    needed = new_masks & ~old_masks
    needed[:, -1] = 0
    return [_batches_from_masks(a, b) for a, b in zip(old_masks, needed)]


def python_dense(num_local, old, new):
    """Build compact expert-to-rank masks without NumPy temporary arrays."""
    ranks = old.size // num_local
    largest = int(max(old.max(), new.max()))
    if ranks > 63 or largest > 65536:
        return masked(num_local, old, new)
    old_masks = [0] * (largest + 1)
    new_masks = [0] * (largest + 1)
    for rank in range(ranks):
        bit = 1 << rank
        start = rank * num_local
        stop = start + num_local
        for expert in old[start:stop]:
            if expert >= 0:
                old_masks[int(expert)] |= bit
        for expert in new[start:stop]:
            if expert >= 0:
                new_masks[int(expert)] |= bit
    flows = {}
    for expert, (source_mask, new_mask) in enumerate(zip(old_masks, new_masks)):
        destination_mask = new_mask & ~source_mask
        if destination_mask:
            for pair in assignment(source_mask, destination_mask):
                flows.setdefault(pair, []).append(expert)
    groups = grouping(tuple(flows))
    values = [MigrationFlow(*pair, tuple(ids)) for pair, ids in flows.items()]
    return [[values[index] for index in group] for group in groups]


def main():
    root = Path(__file__).parent
    placements = []
    for path in sorted(
        (root / "results/scheduler_profile_20260905/placements").glob("*.json")
    ):
        data = json.loads(path.read_text())
        placements.append(
            (
                data["num_instructions"],
                data["num_local_experts"],
                np.asarray(data["old_indices"]),
                np.asarray(data["new_indices"]),
            )
        )
    # Exact equality guards replica choice, ordering, coalescing and boundaries.
    rng = np.random.default_rng(10)
    for ranks in (2, 4, 5, 8, 16):
        for _ in range(100):
            old = rng.integers(-1, 50, size=ranks * 32)
            new = rng.integers(-1, 50, size=ranks * 32)
            assert masked(32, old, new) == schedule_migration_batches(32, old, new)
            assert dense(32, old, new) == schedule_migration_batches(32, old, new)
            assert python_dense(32, old, new) == schedule_migration_batches(
                32, old, new
            )
    variants = {
        "current": schedule_migration_batches,
        "mask_cached": masked,
        "mask_fixed": masked_fixed,
        "mask_uncached_groups": lambda n, a, b: masked(n, a, b, False),
        "dense": dense,
        "python_dense": python_dense,
    }
    for _, n, old, new in placements:
        for function in variants.values():
            assert function(n, old, new) == schedule_migration_batches(n, old, new)
    output = root / "results/scheduler_exploration_20260910/dense_comparison"
    output.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        for migrations, n, old, new in placements:
            for _ in range(300):
                for name in rng.permutation(list(variants)):
                    with torch.profiler.record_function(f"explore.{name}.{migrations}"):
                        variants[name](n, old, new)
    prof.export_chrome_trace(str(output / "trace.json"))
    times = {}
    for event in prof.events():
        if event.name.startswith("explore."):
            times.setdefault(event.name, []).append(event.cpu_time_total / 1000)
    with (output / "summary.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["event", "calls", "p50_ms", "p99_ms"])
        for name, values in sorted(times.items()):
            row = [name, len(values), *np.percentile(values, [50, 99])]
            writer.writerow(row)
            print(row)
    print("500 randomized exact-schedule checks passed")


if __name__ == "__main__":
    main()
