from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MigrationInstruction:
    """One already-decided expert migration instruction."""

    src_rank: int
    dst_rank: int
    layer: int | None = None
    expert: int | None = None


@dataclass(frozen=True)
class NetworkLink:
    """One fabric link used by fixed ECMP shortest-path routing."""

    src_rank: int
    dst_rank: int
    capacity: float | None = None
    background: float | None = None


@dataclass(frozen=True)
class ECMPNetwork:
    """Network topology for fixed ECMP link-load estimation."""

    links: tuple[NetworkLink, ...]
    directed: bool
    link_capacity: float
    num_gpus: int | None = None
    vertex_roles: dict[int, str] | None = None


@dataclass(frozen=True)
class ProblemConfig:
    """Hotspot-contention model for migration batch scheduling."""

    time_limit: float | None
    mip_rel_gap: float | None
    network: ECMPNetwork


def load_problem(path: Path) -> tuple[list[MigrationInstruction], ProblemConfig]:
    data = json.loads(path.read_text())
    instructions = [
        MigrationInstruction(
            src_rank=int(item["src"]),
            dst_rank=int(item["dst"]),
            layer=item.get("layer"),
            expert=item.get("expert"),
        )
        for item in data["migrations"]
    ]
    if not instructions:
        raise ValueError("migrations must not be empty")

    cfg = data.get("config", {})
    network = _load_network(data.get("network"), cfg)
    if network is None:
        raise ValueError("network must be provided for ECMP batch-time modeling")
    for i, inst in enumerate(instructions):
        if inst.src_rank == inst.dst_rank:
            raise ValueError(f"migration {i} has identical src and dst")
        if network.num_gpus is not None and not (
            0 <= inst.src_rank < network.num_gpus
            and 0 <= inst.dst_rank < network.num_gpus
        ):
            raise ValueError(
                f"migration {i} ({inst.src_rank}->{inst.dst_rank}) is outside "
                f"the GPU rank range 0..{network.num_gpus - 1}"
            )
    config = ProblemConfig(
        time_limit=(
            None if cfg.get("time_limit") is None else float(cfg["time_limit"])
        ),
        mip_rel_gap=(
            None if cfg.get("mip_rel_gap") is None else float(cfg["mip_rel_gap"])
        ),
        network=network,
    )
    return instructions, config


def _background_fraction(spec: Any, tier: str) -> float:
    """Resolve an inference-traffic fraction for one link tier."""

    if spec is None:
        return 0.0
    if isinstance(spec, (int, float)):
        fraction = float(spec)
    elif isinstance(spec, dict):
        fraction = float(spec.get(tier, spec.get("default", 0.0)))
    else:
        raise ValueError("network.inference_traffic must be a number or an object")
    if not 0.0 <= fraction < 1.0:
        raise ValueError("inference traffic fractions must be in [0, 1)")
    return fraction


def _parse_link(item: Any, default_background: float) -> NetworkLink:
    if isinstance(item, dict):
        src_rank = int(item["src"])
        dst_rank = int(item["dst"])
        capacity = None if item.get("capacity") is None else float(item["capacity"])
        background = (
            default_background
            if item.get("background") is None
            else float(item["background"])
        )
    else:
        src_rank = int(item[0])
        dst_rank = int(item[1])
        capacity = None
        background = default_background
    if src_rank == dst_rank:
        raise ValueError("network links must connect two distinct ranks")
    if capacity is not None and capacity <= 0:
        raise ValueError("per-link capacity must be positive")
    if not 0.0 <= background < 1.0:
        raise ValueError("per-link background must be in [0, 1)")
    return NetworkLink(src_rank, dst_rank, capacity, background)


def _validate_unique_links(links: list[NetworkLink], directed: bool) -> None:
    seen: set[tuple[int, int]] = set()
    for link in links:
        pair = (link.src_rank, link.dst_rank)
        if pair in seen or (not directed and (pair[1], pair[0]) in seen):
            raise ValueError(f"duplicate network link {_link_label(pair)}")
        seen.add(pair)


def _build_hierarchical_network(
    data: dict[str, Any], link_capacity: float, traffic: Any
) -> ECMPNetwork:
    """Build a 3-tier GPU -> NVSwitch -> cluster-fabric topology.

    Layer 1 (GPU): ranks 0..num_gpus-1.
    Layer 2 (node): every node owns nvswitches_per_node NVSwitch vertices
    (default 1); each of the node's gpus_per_node GPUs connects to all of
    them with nvlink_capacity links (default 18x the fabric capacity, i.e.
    900 GB/s NVLink 4 vs 50 GB/s NDR 400G). A single NVSwitch per node makes
    every pair's shortest path unique and ECMP degenerates to 1.0 per link;
    two or more NVSwitches per node create equal-cost paths through each
    switch, which is what gives ECMP real alternatives to split over.
    Layer 3 (cluster fabric): the NVSwitches interconnect over IB/RoCE
    via one spine vertex with fabric_capacity links.
    """

    try:
        num_gpus = int(data["num_gpus"])
        gpus_per_node = int(data["gpus_per_node"])
    except KeyError as exc:
        raise ValueError(
            "hierarchical network needs num_gpus and gpus_per_node"
        ) from exc
    if num_gpus <= 0 or gpus_per_node <= 0:
        raise ValueError("num_gpus and gpus_per_node must be positive integers")
    if num_gpus % gpus_per_node != 0:
        raise ValueError("num_gpus must be a multiple of gpus_per_node")
    nvswitches_per_node = int(data.get("nvswitches_per_node", 1))
    if nvswitches_per_node <= 0:
        raise ValueError("nvswitches_per_node must be a positive integer")

    num_nodes = num_gpus // gpus_per_node
    nvlink_capacity = float(data.get("nvlink_capacity", 18.0 * link_capacity))
    fabric_capacity = float(data.get("fabric_capacity", link_capacity))
    if nvlink_capacity <= 0 or fabric_capacity <= 0:
        raise ValueError("nvlink_capacity and fabric_capacity must be positive")
    nvlink_background = _background_fraction(traffic, "nvlink")
    fabric_background = _background_fraction(traffic, "fabric")

    spine = num_gpus + num_nodes * nvswitches_per_node
    roles = {spine: "Cluster spine (IB/RoCE)"}
    links: list[NetworkLink] = []
    for node in range(num_nodes):
        for index in range(nvswitches_per_node):
            switch = num_gpus + node * nvswitches_per_node + index
            roles[switch] = f"Node {node} NVSwitch {index}"
            for gpu in range(node * gpus_per_node, (node + 1) * gpus_per_node):
                links.append(
                    NetworkLink(gpu, switch, nvlink_capacity, nvlink_background)
                )
            links.append(NetworkLink(switch, spine, fabric_capacity, fabric_background))

    _validate_unique_links(links, directed=False)
    return ECMPNetwork(
        links=tuple(links),
        directed=False,
        link_capacity=link_capacity,
        num_gpus=num_gpus,
        vertex_roles=roles,
    )


def _load_network(data: Any, cfg: dict[str, Any]) -> ECMPNetwork | None:
    if data is None:
        return None

    link_capacity = float(data.get("link_capacity", cfg.get("link_capacity", 1.0)))
    if link_capacity <= 0:
        raise ValueError("network.link_capacity must be positive")

    traffic = data.get("inference_traffic", cfg.get("inference_traffic"))
    if "num_gpus" in data or "gpus_per_node" in data:
        return _build_hierarchical_network(data, link_capacity, traffic)

    if "links" not in data:
        raise ValueError("network.links must be provided when network is set")

    default_background = _background_fraction(traffic, "default")
    links = [_parse_link(item, default_background) for item in data["links"]]
    if not links:
        raise ValueError("network.links must not be empty")

    directed = bool(data.get("directed", False))
    _validate_unique_links(links, directed)
    return ECMPNetwork(
        links=tuple(links),
        directed=directed,
        link_capacity=link_capacity,
    )


def _link_label(link: tuple[int, int]) -> str:
    return f"{link[0]}->{link[1]}"


def _link_capacities(network: ECMPNetwork) -> dict[tuple[int, int], float]:
    capacities: dict[tuple[int, int], float] = {}
    for link in network.links:
        capacity = network.link_capacity if link.capacity is None else link.capacity
        capacities[(link.src_rank, link.dst_rank)] = capacity
        if not network.directed:
            capacities[(link.dst_rank, link.src_rank)] = capacity
    return capacities


def _background_loads(network: ECMPNetwork) -> dict[tuple[int, int], float]:
    """Absolute inference background traffic reserved on each directed link."""

    capacities = _link_capacities(network)
    loads: dict[tuple[int, int], float] = {}
    for link in network.links:
        fraction = link.background or 0.0
        if not fraction:
            continue
        capacity = capacities[(link.src_rank, link.dst_rank)]
        loads[(link.src_rank, link.dst_rank)] = fraction * capacity
        if not network.directed:
            loads[(link.dst_rank, link.src_rank)] = fraction * capacity
    return loads


def _effective_capacities(network: ECMPNetwork) -> dict[tuple[int, int], float]:
    """Per-directed-link capacity left for migrations after reserving the
    sustained inference traffic rate on that link."""

    capacities = _link_capacities(network)
    background = _background_loads(network)
    return {
        link: capacity - background.get(link, 0.0)
        for link, capacity in capacities.items()
    }


def _network_adjacency(
    network: ECMPNetwork,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    reverse_adjacency: dict[int, set[int]] = defaultdict(set)
    for link in network.links:
        adjacency[link.src_rank].add(link.dst_rank)
        reverse_adjacency[link.dst_rank].add(link.src_rank)
        if not network.directed:
            adjacency[link.dst_rank].add(link.src_rank)
            reverse_adjacency[link.src_rank].add(link.dst_rank)

    forward = {rank: sorted(neighbors) for rank, neighbors in adjacency.items()}
    reverse = {rank: sorted(neighbors) for rank, neighbors in reverse_adjacency.items()}
    return forward, reverse


def _bfs_distances(source: int, adjacency: dict[int, list[int]]) -> dict[int, int]:
    distances = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, []):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def _ecmp_link_fractions_for_pair(
    src_rank: int,
    dst_rank: int,
    adjacency: dict[int, list[int]],
    reverse_adjacency: dict[int, list[int]],
) -> dict[tuple[int, int], float]:
    dist_from_src = _bfs_distances(src_rank, adjacency)
    if dst_rank not in dist_from_src:
        raise ValueError(f"no ECMP path from rank {src_rank} to rank {dst_rank}")

    dist_to_dst = _bfs_distances(dst_rank, reverse_adjacency)
    shortest_len = dist_from_src[dst_rank]
    ordered_nodes = sorted(dist_from_src, key=lambda node: dist_from_src[node])

    prefix_path_counts: dict[int, int] = defaultdict(int)
    prefix_path_counts[src_rank] = 1
    for node in ordered_nodes:
        for neighbor in adjacency.get(node, []):
            if dist_from_src.get(neighbor) == dist_from_src[node] + 1:
                prefix_path_counts[neighbor] += prefix_path_counts[node]

    suffix_path_counts: dict[int, int] = defaultdict(int)
    suffix_path_counts[dst_rank] = 1
    for node in reversed(ordered_nodes):
        for neighbor in adjacency.get(node, []):
            if (
                dist_from_src.get(neighbor) == dist_from_src[node] + 1
                and dist_from_src[node] + 1 + dist_to_dst.get(neighbor, 10**9)
                == shortest_len
            ):
                suffix_path_counts[node] += suffix_path_counts[neighbor]

    total_paths = prefix_path_counts[dst_rank]
    fractions: dict[tuple[int, int], float] = {}
    for node in ordered_nodes:
        for neighbor in adjacency.get(node, []):
            if (
                dist_from_src.get(neighbor) == dist_from_src[node] + 1
                and dist_from_src[node] + 1 + dist_to_dst.get(neighbor, 10**9)
                == shortest_len
            ):
                path_count = prefix_path_counts[node] * suffix_path_counts[neighbor]
                fractions[(node, neighbor)] = path_count / total_paths
    return fractions


def compute_ecmp_link_fractions(
    instructions: list[MigrationInstruction], network: ECMPNetwork
) -> dict[int, dict[tuple[int, int], float]]:
    adjacency, reverse_adjacency = _network_adjacency(network)
    pair_cache: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
    fractions: dict[int, dict[tuple[int, int], float]] = {}
    for i, inst in enumerate(instructions):
        pair = (inst.src_rank, inst.dst_rank)
        if pair not in pair_cache:
            pair_cache[pair] = _ecmp_link_fractions_for_pair(
                inst.src_rank,
                inst.dst_rank,
                adjacency,
                reverse_adjacency,
            )
        fractions[i] = pair_cache[pair]
    return fractions


def _clean_float(value: float) -> float:
    return round(value, 6)


def solve_with_scipy(
    instructions: list[MigrationInstruction], config: ProblemConfig
) -> dict[str, Any]:
    """Solve the discrete-time ILP with scipy.optimize.milp."""

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as exc:
        raise RuntimeError(
            "This solver needs scipy>=1.11. Install it in the vLLM venv with: "
            "uv pip install 'scipy>=1.11'"
        ) from exc

    n = len(instructions)
    # One migration per batch is always feasible (each migration's path
    # fractions fit the effective capacities, checked below), so n batches
    # are a sufficient upper bound; unused batches simply take zero time.
    batch_count = n
    ecmp_link_fractions = compute_ecmp_link_fractions(instructions, config.network)
    links = sorted(
        {link for fractions in ecmp_link_fractions.values() for link in fractions}
    )
    effective_capacities = _effective_capacities(config.network)
    background_loads = _background_loads(config.network)
    x_count = n * batch_count
    batch_time_offset = x_count
    var_count = x_count + batch_count

    def x_idx(i: int, batch: int) -> int:
        return i * batch_count + batch

    def batch_time_idx(batch: int) -> int:
        return batch_time_offset + batch

    rows: list[tuple[dict[int, float], float, float]] = []

    # Every migration is assigned to exactly one batch.
    for i, inst in enumerate(instructions):
        rows.append(({x_idx(i, b): 1.0 for b in range(batch_count)}, 1.0, 1.0))

    # Hard per-batch link capacity: one wave cannot push more concurrent
    # migration traffic over a link than its effective capacity. This is what
    # forces migrations into multiple waves — a purely soft time constraint
    # would always admit a single-wave optimum.
    for i in range(n):
        for link, fraction in ecmp_link_fractions[i].items():
            if fraction > effective_capacities[link] + 1e-9:
                raise ValueError(
                    f"migration {i} ({instructions[i].src_rank}->"
                    f"{instructions[i].dst_rank}) needs {fraction:.6g} on link "
                    f"{_link_label(link)} above its effective capacity "
                    f"{effective_capacities[link]:.6g}; increase link capacity "
                    "or lower inference_traffic"
                )
    for b in range(batch_count):
        for link in links:
            terms = {
                x_idx(i, b): fraction
                for i in range(n)
                if (fraction := ecmp_link_fractions[i].get(link, 0.0))
            }
            if terms:
                rows.append((terms, -float("inf"), effective_capacities[link]))

    # Batch time is determined by the most loaded normalized fabric link.
    # Sustained inference traffic reserves part of each link's bandwidth, so
    # migrations only see the remaining effective capacity.
    for b in range(batch_count):
        for link in links:
            terms = {batch_time_idx(b): -effective_capacities[link]}
            for i in range(n):
                fraction = ecmp_link_fractions[i].get(link, 0.0)
                if fraction:
                    terms[x_idx(i, b)] = terms.get(x_idx(i, b), 0.0) + fraction
            rows.append((terms, -float("inf"), 0.0))

    matrix = lil_matrix((len(rows), var_count), dtype=float)
    lb = np.empty(len(rows), dtype=float)
    ub = np.empty(len(rows), dtype=float)
    for r, (terms, lower, upper) in enumerate(rows):
        for c, value in terms.items():
            matrix[r, c] = value
        lb[r] = lower
        ub[r] = upper

    c = np.zeros(var_count, dtype=float)
    for b in range(batch_count):
        c[batch_time_idx(b)] = 1.0
    integrality = np.zeros(var_count, dtype=int)
    integrality[:x_count] = 1
    lower_bounds = np.zeros(var_count, dtype=float)
    upper_bounds = np.ones(var_count, dtype=float)
    min_capacity = min(effective_capacities[link] for link in links)
    upper_bounds[batch_time_offset:] = n / min_capacity
    options: dict[str, Any] = {}
    if config.time_limit is not None:
        options["time_limit"] = config.time_limit
    if config.mip_rel_gap is not None:
        options["mip_rel_gap"] = config.mip_rel_gap

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(matrix.tocsr(), lb, ub),
        options=options,
    )
    if not result.success:
        raise RuntimeError(
            f"MILP failed: status={result.status}, message={result.message}"
        )

    schedule: list[dict[str, Any]] = []
    values = result.x
    for i, inst in enumerate(instructions):
        chosen_batch = max(range(batch_count), key=lambda b: values[x_idx(i, b)])
        schedule.append(
            {
                "migration_id": i,
                "layer": inst.layer,
                "expert": inst.expert,
                "src": inst.src_rank,
                "dst": inst.dst_rank,
                "batch": chosen_batch,
            }
        )
    # Compact batch ids so empty batches do not leave numbering gaps.
    used_batches = sorted({item["batch"] for item in schedule})
    batch_remap = {batch: new for new, batch in enumerate(used_batches)}
    for item in schedule:
        item["batch"] = batch_remap[item["batch"]]
        item["start"] = item["batch"]
        item["end"] = item["batch"] + 1
    schedule.sort(
        key=lambda item: (
            item["start"],
            item["src"],
            item["dst"],
            item["layer"],
            item["expert"],
        )
    )

    batches = summarize_batches(
        schedule,
        batch_count,
        config.network,
        ecmp_link_fractions,
        background_loads,
    )
    total_batch_time = sum(batch["batch_time"] for batch in batches)
    return {
        "objective_value": _clean_float(total_batch_time),
        "lower_bound": float(result.mip_dual_bound)
        if getattr(result, "mip_dual_bound", None) is not None
        else None,
        "mip_gap": float(result.mip_gap)
        if getattr(result, "mip_gap", None) is not None
        else None,
        "schedule": schedule,
        "batches": batches,
    }


def summarize_batches(
    schedule: list[dict[str, Any]],
    batch_count: int,
    network: ECMPNetwork,
    ecmp_link_fractions: dict[int, dict[tuple[int, int], float]],
    background_loads: dict[tuple[int, int], float] | None = None,
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    effective_capacities = _effective_capacities(network)
    background_loads = background_loads or {}
    for t in range(batch_count):
        active = [item for item in schedule if item["start"] <= t < item["end"]]
        if not active:
            continue
        rank_loads: dict[int, int] = {}
        for item in active:
            rank_loads[item["src"]] = rank_loads.get(item["src"], 0) + 1
            rank_loads[item["dst"]] = rank_loads.get(item["dst"], 0) + 1
        max_rank_load = max(rank_loads.values())
        batch_summary: dict[str, Any] = {
            "batch": t,
            "batch_time": 0.0,
            "max_rank_load": max_rank_load,
            "rank_load": dict(sorted(rank_loads.items())),
            "active": [
                f"{_schedule_expert_label(item)}:{item['src']}->{item['dst']}"
                for item in active
            ],
        }
        link_loads: dict[tuple[int, int], float] = {}
        for item in active:
            migration_id = int(item["migration_id"])
            for link, fraction in ecmp_link_fractions[migration_id].items():
                link_loads[link] = link_loads.get(link, 0.0) + fraction
        link_times = {
            link: load / effective_capacities[link]
            for link, load in sorted(link_loads.items())
        }
        batch_time = max(link_times.values())
        batch_summary["batch_time"] = _clean_float(batch_time)
        batch_summary["max_link_load"] = _clean_float(max(link_loads.values()))
        batch_summary["link_capacity"] = network.link_capacity
        batch_summary["link_load"] = {
            _link_label(link): _clean_float(load)
            for link, load in sorted(link_loads.items())
        }
        if background_loads:
            batch_summary["background_load"] = {
                _link_label(link): _clean_float(background_loads.get(link, 0.0))
                for link in sorted(link_loads)
            }
        batch_summary["link_time"] = {
            _link_label(link): _clean_float(time)
            for link, time in sorted(link_times.items())
        }
        batches.append(batch_summary)
    return batches


def _expert_label(inst: MigrationInstruction) -> str:
    return _migration_label(
        inst.layer,
        inst.expert,
        f"{inst.src_rank}->{inst.dst_rank}",
    )


def _schedule_expert_label(item: dict[str, Any]) -> str:
    return _migration_label(
        item.get("layer"),
        item.get("expert"),
        f"{item['src']}->{item['dst']}",
    )


def _migration_label(layer: int | None, expert: int | None, fallback: str) -> str:
    if layer is not None and expert is not None:
        return f"L{layer}_E{expert}"
    if expert is not None:
        return f"E{expert}"
    return fallback


def _mermaid_id(*parts: object) -> str:
    return "_".join(str(part).replace("-", "_") for part in parts)


def _mermaid_label(text: str) -> str:
    return text.replace('"', "'")


def build_before_mermaid_diagram(instructions: list[MigrationInstruction]) -> str:
    lines = [
        "flowchart LR",
        "  classDef rank fill:#eef5ff,stroke:#275d9f,stroke-width:2px;",
        "  subgraph BEFORE[Before: planned expert transfers]",
        "    direction LR",
    ]
    ranks = sorted(
        {inst.src_rank for inst in instructions}
        | {inst.dst_rank for inst in instructions}
    )
    for rank in ranks:
        node = _mermaid_id("before", "rank", rank)
        lines.append(f'    {node}["Rank {rank}"]')
        lines.append(f"    class {node} rank")
    for inst in instructions:
        src_node = _mermaid_id("before", "rank", inst.src_rank)
        dst_node = _mermaid_id("before", "rank", inst.dst_rank)
        label = _mermaid_label(_expert_label(inst))
        lines.append(f'    {src_node} -->|"{label}"| {dst_node}')
    lines.append("  end")
    return "\n".join(lines) + "\n"


def build_after_mermaid_diagram(schedule: list[dict[str, Any]]) -> str:
    lines = [
        "flowchart TB",
        "  classDef rank fill:#eef5ff,stroke:#275d9f,stroke-width:2px;",
        "  classDef batch fill:#fff8e6,stroke:#d19a00,stroke-width:2px;",
        "  subgraph AFTER[After: scheduled batches]",
        "    direction TB",
    ]
    # flowchart TB does not stack sibling subgraphs on its own, so chain them
    # with invisible links to render batches top to bottom in ascending order.
    batches = sorted({int(item["batch"]) for item in schedule})
    for batch in batches:
        batch_items = [item for item in schedule if int(item["batch"]) == batch]
        ranks = sorted(
            {int(item["src"]) for item in batch_items}
            | {int(item["dst"]) for item in batch_items}
        )
        lines.append(f"    subgraph BA{batch}[Batch {batch}]")
        lines.append("      direction LR")
        for rank in ranks:
            node = _mermaid_id("after", batch, "rank", rank)
            lines.append(f'      {node}["Rank {rank}"]')
            lines.append(f"      class {node} rank")
        for item in batch_items:
            src_node = _mermaid_id("after", batch, "rank", item["src"])
            dst_node = _mermaid_id("after", batch, "rank", item["dst"])
            label = _mermaid_label(_schedule_expert_label(item))
            lines.append(f'      {src_node} -->|"{label}"| {dst_node}')
        lines.append("    end")
        lines.append(f"    class BA{batch} batch")
    for first, second in zip(batches, batches[1:]):
        lines.append(f"    BA{first} ~~~ BA{second}")
    lines.append("  end")
    return "\n".join(lines) + "\n"


def _max_link_loads(
    schedule: list[dict[str, Any]],
    ecmp_link_fractions: dict[int, dict[tuple[int, int], float]],
) -> dict[tuple[int, int], float]:
    max_loads: dict[tuple[int, int], float] = {}
    batches = sorted({int(item["batch"]) for item in schedule})
    for batch in batches:
        loads: dict[tuple[int, int], float] = {}
        for item in schedule:
            if int(item["batch"]) != batch:
                continue
            migration_id = int(item["migration_id"])
            for link, fraction in ecmp_link_fractions[migration_id].items():
                loads[link] = loads.get(link, 0.0) + fraction
        for link, load in loads.items():
            max_loads[link] = max(max_loads.get(link, 0.0), load)
    return max_loads


def build_link_mermaid_diagram(
    instructions: list[MigrationInstruction],
    network: ECMPNetwork,
    schedule: list[dict[str, Any]],
) -> str:
    endpoint_ranks = {inst.src_rank for inst in instructions} | {
        inst.dst_rank for inst in instructions
    }
    ecmp_link_fractions = compute_ecmp_link_fractions(instructions, network)
    used_vertices = set(endpoint_ranks)
    for fractions in ecmp_link_fractions.values():
        for link in fractions:
            used_vertices.add(link[0])
            used_vertices.add(link[1])
    ranks = sorted(
        rank
        for rank in {link.src_rank for link in network.links}
        | {link.dst_rank for link in network.links}
        if rank in used_vertices
    )
    max_loads = _max_link_loads(
        schedule,
        ecmp_link_fractions,
    )
    capacities = _effective_capacities(network)
    roles = network.vertex_roles or {}
    lines = [
        "flowchart LR",
        "  classDef gpu fill:#e9fff5,stroke:#23734d,stroke-width:2px;",
        "  classDef fabric fill:#fff4df,stroke:#9a6200,stroke-width:2px;",
        "  subgraph LINKS[ECMP communication links]",
        "    direction LR",
    ]
    for rank in ranks:
        node = _mermaid_id("link", rank)
        if rank in endpoint_ranks:
            role = f"GPU rank {rank}"
            node_class = "gpu"
        else:
            role = f"{roles.get(rank, 'Fabric switch')} ({rank})"
            node_class = "fabric"
        lines.append(f'    {node}["{role}"]')
        lines.append(f"    class {node} {node_class}")
    edge = "-->" if network.directed else "---"
    hot_edges: list[int] = []
    drawn_links = [
        link
        for link in network.links
        if link.src_rank in used_vertices and link.dst_rank in used_vertices
    ]
    for edge_idx, link in enumerate(drawn_links):
        src_node = _mermaid_id("link", link.src_rank)
        dst_node = _mermaid_id("link", link.dst_rank)
        directions = [(link.src_rank, link.dst_rank)]
        if not network.directed:
            directions.append((link.dst_rank, link.src_rank))
        load = max(max_loads.get(direction, 0.0) for direction in directions)
        cap = capacities[(link.src_rank, link.dst_rank)]
        if load + 1e-9 >= cap:
            label = f"{_clean_float(load)}/{_clean_float(cap)}"
            lines.append(f'    {src_node} {edge}|"{label}"| {dst_node}')
            hot_edges.append(edge_idx)
        else:
            lines.append(f"    {src_node} {edge} {dst_node}")
    for edge_idx in hot_edges:
        lines.append(f"    linkStyle {edge_idx} stroke:#b91c1c,stroke-width:4px")
    lines.append("  end")
    return "\n".join(lines) + "\n"


def _mermaid_command() -> list[str]:
    def which_linux(name: str) -> str | None:
        resolved = shutil.which(name)
        if resolved is not None and not resolved.startswith("/mnt/c/"):
            return resolved
        for directory in ("/home/wu/.nvm/versions/node/v22.21.1/bin", "/usr/bin"):
            candidate = Path(directory) / name
            if candidate.exists():
                return str(candidate)
        return resolved

    mmdc = which_linux("mmdc")
    if mmdc is not None:
        return [mmdc]
    npx = which_linux("npx")
    if npx is not None:
        return [npx, "--yes", "@mermaid-js/mermaid-cli"]
    raise RuntimeError(
        "PNG visualization needs Mermaid CLI. Install Node.js/npm and run with "
        "npx @mermaid-js/mermaid-cli, or install mmdc globally."
    )


def write_migration_visualization(
    instructions: list[MigrationInstruction],
    config: ProblemConfig,
    schedule: list[dict[str, Any]],
    output_path: Path,
) -> list[Path]:
    """Write migration and ECMP link schematics to separate PNG files."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    before_output = output_path.with_name(
        f"{output_path.stem}_before{output_path.suffix}"
    )
    after_output = output_path.with_name(
        f"{output_path.stem}_after{output_path.suffix}"
    )
    link_output = output_path.with_name(f"{output_path.stem}_links{output_path.suffix}")
    rendered_outputs = [before_output, after_output]
    temp_paths: list[Path] = []

    def render_mermaid(mermaid_source: str, png_path: Path) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".mmd", delete=False) as mmd_file:
            mmd_file.write(mermaid_source)
            mmd_path = Path(mmd_file.name)
        temp_paths.append(mmd_path)
        env = dict(os.environ)
        env["PATH"] = "/home/wu/.nvm/versions/node/v22.21.1/bin:" + env.get("PATH", "")
        subprocess.run(
            [
                *_mermaid_command(),
                "-i",
                str(mmd_path),
                "-o",
                str(png_path),
                "-b",
                "white",
                "-s",
                "3",
            ],
            check=True,
            text=True,
            capture_output=True,
            env=env,
        )

    try:
        render_mermaid(build_before_mermaid_diagram(instructions), before_output)
        render_mermaid(build_after_mermaid_diagram(schedule), after_output)
        if config.network is not None:
            render_mermaid(
                build_link_mermaid_diagram(instructions, config.network, schedule),
                link_output,
            )
            rendered_outputs.append(link_output)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Mermaid CLI failed to render PNG:\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)
    return rendered_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve an ILP optimum for EPLB migration instruction ordering."
    )
    parser.add_argument("problem", type=Path, help="Path to a JSON problem file")
    parser.add_argument(
        "--visualize-png",
        type=Path,
        default=None,
        help="Optional path for a PNG before/after migration visualization",
    )
    args = parser.parse_args()

    instructions, config = load_problem(args.problem)
    solution = solve_with_scipy(instructions, config)
    text = json.dumps(solution, indent=2, ensure_ascii=False)
    print(text)
    if args.visualize_png is not None:
        output_paths = write_migration_visualization(
            instructions,
            config,
            solution["schedule"],
            args.visualize_png,
        )
        rendered = ", ".join(str(path) for path in output_paths)
        print(f"Wrote PNG visualizations to {rendered}")


if __name__ == "__main__":
    main()
