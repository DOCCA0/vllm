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


@dataclass(frozen=True)
class ECMPNetwork:
    """Network topology for fixed ECMP link-load estimation."""

    links: tuple[NetworkLink, ...]
    directed: bool
    link_capacity: float


@dataclass(frozen=True)
class ProblemConfig:
    """Hotspot-contention model for migration batch scheduling."""

    max_batch_num: int
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
    max_batch_num = int(
        cfg.get("max_batch_num", cfg.get("horizon", len(instructions)))
    )
    config = ProblemConfig(
        max_batch_num=max_batch_num,
        time_limit=(
            None if cfg.get("time_limit") is None else float(cfg["time_limit"])
        ),
        mip_rel_gap=(
            None if cfg.get("mip_rel_gap") is None else float(cfg["mip_rel_gap"])
        ),
        network=network,
    )
    if config.max_batch_num <= 0:
        raise ValueError("max_batch_num must be a positive integer")
    return instructions, config


def _load_network(data: Any, cfg: dict[str, Any]) -> ECMPNetwork | None:
    if data is None:
        return None
    if "links" not in data:
        raise ValueError("network.links must be provided when network is set")

    link_capacity = float(data.get("link_capacity", cfg.get("link_capacity", 1.0)))
    if link_capacity <= 0:
        raise ValueError("network.link_capacity must be positive")

    links: list[NetworkLink] = []
    for item in data["links"]:
        if isinstance(item, dict):
            src_rank = int(item["src"])
            dst_rank = int(item["dst"])
            capacity = None if item.get("capacity") is None else float(item["capacity"])
        else:
            src_rank = int(item[0])
            dst_rank = int(item[1])
            capacity = None
        if src_rank == dst_rank:
            raise ValueError("network links must connect two distinct ranks")
        if capacity is not None and capacity <= 0:
            raise ValueError("per-link capacity must be positive")
        links.append(NetworkLink(src_rank, dst_rank, capacity))

    if not links:
        raise ValueError("network.links must not be empty")

    return ECMPNetwork(
        links=tuple(links),
        directed=bool(data.get("directed", False)),
        link_capacity=link_capacity,
    )


def _link_label(link: tuple[int, int]) -> str:
    return f"{link[0]}->{link[1]}"


def _directed_links(network: ECMPNetwork) -> tuple[tuple[int, int], ...]:
    links: set[tuple[int, int]] = set()
    for link in network.links:
        forward = (link.src_rank, link.dst_rank)
        if forward in links:
            raise ValueError(f"duplicate network link {_link_label(forward)}")
        links.add(forward)
        if not network.directed:
            reverse = (link.dst_rank, link.src_rank)
            if reverse in links:
                raise ValueError(f"duplicate network link {_link_label(reverse)}")
            links.add(reverse)
    return tuple(sorted(links))


def _link_capacities(network: ECMPNetwork) -> dict[tuple[int, int], float]:
    capacities: dict[tuple[int, int], float] = {}
    for link in network.links:
        capacity = network.link_capacity if link.capacity is None else link.capacity
        capacities[(link.src_rank, link.dst_rank)] = capacity
        if not network.directed:
            capacities[(link.dst_rank, link.src_rank)] = capacity
    return capacities


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
    batch_count = config.max_batch_num
    ecmp_link_fractions = compute_ecmp_link_fractions(instructions, config.network)
    links = _directed_links(config.network)
    link_capacities = _link_capacities(config.network)
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

    # Batch time is determined by the most loaded normalized fabric link.
    for b in range(batch_count):
        for link in links:
            terms = {batch_time_idx(b): -link_capacities[link]}
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
    min_capacity = min(link_capacities.values())
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
                "start": chosen_batch,
                "end": chosen_batch + 1,
            }
        )
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
    )
    total_batch_time = sum(batch["batch_time"] for batch in batches)
    return {
        "objective_value": _clean_float(total_batch_time),
        "objective_total_time": _clean_float(total_batch_time),
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
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    link_capacities = _link_capacities(network)
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
            "max_rank_participation": max_rank_load,
            "max_rank_load": max_rank_load,
            "rank_participation": dict(sorted(rank_loads.items())),
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
            link: load / link_capacities[link]
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
    # Mermaid lays sibling subgraphs right-to-left here, so emit them in reverse
    # to render batch numbers visually from left to right.
    for batch in sorted({int(item["batch"]) for item in schedule}, reverse=True):
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
    ranks = sorted({link.src_rank for link in network.links} | {
        link.dst_rank for link in network.links
    })
    max_loads = _max_link_loads(
        schedule,
        compute_ecmp_link_fractions(instructions, network),
    )
    capacities = _link_capacities(network)
    lines = [
        "flowchart LR",
        "  classDef gpu fill:#e9fff5,stroke:#23734d,stroke-width:2px;",
        "  classDef fabric fill:#fff4df,stroke:#9a6200,stroke-width:2px;",
        "  subgraph LINKS[ECMP communication links]",
        "    direction LR",
    ]
    for rank in ranks:
        node = _mermaid_id("link", rank)
        role = "GPU rank" if rank in endpoint_ranks else "NVSwitch/fabric"
        lines.append(f'    {node}["{role} {rank}"]')
        node_class = "gpu" if rank in endpoint_ranks else "fabric"
        lines.append(f"    class {node} {node_class}")
    edge = "-->" if network.directed else "---"
    hot_edges: list[int] = []
    for edge_idx, link in enumerate(network.links):
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
