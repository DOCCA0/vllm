from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
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
    size_bytes: int | None = None


@dataclass(frozen=True)
class ProblemConfig:
    """Hotspot-contention model for migration batch scheduling."""

    max_batch_num: int
    hotspot_n: int
    time_limit: float | None
    mip_rel_gap: float | None


def load_problem(path: Path) -> tuple[list[MigrationInstruction], ProblemConfig]:
    data = json.loads(path.read_text())
    instructions = [
        MigrationInstruction(
            src_rank=int(item["src"]),
            dst_rank=int(item["dst"]),
            layer=item.get("layer"),
            expert=item.get("expert"),
            size_bytes=item.get("size_bytes"),
        )
        for i, item in enumerate(data["migrations"])
    ]
    if not instructions:
        raise ValueError("migrations must not be empty")

    cfg = data.get("config", {})
    max_batch_num = int(
        cfg.get("max_batch_num", cfg.get("horizon", len(instructions)))
    )
    config = ProblemConfig(
        max_batch_num=max_batch_num,
        hotspot_n=int(cfg.get("hotspot_n", 1)),
        time_limit=(None if cfg.get("time_limit") is None else float(cfg["time_limit"])),
        mip_rel_gap=(
            None if cfg.get("mip_rel_gap") is None else float(cfg["mip_rel_gap"])
        ),
    )
    if config.max_batch_num <= 0:
        raise ValueError("max_batch_num must be a positive integer")
    if config.hotspot_n <= 0:
        raise ValueError("hotspot_n must be a positive integer")
    return instructions, config


def batch_cost(max_rank_participation: int, hotspot_n: int) -> float:
    """Cost of one batch given its hottest rank participation count."""

    if max_rank_participation <= 0:
        return 0.0
    if max_rank_participation <= hotspot_n:
        return float(max_rank_participation)
    return max_rank_participation + 0.1 * (max_rank_participation - hotspot_n)


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
    max_load = n
    x_count = n * batch_count
    z_offset = x_count
    var_count = x_count + batch_count * (max_load + 1)

    def x_idx(i: int, batch: int) -> int:
        return i * batch_count + batch

    def z_idx(batch: int, load: int) -> int:
        return z_offset + batch * (max_load + 1) + load

    rows: list[tuple[dict[int, float], float, float]] = []

    # Every migration is assigned to exactly one batch.
    for i, inst in enumerate(instructions):
        rows.append(({x_idx(i, b): 1.0 for b in range(batch_count)}, 1.0, 1.0))

    # Every batch chooses one maximum-rank-participation level.
    for b in range(batch_count):
        rows.append(({z_idx(b, load): 1.0 for load in range(max_load + 1)}, 1.0, 1.0))

    ranks = sorted({inst.src_rank for inst in instructions} | {inst.dst_rank for inst in instructions})

    # For every rank and batch, the chosen load must cover send + recv count.
    for b in range(batch_count):
        for rank in ranks:
            terms = {z_idx(b, load): -float(load) for load in range(max_load + 1)}
            for i, inst in enumerate(instructions):
                if inst.src_rank == rank:
                    terms[x_idx(i, b)] = terms.get(x_idx(i, b), 0.0) + 1.0
                if inst.dst_rank == rank:
                    terms[x_idx(i, b)] = terms.get(x_idx(i, b), 0.0) + 1.0
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
        for load in range(max_load + 1):
            c[z_idx(b, load)] = batch_cost(load, config.hotspot_n)
    integrality = np.ones(var_count, dtype=int)
    lower_bounds = np.zeros(var_count, dtype=float)
    upper_bounds = np.ones(var_count, dtype=float)
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
        raise RuntimeError(f"MILP failed: status={result.status}, message={result.message}")

    starts: list[int] = []
    schedule: list[dict[str, Any]] = []
    values = result.x
    for i, inst in enumerate(instructions):
        chosen_batch = max(range(batch_count), key=lambda b: values[x_idx(i, b)])
        starts.append(chosen_batch)
        schedule.append(
            {
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
        key=lambda item: (item["start"], item["src"], item["dst"], item["layer"], item["expert"])
    )

    batches = summarize_batches(schedule, batch_count, config.hotspot_n)
    makespan = sum(batch["duration"] for batch in batches)
    return {
        "objective_makespan": makespan,
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
    schedule: list[dict[str, Any]], batch_count: int, hotspot_n: int
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for t in range(batch_count):
        active = [item for item in schedule if item["start"] <= t < item["end"]]
        if not active:
            continue
        rank_counts: dict[int, int] = {}
        for item in active:
            rank_counts[item["src"]] = rank_counts.get(item["src"], 0) + 1
            rank_counts[item["dst"]] = rank_counts.get(item["dst"], 0) + 1
        max_rank_participation = max(rank_counts.values())
        batches.append(
            {
                "batch": t,
                "duration": batch_cost(max_rank_participation, hotspot_n),
                "max_rank_participation": max_rank_participation,
                "hotspot_n": hotspot_n,
                "rank_participation": dict(sorted(rank_counts.items())),
                "active": [
                    f"{_schedule_expert_label(item)}:{item['src']}->{item['dst']}"
                    for item in active
                ],
            }
        )
    return batches



def _expert_label(inst: MigrationInstruction) -> str:
    if inst.layer is not None and inst.expert is not None:
        return f"L{inst.layer}_E{inst.expert}"
    if inst.expert is not None:
        return f"E{inst.expert}"
    return f"{inst.src_rank}->{inst.dst_rank}"


def _schedule_expert_label(item: dict[str, Any]) -> str:
    if item.get("layer") is not None and item.get("expert") is not None:
        return f"L{item['layer']}_E{item['expert']}"
    if item.get("expert") is not None:
        return f"E{item['expert']}"
    return f"{item['src']}->{item['dst']}"


def _mermaid_id(*parts: object) -> str:
    return "_".join(str(part).replace("-", "_") for part in parts)


def _mermaid_label(text: str) -> str:
    return text.replace('"', "'")


def build_before_mermaid_diagram(instructions: list[MigrationInstruction]) -> str:
    lines = [
        "flowchart LR",
        "  classDef rank fill:#eef5ff,stroke:#275d9f,stroke-width:2px;",
        "  classDef expert fill:#ffffff,stroke:#4f9fef,color:#1f4e79;",
        "  subgraph BEFORE[Before: planned expert transfers]",
        "    direction LR",
    ]
    ranks = sorted({inst.src_rank for inst in instructions} | {inst.dst_rank for inst in instructions})
    src_nodes: dict[int, str] = {}
    dst_nodes: dict[int, str] = {}
    for rank in ranks:
        lines.append(f"    subgraph BR{rank}[Rank {rank}]")
        lines.append("      direction TB")
        for i, inst in enumerate(instructions):
            if inst.src_rank == rank:
                node = _mermaid_id("before", "src", i)
                src_nodes[i] = node
                lines.append(f'      {node}["src {_mermaid_label(_expert_label(inst))}"]')
                lines.append(f"      class {node} expert")
            if inst.dst_rank == rank:
                node = _mermaid_id("before", "dst", i)
                dst_nodes[i] = node
                lines.append(f'      {node}["dst {_mermaid_label(_expert_label(inst))}"]')
                lines.append(f"      class {node} expert")
        lines.append("    end")
        lines.append(f"    class BR{rank} rank")
    for i, inst in enumerate(instructions):
        lines.append(f"    {src_nodes[i]} --> {dst_nodes[i]}")
    lines.append("  end")
    return "\n".join(lines) + "\n"


def build_after_mermaid_diagram(schedule: list[dict[str, Any]]) -> str:
    lines = [
        "flowchart TB",
        "  classDef rank fill:#eef5ff,stroke:#275d9f,stroke-width:2px;",
        "  classDef expert fill:#ffffff,stroke:#4f9fef,color:#1f4e79;",
        "  classDef batch fill:#fff8e6,stroke:#d19a00,stroke-width:2px;",
        "  subgraph AFTER[After: scheduled batches]",
        "    direction TB",
    ]
    for batch in sorted({int(item["batch"]) for item in schedule}):
        batch_items = [item for item in schedule if int(item["batch"]) == batch]
        ranks = sorted({int(item["src"]) for item in batch_items} | {int(item["dst"]) for item in batch_items})
        lines.append(f"    subgraph BA{batch}[Batch {batch}]")
        lines.append("      direction LR")
        send_nodes: dict[int, str] = {}
        recv_nodes: dict[int, str] = {}
        for rank in ranks:
            lines.append(f"      subgraph BA{batch}_R{rank}[Rank {rank}]")
            lines.append("        direction TB")
            for idx, item in enumerate(batch_items):
                label = _schedule_expert_label(item)
                if int(item["src"]) == rank:
                    node = _mermaid_id("after", batch, idx, "src")
                    send_nodes[idx] = node
                    lines.append(f'        {node}["send {_mermaid_label(label)}"]')
                    lines.append(f"        class {node} expert")
                if int(item["dst"]) == rank:
                    node = _mermaid_id("after", batch, idx, "dst")
                    recv_nodes[idx] = node
                    lines.append(f'        {node}["recv {_mermaid_label(label)}"]')
                    lines.append(f"        class {node} expert")
            lines.append("      end")
            lines.append(f"      class BA{batch}_R{rank} rank")
        for idx, item in enumerate(batch_items):
            lines.append(f"      {send_nodes[idx]} --> {recv_nodes[idx]}")
        lines.append("    end")
        lines.append(f"    class BA{batch} batch")
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
) -> None:
    """Write before/after migration schematics to separate PNG files via Mermaid."""

    del config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    before_output = output_path.with_name(f"{output_path.stem}_before{output_path.suffix}")
    after_output = output_path.with_name(f"{output_path.stem}_after{output_path.suffix}")
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
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Mermaid CLI failed to render PNG:\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)


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
        write_migration_visualization(
            instructions,
            config,
            solution["schedule"],
            args.visualize_png,
        )
        print(f"Wrote PNG visualization to {args.visualize_png}")


if __name__ == "__main__":
    main()
