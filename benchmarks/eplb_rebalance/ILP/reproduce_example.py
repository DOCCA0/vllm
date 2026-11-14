"""Reproduce the literature-informed example and its assumed-load sensitivity."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

from eplb_migration_ilp import (
    load_problem,
    solve_with_scipy,
    write_migration_visualization,
)


def main() -> None:
    root = Path(__file__).resolve().parent
    problem = json.loads((root / "example_hotspot.json").read_text())
    provenance = problem["provenance"]
    network = problem["network"]
    scale = provenance["normalization_bandwidth_gbps"]
    for key, physical in (
        ("local_capacity", "intra_server_gbps"),
        ("network_capacity", "inter_server_gbps"),
    ):
        assert math.isclose(network[key], provenance[physical] / scale)
    assert math.isclose(
        provenance["reference_interval_ms"],
        provenance["reference_payload_bytes"] * 8 / (scale * 1e9) * 1000,
    )
    instructions, config = load_problem(root / "example_hotspot.json")
    solution = solve_with_scipy(instructions, config)
    (root / "example_hotspot_solution.json").write_text(
        json.dumps({"provenance": provenance, **solution}, indent=2) + "\n"
    )
    rows = []
    for rho in (0.0, 0.3, 0.6, 0.9):
        result = solve_with_scipy(instructions, replace(config, inference_traffic=rho))
        off, on = result["unbatched_lower_bound"], result["optimal"]
        assert on["batch_count"] == 3
        assert math.isclose(on["peak_link_utilization"], 1.0)
        assert math.isclose(off["peak_link_utilization"], 3.0)
        rows.append(
            {
                "assumed_rho": rho,
                "unbatched_inference": off["inference_time"],
                "optimal_inference": on["inference_time"],
                "optimal_batches": on["batch_count"],
                "unbatched_peak_normalized_load": off["peak_link_utilization"],
                "optimal_peak_normalized_load": on["peak_link_utilization"],
            }
        )
    with (root / "rho_sensitivity.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(solution["optimal"], indent=2))
    print("Capacity normalization and four assumed-rho solves verified.")
    for path in write_migration_visualization(
        instructions, config, solution["schedule"], root / "example_hotspot.png"
    ):
        print(path)


if __name__ == "__main__":
    main()
