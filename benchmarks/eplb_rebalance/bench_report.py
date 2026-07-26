#!/usr/bin/env python3
"""汇总 eplb_rebalance/results/*/ 下所有实验组, 输出对比表和 summary.csv"""
import csv
import json
import re
import statistics as st
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
RE_REARRANGE = re.compile(r"Rearranged experts.*?in ([0-9.]+) s")


def parse_run(d: Path) -> dict | None:
    row = {"tag": d.name}
    server_log = d / "server.log"
    if server_log.exists():
        times = [float(t) for t in RE_REARRANGE.findall(server_log.read_text())]
        if times:
            row.update(
                n_rebalance=len(times),
                rearrange_mean_s=round(st.mean(times), 3),
                rearrange_p95_s=round(sorted(times)[int(0.95 * (len(times) - 1))], 3),
                rearrange_max_s=round(max(times), 3),
            )
    bench = d / "bench.json"
    if bench.exists():
        data = json.loads(bench.read_text())
        for k in ("p99_ttft_ms", "p99_tpot_ms", "p99_e2el_ms",
                  "mean_ttft_ms", "mean_tpot_ms", "request_throughput"):
            if k in data:
                row[k] = round(data[k], 2)
    return row if len(row) > 1 else None


def main() -> None:
    rows = [r for d in sorted(RESULTS.iterdir()) if d.is_dir() and (r := parse_run(d))]
    if not rows:
        print("no results found")
        return
    cols = ["tag", "n_rebalance", "rearrange_mean_s", "rearrange_p95_s",
            "rearrange_max_s", "p99_ttft_ms", "p99_tpot_ms", "p99_e2el_ms",
            "request_throughput"]
    widths = {c: max(len(c), *(len(str(r.get(c, "-"))) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "-")).ljust(widths[c]) for c in cols))
    with open(RESULTS / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved -> {RESULTS / 'summary.csv'}")


if __name__ == "__main__":
    main()
