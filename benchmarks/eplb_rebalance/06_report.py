#!/usr/bin/env python3
"""Aggregate results/*/ into a comparison table and summary.csv."""
import csv
import json
import re
import statistics as st
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
RE_REARRANGE = re.compile(r"Rearranged experts.*?in ([0-9.]+) s")
RE_STATS = re.compile(
    r"EPLB migration stats: (\d+) transfers, (\d+) batches, "
    r"per-rank peak=(\d+) mean=([0-9.]+) hotspot_ratio=([0-9.]+)"
)


def _p95(xs: list[float]) -> float:
    return sorted(xs)[int(0.95 * (len(xs) - 1))]


def parse_run(d: Path) -> dict | None:
    row = {"tag": d.name}
    server_log = d / "server.log"
    if server_log.exists():
        text = server_log.read_text()
        times = [float(t) for t in RE_REARRANGE.findall(text)]
        if times:
            row.update(
                n_rebalance=len(times),
                rearrange_mean_s=round(st.mean(times), 3),
                rearrange_p95_s=round(_p95(times), 3),
            )
        stats = [(int(m[0]), int(m[1]), float(m[4])) for m in RE_STATS.findall(text)]
        if stats:
            row.update(
                transfers_mean=round(st.mean(s[0] for s in stats), 1),
                batches_max=max(s[1] for s in stats),
                hotspot_max=round(max(s[2] for s in stats), 2),
            )
    bench = d / "bench.json"
    if bench.exists():
        data = json.loads(bench.read_text())
        for k in ("p99_ttft_ms", "p99_tpot_ms", "p99_e2el_ms"):
            if k in data:
                row[k] = round(data[k], 2)
    nic_peaks = []
    for f in d.glob("nic_*.txt"):
        rates = [
            float(r) + float(t)
            for _, r, t in (line.split() for line in f.read_text().splitlines())
        ]
        if rates:
            nic_peaks.append(max(rates))
    if nic_peaks:
        row["nic_peak_MBps"] = round(max(nic_peaks), 1)
    return row if len(row) > 1 else None


def main() -> None:
    rows = [r for d in sorted(RESULTS.iterdir()) if d.is_dir() and (r := parse_run(d))]
    if not rows:
        print("no results found")
        return
    cols = ["tag", "n_rebalance", "rearrange_mean_s", "rearrange_p95_s",
            "transfers_mean", "batches_max", "hotspot_max", "nic_peak_MBps",
            "p99_ttft_ms", "p99_tpot_ms", "p99_e2el_ms"]
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
