# Local scheduler exploration

Run on local WSL, not the GPU servers. The original scheduler remains unchanged.

```bash
.venv/bin/python benchmarks/eplb_rebalance/explore_scheduler_cost.py
```

The experiment uses the six saved placements, 300 calls per implementation per
placement, randomized implementation order, and CPU PyTorch profiler ranges.
`trace.json` contains the raw trace; `summary.csv` contains milliseconds/call.
500 random placement pairs across 2, 4, 5, 8 and 16 ranks pass exact schedule
equality, including replica assignment, expert order and batch boundaries.

Variants:

- `current`: current implementation.
- `mask_cached`: rank-membership bit masks, bounded memoization of replica
  assignment and of grouping for an ordered tuple of directed rank pairs.
- `mask_uncached_groups`: same membership/assignment optimization, but computes
  the greedy grouping on every call.

At 120 transfers, observed P50 was 0.1088 / 0.0699 / 0.0769 ms respectively.
These are warm-cache results on repeated placements. Grouping cache hits on
changing serving placements are not established. CPU clocks, Python runtime,
instrumentation and local contention can affect these small timings.

Let P be physical placement slots, U distinct migrating expert IDs, M emitted
expert transfers, R ranks, F directed rank pairs and B batches. The current
algorithm scans placements, sorts migrating IDs and groups flows: expected
O(P + U log U + M + F B) under ordinary hash-table assumptions for fixed R.
F is at most R(R-1). Scanning arbitrary supplied placements requires Omega(P),
and explicitly emitting transfers requires Omega(M). Thus the complete API
cannot become constant time for unbounded input/output sizes.

Cached source assignment avoids repeated construction of rank lists and divmod
loops. Cached grouping avoids the F B search on hits, but constructing/hashing
the key and materializing flows still takes work. Full-placement memoization
would need an O(P) key or reliable upstream versioning, and would mostly help
identical placements. Static rank-pair rounds can eliminate online greedy
search, but may introduce more batches for sparse graphs and change transfer
order. Neither establishes constant-time end-to-end migration scheduling.

Follow-up work tested NumPy dense masks, Python dense masks, and static
rank-pair batches. It also collected a real four-rank async NIXL serving A/B.
See `dense_comparison/NOTES.md` and
`../serving_trace_20260910/{baseline_retry,dense_candidate_retry}`. The best
candidate reduced isolated P50 substantially but serving P50 by only 11.1%,
while adding complexity, so it was not selected for the production PR.
