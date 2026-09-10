# Dense membership candidate

Local WSL experiment, 2026-09-10. The production scheduler and PR are unchanged.
See `explore_scheduler_cost.py` for the prototypes and exact equality checks.

Measured with CPU PyTorch profiler, 300 calls per placement per variant:

| 120 transfers | P50 ms/call | P99 ms/call |
| --- | ---: | ---: |
| Current | 0.08499 | 0.17309 |
| Python rank masks + cached assignment/grouping | 0.05132 | 0.12375 |
| Python rank masks + cached assignment, fresh grouping | 0.05538 | 0.16505 |
| Python rank masks + static pair batches | 0.05443 | 0.22654 |
| NumPy rank masks + cached assignment/grouping | 0.04793 | 0.13927 |
| Python dense masks | 0.06013 | 0.15135 |

`trace.json` and `summary.csv` are the generated raw trace and derived statistics.
The NumPy candidate passes exact schedule equality on 500 randomized placement
pairs across 2, 4, 5, 8 and 16 ranks, plus the six saved real placements.

A separate local diagnostic used 1000 varying four-rank placements: 128 unique
experts plus 16 replicas shuffled across 144 slots, seed 4. Wall-clock P50/P99
in microseconds were current 80.705/164.368, Python masks 64.046/119.906 and
NumPy masks 40.206/167.477. These diagnostic numbers came from perf_counter_ns,
not the PyTorch trace. All 1000 outputs matched exactly. Grouping cache had
0 hits/1000 misses; assignment cache had 96636 hits/50 misses. Thus the candidate
does not need repeated placement layouts to improve median cost. Tail latency
did not improve in this diagnostic and requires more measurement.

An indexed-update alternative to `np.bitwise_or.at` was slower for the 144-slot
membership microcheck (4.283 versus 1.845 microseconds), so was not selected.
Precomputing a static rank-pair batch mapping was also slower than cached greedy
grouping at 120 transfers and had substantially worse P99. It can additionally
retain unnecessary rounds for sparse communication graphs, so it was rejected.

The remote A/B used identical four-rank async NIXL serving commands and the
same 8-request profiled random workload (prefix 300, input 100, output 300).
The successful run contained 61 scheduler calls per rank inside the inference
window. Pooling the four ranks gave:

| Scheduler | P50 ms/call | P99 ms/call | Mean ms/call |
| --- | ---: | ---: | ---: |
| Current | 0.57986 | 1.10290 | 0.61137 |
| NumPy dense candidate | 0.51562 | 0.91829 | 0.53681 |

The candidate reduced serving P50 by 11.1% and P99 by 16.7%. This is much less
than the isolated P50 reduction because serving measurements include real CPU
contention and cache state. One candidate run encountered a NIXL transfer
stall after all four ranks had completed the same 93 scheduling calls; an
identical retry completed, including profiler shutdown. The incomplete run is
not used in the table.

Raw successful traces and benchmark logs are in
`../../serving_trace_20260910/{baseline_retry,dense_candidate_retry}`.
The added NumPy and caching complexity is not justified by the modest serving
improvement, so that candidate was rejected. The complete operation still has
lower bounds of Omega(P) to inspect placement slots and Omega(M) to emit
migrations; it cannot be constant time for arbitrary input.

## Bulk precomputation follow-up

A simpler candidate keeps `schedule_migration_batches` unchanged and computes
all 48 layer schedules contiguously at the beginning of each async EPLB cycle.
Layer transfers then consume the precomputed batches. Transfer assignment,
ordering, and batch boundaries are unchanged.

The table normalizes each NVTX range by the number of layer schedules it built:
one layer for the current path and 48 layers for the bulk path. It uses every
scheduler range in each complete trace, including ranges outside the request
execution interval.

| Run | Rank 0 | Rank 1 | Rank 2 | Rank 3 | Four-rank pooled |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current ms/layer | 0.5984 | 0.5703 | 0.5976 | 0.6452 | 0.6029 |
| Bulk run 1 ms/layer | 0.2077 | 0.1450 | 0.1516 | 0.1462 | 0.1626 |
| Bulk run 1 reduction | 65.3% | 74.6% | 74.6% | 77.3% | 73.0% |
| Bulk run 2 ms/layer | 0.2238 | 0.1445 | 0.1449 | 0.1508 | 0.1660 |
| Bulk run 2 reduction | 62.6% | 74.7% | 75.7% | 76.6% | 72.5% |

Both runs exceed the 50% target on every rank. The bulk ranges were 6.9--10.7
ms per 48-layer cycle, rather than 0.57--0.65 ms for each scattered layer call.
This amortizes cache and async-thread scheduling costs without changing the
greedy algorithm. Raw artifacts are in
`../../serving_trace_20260910/{bulk_precompute_retry,bulk_precompute_repeat2}`.
