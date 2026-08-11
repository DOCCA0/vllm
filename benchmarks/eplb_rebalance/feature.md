# EPLB Migration Batching

## Motivation

During EPLB rebalancing, remote expert weight transfers are currently posted
all at once: each rank may participate in many simultaneous P2P sends and
receives. Under heavy rebalancing this concentrates traffic on a few ranks and
creates NIC/RDMA hot spots, increasing tail latency of the rebalance step.

## Feature

This change adds optional **migration batching**: remote transfers are grouped
into sequential waves (batches) where no rank participates in more than one
distinct rank-pair flow at a time. Expert transfers with the same source and
destination are coalesced into one flow and submitted together. Within a wave,
all flows run concurrently; between waves there is a synchronization point.
This spreads network load over time without serializing every expert tensor.

Scheduling is done with a lightweight online **greedy edge coloring** of the
migration graph (ranks = vertices, transfers = directed edges). Coalesced flows
keep their deterministic input order and are placed into the first batch
without an endpoint conflict.

The offline ILP solver in `ILP/eplb_migration_ilp.py` serves as an optimal
baseline to quantify how close the greedy schedules are to the optimum.

## Changes

- **New scheduler module** (`vllm/distributed/eplb/migration_scheduler.py`):
  builds the remote migration instructions from the old/new expert placement
  maps (identical on every rank since the maps are replicated) and groups them
  into conflict-free batches via greedy edge coloring.
- **Execution path** (`vllm/distributed/eplb/rebalance_execute.py`): when
  batching is enabled, `move_to_buffer` replaces the single all-at-once P2P
  phase with one communicator execute per batch. Local copies (unchanged and
  intra-rank moves) are unaffected. When disabled, the original behavior is
  preserved.
- **Configuration** (`vllm/config/parallel.py`): two new `EPLBConfig` fields —
  `use_migration_batching` (default off) and
  `migration_batching_max_batches` (soft limit; exceeding it logs a warning but
  the full schedule still runs to keep correctness).
- **Wiring**: both the synchronous path (`EplbState.rearrange`) and the async
  worker path pass the new config options down to the transfer functions.
- **Tests** (`tests/distributed/test_eplb_migration_scheduler.py`): unit tests
  covering instruction building, conflict-freeness and completeness,
  end-to-end batched execution via a mock communicator, and the soft-limit
  warning.

## Trade-offs

- Fewer NIC hot spots and more predictable transfer concurrency.
- Cost: multiple synchronous P2P waves instead of one, so the rebalance step
  may take slightly longer.
