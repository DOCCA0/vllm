### ILP mathematical model

This model separates two effects that must not be conflated:

- batching makes expert migration take at least as long as one unbatched
  `execute()`, because the batches run sequentially and every call has an
  overhead;
- batching can improve inference by reducing peak link and NIC contention
  while asynchronous migration is in progress.

The script reports the analytical unbatched lower bound and the ILP optimum.
It does not simulate a particular heuristic and does not convert network
contention into a fabricated latency prediction. The serving experiment in
PR #52641 provides the empirical connection: batching reduced head-node NIC
P99 from 599.81 MB/s to 538.40 MB/s and improved output throughput by 2.28%.

#### Sets and parameters

Migration instructions with the same directed `(src, dst)` pair are
coalesced into one flow, matching `migration_scheduler.py`:

$$
P = \{p_0, p_1, \ldots, p_{m-1}\}.
$$

There are at most `n` batches for `n` input migration instructions:

$$
B = \{0, 1, \ldots, n-1\}.
$$

The required `network` section defines a two-tier server topology:

- GPU ranks are grouped by `gpus_per_server`; the example uses two GPUs per
  server.
- GPUs in one server share a fast local path with capacity `local_capacity`.
- Every server has one slow link of capacity `network_capacity` to the shared
  inter-server network.

This is intentionally the smallest topology that captures the optimization
target: concurrent cross-server migrations contend on a server's slow network
link, while same-server migration remains cheap. An explicit `links` list is
still accepted for custom topologies.

For flow $p$ and directed link $e$:

$$
w_p = \text{number of experts in flow } p,
$$

$$
f_{p,e} \in [0,1]
$$

is the fixed routed fraction on link $e$, $C_e$ is the physical link capacity,
and $\rho_e \in [0,1)$ is the background inference utilization configured by
`inference_traffic` or per-link `background`.

`batch_overhead` is the normalized cost $\delta \ge 0$ of one
`communicator.execute()` call. It is required because batching invokes
`execute()` once per batch in `rebalance_execute.py`.

#### Variables

$$
x_{p,b} =
\begin{cases}
1, & \text{flow } p \text{ is assigned to batch } b,\\
0, & \text{otherwise},
\end{cases}
$$

$$
y_b =
\begin{cases}
1, & \text{batch } b \text{ is used},\\
0, & \text{otherwise},
\end{cases}
$$

$$
T_b \ge 0
$$

is the pure transfer time of batch $b$, and

$$
U \ge 0
$$

is the peak total link-utilization proxy seen by inference.

#### Constraints

Every flow executes exactly once:

$$
\sum_{b \in B} x_{p,b} = 1, \qquad \forall p \in P.
$$

The PR scheduler allows each rank to communicate with at most one peer in a
batch. If $r$ is an endpoint of flow $p$, written $r \in p$, then:

$$
\sum_{p: r \in p} x_{p,b} \le 1,
\qquad \forall r,\; \forall b \in B.
$$

Assignments activate their batch, and used batches form a contiguous prefix:

$$
x_{p,b} \le y_b,
\qquad
y_b \ge y_{b+1}.
$$

The migration load on link $e$ in batch $b$ is:

$$
L_{e,b} = \sum_{p \in P} w_p f_{p,e} x_{p,b}.
$$

Pure transfer time uses physical capacity, not capacity with inference
subtracted from it:

$$
T_b \ge \frac{L_{e,b}}{C_e},
\qquad \forall e,\; \forall b.
$$

The inference-contention proxy includes both background inference traffic and
concurrent migration traffic:

$$
U \ge \rho_e + \frac{L_{e,b}}{C_e},
\qquad \forall e,\; \forall b.
$$

Values above 1 mean offered traffic exceeds physical link capacity. Lower $U$
means less peak contention and more headroom for inference; it is not itself a
latency in seconds.

#### Why batching cannot shorten migration

The reported migration completion time is:

$$
M_{batch} = \sum_{b \in B} T_b + \delta \sum_{b \in B} y_b.
$$

The unbatched comparison is:

$$
M_{unbatched} =
\max_e \frac{\sum_p w_p f_{p,e}}{C_e} + \delta.
$$

Because the maximum of sums is no greater than the sum of maxima:

$$
\sum_b \max_e \frac{L_{e,b}}{C_e}
\ge
\max_e \sum_b \frac{L_{e,b}}{C_e},
$$

therefore:

$$
M_{batch} \ge M_{unbatched} + \delta(K-1)
\ge M_{unbatched},
$$

where $K = \sum_b y_b$. This removes the physically impossible result where
enabling sequential batching appears to reduce expert-migration time.

#### Lexicographic objective

The solver runs three MILPs in sequence:

1. Minimize the number of batches $K = \sum_b y_b$.
2. With $K$ fixed, minimize peak contention $U$.
3. With $K$ and $U$ fixed, minimize pure transfer time $\sum_b T_b$.

The result is the theoretical optimum under the rank-disjoint batching
constraints. Comparing this optimum with the analytical unbatched lower bound
is sufficient to show that peak inference contention is optimizable, while
the migration-time proof above prevents claiming that batching accelerates
the migration itself.

### Running the script

```bash
.venv/bin/python benchmarks/eplb_rebalance/ILP/eplb_migration_ilp.py \
  benchmarks/eplb_rebalance/ILP/example_hotspot.json \
  --visualize-png benchmarks/eplb_rebalance/ILP/example_hotspot.png
```
