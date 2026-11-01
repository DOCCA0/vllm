### ILP mathematical model

The model uses two lexicographic objectives. The primary objective is:

$$
\min T_{\mathrm{infer}}
$$

where $T_{\mathrm{infer}}$ is the worst inference time observed while expert
migration is running. Only after fixing its optimum does the solver minimize
the number of batches. This strict ordering does not trade inference time for
fewer batches and requires no arbitrary weights.

### Network

The example uses the smallest topology needed to express network contention:

- each server has two GPUs;
- same-server transfers use a fast link with capacity $C_{local}$;
- cross-server transfers use each server's slow network link with capacity
  $C_{network}$.

For each routed link $e$, $C_e$ is its capacity. For migration flow $p$,
$f_{p,e}$ is the fraction of its traffic routed through $e$. In the example
there is one path, so $f_{p,e}$ is either 0 or 1.

Migration instructions with the same directed `(src, dst)` pair are
coalesced into one flow, matching `migration_scheduler.py`. Its size is:

$$
w_p = \text{number of experts in flow }p.
$$

### Variables

Let $P$ be the set of flows and $B=\{0,\ldots,n-1\}$ be the candidate
batches.

$$
x_{p,b} =
\begin{cases}
1, & \text{flow }p\text{ is assigned to batch }b,\\
0, & \text{otherwise},
\end{cases}
$$

$$
y_b =
\begin{cases}
1, & \text{batch }b\text{ is used},\\
0, & \text{otherwise}.
\end{cases}
$$

$T_{\mathrm{infer}}$ is the continuous inference-time variable.

### Constraints

Every flow runs exactly once:

$$
\sum_{b\in B}x_{p,b}=1,
\qquad \forall p\in P.
$$

Assignments activate their batch, and active batches form a prefix:

$$
x_{p,b}\le y_b,
\qquad
y_b\ge y_{b+1}.
$$

The migration load on link $e$ in batch $b$ is:

$$
L_{e,b}=\sum_{p\in P}w_pf_{p,e}x_{p,b}.
$$

Let $T_0$ be `base_inference_time`, the inference time without migration, and
let $\rho=$ `inference_traffic` be the fraction of link capacity already used
by inference. Migrations therefore contend for the remaining fraction
$1-\rho$. The model assumes inference slowdown grows linearly with migration
load relative to this remaining capacity:

$$
T_{\mathrm{infer}}\ge
T_0\left(1+\frac{L_{e,b}}{C_e(1-\rho)}\right),
\qquad \forall e,\forall b.
$$

Since this constraint covers every link and batch, minimizing
$T_{\mathrm{infer}}$ is equivalent to:

$$
\min T_0\left(
1+\max_{e,b}\frac{L_{e,b}}{C_e(1-\rho)}
\right).
$$

The first solve produces $T_{\mathrm{infer}}^*$. The second solve preserves
that optimum and removes unnecessary batches:

$$
B^*=\min\sum_{b\in B}y_b
\quad\text{s.t.}\quad
T_{\mathrm{infer}}\le T_{\mathrm{infer}}^*+\epsilon.
$$

Thus the priorities are: protect inference first, then avoid unnecessary fixed
per-batch execution and synchronization overhead. Minimizing batch count is
only a tie-break and is not claimed to minimize complete migration time.

### Migration-time diagnostic

Migration time is reported for validation but is not optimized. For batch
$b$:

$$
T_b=\max_e\frac{L_{e,b}}{C_e}.
$$

With per-call overhead $\delta=$ `batch_overhead`, the reported total is:

$$
T_{\mathrm{migration}}=\sum_bT_b+\delta\sum_by_b.
$$

This diagnostic separates variable transfer time from fixed per-batch
overhead. It is reported for validation but is not part of the lexicographic
objectives.

### Example result

For `example_hotspot.json`:

| Metric | Unbatched | ILP optimal |
|---|---:|---:|
| Inference time | 4.75 | 2.25 |
| Migration time | 1.55 | 1.65 |
| Peak link utilization | 1.5 | 0.5 |
| Batches | 1 | 3 |

### Run

```bash
.venv/bin/python benchmarks/eplb_rebalance/ILP/eplb_migration_ilp.py \
  benchmarks/eplb_rebalance/ILP/example_hotspot.json \
  --visualize-png benchmarks/eplb_rebalance/ILP/example_hotspot.png
```
