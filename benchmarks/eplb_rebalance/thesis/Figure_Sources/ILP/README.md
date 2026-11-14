### ILP mathematical model

The model uses two lexicographic objectives. The primary objective is:

$$
\min T_{\mathrm{infer}}
$$

where $T_{\mathrm{infer}}$ is the worst **modeled** inference time while expert
migration is running, not a measured latency prediction. Only after fixing its optimum does the solver minimize
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

### Literature and units

The example adopts the **100 Gb/s intra-server and 50 Gb/s inter-server**
bandwidths reported in Section V-A of [DEMI, IWQoS 2025](https://i.cs.hku.hk/~cwu/papers/zzhang-iwqos25.pdf).
It is a reduced topology inspired by those bandwidths, not a reproduction of
that paper's 16-GPU setup or the four-node Qwen serving experiment.

The formula uses normalized **capacity per reference interval**, not GB/s.
DEMI Section III-B likewise distinguishes per-slot capacity from bandwidth.
We normalize to the reported inter-server bandwidth, 50 Gb/s, giving:

$$
C_e=\begin{cases}100/50=2,&\text{local link},\\50/50=1,&\text{network link}.\end{cases}
$$

One payload unit is one equal-sized expert. For a concrete scale only, DEMI
reports 7.64 MB per expert; interpreting MB as decimal gives a reference interval
of $7.64\times10^6\times8/(50\times10^9)=1.2224$ ms. Thus $C_e$ counts expert
payloads transferable in that interval and $L_{e,b}/C_e$ is dimensionless.
All reported model times are in reference-interval units. $T_0=1$ and the
per-batch overhead 0.05 are illustrative assumptions, not measured latencies.
The original 18:2 had no cited provenance. 9:1 preserves that original ratio,
but halves capacity per interval and therefore changes the modeled slowdown.

**No universal inference bandwidth fraction was found.** The retained
$\rho=0.6$ is explicitly an assumed scenario, not a literature measurement.
`rho_sensitivity.csv` repeats the solve at 0, 0.3, 0.6, and 0.9; all retain
three batches and the same optimal peak load. A uniform $\rho$ only rescales
the primary objective, so these checks do not establish robustness to
heterogeneous, time-varying foreground traffic.

[Varys, SIGCOMM 2014, Section 5.1](https://istc-cc.cmu.edu/publications/papers/2014/varys-sigcomm14.pdf)
supports the bottleneck and residual-bandwidth abstraction, not the exact
linear inference-slowdown law below. [Network Calculus, Theorem 1.4.6](https://leboudec.github.io/netcal/latex/netCalBook.pdf)
explains why serially traversed, pipelined links combine bottleneck rate with
additive fixed latency. We neglect fixed per-hop latency and assume ideal
pipelining; we do not assume that an entire expert must be buffered at each hop.
The affine slowdown law is our illustrative surrogate, not a theorem from
either reference. [DEMI](https://i.cs.hku.hk/~cwu/papers/zzhang-iwqos25.pdf)
also models expert migration and token traffic jointly, but optimizes training
and does not validate this serving-latency surrogate.

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
| Modeled inference time | 8.50 | 3.50 |
| Diagnostic migration time | 3.05 | 3.15 |
| Peak normalized link load (not utilization) | 3.0 | 1.0 |
| Batches | 1 | 3 |

### Run

```bash
.venv/bin/python benchmarks/eplb_rebalance/ILP/reproduce_example.py
```

This writes the solution JSON, assumed-load sensitivity CSV, and all three PNGs.
It also writes individual batch PNGs for the existing slide layout. An independent
enumeration of all 4,140 set partitions of the eight migrations confirmed the
lexicographic optimum (peak normalized load 1.0, three batches). The solver's
variable upper bound includes residual capacity, so high-$\rho$ scenarios are
not inadvertently excluded.
For a single solve:

```bash
.venv/bin/python benchmarks/eplb_rebalance/ILP/eplb_migration_ilp.py \
  benchmarks/eplb_rebalance/ILP/example_hotspot.json \
  --visualize-png benchmarks/eplb_rebalance/ILP/example_hotspot.png
```
