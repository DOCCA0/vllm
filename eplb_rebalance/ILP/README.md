### ILP mathematical model
#### Sets
Instructions:
$$
I = \{0, 1, \ldots, n - 1\}
$$

Instructions should be divided into batches:
$$
B = \{0, 1, \ldots, n - 1\}
$$

At most `n` batches are ever needed, since placing every migration in its own
batch is always feasible; unused batches take zero time and drop out of the
objective.

The required `network` section defines an ECMP topology. It can be given
either as an explicit `links` list, or as a three-tier hierarchical cluster
that the script expands automatically:

- **Layer 1 — GPU**: ranks `0 .. num_gpus - 1`, the compute/HBM units.
- **Layer 2 — node**: every `gpus_per_node` GPUs attach to the node's
  `nvswitches_per_node` NVSwitch vertices (default 1); each GPU connects to
  all of them with `nvlink_capacity` links (default `18 × fabric`, i.e.
  900 GB/s NVLink 4 vs 50 GB/s NDR 400G). A single NVSwitch per node makes
  every pair's shortest path unique, so ECMP degenerates to 1.0 per link;
  two or more NVSwitches per node create equal-cost paths through each
  switch, which is what gives ECMP real alternatives to split over.
- **Layer 3 — cluster fabric**: the NVSwitches interconnect over
  IB/RoCE via one spine vertex with `fabric_capacity` links.

Migrations always reference GPU ranks; intra-node transfers then only
traverse NVLink links, while inter-node transfers additionally cross the
bandwidth-scarce fabric. The solver treats undirected physical links as two
directed links for load accounting:

$$
E = \{(u, v), (v, u) \mid \{u, v\} \text{ is a physical link}\}
$$

Migration instruction:
$$
i = (src_i, dst_i)
$$

Each migration has unit load because all experts are assumed to have the same
size:

$$
w_i = 1
$$

`link_capacity` is the normalized per-directed-link capacity inside one batch.

Rebalancing happens while the cluster keeps serving inference, so each link
also carries sustained background traffic (MoE all-to-all, KV transfers).
This is modeled as a reserved bandwidth fraction

$$
\rho_e \in [0, 1)
$$

per directed link (`inference_traffic`, per-tier with `nvlink`/`fabric` keys
or per-link `background`). It is a *parameter*, not a decision variable: the
scheduler cannot control inference traffic, and a free variable would simply
be driven to zero by the objective. The capacity left for migrations is the
effective capacity

$$
\widehat{link\_capacity}_e = link\_capacity_e \,(1 - \rho_e)
$$

For every migration `i`, fixed ECMP is precomputed from the shortest paths
between `src_i` and `dst_i`. The constant

$$
f_{i,e} \in [0, 1]
$$

is the fraction of migration `i`'s traffic that traverses directed link `e`
after equal splitting across all shortest paths.

#### Variables

$$
x_{i,b} =
\begin{cases}
1, & \text{migration } i \text{ is placed in batch } b \\
0, & \text{otherwise}
\end{cases}
$$

$$
T_b \ge 0
$$

`T_b` is the execution time of batch `b`, normalized by link capacity.

#### Constraints

##### Each migration must be executed once

Each migration must select exactly one batch:

$$
\sum_{b \in B} x_{i,b} = 1,
\quad \forall i \in I
$$

##### Hard per-batch link capacity

One wave cannot push more concurrent migration traffic over a link than its
effective capacity:

$$
link\_load_{e,b} \le link\_capacity_e (1 - \rho_e),
\quad \forall b \in B, \; \forall e \in E
$$

This is the only mechanism that forces migrations into multiple waves: a
purely soft time model (`T_b \ge link\_load_{e,b} / \widehat{cap}_e` alone)
always admits a single-wave optimum, because splitting traffic across waves
cannot reduce the total load on any bottleneck link. Note the hard capacity
is per directed link, so opposite-direction transfers over the same physical
link still share a wave.

##### ECMP link load and batch time

For a given batch `b` and directed link `e`, the load of `e` is the ECMP split
traffic from all migrations assigned to that batch:

$$
link\_load_{e,b}
=
\sum_{i \in I} f_{i,e} x_{i,b}
$$

The batch execution time is determined by the most loaded link, normalized by
its effective capacity:

$$
T_b
=
\max_{e \in E}
\frac{link\_load_{e,b}}{link\_capacity_e (1 - \rho_e)}
$$

#### Objective function

Minimize total migration execution time across batches:

$$
\min \sum_{b \in B} T_b
$$


### Running the script

```bash
.venv/bin/python ILP/eplb_migration_ilp.py ILP/example_hotspot.json --visualize-png ILP/example_hotspot.png
```
