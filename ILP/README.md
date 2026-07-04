### ILP mathematical model
#### Sets
Instructions:
$$
I = \{0, 1, \ldots, n - 1\}
$$

Instructions should be divided into batches:
$$
B = \{0, 1, \ldots, K - 1\}
$$
where `K = max_batch_num`.

The required `network` section defines an ECMP topology. In the multi-GPU
inference setting, each physical link represents a bandwidth-constrained hop in
the GPU communication fabric, such as an NVLink/NVSwitch link. The solver
treats undirected physical links as two directed links for load accounting:

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

##### ECMP link load and batch time

For a given batch `b` and directed link `e`, the load of `e` is the ECMP split
traffic from all migrations assigned to that batch:

$$
link\_load_{e,b}
=
\sum_{i \in I} f_{i,e} x_{i,b}
$$

The batch execution time is determined by the most loaded normalized link:

$$
T_b
=
\max_{e \in E}
\frac{link\_load_{e,b}}{link\_capacity_e}
$$

#### Objective function

Minimize total migration execution time across batches:

$$
\min \sum_{b \in B} T_b
$$


### Running the script


```bash
.venv/bin/python ILP/eplb_migration_ilp.py ILP/example_hotspot.json
```


```bash
.venv/bin/python ILP/eplb_migration_ilp.py ILP/example_hotspot.json > ILP/example_solution.json
```


```bash
.venv/bin/python ILP/eplb_migration_ilp.py ILP/example_hotspot.json --visualize-png ILP/example_hotspot.png
```

With a `network` section, this writes three PNGs:
`example_hotspot_before.png`, `example_hotspot_after.png`, and
`example_hotspot_links.png`.
