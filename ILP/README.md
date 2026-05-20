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

Ranks are inferred from all distinct source and destination ranks in the
migration instructions:

$$
R = \{src_i \mid i \in I\} \cup \{dst_i \mid i \in I\}
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

`hotspot_n` is the per-rank no-hotspot load threshold inside one batch.

#### Variables

$$
x_{i,b} =
\begin{cases}
1, & \text{migration } i \text{ is placed in batch } b \\
0, & \text{otherwise}
\end{cases}
$$

$$
o_{r,b} \ge 0
$$

`o_{r,b}` is the hotspot overflow of rank `r` in batch `b`.

#### Constraints

##### Each migration must be executed once

Each migration must select exactly one batch:

$$
\sum_{b \in B} x_{i,b} = 1,
\quad \forall i \in I
$$

##### Rank load and hotspot overflow

For a given batch `b` and rank `r`, the load of `r` in `b` is its total send
and receive migration load:

$$
load_{r,b}
=
\sum_{i: src_i = r} x_{i,b}
+
\sum_{i: dst_i = r} x_{i,b}
$$

The overflow variable covers the amount above the no-hotspot threshold:

$$
load_{r,b} - o_{r,b} \le hotspot\_n,
\quad \forall r \in R, b \in B
$$

$$
o_{r,b} \ge 0,
\quad \forall r \in R, b \in B
$$

Equivalently:

$$
o_{r,b} = \max(0, load_{r,b} - hotspot\_n)
$$

#### Objective function

Minimize total hotspot overflow across all ranks and batches:

$$
\min \sum_{b \in B} \sum_{r \in R} o_{r,b}
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
