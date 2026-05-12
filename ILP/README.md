### ILP mathematical model
#### Sets
Instructions:
$$
I = \{0, 1, \ldots, n - 1\}
$$

Instructions shoul be divided into batches:
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
#### Variables

$$
x_{i,b} =
\begin{cases}
1, & \text{migration } i \text{ is placed in batch } b \\
0, & \text{otherwise}
\end{cases}
$$

$$
z_{b,l} =
\begin{cases}
1, & \text{the maximum rank participation count of batch } b \text{ is } l \\
0, & \text{otherwise}
\end{cases}
$$

where `l` ranges over `0..n`.

#### Constraints

##### Each migration must be executed once

Each migration must select exactly one batch:

$$
\sum_{b \in B} x_{i,b} = 1,
\quad \forall i \in I
$$

##### Each batch can have only one hotspot level

$$
\sum_{l=0}^{n} z_{b,l} = 1,
\quad \forall b \in B
$$

##### The hotspot level covers each rank's participation count

For a given batch `b` and rank `r`, the participation count of `r` in `b` is:

$$
participation_{r,b}
=
\sum_{i: src_i = r} x_{i,b}
+
\sum_{i: dst_i = r} x_{i,b}
$$

The participation count must be no greater than the batch's hotspot level:

$$
participation_{r,b}
\le
\sum_{l=0}^{n} l \cdot z_{b,l},
\quad \forall r \in R, b \in B
$$
##### Batch cost
- Important assumption: `hotspot_n` is the no-hotspot threshold. A batch whose
  hottest rank has no more than `hotspot_n` source/destination participations
  keeps the normal linear cost `l`. If the hottest rank has more than
  `hotspot_n` participations, every extra participation adds 10% overhead on
  top of the normal linear cost.

$$
cost(l) =
\begin{cases}
0, & l = 0 \\
l, & 0 < l \le hotspot\_n \\
l + 0.1(l - hotspot\_n), & l > hotspot\_n
\end{cases}
$$


Therefore:

$$
cost_b = \sum_{l=0}^{n} cost(l)z_{b,l}
$$

#### <font color="#f79646">Objective function</font>

Minimize the total cost of all non-empty batches:

$$
\min \sum_{b \in B} cost_b
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
