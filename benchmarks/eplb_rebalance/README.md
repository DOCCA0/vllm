# EPLB Migration Batching Benchmark

Benchmark for the migration batching feature (see `feature.md`):
baseline (`off`) vs `first_fit` vs `degree_desc`. Scripts are numbered in
workflow order:

| script | purpose | where |
|---|---|---|
| `01_setup.sh` | venv + vllm + model download | every node |
| `02_firewall.sh` | allow VLAN traffic between nodes | laptop |
| `03_cluster_up.sh` | start/stop the 4-node Ray cluster | laptop |
| `04_run.sh` | one experiment group (serve -> skew -> bench -> collect) | head node / WSL |
| `05_all.sh` | 3-group comparison + report | head node |
| `06_report.py` | aggregate `results/` into table + `summary.csv` | anywhere |
| `nic_sampler.sh` | NIC throughput sampler (invoked remotely by `04_run.sh`) | helper |

## Single Node (laptop / WSL smoke test)

One GPU, small MoE, EPLB off (it requires TP*DP>1). Only smoke-tests the
server/traffic/report pipeline:

```bash
bash 01_setup.sh    # venv + vllm + tiny-random/qwen3-moe
bash 04_run.sh      # MODE=single by default, ~10 min
./06_report.py
```

## Multi Node (4-node VLAN cluster)

Four nodes, one GPU each, TP=4 + expert parallel (`--enable-expert-parallel`,
so the EPLB group is EP=4) over TCP Ethernet: every cross-GPU transfer goes
through the NIC. Update `NODES` in `02_firewall.sh` / `03_cluster_up.sh` /
`04_run.sh` (or export `CLUSTER_NODES`) when the lease changes.

```bash
# 0. On EVERY node:
bash 01_setup.sh

# 1. From the laptop:
bash 02_firewall.sh <FLOAT_IP>
IFACE=<iface> bash 03_cluster_up.sh <FLOAT_IP>     # `... <FLOAT_IP> down` to stop

# 2. On the head node (check iface name with `ip -br addr` first):
MODE=cluster IFACE=<iface> bash 05_all.sh          # small-model smoke
MODE=cluster IFACE=<iface> MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  NUM_REDUNDANT=32 bash 05_all.sh                  # 30B live run
```

`05_all.sh` runs three groups (`off` / `first_fit` / `degree_desc`) and
prints a comparison table; per-group logs land in
`results/multi_<policy>_<model>/`, the aggregate in `results/summary.csv`.

## Hot-spot construction and metrics

- **Hot GPU**: phase A of `04_run.sh` sends requests with a ~98% shared
  prefix (`PREFIX_LEN` + `SKEW_INPUT_LEN`), concentrating token routing on a
  few experts per layer. After rearrangement, the ranks holding those
  experts become migration hot spots.
- **Per-rank stats**: `04_run.sh` sets `VLLM_EPLB_LOG_MIGRATION_STATS=1` in
  cluster mode, so `server.log` contains one line per layer per rebalance:
  `EPLB migration stats: N transfers, B batches, per-rank peak=.. hotspot_ratio=..`
  (peak/mean across ranks; 1.0 = perfectly balanced).
- **NIC throughput**: with `IFACE` set, `04_run.sh` samples
  `/sys/class/net/<iface>/statistics` on every node and stores
  `nic_<ip>.txt` (rx/tx MB/s) per group; the report shows the peak.
- **Before/after**: compare `rearrange_mean/p95_s`, `hotspot_max`,
  `nic_peak_MBps` and `p99_tpot_ms` across the three groups.
- **ILP lower bound**: `ILP/eplb_migration_ilp.py` solves the optimal
  batching offline for comparison with the greedy schedules.

Key knobs (env vars): `EPLB_STEP`, `EPLB_WINDOW`, `NUM_REDUNDANT`,
`NUM_PROMPTS`, `CONCURRENCY`, `PREFIX_LEN`, `SKEW_INPUT_LEN`.
