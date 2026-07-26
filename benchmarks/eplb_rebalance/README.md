# EPLB Migration Batching Benchmark

Benchmark for the migration batching feature (see `feature.md`):
baseline (`off`) vs `first_fit` vs `degree_desc`, measuring rebalance
duration and inference tail latency.

## Single Node (WSL smoke test)

Validates the server/traffic/report pipeline on one GPU with the small
MoE model. EPLB is off (it requires TP*DP>1), so this only smoke-tests
the tooling.

```bash
bash benchmarks/eplb_rebalance/bench_setup.sh     # venv + vllm

# On a small GPU (e.g. 8GB WSL), use a tiny random MoE (Qwen1.5-MoE is
# 14.3B total params and does not fit):
MODE=single MODEL=tiny-random/qwen3-moe bash benchmarks/eplb_rebalance/bench_run.sh
```

Results in `benchmarks/eplb_rebalance/results/single_*/` (`server.log`,
`bench.json`, `bench_main.log`). Inspect with:

```bash
.venv/bin/python benchmarks/eplb_rebalance/bench_report.py
```

## Multi Node (4-node VLAN cluster)

Four nodes, one GPU each, TP=4 + expert parallel over TCP Ethernet
(every cross-GPU transfer goes through the NIC). All scripts run on the
**head node** unless noted.

```bash
# 0. On EVERY node: install env + model, allow VLAN traffic
bash benchmarks/eplb_rebalance/bench_setup.sh
./benchmarks/eplb_rebalance/bootstrap.sh <FLOAT_IP>          # firewall (any node)

# 1. Update NODES in cluster_up.sh, check the VLAN iface name
ip -br addr                                       # e.g. IFACE=enp1s0f1

# 2. Start the Ray cluster
./benchmarks/eplb_rebalance/cluster_up.sh <FLOAT_IP>         # `down` to tear down

# 3. Smoke test with the small model, then run the real benchmark
MODE=cluster IFACE=<iface> bash benchmarks/eplb_rebalance/bench_all.sh
MODE=cluster IFACE=<iface> MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  NUM_REDUNDANT=32 bash benchmarks/eplb_rebalance/bench_all.sh
```

`bench_all.sh` runs three groups (`off` / `first_fit` / `degree_desc`)
and prints a comparison table; per-group logs land in
`results/multi_<policy>_<model>/`, aggregate in `results/summary.csv`.

Key knobs (env vars): `EPLB_STEP`, `EPLB_WINDOW`, `NUM_REDUNDANT`,
`NUM_PROMPTS`, `CONCURRENCY`, `PREFIX_LEN`. See `bench_run.sh` header.
