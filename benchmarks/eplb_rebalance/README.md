# EPLB Migration Batching Benchmark

## Single Node (WSL smoke test, EPLB off)

```bash
cd benchmarks/eplb_rebalance

bash 01_setup.sh tiny-random/qwen3-moe

MODE=single \
MODEL=tiny-random/qwen3-moe \
GPU_MEM_UTIL=0.80 MAX_LEN=2048 \
NUM_PROMPTS=200 CONCURRENCY=8 \
INPUT_LEN=512 OUTPUT_LEN=128 PREFIX_LEN=384 SKEW_INPUT_LEN=8 \
bash 04_run.sh

./06_report.py
```

## Multi Node (4-node VLAN cluster, TP=4 + EP=4)

Topology: 4 nodes x 1 GPU, TP=4 with `--enable-expert-parallel` (EPLB group
is EP=4), NCCL over TCP Ethernet — every cross-GPU transfer goes through the
NIC. A 24GB card cannot hold a 30B FP16 replica, so DP=4 is not an option.

`CLUSTER_NODES` (space-separated node IPs) is required everywhere below.

### A. tiny-random/qwen3-moe smoke (fast pipeline validation)

1. Every node:

```bash
bash 01_setup.sh tiny-random/qwen3-moe
```

2. Laptop (bring the cluster up):

```bash
CLUSTER_NODES="ip1 ip2 ip3 ip4" \
  bash 02_firewall.sh <FLOAT_IP>

CLUSTER_NODES="ip1 ip2 ip3 ip4" \
IFACE=<vlan_iface> \
  bash 03_cluster_up.sh <FLOAT_IP>          # add `down` to tear down
```

3. Head node (3-group comparison):

```bash
MODE=cluster CLUSTER_NODES="ip1 ip2 ip3 ip4" IFACE=<vlan_iface> \
MODEL=tiny-random/qwen3-moe \
TP=4 DTYPE=float16 \
EPLB_STEP=200 EPLB_WINDOW=1000 NUM_REDUNDANT=8 \
NUM_PROMPTS=1000 CONCURRENCY=32 \
INPUT_LEN=512 OUTPUT_LEN=128 PREFIX_LEN=384 SKEW_INPUT_LEN=8 \
GPU_MEM_UTIL=0.90 MAX_LEN=4096 \
bash 05_all.sh
```

### B. Qwen3-30B live run

1. Every node (skip if step A already set up the venv):

```bash
bash 01_setup.sh Qwen/Qwen3-30B-A3B-Instruct-2507
```

2. Laptop (skip if the cluster from step A is still up; otherwise re-run
   `02_firewall.sh` / `03_cluster_up.sh` as above).

3. Head node (3-group comparison):

```bash
MODE=cluster CLUSTER_NODES="ip1 ip2 ip3 ip4" IFACE=<vlan_iface> \
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
TP=4 DTYPE=float16 \
EPLB_STEP=200 EPLB_WINDOW=1000 NUM_REDUNDANT=32 \
NUM_PROMPTS=1000 CONCURRENCY=32 \
INPUT_LEN=512 OUTPUT_LEN=128 PREFIX_LEN=384 SKEW_INPUT_LEN=8 \
GPU_MEM_UTIL=0.90 MAX_LEN=4096 \
bash 05_all.sh
```

Results: `results/multi_<off|first_fit|degree_desc>_<model>/` per group,
`results/summary.csv` aggregate.

