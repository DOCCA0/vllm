# Four-node NIXL EPLB migration batching

This benchmark uses Ubuntu 22.04.5, four Quadro RTX 6000 24 GB GPUs (one per
node), Ray 2.56.1, NIXL 1.3.2, and 10 GbE without RDMA. The tested code is
commit `bfd9399157`. Prefix caching is enabled in every run.

The complete commands and raw profiling method are in
[COMMANDS.md](results/serving_nixl_20260910_cycle_precompute/COMMANDS.md).

## Cluster setup

Pull `ilp` from a Windows terminal:

```bash
ssh cc@192.5.87.75 "cd ~/vllm-ilp && git switch ilp && git pull --ff-only origin ilp"
ssh -J cc@192.5.87.75 cc@10.140.83.20  "cd ~/vllm-ilp && git switch ilp && git pull --ff-only origin ilp"
ssh -J cc@192.5.87.75 cc@10.140.83.141 "cd ~/vllm-ilp && git switch ilp && git pull --ff-only origin ilp"
ssh -J cc@192.5.87.75 cc@10.140.83.105 "cd ~/vllm-ilp && git switch ilp && git pull --ff-only origin ilp"
```

On every node:

```bash
cd ~/vllm-ilp
export PATH=$PWD/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=eno2 GLOO_SOCKET_IFNAME=eno2
export UCX_TLS=all UCX_NET_DEVICES=eno2 UCX_RCACHE_MAX_UNRELEASED=1024
ray stop --force
```

Start the head:

```bash
ray start --head --node-ip-address=10.31.0.89 --port=6379 \
  --num-gpus=1 --disable-usage-stats
```

Start workers with `NODE_IP=10.31.0.87`, `10.31.0.94`, and `10.31.0.91`:

```bash
ray start --address=10.31.0.89:6379 --node-ip-address="$NODE_IP" \
  --num-gpus=1 --disable-usage-stats
```

## Server and benchmark commands

Use a fresh server for every sync/async and off/on run:

```bash
cd ~/vllm-ilp
export PATH=$PWD/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=eno2 GLOO_SOCKET_IFNAME=eno2
export UCX_TLS=all UCX_NET_DEVICES=eno2 UCX_RCACHE_MAX_UNRELEASED=1024
export VLLM_EPLB_LOG_MIGRATION_STATS=1

MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
USE_ASYNC=true     # repeat with false
USE_BATCHING=true  # repeat with false
EPLB_CONFIG="{\"window_size\":50,\"step_interval\":100,\"num_redundant_experts\":16,\"log_balancedness\":false,\"use_async\":$USE_ASYNC,\"communicator\":\"nixl\",\"enable_migration_batching\":$USE_BATCHING}"

vllm serve "$MODEL" --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.8 --max-model-len 4096 \
  --enable-prefix-caching --tensor-parallel-size 4 \
  --distributed-executor-backend ray --enable-expert-parallel \
  --enable-eplb --eplb-config "$EPLB_CONFIG" --enforce-eager
```

Warm up each fresh server with eight requests:

```bash
vllm bench serve --backend vllm --model "$MODEL" --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 100 --random-output-len 300 \
  --num-prompts 8 --max-concurrency 8 --seed 123 --temperature 0 \
  --ignore-eos --disable-tqdm
```

### Random workload

```bash
vllm bench serve --backend vllm --model "$MODEL" --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 100 --random-output-len 300 \
  --num-prompts 200 --max-concurrency 32 --seed 0 --temperature 0 \
  --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,90,95,99 \
  --ignore-eos --disable-tqdm --save-result \
  --result-dir "$RESULTS" --result-filename bench.json
```

### Phased-English workload

The tracked JSONL contains 256 ordered English prompts in eight 32-request
phases. It was generated deterministically by
`bench_dataset/generate_phased_english.py`; ordering is preserved to make the
expert distribution change repeatedly.

```bash
vllm bench serve --backend vllm --model "$MODEL" --port 8000 \
  --dataset-name custom \
  --dataset-path benchmarks/eplb_rebalance/bench_dataset/eplb_phased_english_256.jsonl \
  --disable-shuffle --custom-output-len 300 --num-prompts 256 \
  --max-concurrency 32 --seed 0 --temperature 0 \
  --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,90,95,99 \
  --ignore-eos --disable-tqdm --save-result \
  --result-dir "$RESULTS" --result-filename bench.json
```

NIC P50/P99 is the head node's combined RX+TX rate, sampled from
`/sys/class/net/eno2/statistics/{rx_bytes,tx_bytes}` once per second. It is not
a `vllm bench serve` metric.

## End-to-end results

All requests succeeded. Raw JSON, logs, NIC samples, traces, and summaries are
under `results/serving_nixl_20260910_cycle_precompute/`.

### Random workload

| Mode | Batching | Duration (s) | Output (tok/s) | TTFT P50/P99 (ms) | TPOT P50/P99 (ms) | E2EL P50/P99 (ms) | NIC P50/P99 (MB/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sync | off | 2,312.78 | 25.94 | 29,399.95 / 48,798.73 | 1,021.99 / 1,136.92 | 340,081.87 / 354,373.88 | 107.67 / 635.12 |
| sync | on | 2,468.02 | 24.31 | 28,834.00 / 50,636.72 | 1,101.40 / 1,207.26 | 361,409.21 / 381,648.24 | 107.00 / 597.99 |
| async | off | 1,456.99 | 41.18 | 28,927.16 / 45,012.18 | 628.42 / 725.30 | 221,195.60 / 236,370.29 | 333.97 / 610.51 |
| async | on | 1,437.24 | 41.75 | 28,165.42 / 44,274.41 | 612.41 / 721.67 | 220,980.21 / 236,754.14 | 309.45 / 556.21 |

In async mode, batching raised throughput by 1.37%, reduced TTFT P50/P99 by
2.63%/1.64%, TPOT P50/P99 by 2.55%/0.50%, and NIC P50/P99 by 7.34%/8.90%.
E2EL P50 fell by 0.10%, while E2EL P99 rose by 0.16%. In sync mode, throughput
fell by 6.29% because conflict-free batches execute serially while inference is
paused.

### Phased-English workload

| Mode | Batching | Duration (s) | Output (tok/s) | TTFT P50/P99 (ms) | TPOT P50/P99 (ms) | E2EL P50/P99 (ms) | NIC P50/P99 (MB/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sync | off | 2,924.44 | 26.26 | 40,706.56 / 97,247.70 | 1,099.41 / 1,210.71 | 364,499.70 / 370,849.25 | 106.57 / 625.56 |
| sync | on | 3,041.21 | 25.25 | 40,502.61 / 99,280.08 | 1,146.72 / 1,256.01 | 378,877.16 / 386,795.63 | 109.14 / 608.98 |
| async | off | 1,900.43 | 40.41 | 40,619.13 / 61,971.27 | 676.54 / 780.30 | 237,177.93 / 245,352.56 | 310.42 / 626.69 |
| async | on | 1,883.00 | 40.79 | 40,666.30 / 62,801.18 | 660.08 / 772.75 | 233,319.74 / 247,475.39 | 291.51 / 557.20 |

In async mode, batching raised throughput by 0.93%, reduced TPOT P50/P99 by
2.43%/0.97%, E2EL P50 by 1.63%, and NIC P50/P99 by 6.09%/11.09%. TTFT
P50/P99 rose by 0.12%/1.34%, and E2EL P99 rose by 0.87%. In sync mode,
throughput fell by 3.84%.

Async EPLB is generally the relevant mode for production serving: migration
runs in the background while inference continues, so reducing NIC contention
reduces interference. Sync EPLB pauses inference, and splitting migration into
serial batches extends that pause. Sync users can retain the unbatched path with
`enable_migration_batching=false`.

The phased workload is closer to a changing production request mix. Random
token IDs change expert load less meaningfully, so they exercise migration
batching less consistently even when the number of transfers is similar.

## Scheduler profiling

### Per-layer calls versus one cycle-wide call

This direct short serving comparison used identical four-rank async NIXL server,
warm-up, and eight-request Random commands. The scheduling algorithm is the same;
only its call placement changes.

| Rank | Per-layer mean × 48 (ms) | One measured 48-layer call (ms) | Reduction |
| --- | ---: | ---: | ---: |
| 0 | 28.72 | 9.97 | 65.30% |
| 1 | 27.38 | 6.96 | 74.58% |
| 2 | 28.69 | 7.27 | 74.64% |
| 3 | 30.97 | 7.02 | 77.34% |

The first column multiplies the control's measured mean per-layer call by 48.
The second measures the complete call that builds all 48 schedules. It is not a
single-layer latency or an estimate derived from the new implementation.

### Full async serving traces and measured cost/saved ratio

The Random and Phased async batching-on cases above were captured end to end
with NVTX-only Nsight Systems on all four ranks. Search the reports for
`eplb: schedule migration batches`; every range is one call that builds all 48
layer schedules. Formal-run boundaries recorded alongside each benchmark allow
warm-up calls to be excluded exactly.

| Workload | Calls/rank | Per-rank cumulative cost (ms) | Conservative cost (ms) | Serving saved (ms) | Cost/saved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 13 | 88.75--130.22 | 130.22 | 19,747.50 | 0.659% |
| Phased English | 16 | 110.29--146.53 | 146.53 | 17,436.75 | 0.840% |

Ranks schedule concurrently, so their costs are not summed. The conservative
cost is the largest measured cumulative time among the four ranks. Serving time
saved is the measured batching-off duration minus the batching-on duration from
the same workload. Thus `cost/saved = max-rank cumulative scheduler time /
serving time saved`; neither the call count nor scheduler duration is estimated.

![Scheduler placement and measured cost/saved ratio](https://raw.githubusercontent.com/DOCCA0/vllm/refs/heads/ilp/benchmarks/eplb_rebalance/results/serving_nixl_20260910_cycle_precompute/scheduler_profile.png)

[Raw JSON, NIC samples, four-rank traces, summaries, and complete commands](https://github.com/DOCCA0/vllm/tree/ilp/benchmarks/eplb_rebalance/results/serving_nixl_20260910_cycle_precompute).
Open `random_async_on/trace/rank0.nsys-rep` or
`phased_async_on/trace/rank0.nsys-rep` with Nsight Systems 2024.5.1 or newer.
