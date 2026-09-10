# Four-node NIXL EPLB migration batching

This benchmark uses Ubuntu 22.04.5, four Quadro RTX 6000 24 GB GPUs (one per
node), Ray 2.56.1, NIXL 1.3.2, and 10 GbE without RDMA. The clean end-to-end
runs used commit `a4f561e19a`. Prefix caching is enabled in every run.

The complete end-to-end commands are in
[COMMANDS.md](results/serving_nixl_20260911_clean_e2e/COMMANDS.md). Profiling
commands are documented separately under
`results/serving_nixl_20260910_cycle_precompute/`.

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
export VLLM_EPLB_LOG_MIGRATION_STATS=0

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

All requests succeeded. These are single-run paired comparisons without
profiling or diagnostic migration logging. Raw JSON, logs, NIC samples, and the
summary are under `results/serving_nixl_20260911_clean_e2e/`.

### Random workload

| Mode | Batching | Duration (s) | Output (tok/s) | TTFT P50/P99 (ms) | TPOT P50/P99 (ms) | E2EL P50/P99 (ms) | NIC P50/P99 (MB/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sync | off | 2,300.06 | 26.09 | 28,861.61 / 49,425.56 | 1,026.97 / 1,128.56 | 339,940.86 / 354,637.96 | 106.94 / 659.27 |
| sync | on | 2,428.22 | 24.71 | 29,171.10 / 48,215.95 | 1,083.52 / 1,173.84 | 354,043.93 / 380,462.95 | 110.16 / 620.38 |
| async | off | 1,471.03 | 40.79 | 28,173.62 / 44,766.00 | 631.10 / 738.11 | 220,690.40 / 232,788.46 | 333.70 / 604.52 |
| async | on | 1,444.45 | 41.54 | 28,433.61 / 44,917.12 | 622.99 / 727.35 | 217,276.94 / 239,246.84 | 303.33 / 547.88 |

In async mode, batching raised throughput by 1.84%, reduced TPOT P50/P99 by
1.29%/1.46%, E2EL P50 by 1.55%, and NIC P50/P99 by 9.10%/9.37%. TTFT P50/P99
rose by 0.92%/0.34%, and E2EL P99 rose by 2.77%. In sync mode, throughput fell
by 5.28% because conflict-free batches execute serially while inference is
paused.

### Phased-English workload

| Mode | Batching | Duration (s) | Output (tok/s) | TTFT P50/P99 (ms) | TPOT P50/P99 (ms) | E2EL P50/P99 (ms) | NIC P50/P99 (MB/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sync | off | 2,921.35 | 26.29 | 40,449.89 / 99,029.81 | 1,099.07 / 1,209.36 | 365,319.39 / 371,649.73 | 109.95 / 621.56 |
| sync | on | 3,052.95 | 25.16 | 40,857.21 / 102,399.60 | 1,156.06 / 1,264.49 | 380,874.65 / 389,730.30 | 112.64 / 605.46 |
| async | off | 1,893.46 | 40.56 | 40,987.19 / 61,575.88 | 662.76 / 786.24 | 236,321.17 / 245,964.89 | 305.12 / 627.49 |
| async | on | 1,862.50 | 41.23 | 41,278.66 / 61,895.45 | 652.00 / 767.62 | 231,852.28 / 242,195.02 | 294.56 / 571.11 |

In async mode, batching raised throughput by 1.66%, reduced TPOT P50/P99 by
1.62%/2.37%, E2EL P50/P99 by 1.89%/1.53%, and NIC P50/P99 by 3.46%/8.99%.
TTFT P50/P99 rose by 0.71%/0.52%. In sync mode, throughput fell by 4.31%.

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

This is an independent profiling run, not the clean end-to-end run above. Its
Random and Phased async batching-on cases were captured end to end with
NVTX-only Nsight Systems on all four ranks. Search the reports for
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
The ratio uses the paired off/on durations from this same profiling run, rather
than mixing them with the clean end-to-end results above.

![Scheduler placement and measured cost/saved ratio](https://raw.githubusercontent.com/DOCCA0/vllm/refs/heads/ilp/benchmarks/eplb_rebalance/results/serving_nixl_20260910_cycle_precompute/scheduler_profile.png)

[Raw JSON, NIC samples, four-rank traces, summaries, and complete commands](https://github.com/DOCCA0/vllm/tree/ilp/benchmarks/eplb_rebalance/results/serving_nixl_20260910_cycle_precompute).
Open `random_async_on/trace/rank0.nsys-rep` or
`phased_async_on/trace/rank0.nsys-rep` with Nsight Systems 2024.5.1 or newer.
