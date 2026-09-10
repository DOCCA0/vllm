# Four-node NIXL EPLB migration batching

This benchmark uses Ubuntu 22.04.5, four Quadro RTX 6000 24 GB GPUs (one per
node), Ray 2.56.1, NIXL 1.3.2, and 10 GbE without RDMA. The tested source is
commit `83f052324b` on `ilp`, based on upstream `76ff0cdff2`. Prefix caching is
enabled in every run. The full benchmark tables below predate cycle-wide
precomputation. New serving profiles use candidate `bfd9399157`; their short
8-request results are reported separately.

The full new server, warmup, and profile commands are in
[COMMANDS.md](results/serving_trace_20260910/COMMANDS.md).

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

All requests succeeded. Raw JSON and logs are under
`results/serving_nixl_20260905/`.

### Random workload

| Mode | Batching | Duration (s) | Output (tok/s) | TTFT P50/P99 (ms) | TPOT P50/P99 (ms) | E2EL P50/P99 (ms) | NIC P50/P99 (MB/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sync | off | 2,331.63 | 25.73 | 29,369.51 / 48,692.09 | 1,047.12 / 1,146.84 | 343,648.96 / 365,870.13 | 107.03 / 643.49 |
| sync | on | 2,471.87 | 24.27 | 33,832.50 / 52,842.07 | 1,096.07 / 1,232.19 | 364,251.82 / 388,641.14 | 107.65 / 611.51 |
| async | off | 1,481.68 | 40.49 | 28,827.71 / 44,898.47 | 637.16 / 746.74 | 226,380.20 / 234,041.94 | 323.58 / 614.35 |
| async | on | 1,441.03 | 41.64 | 28,032.03 / 43,914.54 | 623.17 / 717.52 | 218,808.74 / 231,834.29 | 307.36 / 558.16 |

Batching raised throughput by 2.82% and reduced TTFT P50/P99 by 2.76%/2.19%,
TPOT P50/P99 by 2.20%/3.91%, E2EL P50/P99 by 3.34%/0.94%, and NIC P50/P99
by 5.01%/9.15% in async mode. In sync mode, throughput fell by 5.67% because
the conflict-free batches execute serially while inference is paused.

### Phased-English workload

| Mode | Batching | Duration (s) | Output (tok/s) | TTFT P50/P99 (ms) | TPOT P50/P99 (ms) | E2EL P50/P99 (ms) | NIC P50/P99 (MB/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sync | off | 2,907.92 | 26.41 | 40,482.84 / 96,627.16 | 1,096.95 / 1,205.97 | 362,827.06 / 371,150.66 | 107.86 / 655.86 |
| sync | on | 3,029.99 | 25.35 | 41,136.22 / 102,878.97 | 1,138.29 / 1,249.66 | 378,383.85 / 382,688.81 | 113.11 / 610.50 |
| async | off | 1,879.26 | 40.87 | 41,123.20 / 59,359.63 | 667.41 / 776.53 | 233,580.27 / 242,402.25 | 305.56 / 633.24 |
| async | on | 1,861.21 | 41.26 | 40,568.09 / 61,455.58 | 659.14 / 771.25 | 231,309.48 / 239,984.56 | 298.98 / 574.56 |

Batching raised throughput by 0.97% and reduced TTFT P50 by 1.35%, TPOT
P50/P99 by 1.24%/0.68%, E2EL P50/P99 by 0.97%/1.00%, and NIC P50/P99 by
2.15%/9.27% in async mode. TTFT P99 increased by 3.53%. In sync mode,
throughput fell by 4.03%.

Async EPLB is generally the relevant mode for production serving: migration
runs in the background while inference continues, so reducing NIC contention
reduces interference. Sync EPLB pauses inference, and splitting migration into
serial batches extends that pause. Sync users can retain the unbatched path with
`enable_migration_batching=false`.

The phased workload is closer to a changing production request mix. Random
token IDs change expert load less meaningfully, so they exercise migration
batching less consistently even when the number of transfers is similar.

## Scheduler profiling

The isolated PyTorch profile replays six real four-rank placements with 12--120
expert migrations per call. Each point has 50 warmups and 500 measured calls.
It measures the hot CPU algorithm and its scaling with migration count; it is
not an estimate of elapsed scheduler latency during serving.

```bash
.venv/bin/python benchmarks/eplb_rebalance/profile_scheduler_trace.py \
  benchmarks/eplb_rebalance/results/scheduler_profile_20260905/placements \
  --warmup 50 --repeats 500 \
  --trace scheduler.pt.trace.json --summary summary.csv
```

| Expert migrations/call | Flow scheduling P50 (ms/call) |
| ---: | ---: |
| 12 | 0.047 |
| 24 | 0.058 |
| 48 | 0.077 |
| 72 | 0.095 |
| 96 | 0.112 |
| 120 | 0.133 |

Async scheduling is captured as `eplb: schedule migration batches`. Each range
now covers **all 48 layers in one EPLB cycle**, including schedule construction.
The trace includes serving NVTX events such as `execute_context_*`; it does not
include CUDA kernel timing. Profiling builds add an NVTX range alongside the
production `record_function` marker.

The table and plot show Bulk run 1, which completed 8/8 requests and profiler
shutdown. All scheduler ranges in the capture (including warmup) are counted:
each rank had two 48-layer calls. Bulk run 2 repeats the same implementation and
parameters; its raw files remain available for reproducibility.

| Rank | Total / calls | Mean ms/built layer |
| --- | ---: | ---: |
| 0 | 19.936 ms / 2 | 0.208 |
| 1 | 13.920 ms / 2 | 0.145 |
| 2 | 14.549 ms / 2 | 0.152 |
| 3 | 14.037 ms / 2 | 0.146 |

Amortized mean = sum of full scheduling range durations / (number of cycle calls
× 48). This is not a measured single-layer P50 or P99; two calls per rank are
insufficient for useful tail estimates. Full calls average 6.96–9.97 ms.
Precomputation may build schedules for layers not consumed before a cycle stops.
The entire computation is charged here, rather than only the lookup during transfer.

Compared with the separate per-layer-scheduling control, normalized mean cost
fell from 0.603 to 0.163 ms per built layer in the displayed run. This is
preliminary evidence from a short profiling comparison, not a guarantee
for other workloads. Concentrating CPU work may also change inference interference.

![Scheduling cost per built layer and full cycle](https://raw.githubusercontent.com/DOCCA0/vllm/refs/heads/ilp/benchmarks/eplb_rebalance/results/serving_trace_20260910/bulk_scheduler.png)

Short profiled serving runs (not substitutes for the full benchmark above):

| Run | Completed | Duration s | Output tok/s | TTFT P50/P99 ms | TPOT P50/P99 ms | E2EL P50/P99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Per-layer control | 8 | 121.40 | 19.77 | 17,072.31 / 17,074.23 | 348.90 / 387.54 | 121,393.62 / 121,395.09 |
| Bulk run 1 | 8 | 113.91 | 21.07 | 17,451.02 / 17,453.58 | 322.59 / 362.21 | 113,905.64 / 113,908.33 |

TTFT increased in this short comparison; reduced scheduling cost does not establish
that every serving metric improves.

[Unmodified JSON, four-rank reports, and complete commands](https://github.com/DOCCA0/vllm/tree/ilp/benchmarks/eplb_rebalance/results/serving_trace_20260910).
Open `bulk_precompute_retry/rank0.nsys-rep` in Nsight Systems 2024.5.1 or newer.

### Historical cost/saved comparison

For the earlier per-layer implementation only, the serving P50 of 0.572041
ms/call gave these estimates against the earlier full async benchmark:

| Workload | Layer calls/rank | Estimated elapsed/rank ms | Serving time saved ms | Estimated cost/saved |
| --- | ---: | ---: | ---: | ---: |
| Random | 630 | 360.39 | 40,650.40 | 0.89% |
| Phased English | 767 | 438.76 | 18,045.29 | 2.43% |

Estimated elapsed = layer calls × 0.572041 ms; cost/saved = estimated elapsed /
serving time saved × 100%. Ranks run concurrently and are not summed.
These are cross-run estimates, not measured overhead fractions.
A cost/saved ratio for cycle-wide precomputation is **not yet measured**:
the short profile compares two batching-on schedulers, not batching off/on.
The new amortized time must not be multiplied into the old table and presented
as a newly measured benefit.
