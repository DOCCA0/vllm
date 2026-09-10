# Cycle-wide EPLB benchmark and profiling commands

All cases used four Ubuntu 22.04 nodes with one RTX 6000 each, TP/EP 4, NIXL
1.3.2 over 10 GbE without RDMA, prefix caching enabled, and commit
`bfd9399157`. A fresh server and an eight-request warm-up were used for every
case.

## Server

Repeat with both boolean values for `USE_ASYNC` and `USE_BATCHING`.

```bash
export PATH=$PWD/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=eno2 GLOO_SOCKET_IFNAME=eno2
export UCX_TLS=all UCX_NET_DEVICES=eno2 UCX_RCACHE_MAX_UNRELEASED=1024
export VLLM_EPLB_LOG_MIGRATION_STATS=1

MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
USE_ASYNC=true
USE_BATCHING=true
EPLB_CONFIG="{\"window_size\":50,\"step_interval\":100,\"num_redundant_experts\":16,\"log_balancedness\":false,\"use_async\":$USE_ASYNC,\"communicator\":\"nixl\",\"enable_migration_batching\":$USE_BATCHING}"

vllm serve "$MODEL" --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.8 --max-model-len 4096 \
  --enable-prefix-caching --tensor-parallel-size 4 \
  --distributed-executor-backend ray --enable-expert-parallel \
  --enable-eplb --eplb-config "$EPLB_CONFIG" --enforce-eager
```

## Warm-up

```bash
vllm bench serve --backend vllm --model "$MODEL" --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 100 --random-output-len 300 \
  --num-prompts 8 --max-concurrency 8 --seed 123 --temperature 0 \
  --ignore-eos --disable-tqdm
```

## Random workload

```bash
vllm bench serve --backend vllm --model "$MODEL" --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 100 --random-output-len 300 \
  --num-prompts 200 --max-concurrency 32 --seed 0 --temperature 0 \
  --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,90,95,99 \
  --ignore-eos --disable-tqdm --save-result \
  --result-dir "$RESULTS" --result-filename bench.json
```

## Ordered phased-English workload

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

NIC data is the head node's combined RX and TX byte counters sampled once per
second from `/sys/class/net/eno2/statistics/` during each formal benchmark.

## Complete async serving traces

The Random and phased async batching-on cases ran Ray inside an NVTX-only
Nsight Systems session on every node. For rank 0:

```bash
nsys start --session-new=cycle_random_0 --sample=none --cpuctxsw=none \
  -o "$RESULTS/random_async_on/trace/rank0"

nsys launch --session=cycle_random_0 --trace=nvtx --wait=all \
  ray start --head --node-ip-address=10.31.0.89 --port=6379 \
  --num-gpus=1 --disable-usage-stats
```

Workers used the same commands with sessions `cycle_random_1` through
`cycle_random_3`, their own output paths, and:

```bash
ray start --address=10.31.0.89:6379 --node-ip-address="$NODE_IP" \
  --num-gpus=1 --disable-usage-stats
```

The phased capture used session names `cycle_phased_0` through
`cycle_phased_3`. The server and benchmark commands were exactly those above;
`vllm bench serve --profile` was not used because the explicit NVTX ranges were
captured throughout the complete serving process.

Open a `rank*.nsys-rep` file with Nsight Systems 2024.5.1 or newer and search
for `eplb: schedule migration batches`. One range is one call that builds all
48 layer schedules for an async EPLB cycle. Formal benchmark calls were
selected using the recorded `formal_start_epoch_ns` and `formal_end_epoch_ns`
boundaries plus `TARGET_INFO_SESSION_START_TIME.utcEpochNs` in the exported
Nsight SQLite database.

## Per-layer versus cycle-wide profile

The direct short serving comparison used the same server, warm-up, eight-request
Random command, and NVTX setup. The control at commit `4f63218050` called the
scheduler once per layer. Commit `bfd9399157` called it once for all 48 layers.
The table reports the control's measured mean call time multiplied by 48 versus
the measured mean duration of one 48-layer call; no transfer behavior changed.
