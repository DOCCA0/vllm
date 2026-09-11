# Full async serving profile

This repeats the clean batching-on E2E case at commit `eabb08764`. Model,
server, warm-up, and formal workload parameters are identical. Only NVTX-only
Nsight capture around the Ray processes is added.

Head node:

```bash
nsys start --session-new=eplb_final_0 --sample=none --cpuctxsw=none \
  -o "$RESULTS/trace/rank0"

nsys launch --session=eplb_final_0 --trace=nvtx --wait=all \
  ray start --head --node-ip-address=10.31.0.89 --port=6379 \
  --num-gpus=1 --disable-usage-stats
```

Each worker used its own session name, trace filename, and `NODE_IP`:

```bash
nsys start --session-new="$SESSION" --sample=none --cpuctxsw=none \
  -o "$RESULTS/trace/rank${RANK}"

nsys launch --session="$SESSION" --trace=nvtx --wait=all \
  ray start --address=10.31.0.89:6379 --node-ip-address="$NODE_IP" \
  --num-gpus=1 --disable-usage-stats
```

Server:

```bash
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=eno2 GLOO_SOCKET_IFNAME=eno2
export UCX_TLS=all UCX_NET_DEVICES=eno2 UCX_RCACHE_MAX_UNRELEASED=1024
export VLLM_EPLB_LOG_MIGRATION_STATS=0

MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
EPLB_CONFIG='{"window_size":50,"step_interval":50,"num_redundant_experts":32,"log_balancedness":false,"use_async":true,"communicator":"nixl","enable_migration_batching":true}'

vllm serve "$MODEL" --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.9 --max-model-len 4096 \
  --enable-prefix-caching --tensor-parallel-size 4 \
  --distributed-executor-backend ray --enable-expert-parallel \
  --enable-eplb --eplb-config "$EPLB_CONFIG" --enforce-eager
```

Warm-up:

```bash
vllm bench serve --backend vllm --model "$MODEL" --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 200 --random-output-len 300 \
  --num-prompts 8 --max-concurrency 8 --seed 123 --temperature 0 \
  --ignore-eos --disable-tqdm
```

Formal workload:

```bash
vllm bench serve --backend vllm --model "$MODEL" --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 200 --random-output-len 300 \
  --num-prompts 500 --max-concurrency 32 --seed 0 --temperature 0 \
  --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
  --ignore-eos --disable-tqdm --save-result \
  --result-dir "$RESULTS/random_async_on" --result-filename bench.json
```

Open `trace/rank*.nsys-rep` with Nsight Systems 2024.5.1 or newer and search
for `eplb: schedule migration batches`. The timestamps in
`random_async_on/formal_start_epoch_ns` and `formal_end_epoch_ns` delimit the
formal benchmark and exclude model startup and warm-up.
