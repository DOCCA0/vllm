# Commands

Environment: Ubuntu 22.04.5, 4 × Quadro RTX 6000 24 GB, TP=EP=4, Ray
2.56.1, NIXL 1.3.2, and 10 GbE without RDMA. Commit: `eabb08764`.

```bash
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=eno2 GLOO_SOCKET_IFNAME=eno2
export UCX_TLS=all UCX_NET_DEVICES=eno2 UCX_RCACHE_MAX_UNRELEASED=1024
export VLLM_EPLB_LOG_MIGRATION_STATS=0

MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
USE_BATCHING=true  # repeat with false
EPLB_CONFIG="{\"window_size\":50,\"step_interval\":50,\"num_redundant_experts\":32,\"log_balancedness\":false,\"use_async\":true,\"communicator\":\"nixl\",\"enable_migration_batching\":$USE_BATCHING}"

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
  --result-dir "$RESULTS" --result-filename bench.json
```

The head node's combined RX+TX byte counters were sampled once per second from
`/sys/class/net/eno2/statistics/`. The profiler repeated the batching-on case
unchanged, except that Ray was launched inside an NVTX-only Nsight session:

```bash
nsys start --session-new=eplb_final_0 --sample=none --cpuctxsw=none \
  -o "$PROFILE_RESULTS/trace/rank0"

nsys launch --session=eplb_final_0 --trace=nvtx --wait=all \
  ray start --head --node-ip-address=10.31.0.89 --port=6379 \
  --num-gpus=1 --disable-usage-stats
```

Workers used their own session/output names and the normal worker command.
