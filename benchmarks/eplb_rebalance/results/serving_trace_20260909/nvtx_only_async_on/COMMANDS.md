# NVTX-only async serving profile

The four Ray nodes were launched inside separate Nsight Systems sessions. The
head-node form was:

```bash
nsys start --session-new=nvtx0 --sample=none --cpuctxsw=none \
  -o /home/cc/vllm-profile-20260909/nvtx_only/rank0

nsys launch --session=nvtx0 --trace=nvtx --wait=all \
  ray start --head --node-ip-address=10.31.0.89 --port=6379 \
  --num-gpus=1 --disable-usage-stats
```

Workers used the same commands with sessions `nvtx1`--`nvtx3`, outputs
`rank1`--`rank3`, and:

```bash
ray start --address=10.31.0.89:6379 --node-ip-address="$NODE_IP" \
  --num-gpus=1 --disable-usage-stats
```

Before starting Ray and vLLM on every node:

```bash
cd ~/vllm-ilp
export PATH=$PWD/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=eno2 GLOO_SOCKET_IFNAME=eno2
export UCX_TLS=all UCX_NET_DEVICES=eno2 UCX_RCACHE_MAX_UNRELEASED=1024
```

Server:

```bash
export VLLM_EPLB_LOG_MIGRATION_STATS=1
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
EPLB_CONFIG='{"window_size":50,"step_interval":100,"num_redundant_experts":16,"log_balancedness":false,"use_async":true,"communicator":"nixl","enable_migration_batching":true}'

vllm serve "$MODEL" --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.8 --max-model-len 4096 \
  --enable-prefix-caching --tensor-parallel-size 4 \
  --distributed-executor-backend ray --enable-expert-parallel \
  --enable-eplb --eplb-config "$EPLB_CONFIG" --enforce-eager \
  --profiler-config '{"profiler":"cuda"}'
```

Warm-up:

```bash
vllm bench serve --backend vllm --model "$MODEL" --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 100 --random-output-len 300 \
  --num-prompts 8 --max-concurrency 8 --seed 123 --temperature 0 \
  --ignore-eos --disable-tqdm --save-result \
  --result-dir "$RESULTS" --result-filename warmup.json
```

Profiled serving run:

```bash
vllm bench serve --backend vllm --model "$MODEL" --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 100 --random-output-len 300 \
  --num-prompts 8 --max-concurrency 8 --seed 0 --temperature 0 \
  --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,90,95,99 \
  --ignore-eos --disable-tqdm --profile --save-result \
  --result-dir "$RESULTS" --result-filename bench.json
```

After the server stopped, each node finalized its trace with:

```bash
nsys stop --session="$SESSION"
```

The profile-window statistics in `scheduler_summary.csv` include scheduler
ranges between the earliest start and latest end of the captured
`execute_context_*` ranges. The `.nsys-rep` files open directly in Nsight
Systems.

The median of the four per-rank P50 values is 0.572 ms/call. This is elapsed
time in a real async serving thread and includes runtime scheduling effects
absent from a tight microbenchmark. The separate isolated scheduler profile
measures hot algorithm CPU time (0.047--0.133 ms/call for 12--120 migrations),
so it is used only to characterize algorithmic scaling, not to estimate
serving cost.
