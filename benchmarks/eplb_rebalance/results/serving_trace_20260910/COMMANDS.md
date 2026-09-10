# Scheduler serving profile commands

All runs used four RTX 6000 nodes, Ray TP/EP 4, NIXL, async EPLB, migration
batching enabled, prefix caching enabled, and the same model and request shape.

```bash
EPLB_CONFIG='{"window_size":50,"step_interval":100,"num_redundant_experts":16,"log_balancedness":false,"use_async":true,"communicator":"nixl","enable_migration_batching":true}'

VLLM_EPLB_LOG_MIGRATION_STATS=1 vllm serve \
  Qwen/Qwen3-30B-A3B-Instruct-2507 --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.8 --max-model-len 4096 \
  --enable-prefix-caching --tensor-parallel-size 4 \
  --distributed-executor-backend ray --enable-expert-parallel \
  --enable-eplb --eplb-config "$EPLB_CONFIG" --enforce-eager \
  --profiler-config '{"profiler":"cuda"}'
```

Warmup:

```bash
vllm bench serve --backend vllm \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 100 --random-output-len 300 \
  --num-prompts 8 --max-concurrency 8 --seed 123 --temperature 0 \
  --ignore-eos --disable-tqdm --save-result \
  --result-dir "$RESULTS" --result-filename warmup.json
```

Profiled serving window:

```bash
vllm bench serve --backend vllm \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 100 --random-output-len 300 \
  --num-prompts 8 --max-concurrency 8 --seed 0 --temperature 0 \
  --percentile-metrics ttft,tpot,e2el \
  --metric-percentiles 50,90,95,99 --ignore-eos --disable-tqdm \
  --profile --save-result \
  --result-dir "$RESULTS" --result-filename bench.json
```

Nsight Systems was started on every rank with NVTX-only collection:

```bash
nsys start --session-new="$SESSION" --sample=none --cpuctxsw=none \
  -o "$RESULTS/rank$RANK"
nsys launch --session="$SESSION" --trace=nvtx --wait=all ray start ...
```

`baseline_retry` uses one scheduler call per transferred layer.
`bulk_precompute_retry` and `bulk_precompute_repeat2` precompute all 48 layer
schedules once per async EPLB cycle while retaining the same scheduler.
