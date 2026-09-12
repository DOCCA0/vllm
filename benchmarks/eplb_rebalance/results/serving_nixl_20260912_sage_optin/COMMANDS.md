# Async random A/B commands

Commit: `12dac29d594a182941473ea8b11cf0b9949313c0`. Prefix caching and NIXL were
enabled. Both cases used a fresh server after an eight-request warm-up. The
only A/B difference was `enable_migration_batching`.

```bash
EPLB_CONFIG='{"window_size":50,"step_interval":50,"num_redundant_experts":32,"log_balancedness":false,"use_async":true,"communicator":"nixl","enable_migration_batching":false}'

vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.9 --max-model-len 4096 \
  --enable-prefix-caching --tensor-parallel-size 4 \
  --distributed-executor-backend ray --enable-expert-parallel \
  --enable-eplb --eplb-config "$EPLB_CONFIG" --enforce-eager
```

Repeat with `enable_migration_batching` set to `true`.

```bash
vllm bench serve --backend vllm \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 200 --random-output-len 300 \
  --num-prompts 500 --max-concurrency 32 --seed 0 --temperature 0 \
  --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
  --ignore-eos --disable-tqdm --save-result \
  --result-dir "$RESULTS" --result-filename bench.json
```

The head node's combined `eno2` RX+TX counters were sampled once per second.
No profiler or diagnostic migration logging was enabled.
