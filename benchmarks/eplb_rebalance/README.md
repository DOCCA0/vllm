# 四节点 EPLB 迁移分批实验

## 1. 拉取代码

在 Windows 终端依次执行：

```bash
ssh -J cc@192.5.86.236 cc@10.140.83.156 "cd ~/vllm && git switch ilp && git pull --ff-only origin ilp"
ssh -J cc@192.5.86.236 cc@10.140.83.96  "cd ~/vllm && git switch ilp && git pull --ff-only origin ilp"
ssh -J cc@192.5.86.236 cc@10.140.83.28  "cd ~/vllm && git switch ilp && git pull --ff-only origin ilp"
ssh -J cc@192.5.86.236 cc@10.140.83.94  "cd ~/vllm && git switch ilp && git pull --ff-only origin ilp"
```

## 2. 启动 Ray

四台机器都先执行：

```bash
cd ~/vllm
export PATH=$HOME/.venv/bin:$HOME/.local/bin:$PATH
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
ray stop --force
```

head `10.140.83.156` 执行：

```bash
ray start --head --port=6379 --disable-usage-stats
```

其余三台执行：

```bash
ray start --address=10.140.83.156:6379 --disable-usage-stats
```

head 上运行 `ray status`，应看到 4 个节点和 4 张 GPU。

## 3. 运行一组实验

每组都重新启动服务。先选一组：

```bash
# 异步、不分批（对照组）
TAG=async_off
USE_BATCHING=false

# 异步、degree_desc（实验组）
# TAG=async_degree_desc
# USE_BATCHING=true
```

启动服务：

```bash
cd ~/vllm
export PATH=$HOME/.venv/bin:$HOME/.local/bin:$PATH
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
export VLLM_EPLB_LOG_MIGRATION_STATS=1

MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
RESULTS=$HOME/benchmarks/eplb_rebalance/$TAG
mkdir -p "$RESULTS"

EPLB_CONFIG="{\"window_size\":50,\"step_interval\":100,\"num_redundant_experts\":16,\"log_balancedness\":false,\"use_async\":true,\"use_migration_batching\":$USE_BATCHING,\"migration_batching_policy\":\"degree_desc\"}"

vllm serve "$MODEL" --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.8 --max-model-len 4096 \
  --no-enable-prefix-caching --tensor-parallel-size 4 \
  --distributed-executor-backend ray --enable-expert-parallel \
  --enable-eplb --eplb-config "$EPLB_CONFIG" --enforce-eager \
  > "$RESULTS/server.log" 2>&1 &
SERVER_PID=$!

until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 5; done
```

预热并压测：

```bash
curl -sf http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"warm up\",\"max_tokens\":10}" \
  > "$RESULTS/warmup.json"

vllm bench serve --backend vllm --model "$MODEL" --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 200 --random-output-len 100 \
  --num-prompts 100 --max-concurrency 32 --seed 0 \
  --percentile-metrics ttft,tpot,e2el \
  --metric-percentiles 50,90,95,99 --ignore-eos --disable-tqdm \
  --save-result --result-dir "$RESULTS" --result-filename bench.json \
  > "$RESULTS/bench.log" 2>&1

kill "$SERVER_PID"
```

先后运行两组，并交换顺序再重复一次。不要使用同步 EPLB 或单独的 `main`
对照组。

## 4. 已验证结果

两次运行的均值：

| 策略 | 耗时 (s) | 输出 (tok/s) | TPOT P50 (ms) | E2EL P99 (ms) | NIC P99 (MB/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| async off | 809.69 | 12.35 | 2,027.75 | 395,312.18 | 925.13 |
| async degree_desc | 797.68 | 12.54 | 2,006.95 | 391,787.35 | 820.92 |

`degree_desc` 平均快 1.48%，head NIC P99 降 11.3%。每层日志应显示约
`12 flows, 6 batches`。

```bash
cat "$RESULTS/bench.log"
grep -E "EPLB migration stats|Rearranged experts" "$RESULTS/server.log"
```
