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

每组都重新启动服务。四组选一组：

```bash
# 同步、不分批
TAG=01_sync_off
USE_ASYNC=false
USE_BATCHING=false

# 同步、开启迁移分批
# TAG=02_sync_batching_on
# USE_ASYNC=false
# USE_BATCHING=true

# 异步、不分批
# TAG=03_async_off
# USE_ASYNC=true
# USE_BATCHING=false

# 异步、开启迁移分批
# TAG=04_async_batching_on
# USE_ASYNC=true
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
RESULTS=$HOME/benchmarks/eplb_200/$TAG
mkdir -p "$RESULTS"

EPLB_CONFIG="{\"window_size\":50,\"step_interval\":100,\"num_redundant_experts\":16,\"log_balancedness\":false,\"use_async\":$USE_ASYNC,\"enable_migration_batching\":$USE_BATCHING}"

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

while kill -0 "$SERVER_PID"; do
  echo "$(date +%s) $(cat /sys/class/net/eno1np0/statistics/{rx_bytes,tx_bytes} | xargs)"
  sleep 1
done > "$RESULTS/nic.tsv" &
NIC_PID=$!

vllm bench serve --backend vllm --model "$MODEL" --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 200 --random-output-len 100 \
  --num-prompts 200 --max-concurrency 32 --seed 0 --temperature 0 \
  --percentile-metrics ttft,tpot,e2el \
  --metric-percentiles 50,90,95,99 --ignore-eos --disable-tqdm \
  --save-result --result-dir "$RESULTS" --result-filename bench.json \
  > "$RESULTS/bench.log" 2>&1

kill "$NIC_PID" "$SERVER_PID"
```

依次运行四组，不需要单独的 `main` 对照组。

## 4. 200 prompts 结果

四组均为 200 成功、0 失败。NIC 是 head `eno1np0` 每秒 RX+TX。

| 模式 | batching | batches | 耗时 (s) | 输出 (tok/s) | E2EL P99 (ms) | NIC P99 (MB/s) | NIC 峰值 (MB/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sync | false | 1 | 1576.63 | 12.69 | 387,329.45 | 1788.94 | 2026.04 |
| sync | true | 6 | 1638.30 | 12.21 | 408,886.74 | 1165.15 | 1258.95 |
| async | false | 1 | 1504.01 | 13.30 | 382,019.25 | 1030.35 | 1113.71 |
| async | true | 6 | 1480.46 | 13.51 | 378,654.89 | 855.10 | 940.19 |

同步开启分批会增加串行迁移耗时，但 NIC P99 降低 34.9%。异步开启分批时，
NIC P99 降低 17.0%，输出吞吐提高 1.6%，E2EL P99 降低 0.9%。

`batches` 不是配置值。调度器根据每层迁移图动态计算；这次 12 条 flows
恰好得到 6 batches。未分批组记为 1 batch。

```bash
ls -1 "$HOME/benchmarks/eplb_200"/*/bench.log
cat "$RESULTS/bench.log"
grep -E "EPLB migration stats|Rearranged experts" "$RESULTS/server.log"
```
