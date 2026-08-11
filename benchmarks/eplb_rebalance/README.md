# Four-Node EPLB Migration Batching Experiment

## 1. Pull the Code

Run the following commands sequentially in a Windows terminal:

```bash
ssh -J cc@192.5.86.236 cc@10.140.83.156 "cd ~/vllm && git switch ilp && git pull --ff-only origin ilp"
ssh -J cc@192.5.86.236 cc@10.140.83.96  "cd ~/vllm && git switch ilp && git pull --ff-only origin ilp"
ssh -J cc@192.5.86.236 cc@10.140.83.28  "cd ~/vllm && git switch ilp && git pull --ff-only origin ilp"
ssh -J cc@192.5.86.236 cc@10.140.83.94  "cd ~/vllm && git switch ilp && git pull --ff-only origin ilp"
```

## 2. Start Ray

Run the following commands on all four machines first:

```bash
cd ~/vllm
export PATH=$HOME/.venv/bin:$HOME/.local/bin:$PATH
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
ray stop --force
```

Run the following command on the head node, `10.140.83.156`:

```bash
ray start --head --port=6379 --disable-usage-stats
```

Run the following command on the other three machines:

```bash
ray start --address=10.140.83.156:6379 --disable-usage-stats
```

Run `ray status` on the head node. You should see four nodes and four GPUs.

## 3. Run an Experiment

Restart the service for each experiment. Choose one of the following four configurations:

```bash
# Synchronous, without batching
TAG=01_sync_off
USE_ASYNC=false
USE_BATCHING=false

# Synchronous, with migration batching enabled
TAG=02_sync_batching_on
USE_ASYNC=false
USE_BATCHING=true

# Asynchronous, without batching
TAG=03_async_off
USE_ASYNC=true
USE_BATCHING=false

# Asynchronous, with migration batching enabled
TAG=04_async_batching_on
USE_ASYNC=true
USE_BATCHING=true
```

Start the service:

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

Warm up the service and run the benchmark:

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

Run all four configurations sequentially. No separate `main` baseline is needed.

## 4. Results for 200 Prompts

All four configurations completed 200 requests successfully with zero failures. NIC traffic is the per-second RX+TX traffic on the head node's `eno1np0` interface.

| Mode | batching | batches | Duration (s) | Output (tok/s) | E2EL P99 (ms) | NIC P99 (MB/s) | NIC Peak (MB/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sync | false | 1 | 1576.63 | 12.69 | 387,329.45 | 1788.94 | 2026.04 |
| sync | true | 6 | 1638.30 | 12.21 | 408,886.74 | 1165.15 | 1258.95 |
| async | false | 1 | 1504.01 | 13.30 | 382,019.25 | 1030.35 | 1113.71 |
| async | true | 6 | 1480.46 | 13.51 | 378,654.89 | 855.10 | 940.19 |

Enabling batching for synchronous migration increases the serialized migration time, but reduces NIC P99 by 34.9%. With asynchronous migration, enabling batching reduces NIC P99 by 17.0%, increases output throughput by 1.6%, and reduces E2EL P99 by 0.9%.

`batches` is not a configuration value. The scheduler calculates it dynamically from the migration graph for each layer. In this experiment, 12 flows resulted in exactly 6 batches. Configurations without batching are recorded as having 1 batch.

```bash
ls -1 "$HOME/benchmarks/eplb_200"/*/bench.log
cat "$RESULTS/bench.log"
grep -E "EPLB migration stats|Rearranged experts" "$RESULTS/server.log"
```
