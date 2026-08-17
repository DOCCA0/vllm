# Four-Node EPLB Migration Batching Experiment

Both workloads below keep prefix caching enabled. Run four configurations for
each workload: `sync/off`, `sync/on`, `async/off`, and `async/on`.

## 1. Pull the Code

Run in a Windows terminal:

```bash
ssh -J cc@192.5.86.236 cc@10.140.83.156 "cd ~/vllm && git switch ilp && git pull --ff-only origin ilp"
ssh -J cc@192.5.86.236 cc@10.140.83.96  "cd ~/vllm && git switch ilp && git pull --ff-only origin ilp"
ssh -J cc@192.5.86.236 cc@10.140.83.28  "cd ~/vllm && git switch ilp && git pull --ff-only origin ilp"
ssh -J cc@192.5.86.236 cc@10.140.83.94  "cd ~/vllm && git switch ilp && git pull --ff-only origin ilp"
```

## 2. Start Ray

Run on all four nodes:

```bash
cd ~/vllm
export PATH=$HOME/.venv/bin:$HOME/.local/bin:$PATH
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
ray stop --force
```

Run on `10.140.83.156`:

```bash
ray start --head --port=6379 --disable-usage-stats
```

Run on `10.140.83.96`, `10.140.83.28`, and `10.140.83.94`:

```bash
ray start --address=10.140.83.156:6379 --disable-usage-stats
```

`ray status` on the head node should report four active nodes and four GPUs.

## 3. Start vLLM

Choose one configuration before each run:

```bash
# Synchronous, batching disabled
TAG=01_sync_off
USE_ASYNC=false
USE_BATCHING=false

# Synchronous, batching enabled
TAG=02_sync_batching_on
USE_ASYNC=false
USE_BATCHING=true

# Asynchronous, batching disabled
TAG=03_async_off
USE_ASYNC=true
USE_BATCHING=false

# Asynchronous, batching enabled
TAG=04_async_batching_on
USE_ASYNC=true
USE_BATCHING=true
```

Select the result root for the workload being tested:

```bash
# Random workload
RESULTS_ROOT=$HOME/benchmarks/eplb_200_prefix_cache_on_20260817

# Phased English workload
RESULTS_ROOT=$HOME/benchmarks/eplb_phased_english_cache_on_20260817
```

Start a fresh service for every configuration:

```bash
cd ~/vllm
export PATH=$HOME/.venv/bin:$HOME/.local/bin:$PATH
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
export VLLM_EPLB_LOG_MIGRATION_STATS=1

MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
RESULTS=$RESULTS_ROOT/$TAG
mkdir -p "$RESULTS"

EPLB_CONFIG="{\"window_size\":50,\"step_interval\":100,\"num_redundant_experts\":16,\"log_balancedness\":false,\"use_async\":$USE_ASYNC,\"enable_migration_batching\":$USE_BATCHING}"

vllm serve "$MODEL" --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.8 --max-model-len 4096 \
  --enable-prefix-caching --tensor-parallel-size 4 \
  --distributed-executor-backend ray --enable-expert-parallel \
  --enable-eplb --eplb-config "$EPLB_CONFIG" --enforce-eager \
  > "$RESULTS/server.log" 2>&1 &
SERVER_PID=$!

until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 5; done

curl -sf http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"warm up\",\"max_tokens\":10}" \
  > "$RESULTS/warmup.json"

while kill -0 "$SERVER_PID"; do
  echo "$(date +%s) $(cat /sys/class/net/eno1np0/statistics/{rx_bytes,tx_bytes} | xargs)"
  sleep 1
done > "$RESULTS/nic.tsv" &
NIC_PID=$!
```

## 4. Choose One Benchmark Workload

### A. Random workload

This is the ordinary synthetic workload. It has a 300-token shared prefix and
200 random input tokens. Prefix caching remains enabled by the server.

```bash
vllm bench serve --backend vllm --model "$MODEL" --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 200 --random-output-len 100 \
  --num-prompts 200 --max-concurrency 32 --seed 0 --temperature 0 \
  --percentile-metrics ttft,tpot,e2el \
  --metric-percentiles 50,90,95,99 --ignore-eos --disable-tqdm \
  --save-result --result-dir "$RESULTS" --result-filename bench.json \
  > "$RESULTS/bench.log" 2>&1
```

Results are stored under:

```text
/home/cc/benchmarks/eplb_200_prefix_cache_on_20260817/
```

### B. Phased English workload

The JSONL contains 256 English prompts in eight ordered phases of 32 requests:
CUDA, literature, mathematics, and biomedical prompts, repeated twice. The
ordered domain changes make the expert distribution move repeatedly. Each
prompt retains a common 223-token prefix, so prefix caching is still exercised.

The JSONL is stored on the head node at:

```text
/home/cc/benchmarks/eplb_phased_english_cache_on_20260817/workload/eplb_phased_english_256.jsonl
```

If the custom loader reports that bench support is missing, install its required
dependency on the head node:

```bash
uv pip install --python $HOME/.venv/bin/python pandas
```

Run it without shuffling so the phases remain ordered:

```bash
DATASET=$HOME/benchmarks/eplb_phased_english_cache_on_20260817/workload/eplb_phased_english_256.jsonl

vllm bench serve --backend vllm --model "$MODEL" --port 8000 \
  --dataset-name custom --dataset-path "$DATASET" --disable-shuffle \
  --custom-output-len 100 --num-prompts 256 --max-concurrency 32 \
  --seed 0 --temperature 0 --percentile-metrics ttft,tpot,e2el \
  --metric-percentiles 50,90,95,99 --ignore-eos --disable-tqdm \
  --save-result --result-dir "$RESULTS" --result-filename bench.json \
  > "$RESULTS/bench.log" 2>&1
```

Results are stored under:

```text
/home/cc/benchmarks/eplb_phased_english_cache_on_20260817/
```

Stop the sampler and service after each run:

```bash
kill "$NIC_PID" "$SERVER_PID"
```

## 5. Results

### Random workload, prefix cache enabled

| Mode | batching | Duration (s) | Output (tok/s) | E2EL P99 (ms) | NIC P99 (MB/s) | NIC Peak (MB/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sync | false | 979.74 | 20.41 | 211,622.31 | 1863.81 | 2066.54 |
| sync | true | 1028.97 | 19.44 | 211,641.61 | 1188.89 | 1266.94 |
| async | false | 895.46 | 22.33 | 188,001.06 | 1067.90 | 1170.88 |
| async | true | 872.12 | 22.93 | 182,382.76 | 904.43 | 949.66 |

### Phased English workload, prefix cache enabled

| Mode | batching | Duration (s) | Output (tok/s) | E2EL P99 (ms) | NIC P99 (MB/s) | NIC Peak (MB/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sync | false | 926.62 | 27.63 | 144,514.36 | 1917.30 | 2027.71 |
| sync | true | 1009.09 | 25.37 | 154,407.98 | 1177.96 | 1257.16 |
| async | false | 855.80 | 29.91 | 137,387.57 | 1114.55 | 1217.50 |
| async | true | 821.08 | 31.18 | 126,408.37 | 853.07 | 1008.49 |

The main comparison for the proposed optimization is `03_async_off` versus
`04_async_batching_on`. `batches` is calculated from each layer's migration
graph; it is not configured as the constant value six.
