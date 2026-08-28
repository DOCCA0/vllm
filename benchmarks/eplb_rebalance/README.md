# Four-Node EPLB Migration Batching Experiment

Both workloads below keep prefix caching enabled. Run four configurations for
each workload: `sync/off`, `sync/on`, `async/off`, and `async/on`.
Migration batching is enabled by default; set it to `false` for rollback.

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

The benchmark pins Gloo for async runs instead of using communicator
auto-selection, so batching is the only variable between each off/on pair.

```bash
# Synchronous, batching disabled
TAG=01_sync_off
USE_ASYNC=false
USE_BATCHING=false
COMMUNICATOR=torch_nccl

# Synchronous, batching enabled
TAG=02_sync_batching_on
USE_ASYNC=false
USE_BATCHING=true
COMMUNICATOR=torch_nccl

# Asynchronous, batching disabled
TAG=03_async_off
USE_ASYNC=true
USE_BATCHING=false
COMMUNICATOR=torch_gloo

# Asynchronous, batching enabled
TAG=04_async_batching_on
USE_ASYNC=true
USE_BATCHING=true
COMMUNICATOR=torch_gloo
```

Select the result root for the workload being tested:

```bash
# Random workload
RESULTS_ROOT=$HOME/benchmarks/eplb_upstream_random_cache_on_20260817_r1

# Phased English workload
RESULTS_ROOT=$HOME/benchmarks/eplb_upstream_phased_cache_on_20260818_r1
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

EPLB_CONFIG="{\"window_size\":50,\"step_interval\":100,\"num_redundant_experts\":16,\"log_balancedness\":false,\"use_async\":$USE_ASYNC,\"communicator\":\"$COMMUNICATOR\",\"enable_migration_batching\":$USE_BATCHING}"

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
/home/cc/benchmarks/eplb_upstream_random_cache_on_20260817_r1/
```

### B. Phased English workload

The JSONL contains 256 English prompts in eight ordered phases of 32 requests:
CUDA, literature, mathematics, and biomedical prompts, repeated twice. The
ordered domain changes make the expert distribution move repeatedly. Each
prompt retains a common 223-token prefix, so prefix caching is still exercised.

The exact JSONL used for the experiment is tracked in the repository at:

```text
benchmarks/eplb_rebalance/bench_dataset/eplb_phased_english_256.jsonl
```

It was generated deterministically by
`bench_dataset/generate_phased_english.py`. Its SHA-256 is
`15f80f77caf13b44c1ee715a2db0663daf6a5f884fcbb5769998b46b86482dda`.
The experiment directory also retains an identical archived copy under
`workload/`.

If the custom loader reports that bench support is missing, install its required
dependency on the head node:

```bash
uv pip install --python $HOME/.venv/bin/python pandas
```

Run it without shuffling so the phases remain ordered:

```bash
DATASET=$HOME/vllm/benchmarks/eplb_rebalance/bench_dataset/eplb_phased_english_256.jsonl

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
/home/cc/benchmarks/eplb_upstream_phased_cache_on_20260818_r1/
```

Stop the sampler and service after each run:

```bash
kill "$NIC_PID" "$SERVER_PID"
```

## 5. Results

### Random workload, prefix cache enabled

| Mode | batching | Duration (s) | Output (tok/s) | E2EL P99 (ms) | NIC P99 (MB/s) | NIC Peak (MB/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sync | false | 988.94 | 20.22 | 186,001.25 | 1852.22 | 1930.67 |
| sync | true | 1067.07 | 18.74 | 200,128.55 | 1153.45 | 1210.99 |
| async | false | 889.51 | 22.48 | 162,839.73 | 707.21 | 802.06 |
| async | true | 875.30 | 22.85 | 161,767.77 | 649.08 | 701.98 |

### Phased English workload, prefix cache enabled

| Mode | batching | Duration (s) | Output (tok/s) | E2EL P99 (ms) | NIC P99 (MB/s) | NIC Peak (MB/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sync | false | 952.94 | 26.86 | 134,590.88 | 1829.08 | 1982.54 |
| sync | true | 1052.99 | 24.31 | 156,878.38 | 1161.40 | 1208.52 |
| async | false | 845.10 | 30.29 | 108,864.96 | 799.65 | 914.92 |
| async | true | 815.56 | 31.39 | 104,369.71 | 705.32 | 770.31 |

The main comparison for the proposed optimization is `03_async_off` versus
`04_async_batching_on`. `batches` is calculated from each layer's migration
graph; it is not configured as the constant value six.
