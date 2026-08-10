# 四节点 EPLB 迁移分批实验

## 1. 节点

```bash
ssh -J cc@192.5.86.236 cc@10.140.83.156  # head
ssh -J cc@192.5.86.236 cc@10.140.83.96
ssh -J cc@192.5.86.236 cc@10.140.83.28
ssh -J cc@192.5.86.236 cc@10.140.83.94
```

每台机器均使用 `~/vllm`、`ilp` 分支、`.venv` Python 3.12，并确认提交一致：

```bash
cd ~/vllm
git switch ilp
git pull
git rev-parse HEAD

uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt
.venv/bin/pre-commit install
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install "ray[default]" "huggingface_hub[cli]"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-30B-A3B-Instruct-2507')"

sudo firewall-cmd --permanent --zone=trusted --add-source=10.140.80.0/22
sudo firewall-cmd --reload
```

## 2. 启动 Ray

四台机器都先执行：

```bash
cd ~/vllm
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
.venv/bin/ray stop --force
```

head 执行：

```bash
.venv/bin/ray start --head --port=6379 --disable-usage-stats
```

其余三台执行：

```bash
.venv/bin/ray start --address=10.140.83.156:6379 --disable-usage-stats
```

head 验证：

```bash
.venv/bin/ray status
```

应看到 4 个活动节点、4 张 GPU，且没有失败节点。

## 3. 运行一组实验

每组都重新启动服务。先设置组名和策略：

```bash
# off
TAG=off_run1
USE_BATCHING=false
POLICY=first_fit

# first_fit 时改为：
# TAG=first_fit_run1
# USE_BATCHING=true
# POLICY=first_fit

# degree_desc 时改为：
# TAG=degree_desc_run1
# USE_BATCHING=true
# POLICY=degree_desc
```

然后启动服务：

```bash
cd ~/vllm
source .venv/bin/activate
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
export VLLM_EPLB_LOG_MIGRATION_STATS=1

MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
RESULTS=~/benchmarks/eplb_rebalance/$TAG
mkdir -p "$RESULTS"

EPLB_CONFIG="{\"window_size\":50,\"step_interval\":100,\"num_redundant_experts\":16,\"log_balancedness\":false,\"use_async\":false,\"use_migration_batching\":$USE_BATCHING,\"migration_batching_policy\":\"$POLICY\"}"

.venv/bin/vllm serve "$MODEL" --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.8 --max-model-len 4096 \
  --no-enable-prefix-caching --tensor-parallel-size 4 \
  --distributed-executor-backend ray --enable-expert-parallel \
  --enable-eplb --eplb-config "$EPLB_CONFIG" --enforce-eager \
  > "$RESULTS/server.log" 2>&1 &
SERVER_PID=$!

until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 5; done
```

发送一个短预热请求：

```bash
curl -sf http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"warm up\",\"max_tokens\":10}" \
  > "$RESULTS/warmup.json"
```

压测：

```bash
.venv/bin/vllm bench serve --backend vllm --model "$MODEL" --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 200 --random-output-len 50 \
  --num-prompts 200 --max-concurrency 32 --seed 0 \
  --percentile-metrics ttft,tpot,e2el \
  --metric-percentiles 50,90,95,99 --ignore-eos --disable-tqdm \
  --save-result --result-dir "$RESULTS" --result-filename bench.json \
  > "$RESULTS/bench.log" 2>&1

kill "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true
```

依次运行 `off`、`first_fit`、`degree_desc`。正式实验各重复 3 次，修改 `run1` 为
`run2`、`run3`，并轮换三种策略的执行顺序。

## 4. 查看结果

```bash
cat "$RESULTS/bench.log"
rg "EPLB migration stats|Rearranged experts" "$RESULTS/server.log"
```

主要比较：

- 请求吞吐、输出吞吐；
- TTFT、TPOT、E2EL 的 P50/P95/P99；
- 真实 rearrangement 时间；
- expert transfers、合并后的 rank-pair flows、batch 数。

日志格式为：

```text
EPLB migration stats: ... transfers, ... flows, ... batches, ...
Rearranged experts ... in ... s.
```

注意：

- `off` 就是本实验的对照组，不运行单独的 `main`；
- 当前保持同步 EPLB，`use_async=true` 不会让 batch 数变成 1，但属于另一类实验；
- 同一 `src→dst` 的多个 expert 已合并为一个 flow，避免逐 expert 串行；
- 四 rank 密集图中，`first_fit` 和 `degree_desc` 可能产生相同 batch 数；
- P95 比 200 请求下的 P99 更稳定；
- profile rearrangement 不会写入 migration stats，时间日志中带 `(profile)` 的记录也
  不属于正式实验。

## 5. 停止集群

四台机器执行：

```bash
cd ~/vllm
.venv/bin/ray stop --force
```
