# 登录
```bash
ssh -J cc@192.5.86.236 cc@10.140.83.136
ssh -J cc@192.5.86.236 cc@10.140.82.50
ssh -J cc@192.5.86.236 cc@10.140.82.147
ssh -J cc@192.5.86.236 cc@10.140.81.248
```
# clone（每台机器）
```bash
git clone https://github.com/DOCCA0/vllm.git ~/vllm
cd  vllm
git switch ilp
```
# 环境（每台机器）
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv ~/.venv --python 3.11
source ~/.venv/bin/activate

uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"   # True

VLLM_USE_PRECOMPILED=1 uv pip install -e .
uv pip install "huggingface_hub[cli]" "ray[default]"

sudo firewall-cmd --permanent --zone=trusted --add-source=10.140.80.0/22
sudo firewall-cmd --reload
```
## 30B
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-30B-A3B-Instruct-2507')"
```
## tiny
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('tiny-random/qwen3-moe')"
```
# 查网卡名（本集群已确认为 eno1np0）
```bash
ip -4 -brief addr show
```
# 启动 Ray 集群
## head 节点：
```bash
cd ~/vllm
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
~/.venv/bin/ray stop --force
~/.venv/bin/ray start --head --port=6379 --disable-usage-stats
```
## worker 节点
```bash
cd ~/vllm
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
~/.venv/bin/ray stop --force
~/.venv/bin/ray start --address=10.140.83.136:6379 --disable-usage-stats
```
## 验证（head，应为 4 节点 x 1 GPU）：
```bash
~/.venv/bin/ray status
```
# 实验（head 节点；off / first_fit / degree_desc 三组各跑一遍）
## 启动服务（head；Quadro RTX 6000 是 sm75 只能 float16）：：
### smoke
```bash
# baseline（不 batching）
cd ~/vllm
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
export VLLM_EPLB_LOG_MIGRATION_STATS=1
MODEL=tiny-random/qwen3-moe 

TAG=multi_off_tiny-random_qwen3-moe
EPLB_CONFIG='{"window_size":1000,"step_interval":200,"num_redundant_experts":8,"log_balancedness":true,"use_migration_batching":false,"migration_batching_policy":"first_fit"}'

RESULTS=benchmarks/eplb_rebalance/results/$TAG
mkdir -p $RESULTS
~/.venv/bin/vllm serve $MODEL --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.90 --max-model-len 4096 --no-enable-prefix-caching \
  --tensor-parallel-size 2 --distributed-executor-backend ray \
  --enable-expert-parallel --enable-eplb --eplb-config "$EPLB_CONFIG" \
  > $RESULTS/server.log 2>&1 &
until curl -sf http://127.0.0.1:8000/health > /dev/null; do sleep 5; done
```

```bash
# first_fit
cd ~/vllm
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
export VLLM_EPLB_LOG_MIGRATION_STATS=1
MODEL=tiny-random/qwen3-moe 

TAG=multi_first_fit_tiny-random_qwen3-moe
EPLB_CONFIG='{"window_size":1000,"step_interval":200,"num_redundant_experts":8,"log_balancedness":true,"use_migration_batching":true,"migration_batching_policy":"first_fit"}'

RESULTS=benchmarks/eplb_rebalance/results/$TAG
mkdir -p $RESULTS
~/.venv/bin/vllm serve $MODEL --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.90 --max-model-len 4096 --no-enable-prefix-caching \
  --tensor-parallel-size 2 --distributed-executor-backend ray \
  --enable-expert-parallel --enable-eplb --eplb-config "$EPLB_CONFIG" \
  > $RESULTS/server.log 2>&1 &
until curl -sf http://127.0.0.1:8000/health > /dev/null; do sleep 5; done
```

```bash
# degree_desc
cd ~/vllm
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
export VLLM_EPLB_LOG_MIGRATION_STATS=1
MODEL=tiny-random/qwen3-moe 

TAG=multi_degree_desc_tiny-random_qwen3-moe
EPLB_CONFIG='{"window_size":1000,"step_interval":200,"num_redundant_experts":8,"log_balancedness":true,"use_migration_batching":true,"migration_batching_policy":"degree_desc"}'

RESULTS=benchmarks/eplb_rebalance/results/$TAG
mkdir -p $RESULTS
~/.venv/bin/vllm serve $MODEL --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.90 --max-model-len 4096 --no-enable-prefix-caching \
  --tensor-parallel-size 2 --distributed-executor-backend ray \
  --enable-expert-parallel --enable-eplb --eplb-config "$EPLB_CONFIG" \
  > $RESULTS/server.log 2>&1 &
until curl -sf http://127.0.0.1:8000/health > /dev/null; do sleep 5; done
```

### 30B

```bash
# baseline（不 batching）
cd ~/vllm
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
export VLLM_EPLB_LOG_MIGRATION_STATS=1
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507   

TAG=multi_off_Qwen3-30B-A3B-Instruct-2507
EPLB_CONFIG='{"window_size":1000,"step_interval":200,"num_redundant_experts":32,"log_balancedness":true,"use_migration_batching":false,"migration_batching_policy":"first_fit"}'

RESULTS=benchmarks/eplb_rebalance/results/$TAG
mkdir -p $RESULTS
~/.venv/bin/vllm serve $MODEL --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.90 --max-model-len 4096 --no-enable-prefix-caching \
  --tensor-parallel-size 4 --distributed-executor-backend ray \
  --enable-expert-parallel --enable-eplb --eplb-config "$EPLB_CONFIG" \
  > $RESULTS/server.log 2>&1 &
until curl -sf http://127.0.0.1:8000/health > /dev/null; do sleep 5; done
```

```bash
# first_fit
cd ~/vllm
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
export VLLM_EPLB_LOG_MIGRATION_STATS=1
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507   

TAG=multi_first_fit_Qwen3-30B-A3B-Instruct-2507
EPLB_CONFIG='{"window_size":1000,"step_interval":200,"num_redundant_experts":32,"log_balancedness":true,"use_migration_batching":true,"migration_batching_policy":"first_fit"}'

RESULTS=benchmarks/eplb_rebalance/results/$TAG
mkdir -p $RESULTS
~/.venv/bin/vllm serve $MODEL --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.90 --max-model-len 4096 --no-enable-prefix-caching \
  --tensor-parallel-size 4 --distributed-executor-backend ray \
  --enable-expert-parallel --enable-eplb --eplb-config "$EPLB_CONFIG" \
  > $RESULTS/server.log 2>&1 &
until curl -sf http://127.0.0.1:8000/health > /dev/null; do sleep 5; done
```

```bash
# degree_desc
cd ~/vllm
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
export VLLM_EPLB_LOG_MIGRATION_STATS=1
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507   

TAG=multi_degree_desc_Qwen3-30B-A3B-Instruct-2507
EPLB_CONFIG='{"window_size":1000,"step_interval":200,"num_redundant_experts":32,"log_balancedness":true,"use_migration_batching":true,"migration_batching_policy":"degree_desc"}'

RESULTS=benchmarks/eplb_rebalance/results/$TAG
mkdir -p $RESULTS
~/.venv/bin/vllm serve $MODEL --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.90 --max-model-len 4096 --no-enable-prefix-caching \
  --tensor-parallel-size 4 --distributed-executor-backend ray \
  --enable-expert-parallel --enable-eplb --eplb-config "$EPLB_CONFIG" \
  > $RESULTS/server.log 2>&1 &
until curl -sf http://127.0.0.1:8000/health > /dev/null; do sleep 5; done
```

启动 NIC 采样（服务 ready 后、压测前，4 台机器都执行）：
```bash
rm -f /tmp/eplb_nic.txt
nohup bash -c '
IF=eno1np0; OUT=/tmp/eplb_nic.txt
read rx0 < /sys/class/net/$IF/statistics/rx_bytes; read tx0 < /sys/class/net/$IF/statistics/tx_bytes; t0=$(date +%s%N)
while sleep 0.5; do
  read rx < /sys/class/net/$IF/statistics/rx_bytes; read tx < /sys/class/net/$IF/statistics/tx_bytes; t1=$(date +%s%N)
  echo "$t1 $(( (rx-rx0)*8000/(t1-t0) )) $(( (tx-tx0)*8000/(t1-t0) ))" >> $OUT
  rx0=$rx; tx0=$tx; t0=$t1
done' > /dev/null 2>&1 &
```
## 阶段 A：热点构造（head，~98% 共享前缀 -> 少数热 expert -> 热卡）：
```bash
~/.venv/bin/vllm bench serve --backend vllm --model $MODEL --port 8000 \
  --dataset-name random --random-prefix-len 384 \
  --random-input-len 8 --random-output-len 128 \
  --num-prompts 500 --max-concurrency 32 \
  --ignore-eos --disable-tqdm > $RESULTS/bench_skew.log 2>&1
```
阶段 B：主压测（head，延迟指标）：
```bash
~/.venv/bin/vllm bench serve --backend vllm --model $MODEL --port 8000 \
  --dataset-name random --random-prefix-len 384 \
  --random-input-len 512 --random-output-len 128 \
  --num-prompts 1000 --max-concurrency 32 \
  --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
  --ignore-eos --disable-tqdm \
  --save-result --result-dir $RESULTS --result-filename bench.json \
  > $RESULTS/bench_main.log 2>&1
```
收尾（head；停服务 + 停采样并收回 4 台机器的 NIC 数据）：
```bash
pkill -f 'vllm serve'
for ip in 10.140.83.136 10.140.82.50 10.140.82.147 10.140.81.248; do
  ssh cc@$ip "pkill -f 'eplb_nic[.]txt'; cat /tmp/eplb_nic.txt" > $RESULTS/nic_$ip.txt
done
```
换下一组：重设 TAG / EPLB_CONFIG，从「启动服务」重复。

# 汇总（head，三组跑完后）
```bash
~/.venv/bin/python benchmarks/eplb_rebalance/06_report.py
```
# 关闭集群（每台机器）
```bash
~/.venv/bin/ray stop --force
```
