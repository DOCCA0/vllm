# 登录
```bash
ssh -J cc@192.5.86.236 cc@10.140.83.156
ssh -J cc@192.5.86.236 cc@10.140.83.96
ssh -J cc@192.5.86.236 cc@10.140.83.28
ssh -J cc@192.5.86.236 cc@10.140.83.94
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
uv venv ~/.venv --python 3.11 --clear
source ~/.venv/bin/activate

uv pip uninstall torch torchvision torchaudio
uv pip uninstall nvidia-cublas-cu13 nvidia-cuda-cupti-cu13 nvidia-cuda-nvrtc-cu13 nvidia-cuda-runtime-cu13 nvidia-cudnn-cu13 nvidia-cufft-cu13 nvidia-cufile-cu13 nvidia-curand-cu13 nvidia-cusolver-cu13 nvidia-cusparse-cu13 nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink-cu13 nvidia-nvshmem-cu13 nvidia-nvtx-cu13
uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; torch.cuda.init(); print(torch.__version__, torch.version.cuda)"

cd ~/vllm
$-f use_existing_torch.py$ || curl -O https://raw.githubusercontent.com/vllm-project/vllm/main/use_existing_torch.py
python use_existing_torch.py
uv pip install setuptools_scm
VLLM_USE_PRECOMPILED=1 uv pip install -e . --no-build-isolation

sudo firewall-cmd --permanent --zone=trusted --add-source=10.140.80.0/22
sudo firewall-cmd --reload

uv pip install "ray[default]"
uv pip install "huggingface_hub[cli]"

source ~/.venv/bin/activate
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-30B-A3B-Instruct-2507')"
```


# 查网卡名（本集群已确认为 eno1np0）
```bash
ip -4 -brief addr show
```
# 启动 Ray 集群
## head 节点：
```bash
cd ~/vllm
source ~/.venv/bin/activate
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
~/.venv/bin/ray stop --force
~/.venv/bin/ray start --head --port=6379 --disable-usage-stats
```
## worker 节点
```bash
cd ~/vllm
source ~/.venv/bin/activate
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
~/.venv/bin/ray stop --force
~/.venv/bin/ray start --address=10.140.83.156:6379 --disable-usage-stats
```
## 验证（head，应为 4 节点 x 1 GPU）：
```bash
~/.venv/bin/ray status
```
# 实验
## 启动服务（head；Quadro RTX 6000 是 sm75 只能 float16）

```bash
# baseline（不 batching）
cd ~/vllm
source ~/.venv/bin/activate
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
export VLLM_EPLB_LOG_MIGRATION_STATS=1
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507   

EPLB_CONFIG='{"window_size":1000,"step_interval":200,"num_redundant_experts":16,"log_balancedness":true,"use_migration_batching":false,"migration_batching_policy":"first_fit"}'

mkdir -p ~/benchmarks
~/.venv/bin/vllm serve $MODEL --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.8 --max-model-len 4096 --no-enable-prefix-caching \
  --tensor-parallel-size 4 --distributed-executor-backend ray \
  --enable-expert-parallel --enable-eplb --eplb-config "$EPLB_CONFIG" --enforce-eager \
  > ~/benchmarks/start_baseline.log 
```

```bash
# first_fit
cd ~/vllm
source ~/.venv/bin/activate
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
export VLLM_EPLB_LOG_MIGRATION_STATS=1
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507   

EPLB_CONFIG='{"window_size":1000,"step_interval":200,"num_redundant_experts":16,"log_balancedness":true,"use_migration_batching":true,"migration_batching_policy":"first_fit"}'

mkdir -p ~/benchmarks
~/.venv/bin/vllm serve $MODEL --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.8 --max-model-len 4096 --no-enable-prefix-caching \
  --tensor-parallel-size 4 --distributed-executor-backend ray \
  --enable-expert-parallel --enable-eplb --eplb-config "$EPLB_CONFIG" --enforce-eager \
  > ~/benchmarks/start_first_fit.log  
```

```bash
# degree_desc
cd ~/vllm
source ~/.venv/bin/activate
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
export VLLM_EPLB_LOG_MIGRATION_STATS=1
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507   

EPLB_CONFIG='{"window_size":1000,"step_interval":200,"num_redundant_experts":16,"log_balancedness":true,"use_migration_batching":true,"migration_batching_policy":"degree_desc"}'

mkdir -p ~/benchmarks
~/.venv/bin/vllm serve $MODEL --dtype float16 --port 8000 \
  --gpu-memory-utilization 0.8 --max-model-len 4096 --no-enable-prefix-caching \
  --tensor-parallel-size 4 --distributed-executor-backend ray \
  --enable-expert-parallel --enable-eplb --eplb-config "$EPLB_CONFIG" --enforce-eager \
  > ~/benchmarks/start_degree_desc.log
```
### 验证启动成功
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "prompt": "who are you",
    "max_tokens": 10
  }'
```

## 压测（head，延迟指标）：
```bash
~/.venv/bin/vllm bench serve --backend vllm --model Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 200 --random-output-len 100 \
  --num-prompts 100 --max-concurrency 32 \
  --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
  --ignore-eos \
  --save-result --result-dir ~/benchmarks --result-filename result_base.json 
```
  
```bash
~/.venv/bin/vllm bench serve --backend vllm --model Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 200 --random-output-len 100 \
  --num-prompts 100 --max-concurrency 32 \
  --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
  --ignore-eos \
  --save-result --result-dir ~/benchmarks --result-filename result_firstfit.json 
```

```bash
~/.venv/bin/vllm bench serve --backend vllm --model Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8000 \
  --dataset-name random --random-prefix-len 300 \
  --random-input-len 200 --random-output-len 100 \
  --num-prompts 100 --max-concurrency 32 \
  --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
  --ignore-eos \
  --save-result --result-dir ~/benchmarks --result-filename result_degree.json 
```

