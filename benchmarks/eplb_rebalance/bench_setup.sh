#!/bin/bash
# 一次性环境准备：venv + vllm 安装 + 模型下载
# 用法: bash benchmarks/eplb_rebalance/bench_setup.sh [模型ID ...]
#   默认下载冒烟小模型; 上线前再跑:
#   bash benchmarks/eplb_rebalance/bench_setup.sh Qwen/Qwen3-30B-A3B-Instruct-2507
set -euo pipefail
cd "$(dirname "$0")/../.."   # 仓库根目录

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1   # 镜像站不支持 xet CDN, 必须禁用   # 国内镜像加速

# 1. 虚拟环境
if [ ! -d .venv ]; then
  uv venv --python 3.12
fi

# 2. 安装 vllm(纯 Python 改动,直接复用上游预编译 wheel,省去编译)
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install "huggingface_hub[cli]" "ray[default]"   # ray: 集群跨节点 TP 用

# 3. 下载模型(默认冒烟用小 MoE)
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(Qwen/Qwen1.5-MoE-A2.7B-Chat)
for m in "${MODELS[@]}"; do
  .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$m')"
done
echo "setup done"
