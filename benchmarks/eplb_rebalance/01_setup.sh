#!/bin/bash
# 01 - Environment setup: venv + vllm + model download (run on EVERY node)
# Usage: bash 01_setup.sh [MODEL_ID ...]   (default: tiny-random/qwen3-moe)
#   Before the live run: bash 01_setup.sh Qwen/Qwen3-30B-A3B-Instruct-2507
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1   # the mirror does not proxy the xet CDN

[ ! -d .venv ] && uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto   # pure-Python changes: no rebuild
uv pip install "huggingface_hub[cli]" "ray[default]"              # ray: cross-node TP

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(tiny-random/qwen3-moe)
for m in "${MODELS[@]}"; do
  .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$m')"
done
echo "setup done"
