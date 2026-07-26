#!/bin/bash
# 多卡对照实验: baseline(off) / first_fit / degree_desc 三组连跑, 最后汇总
# 用法: bash benchmarks/eplb_rebalance/bench_all.sh
#   上线 30B: MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 NUM_REDUNDANT=32 bash benchmarks/eplb_rebalance/bench_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL_SHORT=$(basename "${MODEL:-Qwen/Qwen1.5-MoE-A2.7B-Chat}" | tr '/.' '__')

for b in off first_fit degree_desc; do
  TAG="multi_${b}_${MODEL_SHORT}" BATCHING=$b bash benchmarks/eplb_rebalance/bench_run.sh
done

.venv/bin/python benchmarks/eplb_rebalance/bench_report.py
