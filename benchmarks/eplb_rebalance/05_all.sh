#!/bin/bash
# 05 - Cluster comparison: baseline(off) / first_fit / degree_desc + report
# Usage (on the head node):
#   MODE=cluster IFACE=<iface> bash 05_all.sh                    # small-model smoke
#   MODE=cluster IFACE=<iface> MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
#     NUM_REDUNDANT=32 bash 05_all.sh                            # 30B live run
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL_SHORT=$(basename "${MODEL:-tiny-random/qwen3-moe}" | tr '/.' '__')

for b in off first_fit degree_desc; do
  TAG="multi_${b}_${MODEL_SHORT}" BATCHING=$b MODE=${MODE:-cluster} bash "$HERE/04_run.sh"
done

"$HERE/06_report.py"
