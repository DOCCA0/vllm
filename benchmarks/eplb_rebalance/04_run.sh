#!/bin/bash
# 04 - One experiment group: serve -> skewed traffic (hot spot) -> bench -> collect
#
# Usage:
#   WSL single-GPU smoke (EPLB off, pipeline validation only):
#     bash 04_run.sh
#   Cluster smoke with the small model (run 03_cluster_up.sh first):
#     MODE=cluster IFACE=<iface> BATCHING=degree_desc bash 04_run.sh
#   Live 30B benchmark:
#     MODE=cluster IFACE=<iface> MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
#       NUM_REDUNDANT=32 BATCHING=degree_desc bash 04_run.sh
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1
VLLM=.venv/bin/vllm
REPO_ROOT=$(pwd)

# ---------- Tunables ----------
MODE=${MODE:-single}              # single | cluster
MODEL=${MODEL:-tiny-random/qwen3-moe}   # live: Qwen/Qwen3-30B-A3B-Instruct-2507
BATCHING=${BATCHING:-degree_desc} # off | first_fit | degree_desc (cluster only)
TP=${TP:-4}                       # cluster: 4 nodes x 1 GPU
DTYPE=${DTYPE:-float16}           # Quadro RTX 6000 is Turing (sm75): no bf16
PORT=${PORT:-8000}

EPLB_STEP=${EPLB_STEP:-200}       # rearrange every N engine steps
EPLB_WINDOW=${EPLB_WINDOW:-1000}
NUM_REDUNDANT=${NUM_REDUNDANT:-8} # more redundancy -> more migrations; use 32 for 30B

INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-128}
PREFIX_LEN=${PREFIX_LEN:-384}     # shared prefix -> routing hot spot
SKEW_INPUT_LEN=${SKEW_INPUT_LEN:-8}   # phase A: ~98% shared tokens
READY_TIMEOUT=${READY_TIMEOUT:-1800}
NODES=(${CLUSTER_NODES:-10.31.0.243 10.31.0.244 10.31.0.247 10.31.0.249})
REPO_DIR=${REPO_DIR:-$REPO_ROOT}

# Mode-dependent defaults
if [[ $MODE == single ]]; then
  # WSL laptop: 8GB card (only ~6.9GB usable), small load
  GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.80}; MAX_LEN=${MAX_LEN:-2048}
  NUM_PROMPTS=${NUM_PROMPTS:-200}; CONCURRENCY=${CONCURRENCY:-8}
else
  GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}; MAX_LEN=${MAX_LEN:-4096}
  NUM_PROMPTS=${NUM_PROMPTS:-1000}; CONCURRENCY=${CONCURRENCY:-32}
fi

# NCCL over the TCP Ethernet VLAN: every cross-GPU transfer hits the NIC
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET
if [[ -n "${IFACE:-}" ]]; then
  export NCCL_SOCKET_IFNAME=$IFACE GLOO_SOCKET_IFNAME=$IFACE
fi

TAG=${TAG:-${MODE}_${BATCHING}_$(basename "$MODEL" | tr '/.' '__')}
RESULTS=benchmarks/eplb_rebalance/results/$TAG
mkdir -p "$RESULTS"
echo "[bench] TAG=$TAG  results -> $RESULTS"

# ---------- Server launch args ----------
SERVE_ARGS=(serve "$MODEL" --dtype "$DTYPE" --port "$PORT"
  --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_LEN"
  --no-enable-prefix-caching)

if [[ $MODE == cluster ]]; then
  USE_BATCHING=true; [[ $BATCHING == off ]] && USE_BATCHING=false
  EPLB_CONFIG="{\"window_size\":$EPLB_WINDOW,\"step_interval\":$EPLB_STEP,\"num_redundant_experts\":$NUM_REDUNDANT,\"log_balancedness\":true,\"use_migration_batching\":$USE_BATCHING,\"migration_batching_policy\":\"$([[ $BATCHING == off ]] && echo first_fit || echo $BATCHING)\"}"
  # A 24GB card cannot hold a 30B FP16 replica (~61GB), ruling out DP=4.
  # TP=4 + --enable-expert-parallel shards experts across 4 ranks, so the
  # EPLB group is EP=4. Cross-node TP requires the Ray executor (03_cluster_up.sh).
  SERVE_ARGS+=(--tensor-parallel-size "$TP" --distributed-executor-backend ray
    --enable-expert-parallel --enable-eplb --eplb-config "$EPLB_CONFIG")
  export VLLM_EPLB_LOG_MIGRATION_STATS=1   # per-rank migration stats (hot spot metric)
else
  # EPLB requires TP*DP>1, so a single GPU cannot enable it; this mode only
  # smoke-tests the server/traffic/report pipeline.
  echo "[bench] single mode: EPLB off, pipeline smoke test only"
fi

# ---------- NIC sampling (cluster only, requires IFACE; every node incl. head) ----------
nic_start_all() {
  [ -z "${IFACE:-}" ] && return 0
  for ip in "${NODES[@]}"; do
    ssh cc@$ip "cd $REPO_DIR && rm -f /tmp/eplb_nic.txt && \
      nohup bash benchmarks/eplb_rebalance/nic_sampler.sh $IFACE /tmp/eplb_nic.txt \
      > /dev/null 2>&1 & echo ok" > /dev/null
  done
}
nic_stop_all() {
  [ -z "${IFACE:-}" ] && return 0
  for ip in "${NODES[@]}"; do
    ssh cc@$ip "pkill -f nic_sampler.sh 2>/dev/null; cat /tmp/eplb_nic.txt 2>/dev/null" \
      > "$RESULTS/nic_$ip.txt" || true
  done
}

# ---------- Launch server ----------
echo "[bench] launch: $VLLM ${SERVE_ARGS[*]}"
"$VLLM" "${SERVE_ARGS[@]}" > "$RESULTS/server.log" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 "$READY_TIMEOUT"); do
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[bench] server exited, see $RESULTS/server.log"; exit 1
  fi
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "[bench] server not ready (timeout)"; kill $SERVER_PID; exit 1; }
echo "[bench] server ready"

[[ $MODE == cluster ]] && nic_start_all

# ---------- Phase A: skewed traffic (~98% shared prefix -> few hot experts -> hot GPU) ----------
echo "[bench] phase A: skewed x$((NUM_PROMPTS/2))"
"$VLLM" bench serve --backend vllm --model "$MODEL" --port "$PORT" \
  --dataset-name random --random-prefix-len "$PREFIX_LEN" \
  --random-input-len "$SKEW_INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$((NUM_PROMPTS/2))" --max-concurrency "$CONCURRENCY" \
  --ignore-eos --disable-tqdm > "$RESULTS/bench_skew.log" 2>&1

# ---------- Phase B: main benchmark (latency metrics) ----------
echo "[bench] phase B: main x$NUM_PROMPTS"
"$VLLM" bench serve --backend vllm --model "$MODEL" --port "$PORT" \
  --dataset-name random --random-prefix-len "$PREFIX_LEN" \
  --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONCURRENCY" \
  --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
  --ignore-eos --disable-tqdm \
  --save-result --result-dir "$RESULTS" --result-filename bench.json \
  > "$RESULTS/bench_main.log" 2>&1

# ---------- Teardown ----------
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
[[ $MODE == cluster ]] && nic_stop_all
echo "[bench] done: $TAG"
