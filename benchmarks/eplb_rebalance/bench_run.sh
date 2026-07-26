#!/bin/bash
# 单组实验: 起服务 -> 倾斜流量预热 EPLB 统计 -> 正式压测 -> 收集日志
#
# 用法:
#   WSL 单卡冒烟(不开 EPLB, 只验证流量/统计链路, 小模型):
#     MODE=single bash benchmarks/eplb_rebalance/bench_run.sh
#   集群冒烟(4节点小模型, 验证 batching 代码路径, 需先 cluster_up.sh):
#     MODE=cluster BATCHING=degree_desc bash benchmarks/eplb_rebalance/bench_run.sh
#   集群上线(Qwen3-30B 正式 benchmark):
#     MODE=cluster MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 NUM_REDUNDANT=32 \
#       BATCHING=degree_desc bash benchmarks/eplb_rebalance/bench_run.sh
set -uo pipefail
cd "$(dirname "$0")/../.."   # 仓库根目录
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1   # 镜像站不支持 xet CDN, 必须禁用
VLLM=.venv/bin/vllm

# ---------- 可调参数 ----------
MODE=${MODE:-single}              # single | cluster
MODEL=${MODEL:-Qwen/Qwen1.5-MoE-A2.7B-Chat}   # 上线: Qwen/Qwen3-30B-A3B-Instruct-2507
BATCHING=${BATCHING:-degree_desc} # off | first_fit | degree_desc (仅 cluster 生效)
TP=${TP:-4}                       # 集群 4 节点 x 1 GPU
DTYPE=${DTYPE:-float16}           # Quadro RTX 6000 是 Turing(sm75), 不支持 bf16
PORT=${PORT:-8000}

EPLB_STEP=${EPLB_STEP:-200}       # 每 200 step 触发一次 rebalance
EPLB_WINDOW=${EPLB_WINDOW:-1000}
NUM_REDUNDANT=${NUM_REDUNDANT:-8} # 冗余 expert 越多迁移量越大; 30B 上线建议 32

# 模式相关的默认值(必须先于通用默认值赋值)
if [[ $MODE == single ]]; then
  # WSL 单卡冒烟: 8GB 显存, 压小规模参数
  DEF_GPU_MEM_UTIL=0.80; DEF_MAX_LEN=2048; DEF_NUM_PROMPTS=200; DEF_CONCURRENCY=8   # WSL 8GB 卡实际可用仅 ~6.9GB
else
  DEF_GPU_MEM_UTIL=0.90; DEF_MAX_LEN=4096; DEF_NUM_PROMPTS=1000; DEF_CONCURRENCY=32
fi

NUM_PROMPTS=${NUM_PROMPTS:-$DEF_NUM_PROMPTS}
CONCURRENCY=${CONCURRENCY:-$DEF_CONCURRENCY}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-128}
PREFIX_LEN=${PREFIX_LEN:-384}     # 共享前缀 -> 路由热点, 制造 expert 倾斜
GPU_MEM_UTIL=${GPU_MEM_UTIL:-$DEF_GPU_MEM_UTIL}
MAX_LEN=${MAX_LEN:-$DEF_MAX_LEN}
READY_TIMEOUT=${READY_TIMEOUT:-1800}

# NCCL 走 TCP 以太网 VLAN(每对 GPU 的流量 100% 过 NIC):
# IFACE 用 ip -br addr 在节点上确认 VLAN 网卡名
IFACE=${IFACE:-}
if [[ -n $IFACE ]]; then
  export NCCL_SOCKET_IFNAME=$IFACE GLOO_SOCKET_IFNAME=$IFACE
fi
export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET

TAG=${TAG:-${MODE}_${BATCHING}_$(basename "$MODEL" | tr '/.' '__')}
RESULTS=benchmarks/eplb_rebalance/results/$TAG
mkdir -p "$RESULTS"
echo "[bench] TAG=$TAG  results -> $RESULTS"

# ---------- 组装启动参数 ----------
SERVE_ARGS=(serve "$MODEL" --dtype "$DTYPE" --port "$PORT"
  --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_LEN"
  --no-enable-prefix-caching)

if [[ $MODE == cluster ]]; then
  if [[ $BATCHING == off ]]; then USE_BATCHING=false; else USE_BATCHING=true; fi
  EPLB_CONFIG=$(cat <<JSON
{"window_size":$EPLB_WINDOW,"step_interval":$EPLB_STEP,
 "num_redundant_experts":$NUM_REDUNDANT,"log_balancedness":true,
 "use_migration_batching":$USE_BATCHING,
 "migration_batching_policy":"$([[ $BATCHING == off ]] && echo first_fit || echo $BATCHING)"}
JSON
)
  # 跨节点 TP 必须走 Ray 执行器(先跑 benchmarks/eplb_rebalance/cluster_up.sh 建好集群)
  # 说明: 24GB 单卡放不下 30B FP16(~61GB)完整副本, 故无法 DP=4;
  # TP=4 + --enable-expert-parallel 下专家切 4 份, EPLB 视角 EP group 即为 4 rank
  SERVE_ARGS+=(--tensor-parallel-size "$TP" --distributed-executor-backend ray
    --enable-expert-parallel --enable-eplb --eplb-config "$EPLB_CONFIG")
else
  # 单卡: EPLB 要求 TP*DP>1(config/parallel.py:481), 单卡不开 EPLB, 只验证流量链路
  echo "[bench] 单卡模式: 不开 EPLB, 仅验证 server/压测/统计链路"
fi

# ---------- 启动服务 ----------
echo "[bench] 启动: $VLLM ${SERVE_ARGS[*]}"
"$VLLM" "${SERVE_ARGS[@]}" > "$RESULTS/server.log" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 "$READY_TIMEOUT"); do
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[bench] 服务进程退出, 启动失败, 日志见 $RESULTS/server.log"; exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "[bench] 服务未就绪(超时)"; kill $SERVER_PID; exit 1; }
echo "[bench] 服务已就绪"

# ---------- 阶段 A: 倾斜流量(共享前缀制造 expert 热点, 预热 EPLB 统计窗口) ----------
echo "[bench] 阶段 A: 倾斜流量 x$((NUM_PROMPTS/2))"
"$VLLM" bench serve --backend vllm --model "$MODEL" --port "$PORT" \
  --dataset-name random --random-prefix-len "$PREFIX_LEN" \
  --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$((NUM_PROMPTS/2))" --max-concurrency "$CONCURRENCY" \
  --ignore-eos --disable-tqdm > "$RESULTS/bench_skew.log" 2>&1

# ---------- 阶段 B: 正式压测(保存延迟指标) ----------
echo "[bench] 阶段 B: 正式压测 x$NUM_PROMPTS"
"$VLLM" bench serve --backend vllm --model "$MODEL" --port "$PORT" \
  --dataset-name random --random-prefix-len "$PREFIX_LEN" \
  --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONCURRENCY" \
  --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
  --ignore-eos --disable-tqdm \
  --save-result --result-dir "$RESULTS" --result-filename bench.json \
  > "$RESULTS/bench_main.log" 2>&1

# ---------- 收尾 ----------
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
grep -oP "Rearranged experts.*?in \K[0-9.]+(?= s)" "$RESULTS/server.log" \
  > "$RESULTS/rearrange_times.txt" || true
N=$(wc -l < "$RESULTS/rearrange_times.txt")
echo "[bench] 完成: $TAG, 共 $N 次 rebalance, 日志在 $RESULTS/"
