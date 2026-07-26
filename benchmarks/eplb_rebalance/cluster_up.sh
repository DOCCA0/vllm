#!/bin/bash
# 在 4 节点 VLAN 上拉起 Ray 集群(每个节点 1 张 GPU), 供跨节点 TP=4 + EPLB 使用
# 前置: 先跑 bootstrap.sh 放行防火墙; 各节点同路径装好 .venv 并下载好模型
# 用法: ./cluster_up.sh <浮动IP> [down]
FLOAT=$1
ACTION=${2:-up}
NODES=(10.31.0.243 10.31.0.244 10.31.0.247 10.31.0.249)   # 每次换新lease只改这行
HEAD=${NODES[0]}
REPO_DIR=${REPO_DIR:-~/vllm}      # 各节点上本仓库的路径(必须一致)
IFACE=${IFACE:-}                  # VLAN 网卡名, 如 enp1s0f1; 不填则 NCCL 自选
RAY_PORT=6379

SSH="ssh -J cc@$FLOAT cc@"
ENV_EXPORT="export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET"
[ -n "$IFACE" ] && ENV_EXPORT="$ENV_EXPORT NCCL_SOCKET_IFNAME=$IFACE GLOO_SOCKET_IFNAME=$IFACE"

if [[ $ACTION == down ]]; then
  for ip in "${NODES[@]}"; do
    $SSH$ip "cd $REPO_DIR && .venv/bin/ray stop --force" &
  done
  wait; echo "ray cluster stopped"; exit 0
fi

# head 节点
$SSH$HEAD "cd $REPO_DIR && $ENV_EXPORT && .venv/bin/ray stop --force; \
  .venv/bin/ray start --head --port=$RAY_PORT --disable-usage-stats"

# worker 节点
for ip in "${NODES[@]:1}"; do
  $SSH$ip "cd $REPO_DIR && $ENV_EXPORT && .venv/bin/ray stop --force; \
    .venv/bin/ray start --address=$HEAD:$RAY_PORT --disable-usage-stats" &
done
wait

# 验证: 应看到 4 个节点各 1 GPU
$SSH$HEAD "cd $REPO_DIR && .venv/bin/ray status"
echo "ray cluster up. 然后在 head 节点($HEAD)上跑:"
echo "  MODE=cluster bash benchmarks/eplb_rebalance/bench_run.sh"
