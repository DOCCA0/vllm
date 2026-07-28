#!/bin/bash
# 03 - Start/stop the 4-node Ray cluster (run from the laptop, via jump host;
# one GPU per node)
# Prerequisites: 01_setup.sh on every node; 02_firewall.sh done
# Usage: CLUSTER_NODES="ip1 ip2 ..." IFACE=<vlan_iface> bash 03_cluster_up.sh <FLOAT_IP> [down]
FLOAT=$1
ACTION=${2:-up}
[ -z "${CLUSTER_NODES:-}" ] && { echo "error: CLUSTER_NODES is required (space-separated node IPs)" >&2; exit 1; }
NODES=($CLUSTER_NODES)
HEAD=${NODES[0]}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}   # must be identical on every node
RAY_PORT=6379

SSH="ssh -J cc@$FLOAT cc@"
ENV_EXPORT="export NCCL_IB_DISABLE=1 NCCL_SOCKET_FAMILY=AF_INET"
[ -n "${IFACE:-}" ] && ENV_EXPORT="$ENV_EXPORT NCCL_SOCKET_IFNAME=$IFACE GLOO_SOCKET_IFNAME=$IFACE"

if [[ $ACTION == down ]]; then
  for ip in "${NODES[@]}"; do
    $SSH$ip "cd $REPO_DIR && .venv/bin/ray stop --force" &
  done
  wait; echo "ray cluster stopped"; exit 0
fi

$SSH$HEAD "cd $REPO_DIR && $ENV_EXPORT && .venv/bin/ray stop --force; \
  .venv/bin/ray start --head --port=$RAY_PORT --disable-usage-stats"
for ip in "${NODES[@]:1}"; do
  $SSH$ip "cd $REPO_DIR && $ENV_EXPORT && .venv/bin/ray stop --force; \
    .venv/bin/ray start --address=$HEAD:$RAY_PORT --disable-usage-stats" &
done
wait

$SSH$HEAD "cd $REPO_DIR && .venv/bin/ray status"   # expect 4 nodes x 1 GPU
echo "ray cluster up. Now run 04_run.sh / 05_all.sh on the head node ($HEAD)"
