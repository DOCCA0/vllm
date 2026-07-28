#!/bin/bash
# 02 - Allow VLAN traffic between nodes (run from the laptop, via jump host)
# Usage: CLUSTER_NODES="ip1 ip2 ..." bash 02_firewall.sh <FLOAT_IP>
FLOAT=$1
[ -z "${CLUSTER_NODES:-}" ] && { echo "error: CLUSTER_NODES is required (space-separated node IPs)" >&2; exit 1; }
NODES=($CLUSTER_NODES)

for ip in "${NODES[@]}"; do
  ssh -J cc@$FLOAT cc@$ip '
    sudo firewall-cmd --permanent --zone=trusted --add-source=10.31.0.0/24
    sudo firewall-cmd --reload'
done
wait
echo "all nodes configured"
