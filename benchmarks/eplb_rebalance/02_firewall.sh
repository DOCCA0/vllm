#!/bin/bash
# 02 - Allow VLAN traffic between nodes (run from the laptop, via jump host)
# Usage: bash 02_firewall.sh <FLOAT_IP>
FLOAT=$1
NODES=(${CLUSTER_NODES:-10.31.0.243 10.31.0.244 10.31.0.247 10.31.0.249})   # update per lease

for ip in "${NODES[@]}"; do
  ssh -J cc@$FLOAT cc@$ip '
    sudo firewall-cmd --permanent --zone=trusted --add-source=10.31.0.0/24
    sudo firewall-cmd --reload'
done
wait
echo "all nodes configured"
