#!/bin/bash
# 用法: ./bootstrap.sh <浮动IP>
FLOAT=$1
NODES=(10.31.0.243 10.31.0.244 10.31.0.247 10.31.0.249)   # 每次换新lease只改这行

for ip in "${NODES[@]}"; do
  ssh -J cc@$FLOAT cc@$ip '
    sudo firewall-cmd --permanent --zone=trusted --add-source=10.31.0.0/24
    sudo firewall-cmd --reload
  ' 
done
wait
echo "all nodes configured"