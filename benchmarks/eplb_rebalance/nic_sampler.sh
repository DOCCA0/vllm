#!/bin/bash
# Helper (invoked remotely by 04_run.sh on each node): sample NIC throughput
# until killed. Writes "<ts_ns> <rx_MBps> <tx_MBps>" lines.
# Usage: bash nic_sampler.sh <iface> <outfile> [interval_sec]
IFACE=$1; OUT=$2; INTERVAL=${3:-0.5}

read rx0 < /sys/class/net/$IFACE/statistics/rx_bytes
read tx0 < /sys/class/net/$IFACE/statistics/tx_bytes
t0=$(date +%s%N)
while sleep "$INTERVAL"; do
  read rx < /sys/class/net/$IFACE/statistics/rx_bytes
  read tx < /sys/class/net/$IFACE/statistics/tx_bytes
  t1=$(date +%s%N)
  echo "$t1 $(( (rx-rx0)*8000/(t1-t0) )) $(( (tx-tx0)*8000/(t1-t0) ))" >> "$OUT"
  rx0=$rx; tx0=$tx; t0=$t1
done
