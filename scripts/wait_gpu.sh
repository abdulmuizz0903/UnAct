#!/usr/bin/env bash
# wait_gpu.sh <gpu> <mib-threshold> -- block until that GPU's used memory drops
# below the threshold for three consecutive samples, then exit 0.
# Used to queue work behind jobs whose pid we do not own.
set -u
GPU="$1"; THRESH="${2:-1500}"; HITS=0
while :; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 99999)
    if [ "$USED" -lt "$THRESH" ]; then HITS=$((HITS+1)); else HITS=0; fi
    [ "$HITS" -ge 3 ] && break
    sleep 30
done
echo "gpu $GPU free (${USED} MiB); proceeding at $(date -Is)"
