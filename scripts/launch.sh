#!/usr/bin/env bash
# launch.sh -- start a long-running job that survives everything.
#
# `setsid` puts the job in its own session and process group, so it is not a
# child of this shell, the SSH connection, or the IDE. It
# keeps running when all of those die. This is the only sanctioned way to start
# a sweep or a training run; anything started as a normal background job of an
# interactive shell will be killed on hangup.
#
# Usage:
#   scripts/launch.sh <gpu> <job-name> <command...>
#
# Example:
#   scripts/launch.sh 1 gold-cifar10-c3 \
#       python "Baseline CIFAR Training/train_retain_cifar.py" \
#              --dataset cifar10 --forget-class 3
#
# Then:  scripts/status.sh          (progress)
#        pgrep -af <job-name>       (is it alive?)
#        tail -f logs/<job>.gpu<n>.log
set -euo pipefail

if [ $# -lt 3 ]; then
    sed -n '2,20p' "$0" >&2
    exit 2
fi

GPU="$1"; shift
JOB="$1"; shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

LOG="logs/${JOB}.gpu${GPU}.log"
PY="${UNLEARN_PYTHON:-python}"
: "${UNLEARN_MODELS_ROOT:=$REPO_ROOT/models}"
export UNLEARN_MODELS_ROOT

# Required for deterministic cuBLAS reductions, and it must be set before the
# CUDA context is created -- setting it from inside Python is too late.
: "${CUBLAS_WORKSPACE_CONFIG:=:4096:8}"
export CUBLAS_WORKSPACE_CONFIG

# Replace a bare `python` with $UNLEARN_PYTHON (default: python on PATH), so
# every job in a run uses the same interpreter.
if [ "$1" = "python" ]; then shift; set -- "$PY" "$@"; fi

{
    echo "=== $JOB ==="
    echo "started:      $(date -Is)"
    echo "gpu:          CUDA_VISIBLE_DEVICES=$GPU"
    echo "models_root:  $UNLEARN_MODELS_ROOT"
    echo "git:          $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "command:      $*"
    echo "==="
} >> "$LOG"

# -u so stdout is unbuffered and the log is useful while the job runs.
setsid nohup env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    "$@" >> "$LOG" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "logs/${JOB}.gpu${GPU}.pid"
echo "launched '$JOB' on GPU $GPU as pid $PID"
echo "  log: $LOG"
