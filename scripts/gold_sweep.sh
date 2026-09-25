#!/usr/bin/env bash
# gold_sweep.sh -- train retrain-from-scratch gold models for one dataset,
# one class at a time (sequential: parallel runs on one GPU only thrash).
#
# Each class is idempotent, so re-running this after an interruption resumes:
# completed classes are skipped, an interrupted one restarts from its last epoch.
#
# Usage:  scripts/gold_sweep.sh <dataset> <class> [class...]
# Launch: scripts/launch.sh 1 gold-cifar10 bash scripts/gold_sweep.sh cifar10 0 2 3 5 8
set -uo pipefail

DATASET="$1"; shift
CLASSES=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/Baseline CIFAR Training"
PY="${UNLEARN_PYTHON:-python}"

echo "gold sweep: $DATASET classes ${CLASSES[*]}"
failed=()
for c in "${CLASSES[@]}"; do
    echo
    echo "################ $DATASET class $c ################"
    if ! "$PY" train_retain_cifar.py --dataset "$DATASET" --forget-class "$c" \
            --device cuda:0 --workers 8; then
        echo "!!! FAILED: $DATASET class $c (continuing)"
        failed+=("$c")
    fi
done

echo
if [ ${#failed[@]} -eq 0 ]; then
    echo "gold sweep complete: $DATASET all ${#CLASSES[@]} classes"
else
    echo "gold sweep finished with failures: ${failed[*]}"
    echo "re-run the same command to retry only those"
    exit 1
fi
