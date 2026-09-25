#!/usr/bin/env bash
# vit_gold_sweep.sh -- train ViT-B/16 retrain gold models for one dataset,
# one forget class at a time (sequential: parallel runs on one GPU only thrash).
#
# Each class is idempotent: completed classes are skipped, an interrupted one
# resumes from its last completed epoch.
#
# Usage:  scripts/vit_gold_sweep.sh <dataset> <class> [class...]
# Launch: scripts/launch.sh 0 vit-gold-cifar10 bash scripts/vit_gold_sweep.sh cifar10 0 2 3 5 8
set -uo pipefail

DATASET="$1"; shift
CLASSES=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${UNLEARN_PYTHON:-python}"

echo "vit gold sweep: $DATASET classes ${CLASSES[*]}"
failed=()
for c in "${CLASSES[@]}"; do
    echo
    echo "################ $DATASET class $c ################"
    if ! "$PY" "Baseline ViT Training/train_vit_cifar.py" --dataset "$DATASET" \
            --forget-class "$c" --device cuda:0; then
        echo "!!! FAILED: $DATASET class $c (continuing)"
        failed+=("$c")
    fi
done

echo
if [ ${#failed[@]} -eq 0 ]; then
    echo "vit gold sweep complete: $DATASET all ${#CLASSES[@]} classes"
else
    echo "vit gold sweep finished with failures: ${failed[*]}"
    exit 1
fi
