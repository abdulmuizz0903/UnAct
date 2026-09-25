#!/usr/bin/env bash
# Retrain gold models for nested multi-class forget sets, for E4.
# Nested prefixes of each dataset's pre-registered 5 classes, so the k=1 golds
# already trained are reused as the k=1 point.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/Baseline CIFAR Training"
PY="${UNLEARN_PYTHON:-python}"

run () { echo; echo "######## $1 : $2 ########"
         "$PY" train_retain_cifar.py --dataset "$1" --forget-class $2 --device cuda:0 --workers 8 || echo "!!! FAILED $1 $2"; }

for spec in "0 2" "0 2 3" "0 2 3 5 8"; do run cifar10 "$spec"; done
for spec in "3 4" "3 4 10" "3 4 10 14 19"; do run cifar20 "$spec"; done
for spec in "3 20" "3 20 51" "3 20 51 69 85"; do run cifar100 "$spec"; done
echo; echo "multi-class gold sweep complete"
