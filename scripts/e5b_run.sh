#!/usr/bin/env bash
# e5b_run.sh -- E5b (EXPLORATORY ViT follow-up): stage-1 grid on cifar10
# class 3, then each scope's best config on all 5 classes at full evaluation.
# Launch: scripts/launch.sh <gpu> e5b bash scripts/e5b_run.sh [pid-to-wait-for]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/queue_lib.sh"
[ $# -ge 1 ] && wait_pid "$1"

R=results/e5b_vit_explore.csv
E5=(experiments/e5_vit.py --datasets cifar10 --token-pool cls --results "$R" --device cuda:0)
K=(--iters 1 3 5 10 20)

step e5b-s1-head_dla      "$PY" "${E5[@]}" --classes 3 --scopes head_dla \
    --percentiles 50 75 85 90 --gammas 0.3 0.1 0.01 "${K[@]}"
step e5b-s1-ffn_dla_last4 "$PY" "${E5[@]}" --classes 3 --scopes ffn_dla_last4 \
    --percentiles 75 85 90 95 --gammas 0.3 0.1 "${K[@]}"
step e5b-s1-ffn_dla_last2 "$PY" "${E5[@]}" --classes 3 --scopes ffn_dla_last2 \
    --percentiles 75 85 90 95 --gammas 0.3 0.1 "${K[@]}"
step e5b-s1-ffn_pos_last4 "$PY" "${E5[@]}" --classes 3 --scopes ffn_pos_last4 \
    --percentiles 90 95 99 99.5 --gammas 0.3 0.1 "${K[@]}"

# Stage 2: selected on class 3, reported on all 5 classes, full evaluation.
"$PY" scripts/e5b_select.py > "$MARK_DIR/e5b_selected.txt" || { echo "selection failed"; FAILED+=(e5b-select); }
while read -r scope p g k; do
    [ -n "$scope" ] || continue
    step "e5b-s2-$scope" "$PY" "${E5[@]}" --scopes "$scope" --percentiles "$p" --gammas "$g" \
        --iters "$k" --profile-n 0 --eval-retain-n 0 --eval-dtype fp32
done < "$MARK_DIR/e5b_selected.txt"

finish_queue
