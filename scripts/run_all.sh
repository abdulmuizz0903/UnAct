#!/usr/bin/env bash
# run_all.sh -- the complete experiment programme, in dependency order.
#
# Every command below is the one that produced the corresponding results/*.csv.
# Each step is idempotent (finished work is skipped), and a completed step leaves
# a marker in logs/queue/, so re-running this script after an interruption
# resumes where it stopped. A failed step is logged and the script moves on.
#
# Launch detached on one GPU (timings in the paper come from a single NVIDIA L4
# with no other job on the card):
#   scripts/launch.sh 0 run-all bash scripts/run_all.sh
#   tail -f logs/run-all.gpu0.log
#
# Roughly: retrained models ~9 min each on an A10 (27 ResNet-18 + 5 ViT-B/16),
# then ~2-3 GPU-days of unlearning sweeps. To regenerate only the paper's tables
# and figures from the shipped results/*.csv, run `python paper/make_all.py`.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/queue_lib.sh"
export UNLEARN_MODELS_ROOT
D=(--device cuda:0)

# ---------------------------------------------------------------- 0. data + models
step data     "$PY" "Baseline CIFAR Training/prepare_data.py"
step baselines bash -c "cd 'Baseline CIFAR Training' && '$PY' train_all.py --gpus 0"
step gold-cifar10  bash scripts/gold_sweep.sh cifar10 0 2 3 5 8
step gold-cifar20  bash scripts/gold_sweep.sh cifar20 3 4 10 14 19
step gold-cifar100 bash scripts/gold_sweep.sh cifar100 3 20 51 69 85
step gold-multi    bash scripts/gold_multi.sh
step e11-gold-cifar100 bash scripts/gold_sweep.sh cifar100 2 40
step e11-gold-cifar20  bash scripts/gold_sweep.sh cifar20 5
step vit-base "$PY" "Baseline ViT Training/train_vit_cifar.py" --dataset cifar10 "${D[@]}"
step vit-gold bash scripts/vit_gold_sweep.sh cifar10 3 0 2 5 8

# ---------------------------------------------------------------- 1. tuning grids
# E1 SSD, E6 UnAct, E9a LFSSD: identical classes, selection rule and gold models.
step e1-ssd   "$PY" experiments/e1_ssd_tune.py --datasets cifar10 cifar20 cifar100 "${D[@]}" --workers 8
step e6-unact "$PY" experiments/e6_unact_grid.py --datasets cifar10 cifar20 cifar100 "${D[@]}" --workers 8
for ds in cifar10 cifar20 cifar100; do
    step "lfssd-tune-$ds" "$PY" experiments/e9_lfssd_tune.py --datasets "$ds" --workers 6 "${D[@]}"
done
"$PY" experiments/analyse_e1.py || true
"$PY" experiments/analyse_e6.py || true

# ---------------------------------------------------------------- 2. main table (E2, E9)
E2=results/e2_main_table_l4.csv
step e2-shared "$PY" experiments/e2_main_table.py --methods baseline gold ssd \
    --results "$E2" "${D[@]}" --workers 8
step e2-unact-cifar10  "$PY" experiments/e2_main_table.py --datasets cifar10 --methods unact \
    --unact-p-cifar10 99 --unact-gamma 0.01 --unact-iters 20 --results "$E2" "${D[@]}" --workers 8
step e2-unact-cifar20  "$PY" experiments/e2_main_table.py --datasets cifar20 --methods unact \
    --unact-p-cifar20 99 --unact-gamma 0.1 --unact-iters 20 --results "$E2" "${D[@]}" --workers 8
step e2-unact-cifar100 "$PY" experiments/e2_main_table.py --datasets cifar100 --methods unact \
    --unact-p-cifar100 90 --unact-gamma 0.3 --unact-iters 1 --results "$E2" "${D[@]}" --workers 8
for ds in cifar10 cifar20 cifar100; do
    step "lfssd-main-$ds" lfssd_main "$ds"      # scripts/queue_lib.sh
done

# ---------------------------------------------------------------- 3. forget-set size (E3)
step e3-size  "$PY" experiments/e3_forget_size.py --datasets cifar10 cifar20 cifar100 "${D[@]}" --workers 8
step e3-lfssd "$PY" experiments/e3_lfssd_forget_size.py "${D[@]}"
step e3-oracle "$PY" experiments/analyse_e3_oracle.py

# ---------------------------------------------------------------- 4. multi-class, sequential (E4, E4b, E15)
step e4-multiclass "$PY" experiments/e4_multiclass.py --mode both --tune "${D[@]}" --workers 8
seq_orders() {  # <dataset> <order>...   the four random class orders per dataset
    local ds="$1" o; shift
    for o in "$@"; do
        # shellcheck disable=SC2086  # $o is a space-separated class list
        "$PY" experiments/e4_multiclass.py --datasets "$ds" --mode sequential \
            --methods unact ssd --order $o --results results/e4b_seq_orders.csv \
            "${D[@]}" --workers 8 || return 1
    done
}
step e4b-cifar10  seq_orders cifar10  "5 6 1 2 0" "8 7 1 5 6" "6 0 3 7 8" "0 4 9 6 7"
step e4b-cifar20  seq_orders cifar20  "5 13 2 11 19" "8 1 19 0 10" "6 12 3 7 1" "10 14 15 2 5"
step e4b-cifar100 seq_orders cifar100 "45 15 90 32 35" "48 97 1 81 90" "86 42 89 92 3" "30 13 49 91 37"
step e15-seq "$PY" experiments/e15_seq_gentle.py

# ---------------------------------------------------------------- 5. mechanism and evidence (E7, E8, E9, E12, E13)
step e7-relearn     "$PY" experiments/e7_relearn.py --datasets cifar10 cifar20 cifar100 "${D[@]}" --workers 8
step e8-selectivity "$PY" experiments/e8_selectivity.py "${D[@]}" --workers 8
step labelfree       "$PY" experiments/e9_labelfree_check.py "${D[@]}" --workers 4
step labelfree-unact "$PY" experiments/e9_labelfree_check.py --methods unact "${D[@]}"
for ds in cifar10 cifar20 cifar100; do
    step "e12-$ds" "$PY" experiments/e12_probe_ablation.py --datasets "$ds" "${D[@]}" \
        --results "results/e12_probe_ablation_$ds.csv"
done
step e13-selection "$PY" experiments/analyse_e13_selection.py

# ---------------------------------------------------------------- 6. cost (timed alone on the card)
C=experiments/measure_cost_breakdown.py
step cost-cifar10  "$PY" $C --datasets cifar10  --unact-p 99 --unact-gamma 0.01 --unact-iters 20 --results results/cost_breakdown_l4.csv "${D[@]}"
step cost-cifar20  "$PY" $C --datasets cifar20  --unact-p 99 --unact-gamma 0.1  --unact-iters 20 --results results/cost_breakdown_l4.csv "${D[@]}"
step cost-cifar100 "$PY" $C --datasets cifar100 --unact-p 90 --unact-gamma 0.3  --unact-iters 1  --results results/cost_breakdown_l4.csv "${D[@]}"
step cost-k1-cifar10  "$PY" $C --datasets cifar10  --unact-p 75 --unact-gamma 0.01 --unact-iters 1 --results results/cost_breakdown_l4_k1.csv "${D[@]}"
step cost-k1-cifar20  "$PY" $C --datasets cifar20  --unact-p 75 --unact-gamma 0.1  --unact-iters 1 --results results/cost_breakdown_l4_k1.csv "${D[@]}"
step cost-k5-cifar10  "$PY" $C --datasets cifar10  --unact-p 90 --unact-gamma 0.03 --unact-iters 5 --results results/cost_breakdown_l4_k5.csv "${D[@]}"
step cost-k5-cifar20  "$PY" $C --datasets cifar20  --unact-p 95 --unact-gamma 0.1  --unact-iters 5 --results results/cost_breakdown_l4_k5.csv "${D[@]}"
step cost-k5-cifar100 "$PY" $C --datasets cifar100 --unact-p 99 --unact-gamma 0.03 --unact-iters 5 --results results/cost_breakdown_l4_k5.csv "${D[@]}"

# ---------------------------------------------------------------- 7. classes SSD's paper reports as hard (E11)
H=results/e11_hard_classes
step e11-ssd-cifar100   "$PY" experiments/e1_ssd_tune.py   --datasets cifar100 --classes 2 40 --results "${H}_ssd.csv"   "${D[@]}" --workers 8
step e11-unact-cifar100 "$PY" experiments/e6_unact_grid.py --datasets cifar100 --classes 2 40 --results "${H}_unact.csv" "${D[@]}" --workers 8
step e11-lfssd-cifar100 "$PY" experiments/e9_lfssd_tune.py --datasets cifar100 --classes 2 40 --results "${H}_lfssd.csv" "${D[@]}" --workers 6
step e11-ssd-cifar20    "$PY" experiments/e1_ssd_tune.py   --datasets cifar20 --classes 5 --results "${H}_ssd.csv"   "${D[@]}" --workers 8
step e11-unact-cifar20  "$PY" experiments/e6_unact_grid.py --datasets cifar20 --classes 5 --results "${H}_unact.csv" "${D[@]}" --workers 8
step e11-lfssd-cifar20  "$PY" experiments/e9_lfssd_tune.py --datasets cifar20 --classes 5 --results "${H}_lfssd.csv" "${D[@]}" --workers 6

# ---------------------------------------------------------------- 8. ViT-B/16 (E5, E5b, E5c, E5d)
# SSD grid, then UnAct's scope search (stage 1 on class 3, stage 2 on all five
# classes at full evaluation), then SSD's matching search at full evaluation.
step e5-ssd "$PY" experiments/e5_vit_ssd.py --datasets cifar10 "${D[@]}"
step e5b-unact bash scripts/e5b_run.sh
step e5b-ssd-search "$PY" experiments/e5_vit_ssd.py --datasets cifar10 \
    --alphas 6 7 8 9 --lambdas 0.1 0.5 1.0 --results results/e5b_vit_ssd_explore.csv "${D[@]}"
SSD_CFG="$("$PY" scripts/e5_select_ssd.py)" || SSD_CFG=""
UN_CFG="$(awk '$1=="ffn_dla_last4"{print $1","$2","$3","$4}' "$MARK_DIR/e5b_selected.txt" 2>/dev/null)"
if [ -n "$SSD_CFG" ] && [ -n "$UN_CFG" ]; then
    IFS=, read -r A L B <<< "$SSD_CFG"
    step e5b-ssd-full "$PY" experiments/e5_vit_ssd.py --datasets cifar10 --alphas "$A" \
        --lambdas "$L" --batch-sizes "$B" --eval-retain-n 0 --eval-dtype fp32 \
        --results results/e5b_vit_ssd_explore.csv "${D[@]}"
    # The paper reports lr 5e-4; the lr 0.01 rows in the CSV are the first probe.
    step e5c-relearn       "$PY" experiments/e5c_vit_relearn.py --ssd "$SSD_CFG" --unact "$UN_CFG" "${D[@]}"
    step e5c-relearn-lr5e4 "$PY" experiments/e5c_vit_relearn.py --ssd "$SSD_CFG" --unact "$UN_CFG" --lr 5e-4 "${D[@]}"
else
    echo "!!! ViT selection unavailable; skipping e5b-ssd-full and e5c"
    FAILED+=(e5b-ssd-full e5c-relearn)
fi
step e5d-vit-forget-size "$PY" experiments/e5d_vit_forget_size.py "${D[@]}"

# ---------------------------------------------------------------- 9. tables and figures
step paper "$PY" paper/make_all.py

finish_queue
