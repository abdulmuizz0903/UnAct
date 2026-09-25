#!/usr/bin/env bash
# queue_lib.sh -- helpers sourced by scripts/run_all.sh and scripts/e5b_run.sh.
#
# A queue is a list of `step <name> <command...>` lines run in order on one GPU
# (inside launch.sh, so CUDA_VISIBLE_DEVICES is already set and cuda:0 is right).
# Every runner below is itself idempotent; the .done markers only make a
# relaunched queue skip finished steps quickly. A failed step is logged and the
# queue moves on -- one bad step must not idle the card for the rest of the run.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${UNLEARN_PYTHON:-python}"
: "${UNLEARN_MODELS_ROOT:=$REPO_ROOT/models}"
MARK_DIR="${QUEUE_MARK_DIR:-logs/queue}"  # overridable for dry runs
mkdir -p "$MARK_DIR"
FAILED=()

step() {
    local name="$1"; shift
    if [ -f "$MARK_DIR/$name.done" ]; then
        echo "--- [$name] already done, skipping"
        return 0
    fi
    echo
    echo "################ [$name] start $(date -Is) ################"
    if "$@"; then
        touch "$MARK_DIR/$name.done"
        echo "################ [$name] done  $(date -Is) ################"
    else
        echo "!!!!!!!!!!!!!!!! [$name] FAILED (exit $?) $(date -Is) -- continuing"
        FAILED+=("$name")
    fi
}

wait_pid() {
    local pid="$1"
    if kill -0 "$pid" 2>/dev/null; then
        echo "waiting for pid $pid to exit ($(date -Is))"
        while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    fi
    echo "pid $pid gone ($(date -Is))"
}

VIT_DIR="$UNLEARN_MODELS_ROOT/vit/cifar10"
VIT_FILES=("$VIT_DIR/vit_b16.pt")
for c in 3 0 2 5 8; do VIT_FILES+=("$VIT_DIR/vit_b16_retain_c$c.pt"); done

vit_missing() {
    local f n=0
    for f in "${VIT_FILES[@]}"; do [ -f "$f" ] || n=$((n+1)); done
    echo "$n"
}

# Block until the ViT baseline and all 5 gold models exist. Gives up (returns 1)
# if they are still missing once every producer pid has exited, so a failed
# gold run cannot hang the queue forever -- and the E5 grids never run without
# the gold models they score against.
wait_vit_ready() {
    local producers=("$@") alive p
    while [ "$(vit_missing)" -gt 0 ]; do
        alive=0
        for p in "${producers[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
        if [ "$alive" -eq 0 ]; then
            # One grace pass: a checkpoint may land just as its job exits.
            sleep 30
            [ "$(vit_missing)" -eq 0 ] && break
            echo "!!! ViT producers exited with $(vit_missing) checkpoint(s) missing; skipping E5"
            return 1
        fi
        sleep 60
    done
    echo "all ViT checkpoints present ($(date -Is))"
}

lfssd_main() {
    # Run LFSSD at its selected operating point, once its grid is complete.
    local ds="$1" spec
    spec="$("$PY" scripts/select_lfssd.py "$ds")" || return 1
    echo "LFSSD selected: $spec"
    "$PY" experiments/e9_baselines.py --methods lfssd --datasets "$ds" \
        --lfssd "$spec" --device cuda:0
}

finish_queue() {
    echo
    if [ ${#FAILED[@]} -eq 0 ]; then
        echo "queue complete, no failures ($(date -Is))"
    else
        echo "queue complete with failures: ${FAILED[*]} ($(date -Is))"
        exit 1
    fi
}
