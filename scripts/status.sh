#!/usr/bin/env bash
# status.sh -- what is running, what finished, how far along.
#
# Safe to run from a fresh SSH session that knows nothing about what was
# launched: everything is reconstructed from logs/, the pid files, and the
# results CSVs.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
: "${UNLEARN_MODELS_ROOT:=$REPO_ROOT/models}"

echo "=== GPUs ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader 2>/dev/null || echo "  nvidia-smi unavailable"

echo
echo "=== jobs ==="
shopt -s nullglob
found=0
for pidfile in logs/*.pid; do
    found=1
    job="$(basename "$pidfile" .pid)"
    pid="$(cat "$pidfile" 2>/dev/null)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        state="RUNNING (pid $pid)"
    else
        state="finished//dead"
    fi
    printf "  %-40s %s\n" "$job" "$state"
    log="logs/${job}.log"
    # Which sub-task, and how far into it.
    cur="$(grep -oE '^#+ [a-z0-9]+ class [0-9]+' "$log" 2>/dev/null | tail -1 | sed 's/#* //')"
    [ -n "$cur" ] && printf "      on: %s\n" "$cur"
    last="$(grep -vE 'Deprecat|pickle\.load' "$log" 2>/dev/null | tail -1)"
    [ -n "$last" ] && printf "      %s\n" "${last:0:150}"
    # Sub-tasks already finished inside this job.
    ndone="$(grep -c 'Saved gold model' "$log" 2>/dev/null)"
    [ "${ndone:-0}" -gt 0 ] && printf "      completed in this job: %s\n" "$ndone"
done
[ "$found" -eq 0 ] && echo "  (no jobs launched yet)"

echo
echo "=== result row counts ==="
for f in results/*.csv; do
    [ -f "$f" ] || continue
    n=$(( $(wc -l < "$f") - 1 ))
    printf "  %-44s %5d rows   (modified %s)\n" "$f" "$n" \
           "$(date -r "$f" '+%m-%d %H:%M' 2>/dev/null)"
done

echo
echo "=== gold models (target: 5 forget classes per dataset) ==="
if [ -f "$UNLEARN_MODELS_ROOT/manifest.csv" ]; then
    # Parsed with the csv module, not awk -F,: the hparams column is JSON and
    # contains commas inside quotes, which shifts every later field in awk.
    "${UNLEARN_PYTHON:-python}" - \
        "$UNLEARN_MODELS_ROOT/manifest.csv" <<'PYEOF'
import csv, sys
from collections import defaultdict
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["role"] == "retain_gold"]
by_ds = defaultdict(list)
for r in rows:
    by_ds[r["dataset"]].append(r)
if not rows:
    print("  none yet")
for ds in sorted(by_ds):
    got = sorted(by_ds[ds], key=lambda r: (len(r["forget_spec"].split("-")), r["forget_spec"]))
    classes = " ".join(r["forget_spec"] for r in got)
    print(f"  {ds:9s} {len(got)}/5  classes: {classes}")
if rows:
    print("  --- accuracies ---")
    for ds in sorted(by_ds):
        for r in sorted(by_ds[ds], key=lambda r: (len(r["forget_spec"].split("-")), r["forget_spec"])):
            t = float(r["train_time_s"] or 0) / 60
            print(f"  {ds:9s} forget={r['forget_spec']:<4s} "
                  f"D_r={float(r['d_r']):6.2f}  D_f={float(r['d_f']):5.2f}  ({t:.0f} min)")
PYEOF
else
    echo "  no manifest yet ($UNLEARN_MODELS_ROOT/manifest.csv)"
fi

echo
echo "=== disk ==="
df -h / "$UNLEARN_MODELS_ROOT" 2>/dev/null | awk 'NR==1 || /\//'
