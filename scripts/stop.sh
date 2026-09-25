#!/usr/bin/env bash
# stop.sh -- stop a job launched by launch.sh, cleanly and completely.
#
# Why this exists: killing the training process alone orphans its DataLoader
# workers. Under Python 3.14's forkserver start method those children are
# reparented to init and keep running, holding ~600 MB RSS each -- eight workers
# is ~5 GB leaked per killed run, and we expect many restarts.
#
# launch.sh runs each job under `setsid`, so the job is its own process-group
# leader and `kill -- -PGID` reaches every descendant.
#
# Usage:
#   scripts/stop.sh <job-name>     # stop one job (as named to launch.sh)
#   scripts/stop.sh --all          # stop every launched job
#   scripts/stop.sh --orphans      # reap leaked workers from earlier kills
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

stop_pidfile () {
    local pidfile="$1"
    local job pid pgid
    job="$(basename "$pidfile" .pid)"
    pid="$(cat "$pidfile" 2>/dev/null)"
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        echo "  $job: not running"
        rm -f "$pidfile"
        return
    fi
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
    if [ -n "$pgid" ]; then
        kill -TERM -- "-$pgid" 2>/dev/null
        for _ in $(seq 20); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.5
        done
        kill -KILL -- "-$pgid" 2>/dev/null
        echo "  $job: stopped (process group $pgid)"
    else
        kill -KILL "$pid" 2>/dev/null
        echo "  $job: stopped (pid $pid; no pgid found)"
    fi
    rm -f "$pidfile"
}

reap_orphans () {
    # Reparented (ppid=1) Python processes belonging to this repo. A live job's
    # children have a real parent, so they are never matched.
    local found=0
    while read -r pid; do
        [ -z "$pid" ] && continue
        found=1
        echo "  reaping orphan $pid"
        kill -KILL "$pid" 2>/dev/null
    done < <(ps -eo pid,ppid,args | awk -v repo="$REPO_ROOT" \
        '$2==1 && index($0, repo) && index($0,"python") && !index($0,"awk") {print $1}')
    [ "$found" -eq 0 ] && echo "  no orphans found"
}

shopt -s nullglob
case "${1:-}" in
    --all)
        echo "stopping all jobs:"
        for f in logs/*.pid; do stop_pidfile "$f"; done
        echo "reaping orphans:"; reap_orphans
        ;;
    --orphans)
        echo "reaping orphans:"; reap_orphans
        ;;
    "")
        sed -n '2,18p' "$0" >&2; exit 2
        ;;
    *)
        matched=0
        for f in logs/"$1".*.pid logs/"$1".pid; do stop_pidfile "$f"; matched=1; done
        [ "$matched" -eq 0 ] && { echo "no pid file for job '$1'"; exit 1; }
        ;;
esac
