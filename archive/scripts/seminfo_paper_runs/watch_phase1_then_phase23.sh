#!/bin/bash
# Wait for externally-launched Phase 1 workers to finish, then chain
# Phase 2 -> aggregate(2) -> Phase 3 -> aggregate(3).
#
# Phase 1 workers are identified by the command-line signature
# `run_optuna_seminfo.py ... --skip-phase2 --skip-phase3` (set by
# launch_optuna_4gpu.sh). Phase 2/3 workers use --phase2-only /
# --phase3-only, so the pgrep pattern never matches them.
#
# Usage (recommended: background with nohup):
#   mkdir -p logs/pipeline_v3
#   nohup bash scripts/watch_phase1_then_phase23.sh hnpcfg-rank1-seminfo-v3 \
#     > logs/pipeline_v3/watcher.log 2>&1 &
#   echo $! > logs/pipeline_v3/watcher.pid
#   tail -f logs/pipeline_v3/watcher.log

set -euo pipefail

STUDY_NAME="${1:-hnpcfg-rank1-seminfo-v3}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"  # seconds between Phase 1 liveness checks
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

timestamp () { date '+%Y-%m-%d %H:%M:%S'; }
log () { printf '[%s] %s\n' "$(timestamp)" "$*"; }

# Count live Phase 1 workers: comm must be "python" (avoids self-match from
# shells whose argv contains the pattern) AND cmdline must include the unique
# --skip-phase2 / --skip-phase3 signature set only by launch_optuna_4gpu.sh.
count_phase1 () {
    ps -eo comm=,args= \
        | awk '$1=="python" && /run_optuna_seminfo\.py/ && /--skip-phase2/ && /--skip-phase3/ {n++} END {print n+0}'
}

log "Watcher start. study=$STUDY_NAME pid=$$ poll_interval=${POLL_INTERVAL}s"
log "Repo: $REPO_ROOT"

initial_count=$(count_phase1)
log "Phase 1 workers detected at start: $initial_count"
if [ "$initial_count" -eq 0 ]; then
    log "WARNING: no Phase 1 workers match. Proceeding to Phase 2 immediately."
fi

while :; do
    count=$(count_phase1)
    if [ "$count" -eq 0 ]; then
        log "Phase 1 workers: 0 -> Phase 1 complete."
        break
    fi
    log "Phase 1 workers still alive: $count. Sleeping ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
done

# Grace period so the final worker's journal writes settle.
sleep 10

# --- Phase 2 ---
log "=== Phase 2: Top-5 x 5 seeds ==="
bash scripts/launch_phase2_4gpu.sh "$STUDY_NAME"
log "Phase 2 launcher returned (all workers done)."

log "=== Phase 2 aggregation ==="
python scripts/aggregate_phase2.py "$STUDY_NAME" --phase 2
log "Phase 2 aggregate DONE."

# --- Phase 3 ---
log "=== Phase 3: Best x 5 new seeds (6-10) ==="
bash scripts/launch_phase3_4gpu.sh "$STUDY_NAME"
log "Phase 3 launcher returned (all workers done)."

log "=== Phase 3 aggregation ==="
python scripts/aggregate_phase2.py "$STUDY_NAME" --phase 3
log "Phase 3 aggregate DONE."

log "ALL DONE. Results:"
log "  Phase 2: logs/optuna_${STUDY_NAME}_phase2_results.json"
log "  Phase 3: logs/optuna_${STUDY_NAME}_phase3_results.json"
