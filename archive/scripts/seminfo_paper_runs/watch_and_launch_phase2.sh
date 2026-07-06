#!/bin/bash
# Watch for Phase 1 to finish, then chain Phase 2 -> aggregate -> Phase 3 -> aggregate.
# Run in the background: nohup bash scripts/watch_and_launch_phase2.sh [STUDY_NAME] &

set -euo pipefail

STUDY_NAME="${1:-hnpcfg-rank1-seminfo-v3}"
CHECK_INTERVAL=300  # seconds

echo "[watcher] Monitoring Phase 1 completion for study: $STUDY_NAME"
echo "[watcher] Checking every ${CHECK_INTERVAL}s..."

while true; do
    n_procs=$(pgrep -fc run_optuna_seminfo 2>/dev/null || echo 0)
    if [ "$n_procs" -eq 0 ]; then
        echo "[watcher] $(date): No run_optuna_seminfo processes found. Phase 1 complete."
        break
    fi
    echo "[watcher] $(date): $n_procs processes still running. Waiting..."
    sleep "$CHECK_INTERVAL"
done

echo "[watcher] Launching Phase 2..."
bash scripts/launch_phase2_4gpu.sh "$STUDY_NAME"

echo "[watcher] Phase 2 done. Aggregating..."
python scripts/aggregate_phase2.py "$STUDY_NAME" --phase 2

echo "[watcher] Launching Phase 3..."
bash scripts/launch_phase3_4gpu.sh "$STUDY_NAME"

echo "[watcher] Phase 3 done. Aggregating..."
python scripts/aggregate_phase2.py "$STUDY_NAME" --phase 3

echo "[watcher] All done."
