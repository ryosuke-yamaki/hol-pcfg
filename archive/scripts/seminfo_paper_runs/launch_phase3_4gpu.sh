#!/bin/bash
# Phase 3 (final validation): Best config x 5 new seeds (6-10) on 5 workers.
#
# Usage:
#   bash scripts/launch_phase3_4gpu.sh [STUDY_NAME]
#
# GPU assignment: workers 0-3 map to GPUs 0-3 (one each), worker 4 reuses GPU 0.
# aggregate_phase2.py --phase 3 afterward for final report.

set -euo pipefail

STUDY_NAME="${1:-hnpcfg-rank1-seminfo-v3}"
JOURNAL="optuna_${STUDY_NAME}_journal.log"
N_WORKERS=5
OMP_THREADS=12
STARTUP_DELAY=5

echo "============================================"
echo "Phase 3: Final Validation (5 workers)"
echo "Study: $STUDY_NAME"
echo "Jobs: 5 (Best x 5 new seeds 6-10)"
echo "============================================"

LOGDIR="logs/optuna_${STUDY_NAME}"
mkdir -p "$LOGDIR"

# Round-robin by worker_id; worker 0..3 -> GPU 0..3, worker 4 -> GPU 0
gpu_assign () {
    case "$1" in
        0) echo 0 ;;
        1) echo 1 ;;
        2) echo 2 ;;
        3) echo 3 ;;
        4) echo 0 ;;
        *) echo 0 ;;
    esac
}

for w in $(seq 0 $((N_WORKERS - 1))); do
    gpu=$(gpu_assign "$w")
    echo "Launching worker $w on GPU $gpu..."
    OMP_NUM_THREADS=$OMP_THREADS MKL_NUM_THREADS=$OMP_THREADS \
    CUDA_VISIBLE_DEVICES=$gpu python scripts/run_optuna_seminfo.py \
        --device $gpu \
        --study-name "$STUDY_NAME" \
        --journal-path "$JOURNAL" \
        --phase3-only \
        --phase3-worker-id $w \
        --phase3-n-workers $N_WORKERS \
        > "$LOGDIR/phase3_w${w}.log" 2>&1 &
    echo "  PID: $!"
    sleep "$STARTUP_DELAY"
done

echo ""
echo "All $N_WORKERS workers launched. Waiting..."
wait

echo ""
echo "All Phase 3 workers done."
echo "Aggregate: python scripts/aggregate_phase2.py $STUDY_NAME --phase 3"
