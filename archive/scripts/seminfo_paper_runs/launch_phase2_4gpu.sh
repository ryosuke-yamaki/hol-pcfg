#!/bin/bash
# Phase 2 (seed verification) on 4 GPUs x 4 workers/GPU = 16 parallel workers.
# Phase 2 does not do TPE sampling, so per-worker TPE seed is not needed.
#
# Usage:
#   bash scripts/launch_phase2_4gpu.sh [STUDY_NAME]
#
# Distributes 25 runs (Top-5 x 5 seeds) round-robin across 16 workers.

set -euo pipefail

STUDY_NAME="${1:-hnpcfg-rank1-seminfo-v3}"
JOURNAL="optuna_${STUDY_NAME}_journal.log"
WORKERS_PER_GPU=4
N_WORKERS=$((WORKERS_PER_GPU * 4))
OMP_THREADS=6
STARTUP_DELAY=5

echo "============================================"
echo "Phase 2: Seed Verification (16 workers)"
echo "Study: $STUDY_NAME"
echo "Workers: $N_WORKERS ($WORKERS_PER_GPU per GPU x 4 GPUs)"
echo "Jobs: 25 (Top-5 x 5 seeds)"
echo "============================================"
echo ""

LOGDIR="logs/optuna_${STUDY_NAME}"
mkdir -p "$LOGDIR"

worker_id=0
for gpu in 0 1 2 3; do
    for w in $(seq 0 $((WORKERS_PER_GPU - 1))); do
        echo "Launching GPU $gpu worker $w (worker_id=$worker_id)..."
        OMP_NUM_THREADS=$OMP_THREADS MKL_NUM_THREADS=$OMP_THREADS \
        CUDA_VISIBLE_DEVICES=$gpu python scripts/run_optuna_seminfo.py \
            --device $gpu \
            --study-name "$STUDY_NAME" \
            --journal-path "$JOURNAL" \
            --phase2-only \
            --phase2-worker-id $worker_id \
            --phase2-n-workers $N_WORKERS \
            > "$LOGDIR/phase2_w${worker_id}.log" 2>&1 &
        echo "  PID: $!"
        worker_id=$((worker_id + 1))
        sleep "$STARTUP_DELAY"
    done
done

echo ""
echo "All $N_WORKERS workers launched. Waiting..."
wait

echo ""
echo "All Phase 2 workers done."
echo "Aggregate: python scripts/aggregate_phase2.py $STUDY_NAME --phase 2"
