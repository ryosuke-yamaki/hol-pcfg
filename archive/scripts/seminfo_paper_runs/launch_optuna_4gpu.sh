#!/bin/bash
# Launch Optuna v3 Phase 1 on 4 GPUs x 4 workers/GPU = 16 parallel workers.
#
# Each worker:
#   - TPE seed = 42 + device_id*4 + worker_subid (prevents intra-GPU TPE collision)
#   - n_trials = 13 per worker => ~208 total, early termination at PHASE1_N_TRIALS=200
#
# Usage:
#   bash scripts/launch_optuna_4gpu.sh [STUDY_NAME]

set -euo pipefail

STUDY_NAME="${1:-hnpcfg-rank1-seminfo-v3}"
JOURNAL="optuna_${STUDY_NAME}_journal.log"
WORKERS_PER_GPU=4
N_TRIALS=13           # 200 / 16 ~= 12.5, round up
STARTUP_DELAY=5       # seconds between consecutive worker launches
OMP_THREADS=6         # 88 cores / 16 workers ~= 5.5

echo "============================================"
echo "Optuna v3 Phase 1: post-refactor HN-PCFG + Sem-Info"
echo "Study: $STUDY_NAME"
echo "Workers per GPU: $WORKERS_PER_GPU (x 4 GPUs = $((WORKERS_PER_GPU * 4)) total)"
echo "Trials per worker: $N_TRIALS"
echo "OMP_NUM_THREADS: $OMP_THREADS"
echo "Journal: $JOURNAL"
echo "============================================"
echo ""

if [ ! -f "$JOURNAL" ]; then
    echo "Starting fresh study."
else
    echo "Resuming existing study from $JOURNAL"
fi

LOGDIR="logs/optuna_${STUDY_NAME}"
mkdir -p "$LOGDIR"

# All 16 workers run Phase 1 with --skip-phase2 --skip-phase3 (Phase 2/3 are
# launched separately by launch_phase2_4gpu.sh and launch_phase3_4gpu.sh after
# all Phase 1 workers finish).
for gpu in 0 1 2 3; do
    for w in $(seq 0 $((WORKERS_PER_GPU - 1))); do
        worker_id="${gpu}_${w}"
        echo "Launching GPU $gpu worker $w (subid=$w)..."
        OMP_NUM_THREADS=$OMP_THREADS MKL_NUM_THREADS=$OMP_THREADS \
        CUDA_VISIBLE_DEVICES=$gpu python scripts/run_optuna_seminfo.py \
            --device $gpu \
            --worker-subid $w \
            --n-trials $N_TRIALS \
            --study-name "$STUDY_NAME" \
            --journal-path "$JOURNAL" \
            --startup-delay $STARTUP_DELAY \
            --skip-phase2 --skip-phase3 \
            > "$LOGDIR/phase1_gpu${worker_id}.log" 2>&1 &
        echo "  PID: $!"
        sleep "$STARTUP_DELAY"
    done
done

echo ""
echo "All $((WORKERS_PER_GPU * 4)) workers launched. Logs in $LOGDIR/"
echo "Monitor: tail -f $LOGDIR/phase1_gpu0_0.log"

wait

echo ""
echo "Phase 1 complete. Kick off Phase 2 / Phase 3 / final aggregation manually:"
echo "  bash scripts/launch_phase2_4gpu.sh $STUDY_NAME"
echo "  python scripts/aggregate_phase2.py $STUDY_NAME --phase 2"
echo "  bash scripts/launch_phase3_4gpu.sh $STUDY_NAME"
echo "  python scripts/aggregate_phase2.py $STUDY_NAME --phase 3"
