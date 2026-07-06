#!/bin/bash
# Phase 0 baseline: rank1-seminfo HP x 5 seeds at 100k steps.
# GPU assignment: seeds 1-4 on GPUs 0-3 (one each), seed 5 on GPU 3 (stacked).
#
# Usage:
#   bash scripts/launch_phase0_5seeds.sh
#
# Waits for all 5 runs to finish, then exits. Logs under logs/phase0_v3/.

set -euo pipefail

OMP_THREADS=12
LOGDIR="logs/phase0_v3"
mkdir -p "$LOGDIR"

echo "============================================"
echo "Phase 0 baseline: rank1-seminfo x 5 seeds @ 100k"
echo "GPU 0-2: 1 seed each, GPU 3: 2 seeds"
echo "============================================"

launch_seed () {
    local seed=$1
    local gpu=$2
    echo "Launching seed ${seed} on GPU ${gpu}..."
    OMP_NUM_THREADS=$OMP_THREADS MKL_NUM_THREADS=$OMP_THREADS \
    CUDA_VISIBLE_DEVICES=$gpu python scripts/run_phase0_baseline.py \
        --seed "$seed" --device "$gpu" \
        > "$LOGDIR/seed${seed}.log" 2>&1 &
    echo "  PID: $!"
}

launch_seed 1 0
launch_seed 2 1
launch_seed 3 2
launch_seed 4 3
sleep 5
launch_seed 5 3

echo ""
echo "All 5 seeds launched. Logs in $LOGDIR/"
echo "Monitor: tail -f $LOGDIR/seed1.log"

wait
echo "Phase 0 done."
