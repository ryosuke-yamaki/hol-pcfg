#!/bin/bash
# Optuna Phase 2: Seed verification for top 5 configurations
# Runs 4 seeds per config on 4 GPUs in parallel, then waits for completion.
#
# Usage: bash scripts/run_optuna_phase2.sh

set -euo pipefail

SEEDS=(1 2 3 4)
CONFIGS=(
    "config/simplepcfg/optuna_phase2_rank1.yaml"
    "config/simplepcfg/optuna_phase2_rank2.yaml"
    "config/simplepcfg/optuna_phase2_rank3.yaml"
    "config/simplepcfg/optuna_phase2_rank4.yaml"
    "config/simplepcfg/optuna_phase2_rank5.yaml"
)
RANK_NAMES=("rank1" "rank2" "rank3" "rank4" "rank5")
LOG_DIR="logs/optuna_phase2"

mkdir -p "$LOG_DIR"

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    rank="${RANK_NAMES[$i]}"
    echo "=== Starting ${rank}: ${config} ==="

    PIDS=()
    for j in "${!SEEDS[@]}"; do
        seed="${SEEDS[$j]}"
        device="$j"
        log_file="${LOG_DIR}/${rank}_seed${seed}_gpu${device}.log"

        echo "  GPU ${device}: seed=${seed} -> ${log_file}"
        nohup python train.py \
            --conf "$config" \
            --device "$device" \
            --seed "$seed" \
            > "$log_file" 2>&1 &
        PIDS+=($!)
    done

    echo "  Waiting for ${rank} (PIDs: ${PIDS[*]})..."
    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "  WARNING: PID $pid exited with error"
            FAILED=$((FAILED + 1))
        fi
    done

    if [ "$FAILED" -gt 0 ]; then
        echo "  ${rank}: ${FAILED}/4 runs failed!"
    else
        echo "  ${rank}: all 4 seeds completed successfully."
    fi
    echo ""
done

echo "=== Phase 2 complete ==="
