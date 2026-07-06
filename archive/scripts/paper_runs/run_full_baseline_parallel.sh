#!/bin/bash
# Run Full HN-PCFG (eji18kkl-tuned) x 5 seeds on a single GPU with 2 lanes.
# Usage:
#   bash scripts/run_full_baseline_parallel.sh
#   GPU=1 bash scripts/run_full_baseline_parallel.sh
set -uo pipefail

GPU="${GPU:-0}"
CONFIG="config/simplepcfg/hn_pcfg_p3_eji18kkl_en_abl_full.yaml"
TAG="full"
SEEDS_A=(0 2 4)
SEEDS_B=(1 3)

LOG_DIR="logs/tab_ablation"
RESULT_DIR="results/tab_ablation"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}/${TAG}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local lane=$1 seed=$2
    local result_path="${RESULT_DIR}/${TAG}/seed${seed}.json"
    local log_file="${LOG_DIR}/${TAG}_seed${seed}.log"
    local run_name="hnpcfg-ablation-ll-en-${TAG}-seed${seed}"

    if [ -f "${result_path}" ]; then
        log "[lane ${lane}] SKIP ${TAG} seed=${seed} (result exists)"
        return 0
    fi
    log "[lane ${lane}] START ${TAG} seed=${seed}"
    local start
    start=$(date +%s)

    CUDA_VISIBLE_DEVICES="${GPU}" \
        python scripts/run_single_train.py \
            --config "${CONFIG}" \
            --seed "${seed}" \
            --result-path "${result_path}" \
            --wandb-name "${run_name}" \
            --wandb-tags "tab-ablation" "abl:${TAG}" "seed:${seed}" \
            > "${log_file}" 2>&1
    local rc=$?
    local elapsed=$(( $(date +%s) - start ))
    local hours
    hours=$(awk -v s="${elapsed}" 'BEGIN{printf "%.2f", s/3600}')

    if [ "${rc}" -ne 0 ]; then
        log "[lane ${lane}] FAILED ${TAG} seed=${seed} rc=${rc} elapsed=${elapsed}s (${hours}h) log=${log_file}"
    else
        log "[lane ${lane}] DONE ${TAG} seed=${seed} elapsed=${elapsed}s (${hours}h)"
    fi
}

run_lane() {
    local lane=$1; shift
    for s in "$@"; do
        run_one "${lane}" "${s}"
    done
}

log "Starting Full HN-PCFG verification on GPU ${GPU} (2 lanes share GPU)"

run_lane A "${SEEDS_A[@]}" &
PID_A=$!
run_lane B "${SEEDS_B[@]}" &
PID_B=$!

wait "${PID_A}" "${PID_B}"
log "All Full baseline runs finished"
