#!/bin/bash
# Run 4 ablation configs x 5 seeds = 20 runs on a single GPU,
# with two sequential lanes that share the GPU concurrently.
#
# Usage:
#   bash scripts/run_table_ablation_parallel.sh
#   GPU=1 bash scripts/run_table_ablation_parallel.sh   # pin to GPU 1
set -uo pipefail

GPU="${GPU:-0}"
SEEDS=(0 1 2 3 4)

LOG_DIR="logs/tab_ablation"
RESULT_DIR="results/tab_ablation"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local lane=$1 config=$2 tag=$3 seed=$4
    local result_subdir="${RESULT_DIR}/${tag}"
    mkdir -p "${result_subdir}"
    local result_path="${result_subdir}/seed${seed}.json"
    local log_file="${LOG_DIR}/${tag}_seed${seed}.log"
    local run_name="hnpcfg-ablation-ll-en-${tag}-seed${seed}"

    if [ -f "${result_path}" ]; then
        log "[lane ${lane}] SKIP ${tag} seed=${seed} (result exists)"
        return 0
    fi
    log "[lane ${lane}] START ${tag} seed=${seed}"
    local start
    start=$(date +%s)

    CUDA_VISIBLE_DEVICES="${GPU}" \
        python scripts/run_single_train.py \
            --config "${config}" \
            --seed "${seed}" \
            --result-path "${result_path}" \
            --wandb-name "${run_name}" \
            --wandb-tags "tab-ablation" "abl:${tag}" "seed:${seed}" \
            > "${log_file}" 2>&1
    local rc=$?
    local elapsed=$(( $(date +%s) - start ))
    local hours
    hours=$(awk -v s="${elapsed}" 'BEGIN{printf "%.2f", s/3600}')

    if [ "${rc}" -ne 0 ]; then
        log "[lane ${lane}] FAILED ${tag} seed=${seed} rc=${rc} elapsed=${elapsed}s (${hours}h) log=${log_file}"
    else
        log "[lane ${lane}] DONE ${tag} seed=${seed} elapsed=${elapsed}s (${hours}h)"
    fi
}

run_lane() {
    local lane=$1
    shift
    while (( "$#" >= 2 )); do
        local config="$1" tag="$2"
        shift 2
        for s in "${SEEDS[@]}"; do
            run_one "${lane}" "${config}" "${tag}" "${s}"
        done
    done
}

log "Starting ablation runs on GPU ${GPU} (2 lanes share the GPU)"

run_lane A \
    "config/simplepcfg/hn_pcfg_p3_eji18kkl_en_abl_hadamard.yaml" "hadamard" \
    "config/simplepcfg/hn_pcfg_p3_eji18kkl_en_abl_no_cnorm.yaml" "no_cnorm" &
PID_A=$!

run_lane B \
    "config/simplepcfg/hn_pcfg_p3_eji18kkl_en_abl_conv.yaml" "conv" \
    "config/simplepcfg/hn_pcfg_p3_eji18kkl_en_abl_no_tau.yaml" "no_tau" &
PID_B=$!

wait "${PID_A}" "${PID_B}"
log "All ablation runs finished"
