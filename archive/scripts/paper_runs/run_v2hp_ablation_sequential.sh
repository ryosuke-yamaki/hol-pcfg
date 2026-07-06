#!/bin/bash
# Sequentially run two v2hp ablation configs x seeds 1-5 (10 runs total).
#   1. NT=8192, T=16384 variant
#   2. s_dim=1024 variant
#
# Usage:
#   bash scripts/run_v2hp_ablation_sequential.sh
#   GPU=1 bash scripts/run_v2hp_ablation_sequential.sh   # pin to GPU 1
set -uo pipefail

GPU="${GPU:-0}"
SEEDS=(1 2 3 4 5)

CONFIGS=(
    "config/simplepcfg/hn_pcfg_nt8192_v2hp_en.yaml"
    "config/simplepcfg/hn_pcfg_nt4096_v2hp_en_sdim1024.yaml"
)

TAGS=(
    "nt8192"
    "sdim1024"
)

LOG_DIR="logs/v2hp_ablation"
RESULT_DIR="results/v2hp_ablation"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

total=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))
idx=0
fail=0

log "Starting ${total} sequential runs on GPU ${GPU}"

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    tag="${TAGS[$i]}"
    config_stem="$(basename "${config}" .yaml)"
    result_subdir="${RESULT_DIR}/${tag}"
    mkdir -p "${result_subdir}"

    for seed in "${SEEDS[@]}"; do
        idx=$(( idx + 1 ))
        log_file="${LOG_DIR}/${tag}_seed${seed}.log"
        result_path="${result_subdir}/seed${seed}.json"
        run_name="hnpcfg-v2hp-en-${tag}-seed${seed}"

        if [ -f "${result_path}" ]; then
            log "[${idx}/${total}] SKIP ${tag} seed=${seed} (result exists)"
            continue
        fi

        log "[${idx}/${total}] START ${tag} seed=${seed} config=${config_stem}"
        start=$(date +%s)

        CUDA_VISIBLE_DEVICES="${GPU}" \
            python scripts/run_single_train.py \
                --config "${config}" \
                --seed "${seed}" \
                --result-path "${result_path}" \
                --wandb-name "${run_name}" \
                --wandb-tags "v2hp-ablation" "variant:${tag}" "seed:${seed}" \
                > "${log_file}" 2>&1
        rc=$?

        elapsed=$(( $(date +%s) - start ))
        hours=$(awk -v s="${elapsed}" 'BEGIN{printf "%.2f", s/3600}')

        if [ "${rc}" -ne 0 ]; then
            log "[${idx}/${total}] FAILED ${tag} seed=${seed} rc=${rc} elapsed=${elapsed}s (${hours}h) log=${log_file}"
            fail=$(( fail + 1 ))
        else
            log "[${idx}/${total}] DONE   ${tag} seed=${seed} elapsed=${elapsed}s (${hours}h)"
        fi
    done
done

log "All runs finished. Failed: ${fail}/${total}"
exit "${fail}"
