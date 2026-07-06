#!/bin/bash
# Paper-ready multilingual HN-PCFG runs with clean v2-hp hyperparameters.
#
# Schedule: 4 languages × 10 seeds (1-10) = 40 runs.
#   - 1 GPU per language: en->0, fr->1, de->2, zh->3
#   - 2 concurrent runs per GPU (8 workers in parallel total)
#   - Slot A handles odd seeds (1,3,5,7,9); slot B handles even (2,4,6,8,10)
#
# Usage:
#   bash scripts/run_paper_v2hp.sh
set -uo pipefail

SEEDS_A=(1 3 5 7 9)
SEEDS_B=(2 4 6 8 10)

LANGS=(en fr de zh)

declare -A CONFIG=(
    [en]="config/simplepcfg/hn_pcfg_nt4096_v2hp_en.yaml"
    [fr]="config/simplepcfg/hn_pcfg_nt4096_v2hp_fr.yaml"
    [de]="config/simplepcfg/hn_pcfg_nt4096_v2hp_de.yaml"
    [zh]="config/simplepcfg/hn_pcfg_nt4096_v2hp_zh.yaml"
)

declare -A GPU=(
    [en]=0
    [fr]=1
    [de]=2
    [zh]=3
)

LOG_DIR="logs/paper_v2hp"
RESULT_DIR="results/paper_v2hp"
mkdir -p "${LOG_DIR}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# A worker processes its assigned seeds sequentially on a pinned GPU.
run_worker() {
    local lang=$1
    local slot=$2
    shift 2
    local seeds=("$@")
    local gpu="${GPU[$lang]}"
    local config="${CONFIG[$lang]}"
    local result_dir="${RESULT_DIR}/${lang}"
    mkdir -p "${result_dir}"

    for seed in "${seeds[@]}"; do
        local log_file="${LOG_DIR}/${lang}_slot${slot}_seed${seed}.log"
        local result_path="${result_dir}/seed${seed}.json"
        local run_name="hnpcfg-v2hp-${lang}-seed${seed}"

        if [ -f "${result_path}" ]; then
            log "[${lang} slot${slot} gpu${gpu}] SKIP seed=${seed} (result exists)"
            continue
        fi

        log "[${lang} slot${slot} gpu${gpu}] START seed=${seed}"
        local start=$(date +%s)

        CUDA_VISIBLE_DEVICES="${gpu}" \
            python scripts/run_single_train.py \
                --config "${config}" \
                --seed "${seed}" \
                --result-path "${result_path}" \
                --wandb-name "${run_name}" \
                --wandb-tags "paper-v2hp" "lang:${lang}" "seed:${seed}" \
                > "${log_file}" 2>&1
        local rc=$?

        local elapsed=$(( $(date +%s) - start ))
        local hours
        hours=$(awk -v s="${elapsed}" 'BEGIN{printf "%.2f", s/3600}')

        if [ "${rc}" -ne 0 ]; then
            log "[${lang} slot${slot} gpu${gpu}] FAILED seed=${seed} rc=${rc} elapsed=${elapsed}s (${hours}h)"
        else
            log "[${lang} slot${slot} gpu${gpu}] DONE   seed=${seed} elapsed=${elapsed}s (${hours}h)"
        fi
    done
}

log "Launching 8 workers (4 langs x 2 slots) for 40 total runs"

declare -A PID2TAG=()
pids=()
for lang in "${LANGS[@]}"; do
    run_worker "${lang}" A "${SEEDS_A[@]}" &
    pid=$!
    pids+=("${pid}")
    PID2TAG[${pid}]="${lang}/A"

    run_worker "${lang}" B "${SEEDS_B[@]}" &
    pid=$!
    pids+=("${pid}")
    PID2TAG[${pid}]="${lang}/B"
done

fail=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        log "Worker ${PID2TAG[${pid}]} (pid=${pid}) exited with non-zero status"
        fail=$(( fail + 1 ))
    fi
done

log "All workers finished. Failed workers: ${fail}/${#pids[@]}"
exit "${fail}"
