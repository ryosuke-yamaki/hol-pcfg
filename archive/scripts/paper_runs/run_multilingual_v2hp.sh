#!/bin/bash
# Multilingual HNPCFG replication using the v2-hp (eji18kkl) best hyperparameters.
# Sequential execution (single GPU, 1 run at a time) with interleaved schedule:
#   round r in 1..5 : zh seed=r -> fr seed=r -> de seed=r
# This yields 15 runs total, ordered so that early rounds produce results
# across all three languages before later seeds.
#
# Usage:
#   bash scripts/run_multilingual_v2hp.sh [DEVICE] [SEEDS_CSV]
# Defaults: DEVICE=0, SEEDS_CSV=1,2,3,4,5
set -euo pipefail

DEVICE=${1:-0}
SEEDS_CSV="${2:-1,2,3,4,5}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"

LANGS=(zh fr de)
declare -A CONFIG=(
    [zh]="config/simplepcfg/hn_pcfg_nt4096_v2hp_zh.yaml"
    [fr]="config/simplepcfg/hn_pcfg_nt4096_v2hp_fr.yaml"
    [de]="config/simplepcfg/hn_pcfg_nt4096_v2hp_de.yaml"
)

mkdir -p logs/multilingual_v2hp results/multilingual_v2hp

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

total_fails=0
run_idx=0
total=$(( ${#SEEDS[@]} * ${#LANGS[@]} ))

for seed in "${SEEDS[@]}"; do
    log "============================================================"
    log "Round seed=${seed}"
    log "============================================================"

    for lang in "${LANGS[@]}"; do
        run_idx=$(( run_idx + 1 ))
        config="${CONFIG[$lang]}"
        config_stem=$(basename "${config}" .yaml)
        log_file="logs/multilingual_v2hp/${config_stem}_seed${seed}.log"
        result_dir="results/multilingual_v2hp/${lang}"
        result_path="${result_dir}/seed${seed}.json"
        run_name="hnpcfg-v2hp-${lang}-seed${seed}"
        mkdir -p "${result_dir}"

        log "[${run_idx}/${total}] START lang=${lang} seed=${seed} -> ${log_file}"
        start=$(date +%s)

        set +e
        CUDA_VISIBLE_DEVICES="${DEVICE}" \
            python scripts/run_single_train.py \
                --config "${config}" \
                --seed "${seed}" \
                --result-path "${result_path}" \
                --wandb-name "${run_name}" \
                --wandb-tags "multilingual-v2hp" "lang:${lang}" "seed:${seed}" \
                > "${log_file}" 2>&1
        rc=$?
        set -e

        elapsed=$(( $(date +%s) - start ))
        hours=$(awk -v s="${elapsed}" 'BEGIN{printf "%.2f", s/3600}')
        if [ "${rc}" -ne 0 ]; then
            log "[${run_idx}/${total}] FAILED lang=${lang} seed=${seed} rc=${rc} elapsed=${elapsed}s (${hours}h)"
            total_fails=$(( total_fails + 1 ))
        else
            log "[${run_idx}/${total}] DONE   lang=${lang} seed=${seed} elapsed=${elapsed}s (${hours}h)"
        fi
    done
done

log "All 15 runs complete. failures=${total_fails}/${total}"
exit "${total_fails}"
