#!/bin/bash
# SemInfo HN-PCFG ablation runs (English, PTB).
#
# 5 variants (full / hadamard / conv / no_cnorm / no_tau) × 5 seeds = 25 runs.
# Hyperparameters mirror W&B run n7e2qm8t (optuna-v3 phase2 rank2 trial 192).
#
# 2 lanes share a single GPU concurrently. Lane A handles
# (full, hadamard, no_cnorm), lane B handles (conv, no_tau).
# This balances heavy / light variants across lanes and keeps
# FFT-based and non-FFT-based variants separated.
#
# Usage:
#   bash scripts/run_table_ablation_seminfo_parallel.sh [gpu]
#   GPU=1 bash scripts/run_table_ablation_seminfo_parallel.sh
#
# Env overrides:
#   VARIANTS_A="full hadamard no_cnorm"   # lane A queue
#   VARIANTS_B="conv no_tau"              # lane B queue
#   SEEDS="0 1 2 3 4"                     # default seeds
set -uo pipefail

GPU="${GPU:-${1:-0}}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"

WANDB_PROJECT="hol-pcfg"
WANDB_ENTITY="ryosuke-yamaki"
WANDB_GROUP="ablation-seminfo"
LANG_FULL="english"

CONFIG_DIR="config/pas-grammar/english-ew-reward-tbtok-idf"
declare -A CONFIGS=(
    [full]="${CONFIG_DIR}/hnpcfg_n7e2qm8t_en_abl_full.yaml"
    [hadamard]="${CONFIG_DIR}/hnpcfg_n7e2qm8t_en_abl_hadamard.yaml"
    [conv]="${CONFIG_DIR}/hnpcfg_n7e2qm8t_en_abl_conv.yaml"
    [no_cnorm]="${CONFIG_DIR}/hnpcfg_n7e2qm8t_en_abl_no_cnorm.yaml"
    [no_tau]="${CONFIG_DIR}/hnpcfg_n7e2qm8t_en_abl_no_tau.yaml"
)

read -r -a VARIANTS_A <<< "${VARIANTS_A:-full hadamard no_cnorm}"
read -r -a VARIANTS_B <<< "${VARIANTS_B:-conv no_tau}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4}"

LOGDIR="logs/tab_ablation_seminfo"
mkdir -p "$LOGDIR"

echo "============================================"
echo "SemInfo HN-PCFG ablation (n7e2qm8t base)"
echo "GPU:          ${GPU} (2 lanes share GPU)"
echo "Lane A:       ${VARIANTS_A[*]}"
echo "Lane B:       ${VARIANTS_B[*]}"
echo "Seeds:        ${SEEDS[*]}"
echo "Group:        ${WANDB_GROUP}"
echo "Start:        $(date)"
echo "Logs:         ${LOGDIR}/"
echo "============================================"

run_one() {
    local lane="$1" var="$2" seed="$3"
    local conf="${CONFIGS[$var]:-}"
    if [ -z "${conf}" ]; then
        echo "[$(date +%H:%M:%S)] [lane ${lane}] WARN unknown variant ${var}, skipping"
        return 0
    fi
    local ts
    ts=$(date +%m%d_%H%M%S)
    local remark="hnpcfg-ablation-seminfo-en-${var}-seed${seed}"
    local ckpt_dir="ckpt/tab_ablation_seminfo/${var}_seed${seed}_${ts}"
    local logfile="${LOGDIR}/${var}_seed${seed}.log"

    if [ -d "${ckpt_dir%/*}/${var}_seed${seed}_done" ]; then
        echo "[$(date +%H:%M:%S)] [lane ${lane}] SKIP ${var} seed=${seed} (marker exists)"
        return 0
    fi

    mkdir -p "$ckpt_dir"
    echo "[$(date +%H:%M:%S)] [lane ${lane}] START ${var} seed=${seed}"

    CUDA_VISIBLE_DEVICES="$GPU" \
    PL_GLOBAL_SEED="$seed" PYTHONHASHSEED="$seed" \
    python -m parsing_by_maxseminfo.train \
        --conf "$conf" \
        --langstr "$LANG_FULL" \
        --remark "$remark" \
        --ckpt_dir "$ckpt_dir" \
        --val_check_interval 2000 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        --wandb_group "$WANDB_GROUP" \
        --wandb_tags "tab-ablation-seminfo" \
        --wandb_tags "n7e2qm8t-base" \
        --wandb_tags "abl:${var}" \
        --wandb_tags "seed:${seed}" \
        > "$logfile" 2>&1

    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] [lane ${lane}] FAIL ${var} seed=${seed} exit=${rc} log=${logfile}"
    else
        echo "[$(date +%H:%M:%S)] [lane ${lane}] DONE ${var} seed=${seed} log=${logfile}"
    fi
    return "$rc"
}

run_lane() {
    local lane="$1"; shift
    local -a vars=("$@")
    for var in "${vars[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_one "$lane" "$var" "$seed" || true
        done
    done
}

run_lane A "${VARIANTS_A[@]}" &
PID_A=$!
run_lane B "${VARIANTS_B[@]}" &
PID_B=$!

wait "${PID_A}" "${PID_B}"

echo ""
echo "============================================"
echo "All SemInfo ablation runs completed: $(date)"
echo "============================================"
