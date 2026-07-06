#!/bin/bash
# Paper-config HN-PCFG variant sweep (English, PTB).
#
# Two configs x seeds 1..5, executed sequentially on a single GPU.
#
#   variant A: (NT, T) = (2048, 4096), s_dim = 512
#   variant B: (NT, T) = (1024, 2048), s_dim = 1024
#
# Total: 10 runs.
#
# Usage:
#   bash scripts/run_paper_variants_5seeds.sh [gpu]
# Default: 0
#
# Env overrides:
#   VARIANTS="A B"         # subset of variants  (default: A B)
#   SEEDS="1 2 3 4 5"      # subset of seeds     (default: 1..5)

set -uo pipefail

GPU=${1:-0}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"

WANDB_PROJECT="hol-pcfg"
WANDB_ENTITY="ryosuke-yamaki"
LANG_FULL="english"

declare -A CONFIGS=(
    [A]="config/pas-grammar/english-ew-reward-tbtok-idf/hnpcfg_nt2048_t4096_rank1_seminfo_paper.yaml"
    [B]="config/pas-grammar/english-ew-reward-tbtok-idf/hnpcfg_nt1024_t2048_rank1_sdim1024_seminfo_paper.yaml"
)
declare -A VARIANT_TAG=(
    [A]="nt2048-t4096"
    [B]="sdim1024"
)

read -r -a VARIANTS <<< "${VARIANTS:-A B}"
read -r -a SEEDS <<< "${SEEDS:-1 2 3 4 5}"

LOGDIR="logs/paper_variants"
mkdir -p "$LOGDIR"

echo "============================================"
echo "Paper HN-PCFG variant sweep"
echo "GPU:          ${GPU}"
echo "Project root: ${PROJECT_ROOT}"
echo "Variants:     ${VARIANTS[*]}"
echo "Seeds:        ${SEEDS[*]}"
echo "Start:        $(date)"
echo "Logs:         ${LOGDIR}/"
echo "============================================"

run_one() {
    local VAR="$1"
    local SEED="$2"
    local CONF="${CONFIGS[$VAR]}"
    local VTAG="${VARIANT_TAG[$VAR]}"

    local TS=$(date +%m%d_%H%M%S)
    local REMARK="hnpcfg-seminfo-paper-${VTAG}-seed${SEED}"
    local CKPT_DIR="ckpt/paper_variants/${VTAG}_seed${SEED}_${TS}"
    local LOGFILE="${LOGDIR}/${VTAG}_seed${SEED}_${TS}.log"

    mkdir -p "$CKPT_DIR"

    echo "[$(date +%H:%M:%S)] START variant=${VTAG} seed=${SEED} gpu=${GPU} log=${LOGFILE}"

    CUDA_VISIBLE_DEVICES="$GPU" \
    PL_GLOBAL_SEED="$SEED" PYTHONHASHSEED="$SEED" \
    python -m parsing_by_maxseminfo.train \
        --conf "$CONF" \
        --langstr "$LANG_FULL" \
        --remark "$REMARK" \
        --ckpt_dir "$CKPT_DIR" \
        --val_check_interval 2000 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        --wandb_tags "paper-variants" \
        --wandb_tags "paper-config" \
        --wandb_tags "variant:${VTAG}" \
        --wandb_tags "seed:${SEED}" \
        > "$LOGFILE" 2>&1

    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] FAIL  variant=${VTAG} seed=${SEED} exit=${rc} log=${LOGFILE}"
    else
        echo "[$(date +%H:%M:%S)] DONE  variant=${VTAG} seed=${SEED} log=${LOGFILE}"
    fi
    return "$rc"
}

total_fails=0
total_runs=0
for VAR in "${VARIANTS[@]}"; do
    if [ -z "${CONFIGS[$VAR]:-}" ]; then
        echo "[WARN] unknown variant key '${VAR}', skipping"
        continue
    fi
    for SEED in "${SEEDS[@]}"; do
        total_runs=$((total_runs + 1))
        if ! run_one "$VAR" "$SEED"; then
            total_fails=$((total_fails + 1))
        fi
    done
done

echo ""
echo "============================================"
echo "All runs completed: $(date)"
echo "Total failures: ${total_fails} / ${total_runs}"
echo "============================================"

exit "$total_fails"
