#!/bin/bash
# Paper-config HN-PCFG smaller-capacity ablation sweep (English, PTB).
#
# Two configs x seeds 1..5 on a single GPU.
# Each variant halves exactly one capacity dimension to isolate its effect.
#
#   variant C: (NT, T) = (512, 1024), s_dim = 512   -- (NT, T) halved only
#   variant D: (NT, T) = (1024, 2048), s_dim = 256  -- s_dim halved only
#
# Total: 10 runs.
#
# By default the two variants are run as concurrent slots on the same GPU
# (each slot runs its 5 seeds sequentially). Set PER_GPU_PARALLEL=1 to
# fall back to fully sequential execution.
#
# Usage:
#   bash scripts/run_paper_smaller_5seeds.sh [gpu]
# Default: 0
#
# Env overrides:
#   VARIANTS="C D"         # subset of variants  (default: C D)
#   SEEDS="1 2 3 4 5"      # subset of seeds     (default: 1..5)
#   PER_GPU_PARALLEL=2     # concurrent slots per GPU (default: 2)

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
    [C]="config/pas-grammar/english-ew-reward-tbtok-idf/hnpcfg_nt512_t1024_rank1_seminfo_paper.yaml"
    [D]="config/pas-grammar/english-ew-reward-tbtok-idf/hnpcfg_nt1024_t2048_rank1_sdim256_seminfo_paper.yaml"
)
declare -A VARIANT_TAG=(
    [C]="nt512-t1024"
    [D]="sdim256"
)

read -r -a VARIANTS <<< "${VARIANTS:-C D}"
read -r -a SEEDS <<< "${SEEDS:-1 2 3 4 5}"
PER_GPU_PARALLEL=${PER_GPU_PARALLEL:-2}

LOGDIR="logs/paper_smaller"
mkdir -p "$LOGDIR"

echo "============================================"
echo "Paper HN-PCFG smaller-capacity ablation sweep"
echo "GPU:              ${GPU}"
echo "Project root:     ${PROJECT_ROOT}"
echo "Variants:         ${VARIANTS[*]}"
echo "Seeds:            ${SEEDS[*]}"
echo "Per-GPU parallel: ${PER_GPU_PARALLEL}"
echo "Start:            $(date)"
echo "Logs:             ${LOGDIR}/"
echo "============================================"

run_one() {
    local VAR="$1"
    local SEED="$2"
    local CONF="${CONFIGS[$VAR]}"
    local VTAG="${VARIANT_TAG[$VAR]}"

    local TS=$(date +%m%d_%H%M%S)
    local REMARK="hnpcfg-seminfo-paper-${VTAG}-seed${SEED}"
    local CKPT_DIR="ckpt/paper_smaller/${VTAG}_seed${SEED}_${TS}"
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
        --wandb_tags "paper-smaller" \
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

# Sequential chain of seeds for one slot. Returns the count of failed seeds.
run_slot() {
    local VAR="$1"
    local fails=0
    for SEED in "${SEEDS[@]}"; do
        if ! run_one "$VAR" "$SEED"; then
            fails=$((fails + 1))
        fi
    done
    return "$fails"
}

# Filter unknown variants up front.
ACTIVE_VARIANTS=()
for VAR in "${VARIANTS[@]}"; do
    if [ -z "${CONFIGS[$VAR]:-}" ]; then
        echo "[WARN] unknown variant key '${VAR}', skipping"
        continue
    fi
    ACTIVE_VARIANTS+=("$VAR")
done

TOTAL_RUNS=$((${#ACTIVE_VARIANTS[@]} * ${#SEEDS[@]}))

if [ "$PER_GPU_PARALLEL" -le 1 ] || [ "${#ACTIVE_VARIANTS[@]}" -le 1 ]; then
    # Sequential fallback: run every (variant, seed) pair in order.
    total_fails=0
    for VAR in "${ACTIVE_VARIANTS[@]}"; do
        if ! run_slot "$VAR"; then
            total_fails=$((total_fails + $?))
        fi
    done
else
    # Parallel-by-variant: launch each variant as its own slot.
    declare -a SLOT_PIDS
    declare -a SLOT_TAGS
    for VAR in "${ACTIVE_VARIANTS[@]}"; do
        run_slot "$VAR" &
        PID=$!
        SLOT_PIDS+=("$PID")
        SLOT_TAGS+=("variant=${VARIANT_TAG[$VAR]} (seeds:${SEEDS[*]})")
        echo "[$(date +%H:%M:%S)] LAUNCH ${SLOT_TAGS[-1]} pid=${PID}"
    done
    total_fails=0
    for IDX in "${!SLOT_PIDS[@]}"; do
        wait "${SLOT_PIDS[$IDX]}"
        rc=$?
        echo "[$(date +%H:%M:%S)] SLOT DONE ${SLOT_TAGS[$IDX]} fails=${rc}"
        total_fails=$((total_fails + rc))
    done
fi

echo ""
echo "============================================"
echo "All runs completed: $(date)"
echo "Total failures: ${total_fails} / ${TOTAL_RUNS}"
echo "============================================"

exit "$total_fails"
