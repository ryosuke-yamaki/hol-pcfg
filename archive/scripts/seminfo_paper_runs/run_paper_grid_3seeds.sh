#!/bin/bash
# Paper-config HN-PCFG (NT, s_dim) grid search (English, PTB).
#
# 3 x 3 grid x 3 seeds = 27 runs. Probes how far capacity can be
# reduced before parsing performance degrades sharply.
#
#   NT     in {1024, 512, 256}      (T = 2 * NT)
#   s_dim  in {256,  128,  64}
#
# Variants are partitioned across PER_GPU_PARALLEL slots in
# round-robin order so every slot gets a mix of large and small
# cells (wall is balanced across slots). Each slot then runs its
# assigned (variant, seed) pairs sequentially on the same GPU.
#
# Usage:
#   bash scripts/run_paper_grid_3seeds.sh [gpu]
# Default: 0
#
# Env overrides:
#   VARIANTS="nt1024_sdim256 nt512_sdim128 ..."  # subset of cells (default: all 9)
#   SEEDS="1 2 3"                                 # default: 1..3
#   PER_GPU_PARALLEL=3                            # concurrent slots (default: 3)

set -uo pipefail

GPU=${1:-0}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"

WANDB_PROJECT="hol-pcfg"
WANDB_ENTITY="ryosuke-yamaki"
LANG_FULL="english"

CONFIG_DIR="config/pas-grammar/english-ew-reward-tbtok-idf"

declare -A CONFIGS=(
    [nt1024_sdim256]="${CONFIG_DIR}/hnpcfg_nt1024_t2048_rank1_sdim256_seminfo_paper.yaml"
    [nt1024_sdim128]="${CONFIG_DIR}/hnpcfg_nt1024_t2048_rank1_sdim128_seminfo_paper.yaml"
    [nt1024_sdim64]="${CONFIG_DIR}/hnpcfg_nt1024_t2048_rank1_sdim64_seminfo_paper.yaml"
    [nt512_sdim256]="${CONFIG_DIR}/hnpcfg_nt512_t1024_rank1_sdim256_seminfo_paper.yaml"
    [nt512_sdim128]="${CONFIG_DIR}/hnpcfg_nt512_t1024_rank1_sdim128_seminfo_paper.yaml"
    [nt512_sdim64]="${CONFIG_DIR}/hnpcfg_nt512_t1024_rank1_sdim64_seminfo_paper.yaml"
    [nt256_sdim256]="${CONFIG_DIR}/hnpcfg_nt256_t512_rank1_sdim256_seminfo_paper.yaml"
    [nt256_sdim128]="${CONFIG_DIR}/hnpcfg_nt256_t512_rank1_sdim128_seminfo_paper.yaml"
    [nt256_sdim64]="${CONFIG_DIR}/hnpcfg_nt256_t512_rank1_sdim64_seminfo_paper.yaml"
)
declare -A VARIANT_TAG=(
    [nt1024_sdim256]="nt1024-sdim256"
    [nt1024_sdim128]="nt1024-sdim128"
    [nt1024_sdim64]="nt1024-sdim64"
    [nt512_sdim256]="nt512-sdim256"
    [nt512_sdim128]="nt512-sdim128"
    [nt512_sdim64]="nt512-sdim64"
    [nt256_sdim256]="nt256-sdim256"
    [nt256_sdim128]="nt256-sdim128"
    [nt256_sdim64]="nt256-sdim64"
)
declare -A VARIANT_NT=(
    [nt1024_sdim256]=1024 [nt1024_sdim128]=1024 [nt1024_sdim64]=1024
    [nt512_sdim256]=512 [nt512_sdim128]=512 [nt512_sdim64]=512
    [nt256_sdim256]=256 [nt256_sdim128]=256 [nt256_sdim64]=256
)
declare -A VARIANT_SDIM=(
    [nt1024_sdim256]=256 [nt1024_sdim128]=128 [nt1024_sdim64]=64
    [nt512_sdim256]=256 [nt512_sdim128]=128 [nt512_sdim64]=64
    [nt256_sdim256]=256 [nt256_sdim128]=128 [nt256_sdim64]=64
)

# Default order: row-major (NT desc, then s_dim desc).
DEFAULT_VARIANTS=(
    nt1024_sdim256 nt1024_sdim128 nt1024_sdim64
    nt512_sdim256  nt512_sdim128  nt512_sdim64
    nt256_sdim256  nt256_sdim128  nt256_sdim64
)

read -r -a VARIANTS <<< "${VARIANTS:-${DEFAULT_VARIANTS[*]}}"
read -r -a SEEDS <<< "${SEEDS:-1 2 3}"
PER_GPU_PARALLEL=${PER_GPU_PARALLEL:-3}

LOGDIR="logs/paper_grid"
mkdir -p "$LOGDIR"

echo "============================================"
echo "Paper HN-PCFG (NT, s_dim) grid search"
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
    local NT="${VARIANT_NT[$VAR]}"
    local SDIM="${VARIANT_SDIM[$VAR]}"

    local TS=$(date +%m%d_%H%M%S)
    local REMARK="hnpcfg-seminfo-paper-${VTAG}-seed${SEED}"
    local CKPT_DIR="ckpt/paper_grid/${VTAG}_seed${SEED}_${TS}"
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
        --wandb_tags "paper-grid" \
        --wandb_tags "paper-config" \
        --wandb_tags "variant:${VTAG}" \
        --wandb_tags "nt:${NT}" \
        --wandb_tags "sdim:${SDIM}" \
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

# Round-robin assignment of variants to slots so each slot gets a mix
# of large/medium/small cells (heavy ones do not all pile on one slot).
declare -A SLOT_VARIANTS
for IDX in "${!ACTIVE_VARIANTS[@]}"; do
    SLOT=$((IDX % PER_GPU_PARALLEL))
    SLOT_VARIANTS[$SLOT]+="${ACTIVE_VARIANTS[$IDX]} "
done

# Sequential chain of (variant, seed) pairs for one slot.
run_slot() {
    local SLOT="$1"
    local fails=0
    for VAR in ${SLOT_VARIANTS[$SLOT]}; do
        for SEED in "${SEEDS[@]}"; do
            if ! run_one "$VAR" "$SEED"; then
                fails=$((fails + 1))
            fi
        done
    done
    return "$fails"
}

if [ "$PER_GPU_PARALLEL" -le 1 ] || [ "${#ACTIVE_VARIANTS[@]}" -le 1 ]; then
    # Sequential fallback.
    total_fails=0
    for VAR in "${ACTIVE_VARIANTS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            if ! run_one "$VAR" "$SEED"; then
                total_fails=$((total_fails + 1))
            fi
        done
    done
else
    declare -a SLOT_PIDS
    declare -a SLOT_TAGS
    for SLOT in $(seq 0 $((PER_GPU_PARALLEL - 1))); do
        if [ -z "${SLOT_VARIANTS[$SLOT]:-}" ]; then
            continue
        fi
        run_slot "$SLOT" &
        PID=$!
        SLOT_PIDS+=("$PID")
        SLOT_TAGS+=("slot=${SLOT} variants=[${SLOT_VARIANTS[$SLOT]}] seeds=[${SEEDS[*]}]")
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
