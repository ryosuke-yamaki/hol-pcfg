#!/bin/bash
# Multilingual paper-config-v2 HN-PCFG training (lr=1.2e-3 variant).
#
# Variant of run_multilingual_paper_10seeds.sh:
#   - Uses *_paper_v2[_<lang>].yaml configs (only optimizer.lr differs:
#     1.0e-3 -> 1.2e-3, closer to the n7e2qm8t Optuna value)
#   - Writes to ckpt/paper_multilingual_v2/ and logs/paper_multilingual_v2/
#   - Adds W&B tag "paper-config-v2" so v1/v2 runs are easy to filter
#
# Layout: 4 GPUs (one language each), 2-way parallelism per GPU, 10 seeds per language.
#
#   GPU 0 -> english  (slot A: seeds 1,3,5,7,9  || slot B: 2,4,6,8,10)
#   GPU 1 -> chinese  (slot A: seeds 1,3,5,7,9  || slot B: 2,4,6,8,10)
#   GPU 2 -> french   (slot A: seeds 1,3,5,7,9  || slot B: 2,4,6,8,10)
#   GPU 3 -> german   (slot A: seeds 1,3,5,7,9  || slot B: 2,4,6,8,10)
#
# Total: 40 runs. Wall clock ~ 5 * (time for one seed).
#
# Usage:
#   bash scripts/run_multilingual_paper_v2_10seeds.sh [en_gpu] [zh_gpu] [fr_gpu] [de_gpu]
# Defaults: 0 1 2 3
#
# Env overrides:
#   LANGS="en zh"                 # subset of languages (default: en zh fr de)
#   SEEDS="1 2 3 4 5"             # subset of seeds     (default: 1..10)
#   PER_GPU_PARALLEL=2            # jobs per GPU        (default: 2)

set -uo pipefail

EN_GPU=${1:-0}
ZH_GPU=${2:-1}
FR_GPU=${3:-2}
DE_GPU=${4:-3}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"

WANDB_PROJECT="hol-pcfg"
WANDB_ENTITY="ryosuke-yamaki"

declare -A CONFIGS=(
    [en]="config/pas-grammar/english-ew-reward-tbtok-idf/hnpcfg_nt1024_t2048_rank1_seminfo_paper_v2.yaml"
    [zh]="config/pas-grammar/chinese-ew-reward-tbtok-idf/hnpcfg_nt1024_t2048_rank1_seminfo_paper_v2_zh.yaml"
    [fr]="config/pas-grammar/french-ew-reward-tbtok-idf/hnpcfg_nt1024_t2048_rank1_seminfo_paper_v2_fr.yaml"
    [de]="config/pas-grammar/german-ew-reward-tbtok-idf-vocab30k/hnpcfg_nt1024_t2048_rank1_seminfo_paper_v2_de.yaml"
)
declare -A LANGSTR=(
    [en]="english"
    [zh]="chinese"
    [fr]="french"
    [de]="german"
)
declare -A LANG_GPU=(
    [en]="$EN_GPU"
    [zh]="$ZH_GPU"
    [fr]="$FR_GPU"
    [de]="$DE_GPU"
)

read -r -a LANGS <<< "${LANGS:-en zh fr de}"
read -r -a SEEDS <<< "${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
PER_GPU_PARALLEL=${PER_GPU_PARALLEL:-2}

LOGDIR="logs/paper_multilingual_v2"
mkdir -p "$LOGDIR"

echo "============================================"
echo "Paper HN-PCFG multilingual training (v2: lr=1.2e-3)"
echo "GPU pinning:      en=GPU${EN_GPU}  zh=GPU${ZH_GPU}  fr=GPU${FR_GPU}  de=GPU${DE_GPU}"
echo "Project root:     $PROJECT_ROOT"
echo "Languages:        ${LANGS[*]}"
echo "Seeds:            ${SEEDS[*]}"
echo "Per-GPU parallel: ${PER_GPU_PARALLEL}"
echo "Start:            $(date)"
echo "Logs:             $LOGDIR/"
echo "============================================"

# ---------------------------------------------------------------------------
# Run one (lang, seed) training on its pinned GPU.
# ---------------------------------------------------------------------------
run_one_seed() {
    local LANG_KEY="$1"
    local SEED="$2"
    local GPU="${LANG_GPU[$LANG_KEY]}"
    local LANG_FULL="${LANGSTR[$LANG_KEY]}"
    local CONF="${CONFIGS[$LANG_KEY]}"

    local TS=$(date +%m%d_%H%M%S)
    local REMARK="hnpcfg-seminfo-paper-v2-${LANG_FULL}-seed${SEED}"
    local CKPT_DIR="ckpt/paper_multilingual_v2/${LANG_FULL}_seed${SEED}_${TS}"
    local LOGFILE="${LOGDIR}/${LANG_FULL}_seed${SEED}_${TS}.log"

    mkdir -p "$CKPT_DIR"

    echo "[$(date +%H:%M:%S)] START lang=${LANG_FULL} seed=${SEED} gpu=${GPU} log=${LOGFILE}"

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
        --wandb_tags "paper-multilingual" \
        --wandb_tags "paper-config-v2" \
        --wandb_tags "lang:${LANG_FULL}" \
        --wandb_tags "seed:${SEED}" \
        > "$LOGFILE" 2>&1

    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] FAIL  lang=${LANG_FULL} seed=${SEED} gpu=${GPU} exit=${rc} log=${LOGFILE}"
    else
        echo "[$(date +%H:%M:%S)] DONE  lang=${LANG_FULL} seed=${SEED} gpu=${GPU} log=${LOGFILE}"
    fi
    return "$rc"
}

# ---------------------------------------------------------------------------
# Sequential chain of seeds for a single slot.
# ---------------------------------------------------------------------------
run_slot() {
    local LANG_KEY="$1"; shift
    local SLOT_SEEDS=("$@")
    local fails=0
    for SEED in "${SLOT_SEEDS[@]}"; do
        if ! run_one_seed "$LANG_KEY" "$SEED"; then
            fails=$((fails + 1))
        fi
    done
    return "$fails"
}

# ---------------------------------------------------------------------------
# Launch PER_GPU_PARALLEL slots per language (round-robin seed split).
# ---------------------------------------------------------------------------
declare -a ALL_PIDS
declare -a ALL_TAGS

for LANG_KEY in "${LANGS[@]}"; do
    LANG_FULL="${LANGSTR[$LANG_KEY]}"
    GPU="${LANG_GPU[$LANG_KEY]}"
    for SLOT in $(seq 0 $((PER_GPU_PARALLEL - 1))); do
        SLOT_SEEDS=()
        for IDX in "${!SEEDS[@]}"; do
            if [ $((IDX % PER_GPU_PARALLEL)) -eq "$SLOT" ]; then
                SLOT_SEEDS+=("${SEEDS[$IDX]}")
            fi
        done
        if [ "${#SLOT_SEEDS[@]}" -eq 0 ]; then
            continue
        fi
        run_slot "$LANG_KEY" "${SLOT_SEEDS[@]}" &
        PID=$!
        ALL_PIDS+=("$PID")
        ALL_TAGS+=("${LANG_FULL}-gpu${GPU}-slot${SLOT}(seeds:${SLOT_SEEDS[*]})")
        echo "[$(date +%H:%M:%S)] LAUNCH lang=${LANG_FULL} gpu=${GPU} slot=${SLOT} seeds=[${SLOT_SEEDS[*]}] pid=${PID}"
    done
done

# ---------------------------------------------------------------------------
# Wait and aggregate failures.
# ---------------------------------------------------------------------------
total_fails=0
for IDX in "${!ALL_PIDS[@]}"; do
    wait "${ALL_PIDS[$IDX]}"
    rc=$?
    echo "[$(date +%H:%M:%S)] SLOT DONE ${ALL_TAGS[$IDX]} fails=${rc}"
    total_fails=$((total_fails + rc))
done

TOTAL_RUNS=$((${#SEEDS[@]} * ${#LANGS[@]}))
echo ""
echo "============================================"
echo "All slots completed: $(date)"
echo "Total failures: ${total_fails} / ${TOTAL_RUNS}"
echo "============================================"

exit "$total_fails"
