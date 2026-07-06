#!/bin/bash
# Multilingual sem-info HN-PCFG replication using W&B run n7e2qm8t HPs.
#
# Layout: 3 GPUs in parallel, one language each, seeds 1..5 sequential per GPU.
#   GPU 0 -> chinese  (seed 1, 2, 3, 4, 5)
#   GPU 1 -> french   (seed 1, 2, 3, 4, 5)
#   GPU 2 -> german   (seed 1, 2, 3, 4, 5)
# Total: 15 runs, ~5 wall-clock runs (one per seed) since languages run concurrently.
#
# Per-language config: config/pas-grammar/<lang_dir>/hnpcfg_nt1024_t2048_n7e2qm8t_<lang>.yaml
#
# Usage:
#   bash scripts/run_multilingual_n7e2qm8t_hp.sh [zh_gpu] [fr_gpu] [de_gpu]
# Defaults: 0 1 2

set -uo pipefail

ZH_GPU=${1:-0}
FR_GPU=${2:-1}
DE_GPU=${3:-2}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"

WANDB_PROJECT="hol-pcfg"
WANDB_ENTITY="ryosuke-yamaki"

declare -A CONFIGS=(
    [zh]="config/pas-grammar/chinese-ew-reward-tbtok-idf/hnpcfg_nt1024_t2048_n7e2qm8t_zh.yaml"
    [fr]="config/pas-grammar/french-ew-reward-tbtok-idf/hnpcfg_nt1024_t2048_n7e2qm8t_fr.yaml"
    [de]="config/pas-grammar/german-ew-reward-tbtok-idf-vocab30k/hnpcfg_nt1024_t2048_n7e2qm8t_de.yaml"
)
declare -A LANGSTR=(
    [zh]="chinese"
    [fr]="french"
    [de]="german"
)
declare -A LANG_GPU=(
    [zh]="$ZH_GPU"
    [fr]="$FR_GPU"
    [de]="$DE_GPU"
)

# LANGS can be overridden via env var, e.g.  LANGS="zh fr" bash scripts/run_multilingual_n7e2qm8t_hp.sh
read -r -a LANGS <<< "${LANGS:-zh fr de}"
# SEEDS can be overridden via env var, e.g.  SEEDS="6 7 8 9 10" bash scripts/run_multilingual_n7e2qm8t_hp.sh
read -r -a SEEDS <<< "${SEEDS:-1 2 3 4 5}"

LOGDIR="logs/multilingual_n7e2qm8t"
mkdir -p "$LOGDIR"

echo "============================================"
echo "Multilingual sem-info HN-PCFG (n7e2qm8t HPs)"
echo "GPU pinning:      zh=GPU${ZH_GPU}  fr=GPU${FR_GPU}  de=GPU${DE_GPU}"
echo "Project root:     $PROJECT_ROOT"
echo "Languages:        ${LANGS[*]}"
echo "Seeds:            ${SEEDS[*]}"
echo "Start:            $(date)"
echo "Logs:             $LOGDIR/"
echo "============================================"

# ---------------------------------------------------------------------------
# Per-language worker: runs seeds sequentially on a single pinned GPU.
# Writes a per-language summary log; per-run logs are <lang>_seed<N>_<ts>.log.
# Returns the number of failed seeds via exit code.
# ---------------------------------------------------------------------------
run_lang_worker() {
    local LANG_KEY="$1"
    local CONF="${CONFIGS[$LANG_KEY]}"
    local LANG_FULL="${LANGSTR[$LANG_KEY]}"
    local GPU="${LANG_GPU[$LANG_KEY]}"
    local SUMMARY_LOG="${LOGDIR}/_worker_${LANG_FULL}_gpu${GPU}.log"

    {
        echo "============================================"
        echo "Worker start: lang=${LANG_FULL} gpu=${GPU}"
        echo "Date:         $(date)"
        echo "============================================"
    } | tee "$SUMMARY_LOG"

    local lang_fails=0
    for SEED in "${SEEDS[@]}"; do
        local TS=$(date +%m%d_%H%M%S)
        local REMARK="hnpcfg-seminfo-n7e2qm8t-${LANG_FULL}-seed${SEED}"
        local CKPT_DIR="ckpt/multilingual_n7e2qm8t/${LANG_FULL}_seed${SEED}_${TS}"
        local LOGFILE="${LOGDIR}/${LANG_FULL}_seed${SEED}_${TS}.log"

        mkdir -p "$CKPT_DIR"

        {
            echo ""
            echo "[$(date +%H:%M:%S)] START lang=${LANG_FULL} seed=${SEED} gpu=${GPU}"
            echo "  conf:      $CONF"
            echo "  ckpt_dir:  $CKPT_DIR"
            echo "  logfile:   $LOGFILE"
        } | tee -a "$SUMMARY_LOG"

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
            --wandb_tags "multilingual-seminfo" \
            --wandb_tags "n7e2qm8t-hp" \
            --wandb_tags "lang:${LANG_FULL}" \
            --wandb_tags "seed:${SEED}" \
            > "$LOGFILE" 2>&1

        local rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "[$(date +%H:%M:%S)] FAIL lang=${LANG_FULL} seed=${SEED} (exit=${rc})" | tee -a "$SUMMARY_LOG"
            lang_fails=$((lang_fails + 1))
        else
            echo "[$(date +%H:%M:%S)] DONE lang=${LANG_FULL} seed=${SEED}" | tee -a "$SUMMARY_LOG"
            grep -E "test/sentence_f1|test/corpus_f1" "$LOGFILE" | tail -2 | tee -a "$SUMMARY_LOG" || true
        fi
    done

    {
        echo ""
        echo "============================================"
        echo "Worker end: lang=${LANG_FULL} gpu=${GPU} fails=${lang_fails}/${#SEEDS[@]}"
        echo "Date:       $(date)"
        echo "============================================"
    } | tee -a "$SUMMARY_LOG"

    return "$lang_fails"
}

# ---------------------------------------------------------------------------
# Launch one worker per language in the background.
# ---------------------------------------------------------------------------
declare -A WORKER_PIDS
for LANG_KEY in "${LANGS[@]}"; do
    run_lang_worker "$LANG_KEY" &
    WORKER_PIDS[$LANG_KEY]=$!
    echo "[$(date +%H:%M:%S)] LAUNCH worker lang=${LANGSTR[$LANG_KEY]} gpu=${LANG_GPU[$LANG_KEY]} pid=${WORKER_PIDS[$LANG_KEY]}"
done

# ---------------------------------------------------------------------------
# Wait for all workers and aggregate failure counts.
# ---------------------------------------------------------------------------
total_fails=0
for LANG_KEY in "${LANGS[@]}"; do
    wait "${WORKER_PIDS[$LANG_KEY]}"
    rc=$?
    LANG_FULL="${LANGSTR[$LANG_KEY]}"
    echo "[$(date +%H:%M:%S)] WORKER DONE lang=${LANG_FULL} fails=${rc}/${#SEEDS[@]}"
    total_fails=$((total_fails + rc))
done

echo ""
echo "============================================"
echo "All workers completed: $(date)"
echo "Total failures: $total_fails / $((${#SEEDS[@]} * ${#LANGS[@]}))"
echo "============================================"

exit "$total_fails"
