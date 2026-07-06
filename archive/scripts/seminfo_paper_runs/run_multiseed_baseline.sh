#!/bin/bash
# Multi-seed baseline experiments for TACL submission
# Phase 1A: SN-PCFG baseline (5 seeds) + HN-PCFG MLP-free (5 seeds)
# All at NT=1024, s_dim=512, SemInfo (rl, log_tfidf)
#
# Usage: bash scripts/run_multiseed_baseline.sh [GPU_ID]

set -euo pipefail

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

SNPCFG_CONF="config/pas-grammar/english-ew-reward-tbtok-idf/snpcfg_nt1024_t2048_en.spacy-10k-merged-0pas-fast-6-3-rlstart0.yaml"
HNPCFG_CONF="config/pas-grammar/english-ew-reward-tbtok-idf/hnpcfg_nt1024_t2048_allproj_cnorm_tau_holeterm_rlstart0.yaml"

COMMON_ARGS="--set_training_mode rl --set_mode_reward log_tfidf --val_check_interval 2000 --wandb_project hol-pcfg --wandb_entity ryosuke-yamaki"

SEEDS=(1 2 3 4 5)

LOGDIR="logs/multiseed_baseline"
mkdir -p "$LOGDIR"

echo "============================================"
echo "TACL Multi-Seed Baseline Experiments"
echo "GPU: $GPU"
echo "Start: $(date)"
echo "============================================"

run_experiment() {
    local CONF="$1"
    local MODEL_NAME="$2"
    local SEED="$3"

    local LOGFILE="$LOGDIR/${MODEL_NAME}_seed${SEED}.log"
    local CKPT_DIR="ckpt/multiseed/${MODEL_NAME}-seed${SEED}"
    local REMARK="${MODEL_NAME}-seed${SEED}"

    echo "[$(date +%H:%M:%S)] Starting ${MODEL_NAME} seed=$SEED ..."

    # Set seed via PL_GLOBAL_SEED and PYTHONHASHSEED for reproducibility
    PL_GLOBAL_SEED=$SEED PYTHONHASHSEED=$SEED \
    python -c "
import lightning.pytorch as L
L.seed_everything($SEED, workers=True)
" 2>/dev/null

    PL_GLOBAL_SEED=$SEED PYTHONHASHSEED=$SEED \
    python -m parsing_by_maxseminfo.train \
        --conf "$CONF" \
        $COMMON_ARGS \
        --ckpt_dir "$CKPT_DIR" \
        --remark "$REMARK" \
        --wandb_tags "multiseed,${MODEL_NAME},seed${SEED}" \
        2>&1 | tee "$LOGFILE"

    echo "[$(date +%H:%M:%S)] ${MODEL_NAME} seed=$SEED done."
    grep -E "test/sentence_f1|test/corpus_f1" "$LOGFILE" | tail -2 || true
    echo ""
}

# --- Phase 1: SN-PCFG + SemInfo (5 seeds) ---
echo ""
echo "========== SN-PCFG + SemInfo (NT=1024) =========="
for seed in "${SEEDS[@]}"; do
    run_experiment "$SNPCFG_CONF" "snpcfg-nt1024-seminfo" "$seed"
done

# --- Phase 2: HN-PCFG MLP-free + SemInfo (5 seeds) ---
echo ""
echo "========== HN-PCFG MLP-free + SemInfo (NT=1024) =========="
for seed in "${SEEDS[@]}"; do
    run_experiment "$HNPCFG_CONF" "hnpcfg-holeterm-nt1024-seminfo" "$seed"
done

# --- Summary ---
echo ""
echo "============================================"
echo "All experiments completed: $(date)"
echo "============================================"

echo ""
echo "=== SN-PCFG Results ==="
for seed in "${SEEDS[@]}"; do
    LOGFILE="$LOGDIR/snpcfg-nt1024-seminfo_seed${seed}.log"
    SF1=$(grep "test/sentence_f1" "$LOGFILE" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+' | tail -1 || echo "N/A")
    CF1=$(grep "test/corpus_f1" "$LOGFILE" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+' | tail -1 || echo "N/A")
    echo "  seed=$seed: SF1=$SF1 CF1=$CF1"
done

echo ""
echo "=== HN-PCFG MLP-free Results ==="
for seed in "${SEEDS[@]}"; do
    LOGFILE="$LOGDIR/hnpcfg-holeterm-nt1024-seminfo_seed${seed}.log"
    SF1=$(grep "test/sentence_f1" "$LOGFILE" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+' | tail -1 || echo "N/A")
    CF1=$(grep "test/corpus_f1" "$LOGFILE" 2>/dev/null | grep -oP '[0-9]+\.[0-9]+' | tail -1 || echo "N/A")
    echo "  seed=$seed: SF1=$SF1 CF1=$CF1"
done

echo ""
echo "Compute statistics with:"
echo "  python -c \"import numpy as np; ..."
echo "Logs saved to: $LOGDIR/"
