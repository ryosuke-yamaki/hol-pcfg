#!/bin/bash
# v_term=delta ablation (Exp 5b from TACL plan)
# HN-PCFG MLP-free with v_term fixed to delta (identity) × 3 seeds
# + SemInfo training
#
# Usage: bash scripts/run_vterm_delta_ablation.sh [GPU_ID]

set -euo pipefail
GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

CONF="config/pas-grammar/english-ew-reward-tbtok-idf/hnpcfg_nt1024_t2048_allproj_cnorm_tau_holeterm_vterm_delta_rlstart0.yaml"
SEEDS=(1 2 3)
LOGDIR="logs/vterm_delta_ablation"
mkdir -p "$LOGDIR"

echo "============================================"
echo "v_term=delta Ablation (Exp 5b)"
echo "GPU: $GPU"
echo "fix_v_term_delta: True (identity, not learned)"
echo "Start: $(date)"
echo "============================================"

for seed in "${SEEDS[@]}"; do
    LOGFILE="$LOGDIR/hnpcfg_holeterm_vterm_delta_seed${seed}.log"
    CKPT_DIR="ckpt/vterm_delta/seed${seed}"
    REMARK="hnpcfg-holeterm-vterm-delta-seed${seed}"

    echo "[$(date +%H:%M:%S)] v_term=delta seed=$seed ..."

    PL_GLOBAL_SEED=$seed \
    python -m parsing_by_maxseminfo.train \
        --conf "$CONF" \
        --set_training_mode rl \
        --set_mode_reward log_tfidf \
        --val_check_interval 2000 \
        --ckpt_dir "$CKPT_DIR" \
        --wandb_project hol-pcfg \
        --wandb_entity ryosuke-yamaki \
        --remark "$REMARK" \
        --wandb_tags "ablation,vterm-delta,seed${seed}" \
        2>&1 | tee "$LOGFILE"

    echo "[$(date +%H:%M:%S)] v_term=delta seed=$seed done."
    grep -E "test/sentence_f1|test/corpus_f1" "$LOGFILE" | tail -2 || true
    echo ""
done

echo "============================================"
echo "All v_term=delta experiments completed: $(date)"
echo "============================================"
