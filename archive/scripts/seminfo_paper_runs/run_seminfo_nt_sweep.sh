#!/bin/bash
# Sequential SemInfo experiments: NT=512, 1024, 2048, 4096
# Each run: 100k steps, patience=100 (early stopping disabled), val every 2000 steps
set -e

cd /workspace/hol-pcfg-seminfo

CONFIGS=(
  "512  1024  hnpcfg_nt512_t1024_allproj_cnorm_tau_100k.yaml"
  "1024 2048  hnpcfg_nt1024_t2048_allproj_cnorm_tau_100k.yaml"
  "2048 4096  hnpcfg_nt2048_t4096_allproj_cnorm_tau_100k.yaml"
  "4096 8192  hnpcfg_nt4096_t8192_allproj_cnorm_tau_100k.yaml"
)

CONFIG_DIR="config/pas-grammar/english-ew-reward-tbtok-idf"

for entry in "${CONFIGS[@]}"; do
  read -r NT T YAML <<< "$entry"
  echo ""
  echo "=============================================="
  echo "  Starting NT=${NT}, T=${T}"
  echo "  $(date)"
  echo "=============================================="

  python3 -m parsing_by_maxseminfo.train \
    -c "${CONFIG_DIR}/${YAML}" \
    --langstr english \
    --ckpt_dir "ckpt/hnpcfg-nt${NT}-100k" \
    --remark "hnpcfg-nt${NT}-seminfo-100k" \
    --wandb_project hol-pcfg \
    --wandb_tags seminfo --wandb_tags hn-pcfg --wandb_tags "nt${NT}" --wandb_tags 100k-overfit-check \
    --set_training_mode rl \
    --mode_reward log_tfidf \
    --val_check_interval 2000 \
    --ngpu 1

  echo ""
  echo "  NT=${NT} completed at $(date)"
  echo "=============================================="
done

echo ""
echo "All 4 runs completed at $(date)"
