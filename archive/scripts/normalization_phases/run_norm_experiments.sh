#!/bin/bash
# Phase 0: Score Scale / Norm experiments
# Runs 7 experiments sequentially with automatic test evaluation after each

set -e

CONFIGS=(
  config/simplepcfg/hn_pcfg_fixedscale10.yaml
  config/simplepcfg/hn_pcfg_tau.yaml
  config/simplepcfg/hn_pcfg_maxnorm5.yaml
  config/simplepcfg/hn_pcfg_maxnorm10.yaml
  config/simplepcfg/hn_pcfg_unit_sphere.yaml
  config/simplepcfg/hn_pcfg_wd0.yaml
  config/simplepcfg/hn_pcfg_wd001.yaml
)

echo "[$(date)] =================================================="
echo "[$(date)] Starting Phase 0: 7 norm experiments"
echo "[$(date)] =================================================="

for conf in "${CONFIGS[@]}"; do
  name=$(basename "$conf" .yaml)
  echo ""
  echo "[$(date)] ===== Starting: $name ====="
  python3 train.py --conf "$conf" --device 0 2>&1 | tee "log/${name}_train.log"

  # Find the latest model directory created by this run
  MODEL_DIR=$(ls -td log/*/HNPCFG* 2>/dev/null | head -1)
  if [ -n "$MODEL_DIR" ] && [ -f "$MODEL_DIR/best.pt" ]; then
    echo "[$(date)] Evaluating test set: $MODEL_DIR"
    python3 evaluate.py --load_from_dir "$MODEL_DIR" --decode_type mbr --device 0 \
      2>&1 | tee "log/${name}_eval.log"
  else
    echo "[$(date)] WARNING: No best.pt found for $name"
  fi

  echo "[$(date)] ===== Finished: $name ====="
done

echo ""
echo "[$(date)] =================================================="
echo "[$(date)] All Phase 0 experiments complete."
echo "[$(date)] =================================================="
