#!/bin/bash
# Sequential training: HC-PCFG-R → HC-PCFG-P → HN-PCFG CP-8
set -e

DEVICE="${1:-0}"
SEED="${2:-}"

SEED_FLAG=""
if [ -n "$SEED" ]; then
    SEED_FLAG="--seed $SEED"
fi

echo "=== [1/3] HC-PCFG-R (relation injection) ==="
python train.py -c config/simplepcfg/hc_pcfg_relation.yaml -d "$DEVICE" $SEED_FLAG

echo "=== [2/3] HC-PCFG-P (parent injection) ==="
python train.py -c config/simplepcfg/hc_pcfg_parent.yaml -d "$DEVICE" $SEED_FLAG

echo "=== [3/3] HN-PCFG CP-8 (multi-head relation, R=8) ==="
python train.py -c config/simplepcfg/hn_pcfg_allproj_cnorm_tau_xavier_cp8.yaml -d "$DEVICE" $SEED_FLAG

echo "=== All training runs completed ==="
