#!/bin/bash
# HC-PCFG Phase 0 Diagnostic Experiments (sequential)
# D1: HN-PCFG@NT=2048 baseline
# D2: HC-PCFG z=0 (encoder frozen)
# D3-R: HC-PCFG beta=0, relation injection
# D3-P: HC-PCFG beta=0, parent injection
set -e

DEVICE="${1:-0}"

echo "============================================"
echo "HC-PCFG Phase 0 Diagnostics"
echo "Device: $DEVICE"
echo "Start: $(date)"
echo "============================================"

echo ""
echo "=== [1/4] D1: HN-PCFG@NT=2048 baseline ==="
echo "Start: $(date)"
python train.py -c config/simplepcfg/hn_pcfg_allproj_cnorm_tau_nt2048.yaml -d "$DEVICE"
echo "Done: $(date)"

echo ""
echo "=== [2/4] D2: HC-PCFG z=0 (encoder frozen) ==="
echo "Start: $(date)"
python train.py -c config/simplepcfg/hc_pcfg_z_zero.yaml -d "$DEVICE"
echo "Done: $(date)"

echo ""
echo "=== [3/4] D3-R: HC-PCFG beta=0 relation ==="
echo "Start: $(date)"
python train.py -c config/simplepcfg/hc_pcfg_beta_zero_relation.yaml -d "$DEVICE"
echo "Done: $(date)"

echo ""
echo "=== [4/4] D3-P: HC-PCFG beta=0 parent ==="
echo "Start: $(date)"
python train.py -c config/simplepcfg/hc_pcfg_beta_zero_parent.yaml -d "$DEVICE"
echo "Done: $(date)"

echo ""
echo "============================================"
echo "All diagnostics completed: $(date)"
echo "============================================"
