#!/bin/bash
set -euo pipefail

DEVICE=${1:-0}
mkdir -p logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_experiment() {
    local config=$1; local label=$2; shift 2; local extra_args="$*"
    log "=== START: ${label} ==="
    log "Config: ${config}"
    python train.py --conf "${config}" --device "${DEVICE}" ${extra_args}
    local config_stem=$(basename "${config}" .yaml)
    local model_dir=$(ls -td log/${config_stem}/*PCFG* 2>/dev/null | head -1)
    if [ -n "${model_dir}" ]; then
        log "Evaluating: ${model_dir}"
        python evaluate.py --load_from_dir "${model_dir}" --device "${DEVICE}"
    else
        log "WARNING: No model directory found for ${label}"
    fi
    log "=== DONE: ${label} ==="
    echo ""
}

# ==========================================
# Phase 4.5 Exp F/G: All-projection cnorm
# ==========================================

log "Phase 4.5 Exp F/G starting on device ${DEVICE}"
echo ""

# Exp G: single tau × 4 runs (先に実行)
log "--- Exp G: allproj cnorm + single tau × 4 runs ---"
for i in 1 2 3 4; do
    run_experiment config/simplepcfg/hn_pcfg_allproj_cnorm_tau.yaml \
        "ExpG: allproj-cnorm-tau-run${i}"
done

# Exp F: multi_tau × 4 runs
log "--- Exp F: allproj cnorm + multi_tau × 4 runs ---"
for i in 1 2 3 4; do
    run_experiment config/simplepcfg/hn_pcfg_allproj_cnorm_multitau.yaml \
        "ExpF: allproj-cnorm-multitau-run${i}"
done

log "All Phase 4.5 Exp F/G experiments completed."
