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
# Phase 4.5: Inline entity cnorm + Xavier
# ==========================================

log "Phase 4.5 starting on device ${DEVICE}"
echo ""

# Exp D: inline entity cnorm + single tau (rule only) × 2 runs
log "--- Exp D: inline ecnorm + single tau × 2 runs ---"
for i in 1 2; do
    run_experiment config/simplepcfg/hn_pcfg_ecnorm_inline_tau.yaml \
        "ExpD: ecnorm-inline-tau-run${i}"
done

# Exp E: inline entity cnorm + multi_tau × 2 runs
log "--- Exp E: inline ecnorm + multi_tau × 2 runs ---"
for i in 1 2; do
    run_experiment config/simplepcfg/hn_pcfg_ecnorm_inline_multitau.yaml \
        "ExpE: ecnorm-inline-multitau-run${i}"
done

log "All Phase 4.5 experiments completed."
