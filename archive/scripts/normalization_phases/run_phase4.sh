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
# Phase 4: Entity cnorm + Multi-tau
# ==========================================

log "Phase 4 starting on device ${DEVICE}"
echo ""

# Exp A: unit_sphere + cnorm + multi_tau (isolate multi_tau effect)
log "--- Exp A: unit_sphere + cnorm + multi_tau × 4 runs ---"
for i in 1 2 3 4; do
    run_experiment config/simplepcfg/hn_pcfg_us_cnorm_multitau.yaml \
        "ExpA: us-cnorm-multitau-run${i}"
done

# Exp B: freq_cnorm + cnorm + multi_tau (pure phase model)
log "--- Exp B: freq_cnorm + cnorm + multi_tau × 4 runs ---"
for i in 1 2 3 4; do
    run_experiment config/simplepcfg/hn_pcfg_ecnorm_cnorm_multitau.yaml \
        "ExpB: ecnorm-cnorm-multitau-run${i}"
done

# Exp C: freq_cnorm + cnorm + scale_c (isolate entity cnorm effect)
log "--- Exp C: freq_cnorm + cnorm + scale_c × 4 runs ---"
for i in 1 2 3 4; do
    run_experiment config/simplepcfg/hn_pcfg_ecnorm_cnorm_scale.yaml \
        "ExpC: ecnorm-cnorm-scale-run${i}"
done

log "All Phase 4 experiments completed."
