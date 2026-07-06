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
# Phase 3.5: rules-only scaling deep dive
# ==========================================

log "Phase 3.5 starting on device ${DEVICE}"
echo ""

# Exp 1: cnorm+us+c rules-only × 2 more runs (total 4 with Phase 3)
log "--- Exp1: cnorm+us+c rules-only × 2 additional runs ---"
for i in 3 4; do
    run_experiment config/simplepcfg/hn_pcfg_cnorm_us_c_rulesonly.yaml \
        "Exp1: cnorm-us-c-rulesonly-run${i}"
done

# Exp 2: us+tau (no cnorm, natural rules-only) × 4 runs
log "--- Exp2: us+tau (no cnorm) × 4 runs ---"
for i in 1 2 3 4; do
    run_experiment config/simplepcfg/hn_pcfg_unit_sphere_tau.yaml \
        "Exp2: us-tau-noseed-run${i}"
done

# Exp 3: cnorm+us+tau × 4 runs
log "--- Exp3: cnorm+us+tau × 4 runs ---"
for i in 1 2 3 4; do
    run_experiment config/simplepcfg/hn_pcfg_unitsphere_cnorm_tau.yaml \
        "Exp3: cnorm-us-tau-noseed-run${i}"
done

log "All Phase 3.5 experiments completed."
