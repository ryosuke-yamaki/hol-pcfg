#!/bin/bash
set -euo pipefail

DEVICE=${1:-0}
mkdir -p logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_experiment() {
    local config=$1
    local label=$2
    shift 2
    local extra_args="$*"

    log "=== START: ${label} ==="
    log "Config: ${config}"
    python train.py --conf "${config}" --device "${DEVICE}" ${extra_args}

    local config_stem
    config_stem=$(basename "${config}" .yaml)
    local model_dir
    model_dir=$(ls -td log/${config_stem}/*PCFG* 2>/dev/null | head -1)

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
# Phase 3: us+c fair evaluation + scale_c stabilization
# ==========================================

log "Phase 3 starting on device ${DEVICE}"
echo ""

# === Axis A: us+c (no cnorm) fair evaluation (no seed, 4 runs) ===
log "--- Axis A: us+c (no cnorm) x 4 runs ---"
for i in 1 2 3 4; do
    run_experiment config/simplepcfg/hn_pcfg_unit_sphere.yaml \
        "A: us+c-noseed-run${i}"
done

# === Axis B: scale_c stabilization (no seed, 2 runs each) ===
log "--- Axis B1: c_init=4.0 x 2 runs ---"
for i in 1 2; do
    run_experiment config/simplepcfg/hn_pcfg_cnorm_us_c_cinit4.yaml \
        "B1: cnorm-us-c-cinit4-run${i}"
done

log "--- Axis B2: lr_c=0.0005 x 2 runs ---"
for i in 1 2; do
    run_experiment config/simplepcfg/hn_pcfg_cnorm_us_c_lrc.yaml \
        "B2: cnorm-us-c-lrc0.0005-run${i}"
done

log "--- Axis B3: scale_c rules-only x 2 runs ---"
for i in 1 2; do
    run_experiment config/simplepcfg/hn_pcfg_cnorm_us_c_rulesonly.yaml \
        "B3: cnorm-us-c-rulesonly-run${i}"
done

log "All Phase 3 experiments completed."
