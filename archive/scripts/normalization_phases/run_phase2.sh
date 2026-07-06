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
# Phase 2: Fair Comparison + NT Scaling
# No --seed → non-deterministic (matching paper conditions)
# ==========================================

log "Phase 2 starting on device ${DEVICE}"
echo ""

# === Axis B: Fair comparison (no seed, 4 runs each) ===

log "--- Axis B: SN-PCFG NT=4096 x 4 runs (no seed) ---"
for i in 1 2 3 4; do
    run_experiment config/simplepcfg/simple_npcfg_nt4096_t8192_curriculum0.yaml \
        "B1: snpcfg-nt4096-noseed-run${i}"
done

log "--- Axis B: cnorm+us+c NT=4096 x 4 runs (no seed) ---"
for i in 1 2 3 4; do
    run_experiment config/simplepcfg/hn_pcfg_unitsphere_cnorm_scale.yaml \
        "B2: hnpcfg-nt4096-cnorm-us-c-noseed-run${i}"
done

# === Axis C: NT scaling (1 run each, no seed) ===

log "--- Axis C: NT=8192 scaling ---"
run_experiment config/simplepcfg/simple_npcfg_nt8192_t16384_curriculum0.yaml \
    "C1: snpcfg-nt8192-noseed"

run_experiment config/simplepcfg/hn_pcfg_nt8192_unitsphere_cnorm_scale.yaml \
    "C2: hnpcfg-nt8192-cnorm-us-c-noseed"

log "All Phase 2 experiments completed."
