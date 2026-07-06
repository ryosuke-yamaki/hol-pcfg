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
# Phase 1: cnorm exploration + stability
# ==========================================

log "Phase 1 starting on device ${DEVICE}"
echo ""

# #1: cnorm + unit_sphere + learnable c
run_experiment config/simplepcfg/hn_pcfg_unitsphere_cnorm_scale.yaml \
    "Exp1: hnpcfg-nt4096-unitsphere-cnorm-learnablescale-wd0.01"

# #2: cnorm + unit_sphere + learnable tau
run_experiment config/simplepcfg/hn_pcfg_unitsphere_cnorm_tau.yaml \
    "Exp2: hnpcfg-nt4096-unitsphere-cnorm-learnabletau-wd0.01"

# #3: unit_sphere + c + tau (double scaling)
run_experiment config/simplepcfg/hn_pcfg_unitsphere_scale_tau.yaml \
    "Exp3: hnpcfg-nt4096-unitsphere-learnablescale-learnabletau-wd0.01"

# #4: cnorm + normless1 + tau
run_experiment config/simplepcfg/hn_pcfg_normless1_cnorm_tau.yaml \
    "Exp4: hnpcfg-nt4096-normless1-cnorm-learnabletau-wd0.01"

# #5: cnorm + unit_sphere + c + wd0 (WD sensitivity for cnorm)
run_experiment config/simplepcfg/hn_pcfg_unitsphere_cnorm_scale_wd0.yaml \
    "Exp5: hnpcfg-nt4096-unitsphere-cnorm-learnablescale-wd0"

# #6: unit_sphere + c additional seed for statistics
run_experiment config/simplepcfg/hn_pcfg_unit_sphere.yaml \
    "Exp6: hnpcfg-nt4096-unitsphere-learnablescale-wd0.01-seed789" \
    --seed 789

log "All Phase 1 experiments completed."
