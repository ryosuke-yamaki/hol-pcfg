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
# Phase 0.5 Control Experiments
# ==========================================

log "Phase 0.5 starting on device ${DEVICE}"
echo ""

# #1: normless1_paper (corrected baseline)
run_experiment config/simplepcfg/hn_pcfg_normless1_paper.yaml \
    "Exp1: hnpcfg-nt4096-normless1paper-wd0.01"

# #2: unit_sphere c=1 fixed (norm uniformity only)
run_experiment config/simplepcfg/hn_pcfg_unit_sphere_noscale.yaml \
    "Exp2: hnpcfg-nt4096-unitsphere-noscale-wd0.01"

# #3: normless1 + global learnable scale (scale scope only)
run_experiment config/simplepcfg/hn_pcfg_normless1_scale.yaml \
    "Exp3: hnpcfg-nt4096-normless1-learnablescale-wd0.01"

# #5: unit_sphere + learnable tau (2x2 factorial completion)
run_experiment config/simplepcfg/hn_pcfg_unit_sphere_tau.yaml \
    "Exp5: hnpcfg-nt4096-unitsphere-learnabletau-wd0.01"

# #4: multi-seed
SEEDS=(42 123 456)

for seed in "${SEEDS[@]}"; do
    run_experiment config/simplepcfg/hn_pcfg_unit_sphere.yaml \
        "Exp4: hnpcfg-nt4096-unitsphere-learnablescale-wd0.01-seed${seed}" \
        --seed "${seed}"
done

for seed in "${SEEDS[@]}"; do
    run_experiment config/simplepcfg/simple_npcfg_nt4096_t8192_curriculum0.yaml \
        "Exp4: snpcfg-nt4096-baseline-seed${seed}" \
        --seed "${seed}"
done

log "All Phase 0.5 experiments completed."
