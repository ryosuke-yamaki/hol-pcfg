#!/bin/bash
set -euo pipefail

STUDY_NAME=${1:-hn-pcfg-hp}
BASE_CONFIG=${2:-archive/configs/normalization_phases/hn_pcfg_allproj_cnorm_tau.yaml}
JOURNAL_PATH=${3:-optuna_journal.log}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

mkdir -p logs

echo "[$(date)] Starting Optuna HP search: ${STUDY_NAME}"
echo "  Base config: ${BASE_CONFIG}"
echo "  Journal: ${JOURNAL_PATH}"
echo "  GPUs: 0, 1, 2, 3"
echo ""

for device in 0 1 2 3; do
    echo "[$(date)] Launching worker on GPU ${device}"
    python "${REPO_ROOT}/scripts/run_optuna.py" \
        --device $device \
        --study-name $STUDY_NAME \
        --base-config $BASE_CONFIG \
        --journal-path $JOURNAL_PATH \
        > logs/optuna_gpu${device}.log 2>&1 &
done

echo ""
echo "[$(date)] All 4 workers launched. Monitor with:"
echo "  tail -f logs/optuna_gpu0.log"
echo "  # Stop all: kill %1 %2 %3 %4"

wait
echo "[$(date)] All workers completed."
