#!/bin/bash
# Run Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 end-to-end, sequentially.
# Total wall-clock: ~12 hours (Phase 0 ~1.5h, Phase 1 ~8h, Phase 2 ~1.5h, Phase 3 ~1h).
#
# Usage (foreground, blocks the terminal):
#   bash scripts/launch_full_pipeline.sh
#
# Usage (background, recommended):
#   mkdir -p logs/pipeline_v3
#   nohup bash scripts/launch_full_pipeline.sh > logs/pipeline_v3/master.log 2>&1 &
#   echo $! > logs/pipeline_v3/master.pid
#   tail -f logs/pipeline_v3/master.log
#
# Per-phase logs stay under logs/phase0_v3/, logs/optuna_${STUDY_NAME}/ etc.
# `set -e` aborts the pipeline on any phase failure.

set -euo pipefail

STUDY_NAME="${1:-hnpcfg-rank1-seminfo-v3}"
SKIP_PHASE0="${SKIP_PHASE0:-0}"          # set to 1 to reuse existing Phase 0 ckpts
SKIP_PHASE1="${SKIP_PHASE1:-0}"          # set to 1 to reuse existing Phase 1 journal
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

timestamp () { date '+%Y-%m-%d %H:%M:%S'; }
log () { printf '\n[%s] === %s ===\n' "$(timestamp)" "$*"; }

log "Full pipeline start for study: $STUDY_NAME"
log "Repo: $REPO_ROOT"
log "PID: $$"

# ------------------------------------------------------------------
# Phase 0: rank1-seminfo baseline x 5 seeds @ 100k
# ------------------------------------------------------------------
if [ "$SKIP_PHASE0" = "1" ]; then
    log "Phase 0 SKIPPED (SKIP_PHASE0=1)"
else
    log "Phase 0: rank1-seminfo baseline (5 seeds)"
    bash scripts/launch_phase0_5seeds.sh
    log "Phase 0 DONE"
fi

# Guard: verify at least one phase0 run produced a ckpt (sanity check).
# Matches both the clean filename=best.ckpt and legacy filename=best-val/... layout.
if ! find ckpt/phase0_baseline -maxdepth 4 -type f -name '*.ckpt' -print -quit 2>/dev/null | grep -q .; then
    log "ERROR: No Phase 0 checkpoints found. Aborting."
    exit 1
fi

# ------------------------------------------------------------------
# Phase 1: Optuna TPE exploration, 200 trials, 16 workers
# ------------------------------------------------------------------
if [ "$SKIP_PHASE1" = "1" ]; then
    log "Phase 1 SKIPPED (SKIP_PHASE1=1)"
else
    log "Phase 1: Optuna TPE (200 trials, 16 workers)"
    bash scripts/launch_optuna_4gpu.sh "$STUDY_NAME"
    log "Phase 1 DONE"
fi

# ------------------------------------------------------------------
# Phase 2: Top-5 x 5 seeds verification
# ------------------------------------------------------------------
log "Phase 2: Top-5 x 5 seeds"
bash scripts/launch_phase2_4gpu.sh "$STUDY_NAME"
log "Phase 2 DONE"

log "Phase 2 aggregation"
python scripts/aggregate_phase2.py "$STUDY_NAME" --phase 2
log "Phase 2 aggregate DONE"

# ------------------------------------------------------------------
# Phase 3: Best config x 5 new seeds (6-10)
# ------------------------------------------------------------------
log "Phase 3: Best x 5 new seeds (6-10)"
bash scripts/launch_phase3_4gpu.sh "$STUDY_NAME"
log "Phase 3 DONE"

log "Phase 3 aggregation"
python scripts/aggregate_phase2.py "$STUDY_NAME" --phase 3
log "Phase 3 aggregate DONE"

log "FULL PIPELINE COMPLETE. Study: $STUDY_NAME"
log "Final results:"
log "  Phase 2: logs/optuna_${STUDY_NAME}_phase2_results.json"
log "  Phase 3: logs/optuna_${STUDY_NAME}_phase3_results.json"
