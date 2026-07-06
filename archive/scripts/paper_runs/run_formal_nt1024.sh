#!/usr/bin/env bash
# Sequentially run HN-PCFG NT=1024 training across seeds 0..3 for one dataset on one GPU,
# then run test evaluation on each saved best.pt. Designed to be launched detached via
# nohup so it survives shell exit.
#
# Usage:
#   nohup bash scripts/run_formal_nt1024.sh <symmath|conala> <gpu_id> </tmp/driver.log> &
#   disown
set -uo pipefail

DATASET=${1:?dataset required (symmath|conala)}
GPU=${2:?gpu id required}

cd /workspace/hol-pcfg

CONF=config/formal/hnpcfg_${DATASET}_nt1024.yaml
RESULTS_DIR=log/formal_nt1024_results
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/${DATASET}.summary.tsv"
echo -e "seed\trun_dir\ttest_sentence_f1\ttest_corpus_f1\ttest_avg_ll" > "${SUMMARY}"

for seed in 0 1 2 3; do
  echo "[$(date -u +%FT%TZ)] === ${DATASET} seed=${seed} gpu=${GPU} training ==="
  python train.py --conf "${CONF}" --device "${GPU}" --seed "${seed}"
  train_exit=$?
  if [[ "${train_exit}" -ne 0 ]]; then
    echo "[$(date -u +%FT%TZ)] WARNING: training exited with ${train_exit} for seed=${seed}, continuing"
  fi

  rundir=$(ls -dt log/hnpcfg_${DATASET}_nt1024_seed${seed}/HNPCFG* 2>/dev/null | head -1)
  if [[ -z "${rundir}" || ! -f "${rundir}/best.pt" ]]; then
    echo "[$(date -u +%FT%TZ)] WARNING: no best.pt for seed=${seed}, skipping eval"
    continue
  fi

  echo "[$(date -u +%FT%TZ)] === ${DATASET} seed=${seed} evaluating ${rundir} ==="
  eval_out=$(python evaluate.py --load_from_dir "${rundir}" --device "${GPU}" 2>&1)
  echo "${eval_out}"
  sent_f1=$(echo "${eval_out}" | grep -oE 'Sentence F1: [0-9.]+%' | tail -1 | grep -oE '[0-9.]+')
  corp_f1=$(echo "${eval_out}" | grep -oE 'Corpus F1: [0-9.]+%' | tail -1 | grep -oE '[0-9.]+')
  avg_ll=$(echo "${eval_out}" | grep -oE 'avg likelihood: -?[0-9.]+' | tail -1 | grep -oE '\-?[0-9.]+$')
  echo -e "${seed}\t${rundir}\t${sent_f1}\t${corp_f1}\t${avg_ll}" >> "${SUMMARY}"
done

echo "[$(date -u +%FT%TZ)] === ${DATASET} all seeds done ==="
touch "${RESULTS_DIR}/${DATASET}.done"
