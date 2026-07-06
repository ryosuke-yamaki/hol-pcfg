#!/usr/bin/env bash
# Build KTB Japanese training pickles for morpheme-level PCFG configs.
#
# Pipeline (Li et al., 2020):
#   1. Clone Keyaki Treebank v1.1 (CC BY 4.0) from GitHub.
#   2. Clone Li et al.'s UnsupConstParseEval preprocessor.
#   3. Apply the standard KTB cleanup (drop empty categories, strip functional
#      labels, lowercase, replace digits with [num], replace singletons with
#      [unk], 80/10/10 shuffled split, length <=40 train filter, punct removed).
#   4. Convert cleaned bracket files to pickles in the format expected by
#      `parser/helper/data_module.py` ({'word', 'pos', 'gold_tree'}).
#
# Usage:
#   bash scripts/prepare_ktb_data.sh [WORKDIR]
# Defaults: WORKDIR=$PWD/.ktb_workspace

set -euo pipefail

WORKDIR="${1:-$PWD/.ktb_workspace}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/data/clean"
PREFIX="japanese-ktb-len40-nopunct-"

mkdir -p "${WORKDIR}" "${OUT_DIR}"

# 1. Keyaki Treebank v1.1
if [ ! -d "${WORKDIR}/KeyakiTreebank" ]; then
    echo "[prepare_ktb] cloning Keyaki Treebank ..."
    git -C "${WORKDIR}" clone --depth 1 https://github.com/ajb129/KeyakiTreebank.git
fi

# 2. Li et al. preprocessor (de facto standard; reused unchanged)
if [ ! -d "${WORKDIR}/UnsupConstParseEval" ]; then
    echo "[prepare_ktb] cloning UnsupConstParseEval (Li et al., 2020) ..."
    git -C "${WORKDIR}" clone --depth 1 https://github.com/i-lijun/UnsupConstParseEval.git
fi

# 3. Run Li et al. preprocessing -> bracketed text splits
CLEAN_ROOT="${WORKDIR}/UnsupConstParseEval/data/cleaned_datasets/japanese/ktb"
LEN40_NOPUNCT="${CLEAN_ROOT}/ktb_len40_nopunct"
if [ ! -f "${LEN40_NOPUNCT}/train" ] || [ ! -f "${LEN40_NOPUNCT}/dev" ] || [ ! -f "${LEN40_NOPUNCT}/test" ]; then
    echo "[prepare_ktb] running Li et al. preprocess ..."
    pushd "${WORKDIR}/UnsupConstParseEval" >/dev/null
    mkdir -p data/japanese
    ln -sfn "${WORKDIR}/KeyakiTreebank" data/japanese/KeyakiTreebank
    python preprocess.py \
        --path_to_raw_ptb "" \
        --path_to_raw_ktb data/japanese/KeyakiTreebank/treebank \
        --path_to_ktb_output_dir data/cleaned_datasets/japanese/ktb
    popd >/dev/null
fi

# 4. Bracketed text -> pickle in the format the data loader expects
echo "[prepare_ktb] building pickles in ${OUT_DIR} ..."
cd "${REPO_ROOT}"
python preprocessing.py \
    --train_file "${LEN40_NOPUNCT}/train" \
    --val_file "${LEN40_NOPUNCT}/dev" \
    --test_file "${LEN40_NOPUNCT}/test" \
    --cache_path "${OUT_DIR}/${PREFIX}"

echo "[prepare_ktb] done. Generated:"
ls -lh "${OUT_DIR}/${PREFIX}"*.pickle
