#!/usr/bin/env bash
# Build CHARACTER-level KTB Japanese pickles for HN-PCFG.
#
# Identical to scripts/prepare_ktb_data.sh through the Li et al. cleanup
# (steps 1-3, idempotent: reuses an existing .ktb_workspace), then converts the
# SAME cleaned bracket files to character-level pickles via
# scripts/preprocess_ktb_char.py instead of preprocessing.py. Each morpheme is
# split into characters so the unsupervised parser must induce morphological
# segmentation; placeholders ([num], [unk], -LRB-, -RRB-) stay atomic.
#
# Usage:
#   bash scripts/prepare_ktb_char_data.sh [WORKDIR]
# Defaults: WORKDIR=$PWD/.ktb_workspace  (shared with prepare_ktb_data.sh)

set -euo pipefail

WORKDIR="${1:-$PWD/.ktb_workspace}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/data/clean"
PREFIX="japanese-ktb-char-"

mkdir -p "${WORKDIR}" "${OUT_DIR}"

# 1. Keyaki Treebank v1.1
if [ ! -d "${WORKDIR}/KeyakiTreebank" ]; then
    echo "[prepare_ktb_char] cloning Keyaki Treebank ..."
    git -C "${WORKDIR}" clone --depth 1 https://github.com/ajb129/KeyakiTreebank.git
fi

# 2. Li et al. preprocessor (de facto standard; reused unchanged)
if [ ! -d "${WORKDIR}/UnsupConstParseEval" ]; then
    echo "[prepare_ktb_char] cloning UnsupConstParseEval (Li et al., 2020) ..."
    git -C "${WORKDIR}" clone --depth 1 https://github.com/i-lijun/UnsupConstParseEval.git
fi

# 3. Run Li et al. preprocessing -> bracketed text splits (morpheme leaves)
CLEAN_ROOT="${WORKDIR}/UnsupConstParseEval/data/cleaned_datasets/japanese/ktb"
LEN40_NOPUNCT="${CLEAN_ROOT}/ktb_len40_nopunct"
if [ ! -f "${LEN40_NOPUNCT}/train" ] || [ ! -f "${LEN40_NOPUNCT}/dev" ] || [ ! -f "${LEN40_NOPUNCT}/test" ]; then
    echo "[prepare_ktb_char] running Li et al. preprocess ..."
    pushd "${WORKDIR}/UnsupConstParseEval" >/dev/null
    mkdir -p data/japanese
    ln -sfn "${WORKDIR}/KeyakiTreebank" data/japanese/KeyakiTreebank
    python preprocess.py \
        --path_to_raw_ptb "" \
        --path_to_raw_ktb data/japanese/KeyakiTreebank/treebank \
        --path_to_ktb_output_dir data/cleaned_datasets/japanese/ktb
    popd >/dev/null
fi

# 4. Bracketed (morpheme) text -> CHARACTER-level pickles
echo "[prepare_ktb_char] building character-level pickles in ${OUT_DIR} ..."
cd "${REPO_ROOT}"
python scripts/preprocess_ktb_char.py \
    --train_file "${LEN40_NOPUNCT}/train" \
    --val_file "${LEN40_NOPUNCT}/dev" \
    --test_file "${LEN40_NOPUNCT}/test" \
    --cache_path "${OUT_DIR}/${PREFIX}"

echo "[prepare_ktb_char] done. Generated:"
ls -lh "${OUT_DIR}/${PREFIX}"*.pickle
