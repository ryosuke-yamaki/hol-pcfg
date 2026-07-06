#!/usr/bin/env bash
# Acquire raw upstream data for the formal-language pilot (symmath).
# Run from the repo root.
#
# Datasets and licenses:
#   - Lample symbolic-math (prim_fwd):
#       URL:     https://dl.fbaipublicfiles.com/SymbolicMathematics/data/prim_fwd.tar.gz
#       License: CC BY-NC 4.0 (research use only)
#       Cite:    Lample & Charton, "Deep Learning for Symbolic Mathematics" (ICLR 2020)
#
# Usage:
#   bash scripts/download_formal_data.sh
set -euo pipefail

mkdir -p data/raw

LAMPLE_URL="https://dl.fbaipublicfiles.com/SymbolicMathematics/data/prim_fwd.tar.gz"
LAMPLE_RANGE_BYTES="${LAMPLE_RANGE_BYTES:-31457280}"  # 30 MiB streamed prefix; sufficient for ~370k expressions

if [[ ! -s data/raw/prim_fwd.train ]]; then
  echo "[lample] fetching first ${LAMPLE_RANGE_BYTES} bytes of prim_fwd.tar.gz"
  TMP_GZ=$(mktemp --suffix=.tar.gz)
  curl -sL --max-time 120 -r "0-${LAMPLE_RANGE_BYTES}" "${LAMPLE_URL}" -o "${TMP_GZ}"
  EXTRACT_DIR=$(mktemp -d)
  # gzip can decompress as much as the truncated stream allows; tar will warn at EOF.
  gzip -cd "${TMP_GZ}" 2>/dev/null | tar -xf - -C "${EXTRACT_DIR}" --warning=none || true
  mv "${EXTRACT_DIR}/prim_fwd.train" data/raw/
  rm -f "${TMP_GZ}"
  rm -rf "${EXTRACT_DIR}"
  echo "[lample] wrote data/raw/prim_fwd.train ($(wc -l < data/raw/prim_fwd.train) lines)"
else
  echo "[lample] data/raw/prim_fwd.train already exists, skipping"
fi

echo "done."
