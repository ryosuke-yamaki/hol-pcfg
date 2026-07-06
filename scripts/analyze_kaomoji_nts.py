"""
Analyse which NT (non-terminal) categories the kaomoji-trained HN-PCFG model
assigns to predicted constituents on the validation split.

Pipeline:
  1. Load the model checkpoint + val pickle.
  2. Run MBR + per-span NT argmax across the full val split.
  3. For each non-root, non-trivial predicted span (i, j), record:
       (NT_id, span_length, leaf_substring_unescaped)
  4. Aggregate by NT_id, list the top-K most frequent ones along with:
       - frequency, mean span length
       - example covered substrings (up to ~12 samples, deduplicated)

Output goes to stdout as plain text (also writable via --out).

Usage:
  python scripts/analyze_kaomoji_nts.py \
      --conf  config/formal/hnpcfg_kaomoji_main.yaml \
      --ckpt  log/hnpcfg_kaomoji_main_seed0/HNPCFG*/best.pt \
      --pickle data/clean/kaomoji-val.pickle \
      --raw-text data/raw/kaomoji-val.txt \
      --top-k 30
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pickle
import yaml
import torch
from easydict import EasyDict as edict


_UNESCAPE = {"-LRB-": "(", "-RRB-": ")", "_": " "}


def display_token(tok: str) -> str:
    return _UNESCAPE.get(tok, tok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--pickle", required=True, type=Path)
    ap.add_argument("--raw-text", type=Path, default=None,
                    help="optional: raw kaomoji file (one per line) for reference; "
                         "if absent, displayed substrings are reconstructed from leaves")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--max-examples", type=int, default=12,
                    help="max unique example substrings to show per NT")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional output file (else stdout)")
    args = ap.parse_args()

    out_f = args.out.open("w", encoding="utf-8") if args.out else sys.stdout

    def emit(line: str = "") -> None:
        print(line, file=out_f)

    # Load config + model
    with args.conf.open() as f:
        cfg = edict(yaml.load(f, Loader=yaml.Loader))
    cfg.device = args.device if torch.cuda.is_available() else "cpu"
    cfg.data.test_file = str(args.pickle)

    from parser.helper.data_module import DataModule
    from parser.helper.util import get_model

    dataset = DataModule(cfg)
    model = get_model(cfg.model, dataset)
    state = torch.load(str(args.ckpt), map_location=cfg.device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(cfg.device).eval()

    with args.pickle.open("rb") as f:
        raw = pickle.load(f)
    raw_words = raw["word"]
    kept_to_raw = [i for i, w in enumerate(raw_words) if len(w) > 1]
    test_dataset = dataset.test_dataset
    sampler = dataset.test_dataloader.batch_sampler

    raw_texts: list[str] | None = None
    if args.raw_text is not None:
        with args.raw_text.open("r", encoding="utf-8") as f:
            raw_texts = [line.rstrip("\n") for line in f]

    # nt_id -> list of (length, substring, raw_idx)
    by_nt: dict[int, list[tuple[int, str, int]]] = defaultdict(list)
    nt_count: Counter = Counter()
    total_spans = 0

    for batch_indices in sampler.total:
        lens = [test_dataset[i]["seq_len"] for i in batch_indices]
        word_ids = torch.tensor(
            [test_dataset[i]["word"] for i in batch_indices], device=cfg.device,
        )
        seq_len = torch.tensor(lens, device=cfg.device)
        outputs = model.evaluate({"word": word_ids, "seq_len": seq_len}, decode_type="mbr")
        preds = outputs["prediction"]
        nt_label_batch = outputs.get("nt_labels", [{}] * len(batch_indices))

        for k, kept_idx in enumerate(batch_indices):
            raw_idx = kept_to_raw[kept_idx]
            leaves = list(raw_words[raw_idx])
            n = len(leaves)
            span_to_nt = nt_label_batch[k] if nt_label_batch[k] is not None else {}
            for (i, j) in preds[k]:
                i, j = int(i), int(j)
                if j - i < 2 or (i == 0 and j == n):
                    continue
                nt_id = span_to_nt.get((i, j))
                if nt_id is None:
                    continue
                # substring of leaves[i:j], displayed with escapes reversed
                substr = "".join(display_token(tok) for tok in leaves[i:j])
                by_nt[int(nt_id)].append((j - i, substr, raw_idx))
                nt_count[int(nt_id)] += 1
                total_spans += 1

    emit(f"=== Kaomoji NT usage analysis ===")
    emit(f"val sentences scored: {len(kept_to_raw)}")
    emit(f"total non-trivial, non-root predicted spans: {total_spans}")
    emit(f"unique NTs used: {len(nt_count)}  (out of NT={cfg.model.NT})")
    emit()
    emit(f"=== Top {args.top_k} NTs by frequency ===")
    emit()

    for rank, (nt_id, cnt) in enumerate(nt_count.most_common(args.top_k), 1):
        rows = by_nt[nt_id]
        lengths = [r[0] for r in rows]
        mean_len = sum(lengths) / len(lengths)
        # Deduplicate examples by substring; show up to max_examples diverse ones
        seen: set[str] = set()
        examples: list[str] = []
        for L, s, _ in rows:
            if s in seen:
                continue
            seen.add(s)
            examples.append(s)
            if len(examples) >= args.max_examples:
                break
        emit(f"--- Rank {rank}: NT={nt_id}  count={cnt}  "
             f"mean_len={mean_len:.2f}  len_range=[{min(lengths)}, {max(lengths)}]  "
             f"unique_subs={len(set(s for _, s, _ in rows))} ---")
        for ex in examples:
            emit(f"    {ex!r}")
        emit()

    if args.out:
        out_f.close()
        print(f"[info] wrote analysis to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
