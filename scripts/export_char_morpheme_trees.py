"""Project a CHARACTER-level HN-PCFG's predictions onto morpheme boundaries and
export MORPHEME-level bracketed trees, so Li et al.'s evaluate.py (Evalb F1) can
score them on exactly the same footing as the morpheme baseline (the 5-seed
`test Evalb F1 (all) = 59.45` number).

Why: a char binary tree's raw spans are not comparable to morpheme-level gold
(sub-morpheme brackets inflate the denominator, different index space). We keep
only predicted char spans whose BOTH endpoints are morpheme boundaries, map them
to morpheme indices, rebuild a morpheme-level tree, and write it next to the
gold morpheme tree. The repo's internal UF1 is also a different metric than the
Evalb F1 reported in the 5-seed summary, so this script exists to produce the
Evalb-comparable trees.

Inputs:
  - char predictions JSONL from dump_predictions.py (sent_id = raw char-pickle
    index, pred_spans = char binary-tree spans)
  - the char test pickle (carries morph_offsets) — for char->morpheme projection
  - the morpheme gold bracketed file (ktb_len40_nopunct/test) — the SAME gold the
    morpheme baseline is scored against

Alignment: the char builder drops single-morpheme sentences (factorize empty
<=> len(tree.pos()) < 2), which is exactly predict_and_export_trees.py's `n < 2`
skip. So the gold lines with >=2 morphemes are parallel, in order, to the char
pickle / sent_id.

Usage:
  python scripts/export_char_morpheme_trees.py \
    --jsonl analysis/ktb_char/preds_test.jsonl \
    --char_pickle data/clean/japanese-ktb-char-test.pickle \
    --gold_text_file .ktb_workspace/UnsupConstParseEval/data/cleaned_datasets/japanese/ktb/ktb_len40_nopunct/test \
    --pred_out analysis/ktb_char/evalb_pred.txt \
    --gold_out analysis/ktb_char/evalb_gold.txt
then:
  cd .ktb_workspace/UnsupConstParseEval && python evaluate.py -g <gold_out> -p <pred_out>
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from nltk import Tree


def _esc(w: str) -> str:
    return w.replace("(", "-LRB-").replace(")", "-RRB-")


def build_morph_tree(proj_spans, M, leaves, label="S") -> str:
    """Build a morpheme-level bracketed tree whose internal nodes are exactly the
    (laminar) projected morpheme spans. Leaves and root are always present."""
    spanset = {(int(a), int(b)) for a, b in proj_spans}
    for i in range(M):
        spanset.add((i, i + 1))
    spanset.add((0, M))

    def build(a, b):
        if b - a == 1:
            return f"({label} {_esc(leaves[a])})"
        parts = []
        i = a
        while i < b:
            k = i + 1
            for kk in range(b, i, -1):
                if (i, kk) in spanset and not (i == a and kk == b):
                    k = kk
                    break
            parts.append(build(i, k))
            i = k
        return f"({label} " + " ".join(parts) + ")"

    return build(0, M)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, help="char predictions JSONL (dump_predictions.py)")
    p.add_argument("--char_pickle", required=True, help="japanese-ktb-char-test.pickle")
    p.add_argument("--gold_text_file", required=True, help="ktb_len40_nopunct/test bracketed gold")
    p.add_argument("--pred_out", required=True)
    p.add_argument("--gold_out", required=True)
    args = p.parse_args()

    with open(args.char_pickle, "rb") as f:
        raw = pickle.load(f)
    morph_offsets = raw["morph_offsets"]
    n_sents = len(morph_offsets)

    # Gold lines with >=2 morphemes are parallel to the char pickle (same drop rule).
    gold_kept = []
    with open(args.gold_text_file, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if len(Tree.fromstring(ln).pos()) >= 2:
                gold_kept.append(ln)
    assert len(gold_kept) == n_sents, (
        f"gold-kept lines {len(gold_kept)} != char pickle sents {n_sents}; "
        "alignment broken"
    )

    preds = {}
    with open(args.jsonl) as f:
        for line in f:
            r = json.loads(line)
            preds[r["sent_id"]] = [(s, e) for s, e in r["pred_spans"]]

    n_written = 0
    with open(args.pred_out, "w", encoding="utf-8") as fpred, \
         open(args.gold_out, "w", encoding="utf-8") as fgold:
        for sid in range(n_sents):
            off = morph_offsets[sid]
            M = len(off) - 1
            gold_line = gold_kept[sid]
            gold_leaves = Tree.fromstring(gold_line).leaves()
            assert len(gold_leaves) == M, (
                f"sent {sid}: gold leaves {len(gold_leaves)} != morphemes {M}"
            )
            pred_char = preds.get(sid, [])
            bset = set(off)
            c2m = {c: i for i, c in enumerate(off)}
            proj = {(c2m[s], c2m[e]) for s, e in pred_char if s in bset and e in bset}
            pred_tree = build_morph_tree(proj, M, gold_leaves, label="S")
            fpred.write(pred_tree + "\n")
            fgold.write(gold_line + "\n")
            n_written += 1

    print(f"wrote {n_written} pred/gold morpheme trees")
    print(f"pred: {args.pred_out}")
    print(f"gold: {args.gold_out}")


if __name__ == "__main__":
    main()
