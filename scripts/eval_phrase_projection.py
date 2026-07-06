"""Fair phrase-structure comparison for the CHARACTER-level KTB parser.

A character-level MBR parse is a binary tree over N characters, so it emits
~N-2 non-trivial brackets, most of them *inside* a morpheme. Comparing those
raw char-space spans against the morpheme baseline UF1 is unfair: the
sub-morpheme brackets count as false positives and the index spaces /
denominators differ (see plan).

This script removes that metric artifact by PROJECTING the char prediction onto
morpheme boundaries: a predicted char span survives only if BOTH endpoints fall
on a morpheme boundary, and is then mapped to morpheme-index space. The
projected spans are scored with the *exact* UF1 logic used at training time
(reused from scripts/verify_f1_from_jsonl.py) against `morph_gold` — the
morpheme-index gold identical to the morpheme baseline. The resulting
projected UF1 is directly comparable to the morpheme-level HN-PCFG.

It also reports, for reference:
  - the raw char-space UF1 (what training prints; NOT comparable to baseline)
  - morpheme-segmentation recall (fraction of multi-char gold morphemes that
    appear as a predicted char constituent) — the direct answer to "did the
    parser recover morpheme boundaries for free?"

Join key: dump_predictions.py writes `sent_id` = raw-pickle index, so
`morph_offsets[sent_id]` / `morph_gold[sent_id]` line up with each JSONL record.

Usage:
    python scripts/eval_phrase_projection.py \
        --jsonl analysis/ktb_char/preds_test.jsonl \
        --char_pickle data/clean/japanese-ktb-char-test.pickle \
        --label HN-PCFG-char \
        --out analysis/ktb_char/projection_report.txt
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

# Reuse the bit-for-bit UF1 helpers from the sibling verifier.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_f1_from_jsonl import (  # noqa: E402
    EPS,
    EXPECTED,
    _filter_spans,
    compute_f1_from_jsonl,
)


def compute_projected_f1(jsonl_path: Path, char_pickle: Path) -> dict:
    """Project char predictions to morpheme boundaries and score vs morph_gold.

    Returns projected phrase-structure UF1 (comparable to the morpheme baseline)
    plus morpheme-segmentation recall.
    """
    with open(char_pickle, "rb") as f:
        raw = pickle.load(f)
    morph_offsets = raw["morph_offsets"]
    morph_gold = raw["morph_gold"]

    tp = fp = fn = 0.0
    sentence_f1_sum = 0.0
    n_used = 0
    n_total = 0
    seg_recovered = 0
    seg_total = 0

    with open(jsonl_path) as fh:
        for line in fh:
            rec = json.loads(line)
            n_total += 1
            sid = rec["sent_id"]
            off = morph_offsets[sid]
            mgold = morph_gold[sid]
            pred_char = [(s, e) for s, e in rec["pred_spans"]]
            if len(pred_char) == 0:
                continue

            bset = set(off)
            c2m = {c: i for i, c in enumerate(off)}

            # --- project to morpheme-index space ---
            proj = [(c2m[s], c2m[e]) for s, e in pred_char if s in bset and e in bset]
            golds = [(a, b) for a, b, _ in mgold]
            length = max(golds, key=lambda x: x[1])[1] if golds else (len(off) - 1)
            preds_f = _filter_spans(proj, length)
            golds_f = _filter_spans(golds, length)

            for span in preds_f:
                if span in golds_f:
                    tp += 1
                else:
                    fp += 1
            for span in golds_f:
                if span not in preds_f:
                    fn += 1

            p_set, g_set = set(preds_f), set(golds_f)
            overlap = p_set & g_set
            prec = len(overlap) / (len(p_set) + EPS)
            reca = len(overlap) / (len(g_set) + EPS)
            if len(g_set) == 0:
                reca = 1.0
                if len(p_set) == 0:
                    prec = 1.0
            sentence_f1_sum += 2 * prec * reca / (prec + reca + 1e-8)
            n_used += 1

            # --- morpheme-segmentation recall (multi-char morphemes only) ---
            morph_char_spans = [
                (off[i], off[i + 1])
                for i in range(len(off) - 1)
                if off[i + 1] - off[i] >= 2
            ]
            pred_char_set = set(pred_char)
            seg_recovered += sum(1 for m in morph_char_spans if m in pred_char_set)
            seg_total += len(morph_char_spans)

    corpus_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    corpus_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    corpus_f1 = (
        2 * corpus_precision * corpus_recall / (corpus_precision + corpus_recall)
        if (corpus_precision + corpus_recall) > 0
        else 0.0
    )
    return {
        "corpus_f1": corpus_f1,
        "sentence_f1": sentence_f1_sum / n_used if n_used > 0 else 0.0,
        "corpus_precision": corpus_precision,
        "corpus_recall": corpus_recall,
        "n_sentences_used": n_used,
        "n_sentences_total": n_total,
        "seg_recall": seg_recovered / seg_total if seg_total > 0 else 0.0,
        "seg_recovered": seg_recovered,
        "seg_total": seg_total,
    }


def _format_report(label, jsonl_path, char_pickle, proj, raw, baseline) -> str:
    lines = [
        f"=== {label} (character-level KTB) ===",
        f"jsonl       : {jsonl_path}",
        f"char_pickle : {char_pickle}",
        f"sentences total / used: {proj['n_sentences_total']} / {proj['n_sentences_used']}",
        "",
        "-- projected phrase-structure UF1 (morpheme-index space; COMPARABLE to baseline) --",
        f"  corpus_f1     = {proj['corpus_f1']:.4f}",
        f"  sentence_f1   = {proj['sentence_f1']:.4f}",
        f"  corpus_prec   = {proj['corpus_precision']:.4f}",
        f"  corpus_recall = {proj['corpus_recall']:.4f}",
        "",
        "-- raw char-space UF1 (what training prints; NOT comparable to baseline) --",
        f"  corpus_f1     = {raw['corpus_f1']:.4f}",
        f"  sentence_f1   = {raw['sentence_f1']:.4f}",
        "",
        "-- morpheme-segmentation recall (multi-char morphemes recovered as constituents) --",
        f"  seg_recall    = {proj['seg_recall']:.4f}  ({proj['seg_recovered']}/{proj['seg_total']})",
    ]
    if baseline is not None:
        lines += [
            "",
            "-- morpheme baseline (reference) --",
            f"  baseline corpus_f1   = {baseline['corpus_f1']:.4f}",
            f"  baseline sentence_f1 = {baseline['sentence_f1']:.4f}",
            f"  Δ corpus_f1   (char_projected - baseline) = {proj['corpus_f1'] - baseline['corpus_f1']:+.4f}",
            f"  Δ sentence_f1 (char_projected - baseline) = {proj['sentence_f1'] - baseline['sentence_f1']:+.4f}",
        ]
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(
        description="Projected (fair) phrase-structure UF1 for char-level KTB predictions."
    )
    p.add_argument("--jsonl", required=True, help="char predictions JSONL (dump_predictions.py).")
    p.add_argument("--char_pickle", required=True,
                   help="japanese-ktb-char-test.pickle (carries morph_offsets/morph_gold).")
    p.add_argument("--label", default="HN-PCFG-char")
    p.add_argument("--baseline_label", default="eji18kkl",
                   help="key into verify_f1_from_jsonl.EXPECTED for the morpheme baseline "
                        "reference (default: eji18kkl = HN-PCFG morpheme KTB JA).")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    jsonl_path = Path(args.jsonl).resolve()
    char_pickle = Path(args.char_pickle).resolve()
    if not jsonl_path.exists():
        sys.exit(f"jsonl not found: {jsonl_path}")
    if not char_pickle.exists():
        sys.exit(f"char_pickle not found: {char_pickle}")

    proj = compute_projected_f1(jsonl_path, char_pickle)
    raw = compute_f1_from_jsonl(jsonl_path)
    baseline = EXPECTED.get(args.baseline_label)

    report = _format_report(args.label, jsonl_path, char_pickle, proj, raw, baseline)
    print(report)

    if args.out is not None:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report + "\n")
        print(f"\n[info] wrote report to {out_path}")


if __name__ == "__main__":
    main()
