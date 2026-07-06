"""Re-compute test/corpus_f1 and test/sentence_f1 from a predictions JSONL.

Reproduces the UF1 metric used at training time
(parser/helper/metric.py:UF1) bit-for-bit. hol-pcfg's evaluate path
(parser/cmds/cmd.py:51) calls `metric_f1(result['prediction'], y['gold_tree'])`
in normal order (preds first, golds second) — NO argument swap like in the
vendored `parsing_by_maxseminfo` harness. We replicate that exact call ordering here.

Usage:
    python scripts/verify_f1_from_jsonl.py \\
        --jsonl analysis/error_analysis/HN-PCFG/predictions_test.jsonl \\
        --label HN-PCFG \\
        --jsonl analysis/error_analysis/SN-PCFG/predictions_test.jsonl \\
        --label SN-PCFG \\
        --out analysis/error_analysis/sanity_check.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EPS = 1e-8


def _filter_spans(spans, length):
    """Mirror UF1's per-span filter: drop trivial (len <= 1) and full-sentence spans."""
    out = []
    for x in spans:
        if x[0] + 1 == x[1] or x[0] == x[1]:
            continue
        if x[0] == 0 and x[1] == length:
            continue
        out.append(x)
    return out


def compute_f1_from_jsonl(jsonl_path: Path) -> dict[str, float]:
    """Re-compute UF1 with the **normal** argument order (preds, golds).

    Returns dict with keys: corpus_f1, sentence_f1, corpus_precision,
    corpus_recall, n_sentences_used, n_sentences_total.
    """
    tp = 0.0
    fp = 0.0
    fn = 0.0
    sentence_f1_sum = 0.0
    n_used = 0.0
    n_total = 0
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            n_total += 1

            preds = [(s, e) for s, e in rec["pred_spans"]]
            golds = [(s, e) for s, e, _ in rec["gold_spans"]]

            if len(preds) == 0:
                continue

            length = max(golds, key=lambda x: x[1])[1] if golds else rec["length"]
            preds_f = _filter_spans(preds, length)
            golds_f = _filter_spans(golds, length)

            for span in preds_f:
                if span in golds_f:
                    tp += 1
                else:
                    fp += 1
            for span in golds_f:
                if span not in preds_f:
                    fn += 1

            p_set = set(preds_f)
            g_set = set(golds_f)
            overlap = p_set & g_set
            prec = float(len(overlap)) / (len(p_set) + EPS)
            reca = float(len(overlap)) / (len(g_set) + EPS)
            if len(g_set) == 0:
                reca = 1.0
                if len(p_set) == 0:
                    prec = 1.0
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            sentence_f1_sum += f1
            n_used += 1

    sentence_f1 = sentence_f1_sum / n_used if n_used > 0 else 0.0
    corpus_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    corpus_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if corpus_precision + corpus_recall > 0:
        corpus_f1 = 2 * corpus_precision * corpus_recall / (corpus_precision + corpus_recall)
    else:
        corpus_f1 = 0.0

    return {
        "corpus_f1": corpus_f1,
        "sentence_f1": sentence_f1,
        "corpus_precision": corpus_precision,
        "corpus_recall": corpus_recall,
        "n_sentences_used": int(n_used),
        "n_sentences_total": n_total,
    }


def _format_results(label, jsonl_path, m, expected):
    lines = [
        f"=== {label} ===",
        f"jsonl: {jsonl_path}",
        f"sentences total / used: {m['n_sentences_total']} / {m['n_sentences_used']}",
        f"corpus_f1     = {m['corpus_f1']:.4f}",
        f"sentence_f1   = {m['sentence_f1']:.4f}",
        f"corpus_prec   = {m['corpus_precision']:.4f}",
        f"corpus_recall = {m['corpus_recall']:.4f}",
    ]
    if expected is not None:
        for key in ("corpus_f1", "sentence_f1"):
            if key in expected:
                diff = m[key] - expected[key]
                tol = 0.005
                ok = abs(diff) <= tol
                tag = "OK" if ok else "MISMATCH"
                lines.append(
                    f"expected {key} = {expected[key]:.4f}  diff = {diff:+.4f}  [{tag}, tol=±{tol}]"
                )
    return "\n".join(lines)


# Hard-coded expectations for the two W&B runs.
#
# eji18kkl: ckpt saved at val_LL best (epoch 25). val/sentence_f1 at that ckpt
# (= W&B's `best/f1_at_best_ll`) = 0.6519. test/* was not logged to W&B
# summary for this run, so the first eval-only run populates it here.
#
# 01cxq9fl: ckpt similarly saved at val_LL best (epoch 24). test/* not in
# W&B summary either; first eval-only run defines the value.
EXPECTED = {
    "eji18kkl": {"corpus_f1": 0.6258, "sentence_f1": 0.6476},
    "01cxq9fl": {"corpus_f1": 0.6302, "sentence_f1": 0.6519},
    "HN-PCFG": {"corpus_f1": 0.6258, "sentence_f1": 0.6476},
    "SN-PCFG": {"corpus_f1": 0.6302, "sentence_f1": 0.6519},
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Re-compute UF1 from predictions JSONL produced by dump_predictions.py"
    )
    p.add_argument("--jsonl", required=True, action="append")
    p.add_argument("--label", required=True, action="append")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if len(args.jsonl) != len(args.label):
        sys.exit("--jsonl and --label must be passed the same number of times")

    blocks = []
    for jsonl, label in zip(args.jsonl, args.label):
        path = Path(jsonl).resolve()
        if not path.exists():
            sys.exit(f"jsonl not found: {path}")
        m = compute_f1_from_jsonl(path)
        expected = EXPECTED.get(label) or None
        block = _format_results(label, path, m, expected)
        print(block)
        print()
        blocks.append(block)

    if args.out is not None:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\n\n".join(blocks) + "\n")
        print(f"[info] wrote report to {out_path}")


if __name__ == "__main__":
    main()
