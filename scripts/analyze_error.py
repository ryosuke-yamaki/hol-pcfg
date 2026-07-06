"""Phase B error analysis: compare treatment vs baseline predictions.

Reads two predictions JSONL files (one per run) produced by
scripts/dump_predictions.py and emits three reports under --out-dir:

  B-1 lowf1_cases.md       — low-F1 case study (4 categories)
  B-2 constituent_recall.md — recall by gold constituent label (NP/VP/...)
  B-3 aux_stats.md / .csv   — F1 by sentence length, recall by span length

UF1 reproduction:
  Span filtering and metric arithmetic mirror parser/helper/metric.py:UF1.
  hol-pcfg's evaluate path (parser/cmds/cmd.py:51) calls
  `metric_f1(result['prediction'], y['gold_tree'])` in **normal** order
  (preds first, golds second) — UNLIKE the vendored `parsing_by_maxseminfo`
  harness, which swaps them in test_step. We faithfully use normal order below.

Usage:
    python scripts/analyze_error.py \\
        --treatment-jsonl analysis/error_analysis/HN-PCFG/predictions_test.jsonl \\
        --baseline-jsonl  analysis/error_analysis/SN-PCFG/predictions_test.jsonl \\
        --treatment-name HN-PCFG \\
        --baseline-name  SN-PCFG \\
        --out-dir analysis/error_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


EPS = 1e-8

# Constituent labels reported in Sem-Info paper Table 7 (PTB defaults).
# Override via --label-rows for treebanks with different label sets
# (e.g. KTB JA: PP, IP, NP, CP, CONJP, ADVP).
LABEL_ROWS: list[str] = ["NP", "VP", "PP", "SBAR", "ADJP", "ADVP"]


def _eval_only_caveat() -> list[str]:
    """Note about checkpoint selection (val-LL-best, not val-F1-best).

    hol-pcfg's run_single_train.py saves best.pt when val likelihood improves,
    NOT when val/sentence_f1 improves. So the ckpt's `val/sentence_f1`
    corresponds to W&B's `best/f1_at_best_ll` rather than the max
    `val/sentence_f1` (= `best/f1_overall`) that may have occurred at a
    different (non-best-LL) epoch.
    """
    return [
        "> **Note on checkpoint selection.** hol-pcfg saves best.pt at the",
        "> epoch with the best **validation log-likelihood**, not the best",
        "> validation F1. So the F1 reproduced from this ckpt equals W&B's",
        "> `best/f1_at_best_ll`, which can differ from `best/f1_overall`",
        "> (the max val F1 across training, possibly at a different epoch).",
        "> Both runs are dumped through the same pipeline, so the Δ",
        "> (treatment − baseline) values remain comparable.",
        "",
    ]

# B-1 sample / pruning thresholds.
MIN_LENGTH = 10
MIN_GOLD_SPANS = 3
TOP_K = 20
CAT_D_FALLBACK_LIMIT = 50


# ------------------------------------------------------------------------
# UF1-compatible F1
# ------------------------------------------------------------------------


def _filter_spans(spans: Iterable[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    """Mirror metric.py:117-121 (drop trivial / sentence spans)."""
    out: list[tuple[int, int]] = []
    for s, e in spans:
        if s + 1 == e or s == e:
            continue
        if s == 0 and e == length:
            continue
        out.append((s, e))
    return out


def per_sentence_metrics(gold_spans_labeled: list[list],
                         pred_spans: list[list]) -> dict[str, float | int]:
    """Return F1 / precision / recall / tp / fp / fn for one sentence.

    Uses **normal** argument order (preds, golds) matching hol-pcfg's
    parser/cmds/cmd.py:51 `metric_f1(result['prediction'], y['gold_tree'])`.
    """
    preds = [(int(s), int(e)) for s, e in pred_spans]
    golds = [(int(s), int(e)) for s, e, _ in gold_spans_labeled]

    if len(preds) == 0:
        return {"f1": 0.0, "prec": 0.0, "reca": 0.0,
                "tp": 0, "fp": 0, "fn": 0, "skipped": True,
                "n_gold_filtered": 0, "n_pred_filtered": 0}
    if len(golds) == 0:
        return {"f1": 0.0, "prec": 0.0, "reca": 0.0,
                "tp": 0, "fp": 0, "fn": 0, "skipped": True,
                "n_gold_filtered": 0, "n_pred_filtered": 0}

    length = max(golds, key=lambda x: x[1])[1]
    preds_f = _filter_spans(preds, length)
    golds_f = _filter_spans(golds, length)

    tp = fp = fn = 0
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

    return {
        "f1": f1, "prec": prec, "reca": reca,
        "tp": tp, "fp": fp, "fn": fn, "skipped": False,
        "n_gold_filtered": len(golds_f),
        "n_pred_filtered": len(preds_f),
    }


# ------------------------------------------------------------------------
# JSONL loading and merging
# ------------------------------------------------------------------------


def load_jsonl(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            sid = int(rec["sent_id"])
            if sid in out:
                raise ValueError(f"duplicate sent_id={sid} in {path}")
            out[sid] = rec
    return out


def attach_metrics(records: dict[int, dict]) -> None:
    """Add per-sentence metrics into each record under key 'm'."""
    for rec in records.values():
        rec["m"] = per_sentence_metrics(rec["gold_spans"], rec["pred_spans"])


def merge_two(treatment: dict[int, dict], baseline: dict[int, dict]) -> list[dict]:
    """Return list of dicts containing both runs' info for shared sent_ids."""
    common = sorted(set(treatment) & set(baseline))
    merged = []
    for sid in common:
        t = treatment[sid]
        b = baseline[sid]
        merged.append({
            "sent_id": sid,
            "words": t["words"],
            "length": t["length"],
            "gold_spans": t["gold_spans"],
            "treatment_pred": t["pred_spans"],
            "baseline_pred": b["pred_spans"],
            "treatment_m": t["m"],
            "baseline_m": b["m"],
        })
    return merged


# ------------------------------------------------------------------------
# Bracketed tree rendering
# ------------------------------------------------------------------------


def _build_tree(spans: list[tuple[int, int, str]], length: int) -> list[tuple[int, int, str]]:
    """Inject leaf spans and the sentence span; sort outermost first.

    Spans are kept as-is (no nesting reconciliation); rendering walks the
    span list top-down. Each input span is (s, e, label) with `label`
    potentially empty (predicted spans).
    """
    out: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for s, e, lbl in spans:
        if (s, e) not in seen:
            out.append((s, e, lbl))
            seen.add((s, e))
    if (0, length) not in seen and length > 0:
        out.append((0, length, "ROOT"))
    for i in range(length):
        if (i, i + 1) not in seen:
            out.append((i, i + 1, ""))
    # Sort: longer spans first, ties broken by left-to-right.
    out.sort(key=lambda t: (-(t[1] - t[0]), t[0]))
    return out


def render_bracket(words: list[str], spans_with_label: list[tuple[int, int, str]]) -> str:
    """Render a bracket-style tree from word list + (s,e,label) spans.

    The renderer assumes the span set is laminar (well-nested). When the
    set is not perfectly laminar (which can happen for predicted spans
    that conflict), the function falls back to a flat rendering.
    """
    length = len(words)
    augmented = _build_tree(spans_with_label, length)
    # Group children: for each span, find immediate sub-spans (largest
    # contiguous spans inside it that don't overlap each other).
    span_set = augmented
    span_set_sorted = sorted(span_set, key=lambda t: (t[0], -(t[1] - t[0])))

    # Build a parent->children map by greedy laminar matching.
    children_map: dict[int, list[int]] = defaultdict(list)
    parent_of: dict[int, int | None] = {}

    # We index spans by id (position in span_set_sorted).
    for i, (s, e, _) in enumerate(span_set_sorted):
        parent_of[i] = None
        # Find the smallest enclosing span among earlier (larger or equal) spans.
        best_j = None
        best_len = None
        for j, (s2, e2, _) in enumerate(span_set_sorted):
            if j == i:
                continue
            if s2 <= s and e <= e2 and (s2, e2) != (s, e):
                ln = e2 - s2
                if best_len is None or ln < best_len:
                    best_len = ln
                    best_j = j
        parent_of[i] = best_j
        if best_j is not None:
            children_map[best_j].append(i)

    # Identify the outermost (parent==None) spans, render each.
    roots = [i for i, p in parent_of.items() if p is None]
    roots.sort(key=lambda i: span_set_sorted[i][0])

    def render(i: int) -> str:
        s, e, lbl = span_set_sorted[i]
        if e - s == 1:
            # leaf
            return words[s] if s < len(words) else "?"
        kids = sorted(children_map[i], key=lambda j: span_set_sorted[j][0])
        # If there are no kids, walk word-by-word.
        if not kids:
            inner = " ".join(words[s:e])
        else:
            # Stitch together kids; words not covered are emitted as bare leaves.
            parts = []
            cursor = s
            for kj in kids:
                ks, ke, _ = span_set_sorted[kj]
                if ks < cursor:
                    # overlap with previous child; skip to keep rendering safe.
                    continue
                # words between cursor and ks are bare leaves
                while cursor < ks:
                    parts.append(words[cursor])
                    cursor += 1
                parts.append(render(kj))
                cursor = max(cursor, ke)
            while cursor < e:
                parts.append(words[cursor])
                cursor += 1
            inner = " ".join(parts)
        if not lbl:
            tag = "*X*"
        else:
            tag = lbl
        return f"({tag} {inner})"

    return " ".join(render(i) for i in roots)


# ------------------------------------------------------------------------
# B-1: low-F1 case study
# ------------------------------------------------------------------------


def passes_b1_filter(rec: dict) -> bool:
    if rec["length"] < MIN_LENGTH:
        return False
    n_gold_nontrivial = sum(
        1 for s, e, _ in rec["gold_spans"]
        if not (s + 1 == e or s == e or (s == 0 and e == rec["length"]))
    )
    return n_gold_nontrivial >= MIN_GOLD_SPANS


def _format_case(rec: dict, treatment_name: str, baseline_name: str) -> list[str]:
    tm = rec["treatment_m"]
    bm = rec["baseline_m"]
    delta = tm["f1"] - bm["f1"]

    pred_t = [(int(s), int(e), "") for s, e in rec["treatment_pred"]]
    pred_b = [(int(s), int(e), "") for s, e in rec["baseline_pred"]]
    gold = [(int(s), int(e), str(lbl)) for s, e, lbl in rec["gold_spans"]]

    out = [
        f"### sent_id={rec['sent_id']} (length={rec['length']})",
        f"- F1 ({treatment_name}) = {tm['f1']:.4f}, F1 ({baseline_name}) = {bm['f1']:.4f}, "
        f"Δ = {delta:+.4f}",
        f"- words: {' '.join(rec['words'])}",
        "",
        "Gold:",
        "    " + render_bracket(rec["words"], gold),
        "",
        f"{treatment_name} pred:",
        "    " + render_bracket(rec["words"], pred_t),
        "",
        f"{baseline_name} pred:",
        "    " + render_bracket(rec["words"], pred_b),
        "",
    ]
    return out


def emit_b1(merged: list[dict], out_path: Path,
            treatment_name: str, baseline_name: str,
            cat_d_high: float, cat_d_low: float) -> dict:
    eligible = [r for r in merged if passes_b1_filter(r)]

    # (a) treatment F1 ascending bottom-K
    cat_a = sorted(eligible, key=lambda r: r["treatment_m"]["f1"])[:TOP_K]
    # (b) Delta ascending bottom-K (degradations: treatment - baseline most negative)
    cat_b = sorted(eligible, key=lambda r: r["treatment_m"]["f1"] - r["baseline_m"]["f1"])[:TOP_K]
    # (c) Delta descending top-K (improvements)
    cat_c = sorted(eligible, key=lambda r: -(r["treatment_m"]["f1"] - r["baseline_m"]["f1"]))[:TOP_K]
    # (d) baseline F1 >= cat_d_high AND treatment F1 <= cat_d_low
    cat_d_all = [r for r in eligible
                 if r["baseline_m"]["f1"] >= cat_d_high and r["treatment_m"]["f1"] <= cat_d_low]
    cat_d_all_sorted = sorted(cat_d_all, key=lambda r: r["treatment_m"]["f1"])
    cat_d = cat_d_all_sorted[:CAT_D_FALLBACK_LIMIT]

    n_warn = []
    for name, items in [("(a)", cat_a), ("(b)", cat_b), ("(c)", cat_c), ("(d)", cat_d)]:
        if len(items) < 5:
            n_warn.append(f"{name} only {len(items)} samples")

    sections: list[str] = []
    sections.append(f"# Phase B-1: Low-F1 Case Study\n")
    sections.append(f"- Treatment: **{treatment_name}**, baseline: **{baseline_name}**")
    sections.append(f"- Filter: length >= {MIN_LENGTH} AND non-trivial gold spans >= {MIN_GOLD_SPANS}")
    sections.append(f"- Eligible sentences: {len(eligible)} / {len(merged)}")
    if n_warn:
        sections.append("- WARN: " + "; ".join(n_warn))
    sections.append("")

    def _section_block(title: str, items: list[dict], extra: str = "") -> list[str]:
        lines = [f"## {title}"]
        if extra:
            lines.append(extra)
        if not items:
            lines.append("(no samples meeting the criteria)")
            lines.append("")
            return lines
        for r in items:
            lines.extend(_format_case(r, treatment_name, baseline_name))
        return lines

    sections.extend(_section_block(
        f"(a) {treatment_name} hardest sentences (treatment F1 ascending, top {TOP_K})",
        cat_a))
    sections.extend(_section_block(
        f"(b) Largest degradations vs baseline (Δ = T − B ascending, top {TOP_K})",
        cat_b))
    sections.extend(_section_block(
        f"(c) Largest improvements vs baseline (Δ descending, top {TOP_K})",
        cat_c))
    sections.extend(_section_block(
        f"(d) Improvement-hint candidates "
        f"(baseline F1 ≥ {cat_d_high} AND treatment F1 ≤ {cat_d_low})",
        cat_d,
        extra=(
            f"- Total matching: {len(cat_d_all)}; "
            f"showing first {min(len(cat_d_all), CAT_D_FALLBACK_LIMIT)} sorted by treatment F1 ascending."
        ),
    ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections))
    return {
        "eligible": len(eligible),
        "cat_a": len(cat_a),
        "cat_b": len(cat_b),
        "cat_c": len(cat_c),
        "cat_d": len(cat_d),
        "cat_d_total": len(cat_d_all),
        "warnings": n_warn,
    }


# ------------------------------------------------------------------------
# B-2: constituent recall
# ------------------------------------------------------------------------


def _normalize_label(label: str, strip_suffix: bool) -> str:
    """KTB-style labels can carry grammatical-function suffixes after ';'
    (e.g. PP;*SBJ*, NP;*OB1*). With strip_suffix=True we collapse them to
    the base label ("PP", "NP") so the label rows aggregate variants."""
    if strip_suffix and ";" in label:
        return label.split(";", 1)[0]
    return label


def _recall_by_label(records: list[dict], pred_key: str,
                     strip_suffix: bool = False) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """For each gold label, count TP and gold occurrences across all sentences.

    `pred_key` is "treatment_pred" or "baseline_pred".
    Trivial / sentence spans are filtered out per metric.py.
    """
    by_label_tp: dict[str, int] = Counter()
    by_label_total: dict[str, int] = Counter()

    for rec in records:
        length = rec["length"]
        pred_set = set(_filter_spans(((int(s), int(e)) for s, e in rec[pred_key]), length))
        for s, e, label in rec["gold_spans"]:
            s, e = int(s), int(e)
            # apply same filters as UF1 (drop trivial / sentence spans).
            if s + 1 == e or s == e:
                continue
            if s == 0 and e == length:
                continue
            lbl = _normalize_label(label, strip_suffix)
            by_label_total[lbl] += 1
            if (s, e) in pred_set:
                by_label_tp[lbl] += 1
    return ({lbl: {"tp": by_label_tp.get(lbl, 0), "total": by_label_total[lbl]}
             for lbl in by_label_total},
            by_label_total)


def emit_b2(merged: list[dict], out_path: Path,
            treatment_name: str, baseline_name: str,
            label_rows: list[str] | None = None,
            strip_label_suffix: bool = False) -> dict:
    rows_keys = label_rows if label_rows is not None else LABEL_ROWS
    treat_stats, totals = _recall_by_label(merged, "treatment_pred", strip_label_suffix)
    base_stats, _ = _recall_by_label(merged, "baseline_pred", strip_label_suffix)

    rows: list[tuple[str, int, float, float, float]] = []
    overall_tp_t = overall_tp_b = overall_total = 0
    for lbl in rows_keys:
        total = totals.get(lbl, 0)
        if total == 0:
            tr = br = 0.0
        else:
            tp_t = treat_stats.get(lbl, {"tp": 0})["tp"]
            tp_b = base_stats.get(lbl, {"tp": 0})["tp"]
            tr = tp_t / total
            br = tp_b / total
        rows.append((lbl, total, tr, br, tr - br))

    # Overall = sum across all labels (including unlisted ones, full universe).
    for lbl, info in treat_stats.items():
        overall_tp_t += info["tp"]
        overall_tp_b += base_stats.get(lbl, {"tp": 0})["tp"]
        overall_total += info["total"]
    if overall_total == 0:
        overall_tr = overall_br = 0.0
    else:
        overall_tr = overall_tp_t / overall_total
        overall_br = overall_tp_b / overall_total
    rows.append(("Overall", overall_total, overall_tr, overall_br, overall_tr - overall_br))

    md_lines: list[str] = []
    md_lines.append("# Phase B-2: Constituent-type recall (Sem-Info paper Table 7 style)\n")
    md_lines.append(f"- Treatment: **{treatment_name}**, baseline: **{baseline_name}**")
    md_lines.append(f"- Trivial / sentence spans excluded from counts (per UF1 filter).")
    md_lines.append("")
    md_lines.extend(_eval_only_caveat())
    md_lines.append("| Label | gold_count | recall_" + treatment_name + " | recall_" + baseline_name + " | Δ |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for lbl, total, tr, br, delta in rows:
        md_lines.append(
            f"| {lbl} | {total} | {tr:.4f} | {br:.4f} | {delta:+.4f} |"
        )
    md_lines.append("")

    # LaTeX table.
    md_lines.append("## LaTeX")
    md_lines.append("```latex")
    md_lines.append(r"\begin{tabular}{lrrrr}")
    md_lines.append(r"\toprule")
    md_lines.append(
        "Label & Gold & Recall(" + treatment_name + ") & Recall(" + baseline_name + ") & $\\Delta$ \\\\"
    )
    md_lines.append(r"\midrule")
    for lbl, total, tr, br, delta in rows:
        md_lines.append(
            f"{lbl} & {total} & {tr:.4f} & {br:.4f} & {delta:+.4f} \\\\"
        )
    md_lines.append(r"\bottomrule")
    md_lines.append(r"\end{tabular}")
    md_lines.append("```")
    md_lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md_lines))
    return {"rows": rows}


# ------------------------------------------------------------------------
# B-3: aux stats
# ------------------------------------------------------------------------


def _length_buckets(lengths: list[int]) -> list[tuple[int, int]]:
    """Return sensible (lo, hi) inclusive bucket bounds covering the range."""
    if not lengths:
        return []
    edges = [(2, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 40), (41, 100), (101, 10**6)]
    return edges


def emit_b3(merged: list[dict], out_dir: Path,
            treatment_name: str, baseline_name: str) -> dict:
    # F1 by sentence length bucket
    buckets = _length_buckets([r["length"] for r in merged])
    bucket_rows: list[dict] = []
    for lo, hi in buckets:
        in_bucket = [r for r in merged if lo <= r["length"] <= hi]
        if not in_bucket:
            continue
        f1_t = mean(r["treatment_m"]["f1"] for r in in_bucket)
        f1_b = mean(r["baseline_m"]["f1"] for r in in_bucket)
        bucket_rows.append({
            "length_bucket": f"{lo}-{hi}" if hi < 10**5 else f"{lo}+",
            "n_sentences": len(in_bucket),
            f"mean_f1_{treatment_name}": round(f1_t, 4),
            f"mean_f1_{baseline_name}": round(f1_b, 4),
            "delta": round(f1_t - f1_b, 4),
        })

    # Recall by gold span length
    span_len_total: Counter = Counter()
    span_len_tp_t: Counter = Counter()
    span_len_tp_b: Counter = Counter()
    for rec in merged:
        length = rec["length"]
        pred_t_set = set(_filter_spans([(int(s), int(e)) for s, e in rec["treatment_pred"]], length))
        pred_b_set = set(_filter_spans([(int(s), int(e)) for s, e in rec["baseline_pred"]], length))
        for s, e, _ in rec["gold_spans"]:
            s, e = int(s), int(e)
            if s + 1 == e or s == e:
                continue
            if s == 0 and e == length:
                continue
            sl = e - s
            span_len_total[sl] += 1
            if (s, e) in pred_t_set:
                span_len_tp_t[sl] += 1
            if (s, e) in pred_b_set:
                span_len_tp_b[sl] += 1

    # Roll up span lengths >= TAIL_THRESHOLD into a single "<thresh>+" row
    # to avoid noisy single-digit-count rows misleading downstream readers.
    SPAN_LEN_TAIL_THRESHOLD = 35
    span_rows = []
    tail_total = tail_tp_t = tail_tp_b = 0
    for sl in sorted(span_len_total):
        total = span_len_total[sl]
        if sl >= SPAN_LEN_TAIL_THRESHOLD:
            tail_total += total
            tail_tp_t += span_len_tp_t[sl]
            tail_tp_b += span_len_tp_b[sl]
            continue
        rt = span_len_tp_t[sl] / total if total else 0.0
        rb = span_len_tp_b[sl] / total if total else 0.0
        span_rows.append({
            "span_len": str(sl),
            "count": total,
            f"recall_{treatment_name}": round(rt, 4),
            f"recall_{baseline_name}": round(rb, 4),
            "delta": round(rt - rb, 4),
        })
    if tail_total > 0:
        rt = tail_tp_t / tail_total
        rb = tail_tp_b / tail_total
        span_rows.append({
            "span_len": f"{SPAN_LEN_TAIL_THRESHOLD}+",
            "count": tail_total,
            f"recall_{treatment_name}": round(rt, 4),
            f"recall_{baseline_name}": round(rb, 4),
            "delta": round(rt - rb, 4),
        })

    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV: combine both into a single CSV with section headers as comments
    csv_path = out_dir / "aux_stats.csv"
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["# section: f1_by_sentence_length"])
        if bucket_rows:
            writer.writerow(list(bucket_rows[0].keys()))
            for row in bucket_rows:
                writer.writerow(list(row.values()))
        writer.writerow([])
        writer.writerow(["# section: recall_by_gold_span_length"])
        if span_rows:
            writer.writerow(list(span_rows[0].keys()))
            for row in span_rows:
                writer.writerow(list(row.values()))

    md_path = out_dir / "aux_stats.md"
    md_lines: list[str] = []
    md_lines.append("# Phase B-3: Auxiliary statistics\n")
    md_lines.append(f"- Treatment: **{treatment_name}**, baseline: **{baseline_name}**")
    md_lines.append("")
    md_lines.extend(_eval_only_caveat())

    md_lines.append("## F1 by sentence length")
    md_lines.append(
        f"| length | n_sentences | mean_f1_{treatment_name} | mean_f1_{baseline_name} | Δ |"
    )
    md_lines.append("|---|---:|---:|---:|---:|")
    for row in bucket_rows:
        md_lines.append(
            f"| {row['length_bucket']} | {row['n_sentences']} | "
            f"{row[f'mean_f1_{treatment_name}']:.4f} | {row[f'mean_f1_{baseline_name}']:.4f} | "
            f"{row['delta']:+.4f} |"
        )
    md_lines.append("")

    md_lines.append("## Recall by gold span length")
    md_lines.append(
        f"| span_len | gold_count | recall_{treatment_name} | recall_{baseline_name} | Δ |"
    )
    md_lines.append("|---|---:|---:|---:|---:|")
    for row in span_rows:
        md_lines.append(
            f"| {row['span_len']} | {row['count']} | "
            f"{row[f'recall_{treatment_name}']:.4f} | {row[f'recall_{baseline_name}']:.4f} | "
            f"{row['delta']:+.4f} |"
        )
    md_lines.append("")
    md_lines.append(f"CSV: `{csv_path.relative_to(out_dir.parent)}`")
    md_lines.append(
        "  - Two sections separated by `# section: ...` comment lines and a blank row "
        "(use `pandas.read_csv(..., skiprows=...)` or split on the comment markers)."
    )

    md_path.write_text("\n".join(md_lines))
    return {"buckets": bucket_rows, "span_lengths": span_rows, "csv": str(csv_path), "md": str(md_path)}


# ------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase B error analysis (analyze_error.py)")
    p.add_argument("--treatment-jsonl", required=True, type=str)
    p.add_argument("--baseline-jsonl", required=True, type=str)
    p.add_argument("--treatment-name", required=True, type=str)
    p.add_argument("--baseline-name", required=True, type=str)
    p.add_argument("--out-dir", required=True, type=str)
    p.add_argument("--cat-d-high", type=float, default=0.7,
                   help="Category (d) baseline F1 lower bound.")
    p.add_argument("--cat-d-low", type=float, default=0.4,
                   help="Category (d) treatment F1 upper bound.")
    p.add_argument("--label-rows", type=str, default=None,
                   help="Comma-separated label list for the B-2 table "
                        "(default: PTB labels NP,VP,PP,SBAR,ADJP,ADVP). "
                        "For KTB JA pass e.g. PP,IP,NP,CP,CONJP,ADVP.")
    p.add_argument("--strip-label-suffix", action="store_true",
                   help="Collapse gold labels at the first ';' before B-2 "
                        "bucketing (KTB tags PP/NP variants like PP;*SBJ* "
                        "as separate labels — strip to group them).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    treatment_path = Path(args.treatment_jsonl).resolve()
    baseline_path = Path(args.baseline_jsonl).resolve()
    out_dir = Path(args.out_dir).resolve()

    treatment = load_jsonl(treatment_path)
    baseline = load_jsonl(baseline_path)
    if set(treatment) != set(baseline):
        only_t = sorted(set(treatment) - set(baseline))[:5]
        only_b = sorted(set(baseline) - set(treatment))[:5]
        print(f"[warn] sent_id mismatch: only-treatment={len(set(treatment)-set(baseline))} "
              f"e.g. {only_t}; only-baseline={len(set(baseline)-set(treatment))} e.g. {only_b}")

    attach_metrics(treatment)
    attach_metrics(baseline)
    merged = merge_two(treatment, baseline)
    print(f"[info] merged sentences: {len(merged)} "
          f"(treatment={len(treatment)}, baseline={len(baseline)})")

    print(f"[info] B-1 lowf1_cases.md ...")
    info1 = emit_b1(merged, out_dir / "lowf1_cases.md",
                    args.treatment_name, args.baseline_name,
                    args.cat_d_high, args.cat_d_low)
    print(f"        eligible={info1['eligible']}, "
          f"a={info1['cat_a']}, b={info1['cat_b']}, c={info1['cat_c']}, "
          f"d={info1['cat_d']}/{info1['cat_d_total']} "
          + (f"warn={info1['warnings']}" if info1['warnings'] else ""))

    label_rows = (
        [s.strip() for s in args.label_rows.split(",") if s.strip()]
        if args.label_rows else None
    )
    print(f"[info] B-2 constituent_recall.md ...")
    info2 = emit_b2(merged, out_dir / "constituent_recall.md",
                    args.treatment_name, args.baseline_name,
                    label_rows=label_rows,
                    strip_label_suffix=args.strip_label_suffix)
    print(f"        rows: {len(info2['rows'])}")

    print(f"[info] B-3 aux_stats.md / .csv ...")
    info3 = emit_b3(merged, out_dir, args.treatment_name, args.baseline_name)
    print(f"        length buckets: {len(info3['buckets'])}, "
          f"span-length rows: {len(info3['span_lengths'])}")

    print(f"[info] reports written under {out_dir}")


if __name__ == "__main__":
    main()
