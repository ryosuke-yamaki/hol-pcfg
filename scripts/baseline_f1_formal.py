"""
Compute right-branching and random-binary baseline UF1 against gold spans
for the math/code pilot datasets.

UF1 follows the existing metric (`parser/helper/metric.py:UF1`): unlabeled, with
trivial spans (length 1) and the root span stripped, sentence-level F1 averaged.

Usage:
  python scripts/baseline_f1_formal.py --pickle data/clean/symmath_infix-val.pickle
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path


def right_branching_spans(n: int) -> list[tuple[int, int]]:
    """Spans of a right-branching binary tree over a length-n sequence."""
    if n < 2:
        return []
    spans = []
    for i in range(n - 1):
        spans.append((i, n))
    return spans


def random_binary_spans(n: int, rng: random.Random) -> list[tuple[int, int]]:
    """Spans of a uniformly random binary bracketing over a length-n sequence."""
    if n < 2:
        return []
    spans: list[tuple[int, int]] = []

    def split(i: int, j: int) -> None:
        if j - i < 2:
            return
        spans.append((i, j))
        if j - i == 2:
            return
        k = rng.randint(i + 1, j - 1)
        split(i, k)
        split(k, j)

    split(0, n)
    return spans


def left_branching_spans(n: int) -> list[tuple[int, int]]:
    """Spans of a left-branching binary tree over a length-n sequence."""
    if n < 2:
        return []
    return [(0, j) for j in range(2, n + 1)]


def balanced_binary_spans(n: int) -> list[tuple[int, int]]:
    """Spans of a balanced binary tree (split at midpoint) over a length-n sequence."""
    if n < 2:
        return []
    spans: list[tuple[int, int]] = []

    def split(i: int, j: int) -> None:
        if j - i < 2:
            return
        spans.append((i, j))
        if j - i == 2:
            return
        k = (i + j) // 2
        split(i, k)
        split(k, j)

    split(0, n)
    return spans


def normalize_gold(gold_tree: list[list]) -> tuple[list[tuple[int, int]], int]:
    """Filter UF1-style: drop trivial spans (len 1) and the full-sentence span.

    Returns (spans, seq_len).
    """
    if not gold_tree:
        return [], 0
    seq_len = max(j for _, j, *_ in gold_tree)
    keep = []
    for span in gold_tree:
        i, j = span[0], span[1]
        if i + 1 == j:
            continue
        if i == 0 and j == seq_len:
            continue
        keep.append((i, j))
    return keep, seq_len


def strip_full_and_trivial(spans: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    return [(i, j) for (i, j) in spans if not (i + 1 == j) and not (i == 0 and j == n)]


def sentence_f1(gold: set[tuple[int, int]], pred: set[tuple[int, int]]) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    overlap = gold & pred
    p = len(overlap) / len(pred)
    r = len(overlap) / len(gold)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def evaluate(pickle_path: Path, n_random_seeds: int = 5) -> None:
    with pickle_path.open("rb") as f:
        data = pickle.load(f)
    gold_trees = data["gold_tree"]

    rb_f1s: list[float] = []
    lb_f1s: list[float] = []
    bal_f1s: list[float] = []
    rand_f1s_per_seed: list[list[float]] = [[] for _ in range(n_random_seeds)]
    n_valid = 0
    n_dropped = 0
    seq_lens: list[int] = []
    gold_span_counts: list[int] = []

    for gold in gold_trees:
        gold_spans, n = normalize_gold(gold)
        if n < 2:
            n_dropped += 1
            continue
        n_valid += 1
        seq_lens.append(n)
        gold_set = set(gold_spans)
        gold_span_counts.append(len(gold_set))

        rb = set(strip_full_and_trivial(right_branching_spans(n), n))
        rb_f1s.append(sentence_f1(gold_set, rb))

        lb = set(strip_full_and_trivial(left_branching_spans(n), n))
        lb_f1s.append(sentence_f1(gold_set, lb))

        bal = set(strip_full_and_trivial(balanced_binary_spans(n), n))
        bal_f1s.append(sentence_f1(gold_set, bal))

        for s in range(n_random_seeds):
            rng = random.Random(s * 10007 + n)  # de-correlate per sentence
            rnd = set(strip_full_and_trivial(random_binary_spans(n, rng), n))
            rand_f1s_per_seed[s].append(sentence_f1(gold_set, rnd))

    def mean(xs: list[float]) -> float:
        return sum(xs) / max(1, len(xs))

    rand_mean_per_seed = [mean(xs) for xs in rand_f1s_per_seed]
    rand_overall = mean(rand_mean_per_seed)
    rand_min = min(rand_mean_per_seed)
    rand_max = max(rand_mean_per_seed)

    seq_lens.sort()
    median = seq_lens[len(seq_lens) // 2] if seq_lens else 0
    p90 = seq_lens[int(len(seq_lens) * 0.9)] if seq_lens else 0

    print(f"=== {pickle_path.name} ===")
    print(f"sentences scored: {n_valid} (dropped {n_dropped} with n<2)")
    print(f"median seq_len: {median}, p90: {p90}, mean gold spans/sent: {mean(gold_span_counts):.2f}")
    print(f"right-branching sentence-F1: {mean(rb_f1s):.4f}")
    print(f"left-branching  sentence-F1: {mean(lb_f1s):.4f}")
    print(f"balanced-binary sentence-F1: {mean(bal_f1s):.4f}")
    print(f"random-binary   sentence-F1: {rand_overall:.4f} (per-seed mean range [{rand_min:.4f}, {rand_max:.4f}])")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pickle", type=Path, required=True, nargs="+")
    ap.add_argument("--n-random-seeds", type=int, default=5)
    args = ap.parse_args()
    for p in args.pickle:
        evaluate(p, n_random_seeds=args.n_random_seeds)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
