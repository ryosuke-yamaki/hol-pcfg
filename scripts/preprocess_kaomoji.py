"""
Convert the ekohrt/emoticon_kaomoji_dataset into PTB-style bracketed trees for
HN-PCFG training.

Tokenisation: **codepoint-level** — every Unicode codepoint of the kaomoji
becomes one leaf token. There is no natural gold parse tree for emoticons, so
each kaomoji is wrapped in a flat unary stub `(S (C c1) (C c2) ... (C cN))`.
With this stub, `preprocessing.factorize` yields zero non-trivial gold spans,
so F1 against gold is degenerate (always 0/1). Use marginal log-likelihood and
visual tree inspection as the primary evaluation signals.

Escapes (so nltk.Tree.fromstring can re-parse the output):
  - `(` -> `-LRB-`
  - `)` -> `-RRB-`
  - any Unicode whitespace -> `_`

Pipeline: read `emoticon_dict.json`, take the top-level keys as the kaomoji
strings (62k unique), filter by length, shuffle with `--seed`, split into
train/val/test, and emit `data/clean/kaomoji-{train,val,test}.txt`.

Upstream data: github.com/ekohrt/emoticon_kaomoji_dataset
  https://raw.githubusercontent.com/ekohrt/emoticon_kaomoji_dataset/main/emoticon_dict.json
License: not specified on the upstream repo as of 2026-05-27. Research-use
only. Source emoticons are aggregated from various public sites enumerated in
the upstream README.

Run:
  python scripts/preprocess_kaomoji.py --train 50000 --val 2000 --test 2000
  python scripts/preprocess_kaomoji.py --self-check
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from nltk import Tree


def _escape_char(ch: str) -> str | None:
    """Return the PTB-safe surface form of a single codepoint, or None to drop."""
    if ch == "(":
        return "-LRB-"
    if ch == ")":
        return "-RRB-"
    if ch.isspace():
        return "_"
    if not ch:
        return None
    return ch


def kaomoji_to_tree(kaomoji: str) -> Tree | None:
    """Build a flat unary PTB stub for one kaomoji, or None if it has no leaves."""
    leaves: list[Tree] = []
    for ch in kaomoji:
        esc = _escape_char(ch)
        if esc is None:
            continue
        leaves.append(Tree("C", [esc]))
    if not leaves:
        return None
    return Tree("S", leaves)


def tree_to_ptb(tree: Tree) -> str:
    return tree._pformat_flat(nodesep="", parens="()", quotes=False)


def load_kaomoji_keys(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return list(obj.keys())
    if isinstance(obj, list):
        # Future-proofing: support a flat list of emoticons too.
        return [str(x) for x in obj]
    raise ValueError(f"unsupported emoticon_dict.json structure: {type(obj).__name__}")


def main_split(args: argparse.Namespace) -> int:
    kaomoji = load_kaomoji_keys(args.source)
    print(f"loaded {len(kaomoji)} kaomoji from {args.source}", file=sys.stderr)

    kept: list[tuple[str, str]] = []
    n_short = 0
    n_long = 0
    n_empty = 0
    seen: set[str] = set()
    for k in kaomoji:
        if k in seen:
            continue
        seen.add(k)
        tree = kaomoji_to_tree(k)
        if tree is None:
            n_empty += 1
            continue
        n_leaves = len(tree.leaves())
        if n_leaves < args.min_len:
            n_short += 1
            continue
        if n_leaves > args.max_len:
            n_long += 1
            continue
        kept.append((k, tree_to_ptb(tree)))
    print(f"after filter: kept={len(kept)} short={n_short} long={n_long} empty={n_empty}",
          file=sys.stderr)

    needed = args.train + args.val + args.test
    if len(kept) < needed:
        raise RuntimeError(f"only {len(kept)} eligible kaomoji, need {needed}")

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    splits = {
        "train": kept[: args.train],
        "val":   kept[args.train : args.train + args.val],
        "test":  kept[args.train + args.val : args.train + args.val + args.test],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.raw_out_dir is not None:
        args.raw_out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        out = args.out_dir / f"kaomoji-{name}.txt"
        with out.open("w", encoding="utf-8") as f:
            for _, ptb in rows:
                f.write(ptb + "\n")
        print(f"wrote {len(rows)} trees to {out}", file=sys.stderr)
        if args.raw_out_dir is not None:
            raw_out = args.raw_out_dir / f"kaomoji-{name}.txt"
            with raw_out.open("w", encoding="utf-8") as f:
                for raw, _ in rows:
                    # Escape embedded newlines so file is line-aligned with the PTB output
                    safe = raw.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
                    f.write(safe + "\n")
            print(f"wrote {len(rows)} raw kaomoji to {raw_out}", file=sys.stderr)
    return 0


FIXTURES: list[tuple[str, list[str]]] = [
    ("(╥﹏╥)",        ["-LRB-", "╥", "﹏", "╥", "-RRB-"]),
    ("（＾ω＾）",       ["（", "＾", "ω", "＾", "）"]),  # fullwidth parens are NOT escaped (they're not '(' / ')')
    ("( ˘͈ ᵕ ˘͈♡)",   ["-LRB-", "_", "˘", "͈", "_", "ᵕ", "_", "˘", "͈", "♡", "-RRB-"]),
    (":)",            [":", "-RRB-"]),
    ("xD",            ["x", "D"]),
]


def _self_check() -> int:
    n_fail = 0
    for kaomoji, expected_leaves in FIXTURES:
        tree = kaomoji_to_tree(kaomoji)
        if tree is None:
            print(f"[FAIL] {kaomoji!r}: produced no tree")
            n_fail += 1
            continue
        ptb = tree_to_ptb(tree)
        re_parsed = Tree.fromstring(ptb)
        leaves = re_parsed.leaves()
        if leaves != expected_leaves:
            print(f"[FAIL] {kaomoji!r}:")
            print(f"       got      {leaves}")
            print(f"       expected {expected_leaves}")
            n_fail += 1
        else:
            print(f"[OK]   {kaomoji!r}  ->  {leaves}")
    # Empty / length=1 / length=0 cases
    for kaomoji, ok in [("", False), (" ", True), ("a", True)]:
        tree = kaomoji_to_tree(kaomoji)
        if (tree is not None) != ok:
            print(f"[FAIL] degenerate {kaomoji!r}: tree-not-None={tree is not None}, expected={ok}")
            n_fail += 1
        else:
            print(f"[OK]   degenerate {kaomoji!r}: tree-not-None={tree is not None}")
    return n_fail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=Path("data/raw/emoticon_dict.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/clean"))
    ap.add_argument("--raw-out-dir", type=Path, default=None,
                    help="if set, also dump one raw kaomoji per line into "
                         "<raw_out_dir>/kaomoji-{train,val,test}.txt aligned with the PTB output")
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        return _self_check()
    return main_split(args)


if __name__ == "__main__":
    raise SystemExit(main())
