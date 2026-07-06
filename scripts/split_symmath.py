"""
Split Lample symbolic-math prefix expressions into train/val/test prefix files.

Reads `data/raw/prim_fwd.train` (one row per line; format
`<id>|<expr_lhs>\\t<expr_rhs>`), keeps both columns as separate prefix examples,
filters by length and de-duplicates across splits to avoid contamination.

Run:
  python scripts/split_symmath.py --source data/raw/prim_fwd.train \\
      --out-dir data/raw --train 20000 --val 2000 --test 2000

Upstream data: facebookresearch/SymbolicMathematics
  https://dl.fbaipublicfiles.com/SymbolicMathematics/data/prim_fwd.tar.gz
License: CC BY-NC 4.0 (research only)
Citation: Lample & Charton, "Deep Learning for Symbolic Mathematics" (ICLR 2020).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def collect_expressions(src: Path, max_len: int, min_len: int, max_lines: int | None,
                        columns: str) -> list[str]:
    """Collect prefix expressions from prim_fwd lines.

    columns:
      "both"  -> both left (ODE form `sub Y' f(x)`) and right (antiderivative)
      "left"  -> only the left column (always begins with `sub Y'`)
      "right" -> only the right column (pure antiderivative expressions)
    """
    use_left = columns in ("both", "left")
    use_right = columns in ("both", "right")
    exprs: list[str] = []
    with src.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                _, rest = line.split("|", 1)
                left, right = rest.split("\t")
            except ValueError:
                continue
            selected: list[str] = []
            if use_left:
                selected.append(left)
            if use_right:
                selected.append(right)
            for col in selected:
                col = col.strip()
                if not col:
                    continue
                n_tok = col.count(" ") + 1
                if min_len <= n_tok <= max_len:
                    exprs.append(col)
    return exprs


def split_dedup(exprs: list[str], n_train: int, n_val: int, n_test: int, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    rng.shuffle(exprs)

    needed = n_train + n_val + n_test
    if len(exprs) < needed:
        raise RuntimeError(f"only {len(exprs)} expressions available, need {needed}")

    # Take the first chunk, but enforce cross-split uniqueness:
    train_set: set[str] = set()
    val_set: set[str] = set()
    test_set: set[str] = set()

    it = iter(exprs)
    train: list[str] = []
    while len(train) < n_train:
        e = next(it)
        if e in train_set:
            continue
        train_set.add(e)
        train.append(e)

    val: list[str] = []
    while len(val) < n_val:
        e = next(it)
        if e in train_set or e in val_set:
            continue
        val_set.add(e)
        val.append(e)

    test: list[str] = []
    while len(test) < n_test:
        e = next(it)
        if e in train_set or e in val_set or e in test_set:
            continue
        test_set.add(e)
        test.append(e)

    return {"train": train, "val": val, "test": test}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=Path("data/raw/prim_fwd.train"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--max-source-lines", type=int, default=200000)
    ap.add_argument("--columns", choices=["both", "left", "right"], default="both",
                    help="which column(s) of prim_fwd to draw expressions from "
                         "(left=ODE form Y' - f(x), right=pure antiderivative)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    exprs = collect_expressions(args.source, args.max_len, args.min_len, args.max_source_lines,
                                args.columns)
    print(f"collected {len(exprs)} raw expressions from {args.source} (min_len={args.min_len}, max_len={args.max_len})")

    splits = split_dedup(exprs, args.train, args.val, args.test, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, lst in splits.items():
        out = args.out_dir / f"symmath-{name}.prefix"
        with out.open("w", encoding="utf-8") as f:
            for e in lst:
                f.write(e + "\n")
        print(f"wrote {len(lst)} expressions to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
