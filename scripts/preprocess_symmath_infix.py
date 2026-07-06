"""
Convert Lample symbolic-math prefix expressions into semantic-aware **binary**
PTB trees over an infix token sequence.

Tokenisation: walks the Lample symbolic-math arity table and emits per-character
infix tokens:

  add A B          -> tokens(A) [+] tokens(B)
  sub A B          -> tokens(A) [-] tokens(B)
  mul/div/pow/...  similarly  (* / ^ mod //)
  cos A            -> [cos, (, *tokens(A), )]
  sin/ln/exp/...   similarly (any unary function in the Lample table)
  derivative A B   -> [derivative, (, *A, ,, *B, )]
  g A B            -> [g, (, *A, ,, *B, )]
  h A B C          -> [h, (, *A, ,, *B, ,, *C, )]
  INT+ d1 d2 ...   -> [d1, d2, ...]
  INT- d1 d2 ...   -> [-, d1, d2, ...]
  atom (x, pi, ..) -> [tok]

The tree shape mirrors the Lample call graph and is **strictly binary with no
unary wrappers**. Each binary operator becomes `(S A_tree (S op B_tree))`,
each function call becomes `(S fn (S ( (S arg_tree ))))`, multi-digit
integers become right-branching binary chains. Single-token subtrees stay
as bare-string leaves and attach directly to their parent.

Parens `(` `)` are escaped to `-LRB-` / `-RRB-` for nltk.Tree.fromstring
compatibility.

Run:
  python scripts/preprocess_symmath_infix.py \
      --input data/raw/symmath-val.prefix \
      --output data/clean/symmath_infix-val.txt \
      --raw-output data/raw/symmath_infix-val.txt

Or batch on all 3 splits via the --all flag.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nltk import Tree


BINARY_OPS: dict[str, str] = {
    "add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "^",
    "mod": "mod", "idiv": "//",
}
UNARY_FUNCS: set[str] = {
    "inv", "pow2", "pow3", "pow4", "pow5", "sqrt", "rac",
    "exp", "ln", "abs", "sign",
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "acot", "asec", "acsc",
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
    "f",
}
DIGITS: set[str] = {str(d) for d in range(10)}


class ParseError(ValueError):
    pass


def _escape(tok: str) -> str:
    if tok == "(":
        return "-LRB-"
    if tok == ")":
        return "-RRB-"
    return tok


def _right_branch(tokens: list[str]) -> tuple[str, list[str]]:
    """Right-branching binary tree over a list of token strings.

    Returns (ptb_string, leaves). For a single token, returns the bare token
    string (no S wrapper) so the parent can attach it directly as a leaf.
    """
    escaped = [_escape(t) for t in tokens]
    if not escaped:
        raise ParseError("cannot build tree over empty token list")
    if len(escaped) == 1:
        return escaped[0], escaped
    if len(escaped) == 2:
        return f"(S {escaped[0]} {escaped[1]})", escaped
    rest, _ = _right_branch(tokens[1:])
    return f"(S {escaped[0]} {rest})", escaped


def _wrap_pair(left: str, right: str) -> str:
    return f"(S {left} {right})"


def prefix_to_binary_ptb(prefix: str) -> tuple[str, list[str]]:
    """Walk a Lample prefix expression and return (PTB tree string, leaf tokens).

    The PTB tree is a strictly binary tree with bare-string leaves.
    """
    src = prefix.strip().split()
    if not src:
        raise ParseError("empty prefix")
    pos = [0]

    def walk() -> tuple[str, list[str]]:
        if pos[0] >= len(src):
            raise ParseError("unexpected end of input")
        tok = src[pos[0]]
        pos[0] += 1

        # Integer literals: sign + greedy digits.
        if tok in ("INT+", "INT-"):
            digits: list[str] = []
            while pos[0] < len(src) and src[pos[0]] in DIGITS:
                digits.append(src[pos[0]])
                pos[0] += 1
            if not digits:
                raise ParseError(f"{tok!r} not followed by any digits")
            seq = digits if tok == "INT+" else ["-"] + digits
            return _right_branch(seq)

        # Binary operators.
        if tok in BINARY_OPS:
            a_ptb, a_leaves = walk()
            b_ptb, b_leaves = walk()
            op = BINARY_OPS[tok]
            # tree: (S A (S op B))
            right_subtree = _wrap_pair(_escape(op), b_ptb)
            tree = _wrap_pair(a_ptb, right_subtree)
            leaves = a_leaves + [_escape(op)] + b_leaves
            return tree, leaves

        # Unary functions: f(arg) -> (S f (S ( (S arg )))) where each level is binary.
        if tok in UNARY_FUNCS:
            a_ptb, a_leaves = walk()
            # tree: (S fn (S ( (S A )))) — innermost wraps arg + ")"
            inner = _wrap_pair(a_ptb, _escape(")"))
            mid = _wrap_pair(_escape("("), inner)
            tree = _wrap_pair(_escape(tok), mid)
            leaves = [_escape(tok), _escape("("), *a_leaves, _escape(")")]
            return tree, leaves

        # 2-argument function-call style: f(A, B)
        if tok in ("derivative", "g"):
            a_ptb, a_leaves = walk()
            b_ptb, b_leaves = walk()
            # tree: (S fn (S ( (S A (S , (S B )))))
            innermost = _wrap_pair(b_ptb, _escape(")"))
            comma_b = _wrap_pair(_escape(","), innermost)
            ab = _wrap_pair(a_ptb, comma_b)
            paren_ab = _wrap_pair(_escape("("), ab)
            tree = _wrap_pair(_escape(tok), paren_ab)
            leaves = [_escape(tok), _escape("(")] + a_leaves + [_escape(",")] + b_leaves + [_escape(")")]
            return tree, leaves

        # 3-argument function-call: h(A, B, C)
        if tok == "h":
            a_ptb, a_leaves = walk()
            b_ptb, b_leaves = walk()
            c_ptb, c_leaves = walk()
            inner_c = _wrap_pair(c_ptb, _escape(")"))
            comma_c = _wrap_pair(_escape(","), inner_c)
            bc = _wrap_pair(b_ptb, comma_c)
            comma_bc = _wrap_pair(_escape(","), bc)
            abc = _wrap_pair(a_ptb, comma_bc)
            paren_abc = _wrap_pair(_escape("("), abc)
            tree = _wrap_pair(_escape(tok), paren_abc)
            leaves = ([_escape("h"), _escape("(")] + a_leaves + [_escape(",")] +
                     b_leaves + [_escape(",")] + c_leaves + [_escape(")")])
            return tree, leaves

        # Atom (variable, constant). Bare-string leaf.
        return _escape(tok), [_escape(tok)]

    tree, leaves = walk()
    if pos[0] != len(src):
        raise ParseError(f"trailing prefix tokens: {src[pos[0]:]!r}")
    return tree, leaves


def convert_file(in_path: Path, out_path: Path, raw_out: Path | None,
                 min_len: int, max_len: int, skip_errors: bool) -> tuple[int, int, int]:
    n_ok = 0
    n_err = 0
    n_len = 0
    raw_handle = raw_out.open("w", encoding="utf-8") if raw_out is not None else None
    try:
        with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
            for line_no, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ptb, leaves = prefix_to_binary_ptb(line)
                except ParseError as exc:
                    n_err += 1
                    if skip_errors:
                        print(f"[skip] line {line_no}: {exc}", file=sys.stderr)
                        continue
                    raise
                if not (min_len <= len(leaves) <= max_len):
                    n_len += 1
                    continue
                # Reject single-leaf "expressions" (cannot form a binary tree).
                if " " not in ptb:
                    n_len += 1
                    continue
                fout.write(ptb + "\n")
                if raw_handle is not None:
                    # raw output uses the unescaped source-form glyphs for readability
                    display = [{"-LRB-": "(", "-RRB-": ")"}.get(t, t) for t in leaves]
                    raw_handle.write(" ".join(display) + "\n")
                n_ok += 1
    finally:
        if raw_handle is not None:
            raw_handle.close()
    return n_ok, n_err, n_len


FIXTURES: list[tuple[str, list[str]]] = [
    # cos(x) + sin(x)^2
    ("add cos x pow sin x INT+ 2",
     ["cos", "-LRB-", "x", "-RRB-", "+", "sin", "-LRB-", "x", "-RRB-", "^", "2"]),
    # cos(x) + sin(x^2)
    ("add cos x sin pow x INT+ 2",
     ["cos", "-LRB-", "x", "-RRB-", "+", "sin", "-LRB-", "x", "^", "2", "-RRB-"]),
    # multi-digit integer 123
    ("INT+ 1 2 3", ["1", "2", "3"]),
    # negative single-digit -5
    ("INT- 5", ["-", "5"]),
    # negative multi-digit -12
    ("INT- 1 2", ["-", "1", "2"]),
    # derivative(x^2, x)
    ("derivative pow x INT+ 2 x",
     ["derivative", "-LRB-", "x", "^", "2", ",", "x", "-RRB-"]),
]


def _self_check() -> int:
    n_fail = 0
    for prefix, expected in FIXTURES:
        try:
            ptb, leaves = prefix_to_binary_ptb(prefix)
        except Exception as exc:
            print(f"[FAIL] {prefix!r}: {exc}")
            n_fail += 1
            continue
        if leaves != expected:
            print(f"[FAIL] {prefix!r}:")
            print(f"       got      {leaves}")
            print(f"       expected {expected}")
            n_fail += 1
            continue
        # Re-parse + verify it's a binary tree (every internal node has exactly 2 children).
        try:
            t = Tree.fromstring(ptb)
        except Exception as exc:
            print(f"[FAIL] {prefix!r}: PTB re-parse: {exc}")
            print(f"       {ptb}")
            n_fail += 1
            continue

        def check_binary(node):
            if isinstance(node, str):
                return True
            if len(node) != 2:
                return False
            return check_binary(node[0]) and check_binary(node[1])

        if not check_binary(t):
            print(f"[FAIL] {prefix!r}: not strictly binary: {ptb}")
            n_fail += 1
            continue
        print(f"[OK]   {prefix!r}  ->  {leaves}")
    # Single-token expression should error or produce no usable tree.
    try:
        ptb, leaves = prefix_to_binary_ptb("x")
        print(f"[INFO] single-atom 'x' -> ptb={ptb!r}, leaves={leaves}  (caller should filter)")
    except Exception as exc:
        print(f"[OK]   single-atom rejected: {exc}")
    return n_fail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path,
                    help="single input .prefix file")
    ap.add_argument("--output", type=Path,
                    help="single output .txt PTB file")
    ap.add_argument("--raw-output", type=Path, default=None,
                    help="optional aligned 1-per-line leaf-joined infix text")
    ap.add_argument("--all", action="store_true",
                    help="batch: convert symmath-{train,val,test}.prefix to "
                         "symmath_infix-{train,val,test}.{txt,raw.txt}")
    ap.add_argument("--in-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/clean"))
    ap.add_argument("--raw-out-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        return _self_check()

    if args.all:
        for split in ("train", "val", "test"):
            inp = args.in_dir / f"symmath-{split}.prefix"
            out = args.out_dir / f"symmath_infix-{split}.txt"
            raw_out = args.raw_out_dir / f"symmath_infix-{split}.txt"
            args.out_dir.mkdir(parents=True, exist_ok=True)
            args.raw_out_dir.mkdir(parents=True, exist_ok=True)
            ok, err, dropped = convert_file(inp, out, raw_out, args.min_len, args.max_len,
                                            skip_errors=not args.strict)
            print(f"{split}: wrote {ok} trees (errors={err}, len-filtered={dropped})",
                  file=sys.stderr)
        return 0

    if not args.input or not args.output:
        ap.error("--input and --output are required unless --self-check or --all")

    ok, err, dropped = convert_file(args.input, args.output, args.raw_output,
                                    args.min_len, args.max_len,
                                    skip_errors=not args.strict)
    print(f"wrote {ok} trees (errors={err}, len-filtered={dropped})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
