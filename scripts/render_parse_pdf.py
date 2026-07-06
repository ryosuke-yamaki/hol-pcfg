"""
Run HN-PCFG inference on a dev split and render the first N samples as a PDF
where each page shows:
  - raw source text (top, large)
  - per-sentence Unlabeled-F1 vs gold (right of raw text)
  - the predicted constituency tree (below)

The trees are rendered with matplotlib (binary tree, leaves bottom, root top).

Usage:
  python scripts/render_parse_pdf.py \
      --conf config/formal/hnpcfg_symmath_infix_nt1024.yaml \
      --ckpt log/hnpcfg_symmath_infix_nt1024_seed0/HNPCFG<run-timestamp>/best.pt \
      --raw-text data/raw/symmath_infix-val.txt \
      --raw-text-mode line \
      --pickle data/clean/symmath_infix-val.pickle \
      --out log/parsing_samples/symmath_infix_val_seed0.pdf \
      --n 50
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch
import yaml
from easydict import EasyDict as edict


# ---------- Lample prefix -> infix (inlined; used only by --prefix_to_infix) ----------

class ParseError(ValueError):
    pass


_INT_SIGN_TOKENS = {"INT+", "INT-"}
_DIGIT_TOKENS = {str(d) for d in range(10)}
_BINOP_SYM = {
    "add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "^",
    "mod": "mod", "idiv": "//",
}
_UNARY_FUNCS = {
    "inv", "pow2", "pow3", "pow4", "pow5", "sqrt", "exp", "ln", "abs", "sign", "rac",
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "acot", "asec", "acsc",
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
    "f",
}


def prefix_to_infix(prefix: str) -> str:
    """Render one Lample symbolic-math prefix expression as human-readable infix.

    Inlined from the former scripts/preprocess_lample_math.py (the rest of which
    drove the now-removed prefix-tree symmath experiment).
    """
    tokens = prefix.strip().split()
    if not tokens:
        return ""
    pos = 0

    def consume() -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise ParseError("unexpected end of input")
        tok = tokens[pos]
        pos += 1
        if tok in _INT_SIGN_TOKENS:
            digits = []
            while pos < len(tokens) and tokens[pos] in _DIGIT_TOKENS:
                digits.append(tokens[pos])
                pos += 1
            sign = "" if tok == "INT+" else "-"
            return sign + ("".join(digits) if digits else "0")
        if tok in _BINOP_SYM:
            a = consume()
            b = consume()
            return f"({a} {_BINOP_SYM[tok]} {b})"
        if tok in _UNARY_FUNCS:
            a = consume()
            return f"{tok}({a})"
        if tok == "derivative":
            a = consume()
            b = consume()
            return f"derivative({a}, {b})"
        if tok == "g":
            a = consume()
            b = consume()
            return f"g({a}, {b})"
        if tok == "h":
            a = consume()
            b = consume()
            c = consume()
            return f"h({a}, {b}, {c})"
        # atom (variable, constant)
        return tok

    out = consume()
    # strip the very outer parens if any (cosmetic)
    if out.startswith("(") and out.endswith(")"):
        depth = 0
        ok = True
        for i, ch in enumerate(out):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(out) - 1:
                    ok = False
                    break
        if ok:
            out = out[1:-1]
    return out


# ---------- raw text loading ----------

def load_raw_texts(path: Path, mode: str) -> list[str]:
    """mode='line': one raw text per line.
       mode='jsonl_snippet': JSONL with `snippet` field."""
    out: list[str] = []
    if mode == "line":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                out.append(line.rstrip("\n"))
    elif mode == "jsonl_snippet":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    out.append("")
                    continue
                obj = json.loads(line)
                out.append(obj.get("snippet") or obj.get("code") or "")
    else:
        raise ValueError(f"unknown raw-text-mode: {mode}")
    return out


# ---------- inference ----------

def run_inference(conf_path: Path, ckpt_path: Path, pickle_path: Path,
                  device: str) -> tuple[list[dict], list[list]]:
    """Run MBR decode on the split pointed to by `pickle_path`.

    Returns (records, raw_words) where records is one dict per kept sentence
    (with raw_idx, gold_spans, pred_spans, leaf_words) and raw_words is the
    raw pickle's word list (one item per raw index)."""
    with conf_path.open() as f:
        cfg = edict(yaml.load(f, Loader=yaml.Loader))
    cfg.device = device if torch.cuda.is_available() else "cpu"
    # point the loader at our split (it can be val or test)
    cfg.data.test_file = str(pickle_path)

    from parser.helper.data_module import DataModule
    from parser.helper.util import get_model

    dataset = DataModule(cfg)
    model = get_model(cfg.model, dataset)
    state = torch.load(str(ckpt_path), map_location=cfg.device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(cfg.device).eval()

    with pickle_path.open("rb") as f:
        raw = pickle.load(f)
    raw_words = raw["word"]
    raw_gold = raw["gold_tree"]
    # Char-level KTB pickles also carry the morpheme-level gold tree + boundary
    # offsets so we can draw the gold morpheme tree above the char prediction.
    raw_morph_gold = raw.get("morph_gold")
    raw_morph_offsets = raw.get("morph_offsets")

    kept_to_raw = [i for i, w in enumerate(raw_words) if len(w) > 1]
    test_dataset = dataset.test_dataset
    sampler = dataset.test_dataloader.batch_sampler

    records: list[dict] = []
    for batch_indices in sampler.total:
        lens = [test_dataset[i]["seq_len"] for i in batch_indices]
        word_ids = torch.tensor(
            [test_dataset[i]["word"] for i in batch_indices], device=cfg.device,
        )
        seq_len = torch.tensor(lens, device=cfg.device)
        x_target = {"word": word_ids, "seq_len": seq_len}
        outputs = model.evaluate(x_target, decode_type="mbr", return_labels=True)
        preds = outputs["prediction"]
        nt_label_batch = outputs.get("nt_labels", [None] * len(batch_indices))
        pt_label_batch = outputs.get("pt_labels", [None] * len(batch_indices))
        for k, kept_idx in enumerate(batch_indices):
            raw_idx = kept_to_raw[kept_idx]
            records.append({
                "raw_idx": int(raw_idx),
                "leaf_words": list(raw_words[raw_idx]),
                "gold_spans": [(int(s), int(e)) for s, e, _ in raw_gold[raw_idx]],
                "pred_spans": [(int(s), int(e)) for s, e in preds[k]],
                "nt_labels": nt_label_batch[k] if nt_label_batch[k] is not None else {},
                "pt_labels": pt_label_batch[k] if pt_label_batch[k] is not None else [],
                "morph_gold": (list(raw_morph_gold[raw_idx])
                               if raw_morph_gold is not None else None),
                "morph_offsets": (list(raw_morph_offsets[raw_idx])
                                  if raw_morph_offsets is not None else None),
            })
    # restore raw-index order (sampler may emit out of order)
    records.sort(key=lambda r: r["raw_idx"])
    return records, raw_words


# ---------- F1 ----------

def sentence_f1(gold: list[tuple[int, int]], pred: list[tuple[int, int]], n: int) -> float:
    def strip(spans):
        return set((i, j) for (i, j) in spans
                   if not (i + 1 == j) and not (i == 0 and j == n))
    g, p = strip(gold), strip(pred)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    overlap = g & p
    prec = len(overlap) / len(p)
    rec = len(overlap) / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ---------- tree layout + render ----------

def build_binary_tree(spans: list[tuple[int, int]], n: int) -> dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]] | None]:
    """Map each non-trivial internal span -> (left_child, right_child); leaves -> None."""
    s = set(spans) | {(i, i + 1) for i in range(n)} | {(0, n)}
    children: dict = {}
    for (i, j) in sorted(s, key=lambda x: x[1] - x[0]):
        if j - i == 1:
            children[(i, j)] = None
            continue
        # find the split k such that (i,k) and (k,j) are both in s
        split = None
        for k in range(i + 1, j):
            if (i, k) in s and (k, j) in s:
                split = k
                break
        if split is None:
            # fall back to right-branching to keep the tree connected
            split = i + 1
            s.add((i, split))
            s.add((split, j))
            children.setdefault((i, split), None)
            if (split, j) not in children and j - split == 1:
                children[(split, j)] = None
        children[(i, j)] = ((i, split), (split, j))
    return children


def layout_tree(children, root):
    """Recursively compute (x, y) for each node. y=0 at root, grows downward."""
    pos: dict[tuple[int, int], tuple[float, float]] = {}

    def recurse(span, depth):
        ch = children.get(span)
        if ch is None:
            x = (span[0] + span[1]) / 2.0
            pos[span] = (x, depth)
            return x
        lx = recurse(ch[0], depth + 1)
        rx = recurse(ch[1], depth + 1)
        x = (lx + rx) / 2.0
        pos[span] = (x, depth)
        return x

    recurse(root, 0)
    return pos


_LEAF_DISPLAY_MAP = {"-LRB-": "(", "-RRB-": ")"}
_FONT_PATH: str | None = None  # set by main() when --font is given


def _make_font_prop(size: int):
    """Return a FontProperties at the requested size for the custom font, or None."""
    if _FONT_PATH is None:
        return None
    from matplotlib import font_manager
    return font_manager.FontProperties(fname=_FONT_PATH, size=size)


def _display_leaf(token: str) -> str:
    """Map preprocessing-time escapes back to their source-form characters."""
    return _LEAF_DISPLAY_MAP.get(token, token)


def build_nary_tree(labeled_spans, M):
    """Build an n-ary children map + label map from a laminar set of labeled
    morpheme spans (the KTB gold tree). `labeled_spans` is a list of
    [a, b, label] over morpheme indices (incl. the root span)."""
    label_map = {(int(a), int(b)): lab for a, b, lab in labeled_spans}
    spanset = set(label_map.keys()) | {(i, i + 1) for i in range(M)} | {(0, M)}

    def collect(a, b):
        kids = []
        i = a
        while i < b:
            k = i + 1
            for kk in range(b, i, -1):
                if (i, kk) in spanset and not (i == a and kk == b):
                    k = kk
                    break
            kids.append((i, k))
            i = k
        return kids

    children: dict = {}
    stack = [(0, M)]
    while stack:
        span = stack.pop()
        a, b = span
        if b - a == 1:
            children[span] = []
            continue
        kids = collect(a, b)
        children[span] = kids
        stack.extend(kids)
    return children, label_map


def layout_nary_tree(children, root, leaf_x):
    """Layout an n-ary tree; leaf x positions are supplied (morpheme -> char-centre)."""
    pos: dict = {}

    def recurse(span, depth):
        kids = children.get(span, [])
        if not kids:
            x = leaf_x[span[0]]
            pos[span] = (x, depth)
            return x
        xs = [recurse(c, depth + 1) for c in kids]
        x = sum(xs) / len(xs)
        pos[span] = (x, depth)
        return x

    recurse(root, 0)
    return pos


def _draw_pred(ax, children, pos, leaves, nt_labels, pt_labels, n, max_depth):
    """Draw the predicted char-level binary tree on `ax` (root top, leaves bottom)."""
    for span, ch in children.items():
        if ch is None:
            continue
        x0, y0 = pos[span]
        for c in ch:
            x1, y1 = pos[c]
            ax.plot([x0, x1], [-y0, -y1], color="black", linewidth=1.0)
    for span, (x, y) in pos.items():
        i, j = span
        if children.get(span) is None:
            continue
        nt_id = nt_labels.get((int(i), int(j))) if nt_labels else None
        label = f"NT={nt_id}" if nt_id is not None else "X"
        ax.text(x, -y + 0.05, label, ha="center", va="bottom",
                fontsize=14, color="navy")
    pt_y = -(max_depth + 0.5)
    leaf_y = -(max_depth + 1.1)
    leaf_kwargs = {"ha": "center", "va": "top", "fontsize": 16}
    fp = _make_font_prop(16)
    if fp is not None:
        leaf_kwargs["fontproperties"] = fp
    else:
        leaf_kwargs["family"] = "monospace"
    for i, w in enumerate(leaves):
        if pt_labels is not None and i < len(pt_labels):
            ax.text(i + 0.5, pt_y, f"PT={pt_labels[i]}",
                    ha="center", va="top", fontsize=10, color="darkgreen",
                    family="monospace")
        ax.text(i + 0.5, leaf_y, _display_leaf(w), **leaf_kwargs)
    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(leaf_y - 0.7, 1.6)
    ax.set_axis_off()
    ax.text(-0.4, 1.4, "HN-PCFG predicted (char)", fontsize=11, color="gray",
            ha="left", va="top")


def _draw_gold(ax, children, pos, label_map, morphemes, off, leaf_x, g_max_depth, n):
    """Draw the KTB morpheme-level gold tree on `ax`. Morpheme leaves sit at the
    centre of their char range so the gold tree aligns with the char panel below."""
    for span, kids in children.items():
        if not kids:
            continue
        x0, y0 = pos[span]
        for c in kids:
            x1, y1 = pos[c]
            ax.plot([x0, x1], [-y0, -y1], color="black", linewidth=1.0)
    for span, (x, y) in pos.items():
        if not children.get(span):
            continue
        lab = label_map.get(span, "")
        if lab and lab != "NULL":
            ax.text(x, -y + 0.05, lab, ha="center", va="bottom",
                    fontsize=12, color="darkred")
    leaf_y = -(g_max_depth + 0.9)
    leaf_kwargs = {"ha": "center", "va": "top", "fontsize": 15}
    fp = _make_font_prop(15)
    if fp is not None:
        leaf_kwargs["fontproperties"] = fp
    else:
        leaf_kwargs["family"] = "monospace"
    for m in range(len(morphemes)):
        ax.text(leaf_x[m], leaf_y, _display_leaf(morphemes[m]), **leaf_kwargs)
    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(leaf_y - 0.6, 1.3)
    ax.set_axis_off()
    ax.text(-0.4, 1.2, "Gold morpheme tree (KTB)", fontsize=11, color="gray",
            ha="left", va="top")


def render_one(raw_text: str, leaves: list[str], pred_spans: list[tuple[int, int]],
               gold_spans: list[tuple[int, int]], f1: float, title_idx: int,
               nt_labels: dict[tuple[int, int], int] | None = None,
               pt_labels: list[int] | None = None,
               morph_gold: list | None = None,
               morph_offsets: list | None = None):
    n = len(leaves)
    children = build_binary_tree(pred_spans, n)
    pos = layout_tree(children, (0, n))
    max_depth = max(p[1] for p in pos.values())

    raw_text_short = raw_text if len(raw_text) <= 200 else raw_text[:197] + "..."
    title_kwargs = {"fontsize": 40, "loc": "left"}
    fp = _make_font_prop(40)
    if fp is not None:
        title_kwargs["fontproperties"] = fp
    else:
        title_kwargs["family"] = "monospace"

    fig_w = max(14.0, 0.9 * n + 3.0)
    has_gold = (morph_gold is not None and morph_offsets is not None
                and len(morph_offsets) >= 2)

    if has_gold:
        off = morph_offsets
        M = len(off) - 1
        leaf_x = {m: (off[m] + off[m + 1]) / 2.0 for m in range(M)}
        g_children, g_labels = build_nary_tree(morph_gold, M)
        g_pos = layout_nary_tree(g_children, (0, M), leaf_x)
        g_max_depth = max(p[1] for p in g_pos.values())
        morphemes = ["".join(leaves[off[m]:off[m + 1]]) for m in range(M)]

        gold_h = 2.2 + 0.75 * g_max_depth
        pred_h = 3.0 + 0.75 * max_depth
        fig, (ax_g, ax_p) = plt.subplots(
            2, 1, figsize=(fig_w, gold_h + pred_h),
            gridspec_kw={"height_ratios": [gold_h, pred_h]})
        _draw_gold(ax_g, g_children, g_pos, g_labels, morphemes, off, leaf_x,
                   g_max_depth, n)
        ax_g.set_title(f"#{title_idx}: {raw_text_short}", **title_kwargs)
        _draw_pred(ax_p, children, pos, leaves, nt_labels, pt_labels, n, max_depth)
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(fig_w, 5.5 + 0.75 * max_depth))
    ax.set_title(f"#{title_idx}: {raw_text_short}", **title_kwargs)
    _draw_pred(ax, children, pos, leaves, nt_labels, pt_labels, n, max_depth)
    fig.tight_layout()
    return fig


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", required=True, type=Path,
                    help="yaml config used at training time")
    ap.add_argument("--ckpt", required=True, type=Path,
                    help="best.pt state_dict")
    ap.add_argument("--raw-text", required=True, type=Path,
                    help="file with raw text aligned to the pickle by line/record index")
    ap.add_argument("--raw-text-mode", choices=["line", "jsonl_snippet"], default="line")
    ap.add_argument("--prefix-to-infix", action="store_true",
                    help="treat raw text as Lample prefix expressions and render them "
                         "in infix form (e.g. 'add cos x sin y' -> 'cos(x) + sin(y)')")
    ap.add_argument("--font", type=Path, default=None,
                    help="path to a TTF/OTF font with wide Unicode coverage "
                         "(e.g. GNU Unifont for kaomoji rendering); applied to "
                         "raw-text title and leaves")
    ap.add_argument("--pickle", required=True, type=Path,
                    help="dev/val pickle (the model will parse the sentences in this file)")
    ap.add_argument("--out", required=True, type=Path,
                    help="output PDF path")
    ap.add_argument("--n", type=int, default=50,
                    help="number of sentences to render (taken in dev-set order)")
    ap.add_argument("--min-leaves", type=int, default=None,
                    help="only render sentences with at least this many leaves")
    ap.add_argument("--max-leaves", type=int, default=None,
                    help="only render sentences with at most this many leaves "
                         "(keeps char-level example trees readable)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    raw_texts = load_raw_texts(args.raw_text, args.raw_text_mode)
    print(f"[info] loaded {len(raw_texts)} raw texts from {args.raw_text}",
          file=sys.stderr)
    if args.prefix_to_infix:
        raw_texts = [prefix_to_infix(t) if t else t for t in raw_texts]
        print(f"[info] converted raw texts to infix notation", file=sys.stderr)

    if args.font is not None:
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(args.font))
        global _FONT_PATH
        _FONT_PATH = str(args.font)
        print(f"[info] using font {args.font}", file=sys.stderr)

    records, raw_words = run_inference(args.conf, args.ckpt, args.pickle, args.device)
    print(f"[info] inferred {len(records)} parsed sentences from {args.pickle}",
          file=sys.stderr)

    if len(raw_texts) != len(raw_words):
        print(f"[warn] raw_text count ({len(raw_texts)}) != pickle word count "
              f"({len(raw_words)}); proceeding by raw_idx, mismatches will be flagged",
              file=sys.stderr)

    if args.min_leaves is not None:
        records = [r for r in records if len(r["leaf_words"]) >= args.min_leaves]
    if args.max_leaves is not None:
        records = [r for r in records if len(r["leaf_words"]) <= args.max_leaves]
    print(f"[info] {len(records)} sentences after leaf-count filter "
          f"(min={args.min_leaves}, max={args.max_leaves})", file=sys.stderr)

    with PdfPages(args.out) as pdf:
        for rec in records[: args.n]:
            raw_idx = rec["raw_idx"]
            raw_text = raw_texts[raw_idx] if raw_idx < len(raw_texts) else f"(no raw text @ idx {raw_idx})"
            leaves = rec["leaf_words"]
            pred = rec["pred_spans"]
            gold = rec["gold_spans"]
            n = len(leaves)
            f1 = sentence_f1(gold, pred, n)
            fig = render_one(raw_text, leaves, pred, gold, f1, raw_idx,
                             nt_labels=rec.get("nt_labels"),
                             pt_labels=rec.get("pt_labels"),
                             morph_gold=rec.get("morph_gold"),
                             morph_offsets=rec.get("morph_offsets"))
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[info] wrote {min(args.n, len(records))} pages to {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
