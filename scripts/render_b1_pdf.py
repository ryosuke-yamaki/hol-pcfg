"""Render B-1 (b), (c), and (d) parse trees as multi-page PDFs.

For each candidate sentence: flat text, gold tree (PTB non-terminals + POS
preterminals), HN-PCFG tree (X labels), SN-PCFG tree (X labels), stacked
vertically per page. One PDF per category (b1_b.pdf, b1_c.pdf, b1_d.pdf).

Selection criteria (mirrors analyze_error.py):
  (b) Δ = F1(HN) − F1(SN) ascending, top 20 (largest HN degradations)
  (c) Δ descending, top 20 (largest HN improvements over SN)
  (d) F1(SN) ≥ 0.7 AND F1(HN) ≤ 0.4 (improvement-hint zone)
  Eligibility: length ≥ 10 AND non-trivial gold spans ≥ 3.

Per-sentence metrics use analyze_error.per_sentence_metrics (UF1 with the
LitXNPCFGFixedCost.test_step argument-swap reproduced) so the (b)/(c)/(d)
sets exactly match `analyze_error.py` output.

Usage:
    python scripts/render_b1_pdf.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nltk.tree import Tree

from analyze_error import per_sentence_metrics, MIN_LENGTH, MIN_GOLD_SPANS, TOP_K


def configure_font(font_path: str | None) -> None:
    """Register a system TTF/OTF (e.g. CJK font) so matplotlib renders
    non-Latin glyphs instead of tofu boxes."""
    if not font_path:
        return
    from matplotlib import font_manager
    fp = Path(font_path)
    if not fp.exists():
        print(f"[warn] font path not found: {font_path}", file=sys.stderr)
        return
    font_manager.fontManager.addfont(str(fp))
    name = font_manager.FontProperties(fname=str(fp)).get_name()
    matplotlib.rcParams["font.family"] = name
    matplotlib.rcParams["pdf.fonttype"] = 42
    print(f"[info] font: {name} ({fp})")


# PTB punctuation POS tags removed during preprocessing; used to align raw_pos
# with cleaned word_form (which has all punct tokens stripped).
PTB_PUNCT_POS: set[str] = {",", ".", ":", "''", "``", "-LRB-", "-RRB-", "#", "$"}


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[int(r["sent_id"])] = r
    return out


def cleaned_pos(words: list[str], raw_pos: list[str]) -> list[str] | None:
    """Filter raw_pos by removing punctuation POS tags so it aligns with the
    punct-removed `words`. Return None if lengths still disagree."""
    p = [x for x in raw_pos if x not in PTB_PUNCT_POS]
    if len(p) == len(words):
        return p
    return None


# ---------------------------------------------------------------------------
# Tree construction (handles unary chains for PTB-style gold trees)
# ---------------------------------------------------------------------------


def build_ptb_tree(
    words: list[str],
    spans_in_dfs_order: list,
    pos: list[str] | None = None,
    default_label: str = "X",
) -> Tree:
    """Build NLTK Tree from labeled spans in PTB DFS top-down order.

    Spans with identical (s, e) are honored as unary chains (outer label first
    in input order). Leaf nodes are either bare words (pos=None) or
    Tree(POS, [word]) preterminals when pos is supplied.
    """
    n = len(words)

    def make_leaf(i: int):
        if pos is not None:
            return Tree(pos[i], [words[i]])
        return words[i]

    # Stack frame: [s, e, label, children, cursor]
    stack: list[list] = []
    root_holder: list[Tree | None] = [None]

    def pop_and_attach() -> None:
        s, e, lbl, children, cursor = stack.pop()
        while cursor < e:
            children.append(make_leaf(cursor))
            cursor += 1
        node = Tree(lbl or default_label, children)
        if stack:
            stack[-1][3].append(node)
            stack[-1][4] = max(stack[-1][4], e)
        else:
            root_holder[0] = node

    for s, e, lbl in spans_in_dfs_order:
        s, e = int(s), int(e)
        # Pop spans that don't enclose (s, e). Identical (s, e) is kept (unary).
        while stack and not (stack[-1][0] <= s and e <= stack[-1][1]):
            pop_and_attach()
        # Fill any leaves between top.cursor and s
        if stack:
            top = stack[-1]
            while top[4] < s:
                top[3].append(make_leaf(top[4]))
                top[4] += 1
        # New span's cursor starts at max(s, parent.cursor) so that leaves
        # already consumed by earlier siblings (e.g. in a unary-overlap case
        # where this span's nominal start lies before the parent's progress)
        # are not re-emitted.
        new_cursor = s if not stack else max(s, stack[-1][4])
        stack.append([s, e, lbl, [], new_cursor])

    while stack:
        pop_and_attach()

    if root_holder[0] is None:
        # Fallback (no spans): wrap leaves under a flat default-label root.
        return Tree(default_label, [make_leaf(i) for i in range(n)])
    return root_holder[0]


# ---------------------------------------------------------------------------
# Tree layout for matplotlib
# ---------------------------------------------------------------------------


def layout_tree(tree: Tree) -> tuple[list[dict], float]:
    """Return (nodes, max_depth). Each node has label/x/y/parent/is_leaf.
    Root is at y=0; leaves at y=max_depth (y grows downward)."""
    nodes: list[dict] = []
    leaf_counter = [0]

    def walk(t, parent_idx: int | None, depth: int) -> int:
        my_idx = len(nodes)
        if isinstance(t, str):
            nodes.append({"label": t, "x": float(leaf_counter[0]),
                          "y": float(depth), "parent": parent_idx,
                          "is_leaf": True})
            leaf_counter[0] += 1
            return my_idx
        nodes.append({"label": t.label(), "x": 0.0, "y": float(depth),
                      "parent": parent_idx, "is_leaf": False})
        if len(t) == 0:
            nodes[my_idx]["x"] = float(leaf_counter[0])
            return my_idx
        child_xs: list[float] = []
        for c in t:
            cidx = walk(c, my_idx, depth + 1)
            child_xs.append(nodes[cidx]["x"])
        nodes[my_idx]["x"] = sum(child_xs) / len(child_xs)
        return my_idx

    walk(tree, None, 0)
    max_depth = max(n["y"] for n in nodes) if nodes else 0.0
    return nodes, max_depth


def draw_tree(ax, tree: Tree, font_size: float = 8.0,
              leaf_font_size: float | None = None) -> None:
    """Draw an NLTK Tree on the given matplotlib Axes."""
    if leaf_font_size is None:
        leaf_font_size = font_size
    nodes, max_depth = layout_tree(tree)

    # edges first (so labels' white bbox masks them)
    for n in nodes:
        if n["parent"] is not None:
            p = nodes[n["parent"]]
            ax.plot([n["x"], p["x"]], [n["y"], p["y"]],
                    color="black", linewidth=0.6, zorder=1)

    for n in nodes:
        fs = leaf_font_size if n["is_leaf"] else font_size
        style = "italic" if n["is_leaf"] else "normal"
        ax.text(n["x"], n["y"], n["label"],
                ha="center", va="center", fontsize=fs,
                style=style,
                bbox=dict(boxstyle="square,pad=0.15", fc="white",
                          ec="none", alpha=1.0),
                zorder=2)

    max_x = max(n["x"] for n in nodes)
    min_x = min(n["x"] for n in nodes)
    ax.set_xlim(min_x - 0.5, max_x + 0.5)
    ax.set_ylim(max_depth + 0.5, -0.5)  # invert so root is at top
    ax.axis("off")


# ---------------------------------------------------------------------------
# Category selection (mirrors analyze_error.py)
# ---------------------------------------------------------------------------


def select_b_c_d(
    treatment: dict[int, dict],
    baseline: dict[int, dict],
    cat_d_high: float = 0.7,
    cat_d_low: float = 0.4,
) -> dict[str, list[tuple[int, float, float]]]:
    """Return {'b': ..., 'c': ..., 'd': ...}; entries are (sent_id, f1_t, f1_b)."""
    eligible: list[tuple[int, float, float]] = []
    for sid in sorted(set(treatment) & set(baseline)):
        rec = treatment[sid]
        if rec["length"] < MIN_LENGTH:
            continue
        n_gold = sum(
            1 for s, e, _ in rec["gold_spans"]
            if not (s + 1 == e or s == e or (s == 0 and e == rec["length"]))
        )
        if n_gold < MIN_GOLD_SPANS:
            continue
        t_m = per_sentence_metrics(rec["gold_spans"], rec["pred_spans"])
        b_m = per_sentence_metrics(rec["gold_spans"], baseline[sid]["pred_spans"])
        eligible.append((sid, t_m["f1"], b_m["f1"]))

    cat_b = sorted(eligible, key=lambda x: x[1] - x[2])[:TOP_K]
    cat_c = sorted(eligible, key=lambda x: -(x[1] - x[2]))[:TOP_K]
    cat_d_all = [x for x in eligible if x[2] >= cat_d_high and x[1] <= cat_d_low]
    cat_d = sorted(cat_d_all, key=lambda x: x[1])
    return {"b": cat_b, "c": cat_c, "d": cat_d}


# ---------------------------------------------------------------------------
# Per-page rendering
# ---------------------------------------------------------------------------


def build_three_trees(t_rec: dict, b_rec: dict) -> tuple[Tree, Tree, Tree]:
    words = t_rec["words"]
    pos = cleaned_pos(words, t_rec["raw_pos"])
    gold_tree = build_ptb_tree(words, t_rec["gold_spans"], pos=pos)
    hn_spans = [[s, e, "X"] for s, e in t_rec["pred_spans"]]
    sn_spans = [[s, e, "X"] for s, e in b_rec["pred_spans"]]
    hn_tree = build_ptb_tree(words, hn_spans, pos=None, default_label="X")
    sn_tree = build_ptb_tree(words, sn_spans, pos=None, default_label="X")
    return gold_tree, hn_tree, sn_tree


def _tree_depth(tree: Tree) -> int:
    if isinstance(tree, str):
        return 1
    if len(tree) == 0:
        return 1
    return 1 + max(_tree_depth(c) for c in tree)


def render_page(pdf: PdfPages, sid: int, t_rec: dict, b_rec: dict,
                f1_t: float, f1_b: float,
                treatment_name: str = "HN-PCFG",
                baseline_name: str = "SN-PCFG",
                gold_label_caption: str = "Gold (non-terminals + POS preterminals)") -> None:
    words = t_rec["words"]
    n = t_rec["length"]
    gold_tree, hn_tree, sn_tree = build_three_trees(t_rec, b_rec)

    # Heights scale with tree depth
    depths = [_tree_depth(t) for t in (gold_tree, hn_tree, sn_tree)]
    tree_heights = [max(2.0, d * 0.4) for d in depths]

    fig_w = max(8.5, min(24.0, n * 0.6 + 2.0))
    fig_h = 1.2 + sum(tree_heights) + 0.6  # header + 3 trees + margins

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        f"sent_id={sid}    length={n}    "
        f"F1({treatment_name})={f1_t:.3f}    F1({baseline_name})={f1_b:.3f}",
        fontsize=11, y=0.985,
    )

    gs = fig.add_gridspec(
        4, 1,
        height_ratios=[0.5] + tree_heights,
        hspace=0.32,
        left=0.02, right=0.98, top=0.95, bottom=0.02,
    )

    ax_s = fig.add_subplot(gs[0])
    ax_s.axis("off")
    ax_s.text(0.5, 0.7, "Sentence", ha="center", va="bottom",
              fontsize=10, weight="bold", transform=ax_s.transAxes)
    ax_s.text(0.5, 0.2, " ".join(words), ha="center", va="center",
              fontsize=10, wrap=True,
              transform=ax_s.transAxes)

    titles = [
        gold_label_caption,
        f"{treatment_name} prediction (all internal nodes labelled X)",
        f"{baseline_name} prediction (all internal nodes labelled X)",
    ]
    trees = [gold_tree, hn_tree, sn_tree]
    for i, (title, tree) in enumerate(zip(titles, trees), start=1):
        ax = fig.add_subplot(gs[i])
        ax.set_title(title, fontsize=10, loc="left")
        draw_tree(ax, tree, font_size=8)

    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--treatment-jsonl",
                   default="analysis/error_analysis/HN-PCFG/predictions_test.jsonl")
    p.add_argument("--baseline-jsonl",
                   default="analysis/error_analysis/SN-PCFG/predictions_test.jsonl")
    p.add_argument("--out-dir", default="analysis/error_analysis/tree_pdfs")
    p.add_argument("--cat-d-high", type=float, default=0.7)
    p.add_argument("--cat-d-low", type=float, default=0.4)
    p.add_argument("--treatment-name", default="HN-PCFG",
                   help="Display name for the treatment model (page titles, file metadata).")
    p.add_argument("--baseline-name", default="SN-PCFG",
                   help="Display name for the baseline model.")
    p.add_argument("--gold-caption",
                   default="Gold (non-terminals + POS preterminals)",
                   help="Caption shown above the gold-tree pane (override for KTB etc.).")
    p.add_argument("--font-path", default=None,
                   help="Path to a TTF/OTF font that supports the data's script "
                        "(e.g. /usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf for JA).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_font(args.font_path)
    t = load_jsonl(Path(args.treatment_jsonl))
    b = load_jsonl(Path(args.baseline_jsonl))

    cats = select_b_c_d(t, b, args.cat_d_high, args.cat_d_low)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ("b", "c", "d"):
        items = cats[name]
        out_pdf = out_dir / f"b1_{name}.pdf"
        with PdfPages(out_pdf) as pdf:
            for sid, f1_t, f1_b in items:
                render_page(pdf, sid, t[sid], b[sid], f1_t, f1_b,
                            treatment_name=args.treatment_name,
                            baseline_name=args.baseline_name,
                            gold_label_caption=args.gold_caption)
        print(f"[info] wrote {len(items)} sentences to {out_pdf}")


if __name__ == "__main__":
    main()
