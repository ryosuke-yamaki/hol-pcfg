"""Sample KTB Japanese strings from an HN-PCFG checkpoint.

The script samples directly from the learned PCFG distributions:
root NT, independent left/right children for each NT, and terminal tokens
from preterminal distributions. It writes one sampled sentence/tree per PDF page.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

import torch
import yaml
from easydict import EasyDict as edict

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class Node:
    label: str
    children: list["Node"]
    token: str | None = None


def get_full_distributions(model):
    """Return root/left/right/unary log-probabilities from HN_PCFG."""
    nt = model.NT
    nonterm_emb = model.rule_state_emb[:nt]
    term_emb = model.rule_state_emb[nt:]

    root_logits = model.root_emb @ nonterm_emb.t()
    root_logp = (root_logits * model.log_tau_root.exp()).log_softmax(-1).squeeze(0)

    term_logits = model._hol_scores(
        model.v_term, term_emb, model.vocab_emb.T, model.log_tau_term
    ).t()
    unary_logp = term_logits.log_softmax(-1)

    left = model._hol_scores(
        model.v_left, nonterm_emb, model.rule_state_emb, model.log_tau_rule
    )
    right = model._hol_scores(
        model.v_right, nonterm_emb, model.rule_state_emb, model.log_tau_rule
    )
    left_logp = left.log_softmax(dim=-2)
    right_logp = right.log_softmax(dim=-2)
    return root_logp.cpu(), left_logp.cpu(), right_logp.cpu(), unary_logp.cpu()


def sample_tree(
    root_logp: torch.Tensor,
    left_logp: torch.Tensor,
    right_logp: torch.Tensor,
    unary_logp: torch.Tensor,
    idx2word,
    nt: int,
    max_leaves: int,
    max_depth: int,
) -> tuple[Node, list[str]] | None:
    """Sample one tree; return None when safeguard limits are exceeded."""
    root_idx = torch.multinomial(root_logp.exp(), 1).item()
    leaves: list[str] = []

    def expand_nt(idx: int, depth: int) -> Node | None:
        if depth > max_depth or len(leaves) > max_leaves:
            return None
        left_idx = torch.multinomial(left_logp[:, idx].exp(), 1).item()
        right_idx = torch.multinomial(right_logp[:, idx].exp(), 1).item()
        children: list[Node] = []
        for child_idx in (left_idx, right_idx):
            if child_idx < nt:
                child = expand_nt(child_idx, depth + 1)
            else:
                child = expand_pt(child_idx - nt)
            if child is None:
                return None
            children.append(child)
            if len(leaves) > max_leaves:
                return None
        return Node(label=f"NT={idx}", children=children)

    def expand_pt(idx: int) -> Node | None:
        vocab_idx = torch.multinomial(unary_logp[idx].exp(), 1).item()
        tok = idx2word(vocab_idx)
        if tok is None:
            return None
        leaves.append(tok)
        return Node(label=f"PT={idx}", children=[Node(label=tok, children=[], token=tok)])

    tree = expand_nt(root_idx, 0)
    if tree is None or len(leaves) > max_leaves:
        return None
    return tree, leaves


def tree_depth(node: Node) -> int:
    if not node.children:
        return 1
    return 1 + max(tree_depth(child) for child in node.children)


def leaf_count(node: Node) -> int:
    if node.token is not None:
        return 1
    return sum(leaf_count(child) for child in node.children)


def layout_tree(node: Node):
    positions: dict[int, tuple[float, float]] = {}
    labels: dict[int, str] = {}
    edges: list[tuple[int, int]] = []
    next_x = 0

    def visit(cur: Node, depth: int) -> float:
        nonlocal next_x
        cur_id = id(cur)
        labels[cur_id] = cur.label
        if not cur.children:
            x = float(next_x)
            next_x += 1
        else:
            child_xs = []
            for child in cur.children:
                edges.append((cur_id, id(child)))
                child_xs.append(visit(child, depth + 1))
            x = sum(child_xs) / len(child_xs)
        positions[cur_id] = (x, -float(depth))
        return x

    visit(node, 0)
    return positions, labels, edges


def render_page(
    pdf: PdfPages,
    sample_id: int,
    sentence: str,
    tree: Node,
    font: FontProperties,
    unit_label: str,
    font_scale: float,
) -> None:
    leaves = leaf_count(tree)
    depth = tree_depth(tree)
    fig_scale = max(1.0, font_scale * 0.72)
    fig_w = max(11.0, min(26.0, 0.28 * leaves + 7.0)) * fig_scale
    fig_h = max(8.5, min(18.0, 0.55 * depth + 3.5)) * fig_scale
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    positions, labels, edges = layout_tree(tree)
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]

    for parent, child in edges:
        x0, y0 = positions[parent]
        x1, y1 = positions[child]
        ax.plot([x0, x1], [y0, y1], color="#606060", linewidth=0.45, zorder=1)

    for node_id, (x, y) in positions.items():
        label = labels[node_id]
        is_leaf = not label.startswith(("NT=", "PT="))
        if label.startswith("NT="):
            bbox = dict(boxstyle="round,pad=0.18", facecolor="#e8f1ff", edgecolor="#2f5f9f", linewidth=0.55)
            size = 5.2 if leaves > 70 else 6.0
            color = "#17375e"
        elif label.startswith("PT="):
            bbox = dict(boxstyle="round,pad=0.14", facecolor="#f3f3f3", edgecolor="#9a9a9a", linewidth=0.4)
            size = 4.4 if leaves > 70 else 5.0
            color = "#555555"
        else:
            bbox = None
            size = 7.0 if leaves <= 70 else 6.0
            color = "#111111"
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=size * font_scale,
            color=color,
            fontproperties=font,
            bbox=bbox,
            zorder=2,
        )

    title = f"Sample {sample_id:02d}   length={leaves} {unit_label}"
    ax.text(
        0.0,
        1.06,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11 * font_scale,
        fontproperties=font,
        color="#111111",
    )
    wrap_width = max(20, int(70 * fig_scale / max(font_scale, 0.1)))
    wrapped = "\n".join(wrap(sentence, width=wrap_width))
    ax.text(
        0.0,
        1.0,
        wrapped,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13 * font_scale,
        fontproperties=font,
        color="#111111",
    )

    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 0.8, 0.8)
    top = max(0.76, 0.93 - 0.06 * max(font_scale - 1.0, 0.0))
    fig.tight_layout(rect=(0.02, 0.02, 0.98, top))
    pdf.savefig(fig)
    plt.close(fig)


def sentence_from_leaves(leaves: list[str], joiner: str) -> str:
    return joiner.join(leaves)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=Path, default=Path("config/multilingual/hnpcfg_japanese_char.yaml"))
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-pdf", type=Path, required=True)
    parser.add_argument("--out-txt", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--min-leaves", type=int, default=8)
    parser.add_argument("--max-leaves", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--max-attempts", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--font", type=Path, default=None,
                        help="path to a .ttf font with the glyphs to render "
                             "(e.g. a CJK font for Japanese KTB char output); "
                             "falls back to the matplotlib default if omitted or missing")
    parser.add_argument("--joiner", default="")
    parser.add_argument("--unit-label", default="tokens")
    parser.add_argument("--font-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.font_scale <= 0:
        raise ValueError("--font-scale must be positive")

    torch.manual_seed(args.seed)

    with args.conf.open() as f:
        cfg = edict(yaml.load(f, Loader=yaml.Loader))
    cfg.device = args.device
    cfg.wandb = edict({"enabled": False})

    from parser.helper.data_module import DataModule
    from parser.helper.util import get_model

    dataset = DataModule(cfg)
    model = get_model(cfg.model, dataset)
    state = torch.load(str(args.ckpt), map_location=args.device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()

    word_vocab = dataset.word_vocab
    blocked = {
        getattr(word_vocab, "padding_idx", None),
        getattr(word_vocab, "unknown_idx", None),
    }

    def idx2word(i: int) -> str | None:
        if i in blocked:
            return None
        tok = word_vocab.to_word(i)
        if tok in {"<pad>", "<unk>", "[unk]"}:
            return None
        return tok

    root_logp, left_logp, right_logp, unary_logp = get_full_distributions(model)

    samples: list[tuple[str, Node]] = []
    seen: set[str] = set()
    rejected = 0
    attempts = 0
    while len(samples) < args.num_samples and attempts < args.max_attempts:
        attempts += 1
        sampled = sample_tree(
            root_logp,
            left_logp,
            right_logp,
            unary_logp,
            idx2word,
            model.NT,
            args.max_leaves,
            args.max_depth,
        )
        if sampled is None:
            rejected += 1
            continue
        tree, leaves = sampled
        if len(leaves) < args.min_leaves:
            rejected += 1
            continue
        sentence = sentence_from_leaves(leaves, args.joiner)
        if sentence in seen:
            rejected += 1
            continue
        seen.add(sentence)
        samples.append((sentence, tree))

    if len(samples) < args.num_samples:
        raise RuntimeError(
            f"collected only {len(samples)} samples after {attempts} attempts "
            f"(rejected={rejected})"
        )

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if args.font is not None and args.font.exists():
        font = FontProperties(fname=str(args.font))
    else:
        detail = f"font not found at {args.font}" if args.font is not None else "no --font given"
        print(
            f"[sample_ktb_char_pdf] {detail}; using the matplotlib default "
            "(non-Latin glyphs such as Japanese may not render).",
            file=sys.stderr,
        )
        font = FontProperties()
    with PdfPages(args.out_pdf) as pdf:
        for i, (sentence, tree) in enumerate(samples, 1):
            render_page(pdf, i, sentence, tree, font, args.unit_label, args.font_scale)

    if args.out_txt is not None:
        args.out_txt.parent.mkdir(parents=True, exist_ok=True)
        with args.out_txt.open("w", encoding="utf-8") as f:
            for i, (sentence, tree) in enumerate(samples, 1):
                f.write(f"{i}\t{sentence}\n")

    print(f"wrote {len(samples)} pages to {args.out_pdf}")
    print(f"attempts={attempts} rejected={rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
