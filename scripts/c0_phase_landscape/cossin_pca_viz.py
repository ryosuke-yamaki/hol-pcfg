"""Torus-aware 2D projection on (cos theta, sin theta) phase embedding.

Each entity x on the phase-only manifold is mapped to
    z(x) = [cos phi_1, sin phi_1, ..., cos phi_{d/2-1}, sin phi_{d/2-1}]
using only the interior FFT phases (k = 1 ... d/2-1). A 2D projection
(PCA, t-SNE, or UMAP) is then applied on the stacked z vectors. Because
adjacent phases 0 and 2pi are mapped to the same (cos, sin) point, the
circular topology of the 255-torus is preserved - the projection sees
no artificial wraparound jumps.

Output: one 2D scatter per checkpoint containing NT / T (and optionally
vocab) entities and the three relation vectors v_left / v_right / v_term.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# POS prefix groups: each PTB tag is bucketed by its leading characters into
# a coarse class. Order in `pos_groups` determines which group wins on
# ambiguity (prefix match is greedy on declared order).
_POS_PREFIX_RULES: dict[str, tuple[str, ...]] = {
    "NN": ("NN", "NNS", "NNP", "NNPS"),
    "VB": ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"),
    "JJ": ("JJ", "JJR", "JJS"),
    "RB": ("RB", "RBR", "RBS"),
    "IN": ("IN",),
    "DT": ("DT", "PDT", "WDT"),
    "CC": ("CC",),
    "PRP": ("PRP", "PRP$"),
    "MD": ("MD",),
    "CD": ("CD",),
    "WP": ("WP", "WP$"),
    "WRB": ("WRB",),
    "TO": ("TO",),
    "EX": ("EX",),
    "FW": ("FW",),
    "UH": ("UH",),
    "SYM": ("SYM",),
    "POS": ("POS",),
    "RP": ("RP",),
}

# Hand-picked (color, marker) per group. NT uses circles for all (filled with
# saturated tab10-derived hues); PT uses non-circle shapes so the two panels
# stay visually separable. Colors are chosen for max perceptual distance
# within each panel and minimal collision across panels.
_NT_VISUAL: dict[str, tuple[str, str]] = {
    "NP":   ("#1f77b4", "o"),  # azure
    "VP":   ("#d62728", "o"),  # red
    "PP":   ("#2ca02c", "o"),  # kelly green
    "S":    ("#9467bd", "o"),  # orchid purple
    "SBAR": ("#ff7f0e", "o"),  # orange
    "ADJP": ("#8c564b", "o"),  # brown
    "ADVP": ("#e377c2", "o"),  # pink
}
_PT_VISUAL: dict[str, tuple[str, str]] = {
    "NN":  ("#0d47a1", "^"),   # deep navy, triangle-up
    "VB":  ("#c2185b", "v"),   # rose, triangle-down
    "JJ":  ("#1b5e20", "s"),   # forest, square
    "RB":  ("#b8860b", "D"),   # goldenrod, diamond
    "IN":  ("#4a148c", "P"),   # deep purple, plus-filled
    "DT":  ("#00838f", "*"),   # teal, star
    "CC":  ("#ad1457", "h"),   # cerise, hexagon
    "PRP": ("#5d4037", "X"),   # taupe brown, x-filled
    "MD":  ("#1565c0", "<"),   # cobalt, triangle-left
    "CD":  ("#827717", ">"),   # olive, triangle-right
}
_OTHER_N_COLOR = "#d0d0d0"
_OTHER_T_COLOR = "#d0d0d0"


def nt_visual(group: str, fallback_idx: int) -> tuple[str, str]:
    """Return (color, marker) for an NT phrase group, falling back to a
    procedural color if the group is not in the curated table (e.g. when the
    user passes a custom --phrase_groups list)."""
    if group in _NT_VISUAL:
        return _NT_VISUAL[group]
    color = plt.cm.tab20(fallback_idx % 20)
    return (color, "o")


def pt_visual(group: str, fallback_idx: int) -> tuple[str, str]:
    if group in _PT_VISUAL:
        return _PT_VISUAL[group]
    color = plt.cm.tab20b(fallback_idx % 20)
    fallback_markers = ("^", "v", "s", "D", "P", "*", "h", "X", "<", ">")
    return (color, fallback_markers[fallback_idx % len(fallback_markers)])


def pos_prefix(pt_tag: str | None, groups: list[str]) -> str:
    """Map a PTB POS tag to one of the requested coarse groups.

    Returns the matching group name or "other-T" when no rule matches or
    the tag is None.
    """
    if pt_tag is None:
        return "other-T"
    for g in groups:
        if pt_tag in _POS_PREFIX_RULES.get(g, (g,)):
            return g
    return "other-T"


def phrase_group(nt_label: str | None, groups: list[str]) -> str:
    if nt_label is None:
        return "other-N"
    return nt_label if nt_label in groups else "other-N"


def set_global_seed(seed: int) -> None:
    """Fix Python/NumPy/Torch global RNGs in addition to the explicit
    np.random.default_rng(seed) for vocab subsampling and the
    random_state=seed passed to sklearn PCA; guards against future RNG
    calls being added without being threaded through args.seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_state(ckpt_path: str) -> dict:
    """Load state dict from either a Lightning .ckpt or a raw torch.save .pt."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    if any(k.startswith("model.") for k in sd.keys()):
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    return dict(sd)


def cossin_embed(x: torch.Tensor) -> np.ndarray:
    """Map (N, d) real vectors to (N, 2*(d/2-1)) via interior phase cos/sin."""
    X = torch.fft.rfft(x, dim=-1)
    theta = torch.angle(X[:, 1:-1]).numpy()
    return np.concatenate([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--NT", type=int, default=1024)
    ap.add_argument("--vocab_sample", type=int, default=3000,
                    help="Vocab subsample (-1 = all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--label_map", default="",
        help="JSON from build_symbol_labels.py. Empty: 3-color legacy plot.",
    )
    ap.add_argument(
        "--phrase_groups",
        default="NP,VP,PP,S,SBAR,ADJP,ADVP",
        help="NT phrase labels to color individually.",
    )
    ap.add_argument(
        "--pos_groups",
        default="NN,VB,JJ,RB,IN,DT,CC,PRP,MD,CD",
        help="PT POS prefix groups to color individually.",
    )
    ap.add_argument(
        "--min_support", type=int, default=5,
        help="Indices with support < this fall back to other-N / other-T.",
    )
    ap.add_argument(
        "--method", choices=("pca", "tsne", "umap"), default="pca",
        help="2D projection method applied to the (cos, sin) embedding.",
    )
    ap.add_argument(
        "--no_vocab", action="store_true",
        help="Exclude vocab embeddings from both the fit and the scatter.",
    )
    ap.add_argument(
        "--tsne_perplexity", type=float, default=30.0,
        help="t-SNE perplexity (ignored for --method != tsne).",
    )
    ap.add_argument(
        "--umap_n_neighbors", type=int, default=15,
        help="UMAP n_neighbors (ignored for --method != umap).",
    )
    ap.add_argument(
        "--umap_min_dist", type=float, default=0.1,
        help="UMAP min_dist (ignored for --method != umap).",
    )
    args = ap.parse_args()

    set_global_seed(args.seed)

    t0 = time.time()
    sd = load_state(args.ckpt)
    rse = sd["rule_state_emb"].float()
    v_left = sd["v_left"].float().unsqueeze(0)
    v_right = sd["v_right"].float().unsqueeze(0)
    v_term = sd["v_term"].float().unsqueeze(0)
    NT = args.NT

    # (cos, sin) embeddings in 2 * (d/2 - 1) = 510 dimensions
    z_nt = cossin_embed(rse[:NT])
    z_t = cossin_embed(rse[NT:])
    z_vs = cossin_embed(torch.cat([v_left, v_right, v_right * 0 + v_term], dim=0))
    # the line above has a typo guard; redo cleanly:
    z_vs = cossin_embed(torch.cat([v_left, v_right, v_term], dim=0))

    if args.no_vocab:
        z_vocab = np.empty((0, z_nt.shape[1]), dtype=np.float32)
    else:
        vocab = sd["vocab_emb"].float().t()
        rng = np.random.default_rng(args.seed)
        if 0 < args.vocab_sample < len(vocab):
            idx = rng.choice(len(vocab), size=args.vocab_sample, replace=False)
            vocab = vocab[idx]
        z_vocab = cossin_embed(vocab)

    print(f"[{args.lang}] loaded   NT={len(z_nt)}, T={len(z_t)}, "
          f"vocab={len(z_vocab)}, v_*=3,  emb_dim={z_nt.shape[1]}   "
          f"method={args.method}  dt={time.time()-t0:.1f}s")

    # Fit the chosen 2D projection so all points share coordinates.
    z_all = np.vstack([z_nt, z_t, z_vocab, z_vs])
    if args.method == "pca":
        reducer = PCA(n_components=2, random_state=args.seed)
        xy = reducer.fit_transform(z_all)
        evr = reducer.explained_variance_ratio_ * 100
    elif args.method == "tsne":
        # perplexity must be < n_samples; cap it for tiny ckpts.
        perplexity = min(args.tsne_perplexity, max(2.0, len(z_all) / 3.0 - 1.0))
        reducer = TSNE(
            n_components=2, random_state=args.seed,
            perplexity=perplexity, init="pca", learning_rate="auto",
        )
        xy = reducer.fit_transform(z_all)
        evr = None
    elif args.method == "umap":
        import umap  # imported lazily to keep PCA-only runs dependency-free.
        reducer = umap.UMAP(
            n_components=2, random_state=args.seed,
            n_neighbors=min(args.umap_n_neighbors, max(2, len(z_all) - 1)),
            min_dist=args.umap_min_dist,
        )
        xy = reducer.fit_transform(z_all)
        evr = None
    else:
        raise ValueError(f"unknown method: {args.method}")

    n_nt, n_t, n_v = len(z_nt), len(z_t), len(z_vocab)
    xy_nt = xy[:n_nt]
    xy_t = xy[n_nt:n_nt + n_t]
    xy_vocab = xy[n_nt + n_t:n_nt + n_t + n_v]
    xy_vs = xy[n_nt + n_t + n_v:]

    # Axis labels: only PCA has a meaningful "explained variance ratio" to
    # report; t-SNE / UMAP coordinates are dimensionless so we just number them.
    if args.method == "pca":
        xlabel = f"PC1 ({evr[0]:.1f}%)"
        ylabel = f"PC2 ({evr[1]:.1f}%)"
    elif args.method == "tsne":
        xlabel, ylabel = "t-SNE 1", "t-SNE 2"
    else:
        xlabel, ylabel = "UMAP 1", "UMAP 2"

    # ---- Plot -------------------------------------------------------------
    # Labeled mode needs extra horizontal room for the three external legends.
    fig_w = 13 if args.label_map else 10
    fig, ax = plt.subplots(figsize=(fig_w, 10))

    if args.label_map:
        # Labeled mode: per-group scatter for NT and PT using build_symbol_labels.py JSON.
        with open(args.label_map) as f:
            lm = json.load(f)
        if lm["NT"] != n_nt or lm["T"] != n_t:
            raise ValueError(
                f"label_map shape mismatch: NT={lm['NT']}/{n_nt}, T={lm['T']}/{n_t}"
            )
        phrase_groups = [g for g in args.phrase_groups.split(",") if g]
        pos_groups = [g for g in args.pos_groups.split(",") if g]

        # Effective per-index labels with min_support cutoff.
        nt_eff: list[str | None] = [
            lab if (lab is not None and sup >= args.min_support) else None
            for lab, sup in zip(lm["nt_label"], lm["nt_support"])
        ]
        pt_eff: list[str | None] = [
            lab if (lab is not None and sup >= args.min_support) else None
            for lab, sup in zip(lm["pt_label"], lm["pt_support"])
        ]
        nt_groups_per_idx = [phrase_group(lab, phrase_groups) for lab in nt_eff]
        pt_groups_per_idx = [pos_prefix(lab, pos_groups) for lab in pt_eff]

        # Vocab background first. Muted lavender so Σ stays clearly distinct
        # from the cool-gray other-N / other-T points beneath the named groups.
        h_vocab = None
        if not args.no_vocab:
            h_vocab = ax.scatter(
                xy_vocab[:, 0], xy_vocab[:, 1],
                s=5, c="#b8a8d0", alpha=0.32, edgecolors="none",
                zorder=1,
            )
        # Other-N / Other-T drawn beneath the named groups, but their legend
        # handles are appended last so "other" sits at the bottom of each
        # legend section.
        nt_groups_arr = np.array(nt_groups_per_idx)
        pt_groups_arr = np.array(pt_groups_per_idx)
        nt_handles: list = []
        pt_handles: list = []
        h_other_n = None
        mask_other_n = nt_groups_arr == "other-N"
        if mask_other_n.any():
            h_other_n = ax.scatter(
                xy_nt[mask_other_n, 0], xy_nt[mask_other_n, 1],
                s=8, c=_OTHER_N_COLOR, alpha=0.35, edgecolors="none",
                marker="o", zorder=2,
            )
        h_other_t = None
        mask_other_t = pt_groups_arr == "other-T"
        if mask_other_t.any():
            h_other_t = ax.scatter(
                xy_t[mask_other_t, 0], xy_t[mask_other_t, 1],
                s=8, c=_OTHER_T_COLOR, alpha=0.35, edgecolors="none",
                marker="^", zorder=2,
            )
        for gi, g in enumerate(phrase_groups):
            mask = nt_groups_arr == g
            if not mask.any():
                continue
            color, marker = nt_visual(g, gi)
            h = ax.scatter(
                xy_nt[mask, 0], xy_nt[mask, 1],
                s=32, color=color, alpha=0.8,
                edgecolors="white", linewidths=0.4,
                marker=marker, zorder=3,
            )
            nt_handles.append((g, h))
        for gi, g in enumerate(pos_groups):
            mask = pt_groups_arr == g
            if not mask.any():
                continue
            color, marker = pt_visual(g, gi)
            h = ax.scatter(
                xy_t[mask, 0], xy_t[mask, 1],
                s=36, color=color, alpha=0.8,
                edgecolors="white", linewidths=0.4,
                marker=marker, zorder=4,
            )
            pt_handles.append((g, h))
        if h_other_n is not None:
            nt_handles.append(("other", h_other_n))
        if h_other_t is not None:
            pt_handles.append(("other", h_other_t))

        star_colors = ["black", "black", "black"]
        star_markers = [r"$\mathbf{v}^{(L)}$", r"$\mathbf{v}^{(R)}$", r"$\mathbf{v}^{(T)}$"]
        for i, (c, m) in enumerate(zip(star_colors, star_markers)):
            ax.scatter(
                xy_vs[i, 0], xy_vs[i, 1],
                s=800, c=c, marker=m,
                zorder=5,
            )

        ax.set_xlabel(xlabel, fontsize=24)
        ax.set_ylabel(ylabel, fontsize=24)
        ax.tick_params(labelsize=20)
        ax.axhline(0, color="#d0d0d0", linewidth=0.5, zorder=0)
        ax.axvline(0, color="#d0d0d0", linewidth=0.5, zorder=0)
        ax.set_aspect("equal", adjustable="datalim")
        # Legends placed outside the plot (right side, stacked vertically) so
        # they never occlude data points: N (top), P (middle), and Vocab Σ
        # (bottom) when present. With --no_vocab the Σ legend is omitted.
        leg_nt = None
        leg_pt = None
        if nt_handles:
            leg_nt = ax.legend(
                [h for _, h in nt_handles], [g for g, _ in nt_handles],
                title=r"$N$", loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                fontsize=14, title_fontsize=16, framealpha=0.9,
                borderaxespad=0.0,
            )
            for handle in leg_nt.legend_handles:
                handle.set_sizes([120])
        if pt_handles:
            leg_pt = ax.legend(
                [h for _, h in pt_handles], [g for g, _ in pt_handles],
                title=r"$P$", loc="upper left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=14, title_fontsize=16, framealpha=0.9,
                borderaxespad=0.0,
            )
            for handle in leg_pt.legend_handles:
                handle.set_sizes([120])
        leg_vocab = None
        if h_vocab is not None:
            leg_vocab = ax.legend(
                [h_vocab], [r"$\Sigma$"],
                loc="lower left",
                bbox_to_anchor=(1.02, 0.0),
                fontsize=14, framealpha=0.9, borderaxespad=0.0,
            )
            for handle in leg_vocab.legend_handles:
                handle.set_sizes([120])
        if leg_nt is not None:
            ax.add_artist(leg_nt)
        if leg_pt is not None:
            ax.add_artist(leg_pt)
        ax.grid(alpha=0.25)
        stem_prefix = f"cossin_{args.method}_labeled"
        if args.no_vocab:
            stem_prefix += "_NP"
    else:
        if not args.no_vocab:
            ax.scatter(
                xy_vocab[:, 0], xy_vocab[:, 1],
                s=5, c="#9a9a9a", alpha=0.28, edgecolors="none",
                label=r"$\Sigma$", zorder=1,
            )
        ax.scatter(
            xy_nt[:, 0], xy_nt[:, 1],
            s=12, c="#1f77b4", alpha=0.55, edgecolors="none",
            label=r"$N$", zorder=2,
        )
        ax.scatter(
            xy_t[:, 0], xy_t[:, 1],
            s=12, c="#f39c12", alpha=0.55, edgecolors="none",
            label=r"$P$", zorder=3,
        )

        star_colors = ["black", "black", "black"]
        star_markers = [r"$\mathbf{v}^{(L)}$", r"$\mathbf{v}^{(R)}$", r"$\mathbf{v}^{(T)}$"]
        for i, (c, m) in enumerate(zip(star_colors, star_markers)):
            ax.scatter(
                xy_vs[i, 0], xy_vs[i, 1],
                s=800, c=c, marker=m,
                zorder=5,
            )

        ax.set_xlabel(xlabel, fontsize=24)
        ax.set_ylabel(ylabel, fontsize=24)
        ax.tick_params(labelsize=20)
        ax.axhline(0, color="#d0d0d0", linewidth=0.5, zorder=0)
        ax.axvline(0, color="#d0d0d0", linewidth=0.5, zorder=0)
        ax.set_aspect("equal", adjustable="datalim")
        # Reorder legend so entries read N, P (, Σ) top-to-bottom.
        handles, labels = ax.get_legend_handles_labels()
        if args.no_vocab:
            # Auto-collected order is N, P -> already correct.
            order = list(range(len(handles)))
        else:
            # Auto-collected order is Σ, N, P -> rotate to N, P, Σ.
            order = [1, 2, 0]
        legend = ax.legend(
            [handles[i] for i in order], [labels[i] for i in order],
            loc="best", fontsize=20, framealpha=0.9,
        )
        # Scatter dots (N / P / Σ) are drawn at s=5..12 in data so legend
        # handles are too small to read; resize them up.
        for handle in legend.legend_handles:
            handle.set_sizes([200])
        ax.grid(alpha=0.25)
        stem_prefix = Path(args.out).stem

    out_path = Path(args.out)
    # `tight_layout` shrinks the axes box without accounting for the external
    # legends in labeled mode, which then clips them on save. Reserve right-
    # side room via the rect kwarg in that case.
    if args.label_map:
        fig.tight_layout(rect=(0, 0, 0.78, 1))
    else:
        fig.tight_layout()
    out_dir = out_path.parent.parent / args.lang / out_path.parent.name
    if args.label_map:
        # Keep labeled artifacts in their own subdir so they sit alongside the
        # unlabeled cossin_pca / heatmap outputs without overwriting them.
        out_dir = out_dir / "label"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_id = Path(args.ckpt).parent.name
    stem = f"{stem_prefix}_{args.lang}_{ckpt_id}"
    extra_artists = tuple(
        a for a in (
            locals().get("leg_nt"),
            locals().get("leg_pt"),
            locals().get("leg_vocab"),
        ) if a is not None
    )
    written = []
    for ext in (".png", ".svg", ".pdf"):
        p = out_dir / (stem + ext)
        fig.savefig(
            p, dpi=140, bbox_inches="tight", pad_inches=0.15,
            bbox_extra_artists=extra_artists,
        )
        written.append(str(p))
    plt.close(fig)
    summary = (
        f"EVR: PC1={evr[0]:.1f}% PC2={evr[1]:.1f}%"
        if evr is not None
        else f"method={args.method}"
    )
    print(f"[{args.lang}] wrote {', '.join(written)}   "
          f"{summary}  total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
