"""Generate publication-quality figures for SP3 Relation Extraction analysis.

Produces 5 figures for ACL/EMNLP submission showing that
  r_ext = e_A ⋆ e_B  ≈  v_left  (or v_right)
holds in a trained HN-PCFG with freq_cnorm.

Usage:
    python scripts/generate_sp3_figures.py [--checkpoint PATH]
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Matplotlib configuration (ACL style) ──────────────────────────────
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

plt.rcParams.update(
    {
        "font.size": 9,
        "font.family": "sans-serif",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "svg",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# ── Constants ─────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = (
    "log/hn_pcfg_allproj_cnorm_tau/HNPCFG2026-03-31-12_43_40/best.pt"
)
NT = 4096
T = 8192
S_DIM = 512


# ── Core math ─────────────────────────────────────────────────────────
def circular_correlation(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    """Circular correlation  a ⋆ b = IFFT(conj(FFT(a)) * FFT(b))."""
    a_f = torch.fft.rfft(a, dim=-1)
    b_f = torch.fft.rfft(b, dim=-1)
    return torch.fft.irfft(a_f.conj() * b_f, n=n, dim=-1)


def compute_rule_probs(
    v: torch.Tensor,
    nt_emb: torch.Tensor,
    all_emb: torch.Tensor,
    tau: float,
    s_dim: int,
) -> torch.Tensor:
    """P(child | parent, relation) via HolE scoring + softmax.

    Returns: (NT+T, NT) probability matrix (softmax over children dim=0).
    """
    v_f = torch.fft.rfft(v.unsqueeze(0), dim=-1)
    parent_f = torch.fft.rfft(nt_emb, dim=-1)
    template = torch.fft.irfft(
        v_f.unsqueeze(1) * parent_f.unsqueeze(0), n=s_dim, dim=-1
    )
    scores = torch.einsum("cs, rps -> rcp", all_emb, template).squeeze(0) * tau
    return scores.softmax(dim=0)


# ── Data loading & precomputation ─────────────────────────────────────
def load_checkpoint(path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    emb = ckpt["rule_state_emb"]
    v_left = ckpt["v_left"]
    v_right = ckpt["v_right"]
    tau = ckpt["log_tau"].exp().item()
    return {
        "emb": emb,
        "nt_emb": emb[:NT],
        "all_emb": emb,
        "v_left": v_left,
        "v_right": v_right,
        "tau": tau,
    }


def precompute(data: dict, top_k: int, device: torch.device) -> dict:
    """Compute rule probabilities, top-k children, extracted relations, and cosines."""
    nt_emb = data["nt_emb"]
    all_emb = data["all_emb"]
    v_left = data["v_left"]
    v_right = data["v_right"]
    tau = data["tau"]

    results = {}
    for label, v, v_other in [
        ("left", v_left, v_right),
        ("right", v_right, v_left),
    ]:
        probs = compute_rule_probs(v, nt_emb, all_emb, tau, S_DIM)
        topk_probs, topk_indices = probs.topk(top_k, dim=0)
        topk_probs = topk_probs.t()  # (NT, k)
        topk_indices = topk_indices.t()  # (NT, k)

        # Extract relations for all top-k
        child_emb = all_emb[topk_indices.reshape(-1)].reshape(NT, top_k, S_DIM)
        parent_emb = nt_emb.unsqueeze(1).expand(-1, top_k, -1)
        r_ext = circular_correlation(parent_emb, child_emb, S_DIM)

        cos_v = F.cosine_similarity(r_ext, v.unsqueeze(0).unsqueeze(0), dim=-1)
        cos_other = F.cosine_similarity(
            r_ext, v_other.unsqueeze(0).unsqueeze(0), dim=-1
        )

        results[label] = {
            "probs": probs.cpu(),
            "topk_probs": topk_probs.cpu(),
            "topk_indices": topk_indices.cpu(),
            "r_ext": r_ext.cpu(),
            "r_ext_top1": r_ext[:, 0, :].cpu(),
            "cos_v": cos_v.cpu(),
            "cos_other": cos_other.cpu(),
            "cos_top1": cos_v[:, 0].cpu(),
        }

    return results


# ── Label helpers ─────────────────────────────────────────────────────
def load_nt_labels(path: Path) -> dict | None:
    """Load NT labels pickle.  Returns the full dict or None."""
    if not path.exists():
        print(f"  [INFO] {path} not found — skipping label-annotated figures.")
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Expect keys: nt_labels, vocab, t_labels, ...
    if "nt_labels" in data:
        print(f"  [INFO] Loaded {len(data['nt_labels'])} NT labels from {path}")
        return data
    print(f"  [INFO] Loaded labels file but unexpected format — keys: {list(data.keys())}")
    return None


def _short_child_desc(entry: tuple) -> str:
    """Extract a short description from a top-child tuple (idx, prob, label_str)."""
    if len(entry) >= 3:
        return str(entry[2])
    return str(entry[0])


def nt_label(idx: int, label_data: dict | None) -> str:
    """Generate a concise label for NT `idx` from the labels data.

    Format: 'NT{idx} [top-1-left-child-category]'
    """
    if label_data is None:
        return f"NT{idx}"
    nt_labels = label_data.get("nt_labels", {})
    info = nt_labels.get(idx)
    if info is None:
        return f"NT{idx}"
    # Extract top-1 left child category for a short descriptor
    left_top = info.get("left_top5", [])
    if left_top:
        desc = _short_child_desc(left_top[0])
        # Shorten: strip NT- / T- prefix details for brevity
        # e.g. 'T-3845(IN-like)' -> 'IN-like', 'NT-940' -> 'NT'
        import re
        m = re.search(r"\(([^)]+)\)", desc)
        if m:
            cat = m.group(1)
            return f"NT{idx} [{cat}]"
        elif desc.startswith("NT-"):
            return f"NT{idx} [NT]"
    return f"NT{idx}"


# ── Figure 1: Relation Extraction Heatmap ─────────────────────────────
def fig1_heatmap(
    precomp: dict,
    data: dict,
    label_data: dict | None,
    out: Path,
    n_parents: int = 45,
    n_children: int = 12,
) -> None:
    """Heatmap of cos(r_ext, v_left) for top parent-child pairs."""
    print("  Figure 1: Relation Extraction Heatmap ...")

    from sklearn.cluster import KMeans

    res_left = precomp["left"]
    probs_left = res_left["probs"]  # (NT+T, NT)
    v_left = data["v_left"]
    v_right = data["v_right"]
    nt_emb = data["nt_emb"]
    all_emb = data["all_emb"]

    # Select parents: highest root prob + highest incoming mass (combine both signals)
    incoming_mass = probs_left[:NT, :].sum(dim=1)  # (NT,)
    # Normalize and combine
    mass_norm = incoming_mass / incoming_mass.max()
    # Use top incoming mass as primary criterion
    parent_indices = mass_norm.topk(n_parents).indices.cpu().numpy()

    # Sort parents by clustering their top-1 r_ext vectors
    r_ext_selected = res_left["r_ext_top1"][parent_indices].numpy()
    n_clusters = min(8, n_parents)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_ids = km.fit_predict(r_ext_selected)

    # Within each cluster, sort by cos(r_ext_top1, v_left) descending
    cos_selected = res_left["cos_top1"][parent_indices].numpy()
    sort_order = np.lexsort((-cos_selected, cluster_ids))
    parent_indices = parent_indices[sort_order]
    cluster_ids = cluster_ids[sort_order]

    # Build heatmap matrices
    cos_left_matrix = np.full((n_parents, n_children), np.nan)
    cos_right_matrix = np.full((n_parents, n_children), np.nan)

    for i, pidx in enumerate(parent_indices):
        col_probs = probs_left[:, pidx]
        topk_p, topk_i = col_probs.topk(n_children)
        for j in range(n_children):
            cidx = topk_i[j].item()
            child_e = all_emb[cidx]
            parent_e = nt_emb[pidx]
            r = circular_correlation(
                parent_e.unsqueeze(0), child_e.unsqueeze(0), S_DIM
            )
            cos_left_matrix[i, j] = F.cosine_similarity(
                r, v_left.unsqueeze(0), dim=-1
            ).item()
            cos_right_matrix[i, j] = F.cosine_similarity(
                r, v_right.unsqueeze(0), dim=-1
            ).item()

    # Parent tick labels
    parent_tick_labels = [
        nt_label(int(pidx), label_data) for pidx in parent_indices
    ]

    # ── Plot ──
    fig = plt.figure(figsize=(14, 9.0))
    gs = GridSpec(1, 3, width_ratios=[5, 5, 0.3], wspace=0.30)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    norm = TwoSlopeNorm(vmin=-0.1, vcenter=0.3, vmax=0.85)

    for ax, mat, title in [
        (
            ax_left,
            cos_left_matrix,
            r"$\cos(\mathbf{r}_{\mathrm{ext}},\;\mathbf{v}_{\mathrm{left}})$",
        ),
        (
            ax_right,
            cos_right_matrix,
            r"$\cos(\mathbf{r}_{\mathrm{ext}},\;\mathbf{v}_{\mathrm{right}})$",
        ),
    ]:
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="RdYlGn",
            norm=norm,
            interpolation="nearest",
        )
        ax.set_xlabel("Child rank (by probability)", fontsize=9)
        ax.set_xticks(range(n_children))
        ax.set_xticklabels(
            [str(r + 1) for r in range(n_children)], fontsize=7
        )
        ax.set_yticks(range(n_parents))
        ax.set_yticklabels(parent_tick_labels, fontsize=5.5)
        ax.set_title(title, fontsize=10, pad=6)

        # Cluster separators
        boundaries = np.where(np.diff(cluster_ids) != 0)[0]
        for b in boundaries:
            ax.axhline(y=b + 0.5, color="white", linewidth=1.5)

    ax_left.set_ylabel("Parent NT (sorted by cluster)", fontsize=9)

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Cosine similarity", fontsize=9)

    _save(fig, out, "fig1_heatmap")


# ── Figure 2: t-SNE / PCA of relation space ──────────────────────────
def fig2_relation_tsne(
    precomp: dict,
    data: dict,
    label_data: dict | None,
    out: Path,
) -> None:
    """t-SNE of top-1 r_ext (left relation); left panel colored by cos(v_left),
    right panel colored by cos(v_right) for contrast."""
    print("  Figure 2: Relation Space t-SNE ...")

    from sklearn.manifold import TSNE

    r_ext_left = precomp["left"]["r_ext_top1"].numpy()  # (NT, 512)
    cos_left = precomp["left"]["cos_top1"].numpy()
    # cos(r_ext_left, v_right) — should be LOW if v_left is the correct relation
    cos_right_on_left = precomp["left"]["cos_other"][:, 0].numpy()
    topk_idx_left = precomp["left"]["topk_indices"][:, 0].numpy()
    v_left_np = data["v_left"].cpu().numpy()
    v_right_np = data["v_right"].cpu().numpy()

    # t-SNE on r_ext_left + reference vectors
    combined = np.vstack([r_ext_left, v_left_np[None], v_right_np[None]])
    print("    Running t-SNE (perplexity=40) ...")
    tsne = TSNE(
        n_components=2, perplexity=40, random_state=42,
        max_iter=1000, init="pca",
    )
    proj = tsne.fit_transform(combined)
    pts = proj[:-2]
    v_left_pt = proj[-2]
    v_right_pt = proj[-1]

    is_nt_child = topk_idx_left < NT

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    for ax, cos_vals, cbar_label, panel_title in [
        (
            axes[0],
            cos_left,
            r"$\cos(\mathbf{r}_{\mathrm{ext}},\,\mathbf{v}_{\mathrm{left}})$",
            r"Colored by $\cos(\mathbf{r}_{\mathrm{ext}},\,\mathbf{v}_{\mathrm{left}})$",
        ),
        (
            axes[1],
            cos_right_on_left,
            r"$\cos(\mathbf{r}_{\mathrm{ext}},\,\mathbf{v}_{\mathrm{right}})$",
            r"Colored by $\cos(\mathbf{r}_{\mathrm{ext}},\,\mathbf{v}_{\mathrm{right}})$",
        ),
    ]:
        # T children — small crosses
        sc = ax.scatter(
            pts[~is_nt_child, 0],
            pts[~is_nt_child, 1],
            c=cos_vals[~is_nt_child],
            s=4, alpha=0.35, cmap="RdYlGn",
            vmin=-0.05, vmax=0.8,
            marker="x", linewidths=0.5,
            label="T child", rasterized=True,
        )
        # NT children — circles
        ax.scatter(
            pts[is_nt_child, 0],
            pts[is_nt_child, 1],
            c=cos_vals[is_nt_child],
            s=6, alpha=0.5, cmap="RdYlGn",
            vmin=-0.05, vmax=0.8,
            marker="o", edgecolors="none",
            label="NT child", rasterized=True,
        )
        # Reference vectors
        ax.scatter(
            v_left_pt[0], v_left_pt[1],
            c="tab:blue", marker="*", s=220, edgecolors="black",
            linewidths=0.8, zorder=10,
            label=r"$\mathbf{v}_{\mathrm{left}}$",
        )
        ax.scatter(
            v_right_pt[0], v_right_pt[1],
            c="tab:red", marker="*", s=220, edgecolors="black",
            linewidths=0.8, zorder=10,
            label=r"$\mathbf{v}_{\mathrm{right}}$",
        )

        ax.set_xlabel("t-SNE dim 1", fontsize=9)
        ax.set_ylabel("t-SNE dim 2", fontsize=9)
        ax.set_title(panel_title, fontsize=10)
        ax.legend(
            loc="lower left", fontsize=7, markerscale=1.5,
            framealpha=0.85, edgecolor="gray",
        )

        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label(cbar_label, fontsize=8)

    fig.tight_layout()
    _save(fig, out, "fig2_relation_tsne")


# ── Figure 3: Cosine vs Rule Probability (hexbin) ────────────────────
def fig3_cosine_vs_prob(
    precomp: dict,
    out: Path,
) -> None:
    """Hexbin scatter: log P(child|parent) vs cos(r_ext, v)."""
    print("  Figure 3: Cosine vs Rule Probability ...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for ax, side, title_v in [
        (axes[0], "left", r"$\mathbf{v}_{\mathrm{left}}$"),
        (axes[1], "right", r"$\mathbf{v}_{\mathrm{right}}$"),
    ]:
        probs = precomp[side]["topk_probs"][:, 0].numpy()  # top-1
        cos_vals = precomp[side]["cos_top1"].numpy()
        log_probs = np.log10(np.clip(probs, 1e-12, None))

        hb = ax.hexbin(
            log_probs, cos_vals,
            gridsize=60, cmap="viridis", mincnt=1,
            linewidths=0.2, edgecolors="face",
        )
        cb = fig.colorbar(hb, ax=ax, shrink=0.85)
        cb.set_label("Count", fontsize=8)

        ax.set_xlabel(r"$\log_{10}\, P(\mathrm{child} \mid \mathrm{parent})$", fontsize=9)
        ax.set_ylabel(f"cos(r_ext, {title_v})", fontsize=9)
        ax.set_title(f"Relation similarity vs rule probability — {title_v}", fontsize=10)

        # Reference lines
        ax.axhline(y=cos_vals.mean(), color="white", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.text(
            log_probs.min() + 0.05,
            cos_vals.mean() + 0.02,
            f"mean = {cos_vals.mean():.3f}",
            color="white", fontsize=7, alpha=0.9,
        )

    fig.tight_layout()
    _save(fig, out, "fig3_cosine_vs_prob")


# ── Figure 4: Cluster-averaged cosine heatmap ────────────────────────
def fig4_cluster_heatmap(
    precomp: dict,
    out: Path,
    n_clusters: int = 10,
) -> None:
    """K-means clustering of r_ext, then mean cosine per cluster."""
    print("  Figure 4: Cluster-averaged Cosine Heatmap ...")

    from sklearn.cluster import KMeans

    r_ext = precomp["left"]["r_ext_top1"].numpy()
    cos_left = precomp["left"]["cos_top1"].numpy()
    cos_right = precomp["left"]["cos_other"][:, 0].numpy()  # cos(r_left_ext, v_right)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_ids = km.fit_predict(r_ext)

    # Aggregate
    mean_cos_left = np.zeros(n_clusters)
    mean_cos_right = np.zeros(n_clusters)
    cluster_sizes = np.zeros(n_clusters, dtype=int)
    for c in range(n_clusters):
        mask = cluster_ids == c
        mean_cos_left[c] = cos_left[mask].mean()
        mean_cos_right[c] = cos_right[mask].mean()
        cluster_sizes[c] = mask.sum()

    # Sort clusters by mean_cos_left descending
    order = np.argsort(-mean_cos_left)
    mean_cos_left = mean_cos_left[order]
    mean_cos_right = mean_cos_right[order]
    cluster_sizes = cluster_sizes[order]

    mat = np.stack([mean_cos_left, mean_cos_right], axis=1)  # (k, 2)

    fig, ax = plt.subplots(figsize=(4.0, 5.0))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-0.05, vmax=0.75)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [r"$\mathbf{v}_{\mathrm{left}}$", r"$\mathbf{v}_{\mathrm{right}}$"],
        fontsize=9,
    )
    ax.set_yticks(range(n_clusters))
    ylabels = [f"C{order[i]}  (n={cluster_sizes[i]})" for i in range(n_clusters)]
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_ylabel("Cluster (sorted by left cosine)", fontsize=9)
    ax.set_title("Mean cosine similarity per cluster", fontsize=10, pad=8)

    # Annotate cells
    for i in range(n_clusters):
        for j in range(2):
            val = mat[i, j]
            color = "white" if val < 0.35 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Mean cosine similarity", fontsize=8)

    fig.tight_layout()
    _save(fig, out, "fig4_cluster_heatmap")


# ── Figure 5: NT vs T child violin plot ──────────────────────────────
def fig5_nt_vs_t_violin(
    precomp: dict,
    out: Path,
) -> None:
    """Violin plot of cos(r_ext, v) split by child type and relation."""
    print("  Figure 5: NT vs T Child Violin Plot ...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, side, title_v in [
        (axes[0], "left", r"$\mathbf{v}_{\mathrm{left}}$"),
        (axes[1], "right", r"$\mathbf{v}_{\mathrm{right}}$"),
    ]:
        topk_idx = precomp[side]["topk_indices"][:, 0].numpy()
        cos_vals = precomp[side]["cos_top1"].numpy()
        is_nt = topk_idx < NT

        cos_nt = cos_vals[is_nt]
        cos_t = cos_vals[~is_nt]

        data_list = [cos_nt, cos_t]
        positions = [1, 2]

        parts = ax.violinplot(
            data_list, positions=positions, showmeans=True,
            showmedians=True, showextrema=False,
        )

        # Style violins
        colors = ["#4C72B0", "#55A868"]
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmeans"].set_color("black")
        parts["cmeans"].set_linewidth(1.5)
        parts["cmedians"].set_color("gray")
        parts["cmedians"].set_linestyle("--")

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"NT child\n(n={len(cos_nt)})", f"T child\n(n={len(cos_t)})"],
            fontsize=8,
        )
        ax.set_ylabel(f"cos(r_ext, {title_v})", fontsize=9)
        ax.set_title(f"Child type distribution — {title_v}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Annotate means with background for readability
        for pos, d, color in zip(positions, data_list, colors):
            ax.annotate(
                f"$\\mu$={d.mean():.3f}",
                xy=(pos, d.mean()),
                xytext=(pos + 0.35, d.mean() + 0.04),
                fontsize=7.5, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.8, lw=0.5),
                arrowprops=dict(arrowstyle="-", color=color, lw=0.5),
            )

    fig.tight_layout()
    _save(fig, out, "fig5_nt_vs_t_violin")


# ── Save helper ───────────────────────────────────────────────────────
def _save(fig: plt.Figure, out: Path, name: str) -> None:
    p = out / f"{name}.svg"
    fig.savefig(p, format="svg")
    print(f"    -> {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--output-dir", default="results/sp3/figures")
    p.add_argument("--labels-path", default="results/sp3/nt_labels.pkl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")
    print(f"Output     : {out}")

    # Load model
    data = load_checkpoint(args.checkpoint, device)
    print(f"tau = {data['tau']:.4f}")

    # Precompute
    print("\nPrecomputing rule probs + relation extraction ...")
    precomp = precompute(data, args.top_k, device)
    print("  Done.\n")

    # Move relation vectors to CPU for figure code
    data_cpu = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}

    # Load NT labels (may not exist yet)
    label_data = load_nt_labels(Path(args.labels_path))

    # ── Generate figures that do NOT need labels ──
    fig3_cosine_vs_prob(precomp, out)
    fig4_cluster_heatmap(precomp, out)
    fig5_nt_vs_t_violin(precomp, out)
    fig2_relation_tsne(precomp, data_cpu, label_data, out)

    # ── Heatmap (benefits from labels but works without) ──
    fig1_heatmap(precomp, data_cpu, label_data, out)

    print("\n  All figures saved to:", out)


if __name__ == "__main__":
    main()
