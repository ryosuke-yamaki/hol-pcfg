"""C0: Entity-wide phase landscape heatmap.

For a HN-PCFG checkpoint, extract the full FFT phases (k = 0..d/2) of
every entity embedding group and display them as a cyclic heatmap.
Under the phase-only manifold, DC (k=0) and Nyquist (k=d/2) are
discrete: their phase is either 0 (sign = +1) or π (sign = −1), but
they are plotted on the same [−π, π] cyclic scale as the interior
bins. v_left / v_right / v_term appear as a 3-row reference strip on
top.

Rows within each entity group are re-ordered by hierarchical (Ward)
clustering on the (cos theta, sin theta) embedding so that entities
with similar phase profiles appear adjacent. For NT and T panels the
dendrogram itself is plotted alongside the heatmap.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram


def set_global_seed(seed: int) -> None:
    """Fix Python/NumPy/Torch global RNGs in addition to the explicit
    np.random.default_rng(seed) used by subsample(); guards against future
    RNG calls being added without being threaded through args.seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


_CYCLIC_DEFAULT = plt.cm.twilight_shifted
_CYCLIC_NAME = "twilight_shifted"


# ---------------------------------------------------------------------------
# Loading / phase extraction
# ---------------------------------------------------------------------------


def load_state(ckpt_path: str) -> dict:
    """Load state dict from either a Lightning .ckpt or a raw torch.save .pt."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    if any(k.startswith("model.") for k in sd.keys()):
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    return dict(sd)


def full_phase(x: torch.Tensor) -> np.ndarray:
    """Return (N, d/2+1) full FFT phases including DC and Nyquist."""
    X = torch.fft.rfft(x, dim=-1)
    return torch.angle(X).numpy()


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_linkage(theta: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """Return (Z, leaf_order) from Ward linkage on (cos, sin) embedding."""
    if theta.shape[0] <= 2:
        return None, np.arange(theta.shape[0])
    cs = np.concatenate([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    Z = linkage(cs, method="ward")
    return Z, leaves_list(Z)


def subsample(theta, n_max, seed=42):
    N = theta.shape[0]
    if N <= n_max:
        return theta, np.arange(N)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=n_max, replace=False)
    idx.sort()
    return theta[idx], idx


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_heatmap(ax, theta, title, yticklabels=None):
    im = ax.imshow(
        theta, aspect="auto", cmap=_CYCLIC_DEFAULT,
        vmin=-np.pi, vmax=np.pi, interpolation="nearest",
    )
    if yticklabels is None:
        ax.set_yticks([])
    else:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=22)
        ax.tick_params(axis="y", pad=2, length=0)
    if title is not None:
        ax.set_title(title, fontsize=24, loc="left", pad=6)
    return im


def _draw_dendrogram(ax, Z):
    """Dendrogram on the left, leaves aligned with heatmap rows (top → bottom)."""
    dendrogram(
        Z, orientation="left", ax=ax,
        no_labels=True, color_threshold=0,
        above_threshold_color="#555555",
        link_color_func=lambda _: "#555555",
    )
    # scipy default plots leaf index 0 at the bottom; invert so row 0 is at top
    ax.invert_yaxis()
    ax.set_axis_off()


def _make_panel(fig, gs_slice, theta, title, Z=None, yticklabels=None):
    """Panel = [dendrogram slot] + heatmap. The left slot is reserved
    even when Z is None so heatmap widths align across panels."""
    sub = gs_slice.subgridspec(
        1, 2, width_ratios=[0.07, 1], wspace=0.005,
    )
    ax_dendro = fig.add_subplot(sub[0, 0])
    ax_main = fig.add_subplot(sub[0, 1])
    if Z is not None:
        _draw_dendrogram(ax_dendro, Z)
    else:
        ax_dendro.set_axis_off()
    im = _draw_heatmap(ax_main, theta, title, yticklabels=yticklabels)
    return ax_main, im


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--NT", type=int, default=1024)
    ap.add_argument("--vocab_sample", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_global_seed(args.seed)

    t0 = time.time()
    sd = load_state(args.ckpt)
    rse = sd["rule_state_emb"].float()
    vocab = sd["vocab_emb"].float().t()  # (V, d)
    v_left = sd["v_left"].float().unsqueeze(0)
    v_right = sd["v_right"].float().unsqueeze(0)
    v_term = sd["v_term"].float().unsqueeze(0)

    NT = args.NT
    nt_emb = rse[:NT]
    t_emb = rse[NT:]

    v_stack = torch.cat([v_left, v_right, v_term], dim=0)       # (3, d)
    theta_v = full_phase(v_stack)                                # (3, d/2+1)
    v_names = [r"$\mathbf{v}^{(L)}$", r"$\mathbf{v}^{(R)}$", r"$\mathbf{v}^{(T)}$"]

    theta_nt = full_phase(nt_emb)                                # (NT, 257)
    theta_t = full_phase(t_emb)
    theta_vocab_full = full_phase(vocab)
    theta_vocab, _ = subsample(theta_vocab_full, args.vocab_sample, seed=args.seed)

    n_freq = theta_v.shape[1]
    print(f"[{args.lang}] loaded  (NT={len(theta_nt)}, T={len(theta_t)}, "
          f"V_full={len(theta_vocab_full)}, V_plot={len(theta_vocab)}, "
          f"bins={n_freq})  dt={time.time()-t0:.1f}s")

    # Ward clustering on every entity group; keep linkage for dendrograms.
    t1 = time.time()
    Z_nt, o_nt = cluster_linkage(theta_nt)
    theta_nt = theta_nt[o_nt]
    Z_t, o_t = cluster_linkage(theta_t)
    theta_t = theta_t[o_t]
    Z_v, o_v = cluster_linkage(theta_vocab)
    theta_vocab = theta_vocab[o_v]
    print(f"[{args.lang}] cluster-sort total={time.time()-t1:.1f}s")

    # Figure layout
    heights = [0.85, 2.8, 3.8, 3.8]
    fig = plt.figure(figsize=(15.5, sum(heights) + 2.0))
    gs = GridSpec(
        nrows=4, ncols=1,
        height_ratios=heights,
        hspace=0.20, left=0.07, right=0.89, top=0.93, bottom=0.07,
    )

    ax_ref, _ = _make_panel(fig, gs[0, 0], theta_v, "Relation Vectors",
                            Z=None, yticklabels=v_names)
    ax_nt, im = _make_panel(fig, gs[1, 0], theta_nt,
                            rf"$N$ (Nonterminals, $n={len(theta_nt)}$)", Z=Z_nt)
    ax_t, _ = _make_panel(fig, gs[2, 0], theta_t,
                          rf"$P$ (Preterminals, $n={len(theta_t)}$)", Z=Z_t)
    ax_v, _ = _make_panel(fig, gs[3, 0], theta_vocab,
                          rf"$\Sigma$ (Vocabulary Items, $n={args.vocab_sample}$, subsample)",
                          Z=Z_v)

    # Shared x range on heatmap axes
    for ax in (ax_ref, ax_nt, ax_t, ax_v):
        ax.set_xlim(-0.5, n_freq - 0.5)
    for ax in (ax_ref, ax_nt, ax_t):
        ax.set_xticks([])

    xt = np.array([0, 64, 128, 192, 256])
    ax_v.set_xticks(xt)
    ax_v.set_xticklabels([rf"$k={k}$" for k in xt], fontsize=20)

    # Colorbar
    cbar_ax = fig.add_axes([0.920, 0.10, 0.010, 0.78])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cb.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    cb.ax.tick_params(labelsize=20, pad=2)
    cb.set_label("phase [rad]", fontsize=22, labelpad=-30)

    out_path = Path(args.out)
    # Insert <lang> as a directory above the run-id dir so output lays out as
    # results/c0_phase_landscape/<lang>/<run_id>/<stem>_<lang>_<ckpt_id>.{ext}
    out_dir = out_path.parent.parent / args.lang / out_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_id = Path(args.ckpt).parent.name
    stem = f"{out_path.stem}_{args.lang}_{ckpt_id}"
    written = []
    for ext in (".png", ".svg", ".pdf"):
        p = out_dir / (stem + ext)
        fig.savefig(p, bbox_inches="tight", pad_inches=0.005, dpi=140)
        written.append(str(p))
    plt.close(fig)
    print(f"[{args.lang}] wrote {', '.join(written)}   total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
