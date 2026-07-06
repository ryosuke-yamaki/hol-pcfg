"""Relation residual phase heatmap.

For each top-1 (parent A, child B) pair chosen by the model, compute
the per-frequency phase residual

    delta_k = wrap(theta_B[k] - theta_A[k] - theta_v[k])

which should be near 0 under SP4's "r_ext ~= v_relation" claim. This
script displays delta_k as a cyclic heatmap with rows = parent-child
pairs, columns = frequency bins (interior k = 1..d/2-1). A tight
band of one color (ideally cream = 0) indicates that the relation
law is satisfied at that frequency; a noisy column indicates the law
fails there.

Three panels:
  1. Rule-Left   (1024 NT parents -> top-1 child via v_left)
  2. Rule-Right  (1024 NT parents -> top-1 child via v_right)
  3. Terminal    (2048 preterminals -> top-1 word via v_term)

Dendrograms on the left show Ward hierarchical clustering of
parent-child pairs on the (cos delta, sin delta) embedding, so pairs
with similar residual patterns become adjacent.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram


_CYCLIC = plt.cm.twilight_shifted


# ---------------------------------------------------------------------------
# Model primitives
# ---------------------------------------------------------------------------


def load_state(ckpt_path: str) -> dict:
    """Load state dict from either a Lightning .ckpt or a raw torch.save .pt."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    if any(k.startswith("model.") for k in sd.keys()):
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    return dict(sd)


def circconv_template(v: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    s_dim = source.shape[-1]
    vf = torch.fft.rfft(v)
    sf = torch.fft.rfft(source, dim=-1)
    return torch.fft.irfft(vf.unsqueeze(0) * sf, n=s_dim, dim=-1)


def hol_score(v, source, target, tau):
    """HolE score matrix (target, source)."""
    template = circconv_template(v, source)
    return (target @ template.t()) * tau


def interior_phase(x: torch.Tensor) -> torch.Tensor:
    """Return phases at interior bins k = 1..d/2-1, shape (..., d/2-1)."""
    X = torch.fft.rfft(x, dim=-1)
    return torch.angle(X[..., 1:-1])


def wrap(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


@torch.no_grad()
def compute_residuals(parents, candidates, v, tau) -> np.ndarray:
    """Return (P, d/2-1) residual phases for top-1 child per parent."""
    scores = hol_score(v, parents, candidates, tau)     # (C, P)
    top_idx = scores.argmax(dim=-2)                     # (P,)
    top_children = candidates[top_idx]                  # (P, d)

    phi_p = interior_phase(parents)                     # (P, d/2-1)
    phi_c = interior_phase(top_children)                # (P, d/2-1)
    phi_v = interior_phase(v)                           # (d/2-1,)

    delta = wrap(phi_c - phi_p - phi_v.unsqueeze(0))
    return delta.numpy()


# ---------------------------------------------------------------------------
# Clustering + drawing
# ---------------------------------------------------------------------------


def cluster_linkage(x: np.ndarray):
    if x.shape[0] <= 2:
        return None, np.arange(x.shape[0])
    cs = np.concatenate([np.cos(x), np.sin(x)], axis=1).astype(np.float32)
    Z = linkage(cs, method="ward")
    return Z, leaves_list(Z)


def _draw_heatmap(ax, delta, title):
    im = ax.imshow(
        delta, aspect="auto", cmap=_CYCLIC,
        vmin=-np.pi, vmax=np.pi, interpolation="nearest",
    )
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=12, loc="left", pad=6)
    return im


def _draw_dendrogram(ax, Z):
    dendrogram(
        Z, orientation="left", ax=ax,
        no_labels=True, color_threshold=0,
        above_threshold_color="#555555",
        link_color_func=lambda _: "#555555",
    )
    ax.invert_yaxis()
    ax.set_axis_off()


def _make_panel(fig, gs_slice, delta, Z, title):
    sub = gs_slice.subgridspec(
        1, 2, width_ratios=[0.07, 1], wspace=0.005,
    )
    ax_dendro = fig.add_subplot(sub[0, 0])
    ax_main = fig.add_subplot(sub[0, 1])
    if Z is not None:
        _draw_dendrogram(ax_dendro, Z)
    else:
        ax_dendro.set_axis_off()
    im = _draw_heatmap(ax_main, delta, title)
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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    sd = load_state(args.ckpt)
    rse = sd["rule_state_emb"].float()
    vocab = sd["vocab_emb"].float().t()
    v_left = sd["v_left"].float()
    v_right = sd["v_right"].float()
    v_term = sd["v_term"].float()
    tau_rule = sd["log_tau_rule"].float().exp()
    tau_term = sd["log_tau_term"].float().exp()

    NT = args.NT
    nt_emb = rse[:NT]
    t_emb = rse[NT:]

    # Compute residuals
    delta_L = compute_residuals(nt_emb, rse, v_left, tau_rule)
    delta_R = compute_residuals(nt_emb, rse, v_right, tau_rule)
    delta_T = compute_residuals(t_emb, vocab, v_term, tau_term)

    n_freq_int = delta_L.shape[1]
    print(f"[{args.lang}] computed residuals  "
          f"L={delta_L.shape}, R={delta_R.shape}, T={delta_T.shape}  "
          f"interior_bins={n_freq_int}  dt={time.time()-t0:.1f}s")

    # Summary stats
    def _summary(delta, name):
        R_k = np.abs(np.exp(1j * delta).mean(axis=0))
        frac_close = (np.abs(delta) < np.pi / 4).mean()
        print(f"   {name:12s}: mean|δ|={np.abs(delta).mean():.3f} rad  "
              f"R_k mean={R_k.mean():.3f}  frac(|δ|<π/4)={frac_close:.3f}")
    _summary(delta_L, "rule_left")
    _summary(delta_R, "rule_right")
    _summary(delta_T, "terminal")

    # Cluster sort
    t1 = time.time()
    Z_L, o_L = cluster_linkage(delta_L); delta_L = delta_L[o_L]
    Z_R, o_R = cluster_linkage(delta_R); delta_R = delta_R[o_R]
    Z_T, o_T = cluster_linkage(delta_T); delta_T = delta_T[o_T]
    print(f"[{args.lang}] cluster-sort  dt={time.time()-t1:.1f}s")

    # Figure
    heights = [3.6, 3.6, 4.2]
    fig = plt.figure(figsize=(15.5, sum(heights) + 1.6))
    gs = GridSpec(
        nrows=3, ncols=1,
        height_ratios=heights,
        hspace=0.4, left=0.06, right=0.94, top=0.94, bottom=0.07,
    )

    ax_L, im = _make_panel(
        fig, gs[0, 0], delta_L, Z_L,
        f"Rule-Left   N = {delta_L.shape[0]} pairs",
    )
    ax_R, _ = _make_panel(
        fig, gs[1, 0], delta_R, Z_R,
        f"Rule-Right   N = {delta_R.shape[0]} pairs",
    )
    ax_T, _ = _make_panel(
        fig, gs[2, 0], delta_T, Z_T,
        f"Terminal   N = {delta_T.shape[0]} pairs",
    )

    for ax in (ax_L, ax_R, ax_T):
        ax.set_xlim(-0.5, n_freq_int - 0.5)
    ax_L.set_xticks([]); ax_R.set_xticks([])
    xt = np.linspace(0, n_freq_int - 1, 6).astype(int)
    ax_T.set_xticks(xt)
    ax_T.set_xticklabels([f"k={k+1}" for k in xt], fontsize=10)
    ax_T.set_xlabel("Frequency bin (interior: k = 1 ... d/2-1)",
                    fontsize=11, labelpad=6)

    cbar_ax = fig.add_axes([0.955, 0.10, 0.010, 0.78])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cb.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    cb.ax.tick_params(labelsize=10)
    cb.set_label(r"residual phase $\delta = $ wrap($\theta_B - \theta_A - \theta_v$)   [rad]",
                 fontsize=11, labelpad=6)

    fig.suptitle(
        f"Relation Residual Phase - {args.lang.capitalize()}",
        fontsize=14, y=0.98,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[{args.lang}] wrote {out_path}   total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
