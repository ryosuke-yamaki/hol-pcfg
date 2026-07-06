"""A1: Visualize extracted relation cloud {r_ext_i} with v_relation.

For each condition (rule_left / rule_right / terminal), compute
r_ext_i = circcorr(parent_i, top1_child_i) for all parents, unit-
normalize {r_i} together with v_relation, fit a standard PCA on that
set, and scatter r_i colored by logP(top-1 child | parent). v_relation
is overlaid as a star. Summary stats (mean_cos, R, theta_deg) shown in
the subplot title.

Output: one SVG per checkpoint with 3 side-by-side subplots.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


def circcorr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    s_dim = x.shape[-1]
    xf = torch.fft.rfft(x, dim=-1)
    yf = torch.fft.rfft(y, dim=-1)
    return torch.fft.irfft(xf.conj() * yf, n=s_dim, dim=-1)


def circconv(v: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    s_dim = source.shape[-1]
    vf = torch.fft.rfft(v, dim=-1)
    sf = torch.fft.rfft(source, dim=-1)
    return torch.fft.irfft(vf.unsqueeze(0) * sf, n=s_dim, dim=-1)


def load_state(ckpt_path: str) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    return {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}


@torch.no_grad()
def extract_cloud(
    parents: torch.Tensor,      # (P, d)
    candidates: torch.Tensor,   # (C, d)
    v: torch.Tensor,            # (d,)
    tau: torch.Tensor,          # scalar
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (r_ext (P, d), logP_top1 per parent (P,))."""
    templates = circconv(v, parents)              # (P, d)
    scores = candidates @ templates.t()           # (C, P)
    logits = scores * tau
    logp = F.log_softmax(logits, dim=-2)          # over candidates
    top_idx = logits.argmax(dim=-2)               # (P,)
    top_logp = logp.gather(dim=-2, index=top_idx.unsqueeze(0)).squeeze(0)
    r_ext = circcorr(parents, candidates[top_idx])
    return r_ext, top_logp


V_LABEL = {"rule_left": "v_left", "rule_right": "v_right", "terminal": "v_term"}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--NT", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sd = load_state(args.ckpt)
    rse = sd["rule_state_emb"].float()
    vocab = sd["vocab_emb"].float().t()
    NT = args.NT
    nt_emb = rse[:NT]
    t_emb = rse[NT:]

    v_left = sd["v_left"].float()
    v_right = sd["v_right"].float()
    v_term = sd["v_term"].float()
    tau_rule = sd["log_tau_rule"].float().exp()
    tau_term = sd["log_tau_term"].float().exp()

    conditions = [
        ("rule_left",  nt_emb, rse,   v_left,  tau_rule),
        ("rule_right", nt_emb, rse,   v_right, tau_rule),
        ("terminal",   t_emb,  vocab, v_term,  tau_term),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))

    for ax, (name, parents, cands, v, tau) in zip(axes, conditions):
        r_ext, logp = extract_cloud(parents, cands, v, tau)

        cos_iv = F.cosine_similarity(r_ext, v.unsqueeze(0), dim=-1)
        mean_cos = cos_iv.mean().item()
        u = F.normalize(r_ext, dim=-1)
        mu = u.mean(dim=0)
        R = mu.norm().item()
        mu_unit = mu / mu.norm().clamp(min=1e-12)
        v_unit_t = v / v.norm().clamp(min=1e-12)
        theta_deg = math.degrees(math.acos(
            torch.clamp(torch.dot(mu_unit, v_unit_t), -1.0, 1.0).item()
        ))

        r_unit = u.numpy()
        v_unit = v_unit_t.numpy()[None, :]
        stacked = np.vstack([r_unit, v_unit])
        pca = PCA(n_components=2, random_state=0)
        xy = pca.fit_transform(stacked)
        evr = pca.explained_variance_ratio_
        r_xy, v_xy = xy[:-1], xy[-1:]

        logp_np = logp.numpy()
        vmin = float(np.percentile(logp_np, 2))
        vmax = float(np.percentile(logp_np, 98))

        sc = ax.scatter(
            r_xy[:, 0], r_xy[:, 1],
            c=logp_np, cmap="viridis",
            s=16, alpha=0.70, edgecolors="none",
            vmin=vmin, vmax=vmax, zorder=2,
            label=f"r_ext (n={len(r_xy)})",
        )
        ax.scatter(
            v_xy[:, 0], v_xy[:, 1],
            c="#d62246", marker="*", s=520,
            edgecolors="black", linewidths=1.4,
            label=V_LABEL[name], zorder=4,
        )

        ax.set_title(
            f"{name}\nmean_cos={mean_cos:.3f}  R={R:.3f}  θ={theta_deg:.1f}°",
            fontsize=11,
        )
        ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
        ax.axhline(0, color="#d0d0d0", linewidth=0.3, zorder=0)
        ax.axvline(0, color="#d0d0d0", linewidth=0.3, zorder=0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        plt.colorbar(sc, ax=ax, shrink=0.72,
                     label="logP(top-1 child | parent)")

    fig.suptitle(
        f"A1: Extracted relation cloud vs. v — {args.lang}",
        fontsize=13, y=1.02,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
