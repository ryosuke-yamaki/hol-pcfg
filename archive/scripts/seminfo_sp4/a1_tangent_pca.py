"""A1 (Tangent-PCA variant): visualize {r_ext_i} in T_v S^{d-1}.

Under the Phase-Only Manifold all vectors have ||x|| = 1 (Parseval with
|FFT[k]|=1), so they live on the unit sphere S^{d-1}. We lift each r_i
to the tangent plane at v via the sphere log map,
    log_v(x) = (θ / sin θ) (x - cos θ · v),  θ = arccos(v·x),
and apply *uncentered* PCA (SVD) on {log_v(r_i)}. This keeps v at the
origin of the plot; ||log_v(r_i)|| equals the geodesic distance θ_i.

Output: one SVG per checkpoint, 3 subplots (rule_left, rule_right,
terminal) with reference circles at 15°, 30°, 60°, 90° (PC-space
radii; these are lower bounds on the true geodesic θ_i because the 2D
plane is a projection of the (d-1)-dim tangent space).
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


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


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


def log_map(v_unit: torch.Tensor, x_unit: torch.Tensor,
            eps: float = 1e-9) -> tuple[torch.Tensor, torch.Tensor]:
    """Sphere log map at v_unit.

    v_unit: (d,) unit,   x_unit: (N, d) unit.
    Returns (u: (N, d) in T_v, theta: (N,) geodesic distance in rad).
    """
    cos_theta = (x_unit * v_unit).sum(-1).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta).clamp(min=eps)
    coeff = (theta / sin_theta).unsqueeze(-1)
    shifted = x_unit - cos_theta.unsqueeze(-1) * v_unit.unsqueeze(0)
    return coeff * shifted, theta


def load_state(ckpt_path: str) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    return {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}


@torch.no_grad()
def extract_cloud(
    parents: torch.Tensor, candidates: torch.Tensor,
    v: torch.Tensor, tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    templates = circconv(v, parents)
    scores = candidates @ templates.t()
    logits = scores * tau
    logp = F.log_softmax(logits, dim=-2)
    top_idx = logits.argmax(dim=-2)
    top_logp = logp.gather(dim=-2, index=top_idx.unsqueeze(0)).squeeze(0)
    r_ext = circcorr(parents, candidates[top_idx])
    return r_ext, top_logp


V_LABEL = {"rule_left": "v_left", "rule_right": "v_right", "terminal": "v_term"}


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

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))

    for ax, (name, parents, cands, v, tau) in zip(axes, conditions):
        r_ext, logp = extract_cloud(parents, cands, v, tau)
        r_unit = F.normalize(r_ext, dim=-1)
        v_unit = v / v.norm().clamp(min=1e-12)

        u, theta = log_map(v_unit, r_unit)         # u: (N, d), theta: (N,) rad
        mean_theta_deg = math.degrees(theta.mean().item())

        cos_iv = F.cosine_similarity(r_ext, v.unsqueeze(0), dim=-1)
        mean_cos = cos_iv.mean().item()
        mu_r = r_unit.mean(dim=0)
        R = mu_r.norm().item()
        mu_r_unit = mu_r / mu_r.norm().clamp(min=1e-12)
        theta_R_deg = math.degrees(math.acos(
            torch.clamp(torch.dot(mu_r_unit, v_unit), -1.0, 1.0).item()
        ))

        # Uncentered PCA via SVD of u  — keeps v (u=0) at the origin
        u_np = u.numpy()
        U_, S_, Vt_ = np.linalg.svd(u_np, full_matrices=False)
        xy = U_[:, :2] * S_[:2]                    # (N, 2) in rad
        evr = (S_[:2] ** 2) / (S_ ** 2).sum()
        xy_deg = np.degrees(xy)

        # Project mean tangent vector into PC space
        mu_u = u.mean(dim=0).numpy()
        mu_xy_deg = np.degrees(mu_u @ Vt_[:2].T)

        # Reference circles (PC-space radii in degrees)
        for r_deg in (15, 30, 60, 90):
            ax.add_patch(plt.Circle(
                (0, 0), r_deg, fill=False,
                color="#b8b8b8", linestyle="--", linewidth=0.55, zorder=0,
            ))
            ax.text(r_deg * 0.707, r_deg * 0.707, f"{r_deg}°",
                    fontsize=7, color="#909090",
                    ha="center", va="center", zorder=0)

        logp_np = logp.numpy()
        vmin = float(np.percentile(logp_np, 2))
        vmax = float(np.percentile(logp_np, 98))

        sc = ax.scatter(
            xy_deg[:, 0], xy_deg[:, 1],
            c=logp_np, cmap="viridis",
            s=16, alpha=0.70, edgecolors="none",
            vmin=vmin, vmax=vmax, zorder=2,
            label=f"log_v(r_i) (n={len(xy_deg)})",
        )
        ax.scatter(
            [0], [0], c="#d62246", marker="*", s=520,
            edgecolors="black", linewidths=1.4,
            label=f"{V_LABEL[name]} (origin)", zorder=5,
        )
        ax.scatter(
            [mu_xy_deg[0]], [mu_xy_deg[1]],
            facecolors="none", edgecolors="#222222", linewidths=1.8,
            s=300, marker="o", zorder=4,
            label=f"⟨log_v(r_i)⟩",
        )

        ax.set_title(
            f"{name}\nmean_cos={mean_cos:.3f}  R={R:.3f}  "
            f"θ(mean dir)={theta_R_deg:.1f}°  ⟨θ_i⟩={mean_theta_deg:.1f}°",
            fontsize=10,
        )
        ax.set_xlabel(f"PC1 (T_v, {evr[0]*100:.1f}%)  [deg]")
        ax.set_ylabel(f"PC2 (T_v, {evr[1]*100:.1f}%)  [deg]")
        ax.axhline(0, color="#e0e0e0", linewidth=0.3, zorder=0)
        ax.axvline(0, color="#e0e0e0", linewidth=0.3, zorder=0)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        plt.colorbar(sc, ax=ax, shrink=0.72,
                     label="logP(top-1 child | parent)")

    fig.suptitle(
        f"A1 (Tangent PCA at v): log_v(r_ext) — {args.lang}",
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
