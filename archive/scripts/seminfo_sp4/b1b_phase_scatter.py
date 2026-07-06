"""B1-b: Bivariate phase scatter at selected frequencies.

For each language plots phi_i(k1) vs phi_i(k2) across all entities
(NT and T split out) for three frequency regimes:
  * Low freq   : (k=1, k=2)
  * Mid freq   : (k=128, k=129)
  * Top-R      : the two frequencies with the highest mean resultant
                 length across entities (most "consensus" phases)
v_left/v_right/v_term overlaid as stars.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_state(ckpt_path: str) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    return {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}


V_COLORS = {"v_left": "#d62246", "v_right": "#0891b2", "v_term": "#9d4edd"}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--NT", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sd = load_state(args.ckpt)
    rse = sd["rule_state_emb"].float()          # (NT+T, d)
    NT = args.NT
    rse_f = torch.fft.rfft(rse, dim=-1)         # (NT+T, 257)
    phi = torch.angle(rse_f).numpy()

    v_phi = {
        n: torch.angle(torch.fft.rfft(sd[n].float())).numpy()
        for n in ("v_left", "v_right", "v_term")
    }

    # Mean resultant length across entities per frequency (exclude k=0, Nyquist)
    unit = rse_f / rse_f.abs().clamp(min=1e-12)
    R_k = unit.mean(dim=0).abs().numpy()
    R_k_for_top = R_k.copy()
    R_k_for_top[0] = -np.inf
    R_k_for_top[-1] = -np.inf
    top2 = np.argsort(R_k_for_top)[::-1][:2]

    pairs = [
        ("low freq", 1, 2),
        ("mid freq", 128, 129),
        ("top-R", int(top2[0]), int(top2[1])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))

    for ax, (label, k1, k2) in zip(axes, pairs):
        ax.scatter(
            phi[:NT, k1], phi[:NT, k2],
            s=7, alpha=0.40, c="#1f77b4",
            label=f"NT (n={NT})", edgecolors="none",
        )
        ax.scatter(
            phi[NT:, k1], phi[NT:, k2],
            s=7, alpha=0.40, c="#f39c12",
            label=f"T (n={phi.shape[0] - NT})", edgecolors="none",
        )
        for vname, color in V_COLORS.items():
            ax.scatter(
                [v_phi[vname][k1]], [v_phi[vname][k2]],
                c=color, marker="*", s=300,
                edgecolors="black", linewidths=1.2,
                label=vname, zorder=5,
            )
        ax.set_title(
            f"{label}  (k={k1}, k={k2})\nR(k={k1})={R_k[k1]:.3f}  "
            f"R(k={k2})={R_k[k2]:.3f}",
            fontsize=10,
        )
        ax.set_xlim(-np.pi - 0.1, np.pi + 0.1)
        ax.set_ylim(-np.pi - 0.1, np.pi + 0.1)
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        ax.set_yticks([-np.pi, 0, np.pi])
        ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
        ax.set_xlabel(f"phase(k={k1})")
        ax.set_ylabel(f"phase(k={k2})")
        ax.axhline(0, color="#d0d0d0", linewidth=0.3, zorder=0)
        ax.axvline(0, color="#d0d0d0", linewidth=0.3, zorder=0)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    fig.suptitle(
        f"B1-b: Bivariate phase scatter — {args.lang}",
        fontsize=12, y=1.01,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  (top-R pair: k={top2[0]}, k={top2[1]})")


if __name__ == "__main__":
    main()
