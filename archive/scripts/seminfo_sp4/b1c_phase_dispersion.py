"""B1-c: Per-frequency phase concentration (mean resultant length).

For each frequency bin k, compute R_k = | mean_i exp(i * phi_i[k]) |
across NT / T / vocab entity groups. R_k near 1 means all entities
share the same phase at k (strong "class-shared" signal). R_k near 0
means phases are uniform/random.

Also draws dashed reference lines at the uniform-phase expectation
E[R_k] ≈ sqrt(pi / (4 N)) per group for comparison.
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


def R_per_freq(group: torch.Tensor) -> np.ndarray:
    """Mean resultant length R_k across rows of `group`, per frequency bin."""
    gf = torch.fft.rfft(group, dim=-1)
    unit = gf / gf.abs().clamp(min=1e-12)
    return unit.mean(dim=0).abs().numpy()


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
    vocab = sd["vocab_emb"].float().t()        # (V, d)
    NT = args.NT
    groups = [
        ("NT",    rse[:NT], "#1f77b4"),
        ("T",     rse[NT:], "#f39c12"),
        ("vocab", vocab,    "#666666"),
    ]
    Rs = {name: R_per_freq(t) for name, t, _ in groups}

    n_freq = next(iter(Rs.values())).shape[0]
    k_axis = np.arange(n_freq)

    fig, ax = plt.subplots(figsize=(14, 5.4))
    for name, tensor, color in groups:
        N = tensor.shape[0]
        ax.plot(k_axis, Rs[name], color=color, linewidth=1.0,
                label=f"{name} (n={N})")
        ref = np.sqrt(np.pi / (4 * N))
        ax.axhline(ref, color=color, linestyle=":", linewidth=0.7, alpha=0.6)

    ax.set_xlabel("Frequency bin k")
    ax.set_ylabel(r"$R_k$ (mean resultant length)")
    ax.set_title(
        f"B1-c: Per-frequency phase concentration — {args.lang}  "
        f"(dotted lines: uniform-phase expectation $\\sqrt{{\\pi/(4N)}}$)",
        fontsize=11,
    )
    ax.set_xlim(-2, n_freq + 1)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  (max R_k across groups: "
          f"{max(r.max() for r in Rs.values()):.3f})")


if __name__ == "__main__":
    main()
