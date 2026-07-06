"""B1-a: FFT-phase heatmap of the relation vectors.

Under Phase-Only Manifold every FFT coefficient has unit magnitude, so
the only learned degree of freedom per frequency bin is the angle
phi_k in [-pi, pi]. This figure shows the 257 phases of v_left,
v_right, and v_term side-by-side via a cyclic colormap.
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


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sd = load_state(args.ckpt)
    names = ["v_left", "v_right", "v_term"]
    phases = []
    for n in names:
        vec = sd[n].float()
        phi = torch.angle(torch.fft.rfft(vec)).numpy()
        phases.append(phi)
    phases = np.stack(phases, axis=0)           # (3, 257)
    n_freq = phases.shape[1]

    fig, ax = plt.subplots(figsize=(16, 2.8))
    im = ax.imshow(
        phases, aspect="auto", cmap="twilight_shifted",
        vmin=-np.pi, vmax=np.pi, interpolation="nearest",
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Frequency bin k")
    ax.set_xticks([0, 64, 128, 192, 256])
    ax.set_title(f"B1-a: FFT phase of relation vectors — {args.lang}")
    cbar = plt.colorbar(im, ax=ax, shrink=0.9, label="phase [rad]")
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  ({n_freq} frequency bins)")


if __name__ == "__main__":
    main()
