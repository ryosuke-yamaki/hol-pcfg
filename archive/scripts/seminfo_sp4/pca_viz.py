"""Cosine-PCA 2D visualization of HN-PCFG embedding spaces.

Unit-normalizes every vector (rule_state_emb NT/T split, vocab_emb,
root_emb, v_left/v_right/v_term) then fits a standard PCA on the
stacked set. On the unit sphere, Euclidean geometry matches cosine
geometry, so this is the standard "cosine PCA".

Output: one .svg per checkpoint under results/sp4/pca_<lang>.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def load_embeddings(ckpt_path: str, NT: int) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    rule = sd["rule_state_emb"].float().numpy()
    return {
        "nt_emb": rule[:NT],
        "t_emb": rule[NT:],
        "vocab_emb": sd["vocab_emb"].float().numpy().T,
        "v_left": sd["v_left"].float().numpy()[None, :],
        "v_right": sd["v_right"].float().numpy()[None, :],
        "v_term": sd["v_term"].float().numpy()[None, :],
    }


def _unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--NT", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab_sample", type=int, default=-1,
                    help="Random subsample of vocab points (-1 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    emb = load_embeddings(args.ckpt, args.NT)
    rng = np.random.default_rng(args.seed)

    vocab = emb["vocab_emb"]
    if 0 < args.vocab_sample < vocab.shape[0]:
        idx = rng.choice(vocab.shape[0], size=args.vocab_sample, replace=False)
        vocab = vocab[idx]

    groups = [
        ("vocab",   _unit(vocab),          "#bdbdbd", 12,  0.25, "o"),
        ("NT",      _unit(emb["nt_emb"]),  "#1f77b4", 12,  0.65, "o"),
        ("T",       _unit(emb["t_emb"]),   "#f39c12", 12,  0.65, "o"),
        ("v_left",  _unit(emb["v_left"]),  "#d62246", 280, 1.0,  "*"),
        ("v_right", _unit(emb["v_right"]), "#0891b2", 280, 1.0,  "*"),
        ("v_term",  _unit(emb["v_term"]),  "#9d4edd", 280, 1.0,  "*"),
    ]

    stacked = np.vstack([g[1] for g in groups])
    pca = PCA(n_components=2, random_state=args.seed)
    xy = pca.fit_transform(stacked)

    fig, ax = plt.subplots(figsize=(8, 8))
    start = 0
    for name, arr, color, size, alpha, marker in groups:
        end = start + arr.shape[0]
        pts = xy[start:end]
        zorder = 1 if name in ("vocab", "NT", "T") else 3
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=size, c=color, alpha=alpha, marker=marker,
            edgecolors="black" if marker in ("*", "X") else "none",
            linewidths=0.5 if marker in ("*", "X") else 0.0,
            label=f"{name} (n={arr.shape[0]})",
            zorder=zorder,
        )
        start = end

    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title(f"Cosine-PCA of HN-PCFG embeddings — {args.lang}")
    ax.axhline(0, color="#808080", linewidth=0.3, zorder=0)
    ax.axvline(0, color="#808080", linewidth=0.3, zorder=0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"Wrote {out_path}  (EVR: PC1={evr[0]:.3f}, PC2={evr[1]:.3f})")


if __name__ == "__main__":
    main()
