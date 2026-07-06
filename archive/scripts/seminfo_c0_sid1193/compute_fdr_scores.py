"""Original circular-FDR torus-axis selector reproducing PR #25's visualisation.

Selects two interior FFT phase bins (k*, l*) by maximizing the joint
Fisher-discriminant-ratio of the NT phases in their (cos phi, sin phi)
2D representation:

    z_i(k)     = [cos phi_i^k, sin phi_i^k]              # per (NT i, axis k)
    m_global   = mean_i z_i                              # (axes, 2)
    m_c        = mean_{i in c} z_i                       # (axes, 2)
    between(k) = sum_c n_c * ||m_c(k) - m_global(k)||^2
    within(k)  = sum_c n_c * (1 - ||m_c(k)||^2)
    score(k,l) = (between(k) + between(l)) / (within(k) + within(l) + eps)

The pair (k*, l*) is the argmax over k != l.

Classes:
- The argmax NT label per index from build_symbol_labels.py with
  support >= --min_support is kept.
- The top --top_labels labels by total support form the class set; every
  other labeled NT is mapped to the synthetic "other" class
  (--other_policy labeled). Unlabeled NTs are excluded entirely.

This formulation reproduces the (k*, l*) computed in PR #25's
fdr_scores_gold JSON exactly bit-for-bit on the same checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (str(REPO), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fdr_torus_viz import compute_phases  # noqa: E402


def load_nt_emb(ckpt_path: str) -> tuple[torch.Tensor, int]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    NT = ck["hyper_parameters"]["model_params"]["NT"]
    sd = {
        k[len("model."):]: v
        for k, v in ck["state_dict"].items()
        if k.startswith("model.")
    }
    return sd["rule_state_emb"].float()[:NT], NT


def effective_labels(
    labels: list[str | None], support: list[int], min_support: int,
) -> list[str | None]:
    return [
        lab if (lab is not None and sup >= min_support) else None
        for lab, sup in zip(labels, support)
    ]


def choose_target_labels(
    labels: list[str | None], support: list[int], top_labels: int,
) -> list[str]:
    totals: dict[str, int] = {}
    for lab, sup in zip(labels, support):
        if lab is None or sup <= 0:
            continue
        totals[lab] = totals.get(lab, 0) + int(sup)
    return [
        lab for lab, _ in
        sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:top_labels]
    ]


def build_fdr_classes(
    labels: list[str | None], target_labels: list[str], other_policy: str,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    class_names = list(target_labels) + ["other"]
    class_of = {lab: i for i, lab in enumerate(target_labels)}
    y = np.full(len(labels), -1, dtype=np.int64)
    for i, lab in enumerate(labels):
        if lab in class_of:
            y[i] = class_of[lab]
        elif other_policy == "all":
            y[i] = len(target_labels)
        elif other_policy == "labeled" and lab is not None:
            y[i] = len(target_labels)
    counts = {name: int((y == ci).sum()) for ci, name in enumerate(class_names)}
    return y, class_names, counts


def circular_fdr_components(
    phi: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis between / within using the (cos, sin) 2D vector representation."""
    valid = y >= 0
    phi = phi[valid].astype(np.float64, copy=False)
    y = y[valid]
    z = np.stack([np.cos(phi), np.sin(phi)], axis=-1)        # (N, K, 2)
    global_mean = z.mean(axis=0)                              # (K, 2)
    n_axes = z.shape[1]
    between = np.zeros(n_axes, dtype=np.float64)
    within = np.zeros(n_axes, dtype=np.float64)
    for ci in sorted({int(v) for v in y.tolist()}):
        mask = y == ci
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        m_c = z[mask].mean(axis=0)                            # (K, 2)
        diff = m_c - global_mean
        between += n_c * np.einsum("kd,kd->k", diff, diff)
        within += n_c * (1.0 - np.einsum("kd,kd->k", m_c, m_c))
    return np.maximum(between, 0.0), np.maximum(within, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--labels", required=True,
        help="symbol_labels JSON from build_symbol_labels.py",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_support", type=int, default=5)
    ap.add_argument(
        "--top_labels", type=int, default=2,
        help="number of top-support NT labels to keep as named classes; "
             "every other labeled NT is mapped to 'other'.",
    )
    ap.add_argument(
        "--other_policy", choices=("labeled", "all"), default="labeled",
        help="'labeled' = unlabeled NTs are excluded; 'all' = they go to 'other'.",
    )
    args = ap.parse_args()

    nt_emb, NT = load_nt_emb(args.ckpt)
    phi_nt = compute_phases(nt_emb)                          # (NT, K)
    K = phi_nt.shape[1]

    lm = json.loads(Path(args.labels).read_text())
    assert lm["NT"] == NT, f"label/NT mismatch: {lm['NT']} vs {NT}"
    support = [int(x) for x in lm["nt_support"]]
    labels_eff = effective_labels(list(lm["nt_label"]), support, args.min_support)

    target_labels = choose_target_labels(labels_eff, support, args.top_labels)
    y, class_names, counts = build_fdr_classes(
        labels_eff, target_labels, args.other_policy,
    )
    between, within = circular_fdr_components(phi_nt, y)
    eps = 1e-12
    score = (between[:, None] + between[None, :]) / (within[:, None] + within[None, :] + eps)
    np.fill_diagonal(score, -np.inf)
    k_idx, l_idx = np.unravel_index(np.nanargmax(score), score.shape)
    if l_idx < k_idx:
        k_idx, l_idx = l_idx, k_idx
    k_idx, l_idx = int(k_idx), int(l_idx)

    out = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "labels_json": str(Path(args.labels).resolve()),
        "selection_mode": "circular_fdr",
        "k_star_bin": k_idx + 1,
        "l_star_bin": l_idx + 1,
        "k_star_idx": k_idx,
        "l_star_idx": l_idx,
        "score": float(score[k_idx, l_idx]),
        "selected_labels": target_labels,
        "class_counts": {name: counts.get(name, 0) for name in class_names},
        "min_support": args.min_support,
        "top_labels": args.top_labels,
        "other_policy": args.other_policy,
        "phase_axes": int(K),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    order = np.argsort(-(between / np.maximum(within, eps)))
    print(f"[fdr] NT={NT}  classes={class_names}  counts={counts}")
    print(f"[fdr] k*={k_idx + 1}  l*={l_idx + 1}  score={out['score']:.4f}")
    print(f"[fdr] top10 axes by (between/within): "
          f"{[(int(o + 1), round(float(between[o] / max(within[o], eps)), 3)) for o in order[:10]]}")
    print(f"[fdr] wrote {out_path}")


if __name__ == "__main__":
    main()
