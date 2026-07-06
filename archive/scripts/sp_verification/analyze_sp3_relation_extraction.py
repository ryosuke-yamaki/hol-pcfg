"""SP3 Verification: Relation Extraction from Learned HN-PCFG Grammar.

Verifies that the HolE algebraic identity holds in a trained HN-PCFG model:
    r_ext = e_A ⋆ e_B ≈ v_left   (for high-probability parent-child pairs)

where ⋆ is circular correlation and freq_cnorm guarantees |FFT(e)[k]| = 1.

Usage:
    python scripts/analyze_sp3_relation_extraction.py [--checkpoint PATH] [--top-k K]
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = (
    "log/hn_pcfg_allproj_cnorm_tau/HNPCFG2026-03-31-12_43_40/best.pt"
)
NT = 4096
T = 8192
S_DIM = 512


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--top-k", type=int, default=10,
                   help="Number of top children per parent to analyze")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="results/sp3")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────────────────────────────
def circular_correlation(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    """Compute circular correlation a ⋆ b = IFFT(conj(FFT(a)) * FFT(b)).

    Args:
        a: (..., s_dim)
        b: (..., s_dim)
        n: signal length for irfft

    Returns:
        (..., s_dim) circular correlation
    """
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
    """Compute P(child | parent, relation) via HolE scoring + softmax.

    Args:
        v: (s_dim,) relation vector
        nt_emb: (NT, s_dim) nonterminal embeddings
        all_emb: (NT+T, s_dim) all entity embeddings
        tau: temperature scalar
        s_dim: embedding dimension

    Returns:
        (NT+T, NT) probability matrix, softmax over children (dim=0)
    """
    v_f = torch.fft.rfft(v.unsqueeze(0), dim=-1)          # (1, F)
    parent_f = torch.fft.rfft(nt_emb, dim=-1)             # (NT, F)
    template = torch.fft.irfft(
        v_f.unsqueeze(1) * parent_f.unsqueeze(0),
        n=s_dim, dim=-1
    )                                                       # (1, NT, s_dim)
    scores = torch.einsum("cs, rps -> rcp", all_emb, template)  # (1, C, NT)
    scores = scores.squeeze(0) * tau                        # (C, NT)
    return scores.softmax(dim=0)


def extract_relations_topk(
    nt_emb: torch.Tensor,
    all_emb: torch.Tensor,
    probs: torch.Tensor,
    k: int,
    s_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract relation vectors for top-k children of each parent.

    Returns:
        r_ext: (NT, k, s_dim) extracted relation vectors
        cos_with_parent: (NT, k) cosine similarity with parent's own embedding
        topk_probs: (NT, k) probabilities of top-k children
        topk_indices: (NT, k) indices of top-k children
    """
    # Top-k children per parent
    topk_probs, topk_indices = probs.topk(k, dim=0)  # (k, NT)
    topk_probs = topk_probs.t()    # (NT, k)
    topk_indices = topk_indices.t()  # (NT, k)

    # Batch extract: gather child embeddings
    child_emb = all_emb[topk_indices.reshape(-1)].reshape(NT, k, s_dim)  # (NT, k, s_dim)
    parent_emb = nt_emb.unsqueeze(1).expand(-1, k, -1)  # (NT, k, s_dim)

    # Circular correlation: r_ext = e_A ⋆ e_B
    r_ext = circular_correlation(parent_emb, child_emb, s_dim)  # (NT, k, s_dim)

    return r_ext, topk_probs, topk_indices


def cosine_sim_with_relation(
    r_ext: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Compute cosine similarity between extracted relations and a relation vector.

    Args:
        r_ext: (NT, k, s_dim)
        v: (s_dim,)

    Returns:
        (NT, k) cosine similarities
    """
    return F.cosine_similarity(r_ext, v.unsqueeze(0).unsqueeze(0), dim=-1)


# ──────────────────────────────────────────────────────────────────────
# Analysis routines
# ──────────────────────────────────────────────────────────────────────
def analyze_relation(
    label: str,
    v: torch.Tensor,
    v_other: torch.Tensor,
    nt_emb: torch.Tensor,
    all_emb: torch.Tensor,
    tau: float,
    k: int,
    s_dim: int,
) -> dict:
    """Run full SP3 analysis for one relation (left or right)."""
    print(f"\n{'='*60}")
    print(f"  Analyzing: {label}")
    print(f"{'='*60}")

    # Step 1: Compute rule probabilities
    probs = compute_rule_probs(v, nt_emb, all_emb, tau, s_dim)
    print(f"Rule probs shape: {probs.shape}  (children x parents)")

    # Step 2: Extract relations for top-k children
    r_ext, topk_probs, topk_indices = extract_relations_topk(
        nt_emb, all_emb, probs, k, s_dim
    )

    # Step 3: Cosine similarity with the correct relation
    cos_correct = cosine_sim_with_relation(r_ext, v)      # (NT, k)
    # Cosine similarity with the OTHER relation (discrimination test)
    cos_other = cosine_sim_with_relation(r_ext, v_other)  # (NT, k)

    # Step 4: Random baseline — shuffle parent-child pairing
    rand_perm = torch.randperm(NT)
    r_ext_random = circular_correlation(
        nt_emb,
        all_emb[topk_indices[:, 0][rand_perm]],
        s_dim,
    )
    cos_random = F.cosine_similarity(
        r_ext_random, v.unsqueeze(0), dim=-1
    )

    # ── Collect statistics ──
    # Top-1 analysis
    cos_top1 = cos_correct[:, 0]
    cos_other_top1 = cos_other[:, 0]

    # Probability-weighted analysis (all top-k)
    weights = topk_probs / topk_probs.sum(dim=1, keepdim=True)  # normalize
    cos_weighted = (cos_correct * weights).sum(dim=1)  # (NT,)

    results = {
        "label": label,
        # Top-1
        "top1_cos_mean": cos_top1.mean().item(),
        "top1_cos_std": cos_top1.std().item(),
        "top1_cos_median": cos_top1.median().item(),
        "top1_cos_min": cos_top1.min().item(),
        "top1_cos_max": cos_top1.max().item(),
        "top1_cos_q25": cos_top1.quantile(0.25).item(),
        "top1_cos_q75": cos_top1.quantile(0.75).item(),
        # Top-1 vs other relation (discrimination)
        "top1_cos_other_mean": cos_other_top1.mean().item(),
        "top1_cos_other_std": cos_other_top1.std().item(),
        # Random baseline
        "random_cos_mean": cos_random.mean().item(),
        "random_cos_std": cos_random.std().item(),
        # Weighted top-k
        "weighted_cos_mean": cos_weighted.mean().item(),
        "weighted_cos_std": cos_weighted.std().item(),
        # Top-k breakdown
        "per_rank_cos_mean": [cos_correct[:, i].mean().item() for i in range(k)],
        # Probability stats
        "top1_prob_mean": topk_probs[:, 0].mean().item(),
        "top1_prob_std": topk_probs[:, 0].std().item(),
        "topk_cumprob_mean": topk_probs.sum(dim=1).mean().item(),
    }

    # ── Print summary ──
    print(f"\n  Top-1 cosine(r_ext, {label}):")
    print(f"    mean={results['top1_cos_mean']:.4f}  std={results['top1_cos_std']:.4f}")
    print(f"    median={results['top1_cos_median']:.4f}  "
          f"[Q25={results['top1_cos_q25']:.4f}, Q75={results['top1_cos_q75']:.4f}]")
    print(f"    min={results['top1_cos_min']:.4f}  max={results['top1_cos_max']:.4f}")

    print(f"\n  Discrimination (top-1 vs other relation):")
    print(f"    cos({label}): {results['top1_cos_mean']:.4f}")
    other_label = "v_right" if "left" in label else "v_left"
    print(f"    cos({other_label}): {results['top1_cos_other_mean']:.4f}")
    gap = results["top1_cos_mean"] - results["top1_cos_other_mean"]
    print(f"    gap: {gap:+.4f}")

    print(f"\n  Random baseline:")
    print(f"    cos(random): {results['random_cos_mean']:.4f} ± {results['random_cos_std']:.4f}")

    print(f"\n  Per-rank cosine(r_ext, {label}) mean:")
    for i, c in enumerate(results["per_rank_cos_mean"]):
        print(f"    rank {i+1}: {c:.4f}")

    print(f"\n  Top-1 prob: {results['top1_prob_mean']:.4f} ± {results['top1_prob_std']:.4f}")
    print(f"  Top-{k} cumulative prob: {results['topk_cumprob_mean']:.4f}")

    # Return raw tensors for visualization
    results["_cos_top1"] = cos_top1.cpu()
    results["_cos_other_top1"] = cos_other_top1.cpu()
    results["_cos_random"] = cos_random.cpu()
    results["_cos_correct"] = cos_correct.cpu()
    results["_topk_probs"] = topk_probs.cpu()
    results["_topk_indices"] = topk_indices.cpu()
    results["_r_ext_top1"] = r_ext[:, 0, :].cpu()

    return results


# ──────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────
def save_visualizations(
    left_results: dict,
    right_results: dict,
    v_left: torch.Tensor,
    v_right: torch.Tensor,
    output_dir: Path,
) -> None:
    """Generate and save analysis plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\nmatplotlib not available, skipping visualizations.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Histogram: cosine similarity distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, res in zip(axes, [left_results, right_results]):
        cos_top1 = res["_cos_top1"].numpy()
        cos_other = res["_cos_other_top1"].numpy()
        cos_rand = res["_cos_random"].numpy()

        ax.hist(cos_top1, bins=80, alpha=0.7, label=f"cos(r_ext, {res['label']})", color="C0")
        ax.hist(cos_other, bins=80, alpha=0.5, label="cos(r_ext, other)", color="C1")
        ax.hist(cos_rand, bins=80, alpha=0.4, label="random baseline", color="gray")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title(f"{res['label']} — Top-1 Extracted Relation")
        ax.legend()
        ax.axvline(x=res["top1_cos_mean"], color="C0", linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "sp3_cosine_histogram.svg", format="svg")
    plt.close(fig)
    print(f"\n  Saved: {output_dir / 'sp3_cosine_histogram.svg'}")

    # ── 2. Per-rank cosine decay ──
    fig, ax = plt.subplots(figsize=(8, 5))
    k = len(left_results["per_rank_cos_mean"])
    ranks = list(range(1, k + 1))
    ax.plot(ranks, left_results["per_rank_cos_mean"], "o-", label="v_left")
    ax.plot(ranks, right_results["per_rank_cos_mean"], "s-", label="v_right")
    ax.set_xlabel("Child Rank (by probability)")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Cosine(r_ext, v) by Child Rank")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "sp3_per_rank_cosine.svg", format="svg")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'sp3_per_rank_cosine.svg'}")

    # ── 3. Scatter: cosine similarity vs probability ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, res in zip(axes, [left_results, right_results]):
        cos_all = res["_cos_correct"].numpy()  # (NT, k)
        prob_all = res["_topk_probs"].numpy()  # (NT, k)
        ax.scatter(prob_all.ravel(), cos_all.ravel(), s=1, alpha=0.1, c="C0")
        ax.set_xlabel("P(child | parent)")
        ax.set_ylabel(f"cos(r_ext, {res['label']})")
        ax.set_title(f"{res['label']} — Cosine vs Probability")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "sp3_cosine_vs_prob.svg", format="svg")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'sp3_cosine_vs_prob.svg'}")

    # ── 4. Relation space structure (PCA of extracted relations) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, res, v_ref in zip(axes,
                               [left_results, right_results],
                               [v_left, v_right]):
        r_ext_top1 = res["_r_ext_top1"].numpy()  # (NT, s_dim)
        v_np = v_ref.cpu().numpy()

        # PCA via SVD
        centered = r_ext_top1 - r_ext_top1.mean(axis=0)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pc = U[:, :2] * S[:2]  # first 2 PCs

        # Project v onto same space
        v_centered = v_np - r_ext_top1.mean(axis=0)
        v_pc = (Vt[:2] @ v_centered)

        cos_vals = res["_cos_top1"].numpy()
        scatter = ax.scatter(pc[:, 0], pc[:, 1], c=cos_vals, s=2, alpha=0.4,
                             cmap="RdYlGn", vmin=-0.2, vmax=1.0)
        ax.scatter([v_pc[0]], [v_pc[1]], c="red", marker="*", s=200,
                   edgecolors="black", zorder=10, label=res["label"])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA of Extracted Relations ({res['label']})")
        ax.legend()
        plt.colorbar(scatter, ax=ax, label="cos(r_ext, v)")
    fig.tight_layout()
    fig.savefig(output_dir / "sp3_relation_pca.svg", format="svg")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'sp3_relation_pca.svg'}")

    # ── 5. NT vs T child breakdown ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, res in zip(axes, [left_results, right_results]):
        topk_idx = res["_topk_indices"][:, 0].numpy()  # (NT,) top-1 child indices
        is_nt = topk_idx < NT
        cos_top1 = res["_cos_top1"].numpy()

        cos_nt = cos_top1[is_nt]
        cos_t = cos_top1[~is_nt]

        ax.hist(cos_nt, bins=60, alpha=0.7, label=f"NT child (n={len(cos_nt)})", color="C0")
        ax.hist(cos_t, bins=60, alpha=0.7, label=f"T child (n={len(cos_t)})", color="C2")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title(f"{res['label']} — NT vs T children")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sp3_nt_vs_t_children.svg", format="svg")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'sp3_nt_vs_t_children.svg'}")


# ──────────────────────────────────────────────────────────────────────
# Normalization verification
# ──────────────────────────────────────────────────────────────────────
def verify_normalization(ckpt: dict) -> None:
    """Verify freq_cnorm and relation projection cnorm conditions."""
    print("\n" + "=" * 60)
    print("  Normalization Verification")
    print("=" * 60)

    emb = ckpt["rule_state_emb"]
    v_left = ckpt["v_left"]
    v_right = ckpt["v_right"]

    # Entity: |FFT(e)[k]| = 1
    emb_f = torch.fft.rfft(emb, dim=-1)
    emb_mag = emb_f.abs()
    print(f"\n  Entity embeddings (freq_cnorm):")
    print(f"    |FFT(e)[k]| — mean: {emb_mag.mean():.8f}  std: {emb_mag.std():.8f}")
    print(f"    max deviation from 1: {(emb_mag - 1.0).abs().max():.2e}")

    # Relation: |FFT(v)[k]| = 1
    for name, v in [("v_left", v_left), ("v_right", v_right)]:
        v_f = torch.fft.rfft(v.unsqueeze(0) if v.dim() == 1 else v, dim=-1)
        v_mag = v_f.abs()
        print(f"\n  {name} (projection cnorm):")
        print(f"    |FFT(v)[k]| — mean: {v_mag.mean():.8f}  std: {v_mag.std():.8f}")
        print(f"    max deviation from 1: {(v_mag - 1.0).abs().max():.2e}")

    # Auto-correlation check: e ⋆ e ≈ δ
    sample_idx = torch.randint(0, emb.shape[0], (100,))
    sample = emb[sample_idx]
    auto_corr = circular_correlation(sample, sample, S_DIM)  # (100, s_dim)
    # Delta should have peak at index 0, ~0 elsewhere
    peak = auto_corr[:, 0].mean().item()
    off_peak = auto_corr[:, 1:].abs().mean().item()
    print(f"\n  Auto-correlation e ⋆ e (100 random entities):")
    print(f"    peak (index 0): {peak:.6f}")
    print(f"    off-peak mean |val|: {off_peak:.6f}")
    print(f"    ratio (peak / off-peak): {peak / max(off_peak, 1e-12):.1f}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Top-k: {args.top_k}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    emb = ckpt["rule_state_emb"]      # (NT+T, s_dim)
    v_left = ckpt["v_left"]            # (s_dim,)
    v_right = ckpt["v_right"]          # (s_dim,)
    log_tau = ckpt["log_tau"]          # scalar
    tau = log_tau.exp().item()

    nt_emb = emb[:NT]                  # (NT, s_dim)
    all_emb = emb                      # (NT+T, s_dim)

    print(f"\nModel parameters:")
    print(f"  rule_state_emb: {emb.shape}")
    print(f"  v_left: {v_left.shape}  v_right: {v_right.shape}")
    print(f"  tau: {tau:.4f}")

    # Normalization verification
    verify_normalization(ckpt)

    # Left relation analysis
    left_results = analyze_relation(
        label="v_left", v=v_left, v_other=v_right,
        nt_emb=nt_emb, all_emb=all_emb, tau=tau,
        k=args.top_k, s_dim=S_DIM,
    )

    # Right relation analysis
    right_results = analyze_relation(
        label="v_right", v=v_right, v_other=v_left,
        nt_emb=nt_emb, all_emb=all_emb, tau=tau,
        k=args.top_k, s_dim=S_DIM,
    )

    # Save visualizations
    save_visualizations(left_results, right_results, v_left, v_right, output_dir)

    # Save numerical results (exclude tensor fields)
    json_results = {}
    for key, res in [("left", left_results), ("right", right_results)]:
        json_results[key] = {
            k: v for k, v in res.items() if not k.startswith("_")
        }
    json_results["checkpoint"] = args.checkpoint
    json_results["tau"] = tau
    json_results["top_k"] = args.top_k

    json_path = output_dir / "sp3_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("  SP3 Verification Summary")
    print("=" * 60)
    print(f"\n  {'Metric':<40} {'v_left':>8} {'v_right':>8}")
    print(f"  {'-'*56}")
    print(f"  {'Top-1 cos(r_ext, v) mean':<40} "
          f"{left_results['top1_cos_mean']:>8.4f} {right_results['top1_cos_mean']:>8.4f}")
    print(f"  {'Top-1 cos(r_ext, v_other) mean':<40} "
          f"{left_results['top1_cos_other_mean']:>8.4f} {right_results['top1_cos_other_mean']:>8.4f}")
    print(f"  {'Random baseline mean':<40} "
          f"{left_results['random_cos_mean']:>8.4f} {right_results['random_cos_mean']:>8.4f}")
    gap_l = left_results["top1_cos_mean"] - left_results["top1_cos_other_mean"]
    gap_r = right_results["top1_cos_mean"] - right_results["top1_cos_other_mean"]
    print(f"  {'Discrimination gap':<40} {gap_l:>+8.4f} {gap_r:>+8.4f}")


if __name__ == "__main__":
    main()
