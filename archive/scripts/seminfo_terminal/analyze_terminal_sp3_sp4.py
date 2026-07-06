#!/usr/bin/env python3
"""Terminal SP3/SP4 Analysis for MLP-free HN-PCFG (holeterm model).

Analyzes whether the HolE algebraic structure holds for the terminal
(preterminal -> word) production in the MLP-free model:
    P(w|A) = softmax(tau_term * vocab_emb^T @ circonv(v_term, e_A))

SP3: Relation extraction from learned terminal productions
    r_term(A, w) = circular_correlation(e_A, vocab_emb_w)
    Measures cos(r_term, v_term) for top-ranked words.

SP4: Systematicity — whether r_term ≈ v_term holds broadly,
    not just for top-1 words.

Checkpoint: Lightning format, keys prefixed with "model."
    - NT=1024, T=2048, s_dim=512, V=10020
    - term_emb = rule_state_emb[NT:]  (T=2048 preterminals)
    - All embeddings are freq_cnorm'd (|FFT[k]|=1)

Usage:
    python scripts/analyze_terminal_sp3_sp4.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
CKPT_PATH = (
    "/workspace/hol-pcfg-seminfo/ckpt/holeterm-nt1024/"
    "ckpt-sf1_val/sentence_f1=0.67-v2.ckpt"
)
OUTPUT_DIR = Path("/workspace/hol-pcfg/results/terminal_sp3")

NT = 1024
T = 2048
S_DIM = 512
V = 10020


# ──────────────────────────────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────────────────────────────
def circular_correlation(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    """a ⋆ b = IFFT(conj(FFT(a)) * FFT(b))."""
    a_f = torch.fft.rfft(a, dim=-1)
    b_f = torch.fft.rfft(b, dim=-1)
    return torch.fft.irfft(a_f.conj() * b_f, n=n, dim=-1)


def circular_convolution(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    """a * b = IFFT(FFT(a) * FFT(b))."""
    a_f = torch.fft.rfft(a, dim=-1)
    b_f = torch.fft.rfft(b, dim=-1)
    return torch.fft.irfft(a_f * b_f, n=n, dim=-1)


def hol_cosine(r: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between vectors under freq_cnorm.

    Under freq_cnorm with torch convention: ||x|| = 1 (not sqrt(d)),
    since |FFT[k]|=1 and Parseval gives sum|x|^2 = (1/n)*sum|FFT|^2 = 1.
    Therefore cos = <r, v> / (||r|| * ||v||) = <r, v>.
    """
    return (r * v).sum(dim=-1)


# ──────────────────────────────────────────────────────────────────────
# Matplotlib setup
# ──────────────────────────────────────────────────────────────────────
def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.format": "svg",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


# ──────────────────────────────────────────────────────────────────────
# Load checkpoint
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def load_model(device: str = "cuda") -> dict:
    """Load holeterm checkpoint and extract key tensors."""
    dev = torch.device(device)
    raw = torch.load(CKPT_PATH, map_location=dev, weights_only=False)
    sd = raw["state_dict"]

    # Strip "model." prefix
    def get(key: str) -> torch.Tensor:
        return sd[f"model.{key}"]

    rule_state_emb = get("rule_state_emb")           # (3072, 512)
    term_emb = rule_state_emb[NT:]                    # (2048, 512)
    vocab_emb = get("vocab_emb")                      # (512, 10020)
    v_term = get("v_term")                            # (512,)
    v_left = get("v_left").squeeze(0)                 # (512,)
    v_right = get("v_right").squeeze(0)               # (512,)
    tau_term = get("log_tau_term").exp().item()        # scalar
    tau = get("log_tau").exp().item()                  # scalar

    return {
        "term_emb": term_emb,
        "vocab_emb": vocab_emb,
        "v_term": v_term,
        "v_left": v_left,
        "v_right": v_right,
        "tau_term": tau_term,
        "tau": tau,
    }


# ──────────────────────────────────────────────────────────────────────
# Verify freq_cnorm
# ──────────────────────────────────────────────────────────────────────
def verify_cnorm(data: dict) -> None:
    """Verify that embeddings satisfy freq_cnorm."""
    print("=" * 60)
    print("  Normalization Verification (freq_cnorm)")
    print("=" * 60)

    for name, t in [
        ("term_emb", data["term_emb"]),
        ("vocab_emb", data["vocab_emb"].t()),
        ("v_term", data["v_term"].unsqueeze(0)),
        ("v_left", data["v_left"].unsqueeze(0)),
        ("v_right", data["v_right"].unsqueeze(0)),
    ]:
        tf = torch.fft.rfft(t, dim=-1)
        mag = tf.abs()
        dev = (mag - 1.0).abs()
        print(f"  {name:12s}: |FFT| mean={mag.mean():.6f}  "
              f"max_dev={dev.max():.2e}  ||x||={t.norm(dim=-1).mean():.6f}")


# ──────────────────────────────────────────────────────────────────────
# SP3: Terminal Relation Extraction
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_sp3(data: dict) -> dict:
    """SP3 analysis for terminal productions."""
    term_emb = data["term_emb"]      # (T, d)
    vocab_emb = data["vocab_emb"]    # (d, V)
    v_term = data["v_term"]          # (d,)
    tau_term = data["tau_term"]

    print("\n" + "=" * 60)
    print("  SP3: Terminal Relation Extraction")
    print("=" * 60)
    print(f"  tau_term = {tau_term:.4f}")
    print(f"  ||v_term|| = {v_term.norm().item():.6f}")

    # Step 1: P(w|A) for each preterminal A
    # template = circonv(v_term, e_A)  →  (T, d)
    template = circular_convolution(
        v_term.unsqueeze(0).expand(T, -1), term_emb, S_DIM
    )  # (T, d)
    logits = template @ vocab_emb  # (T, V)
    logits = logits * tau_term
    probs = logits.softmax(dim=-1)  # (T, V)

    # Top-K words per preterminal
    K = 10
    topk_probs, topk_indices = probs.topk(K, dim=-1)  # (T, K)

    # Step 2: r_term(A, w) = circ_corr(e_A, vocab_emb_w) for top-K words
    # vocab_emb is (d, V), transpose to (V, d) for indexing
    vocab_emb_t = vocab_emb.t()  # (V, d)

    cos_by_rank = []  # list of (T,) tensors for each rank
    for k in range(K):
        w_indices = topk_indices[:, k]  # (T,)
        w_emb = vocab_emb_t[w_indices]  # (T, d)
        r_term = circular_correlation(term_emb, w_emb, S_DIM)  # (T, d)
        cos_k = hol_cosine(r_term, v_term.unsqueeze(0))  # (T,)
        cos_by_rank.append(cos_k)

    cos_by_rank = torch.stack(cos_by_rank, dim=1)  # (T, K)
    cos_top1 = cos_by_rank[:, 0]  # (T,)

    # Step 3: Random baseline — random (A, w) pairs
    rand_A = torch.randint(0, T, (T,), device=term_emb.device)
    rand_w = torch.randint(0, V, (T,), device=term_emb.device)
    r_rand = circular_correlation(term_emb[rand_A], vocab_emb_t[rand_w], S_DIM)
    cos_random = hol_cosine(r_rand, v_term.unsqueeze(0))

    # Step 4: "Other" baseline — for each A, pick a non-top-1 word from top-10
    other_indices = topk_indices[:, 1]  # rank-2 word
    r_other = circular_correlation(
        term_emb, vocab_emb_t[other_indices], S_DIM
    )
    cos_other = hol_cosine(r_other, v_term.unsqueeze(0))

    # ── Statistics ──
    top1_p = topk_probs[:, 0]  # (T,)
    log_top1_p = top1_p.log()

    stats = {
        "top1_cos_mean": cos_top1.mean().item(),
        "top1_cos_std": cos_top1.std().item(),
        "top1_cos_median": cos_top1.median().item(),
        "top1_cos_q25": cos_top1.quantile(0.25).item(),
        "top1_cos_q75": cos_top1.quantile(0.75).item(),
        "top1_cos_min": cos_top1.min().item(),
        "top1_cos_max": cos_top1.max().item(),
        "other_cos_mean": cos_other.mean().item(),
        "other_cos_std": cos_other.std().item(),
        "random_cos_mean": cos_random.mean().item(),
        "random_cos_std": cos_random.std().item(),
        "top1_prob_mean": top1_p.mean().item(),
        "top1_prob_std": top1_p.std().item(),
        "per_rank_cos_mean": [cos_by_rank[:, i].mean().item() for i in range(K)],
        "per_rank_cos_std": [cos_by_rank[:, i].std().item() for i in range(K)],
    }

    # Correlation between P(top-1) and cos
    p_np = log_top1_p.cpu().numpy()
    c_np = cos_top1.cpu().numpy()
    from scipy.stats import pearsonr, spearmanr
    pearson_r, pearson_p = pearsonr(p_np, c_np)
    spearman_r, spearman_p = spearmanr(p_np, c_np)
    stats["pearson_r"] = pearson_r
    stats["pearson_p"] = pearson_p
    stats["spearman_r"] = spearman_r
    stats["spearman_p"] = spearman_p

    # ── Print ──
    print(f"\n  Top-1 cos(r_term, v_term):")
    print(f"    mean  = {stats['top1_cos_mean']:.4f}")
    print(f"    std   = {stats['top1_cos_std']:.4f}")
    print(f"    median= {stats['top1_cos_median']:.4f}")
    print(f"    Q25   = {stats['top1_cos_q25']:.4f}")
    print(f"    Q75   = {stats['top1_cos_q75']:.4f}")
    print(f"    min   = {stats['top1_cos_min']:.4f}")
    print(f"    max   = {stats['top1_cos_max']:.4f}")

    print(f"\n  Discrimination gap:")
    print(f"    top-1 (correct):  {stats['top1_cos_mean']:.4f} ± {stats['top1_cos_std']:.4f}")
    print(f"    rank-2 (other):   {stats['other_cos_mean']:.4f} ± {stats['other_cos_std']:.4f}")
    print(f"    random baseline:  {stats['random_cos_mean']:.4f} ± {stats['random_cos_std']:.4f}")
    gap = stats['top1_cos_mean'] - stats['other_cos_mean']
    print(f"    gap (top1 - rank2): {gap:+.4f}")

    print(f"\n  Per-rank cos(r_term, v_term) mean:")
    for i in range(K):
        print(f"    rank {i+1:2d}: {stats['per_rank_cos_mean'][i]:.4f} "
              f"± {stats['per_rank_cos_std'][i]:.4f}")

    print(f"\n  Correlation: log P(top-1) vs cos(r_term, v_term):")
    print(f"    Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.2e})")
    print(f"    Spearman ρ = {spearman_r:.4f}  (p = {spearman_p:.2e})")

    print(f"\n  Top-1 probability:")
    print(f"    mean = {stats['top1_prob_mean']:.4f} ± {stats['top1_prob_std']:.4f}")

    return {
        "stats": stats,
        "cos_top1": cos_top1.cpu(),
        "cos_other": cos_other.cpu(),
        "cos_random": cos_random.cpu(),
        "cos_by_rank": cos_by_rank.cpu(),
        "topk_probs": topk_probs.cpu(),
        "topk_indices": topk_indices.cpu(),
        "log_top1_p": log_top1_p.cpu(),
    }


# ──────────────────────────────────────────────────────────────────────
# SP4: Terminal Systematicity
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_sp4(data: dict) -> dict:
    """SP4 analysis: weighted-average cos(r_term, v_term) over all words."""
    term_emb = data["term_emb"]      # (T, d)
    vocab_emb = data["vocab_emb"]    # (d, V)
    v_term = data["v_term"]          # (d,)
    tau_term = data["tau_term"]

    print("\n" + "=" * 60)
    print("  SP4: Terminal Systematicity")
    print("=" * 60)

    vocab_emb_t = vocab_emb.t()  # (V, d)

    # Compute P(w|A) for all (A, w)
    template = circular_convolution(
        v_term.unsqueeze(0).expand(T, -1), term_emb, S_DIM
    )  # (T, d)
    logits = template @ vocab_emb  # (T, V)
    logits = logits * tau_term
    probs = logits.softmax(dim=-1)  # (T, V)

    # Compute cos(r_term(A,w), v_term) for ALL w efficiently.
    # Under freq_cnorm: ||r_term|| = ||v_term|| = 1, so cos = <r_term, v_term>.
    # By HolE identity: <r_term, v_term> = <circ_corr(e_A, vocab_w), v_term>
    #   = (circonv(v_term, e_A))^T @ vocab_w = template[A]^T @ vocab_w
    # Since logits = tau_term * template @ vocab_emb:
    #   cos(r_term(A,w), v_term) = logits[A,w] / tau_term

    cos_all = logits / tau_term  # (T, V) -- cos(r_term(A,w), v_term)

    # Weighted average cos over all words for each preterminal
    weighted_cos = (probs * cos_all).sum(dim=-1)  # (T,)

    # Random baseline: random cnorm embeddings for term_emb
    rand_emb = torch.randn_like(term_emb)
    rand_f = torch.fft.rfft(rand_emb, dim=-1)
    rand_f = rand_f / rand_f.abs().clamp(min=1e-12)
    rand_emb = torch.fft.irfft(rand_f, n=S_DIM, dim=-1)

    rand_template = circular_convolution(
        v_term.unsqueeze(0).expand(T, -1), rand_emb, S_DIM
    )
    rand_logits = rand_template @ vocab_emb
    rand_cos_all = rand_logits  # raw cos = template^T @ vocab_w (no tau)
    rand_probs = (rand_logits * tau_term).softmax(dim=-1)
    rand_weighted_cos = (rand_probs * rand_cos_all).sum(dim=-1)

    stats = {
        "learned_weighted_cos_mean": weighted_cos.mean().item(),
        "learned_weighted_cos_std": weighted_cos.std().item(),
        "learned_weighted_cos_median": weighted_cos.median().item(),
        "random_weighted_cos_mean": rand_weighted_cos.mean().item(),
        "random_weighted_cos_std": rand_weighted_cos.std().item(),
    }

    # Effect size (Cohen's d)
    pooled_std = ((weighted_cos.std() ** 2 + rand_weighted_cos.std() ** 2) / 2).sqrt()
    cohens_d = (weighted_cos.mean() - rand_weighted_cos.mean()) / pooled_std
    stats["cohens_d"] = cohens_d.item()

    print(f"\n  Weighted-average cos(r_term, v_term) over all words:")
    print(f"    Learned:  {stats['learned_weighted_cos_mean']:.4f} "
          f"± {stats['learned_weighted_cos_std']:.4f}  "
          f"(median={stats['learned_weighted_cos_median']:.4f})")
    print(f"    Random:   {stats['random_weighted_cos_mean']:.4f} "
          f"± {stats['random_weighted_cos_std']:.4f}")
    print(f"    Cohen's d = {stats['cohens_d']:.2f}")

    # Group by top-1 probability (proxy for "certainty")
    top1_p = probs.max(dim=-1).values
    q_low = top1_p.quantile(0.25)
    q_high = top1_p.quantile(0.75)
    low_mask = top1_p <= q_low
    mid_mask = (top1_p > q_low) & (top1_p <= q_high)
    high_mask = top1_p > q_high

    group_stats = {}
    for name, mask in [("low_p", low_mask), ("mid_p", mid_mask), ("high_p", high_mask)]:
        wc = weighted_cos[mask]
        group_stats[name] = {
            "n": mask.sum().item(),
            "mean": wc.mean().item(),
            "std": wc.std().item(),
        }
        print(f"    {name} (n={mask.sum().item():4d}): "
              f"weighted cos = {wc.mean():.4f} ± {wc.std():.4f}")

    stats["group_stats"] = group_stats

    return {
        "stats": stats,
        "weighted_cos": weighted_cos.cpu(),
        "rand_weighted_cos": rand_weighted_cos.cpu(),
    }


# ──────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────
def plot_figures(sp3: dict, sp4: dict) -> None:
    """Generate all publication-quality figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    # ── Fig 1: Cosine distribution histogram ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cos_top1 = sp3["cos_top1"].numpy()
    cos_random = sp3["cos_random"].numpy()

    ax.hist(cos_top1, bins=60, alpha=0.75, label="Top-1 word", color="#2196F3",
            edgecolor="white", linewidth=0.3)
    ax.hist(cos_random, bins=60, alpha=0.55, label="Random (A, w)", color="#9E9E9E",
            edgecolor="white", linewidth=0.3)
    ax.axvline(cos_top1.mean(), color="#1565C0", linestyle="--", linewidth=1.5,
               label=f"Mean = {cos_top1.mean():.3f}")
    ax.set_xlabel(r"$\cos(r_{\mathrm{term}},\, v_{\mathrm{term}})$")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Terminal Relation Cosine (Top-1)")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_cosine_distribution.svg", format="svg")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'fig1_cosine_distribution.svg'}")

    # ── Fig 2: Cosine vs Probability scatter ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    log_p = sp3["log_top1_p"].numpy()
    cos_v = sp3["cos_top1"].numpy()

    ax.scatter(log_p, cos_v, s=3, alpha=0.25, c="#2196F3", edgecolors="none")
    # Trend line
    z = np.polyfit(log_p, cos_v, 1)
    x_line = np.linspace(log_p.min(), log_p.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r-", linewidth=1.5,
            label=f"Linear fit (slope={z[0]:.3f})")
    r_val = sp3["stats"]["pearson_r"]
    ax.set_xlabel(r"$\log\, P(\mathrm{top\text{-}1}\,|\,A)$")
    ax.set_ylabel(r"$\cos(r_{\mathrm{term}},\, v_{\mathrm{term}})$")
    ax.set_title(f"Cosine vs Log-Probability (Pearson r = {r_val:.3f})")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_cosine_vs_probability.svg", format="svg")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'fig2_cosine_vs_probability.svg'}")

    # ── Fig 3: Per-rank decay bar chart ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    K = len(sp3["stats"]["per_rank_cos_mean"])
    ranks = list(range(1, K + 1))
    means = sp3["stats"]["per_rank_cos_mean"]
    stds = sp3["stats"]["per_rank_cos_std"]

    bars = ax.bar(ranks, means, yerr=stds, capsize=3, color="#2196F3",
                  edgecolor="white", linewidth=0.5, alpha=0.85,
                  error_kw={"linewidth": 1.0})
    # Annotate values on bars
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(sp3["stats"]["random_cos_mean"], color="#9E9E9E", linestyle="--",
               linewidth=1.2, label=f"Random = {sp3['stats']['random_cos_mean']:.3f}")
    ax.set_xlabel("Word Rank (by probability)")
    ax.set_ylabel(r"Mean $\cos(r_{\mathrm{term}},\, v_{\mathrm{term}})$")
    ax.set_title("Cosine by Word Rank")
    ax.set_xticks(ranks)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_per_rank_decay.svg", format="svg")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'fig3_per_rank_decay.svg'}")

    # ── Fig 4: Discrimination box plot ──
    fig, ax = plt.subplots(figsize=(6, 4.5))
    box_data = [
        sp3["cos_top1"].numpy(),
        sp3["cos_other"].numpy(),
        sp3["cos_random"].numpy(),
    ]
    labels = ["Top-1\n(correct)", "Rank-2\n(other)", "Random\n(baseline)"]
    colors = ["#2196F3", "#FF9800", "#9E9E9E"]

    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True,
                    showfliers=False, widths=0.5,
                    medianprops={"color": "black", "linewidth": 1.5})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Annotate medians
    for i, d in enumerate(box_data):
        med = np.median(d)
        ax.text(i + 1, med + 0.01, f"{med:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_ylabel(r"$\cos(r_{\mathrm{term}},\, v_{\mathrm{term}})$")
    ax.set_title("Discrimination: Top-1 vs Other vs Random")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_discrimination.svg", format="svg")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'fig4_discrimination.svg'}")

    # ── Fig 5 (bonus): SP4 weighted cos distribution ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    wc_learned = sp4["weighted_cos"].numpy()
    wc_random = sp4["rand_weighted_cos"].numpy()

    ax.hist(wc_learned, bins=60, alpha=0.75, label="Learned", color="#4CAF50",
            edgecolor="white", linewidth=0.3)
    ax.hist(wc_random, bins=60, alpha=0.55, label="Random baseline", color="#9E9E9E",
            edgecolor="white", linewidth=0.3)
    ax.axvline(wc_learned.mean(), color="#2E7D32", linestyle="--", linewidth=1.5,
               label=f"Learned mean = {wc_learned.mean():.3f}")
    ax.axvline(wc_random.mean(), color="#616161", linestyle="--", linewidth=1.5,
               label=f"Random mean = {wc_random.mean():.3f}")
    ax.set_xlabel(r"Weighted avg $\cos(r_{\mathrm{term}},\, v_{\mathrm{term}})$")
    ax.set_ylabel("Count")
    ax.set_title(f"SP4: Systematicity (Cohen's d = {sp4['stats']['cohens_d']:.2f})")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_sp4_systematicity.svg", format="svg")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'fig5_sp4_systematicity.svg'}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Model: NT={NT}, T={T}, s_dim={S_DIM}, V={V}\n")

    data = load_model()

    # Print basic info
    print(f"  tau_term = {data['tau_term']:.4f}")
    print(f"  tau (rule) = {data['tau']:.4f}")
    print(f"  term_emb: {data['term_emb'].shape}")
    print(f"  vocab_emb: {data['vocab_emb'].shape}")
    print(f"  v_term: {data['v_term'].shape}")

    verify_cnorm(data)

    sp3_results = run_sp3(data)
    sp4_results = run_sp4(data)

    plot_figures(sp3_results, sp4_results)

    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    s3 = sp3_results["stats"]
    s4 = sp4_results["stats"]
    print(f"\n  SP3 (Relation Extraction):")
    print(f"    Top-1 cos(r_term, v_term) = {s3['top1_cos_mean']:.4f} "
          f"± {s3['top1_cos_std']:.4f}")
    print(f"    Discrimination gap (top1 - rank2) = "
          f"{s3['top1_cos_mean'] - s3['other_cos_mean']:+.4f}")
    print(f"    Random baseline = {s3['random_cos_mean']:.4f}")
    print(f"    Pearson r(log P, cos) = {s3['pearson_r']:.4f}")
    print(f"\n  SP4 (Systematicity):")
    print(f"    Weighted cos (learned) = {s4['learned_weighted_cos_mean']:.4f} "
          f"± {s4['learned_weighted_cos_std']:.4f}")
    print(f"    Weighted cos (random)  = {s4['random_weighted_cos_mean']:.4f} "
          f"± {s4['random_weighted_cos_std']:.4f}")
    print(f"    Cohen's d = {s4['cohens_d']:.2f}")
    print()


if __name__ == "__main__":
    main()
