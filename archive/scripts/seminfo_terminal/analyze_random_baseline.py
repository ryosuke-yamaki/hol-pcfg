#!/usr/bin/env python3
"""Random initialization baseline for HN-PCFG SP analyses.

Compares learned model vs randomly initialized model (same architecture)
to separate "learning effects" from "freq_cnorm algebraic properties."

Analyses:
  - SP3: Terminal relation extraction (cos(r_term, v_term))
  - SP4: Terminal systematicity (weighted cos)
  - SP1: Inverse retrieval P@1 on random model
  - v_term delta proximity: random baseline distribution
  - Rule SP3 vs Terminal SP3 on the same checkpoint

Usage:
    python scripts/analyze_random_baseline.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
CKPT_PATH = (
    "/workspace/hol-pcfg-seminfo/ckpt/holeterm-nt1024/"
    "ckpt-sf1_val/sentence_f1=0.67-v2.ckpt"
)
OUTPUT_DIR = Path("/workspace/hol-pcfg/results/random_baseline")

NT = 1024
T = 2048
S_DIM = 512
V = 10020

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────────────────────────────
def freq_cnorm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Frequency-domain circular normalization: |FFT(x)[k]| = 1."""
    x_f = torch.fft.rfft(x, dim=dim)
    x_f = x_f / x_f.abs().clamp(min=1e-12)
    return torch.fft.irfft(x_f, n=x.shape[dim], dim=dim)


def circular_convolution(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    """circonv(a, b) = IFFT(FFT(a) * FFT(b))."""
    a_f = torch.fft.rfft(a, dim=-1)
    b_f = torch.fft.rfft(b, dim=-1)
    return torch.fft.irfft(a_f * b_f, n=n, dim=-1)


def circular_correlation(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    """a star b = IFFT(conj(FFT(a)) * FFT(b))."""
    a_f = torch.fft.rfft(a, dim=-1)
    b_f = torch.fft.rfft(b, dim=-1)
    return torch.fft.irfft(a_f.conj() * b_f, n=n, dim=-1)


def hol_cosine(r: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Cosine for freq_cnorm'd vectors: cos = <r, v> (since ||x|| = 1)."""
    return (r * v).sum(dim=-1)


def make_delta(d: int, device: torch.device) -> torch.Tensor:
    """Identity element for circular convolution."""
    return torch.fft.irfft(torch.ones(d // 2 + 1, device=device), n=d)


# ──────────────────────────────────────────────────────────────────────
# Matplotlib setup
# ──────────────────────────────────────────────────────────────────────
def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.format": "svg",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


# ──────────────────────────────────────────────────────────────────────
# Load checkpoint (learned model)
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def load_learned_model() -> dict:
    """Load learned checkpoint and extract key tensors."""
    raw = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    sd = raw["state_dict"]

    def get(key: str) -> torch.Tensor:
        return sd[f"model.{key}"]

    rule_state_emb = get("rule_state_emb")       # (3072, 512)
    vocab_emb = get("vocab_emb")                  # (512, 10020)
    v_term = get("v_term")                        # (512,)
    v_left = get("v_left").squeeze(0)             # (512,)
    v_right = get("v_right").squeeze(0)           # (512,)
    tau_term = get("log_tau_term").exp().item()
    tau = get("log_tau").exp().item()

    return {
        "rule_state_emb": rule_state_emb,
        "term_emb": rule_state_emb[NT:],
        "nonterm_emb": rule_state_emb[:NT],
        "vocab_emb": vocab_emb,
        "v_term": v_term,
        "v_left": v_left,
        "v_right": v_right,
        "tau_term": tau_term,
        "tau": tau,
    }


# ──────────────────────────────────────────────────────────────────────
# Create random model (epoch 0 equivalent)
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def create_random_model() -> dict:
    """Create a randomly initialized model with the same architecture.

    Matches HN_PCFG.__init__ + project_embeddings(freq_cnorm).
    """
    torch.manual_seed(42)

    # Entity embeddings: N(0, 1/d)
    rule_state_emb = torch.randn(NT + T, S_DIM, device=DEVICE) / math.sqrt(S_DIM)

    # vocab_emb: N(0, 1) then xavier_uniform_ in _initialize()
    # _initialize applies xavier_uniform_ to all parameters with dim > 1
    # EXCEPT v_left, v_right, v_term, scale_c, log_tau*
    # So vocab_emb gets xavier_uniform_
    vocab_emb = torch.empty(S_DIM, V, device=DEVICE)
    torch.nn.init.xavier_uniform_(vocab_emb)

    # Relation vectors: Xavier-scale uniform
    bnd = math.sqrt(6.0 / (1 + S_DIM))
    v_term = torch.empty(S_DIM, device=DEVICE).uniform_(-bnd, bnd)
    v_left = torch.empty(1, S_DIM, device=DEVICE).uniform_(-bnd, bnd).squeeze(0)
    v_right = torch.empty(1, S_DIM, device=DEVICE).uniform_(-bnd, bnd).squeeze(0)

    # Apply freq_cnorm projection (project_embeddings)
    rule_state_emb = freq_cnorm(rule_state_emb, dim=-1)

    # Relation projection cnorm
    v_term = freq_cnorm(v_term.unsqueeze(0), dim=-1).squeeze(0)
    v_left = freq_cnorm(v_left.unsqueeze(0), dim=-1).squeeze(0)
    v_right = freq_cnorm(v_right.unsqueeze(0), dim=-1).squeeze(0)

    # vocab_emb cnorm: applied on (V, s_dim) then transposed back
    ve = vocab_emb.T  # (V, s_dim)
    ve = freq_cnorm(ve, dim=-1)
    vocab_emb = ve.T  # (s_dim, V)

    return {
        "rule_state_emb": rule_state_emb,
        "term_emb": rule_state_emb[NT:],
        "nonterm_emb": rule_state_emb[:NT],
        "vocab_emb": vocab_emb,
        "v_term": v_term,
        "v_left": v_left,
        "v_right": v_right,
        "tau_term": 1.0,  # No learned temperature
        "tau": 1.0,
    }


# ──────────────────────────────────────────────────────────────────────
# SP3: Terminal relation extraction
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_sp3(data: dict, label: str) -> dict:
    """SP3 analysis for terminal productions."""
    term_emb = data["term_emb"]      # (T, d)
    vocab_emb = data["vocab_emb"]    # (d, V)
    v_term = data["v_term"]          # (d,)
    tau_term = data["tau_term"]

    print(f"\n{'=' * 60}")
    print(f"  SP3: Terminal Relation Extraction [{label}]")
    print(f"{'=' * 60}")
    print(f"  tau_term = {tau_term:.4f}")

    # P(w|A) for each preterminal
    template = circular_convolution(
        v_term.unsqueeze(0).expand(T, -1), term_emb, S_DIM
    )
    logits = template @ vocab_emb
    logits_scaled = logits * tau_term
    probs = logits_scaled.softmax(dim=-1)

    # Top-K
    K = 10
    topk_probs, topk_indices = probs.topk(K, dim=-1)

    # vocab_emb transposed for indexing
    vocab_emb_t = vocab_emb.t()  # (V, d)

    # cos(r_term, v_term) for each rank
    cos_by_rank = []
    for k in range(K):
        w_indices = topk_indices[:, k]
        w_emb = vocab_emb_t[w_indices]
        r_term = circular_correlation(term_emb, w_emb, S_DIM)
        cos_k = hol_cosine(r_term, v_term.unsqueeze(0))
        cos_by_rank.append(cos_k)
    cos_by_rank = torch.stack(cos_by_rank, dim=1)  # (T, K)
    cos_top1 = cos_by_rank[:, 0]

    # Random baseline: random (A, w) pairs
    rand_A = torch.randint(0, T, (T,), device=term_emb.device)
    rand_w = torch.randint(0, V, (T,), device=term_emb.device)
    r_rand = circular_correlation(term_emb[rand_A], vocab_emb_t[rand_w], S_DIM)
    cos_random = hol_cosine(r_rand, v_term.unsqueeze(0))

    # Rank-2
    other_indices = topk_indices[:, 1]
    r_other = circular_correlation(term_emb, vocab_emb_t[other_indices], S_DIM)
    cos_other = hol_cosine(r_other, v_term.unsqueeze(0))

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
        "top1_prob_mean": topk_probs[:, 0].mean().item(),
        "top1_prob_std": topk_probs[:, 0].std().item(),
        "per_rank_cos_mean": [cos_by_rank[:, i].mean().item() for i in range(K)],
    }

    # Top-1 probability entropy (how peaked is the distribution?)
    entropy = -(probs * probs.clamp(min=1e-30).log()).sum(dim=-1)
    stats["entropy_mean"] = entropy.mean().item()
    stats["entropy_std"] = entropy.std().item()
    max_entropy = math.log(V)
    stats["entropy_ratio"] = entropy.mean().item() / max_entropy

    # SP4: weighted cos
    cos_all = logits  # cos(r_term(A,w), v_term) = template^T @ vocab_w (no tau)
    weighted_cos = (probs * cos_all).sum(dim=-1)
    stats["sp4_weighted_cos_mean"] = weighted_cos.mean().item()
    stats["sp4_weighted_cos_std"] = weighted_cos.std().item()

    print(f"  Top-1 cos(r_term, v_term): {stats['top1_cos_mean']:.4f} +/- {stats['top1_cos_std']:.4f}")
    print(f"  Top-1 cos median: {stats['top1_cos_median']:.4f}")
    print(f"  Rank-2 cos: {stats['other_cos_mean']:.4f} +/- {stats['other_cos_std']:.4f}")
    print(f"  Random cos: {stats['random_cos_mean']:.4f} +/- {stats['random_cos_std']:.4f}")
    print(f"  Discrimination gap (top1 - rank2): {stats['top1_cos_mean'] - stats['other_cos_mean']:+.4f}")
    print(f"  Top-1 prob: {stats['top1_prob_mean']:.4f} +/- {stats['top1_prob_std']:.4f}")
    print(f"  Entropy: {stats['entropy_mean']:.2f} / {max_entropy:.2f} "
          f"(ratio={stats['entropy_ratio']:.4f})")
    print(f"  SP4 weighted cos: {stats['sp4_weighted_cos_mean']:.4f} +/- {stats['sp4_weighted_cos_std']:.4f}")

    print(f"  Per-rank cos(r_term, v_term) mean:")
    for i in range(K):
        print(f"    rank {i+1:2d}: {stats['per_rank_cos_mean'][i]:.4f}")

    return {
        "stats": stats,
        "cos_top1": cos_top1.cpu(),
        "cos_other": cos_other.cpu(),
        "cos_random": cos_random.cpu(),
        "cos_by_rank": cos_by_rank.cpu(),
        "weighted_cos": weighted_cos.cpu(),
    }


# ──────────────────────────────────────────────────────────────────────
# SP1: Inverse retrieval on random model
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_sp1_random(data: dict) -> dict:
    """SP1 inverse retrieval verification on random model."""
    print(f"\n{'=' * 60}")
    print(f"  SP1: Inverse Retrieval (Random Model)")
    print(f"{'=' * 60}")

    term_emb = data["term_emb"]       # (T, d)
    vocab_emb = data["vocab_emb"]     # (d, V)
    v_term = data["v_term"]           # (d,)
    tau_term = data["tau_term"]

    # Forward: template = circonv(v_term, e_A), scores = template @ vocab_emb
    v_f = torch.fft.rfft(v_term)
    term_f = torch.fft.rfft(term_emb, dim=-1)
    template = torch.fft.irfft(v_f.unsqueeze(0) * term_f, n=S_DIM, dim=-1)
    full_scores = template @ vocab_emb  # (T, V)

    # Inverse: inv_template = circonv(v_term_inv, vocab_w), inv_score = e_A @ inv_template
    v_term_inv_f = v_f.conj()
    vocab_emb_t = vocab_emb.t()  # (V, d)
    vocab_f = torch.fft.rfft(vocab_emb_t, dim=-1)
    inv_templates = torch.fft.irfft(
        v_term_inv_f.unsqueeze(0) * vocab_f, n=S_DIM, dim=-1
    )  # (V, d)
    inv_scores_wt = inv_templates @ term_emb.T  # (V, T)

    # For each word: find forward-best PT, then check inverse rank
    fwd_best_per_word = full_scores.argmax(dim=0)  # (V,)

    word_start = 20
    eval_range = range(word_start, V)
    ranks = []
    for w_idx in eval_range:
        gt_pt = fwd_best_per_word[w_idx].item()
        inv_sc = inv_scores_wt[w_idx]
        rank = (inv_sc > inv_sc[gt_pt]).sum().item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    p_at_1 = (ranks == 1).mean()
    mrr = (1.0 / ranks).mean()

    print(f"  P@1 (random model): {p_at_1:.6f}")
    print(f"  MRR (random model): {mrr:.6f}")
    print(f"  Median rank: {np.median(ranks):.1f}")
    print(f"  Mean rank:   {ranks.mean():.2f}")

    # Verify algebraic identity: forward == inverse score
    max_diff = (full_scores - inv_scores_wt.T).abs().max().item()
    print(f"  max |forward - inverse| score: {max_diff:.2e} (should be ~0)")

    return {
        "p_at_1": p_at_1,
        "mrr": mrr,
        "ranks": ranks,
        "max_score_diff": max_diff,
    }


# ──────────────────────────────────────────────────────────────────────
# v_term delta analysis with random baseline distribution
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_vterm_delta_analysis(learned: dict) -> dict:
    """Analyze v_term proximity to delta with random baseline distribution."""
    print(f"\n{'=' * 60}")
    print(f"  v_term Delta Proximity Analysis")
    print(f"{'=' * 60}")

    v_term = learned["v_term"]
    delta = make_delta(S_DIM, DEVICE)

    cos_vt_delta = F.cosine_similarity(
        v_term.unsqueeze(0), delta.unsqueeze(0)
    ).item()
    print(f"  cos(v_term_learned, delta) = {cos_vt_delta:.6f}")

    # Generate 1000 random freq_cnorm vectors and compute cos with delta
    N_RAND = 1000
    rand_vecs = torch.randn(N_RAND, S_DIM, device=DEVICE)
    rand_vecs = freq_cnorm(rand_vecs, dim=-1)

    cos_rand_delta = F.cosine_similarity(
        rand_vecs, delta.unsqueeze(0), dim=-1
    ).cpu().numpy()

    print(f"  Random cos(v_random, delta): mean={cos_rand_delta.mean():.4f} "
          f"+/- {cos_rand_delta.std():.4f}")
    print(f"  Random cos range: [{cos_rand_delta.min():.4f}, {cos_rand_delta.max():.4f}]")

    # Where does the learned v_term fall?
    percentile = (cos_rand_delta < cos_vt_delta).mean() * 100
    print(f"  Learned v_term percentile in random dist: {percentile:.1f}%")

    # z-score
    z_score = (cos_vt_delta - cos_rand_delta.mean()) / cos_rand_delta.std()
    print(f"  z-score: {z_score:.2f}")

    # FFT phase analysis: learned vs random
    v_f_learned = torch.fft.rfft(v_term)
    phases_learned = torch.angle(v_f_learned).cpu().numpy()

    rand_v = freq_cnorm(torch.randn(1, S_DIM, device=DEVICE)).squeeze(0)
    v_f_random = torch.fft.rfft(rand_v)
    phases_random = torch.angle(v_f_random).cpu().numpy()

    # Phase entropy (uniformity test)
    # For uniform phase: mean should be ~0, std should be ~pi/sqrt(3)
    expected_std = math.pi / math.sqrt(3)
    print(f"\n  FFT Phase Distribution:")
    print(f"    Learned: mean={phases_learned.mean():.4f}, std={phases_learned.std():.4f}")
    print(f"    Random:  mean={phases_random.mean():.4f}, std={phases_random.std():.4f}")
    print(f"    Uniform expected std: {expected_std:.4f}")

    return {
        "cos_vt_delta": cos_vt_delta,
        "cos_rand_delta": cos_rand_delta,
        "percentile": percentile,
        "z_score": z_score,
        "phases_learned": phases_learned,
        "phases_random": phases_random,
    }


# ──────────────────────────────────────────────────────────────────────
# Rule SP3 on learned model
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_rule_sp3(data: dict) -> dict:
    """Rule SP3 analysis using v_left on nonterminal embeddings."""
    print(f"\n{'=' * 60}")
    print(f"  Rule SP3: Nonterminal Relation Extraction (Learned Model)")
    print(f"{'=' * 60}")

    nonterm_emb = data["nonterm_emb"]     # (NT, d)
    all_emb = data["rule_state_emb"]       # (NT+T, d)
    v_left = data["v_left"]                # (d,)
    tau = data["tau"]

    print(f"  tau (rule) = {tau:.4f}")
    print(f"  NT = {NT}, total symbols = {NT + T}")

    # Template = circonv(v_left, e_parent) for each parent NT
    template = circular_convolution(
        v_left.unsqueeze(0).expand(NT, -1), nonterm_emb, S_DIM
    )  # (NT, d)

    # Logits: score of each child given each parent
    # all_emb @ template.T = (NT+T, d) @ (d, NT) = (NT+T, NT)
    logits = all_emb @ template.T  # (NT+T, NT)
    logits_scaled = logits * tau
    probs = logits_scaled.softmax(dim=0)  # softmax over children (dim=0)

    # Top-1 child per parent
    top1_indices = probs.argmax(dim=0)  # (NT,) -- index into all_emb

    # r_rule = circ_corr(e_parent, e_child_top1)
    top1_child_emb = all_emb[top1_indices]  # (NT, d)
    r_rule = circular_correlation(nonterm_emb, top1_child_emb, S_DIM)  # (NT, d)
    cos_top1 = hol_cosine(r_rule, v_left.unsqueeze(0))  # (NT,)

    # Rank-2 child
    _, topk_indices = probs.topk(2, dim=0)  # (2, NT)
    rank2_indices = topk_indices[1]  # (NT,)
    rank2_child_emb = all_emb[rank2_indices]
    r_rank2 = circular_correlation(nonterm_emb, rank2_child_emb, S_DIM)
    cos_rank2 = hol_cosine(r_rank2, v_left.unsqueeze(0))

    # Random pairs
    rand_parent = torch.randint(0, NT, (NT,), device=nonterm_emb.device)
    rand_child = torch.randint(0, NT + T, (NT,), device=nonterm_emb.device)
    r_rand = circular_correlation(nonterm_emb[rand_parent], all_emb[rand_child], S_DIM)
    cos_random = hol_cosine(r_rand, v_left.unsqueeze(0))

    stats = {
        "top1_cos_mean": cos_top1.mean().item(),
        "top1_cos_std": cos_top1.std().item(),
        "top1_cos_median": cos_top1.median().item(),
        "rank2_cos_mean": cos_rank2.mean().item(),
        "rank2_cos_std": cos_rank2.std().item(),
        "random_cos_mean": cos_random.mean().item(),
        "random_cos_std": cos_random.std().item(),
    }

    print(f"  Top-1 cos(r_rule, v_left): {stats['top1_cos_mean']:.4f} +/- {stats['top1_cos_std']:.4f}")
    print(f"  Top-1 cos median: {stats['top1_cos_median']:.4f}")
    print(f"  Rank-2 cos: {stats['rank2_cos_mean']:.4f} +/- {stats['rank2_cos_std']:.4f}")
    print(f"  Random cos: {stats['random_cos_mean']:.4f} +/- {stats['random_cos_std']:.4f}")
    print(f"  Gap (top1 - rank2): {stats['top1_cos_mean'] - stats['rank2_cos_mean']:+.4f}")

    return {
        "stats": stats,
        "cos_top1": cos_top1.cpu(),
        "cos_rank2": cos_rank2.cpu(),
        "cos_random": cos_random.cpu(),
    }


# ──────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────
def plot_all(
    sp3_learned: dict,
    sp3_random: dict,
    sp1_random: dict,
    vterm_delta: dict,
    rule_sp3: dict,
) -> None:
    """Generate all publication-quality figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    # ── Fig 1: SP3 learned vs random -- side-by-side histograms ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    ax = axes[0]
    cos_l = sp3_learned["cos_top1"].numpy()
    ax.hist(cos_l, bins=60, alpha=0.8, color="#2196F3", edgecolor="white", linewidth=0.3)
    ax.axvline(cos_l.mean(), color="#1565C0", linestyle="--", linewidth=1.5,
               label=f"Mean = {cos_l.mean():.3f}")
    ax.set_xlabel(r"$\cos(r_{\mathrm{term}},\, v_{\mathrm{term}})$")
    ax.set_ylabel("Count")
    ax.set_title("Learned Model (Top-1)")
    ax.legend(framealpha=0.9)
    ax.set_xlim(-0.2, 1.0)

    ax = axes[1]
    cos_r = sp3_random["cos_top1"].numpy()
    ax.hist(cos_r, bins=60, alpha=0.8, color="#FF9800", edgecolor="white", linewidth=0.3)
    ax.axvline(cos_r.mean(), color="#E65100", linestyle="--", linewidth=1.5,
               label=f"Mean = {cos_r.mean():.3f}")
    ax.set_xlabel(r"$\cos(r_{\mathrm{term}},\, v_{\mathrm{term}})$")
    ax.set_title("Random Init Model (Top-1)")
    ax.legend(framealpha=0.9)
    ax.set_xlim(-0.2, 1.0)

    fig.suptitle(
        "SP3 Top-1 $\\cos(r_{\\mathrm{term}}, v_{\\mathrm{term}})$: "
        "Learned vs Random Initialization",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_DIR / "fig1_sp3_learned_vs_random.svg", format="svg")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'fig1_sp3_learned_vs_random.svg'}")

    # ── Fig 2: SP3 discrimination comparison (grouped bar) ──
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["Top-1", "Rank-2", "Random pair"]
    learned_vals = [
        sp3_learned["stats"]["top1_cos_mean"],
        sp3_learned["stats"]["other_cos_mean"],
        sp3_learned["stats"]["random_cos_mean"],
    ]
    learned_errs = [
        sp3_learned["stats"]["top1_cos_std"],
        sp3_learned["stats"]["other_cos_std"],
        sp3_learned["stats"]["random_cos_std"],
    ]
    random_vals = [
        sp3_random["stats"]["top1_cos_mean"],
        sp3_random["stats"]["other_cos_mean"],
        sp3_random["stats"]["random_cos_mean"],
    ]
    random_errs = [
        sp3_random["stats"]["top1_cos_std"],
        sp3_random["stats"]["other_cos_std"],
        sp3_random["stats"]["random_cos_std"],
    ]

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width / 2, learned_vals, width, yerr=learned_errs,
                   label="Learned", color="#2196F3", capsize=4, alpha=0.85,
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, random_vals, width, yerr=random_errs,
                   label="Random Init", color="#FF9800", capsize=4, alpha=0.85,
                   edgecolor="white", linewidth=0.5)

    # Annotate values
    for bar, val in zip(bars1, learned_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, random_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel(r"Mean $\cos(r_{\mathrm{term}},\, v_{\mathrm{term}})$")
    ax.set_title("SP3 Discrimination: Learned vs Random Initialization")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(framealpha=0.9)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_sp3_discrimination_comparison.svg", format="svg")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'fig2_sp3_discrimination_comparison.svg'}")

    # ── Fig 3: SP1 random verification ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ranks = sp1_random["ranks"]
    rank_bins = [1, 2, 5, 10, 50, 100, 500, 2048]
    cumulative = [(ranks <= r).mean() * 100 for r in rank_bins]
    ax.bar(range(len(rank_bins)), cumulative, color="#4CAF50", alpha=0.85,
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(rank_bins)))
    ax.set_xticklabels([str(r) for r in rank_bins])
    ax.set_xlabel("Rank Threshold")
    ax.set_ylabel("Cumulative Accuracy (%)")
    ax.set_title(f"SP1 Inverse Retrieval (Random)\nP@1 = {sp1_random['p_at_1']:.4f}")
    for i, v in enumerate(cumulative):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 110)

    ax = axes[1]
    # Show rank distribution (log scale)
    rank_vals, rank_counts = np.unique(ranks, return_counts=True)
    ax.bar(rank_vals[:30], rank_counts[:30], color="#4CAF50", alpha=0.85,
           edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Inverse Retrieval Rank")
    ax.set_ylabel("Count")
    ax.set_title(f"Rank Distribution (first 30)\nMRR = {sp1_random['mrr']:.4f}")
    ax.set_yscale("log")

    fig.suptitle(
        "SP1 Verification on Random Model: "
        "Algebraic Identity Guarantees P@1 = 1.0",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTPUT_DIR / "fig3_sp1_random_verification.svg", format="svg")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'fig3_sp1_random_verification.svg'}")

    # ── Fig 4: v_term vs random distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    cos_rand = vterm_delta["cos_rand_delta"]
    ax.hist(cos_rand, bins=50, alpha=0.8, color="#9E9E9E", edgecolor="white",
            linewidth=0.3, label="Random freq_cnorm vectors")
    ax.axvline(vterm_delta["cos_vt_delta"], color="#F44336", linewidth=2,
               linestyle="-", label=f"Learned v_term = {vterm_delta['cos_vt_delta']:.3f}")
    ax.axvline(cos_rand.mean(), color="#616161", linewidth=1.2, linestyle="--",
               label=f"Random mean = {cos_rand.mean():.3f}")
    ax.set_xlabel(r"$\cos(v, \delta)$")
    ax.set_ylabel("Count")
    ax.set_title(
        f"v_term vs Delta: Learned vs Random\n"
        f"(z = {vterm_delta['z_score']:.2f}, "
        f"percentile = {vterm_delta['percentile']:.1f}%)"
    )
    ax.legend(framealpha=0.9, fontsize=8)

    ax = axes[1]
    ax.hist(vterm_delta["phases_learned"], bins=50, alpha=0.7, color="#2196F3",
            edgecolor="white", linewidth=0.3, label="Learned v_term", density=True)
    ax.hist(vterm_delta["phases_random"], bins=50, alpha=0.5, color="#FF9800",
            edgecolor="white", linewidth=0.3, label="Random v", density=True)
    ax.axvline(0, color="#F44336", linestyle="--", linewidth=1.5, label="delta phase (0)")
    ax.set_xlabel("FFT Phase (radians)")
    ax.set_ylabel("Density")
    ax.set_title("FFT Phase Distribution: Learned vs Random")
    ax.set_xlim(-math.pi, math.pi)
    ax.legend(framealpha=0.9, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_vterm_vs_random_distribution.svg", format="svg")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'fig4_vterm_vs_random_distribution.svg'}")

    # ── Fig 5: Rule SP3 vs Terminal SP3 on same checkpoint ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    cos_term = sp3_learned["cos_top1"].numpy()
    cos_rule = rule_sp3["cos_top1"].numpy()
    ax.hist(cos_term, bins=60, alpha=0.7, color="#2196F3",
            edgecolor="white", linewidth=0.3, label="Terminal SP3")
    ax.hist(cos_rule, bins=60, alpha=0.6, color="#9C27B0",
            edgecolor="white", linewidth=0.3, label="Rule SP3")
    ax.axvline(cos_term.mean(), color="#1565C0", linestyle="--", linewidth=1.5,
               label=f"Term mean = {cos_term.mean():.3f}")
    ax.axvline(cos_rule.mean(), color="#6A1B9A", linestyle="--", linewidth=1.5,
               label=f"Rule mean = {cos_rule.mean():.3f}")
    ax.set_xlabel(r"$\cos(r, v)$ (Top-1)")
    ax.set_ylabel("Count")
    ax.set_title("Top-1 Cosine Distribution")
    ax.legend(framealpha=0.9, fontsize=8)

    ax = axes[1]
    # Discrimination comparison
    categories = ["Top-1", "Rank-2", "Random"]
    term_vals = [
        sp3_learned["stats"]["top1_cos_mean"],
        sp3_learned["stats"]["other_cos_mean"],
        sp3_learned["stats"]["random_cos_mean"],
    ]
    rule_vals = [
        rule_sp3["stats"]["top1_cos_mean"],
        rule_sp3["stats"]["rank2_cos_mean"],
        rule_sp3["stats"]["random_cos_mean"],
    ]
    x = np.arange(len(categories))
    width = 0.35
    b1 = ax.bar(x - width / 2, term_vals, width, label="Terminal (v_term)",
                color="#2196F3", alpha=0.85, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + width / 2, rule_vals, width, label="Rule (v_left)",
                color="#9C27B0", alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(b1, term_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(b2, rule_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel(r"Mean $\cos(r, v)$")
    ax.set_title("Discrimination: Terminal vs Rule SP3")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(framealpha=0.9)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-")

    fig.suptitle(
        "Rule SP3 vs Terminal SP3 on Same Checkpoint (Learned)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTPUT_DIR / "fig5_rule_vs_terminal_sp3_same_ckpt.svg", format="svg")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'fig5_rule_vs_terminal_sp3_same_ckpt.svg'}")


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
def print_summary(
    sp3_learned: dict,
    sp3_random: dict,
    sp1_random: dict,
    vterm_delta: dict,
    rule_sp3: dict,
) -> None:
    sl = sp3_learned["stats"]
    sr = sp3_random["stats"]
    rl = rule_sp3["stats"]

    print("\n" + "=" * 70)
    print("  SUMMARY (Random Initialization Baseline)")
    print("=" * 70)

    print(f"\n  [SP3 Terminal Relation Extraction]")
    print(f"    {'Metric':<30s}  {'Learned':>10s}  {'Random':>10s}  {'Delta':>10s}")
    print(f"    {'-' * 65}")
    for key, label in [
        ("top1_cos_mean", "Top-1 cos mean"),
        ("top1_cos_median", "Top-1 cos median"),
        ("other_cos_mean", "Rank-2 cos mean"),
        ("random_cos_mean", "Random cos mean"),
        ("top1_prob_mean", "Top-1 prob mean"),
    ]:
        lv = sl[key]
        rv = sr[key]
        delta = lv - rv
        print(f"    {label:<30s}  {lv:>10.4f}  {rv:>10.4f}  {delta:>+10.4f}")

    disc_l = sl["top1_cos_mean"] - sl["other_cos_mean"]
    disc_r = sr["top1_cos_mean"] - sr["other_cos_mean"]
    print(f"    {'Discrimination gap':<30s}  {disc_l:>10.4f}  {disc_r:>10.4f}  {disc_l - disc_r:>+10.4f}")

    print(f"\n  [SP4 Terminal Systematicity]")
    print(f"    Learned weighted cos: {sl['sp4_weighted_cos_mean']:.4f} +/- {sl['sp4_weighted_cos_std']:.4f}")
    print(f"    Random  weighted cos: {sr['sp4_weighted_cos_mean']:.4f} +/- {sr['sp4_weighted_cos_std']:.4f}")

    print(f"\n  [SP1 Inverse Retrieval]")
    print(f"    Random model P@1: {sp1_random['p_at_1']:.6f}")
    print(f"    Random model MRR: {sp1_random['mrr']:.6f}")
    print(f"    => {'Algebraic identity confirmed' if sp1_random['p_at_1'] > 0.999 else 'UNEXPECTED: P@1 < 1.0'}")

    print(f"\n  [v_term Delta Proximity]")
    print(f"    cos(v_term_learned, delta): {vterm_delta['cos_vt_delta']:.4f}")
    print(f"    Random cos(v, delta) mean: {vterm_delta['cos_rand_delta'].mean():.4f} "
          f"+/- {vterm_delta['cos_rand_delta'].std():.4f}")
    print(f"    z-score: {vterm_delta['z_score']:.2f}")
    print(f"    percentile: {vterm_delta['percentile']:.1f}%")

    print(f"\n  [Rule SP3 vs Terminal SP3 (Same Checkpoint)]")
    print(f"    {'Metric':<30s}  {'Terminal':>10s}  {'Rule':>10s}")
    print(f"    {'-' * 55}")
    print(f"    {'Top-1 cos mean':<30s}  {sl['top1_cos_mean']:>10.4f}  {rl['top1_cos_mean']:>10.4f}")
    print(f"    {'Top-1 cos median':<30s}  {sl['top1_cos_median']:>10.4f}  {rl['top1_cos_median']:>10.4f}")
    print(f"    {'Rank-2 cos mean':<30s}  {sl['other_cos_mean']:>10.4f}  {rl['rank2_cos_mean']:>10.4f}")
    print(f"    {'Random cos mean':<30s}  {sl['random_cos_mean']:>10.4f}  {rl['random_cos_mean']:>10.4f}")
    disc_term = sl["top1_cos_mean"] - sl["other_cos_mean"]
    disc_rule = rl["top1_cos_mean"] - rl["rank2_cos_mean"]
    print(f"    {'Discrimination gap':<30s}  {disc_term:>10.4f}  {disc_rule:>10.4f}")

    print(f"\n  [Entropy Analysis]")
    max_ent = math.log(V)
    print(f"    Learned entropy: {sl['entropy_mean']:.2f} / {max_ent:.2f} "
          f"(ratio={sl['entropy_ratio']:.4f})")
    print(f"    Random  entropy: {sr['entropy_mean']:.2f} / {max_ent:.2f} "
          f"(ratio={sr['entropy_ratio']:.4f})")

    # ── Key conclusions ──
    print(f"\n  {'=' * 60}")
    print(f"  KEY CONCLUSIONS")
    print(f"  {'=' * 60}")

    random_baseline = sr["top1_cos_mean"]
    learned_val = sl["top1_cos_mean"]
    learning_contrib = learned_val - random_baseline
    total_range = 1.0 - 0.0  # cos range [0, 1] for top-1

    print(f"\n  1. SP3 Top-1 cos: Random baseline = {random_baseline:.4f}, "
          f"Learned = {learned_val:.4f}")
    print(f"     Learning contribution: {learning_contrib:+.4f}")
    if random_baseline > 0.5:
        print(f"     => freq_cnorm algebra alone explains a LARGE portion ({random_baseline:.1%}) of the signal")
    else:
        print(f"     => Learning provides SUBSTANTIAL improvement over algebraic baseline")

    if sp1_random["p_at_1"] > 0.999:
        print(f"\n  2. SP1 P@1 = 1.0 confirmed for random model")
        print(f"     => SP1 is a trivial algebraic identity, NOT evidence of learning")

    print()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Random Initialization Baseline Analysis")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Model: NT={NT}, T={T}, s_dim={S_DIM}, V={V}\n")

    # Load both models
    print("Loading learned model...")
    learned = load_learned_model()
    print(f"  tau_term = {learned['tau_term']:.4f}")
    print(f"  tau (rule) = {learned['tau']:.4f}")

    print("\nCreating random model...")
    random_model = create_random_model()

    # Verify both are freq_cnorm'd
    for name, model in [("Learned", learned), ("Random", random_model)]:
        tf = torch.fft.rfft(model["term_emb"], dim=-1)
        dev = (tf.abs() - 1.0).abs().max().item()
        vf = torch.fft.rfft(model["v_term"])
        v_dev = (vf.abs() - 1.0).abs().max().item()
        print(f"  {name}: emb max |FFT|-1 dev = {dev:.2e}, v_term max |FFT|-1 dev = {v_dev:.2e}")

    # Run analyses
    sp3_learned = run_sp3(learned, "Learned")
    sp3_random = run_sp3(random_model, "Random Init")
    sp1_random = run_sp1_random(random_model)
    vterm_delta = run_vterm_delta_analysis(learned)
    rule_sp3 = run_rule_sp3(learned)

    # Visualize
    plot_all(sp3_learned, sp3_random, sp1_random, vterm_delta, rule_sp3)

    # Summary
    print_summary(sp3_learned, sp3_random, sp1_random, vterm_delta, rule_sp3)


if __name__ == "__main__":
    main()
