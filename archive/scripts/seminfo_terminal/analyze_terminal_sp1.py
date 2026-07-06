#!/usr/bin/env python3
"""Terminal SP1 (inverse retrieval) and v_term delta proximity analysis.

Analyzes the MLP-free HN-PCFG model (holeterm-nt1024) to verify:
1. SP1: Inverse retrieval via circonv with v_term_inv
2. v_term proximity to the identity element (delta)
3. Cross-domain relation vector relationships

Usage:
    python scripts/analyze_terminal_sp1.py
"""

import json
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ============================================================
# Configuration
# ============================================================

CKPT_PATH = (
    "/workspace/hol-pcfg-seminfo/ckpt/holeterm-nt1024/ckpt-sf1_val/"
    "sentence_f1=0.67-v2.ckpt"
)
VOCAB_PATH = "/tmp/vocab_idx2word.json"
OUTPUT_DIR = Path("/workspace/hol-pcfg/results/terminal_sp1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NT = 1024
T = 2048
S_DIM = 512
V = 10020

# Representative words: high-freq function words + content words
REPRESENTATIVE_WORDS = [
    "the", "of", "to", "a", "and",      # function words
    "said", "million", "new", "year",    # content words
    "company",                            # content word
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Helpers
# ============================================================


def freq_cnorm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Frequency-domain circular normalization: |FFT(x)[k]| = 1 for all k."""
    x_f = torch.fft.rfft(x, dim=dim)
    x_f = x_f / x_f.abs().clamp(min=1e-12)
    return torch.fft.irfft(x_f, n=x.shape[dim], dim=dim)


def circonv(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Circular convolution via FFT: circonv(a, b) = IFFT(FFT(a) * FFT(b))."""
    n = a.shape[dim]
    a_f = torch.fft.rfft(a, dim=dim)
    b_f = torch.fft.rfft(b, dim=dim)
    return torch.fft.irfft(a_f * b_f, n=n, dim=dim)


def cirinv(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Circular inverse: IFFT(conj(FFT(v))), so circonv(v, cirinv(v)) ≈ delta."""
    n = v.shape[dim]
    v_f = torch.fft.rfft(v, dim=dim)
    return torch.fft.irfft(v_f.conj(), n=n, dim=dim)


def make_delta(d: int, device: torch.device) -> torch.Tensor:
    """Identity element for circular convolution: IFFT([1,1,...,1])."""
    return torch.fft.irfft(torch.ones(d // 2 + 1, device=device), n=d)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two vectors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ============================================================
# Load model
# ============================================================


def load_model() -> dict:
    """Load checkpoint and extract parameters."""
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    sd = ckpt["state_dict"]

    # Strip "model." prefix
    params = {}
    for k, v in sd.items():
        name = k.replace("model.", "")
        params[name] = v.to(DEVICE)

    with open(VOCAB_PATH) as f:
        idx2word = json.load(f)

    return {
        "rule_state_emb": params["rule_state_emb"],   # (NT+T, 512)
        "vocab_emb": params["vocab_emb"],               # (512, V)
        "v_term": params["v_term"],                      # (512,)
        "v_left": params["v_left"],                      # (1, 512)
        "v_right": params["v_right"],                    # (1, 512)
        "log_tau": params["log_tau"],                    # scalar
        "log_tau_term": params["log_tau_term"],          # scalar
        "root_emb": params["root_emb"],                  # (1, 512)
        "idx2word": idx2word,
    }


# ============================================================
# Part 1: SP1 Terminal Inverse Retrieval
# ============================================================


def part1_sp1_inverse_retrieval(model: dict) -> None:
    """SP1: Verify inverse retrieval via circonv with v_term_inv."""
    print("=" * 70)
    print("Part 1: SP1 Terminal Inverse Retrieval")
    print("=" * 70)

    v_term = model["v_term"]                         # (512,)
    term_emb = model["rule_state_emb"][NT:]          # (T, 512)
    vocab_emb = model["vocab_emb"]                   # (512, V)
    vocab_emb_t = vocab_emb.T                        # (V, 512)
    idx2word = model["idx2word"]
    tau_term = model["log_tau_term"].exp().item()

    word2idx = {w: i for i, w in enumerate(idx2word)}

    # --- Step 1: Compute v_term_inv ---
    v_term_inv = cirinv(v_term)

    # --- Step 2: Verify circonv(v_term, v_term_inv) ≈ delta ---
    recovery = circonv(v_term, v_term_inv)
    delta = make_delta(S_DIM, DEVICE)

    peak_val = recovery[0].item()
    off_peak_max = recovery[1:].abs().max().item()
    cos_recovery_delta = cos_sim(recovery, delta)

    print(f"\n--- v_term 逆元の検証 ---")
    print(f"  circonv(v_term, v_term_inv)[0] (peak): {peak_val:.6f}")
    print(f"  circonv(v_term, v_term_inv)[1:] (off-peak max): {off_peak_max:.2e}")
    print(f"  cos(recovery, delta): {cos_recovery_delta:.6f}")
    print(f"  peak / off-peak ratio: {peak_val / off_peak_max:.1f}x")

    # Verify v_term is freq_cnorm'd
    v_f = torch.fft.rfft(v_term)
    v_f_mag = v_f.abs()
    print(f"\n--- v_term の freq_cnorm 検証 ---")
    print(f"  |FFT(v_term)[k]| mean: {v_f_mag.mean():.6f}")
    print(f"  |FFT(v_term)[k]| max deviation from 1: {(v_f_mag - 1).abs().max():.2e}")
    print(f"  ||v_term||: {v_term.norm():.6f}")
    print(f"  tau_term (exp(log_tau_term)): {tau_term:.4f}")

    # --- Step 3: Inverse retrieval for representative words ---
    print(f"\n--- 代表単語の逆検索 (Inverse Retrieval) ---")

    # Forward computation: term_scores[a, w] = tau * (circonv(v_term, e_A) @ vocab_emb_w)
    # All term_emb and vocab_emb are already freq_cnorm'd in the checkpoint
    term_f = torch.fft.rfft(term_emb, dim=-1)
    v_f_expand = torch.fft.rfft(v_term).unsqueeze(0)  # (1, d//2+1)
    templates = torch.fft.irfft(v_f_expand * term_f, n=S_DIM, dim=-1)  # (T, 512)
    full_scores = templates @ vocab_emb  # (T, V)

    # Inverse retrieval: given word w, find best preterminal A
    # inv_template_w = circonv(v_term_inv, vocab_emb_w)
    # score(A) = e_A @ inv_template_w
    vinv_f = torch.fft.rfft(v_term_inv)
    vocab_f = torch.fft.rfft(vocab_emb_t, dim=-1)  # (V, d//2+1)
    inv_templates = torch.fft.irfft(
        vinv_f.unsqueeze(0) * vocab_f, n=S_DIM, dim=-1
    )  # (V, 512)

    # Inverse scores: for each word, score all preterminals
    inv_scores = term_emb @ inv_templates.T  # (T, V) -> but we want (V, T)
    # inv_scores[w, a] = e_A @ circonv(v_term_inv, vocab_emb_w)
    inv_scores_wt = inv_templates @ term_emb.T  # (V, T)

    # --- Parseval identity verification ---
    print(f"\n--- Parseval 恒等式の検証: forward score == inverse score ---")
    fwd_sample = full_scores[:10, :10]
    inv_sample = inv_scores_wt[:10, :10].T
    max_diff = (fwd_sample - inv_sample).abs().max().item()
    mean_diff = (fwd_sample - inv_sample).abs().mean().item()
    print(f"  max |forward - inverse|: {max_diff:.2e}")
    print(f"  mean |forward - inverse|: {mean_diff:.2e}")

    # Full check across all
    fwd_flat = full_scores.flatten()
    inv_flat = inv_scores_wt.T.flatten()
    global_max_diff = (fwd_flat - inv_flat).abs().max().item()
    print(f"  全要素 max |forward - inverse|: {global_max_diff:.2e}")

    # --- Per-word inverse retrieval demo ---
    all_ranks = []
    all_rrs = []
    all_p_at_1 = []
    demo_results = []

    words_to_analyze = [w for w in REPRESENTATIVE_WORDS if w in word2idx]
    print(f"\n  分析対象単語: {words_to_analyze}")

    for word in words_to_analyze:
        w_idx = word2idx[word]

        # Inverse retrieval: rank preterminals for this word
        scores_for_word = inv_scores_wt[w_idx]  # (T,)
        ranked_pts = scores_for_word.argsort(descending=True)

        # Forward check: for top-5 PTs, what's their top-1 word?
        top5_pts = ranked_pts[:5].tolist()
        top5_scores = scores_for_word[ranked_pts[:5]].tolist()

        forward_top1 = []
        for pt_idx in top5_pts:
            fwd_scores_pt = full_scores[pt_idx]  # (V,)
            top1_word_idx = fwd_scores_pt.argmax().item()
            top1_word = idx2word[top1_word_idx]
            forward_top1.append(top1_word)

        # Forward rank of this word under top-1 PT
        best_pt = top5_pts[0]
        fwd_scores_best = full_scores[best_pt]
        fwd_rank = (fwd_scores_best > fwd_scores_best[w_idx]).sum().item() + 1

        # What rank does the ground-truth best PT get?
        # "Best PT" = the one that gives highest forward score for this word
        fwd_scores_for_word = full_scores[:, w_idx]  # (T,)
        gt_best_pt = fwd_scores_for_word.argmax().item()
        inv_rank_of_gt = (scores_for_word > scores_for_word[gt_best_pt]).sum().item() + 1

        print(f"\n  [{word}] (vocab idx={w_idx})")
        print(f"    逆検索 Top-5 PT:")
        for i, (pt, sc) in enumerate(zip(top5_pts, top5_scores)):
            print(f"      #{i+1}: PT_{pt:4d} (score={sc:.4f}, "
                  f"forward top-1='{forward_top1[i]}')")
        print(f"    Forward best PT: PT_{gt_best_pt} "
              f"(forward score={fwd_scores_for_word[gt_best_pt]:.4f})")
        print(f"    Inverse rank of forward-best PT: {inv_rank_of_gt}")
        print(f"    Forward rank of '{word}' under inverse-best PT: {fwd_rank}")

        demo_results.append({
            "word": word,
            "top5_pts": top5_pts,
            "top5_scores": top5_scores,
            "forward_top1": forward_top1,
        })

    # --- Global P@1, MRR, median rank ---
    print(f"\n--- 全語彙に対するグローバル検索性能 ---")

    # For each word w: find forward-best PT, then check inverse rank
    # Forward best: argmax_A full_scores[A, w]
    fwd_best_per_word = full_scores.argmax(dim=0)  # (V,)
    # Inverse best: argmax_A inv_scores_wt[w, A]
    inv_best_per_word = inv_scores_wt.argmax(dim=1)  # (V,)

    # Only evaluate on actual words (skip special tokens)
    word_start = 20  # after special tokens
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
    median_rank = np.median(ranks)
    mean_rank = ranks.mean()

    print(f"  対象語数: {len(ranks)}")
    print(f"  P@1 (inverse rank of forward-best PT == 1): {p_at_1:.4f}")
    print(f"  MRR: {mrr:.4f}")
    print(f"  Median rank: {median_rank:.1f}")
    print(f"  Mean rank: {mean_rank:.2f}")
    print(f"  Rank 分布: 1={np.sum(ranks==1)}, "
          f"<=5={np.sum(ranks<=5)}, "
          f"<=10={np.sum(ranks<=10)}, "
          f"<=50={np.sum(ranks<=50)}, "
          f"<=100={np.sum(ranks<=100)}")

    # --- Figure 1: Bar chart for representative words ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    n_demo = min(len(demo_results), 6)

    for i, res in enumerate(demo_results[:n_demo]):
        ax = axes[i]
        w_idx = word2idx[res["word"]]
        scores = inv_scores_wt[w_idx]
        top10 = scores.argsort(descending=True)[:10]
        top10_scores = scores[top10].cpu().numpy()

        labels = [f"PT_{pt}" for pt in top10.cpu().numpy()]
        colors = ["#2196F3"] * 10
        # Highlight PT whose forward top-1 matches the query word
        for j, pt in enumerate(top10.cpu().numpy()):
            fwd_top1_idx = full_scores[pt].argmax().item()
            if idx2word[fwd_top1_idx] == res["word"]:
                colors[j] = "#FF5722"

        ax.barh(range(10), top10_scores[::-1], color=colors[::-1])
        ax.set_yticks(range(10))
        ax.set_yticklabels(labels[::-1], fontsize=8)
        ax.set_xlabel("Score", fontsize=9)
        ax.set_title(f'"{res["word"]}"', fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)

    # Hide unused axes
    for i in range(n_demo, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "SP1 Inverse Retrieval: Top-10 Preterminals per Word\n"
        "(red = forward top-1 word matches query)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_DIR / "fig1_inverse_retrieval_demo.svg", format="svg", dpi=150)
    plt.close(fig)
    print(f"\n  => fig1_inverse_retrieval_demo.svg 保存完了")


# ============================================================
# Part 2: v_term Delta Proximity Analysis
# ============================================================


def part2_vterm_delta_proximity(model: dict) -> None:
    """Analyze how close v_term is to the identity element (delta)."""
    print("\n" + "=" * 70)
    print("Part 2: v_term Delta Proximity Analysis")
    print("=" * 70)

    v_term = model["v_term"]
    term_emb = model["rule_state_emb"][NT:]
    v_left = model["v_left"].squeeze(0)
    v_right = model["v_right"].squeeze(0)

    # --- Step 1: Delta (identity element) ---
    delta = make_delta(S_DIM, DEVICE)
    print(f"\n--- Delta (恒等元) ---")
    print(f"  delta[0]: {delta[0]:.6f}")
    print(f"  delta[1:5]: {delta[1:5].tolist()}")
    print(f"  ||delta||: {delta.norm():.6f}")

    # --- Step 2: cos(v_term, delta) ---
    cos_vt_delta = cos_sim(v_term, delta)
    print(f"\n--- cos(v_term, delta) ---")
    print(f"  cos(v_term, delta): {cos_vt_delta:.6f}")

    # --- Step 3: FFT phase distribution ---
    v_f = torch.fft.rfft(v_term)
    phases = torch.angle(v_f).cpu().numpy()
    magnitudes = v_f.abs().cpu().numpy()

    print(f"\n--- v_term FFT 位相分布 ---")
    print(f"  |FFT(v_term)[k]| mean: {magnitudes.mean():.6f}, std: {magnitudes.std():.6f}")
    print(f"  phase mean: {phases.mean():.4f} rad ({np.degrees(phases.mean()):.2f} deg)")
    print(f"  phase std: {phases.std():.4f} rad ({np.degrees(phases.std()):.2f} deg)")
    print(f"  phase range: [{phases.min():.4f}, {phases.max():.4f}] rad")

    # For delta: all phases should be 0
    delta_f = torch.fft.rfft(delta)
    delta_phases = torch.angle(delta_f).cpu().numpy()
    print(f"  delta phase mean: {delta_phases.mean():.6f} (should be 0)")

    # --- Figure 2: Phase histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(phases, bins=50, color="#2196F3", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#FF5722", linestyle="--", linewidth=1.5, label="delta phase (0)")
    ax.set_xlabel("Phase (radians)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("FFT Phase Distribution of v_term", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-np.pi, np.pi)

    ax = axes[1]
    ax.hist(magnitudes, bins=50, color="#4CAF50", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axvline(1.0, color="#FF5722", linestyle="--", linewidth=1.5, label="cnorm target (1.0)")
    ax.set_xlabel("|FFT(v_term)[k]|", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("FFT Magnitude Distribution of v_term", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_vterm_phase_distribution.svg", format="svg", dpi=150)
    plt.close(fig)
    print(f"  => fig2_vterm_phase_distribution.svg 保存完了")

    # --- Step 4: Identity proximity for all preterminals ---
    print(f"\n--- 恒等変換近似度: cos(circonv(v_term, e_A), e_A) ---")

    convolved = circonv(v_term.unsqueeze(0), term_emb)  # (T, 512)
    # cos(circonv(v_term, e_A), e_A) for each A
    identity_cos = F.cosine_similarity(convolved, term_emb, dim=-1).cpu().numpy()

    print(f"  mean cos: {identity_cos.mean():.6f}")
    print(f"  std cos:  {identity_cos.std():.6f}")
    print(f"  Note: for cnorm'd vectors, this equals v_term[0] = {v_term[0].item():.6f}")
    print(f"  v_term is NOT close to delta (cos=-0.03 ≈ random)")

    # More informative: score distortion analysis
    # Compare: score_with_vterm(A, w) = circonv(v_term, e_A) @ vocab_w
    #          score_identity(A, w)   = e_A @ vocab_w  (if v_term were delta)
    vocab_emb = model["vocab_emb"]  # (512, V)
    scores_with_vterm = convolved @ vocab_emb  # (T, V)
    scores_identity = term_emb @ vocab_emb     # (T, V)

    # Per-PT correlation: how well do the two scorings agree?
    per_pt_corr = []
    for a in range(T):
        c = torch.corrcoef(torch.stack([scores_with_vterm[a], scores_identity[a]]))[0, 1].item()
        per_pt_corr.append(c)
    per_pt_corr = np.array(per_pt_corr)

    print(f"\n--- Score distortion: circonv(v_term, e_A)@w vs e_A@w ---")
    print(f"  Per-PT Pearson corr: mean={per_pt_corr.mean():.4f}, "
          f"std={per_pt_corr.std():.4f}")
    print(f"  min corr: {per_pt_corr.min():.4f}, max corr: {per_pt_corr.max():.4f}")

    # Top-1 word agreement: does applying v_term change which word is ranked first?
    top1_with = scores_with_vterm.argmax(dim=1)  # (T,)
    top1_without = scores_identity.argmax(dim=1)  # (T,)
    top1_agree = (top1_with == top1_without).float().mean().item()
    print(f"  Top-1 word agreement (with vs without v_term): {top1_agree:.4f}")

    # Rank correlation (Kendall) is expensive, use Spearman on sample
    from scipy.stats import spearmanr
    spearman_corrs = []
    sample_pts = np.random.choice(T, min(200, T), replace=False)
    for a in sample_pts:
        rho, _ = spearmanr(
            scores_with_vterm[a].cpu().numpy(),
            scores_identity[a].cpu().numpy(),
        )
        spearman_corrs.append(rho)
    spearman_corrs = np.array(spearman_corrs)
    print(f"  Spearman rank corr (200 sample PTs): mean={spearman_corrs.mean():.4f}, "
          f"std={spearman_corrs.std():.4f}")

    # --- Figure 3: Score distortion analysis ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Pearson correlation histogram
    ax = axes[0]
    ax.hist(per_pt_corr, bins=50, color="#9C27B0", alpha=0.8,
            edgecolor="white", linewidth=0.5)
    ax.axvline(per_pt_corr.mean(), color="#2196F3", linestyle="-", linewidth=1.5,
               label=f"Mean ({per_pt_corr.mean():.3f})")
    ax.set_xlabel("Pearson r (per preterminal)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Score Correlation:\ncirconv(v_term, e_A)@w vs e_A@w",
                  fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel 2: Phase histogram with delta comparison
    ax = axes[1]
    ax.hist(phases, bins=50, color="#2196F3", alpha=0.8,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#FF5722", linestyle="--", linewidth=1.5, label="delta (phase=0)")
    ax.set_xlabel("FFT Phase of v_term (rad)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("v_term Phase Distribution\n(delta would be all-zero)",
                  fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-np.pi, np.pi)

    # Panel 3: Spearman correlation histogram
    ax = axes[2]
    ax.hist(spearman_corrs, bins=40, color="#4CAF50", alpha=0.8,
            edgecolor="white", linewidth=0.5)
    ax.axvline(spearman_corrs.mean(), color="#2196F3", linestyle="-", linewidth=1.5,
               label=f"Mean ({spearman_corrs.mean():.3f})")
    ax.set_xlabel("Spearman rho (per preterminal)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Rank Correlation:\nv_term vs Identity Scoring",
                  fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Identity Proximity: cos(v_term, delta) = {cos_vt_delta:.3f}  |  "
        f"Top-1 agreement = {top1_agree:.3f}",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_identity_proximity.svg", format="svg",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  => fig3_identity_proximity.svg 保存完了")

    # --- Step 5: Relation vector separation ---
    print(f"\n--- Relation vector 間の距離 ---")
    cos_lr = cos_sim(v_left, v_right)
    cos_lt = cos_sim(v_left, v_term)
    cos_rt = cos_sim(v_right, v_term)
    print(f"  cos(v_left, v_right): {cos_lr:.6f}")
    print(f"  cos(v_left, v_term):  {cos_lt:.6f}")
    print(f"  cos(v_right, v_term): {cos_rt:.6f}")


# ============================================================
# Part 3: Cross-domain Relation Vectors
# ============================================================


def part3_cross_domain_relations(model: dict) -> None:
    """Analyze relationships between v_left, v_right, v_term."""
    print("\n" + "=" * 70)
    print("Part 3: Cross-domain Relation Vectors")
    print("=" * 70)

    v_left = model["v_left"].squeeze(0)   # (512,)
    v_right = model["v_right"].squeeze(0)  # (512,)
    v_term = model["v_term"]               # (512,)

    # --- Pairwise cosines ---
    cos_lr = cos_sim(v_left, v_right)
    cos_lt = cos_sim(v_left, v_term)
    cos_rt = cos_sim(v_right, v_term)

    print(f"\n--- Pairwise cosine similarities ---")
    print(f"  cos(v_left, v_right): {cos_lr:.6f}")
    print(f"  cos(v_left, v_term):  {cos_lt:.6f}")
    print(f"  cos(v_right, v_term): {cos_rt:.6f}")

    # Norms
    print(f"\n--- ノルム ---")
    print(f"  ||v_left||:  {v_left.norm():.6f}")
    print(f"  ||v_right||: {v_right.norm():.6f}")
    print(f"  ||v_term||:  {v_term.norm():.6f}")

    # --- Composed relation: v_lt = circonv(v_left, v_term) ---
    v_lt = circonv(v_left, v_term)
    print(f"\n--- 合成ベクトル: v_lt = circonv(v_left, v_term) ---")
    print(f"  ||v_lt||: {v_lt.norm():.6f}")

    # cnorm preservation check
    v_lt_f = torch.fft.rfft(v_lt)
    print(f"  |FFT(v_lt)[k]| mean: {v_lt_f.abs().mean():.6f}")
    print(f"  |FFT(v_lt)[k]| max deviation from 1: {(v_lt_f.abs() - 1).abs().max():.2e}")

    cos_lt_lr = cos_sim(v_lt, v_right)
    cos_lt_left = cos_sim(v_lt, v_left)
    cos_lt_term = cos_sim(v_lt, v_term)
    print(f"\n--- v_lt との cosine ---")
    print(f"  cos(v_lt, v_left):  {cos_lt_left:.6f}")
    print(f"  cos(v_lt, v_right): {cos_lt_lr:.6f}")
    print(f"  cos(v_lt, v_term):  {cos_lt_term:.6f}")

    # Also check v_rt = circonv(v_right, v_term)
    v_rt = circonv(v_right, v_term)
    print(f"\n--- 合成ベクトル: v_rt = circonv(v_right, v_term) ---")
    print(f"  ||v_rt||: {v_rt.norm():.6f}")

    # Spectral analysis: phase differences
    vl_f = torch.fft.rfft(v_left)
    vr_f = torch.fft.rfft(v_right)
    vt_f = torch.fft.rfft(v_term)

    phase_l = torch.angle(vl_f).cpu().numpy()
    phase_r = torch.angle(vr_f).cpu().numpy()
    phase_t = torch.angle(vt_f).cpu().numpy()

    # Phase difference distributions
    diff_lr = np.angle(np.exp(1j * (phase_l - phase_r)))
    diff_lt = np.angle(np.exp(1j * (phase_l - phase_t)))
    diff_rt = np.angle(np.exp(1j * (phase_r - phase_t)))

    print(f"\n--- 位相差分布 ---")
    print(f"  phase_diff(L,R) std: {diff_lr.std():.4f} rad ({np.degrees(diff_lr.std()):.2f} deg)")
    print(f"  phase_diff(L,T) std: {diff_lt.std():.4f} rad ({np.degrees(diff_lt.std()):.2f} deg)")
    print(f"  phase_diff(R,T) std: {diff_rt.std():.4f} rad ({np.degrees(diff_rt.std()):.2f} deg)")

    # --- Figure 4: Relation vector comparison heatmap ---
    delta = make_delta(S_DIM, DEVICE)
    all_vecs = [v_left, v_right, v_term, v_lt, v_rt, delta]
    vec_names = ["v_left", "v_right", "v_term", "v_left*v_term", "v_right*v_term", "delta"]

    n = len(all_vecs)
    cos_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_matrix[i, j] = cos_sim(all_vecs[i], all_vecs[j])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cos_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = cos_matrix[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_xticklabels(vec_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(vec_names, fontsize=9)
    ax.set_title(
        "Pairwise Cosine Similarities: Relation Vectors",
        fontsize=11, fontweight="bold", pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Similarity", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_relation_vector_comparison.svg", format="svg", dpi=150)
    plt.close(fig)
    print(f"\n  => fig4_relation_vector_comparison.svg 保存完了")


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("Terminal SP1 + v_term Delta Proximity Analysis")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Model: NT={NT}, T={T}, s_dim={S_DIM}, V={V}")
    print()

    model = load_model()

    # Verify shapes
    print("--- パラメータ形状 ---")
    for k in ["rule_state_emb", "vocab_emb", "v_term", "v_left", "v_right", "log_tau", "log_tau_term"]:
        v = model[k]
        shape = v.shape if hasattr(v, "shape") else "scalar"
        print(f"  {k}: {shape}")
    print()

    with torch.no_grad():
        part1_sp1_inverse_retrieval(model)
        part2_vterm_delta_proximity(model)
        part3_cross_domain_relations(model)

    print("\n" + "=" * 70)
    print("分析完了。出力先: " + str(OUTPUT_DIR))
    print("=" * 70)


if __name__ == "__main__":
    main()
