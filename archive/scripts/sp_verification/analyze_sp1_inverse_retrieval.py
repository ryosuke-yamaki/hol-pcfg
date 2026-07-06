"""SP1 Verification: Inverse Retrieval under cnorm.

Demonstrates that under cnorm (|FFT(v)[k]|=1), the inverse relation v⁻¹
trivially exists and enables child→parent retrieval.

Key result: forward score e_B^T circonv(v, e_A) = inverse score e_A^T circonv(v⁻¹, e_B)
due to Parseval + real signals, so the raw score matrix is symmetric.

Run from project root:
    python scripts/analyze_sp1_inverse_retrieval.py
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Setup imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from sp_utils import (
    NT, T, S_DIM, DEFAULT_CHECKPOINT,
    circular_convolution, compute_inverse_relation,
    compute_raw_scores, compute_rule_probs,
    load_checkpoint, load_nt_labels, verify_normalization, print_normalization,
    setup_matplotlib, save_svg,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "sp1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DATA_PATH = "data/ptb-train.pickle"
VOCAB_SIZE = 10000


# ---------------------------------------------------------------------------
# ResLayer (matches parser/modules/res.py)
# ---------------------------------------------------------------------------
class ResLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + x


# ---------------------------------------------------------------------------
# Vocabulary and term_mlp reconstruction
# ---------------------------------------------------------------------------
def build_vocab() -> list[str]:
    """Reconstruct the vocabulary exactly as data_module.py does."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from fastNLP.core.dataset import DataSet
    from fastNLP.core.vocabulary import Vocabulary

    with open(PROJECT_ROOT / TRAIN_DATA_PATH, "rb") as f:
        train_data = pickle.load(f)

    train_dataset = DataSet()
    train_dataset.add_field("word", train_data["word"])

    def clean_word(words: list[str]) -> list[str]:
        def clean_number(w: str) -> str:
            return re.sub(r"[0-9]{1,}([,.][0-9]*)*", "N", w)
        return [clean_number(word.lower()) for word in words]

    train_dataset.apply_field(clean_word, "word", "word")
    word_vocab = Vocabulary(max_size=VOCAB_SIZE)
    word_vocab.from_dataset(train_dataset, field_name="word")
    idx2word: dict[int, str] = {v: k for k, v in word_vocab.word2idx.items()}
    vocab = [idx2word.get(i, f"<idx{i}>") for i in range(len(word_vocab))]
    return vocab


def build_term_mlp(ckpt: dict[str, torch.Tensor]) -> nn.Sequential:
    """Reconstruct term_mlp from checkpoint weights."""
    mlp = nn.Sequential(
        nn.Linear(S_DIM, S_DIM),
        ResLayer(S_DIM, S_DIM),
        ResLayer(S_DIM, S_DIM),
        ResLayer(S_DIM, S_DIM),
    )
    state = {}
    prefix = "term_mlp."
    for k, v in ckpt.items():
        if k.startswith(prefix):
            state[k[len(prefix):]] = v
    mlp.load_state_dict(state)
    mlp.eval()
    return mlp


def get_preterminal_top_words(
    ckpt_raw: dict,
    all_emb: torch.Tensor,
    vocab: list[str],
    top_k: int = 5,
) -> dict[int, list[str]]:
    """Get top-k words for each preterminal (T indices NT..NT+T-1)."""
    term_mlp = build_term_mlp(ckpt_raw).to(DEVICE)
    vocab_emb = ckpt_raw["vocab_emb"].to(DEVICE)  # (S_DIM, V)

    t_emb = all_emb[NT: NT + T]  # (T, S_DIM)
    result: dict[int, list[str]] = {}

    with torch.no_grad():
        # Process in chunks to avoid OOM
        chunk = 2048
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            emb_chunk = t_emb[start:end]
            logits = (term_mlp(emb_chunk) + emb_chunk) @ vocab_emb  # (chunk, V)
            probs = logits.softmax(dim=-1)
            topk_vals, topk_idx = probs.topk(top_k, dim=-1)
            for i in range(end - start):
                t_idx = NT + start + i
                words = [vocab[j.item()] if j.item() < len(vocab) else f"<{j.item()}>"
                         for j in topk_idx[i]]
                result[t_idx] = words

    return result


# ---------------------------------------------------------------------------
# POS label heuristic (simplified from label_nonterminals.py)
# ---------------------------------------------------------------------------
_LABEL_RULES: list[tuple[set[str], str]] = [
    ({"the", "a", "an", "this", "that", "these", "those", "his", "her",
      "its", "my", "your", "our", "their", "no", "some", "any", "each",
      "every", "all"}, "DT"),
    ({"of", "in", "for", "on", "to", "with", "at", "from", "by", "about",
      "into", "through", "during", "before", "after", "between", "under"}, "IN"),
    ({"is", "was", "are", "were", "be", "been", "being", "am"}, "VB-cop"),
    ({"has", "have", "had"}, "VB-aux"),
    ({"will", "would", "could", "should", "may", "might", "can", "shall",
      "must"}, "MD"),
    ({"said", "says"}, "VBD-said"),
    ({"not", "n't"}, "RB-neg"),
    ({"and", "or", "but", "nor"}, "CC"),
    ({",", ".", ":", ";", "!", "?", "...", "--", "``", "''"}, "PUNCT"),
    ({"$", "#", "%"}, "SYM"),
    ({"'s"}, "POS"),
    ({"that", "which", "who", "whom", "whose", "where", "when"}, "WH"),
    ({"do", "does", "did"}, "VB-do"),
]


def infer_t_label(top_words: list[str]) -> str:
    """Heuristic POS-like label for a preterminal based on its top words."""
    top_set = set(top_words[:5])
    for keyword_set, label in _LABEL_RULES:
        if len(top_set & keyword_set) >= 2:
            return label
    # Fallback: use top word
    return top_words[0] if top_words else "?"


# =========================================================================
# Pillar 1: Algebraic Property Verification
# =========================================================================
def verify_algebraic_properties(data: dict) -> dict:
    """Verify v_inv properties and forward/inverse score identity."""
    print("\n" + "=" * 70)
    print("  PILLAR 1: Algebraic Property Verification")
    print("=" * 70)

    v_left = data["v_left"]
    all_emb = data["all_emb"]
    nt_emb = data["nt_emb"]

    # 1. Compute v_left_inv
    v_left_inv = compute_inverse_relation(v_left, S_DIM)

    # 2. Verify |FFT(v_inv)[k]| = 1
    v_inv_f = torch.fft.rfft(v_left_inv)
    v_inv_mag = v_inv_f.abs()
    mag_dev = (v_inv_mag - 1.0).abs().max().item()
    print(f"\n  |FFT(v_inv)[k]| max deviation from 1: {mag_dev:.2e}")

    # 3. Verify circonv(v, v_inv) ≈ δ
    delta = circular_convolution(v_left, v_left_inv, S_DIM)
    peak = delta[0].item()
    off_peak_max = delta[1:].abs().max().item()
    off_peak_mean = delta[1:].abs().mean().item()
    print(f"  circonv(v, v⁻¹): peak={peak:.8f}, off_peak_max={off_peak_max:.2e}, off_peak_mean={off_peak_mean:.2e}")

    # 4. Forward vs inverse score identity (1000 random pairs)
    print("\n  Verifying forward_score ≈ inverse_score for 1000 random (parent, child) pairs...")
    torch.manual_seed(42)
    parent_idx = torch.randint(0, NT, (1000,), device=DEVICE)
    child_idx = torch.randint(0, NT + T, (1000,), device=DEVICE)

    diffs = []
    fwd_scores = []
    inv_scores = []
    for i in range(1000):
        e_A = nt_emb[parent_idx[i]]
        e_B = all_emb[child_idx[i]]
        fwd = e_B @ circular_convolution(v_left, e_A, S_DIM)
        inv = e_A @ circular_convolution(v_left_inv, e_B, S_DIM)
        diffs.append(abs(fwd.item() - inv.item()))
        fwd_scores.append(fwd.item())
        inv_scores.append(inv.item())

    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    print(f"  Max |forward - inverse| = {max_diff:.2e}")
    print(f"  Mean |forward - inverse| = {mean_diff:.2e}")
    passed = max_diff < 1e-4
    print(f"  Score identity check: {'PASSED' if passed else 'FAILED'} (threshold 1e-4)")

    return {
        "v_inv_mag_max_deviation": mag_dev,
        "delta_peak": peak,
        "delta_off_peak_max": off_peak_max,
        "delta_off_peak_mean": off_peak_mean,
        "score_identity_max_diff": max_diff,
        "score_identity_mean_diff": mean_diff,
        "score_identity_passed": passed,
        "fwd_scores": fwd_scores,
        "inv_scores": inv_scores,
    }


# =========================================================================
# Pillar 2: Practical Use Case — Child → Parent Cluster
# =========================================================================
def analyze_child_to_parent(
    data: dict,
    t_top_words: dict[int, list[str]],
) -> dict:
    """For representative preterminals, find top parents via inverse retrieval."""
    print("\n" + "=" * 70)
    print("  PILLAR 2: Child → Parent Cluster (Inverse Retrieval Demo)")
    print("=" * 70)

    v_left = data["v_left"]
    v_right = data["v_right"]
    v_left_inv = compute_inverse_relation(v_left, S_DIM)
    nt_emb = data["nt_emb"]
    all_emb = data["all_emb"]
    tau = data["tau"]

    # Step 1: Find top-1 left children for all parents
    print("\n  Computing left-child top-1 for all parents...")
    left_scores = compute_raw_scores(v_left, nt_emb, all_emb, S_DIM)  # (NT+T, NT)
    left_top1 = left_scores.argmax(dim=0)  # (NT,) — top-1 child for each parent

    # Count how many parents select each child
    child_counts: Counter = Counter(left_top1.cpu().tolist())
    print(f"  Unique left children used: {len(child_counts)} / {NT + T}")
    print(f"  Top-20 most-shared children:")
    for child_id, count in child_counts.most_common(20):
        words = t_top_words.get(child_id, [f"NT-{child_id}"])
        label = infer_t_label(words) if child_id >= NT else f"NT-{child_id}"
        print(f"    idx={child_id:>5d} ({label:>10s}) -> {count:>3d} parents   top_words={words[:3]}")

    # Step 2: Select representative children
    # Function words (shared, high count)
    func_word_targets = {"DT", "IN", "CC", "MD", "VB-cop", "VB-aux", "RB-neg", "PUNCT"}
    selected_children: list[dict] = []

    # Find preterminals matching function-word categories
    for child_id, count in child_counts.most_common(200):
        if child_id < NT:
            continue
        words = t_top_words.get(child_id, [])
        label = infer_t_label(words)
        if label in func_word_targets and count >= 3:
            selected_children.append({
                "idx": child_id, "label": label, "words": words[:5],
                "parent_count": count, "category": "function",
            })
            func_word_targets.discard(label)
        if len(func_word_targets) == 0:
            break

    # Add some content-word preterminals (less shared) for contrast
    content_added = 0
    for child_id, count in child_counts.most_common():
        if child_id < NT or count > 5:
            continue
        words = t_top_words.get(child_id, [])
        label = infer_t_label(words)
        # Skip if it's a function word
        if label in {"DT", "IN", "CC", "MD", "VB-cop", "VB-aux", "RB-neg", "PUNCT", "POS", "WH", "SYM"}:
            continue
        selected_children.append({
            "idx": child_id, "label": label, "words": words[:5],
            "parent_count": count, "category": "content",
        })
        content_added += 1
        if content_added >= 3:
            break

    print(f"\n  Selected {len(selected_children)} representative children:")
    for c in selected_children:
        print(f"    {c['label']:>10s}  words={c['words'][:3]}  parents={c['parent_count']}")

    # Step 3: For each child, inverse retrieve top-10 parents + their right children
    print("\n  Inverse retrieval results:")
    child_results = []

    for c in selected_children:
        child_id = c["idx"]
        e_B = all_emb[child_id]

        # Inverse template
        inv_template = circular_convolution(v_left_inv, e_B, S_DIM)
        parent_scores = nt_emb @ inv_template  # (NT,)
        top10_scores, top10_parents = parent_scores.topk(10)

        # For each top parent, find right child top-1
        parent_info = []
        for rank, (p_idx, p_score) in enumerate(zip(top10_parents.tolist(), top10_scores.tolist())):
            e_parent = nt_emb[p_idx]
            right_template = circular_convolution(v_right, e_parent, S_DIM)
            right_scores = all_emb @ right_template
            right_top1_idx = right_scores.argmax().item()
            right_words = t_top_words.get(right_top1_idx, [f"NT-{right_top1_idx}"])
            right_label = infer_t_label(right_words) if right_top1_idx >= NT else f"NT-{right_top1_idx}"

            parent_info.append({
                "parent_idx": p_idx,
                "score": round(p_score, 4),
                "right_child_idx": right_top1_idx,
                "right_child_label": right_label,
                "right_child_words": right_words[:3],
            })

        c_result = {
            "child_idx": child_id,
            "child_label": c["label"],
            "child_words": c["words"],
            "parent_count": c["parent_count"],
            "category": c["category"],
            "top10_parents": parent_info,
        }
        child_results.append(c_result)

        # Print
        print(f"\n    Child: {c['label']} (words: {c['words'][:3]}, {c['parent_count']} parents)")
        for rank, pi in enumerate(parent_info[:5]):
            print(f"      #{rank+1}: NT-{pi['parent_idx']:<5d} score={pi['score']:.4f}  "
                  f"right_child={pi['right_child_label']} ({pi['right_child_words']})")

    return {
        "child_counts_top20": [
            {"child_idx": cid, "count": cnt,
             "words": t_top_words.get(cid, [f"NT-{cid}"])[:3]}
            for cid, cnt in child_counts.most_common(20)
        ],
        "selected_children": child_results,
    }


# =========================================================================
# Pillar 3: Shared vs Exclusive Child Analysis
# =========================================================================
def analyze_shared_vs_exclusive(
    data: dict,
    t_top_words: dict[int, list[str]],
) -> dict:
    """Compare inverse retrieval metrics for shared vs exclusive children."""
    print("\n" + "=" * 70)
    print("  PILLAR 3: Shared vs Exclusive Child Analysis")
    print("=" * 70)

    v_left = data["v_left"]
    v_left_inv = compute_inverse_relation(v_left, S_DIM)
    nt_emb = data["nt_emb"]
    all_emb = data["all_emb"]

    # Compute left scores and top-1 children
    left_scores = compute_raw_scores(v_left, nt_emb, all_emb, S_DIM)  # (NT+T, NT)
    left_top1 = left_scores.argmax(dim=0)  # (NT,)

    # Build child → set of parents mapping
    child_to_parents: dict[int, list[int]] = defaultdict(list)
    for parent_id in range(NT):
        child_id = left_top1[parent_id].item()
        child_to_parents[child_id].append(parent_id)

    child_counts = {c: len(ps) for c, ps in child_to_parents.items()}

    # Classify: shared (14+ parents) vs exclusive (1-2 parents)
    shared_threshold = 10
    exclusive_threshold = 2
    shared_children = {c: ps for c, ps in child_to_parents.items()
                       if len(ps) >= shared_threshold}
    exclusive_children = {c: ps for c, ps in child_to_parents.items()
                          if len(ps) <= exclusive_threshold}

    print(f"\n  Shared children (>={shared_threshold} parents): {len(shared_children)}")
    print(f"  Exclusive children (<={exclusive_threshold} parents): {len(exclusive_children)}")

    def compute_retrieval_metrics(
        children_dict: dict[int, list[int]],
        label: str,
    ) -> dict:
        """For each child, pick a correct parent and measure inverse retrieval rank."""
        ranks = []
        torch.manual_seed(123)

        for child_id, parent_list in children_dict.items():
            e_B = all_emb[child_id]
            inv_template = circular_convolution(v_left_inv, e_B, S_DIM)
            parent_scores = nt_emb @ inv_template  # (NT,)

            # Sort parents by score (descending)
            sorted_indices = parent_scores.argsort(descending=True)

            # Pick a random correct parent
            correct_parent = parent_list[torch.randint(0, len(parent_list), (1,)).item()]
            rank = (sorted_indices == correct_parent).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

        ranks_t = torch.tensor(ranks, dtype=torch.float)
        p_at_1 = (ranks_t <= 1).float().mean().item()
        p_at_5 = (ranks_t <= 5).float().mean().item()
        p_at_10 = (ranks_t <= 10).float().mean().item()
        mrr = (1.0 / ranks_t).mean().item()
        median_rank = ranks_t.median().item()

        print(f"\n  {label}:")
        print(f"    N = {len(ranks)}")
        print(f"    P@1  = {p_at_1:.4f}")
        print(f"    P@5  = {p_at_5:.4f}")
        print(f"    P@10 = {p_at_10:.4f}")
        print(f"    MRR  = {mrr:.4f}")
        print(f"    Median rank = {median_rank:.0f}")

        return {
            "n": len(ranks),
            "p_at_1": round(p_at_1, 4),
            "p_at_5": round(p_at_5, 4),
            "p_at_10": round(p_at_10, 4),
            "mrr": round(mrr, 4),
            "median_rank": round(median_rank, 1),
        }

    shared_metrics = compute_retrieval_metrics(shared_children, f"Shared (>={shared_threshold} parents)")
    exclusive_metrics = compute_retrieval_metrics(exclusive_children, f"Exclusive (<={exclusive_threshold} parents)")

    print(f"\n  Interpretation: Exclusive children have {'higher' if exclusive_metrics['p_at_1'] > shared_metrics['p_at_1'] else 'lower'} P@1 "
          f"({exclusive_metrics['p_at_1']:.4f} vs {shared_metrics['p_at_1']:.4f})")

    return {
        "shared_threshold": shared_threshold,
        "exclusive_threshold": exclusive_threshold,
        "n_shared": len(shared_children),
        "n_exclusive": len(exclusive_children),
        "shared_metrics": shared_metrics,
        "exclusive_metrics": exclusive_metrics,
    }


# =========================================================================
# P@k Metrics (supplementary)
# =========================================================================
def compute_pk_metrics(data: dict) -> dict:
    """Compute P@1, P@5, MRR for all parent-child pairs."""
    print("\n" + "=" * 70)
    print("  Supplementary: P@k Metrics (all 4096 parent-child pairs)")
    print("=" * 70)

    v_left = data["v_left"]
    v_left_inv = compute_inverse_relation(v_left, S_DIM)
    nt_emb = data["nt_emb"]
    all_emb = data["all_emb"]

    # Forward: top-1 child for each parent
    left_scores = compute_raw_scores(v_left, nt_emb, all_emb, S_DIM)  # (NT+T, NT)
    left_top1 = left_scores.argmax(dim=0)  # (NT,)

    # Inverse retrieval: for each parent's top-1 child, retrieve parents
    ranks = []
    for parent_id in range(NT):
        child_id = left_top1[parent_id].item()
        e_B = all_emb[child_id]
        inv_template = circular_convolution(v_left_inv, e_B, S_DIM)
        parent_scores = nt_emb @ inv_template

        sorted_indices = parent_scores.argsort(descending=True)
        rank = (sorted_indices == parent_id).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks_t = torch.tensor(ranks, dtype=torch.float)
    p_at_1 = (ranks_t <= 1).float().mean().item()
    p_at_5 = (ranks_t <= 5).float().mean().item()
    p_at_10 = (ranks_t <= 10).float().mean().item()
    mrr = (1.0 / ranks_t).mean().item()
    median_rank = ranks_t.median().item()

    print(f"\n  P@1  = {p_at_1:.4f}")
    print(f"  P@5  = {p_at_5:.4f}")
    print(f"  P@10 = {p_at_10:.4f}")
    print(f"  MRR  = {mrr:.4f}")
    print(f"  Median rank = {median_rank:.0f}")
    print(f"\n  NOTE: High P@1 is expected because forward score = inverse score (Parseval identity).")
    print(f"        If child B is unique top-1 for parent A, then A is automatically top-1 for child B.")

    return {
        "p_at_1": round(p_at_1, 4),
        "p_at_5": round(p_at_5, 4),
        "p_at_10": round(p_at_10, 4),
        "mrr": round(mrr, 4),
        "median_rank": round(median_rank, 1),
        "note": "High P@1 expected due to forward=inverse score identity (Parseval).",
    }


# =========================================================================
# Figures
# =========================================================================
def generate_figures(
    alg_results: dict,
    child_parent_results: dict,
    shared_exclusive_results: dict,
    data: dict,
) -> None:
    """Generate all SVG figures."""
    import matplotlib.pyplot as plt
    import numpy as np
    setup_matplotlib()

    v_left = data["v_left"]
    v_left_inv = compute_inverse_relation(v_left, S_DIM)

    # --- Figure 1: Score Identity Scatter ---
    fig, ax = plt.subplots(figsize=(4, 4))
    fwd = alg_results["fwd_scores"]
    inv = alg_results["inv_scores"]
    ax.scatter(fwd, inv, s=3, alpha=0.4, c="#2196F3", edgecolors="none")
    lims = [min(min(fwd), min(inv)) - 0.5, max(max(fwd), max(inv)) + 0.5]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5, label="y = x")
    ax.set_xlabel(r"Forward score $\mathbf{e}_B^\top \mathrm{circonv}(\mathbf{v}, \mathbf{e}_A)$")
    ax.set_ylabel(r"Inverse score $\mathbf{e}_A^\top \mathrm{circonv}(\mathbf{v}^{-1}, \mathbf{e}_B)$")
    ax.set_title("Score Identity (1000 random pairs)")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    save_svg(fig, OUTPUT_DIR, "sp1_score_identity")
    plt.close(fig)

    # --- Figure 2: Inverse Delta ---
    delta = circular_convolution(v_left, v_left_inv, S_DIM).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(S_DIM), delta, width=1.0, color="#4CAF50", alpha=0.8)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.set_title(r"$\mathrm{circonv}(\mathbf{v}, \mathbf{v}^{-1}) \approx \delta$")
    ax.axhline(y=0, color="gray", lw=0.5)
    # Inset for peak
    ax_inset = fig.add_axes([0.55, 0.45, 0.35, 0.4])
    ax_inset.bar(range(min(20, S_DIM)), delta[:20], width=1.0, color="#4CAF50", alpha=0.8)
    ax_inset.set_title("First 20 indices", fontsize=7)
    ax_inset.tick_params(labelsize=6)
    fig.tight_layout()
    save_svg(fig, OUTPUT_DIR, "sp1_inverse_delta")
    plt.close(fig)

    # --- Figure 3: Child → Parent Demo (horizontal bar chart) ---
    children_data = child_parent_results["selected_children"]
    # Take up to 8 children
    children_to_plot = children_data[:8]
    n_children = len(children_to_plot)
    n_parents_show = 10

    fig, axes = plt.subplots(n_children, 1, figsize=(8, 1.8 * n_children), sharex=False)
    if n_children == 1:
        axes = [axes]

    for ax_i, c_data in enumerate(children_to_plot):
        ax = axes[ax_i]
        parents = c_data["top10_parents"][:n_parents_show]
        labels = [f"NT-{p['parent_idx']} [{p['right_child_label']}]" for p in parents]
        scores = [p["score"] for p in parents]
        colors = ["#FF9800" if c_data["category"] == "function" else "#2196F3"] * len(scores)

        y_pos = range(len(scores))
        ax.barh(y_pos, scores, color=colors, alpha=0.8, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6)
        ax.invert_yaxis()
        child_label = f"{c_data['child_label']} ({', '.join(c_data['child_words'][:3])})"
        ax.set_title(f"Child: {child_label}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Score", fontsize=7)

    fig.suptitle("SP1: Inverse Retrieval — Child → Top Parents", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_svg(fig, OUTPUT_DIR, "sp1_child_to_parent_demo")
    plt.close(fig)

    # --- Figure 4: Shared vs Exclusive P@k ---
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ks = [1, 5, 10]
    shared = shared_exclusive_results["shared_metrics"]
    exclusive = shared_exclusive_results["exclusive_metrics"]

    shared_vals = [shared["p_at_1"], shared["p_at_5"], shared["p_at_10"]]
    exclusive_vals = [exclusive["p_at_1"], exclusive["p_at_5"], exclusive["p_at_10"]]

    x = np.arange(len(ks))
    width = 0.3
    ax.bar(x - width / 2, shared_vals, width, label=f"Shared (n={shared['n']})", color="#FF9800", alpha=0.8)
    ax.bar(x + width / 2, exclusive_vals, width, label=f"Exclusive (n={exclusive['n']})", color="#2196F3", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P@{k}" for k in ks])
    ax.set_ylabel("Precision")
    ax.set_title("Shared vs Exclusive Children: Inverse Retrieval P@k")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Add MRR annotation
    ax.text(0.98, 0.02,
            f"MRR: shared={shared['mrr']:.3f}, exclusive={exclusive['mrr']:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    save_svg(fig, OUTPUT_DIR, "sp1_shared_vs_exclusive")
    plt.close(fig)


# =========================================================================
# Text Report
# =========================================================================
def write_text_report(
    alg_results: dict,
    child_parent_results: dict,
    shared_exclusive_results: dict,
    pk_results: dict,
) -> None:
    """Write qualitative text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("SP1: Inverse Retrieval Verification Report")
    lines.append("=" * 70)

    lines.append("\n--- Algebraic Verification ---")
    lines.append(f"|FFT(v_inv)[k]| max deviation from 1: {alg_results['v_inv_mag_max_deviation']:.2e}")
    lines.append(f"circonv(v, v⁻¹) peak: {alg_results['delta_peak']:.8f}")
    lines.append(f"circonv(v, v⁻¹) off-peak max: {alg_results['delta_off_peak_max']:.2e}")
    lines.append(f"Forward = Inverse score max diff: {alg_results['score_identity_max_diff']:.2e}")
    lines.append(f"Score identity check: {'PASSED' if alg_results['score_identity_passed'] else 'FAILED'}")

    lines.append("\n--- Child → Parent Cluster Demo ---")
    for c in child_parent_results["selected_children"]:
        lines.append(f"\nChild: {c['child_label']} (words: {', '.join(c['child_words'][:5])}) "
                      f"[{c['category']}, {c['parent_count']} parents]")
        for rank, p in enumerate(c["top10_parents"]):
            lines.append(f"  #{rank+1}: NT-{p['parent_idx']:<5d} score={p['score']:.4f}  "
                          f"right_child={p['right_child_label']} ({', '.join(p['right_child_words'][:3])})")

    lines.append("\n\n--- Linguistic Interpretation ---")
    for c in child_parent_results["selected_children"]:
        if c["category"] == "function":
            right_labels = [p["right_child_label"] for p in c["top10_parents"][:5]]
            lines.append(f"\n{c['child_label']} ({', '.join(c['child_words'][:3])}):")
            lines.append(f"  Top-5 right siblings: {right_labels}")
            lines.append(f"  Interpretation: Function word '{c['child_words'][0]}' combines with "
                          f"various right children to form phrasal categories.")

    lines.append("\n\n--- Shared vs Exclusive Child Analysis ---")
    se = shared_exclusive_results
    lines.append(f"Shared children (>={se['shared_threshold']} parents): n={se['n_shared']}")
    lines.append(f"  P@1={se['shared_metrics']['p_at_1']:.4f}, P@5={se['shared_metrics']['p_at_5']:.4f}, "
                  f"MRR={se['shared_metrics']['mrr']:.4f}, Median rank={se['shared_metrics']['median_rank']:.0f}")
    lines.append(f"Exclusive children (<={se['exclusive_threshold']} parents): n={se['n_exclusive']}")
    lines.append(f"  P@1={se['exclusive_metrics']['p_at_1']:.4f}, P@5={se['exclusive_metrics']['p_at_5']:.4f}, "
                  f"MRR={se['exclusive_metrics']['mrr']:.4f}, Median rank={se['exclusive_metrics']['median_rank']:.0f}")

    lines.append(f"\n\n--- Supplementary P@k (all 4096 parents) ---")
    lines.append(f"P@1={pk_results['p_at_1']:.4f}, P@5={pk_results['p_at_5']:.4f}, "
                  f"MRR={pk_results['mrr']:.4f}, Median rank={pk_results['median_rank']:.0f}")
    lines.append(f"Note: {pk_results['note']}")

    report_path = OUTPUT_DIR / "sp1_qualitative.txt"
    report_path.write_text("\n".join(lines))
    print(f"\n  Saved text report: {report_path}")


# =========================================================================
# Main
# =========================================================================
def main() -> None:
    print("SP1: Inverse Retrieval Verification")
    print("=" * 70)

    # Load checkpoint
    print("\n  Loading checkpoint...")
    data = load_checkpoint(DEFAULT_CHECKPOINT, DEVICE)
    print(f"  tau = {data['tau']:.4f}")

    # Build vocabulary and get preterminal labels
    print("\n  Building vocabulary...")
    vocab = build_vocab()
    print(f"  Vocab size: {len(vocab)}")

    print("\n  Computing preterminal top words...")
    t_top_words = get_preterminal_top_words(data["ckpt"], data["all_emb"], vocab)
    print(f"  Labeled {len(t_top_words)} preterminals")

    # Pillar 1: Algebraic verification
    alg_results = verify_algebraic_properties(data)

    # Pillar 2: Child → Parent Cluster
    child_parent_results = analyze_child_to_parent(data, t_top_words)

    # Pillar 3: Shared vs Exclusive
    shared_exclusive_results = analyze_shared_vs_exclusive(data, t_top_words)

    # Supplementary P@k
    pk_results = compute_pk_metrics(data)

    # Generate figures
    print("\n  Generating figures...")
    generate_figures(alg_results, child_parent_results, shared_exclusive_results, data)

    # Write text report
    write_text_report(alg_results, child_parent_results, shared_exclusive_results, pk_results)

    # Save JSON
    json_results = {
        "algebraic": {k: v for k, v in alg_results.items()
                      if k not in ("fwd_scores", "inv_scores")},
        "child_to_parent": {
            "child_counts_top20": child_parent_results["child_counts_top20"],
            "selected_children": child_parent_results["selected_children"],
        },
        "shared_vs_exclusive": shared_exclusive_results,
        "pk_metrics": pk_results,
    }
    json_path = OUTPUT_DIR / "sp1_results.json"
    json_path.write_text(json.dumps(json_results, indent=2, ensure_ascii=False))
    print(f"\n  Saved JSON results: {json_path}")

    print("\n" + "=" * 70)
    print("  SP1 Verification Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
