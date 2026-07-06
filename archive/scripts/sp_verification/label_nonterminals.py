"""Label nonterminals and preterminals of an HN-PCFG checkpoint.

Produces `results/sp3/nt_labels.pkl` consumed by downstream analysis agents.
"""

from __future__ import annotations

import math
import os
import pickle
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CKPT_PATH = "log/hn_pcfg_allproj_cnorm_tau/HNPCFG2026-03-31-12_43_40/best.pt"
TRAIN_DATA_PATH = "data/ptb-train.pickle"
OUTPUT_PATH = "results/sp3/nt_labels.pkl"

NT = 4096
T = 8192
S_DIM = 512
VOCAB_SIZE = 10000  # max_size passed to Vocabulary (actual size = 10002 with pad/unk)

# Thresholds for "active" NT detection
ROOT_THRESHOLD = 1e-4
# child_activity = sum_over_parents P(child|parent).  With NT=4096 parents,
# median ≈ 0.5, so use 0.1 (above bottom quartile) as a generous cutoff.
CHILD_ACTIVITY_THRESHOLD = 0.1

# Chunk size for large matmuls (NT+T, NT) to avoid OOM
CHUNK_SIZE = 1024

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
# Vocabulary reconstruction
# ---------------------------------------------------------------------------
def build_vocab() -> list[str]:
    """Reconstruct the vocabulary exactly as data_module.py does."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from fastNLP.core.dataset import DataSet
    from fastNLP.core.vocabulary import Vocabulary

    print("[vocab] Loading training data...")
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
    print(f"[vocab] Built vocabulary: {len(vocab)} words")
    return vocab


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint() -> dict[str, torch.Tensor]:
    print("[ckpt] Loading checkpoint...")
    ckpt = torch.load(PROJECT_ROOT / CKPT_PATH, map_location="cpu")
    return ckpt


# ---------------------------------------------------------------------------
# Build term_mlp from checkpoint weights
# ---------------------------------------------------------------------------
def build_term_mlp(ckpt: dict[str, torch.Tensor]) -> nn.Sequential:
    """Reconstruct term_mlp: Linear -> ResLayer -> ResLayer -> ResLayer."""
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


# ---------------------------------------------------------------------------
# POS-label heuristic
# ---------------------------------------------------------------------------
# Map dominant words to PTB POS-like labels
_LABEL_RULES: list[tuple[set[str], str]] = [
    ({"the", "a", "an", "this", "that", "these", "those", "his", "her",
      "its", "my", "your", "our", "their", "no", "some", "any", "each",
      "every", "all"}, "DT-like"),
    ({"of", "in", "for", "on", "to", "with", "at", "from", "by", "about",
      "into", "through", "during", "before", "after", "between", "under",
      "over", "against", "among"}, "IN-like"),
    ({"is", "was", "are", "were", "be", "been", "being", "am"}, "VBZ/VBD-like"),
    ({"has", "have", "had"}, "VB-aux-like"),
    ({"will", "would", "could", "should", "may", "might", "can", "shall",
      "must"}, "MD-like"),
    ({"said", "says"}, "VBD-said-like"),
    ({"not", "n't"}, "RB-neg-like"),
    ({"and", "or", "but", "nor"}, "CC-like"),
    ({",", ".", ":", ";", "!", "?", "...", "--", "``", "''"}, "PUNCT-like"),
    ({"-lrb-", "-rrb-", "-lcb-", "-rcb-"}, "BRACKET-like"),
    ({"$", "#", "%"}, "SYM-like"),
    ({"'s"}, "POS-like"),
    ({"that", "which", "who", "whom", "whose", "where", "when"}, "WH-like"),
    ({"do", "does", "did"}, "VB-do-like"),
    ({"mr.", "mrs.", "dr.", "ms.", "prof.", "rep.", "sen.", "gov."}, "NNP-title-like"),
]


def infer_t_label(top_words: list[str], entropy: float) -> str:
    """Heuristic POS-like label for a preterminal based on its top words."""
    top_set = set(top_words[:5])

    # Check rule-based patterns first
    for keyword_set, label in _LABEL_RULES:
        if len(top_set & keyword_set) >= 2:
            return label

    # Single-word dominated checks
    top1 = top_words[0] if top_words else ""
    if top1 in {",", ".", ":", ";", "!", "?", "--", "``", "''", "..."}:
        return f"PUNCT({top1})"
    if top1 in {"$", "#"}:
        return f"SYM({top1})"

    # Low entropy = very specific
    if entropy < 1.0:
        return f"LEX({top1})"

    # Number-dominated
    if "n" in top_set and entropy < 3.0:
        return "CD-like"

    # High entropy = likely open-class (noun/verb/adj)
    if entropy > 5.0:
        return "OPEN-class"

    # Medium entropy heuristics
    return "T-misc"


# ---------------------------------------------------------------------------
# NT role inference
# ---------------------------------------------------------------------------
def infer_nt_role(
    left_top5: list[tuple[int, float]],
    right_top5: list[tuple[int, float]],
) -> str:
    """Infer a role description for an NT based on its top children."""
    left_has_t = any(idx >= NT for idx, _ in left_top5[:3])
    right_has_t = any(idx >= NT for idx, _ in right_top5[:3])
    left_has_nt = any(idx < NT for idx, _ in left_top5[:3])
    right_has_nt = any(idx < NT for idx, _ in right_top5[:3])

    if left_has_t and right_has_nt:
        return "head-left(T->NT)"
    if left_has_nt and right_has_t:
        return "head-right(NT->T)"
    if left_has_t and right_has_t:
        return "leaf-pair(T->T)"
    if left_has_nt and right_has_nt:
        return "branching(NT->NT)"
    return "mixed"


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
def main() -> None:
    os.chdir(PROJECT_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Device: {device}")

    vocab = build_vocab()
    ckpt = load_checkpoint()

    # Extract parameters
    rule_state_emb = ckpt["rule_state_emb"].to(device)   # (12288, 512)
    root_emb = ckpt["root_emb"].to(device)               # (1, 512)
    vocab_emb_w = ckpt["vocab_emb"].to(device)            # (512, 10002)
    v_left = ckpt["v_left"].to(device)                    # (512,) — R=1 stored as 1D
    v_right = ckpt["v_right"].to(device)
    log_tau = ckpt["log_tau"].to(device)
    tau = log_tau.exp().item()
    print(f"[main] tau = {tau:.4f}")

    # Ensure v_left/v_right are 1D (R=1 checkpoint may save as (1,512))
    if v_left.dim() == 2:
        v_left = v_left.squeeze(0)
    if v_right.dim() == 2:
        v_right = v_right.squeeze(0)

    term_mlp = build_term_mlp(ckpt).to(device)

    # -----------------------------------------------------------------------
    # 1. Preterminal (T) labeling
    # -----------------------------------------------------------------------
    print("[T-label] Computing term emission probabilities...")
    nt_emb = rule_state_emb[:NT]       # (4096, 512)
    term_emb = rule_state_emb[NT:]     # (8192, 512)

    with torch.no_grad():
        term_logits = (term_mlp(term_emb) + term_emb) @ vocab_emb_w  # (T, V)
        term_prob = term_logits.softmax(dim=-1)                       # (T, V)

    # Entropy: H = -sum p log p
    log_term_prob = (term_prob + 1e-30).log()
    term_entropy = -(term_prob * log_term_prob).sum(dim=-1)  # (T,)

    # Top-5 words per preterminal
    top5_vals, top5_ids = term_prob.topk(5, dim=-1)  # (T, 5)

    print("[T-label] Assigning labels...")
    t_labels: dict[int, dict] = {}
    for t_idx in range(T):
        top_word_ids = top5_ids[t_idx].cpu().tolist()
        top_word_probs = top5_vals[t_idx].cpu().tolist()
        top_words = [vocab[wid] for wid in top_word_ids]
        ent = term_entropy[t_idx].item()
        label = infer_t_label(top_words, ent)
        t_labels[t_idx] = {
            "label": label,
            "top_words": top_words,
            "entropy": ent,
            "word_probs_top5": list(zip(top_words, top_word_probs)),
        }

    # Label distribution summary
    from collections import Counter
    t_label_counts = Counter(v["label"] for v in t_labels.values())
    print(f"[T-label] Label distribution (top 20):")
    for label, cnt in t_label_counts.most_common(20):
        print(f"  {label:25s}  {cnt:5d}")

    # -----------------------------------------------------------------------
    # 3. Root distribution (compute before NT labels since we need it)
    # -----------------------------------------------------------------------
    print("[root] Computing root distribution...")
    with torch.no_grad():
        root_logits = root_emb @ nt_emb.t()  # (1, NT)
        root_probs = root_logits.softmax(dim=-1).squeeze(0)  # (NT,)

    top_root_vals, top_root_ids = root_probs.topk(30)
    print("[root] Top-10 root NTs:")
    for i in range(10):
        print(f"  NT-{top_root_ids[i].item():4d}  P={top_root_vals[i].item():.6f}")

    # -----------------------------------------------------------------------
    # 2. Nonterminal (NT) labeling — HolE scoring
    # -----------------------------------------------------------------------
    print("[NT-label] Computing HolE rule scores (chunked)...")
    all_emb = rule_state_emb  # (NT+T, s_dim)
    total_children = NT + T   # 12288

    # We compute left_probs and right_probs in chunks over the child dimension
    # to avoid allocating a huge (12288, 4096) float32 tensor all at once.
    # Store top-k per parent for the label dict, and also accumulate
    # child activity scores.

    def compute_hole_probs_chunked(
        v: torch.Tensor,
        parent_emb: torch.Tensor,
        child_emb: torch.Tensor,
        tau_val: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute softmax(HolE_scores * tau) over the child dimension.

        Returns:
            probs: (NT+T, NT) float16 probability matrix
            scores_raw: (NT+T, NT) float32 raw tau-scaled scores (for softmax)
        """
        s_dim = v.shape[-1]
        v_f = torch.fft.rfft(v.unsqueeze(0), dim=-1)      # (1, F)
        parent_f = torch.fft.rfft(parent_emb, dim=-1)       # (NT, F)
        # template: (NT, s_dim)
        template = torch.fft.irfft(
            v_f * parent_f, n=s_dim, dim=-1
        )  # (NT, s_dim)

        # Compute raw scores in chunks, then softmax over full child dim
        n_children = child_emb.shape[0]
        n_parents = parent_emb.shape[0]
        scores_all = torch.empty(n_children, n_parents, device=device, dtype=torch.float32)

        for start in range(0, n_children, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_children)
            chunk = child_emb[start:end]  # (chunk, s_dim)
            # (chunk, s_dim) @ (s_dim, NT) -> (chunk, NT)
            scores_all[start:end] = chunk @ template.t() * tau_val

        # Softmax over child dimension (dim=0)
        probs = scores_all.softmax(dim=0)
        return probs.to(torch.float16)

    with torch.no_grad():
        print("[NT-label]   Left probs...")
        left_probs = compute_hole_probs_chunked(v_left, nt_emb, all_emb, tau)
        print("[NT-label]   Right probs...")
        right_probs = compute_hole_probs_chunked(v_right, nt_emb, all_emb, tau)

    print("[NT-label] Extracting top-5 children per NT parent...")
    # left_probs, right_probs: (NT+T, NT) float16
    # Top-5 children for each parent NT (column)
    left_top5_vals, left_top5_ids = left_probs.topk(5, dim=0)   # (5, NT)
    right_top5_vals, right_top5_ids = right_probs.topk(5, dim=0)

    # -----------------------------------------------------------------------
    # 4. Active NTs
    # -----------------------------------------------------------------------
    print("[active] Identifying active NTs...")
    active_as_root = (root_probs > ROOT_THRESHOLD).cpu()

    # Active as child: sum_over_parents P(child|parent, left) + P(child|parent, right).
    # With NT=4096 parents each column sums to 1, so total mass = 4096.
    # Uniform baseline = 4096 / (NT+T) ≈ 0.33.  Median empirically ≈ 0.5.
    left_child_activity = left_probs.float().sum(dim=1)   # (NT+T,)
    right_child_activity = right_probs.float().sum(dim=1)
    nt_left_activity = left_child_activity[:NT].cpu()
    nt_right_activity = right_child_activity[:NT].cpu()
    nt_child_activity = nt_left_activity + nt_right_activity
    active_as_child = nt_child_activity > CHILD_ACTIVITY_THRESHOLD

    active_mask = active_as_root | active_as_child
    active_nt_indices = torch.where(active_mask)[0].tolist()
    print(f"[active] {len(active_nt_indices)} active NTs "
          f"(root: {active_as_root.sum().item()}, child: {active_as_child.sum().item()})")

    # -----------------------------------------------------------------------
    # Build NT labels dict
    # -----------------------------------------------------------------------
    print("[NT-label] Building NT label dict...")

    def _child_label(child_idx: int) -> str:
        if child_idx < NT:
            return f"NT-{child_idx}"
        else:
            t_idx = child_idx - NT
            return f"T-{t_idx}({t_labels[t_idx]['label']})"

    nt_labels: dict[int, dict] = {}
    for nt_idx in range(NT):
        l5 = [(left_top5_ids[k, nt_idx].item(), left_top5_vals[k, nt_idx].item())
               for k in range(5)]
        r5 = [(right_top5_ids[k, nt_idx].item(), right_top5_vals[k, nt_idx].item())
               for k in range(5)]
        role = infer_nt_role(l5, r5)
        rp = root_probs[nt_idx].item()
        is_active = active_mask[nt_idx].item()

        nt_labels[nt_idx] = {
            "left_top5": [(idx, prob, _child_label(idx)) for idx, prob in l5],
            "right_top5": [(idx, prob, _child_label(idx)) for idx, prob in r5],
            "root_prob": rp,
            "is_active": bool(is_active),
            "role_description": role,
        }

    # -----------------------------------------------------------------------
    # 5. Save
    # -----------------------------------------------------------------------
    output_path = PROJECT_ROOT / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "vocab": vocab,
        "t_labels": t_labels,
        "nt_labels": nt_labels,
        "root_probs": root_probs.cpu(),
        "left_probs": left_probs.cpu(),    # (NT+T, NT) float16
        "right_probs": right_probs.cpu(),  # (NT+T, NT) float16
        "active_nt_indices": active_nt_indices,
    }

    print(f"[save] Writing {output_path} ...")
    with open(output_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[save] Done. File size: {file_size_mb:.1f} MB")

    # -----------------------------------------------------------------------
    # Human-readable summary: top-30 active NTs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TOP-30 MOST ACTIVE NTs (by root_prob + child_activity)")
    print("=" * 90)

    # Rank by combined score: root_prob (normalized) + child_activity (normalized)
    rp_cpu = root_probs.cpu()
    combined_score = rp_cpu / rp_cpu.max() + nt_child_activity / nt_child_activity.max()
    top30_indices = combined_score.topk(30).indices.tolist()

    for rank, nt_idx in enumerate(top30_indices, 1):
        info = nt_labels[nt_idx]
        rp = info["root_prob"]
        role = info["role_description"]
        l_children = "  ".join(
            f"{lbl}({p:.3f})" for _, p, lbl in info["left_top5"][:3]
        )
        r_children = "  ".join(
            f"{lbl}({p:.3f})" for _, p, lbl in info["right_top5"][:3]
        )
        print(f"\n#{rank:2d}  NT-{nt_idx:<4d}  root_P={rp:.6f}  role={role}")
        print(f"     Left:  {l_children}")
        print(f"     Right: {r_children}")

    # Additional summary stats
    print("\n" + "=" * 90)
    print("SUMMARY STATISTICS")
    print("=" * 90)
    role_counts = Counter(v["role_description"] for v in nt_labels.values() if v["is_active"])
    print(f"Active NTs: {len(active_nt_indices)} / {NT}")
    print(f"Role distribution among active NTs:")
    for role, cnt in role_counts.most_common():
        print(f"  {role:25s}  {cnt:5d}")

    # Entropy stats for preterminals
    ent_tensor = term_entropy.cpu()
    print(f"\nPreterm entropy: min={ent_tensor.min():.3f}  "
          f"median={ent_tensor.median():.3f}  "
          f"max={ent_tensor.max():.3f}  "
          f"mean={ent_tensor.mean():.3f}")

    print("\n[done] Label data saved to", output_path)


if __name__ == "__main__":
    main()
