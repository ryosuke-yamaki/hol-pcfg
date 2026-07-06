"""SP3 Qualitative Evaluation: Concrete Examples of Relation Extraction.

Produces a linguistically interpretable report showing specific parent-child
pairs that demonstrate systematic vs exceptional rules in a trained HN-PCFG.

Usage:
    python scripts/analyze_sp3_examples.py [--checkpoint PATH] [--device DEVICE]
"""

import argparse
import math
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = (
    "log/hn_pcfg_allproj_cnorm_tau/HNPCFG2026-03-31-12_43_40/best.pt"
)
TRAIN_DATA = "data/ptb-train.pickle"
NT = 4096
T = 8192
S_DIM = 512
VOCAB_MAX_SIZE = 10000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="results/sp3")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# ResLayer (matches parser/modules/res.py)
# ──────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
# Vocabulary reconstruction (mirrors data_module.py pipeline)
# ──────────────────────────────────────────────────────────────────────
def build_vocabulary(train_file: str, max_size: int = VOCAB_MAX_SIZE) -> dict:
    """Build idx->word and word->idx mappings matching the training pipeline.

    Uses the same fastNLP Vocabulary to get identical indexing as training.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fastNLP.core.dataset import DataSet
    from fastNLP.core.vocabulary import Vocabulary

    with open(train_file, "rb") as f:
        data = pickle.load(f)

    dataset = DataSet()
    dataset.add_field("word", data["word"])

    def clean_word(words):
        def clean_number(w):
            return re.sub(r"[0-9]{1,}([,.]?[0-9]*)*", "N", w)
        return [clean_number(word.lower()) for word in words]

    dataset.apply_field(clean_word, "word", "word")

    vocab = Vocabulary(max_size=max_size)
    vocab.from_dataset(dataset, field_name="word")
    vocab.build_vocab()

    idx2word = {}
    for word, idx in vocab.word2idx.items():
        idx2word[idx] = word
    return {"idx2word": idx2word, "word2idx": vocab.word2idx, "vocab_obj": vocab}


# ──────────────────────────────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────────────────────────────
def circular_correlation(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    """Circular correlation: a ⋆ b = IFFT(conj(FFT(a)) * FFT(b))."""
    a_f = torch.fft.rfft(a, dim=-1)
    b_f = torch.fft.rfft(b, dim=-1)
    return torch.fft.irfft(a_f.conj() * b_f, n=n, dim=-1)


def build_term_mlp(ckpt: dict, device: torch.device) -> nn.Sequential:
    """Reconstruct term_mlp from checkpoint weights."""
    term_mlp = nn.Sequential(
        nn.Linear(S_DIM, S_DIM),
        ResLayer(S_DIM, S_DIM),
        ResLayer(S_DIM, S_DIM),
        ResLayer(S_DIM, S_DIM),
    )
    mlp_state = {
        k.replace("term_mlp.", ""): v
        for k, v in ckpt.items()
        if k.startswith("term_mlp.")
    }
    term_mlp.load_state_dict(mlp_state)
    term_mlp.to(device)
    term_mlp.eval()
    return term_mlp


def compute_term_distributions(
    term_mlp: nn.Sequential,
    rule_state_emb: torch.Tensor,
    vocab_emb: torch.Tensor,
) -> torch.Tensor:
    """Compute P(word | preterminal) for all T preterminals.

    Returns:
        (T, V) probability matrix.
    """
    term_emb = rule_state_emb[NT:]  # (T, s_dim)
    with torch.no_grad():
        logits = (term_mlp(term_emb) + term_emb) @ vocab_emb  # (T, V)
    return logits.softmax(dim=-1)


def compute_rule_probs(
    v: torch.Tensor,
    nt_emb: torch.Tensor,
    all_emb: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Compute P(child | parent, relation) via HolE scoring + softmax.

    Returns:
        (NT+T, NT) probability matrix, softmax over children dim.
    """
    v_f = torch.fft.rfft(v.unsqueeze(0), dim=-1)       # (1, F)
    parent_f = torch.fft.rfft(nt_emb, dim=-1)           # (NT, F)
    template = torch.fft.irfft(
        v_f.unsqueeze(1) * parent_f.unsqueeze(0),
        n=S_DIM, dim=-1,
    )  # (1, NT, s_dim)
    scores = torch.einsum("cs, rps -> rcp", all_emb, template)  # (1, C, NT)
    scores = scores.squeeze(0) * tau
    return scores.softmax(dim=0)


# ──────────────────────────────────────────────────────────────────────
# Labeling helpers
# ──────────────────────────────────────────────────────────────────────
CLOSED_CLASS_WORDS = {
    "the", "a", "an", "of", "in", "to", "for", "on", "at", "by", "with",
    "from", "as", "into", "that", "which", "who", "whom", "whose", "this",
    "these", "those", "it", "its", "is", "are", "was", "were", "be", "been",
    "being", "has", "have", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "can", "could", "must",
    "not", "n't", "'s", "'re", "'ve", "'ll", "'d", "'m",
    "and", "or", "but", "if", "than", "when", "while", "because", "although",
    "no", "some", "any", "all", "each", "every", "both", "either", "neither",
    "more", "most", "much", "many", "few", "several",
}

# POS-like category guesses based on dominant words
POS_PATTERNS = {
    "DT": {"the", "a", "an", "this", "that", "these", "those", "each",
            "every", "some", "any", "no", "all", "both", "either", "neither"},
    "IN/TO": {"of", "in", "to", "for", "on", "at", "by", "with", "from",
              "into", "about", "between", "through", "after", "before",
              "during", "under", "over", "against", "without", "as", "than"},
    "CC": {"and", "or", "but", "nor", "yet"},
    "MD": {"will", "would", "can", "could", "may", "might", "must",
            "shall", "should"},
    "AUX": {"is", "are", "was", "were", "be", "been", "being",
            "has", "have", "had", "do", "does", "did"},
    "PRP": {"he", "she", "it", "they", "we", "i", "you", "him", "her",
            "them", "us", "me"},
    "PRP$": {"his", "her", "its", "their", "our", "my", "your"},
    "RB-NEG": {"not", "n't"},
    ",": {","},
    ".": {".", "!", "?"},
    "``": {"``", "''", '"'},
}


def classify_preterminal(top_words: list[str]) -> str:
    """Guess a POS-like category for a preterminal from its top words."""
    word_set = set(top_words[:5])
    for pos, pattern_words in POS_PATTERNS.items():
        overlap = word_set & pattern_words
        if len(overlap) >= 2 or (len(overlap) >= 1 and top_words[0] in pattern_words):
            return pos
    # Heuristic: check if most words share a suffix
    if all(w.endswith("ly") for w in top_words[:3]):
        return "RB"
    if all(w.endswith("ed") for w in top_words[:3]):
        return "VBD/VBN"
    if all(w.endswith("ing") for w in top_words[:3]):
        return "VBG"
    if all(w.endswith("tion") or w.endswith("ment") or w.endswith("ness")
           for w in top_words[:3]):
        return "NN-abstract"
    # More content-word heuristics
    if top_words[0] == "n" or (top_words[0] == "N" and top_words[1] in ("'s", "five", "three", "two")):
        return "CD"
    if top_words[0] in ("mr.", "mrs.", "ms.", "dr.", "rep.", "sen."):
        return "NNP-title"
    if top_words[0] in ("said", "says", "added", "noted"):
        return "VBD-report"
    if top_words[0] in ("%",):
        return "%"
    if top_words[0] in ("million", "billion"):
        return "CD-amount"
    if top_words[0] in ("year", "week", "month", "day"):
        return "NN-time"
    return "?"


def label_preterminal(
    t_idx: int,
    term_probs: torch.Tensor,
    idx2word: dict,
    top_n: int = 5,
) -> tuple[str, list[tuple[str, float]]]:
    """Get word label and POS guess for a preterminal.

    Returns:
        (pos_guess, [(word, prob), ...])
    """
    probs = term_probs[t_idx]
    topk_probs, topk_ids = probs.topk(top_n)
    word_prob_pairs = [
        (idx2word.get(idx.item(), f"<unk:{idx.item()}>"), p.item())
        for idx, p in zip(topk_ids, topk_probs)
    ]
    top_words = [w for w, _ in word_prob_pairs]
    pos = classify_preterminal(top_words)
    return pos, word_prob_pairs


def format_child_label(
    child_idx: int,
    term_probs: torch.Tensor,
    idx2word: dict,
    left_probs: torch.Tensor,
    right_probs: torch.Tensor,
) -> str:
    """Human-readable label for a child (NT or T)."""
    if child_idx >= NT:
        # Preterminal
        t_idx = child_idx - NT
        pos, word_probs = label_preterminal(t_idx, term_probs, idx2word)
        words_str = ", ".join(f"{w}({p:.2f})" for w, p in word_probs[:3])
        return f"T{t_idx} [{pos}] -> {words_str}"
    else:
        # Nonterminal -- describe by its own top children
        return f"NT{child_idx}"


# ──────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # ── Load checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    rule_state_emb = ckpt["rule_state_emb"]  # (NT+T, s_dim)
    v_left = ckpt["v_left"]                  # (s_dim,)
    v_right = ckpt["v_right"]                # (s_dim,)
    log_tau = ckpt["log_tau"]
    tau = log_tau.exp().item()
    root_emb = ckpt["root_emb"]              # (1, s_dim)
    vocab_emb = ckpt["vocab_emb"]            # (s_dim, V)

    nt_emb = rule_state_emb[:NT]
    all_emb = rule_state_emb

    print(f"tau = {tau:.4f}")
    print(f"NT = {NT}, T = {T}, V = {vocab_emb.shape[1]}")

    # ── Build vocabulary ──
    print("\nBuilding vocabulary...")
    vocab_data = build_vocabulary(TRAIN_DATA, VOCAB_MAX_SIZE)
    idx2word = vocab_data["idx2word"]
    print(f"Vocabulary size: {len(idx2word)}")

    # ── Build term_mlp and compute term distributions ──
    print("Computing terminal distributions...")
    term_mlp = build_term_mlp(ckpt, device)
    term_probs = compute_term_distributions(term_mlp, rule_state_emb, vocab_emb)
    print(f"term_probs shape: {term_probs.shape}")

    # ── Compute rule probabilities ──
    print("Computing rule probabilities...")
    left_probs = compute_rule_probs(v_left, nt_emb, all_emb, tau)   # (C, NT)
    right_probs = compute_rule_probs(v_right, nt_emb, all_emb, tau)  # (C, NT)
    print(f"left_probs shape: {left_probs.shape}")

    # ── Compute root distribution ──
    root_logits = root_emb @ nt_emb.t()  # (1, NT)
    root_dist = root_logits.softmax(dim=-1).squeeze(0)  # (NT,)

    # ── For each NT: get top-1 left/right child and compute r_ext ──
    print("Computing r_ext and cosine similarities...")

    # Top-1 left child
    left_top1_prob, left_top1_idx = left_probs.max(dim=0)   # (NT,), (NT,)
    right_top1_prob, right_top1_idx = right_probs.max(dim=0)

    # Top-5 for richer labeling
    left_top5_prob, left_top5_idx = left_probs.topk(5, dim=0)   # (5, NT)
    right_top5_prob, right_top5_idx = right_probs.topk(5, dim=0)
    left_top5_prob = left_top5_prob.t()   # (NT, 5)
    left_top5_idx = left_top5_idx.t()
    right_top5_prob = right_top5_prob.t()
    right_top5_idx = right_top5_idx.t()

    # Extract r_ext for top-1 pairs
    left_child_emb = all_emb[left_top1_idx]    # (NT, s_dim)
    right_child_emb = all_emb[right_top1_idx]  # (NT, s_dim)

    r_ext_left = circular_correlation(nt_emb, left_child_emb, S_DIM)   # (NT, s_dim)
    r_ext_right = circular_correlation(nt_emb, right_child_emb, S_DIM)

    cos_left = F.cosine_similarity(r_ext_left, v_left.unsqueeze(0), dim=-1)    # (NT,)
    cos_right = F.cosine_similarity(r_ext_right, v_right.unsqueeze(0), dim=-1)

    # Cross-relation cosine (left r_ext vs right r_ext for same parent)
    cos_cross = F.cosine_similarity(r_ext_left, r_ext_right, dim=-1)  # (NT,)

    # ── Pre-compute preterminal labels (batch) ──
    print("Labeling preterminals...")
    preterm_labels = {}
    preterm_top_words = {}
    for t_idx in range(T):
        pos, word_probs = label_preterminal(t_idx, term_probs, idx2word)
        preterm_labels[t_idx] = pos
        preterm_top_words[t_idx] = word_probs

    # ── Helper: describe a child index ──
    def describe_child(cidx: int) -> str:
        if cidx >= NT:
            t = cidx - NT
            pos = preterm_labels[t]
            wps = preterm_top_words[t]
            ws = "/".join(w for w, _ in wps[:3])
            return f"T{t} [{pos}] ({ws})"
        else:
            return f"NT{cidx}"

    def describe_child_detail(cidx: int) -> str:
        if cidx >= NT:
            t = cidx - NT
            pos = preterm_labels[t]
            wps = preterm_top_words[t]
            ws = ", ".join(f'"{w}"({p:.3f})' for w, p in wps[:5])
            return f"T{t}  POS=[{pos}]  top-words: {ws}"
        else:
            # Describe NT by its children
            lc = left_top5_idx[cidx]
            rc = right_top5_idx[cidx]
            lc_str = describe_child(lc[0].item())
            rc_str = describe_child(rc[0].item())
            lp = left_top5_prob[cidx, 0].item()
            rp = right_top5_prob[cidx, 0].item()
            return f"NT{cidx}  left={lc_str}(p={lp:.3f})  right={rc_str}(p={rp:.3f})"

    def describe_nt_role(nt_idx: int) -> str:
        """Infer the grammatical role of an NT from its children."""
        lc_idx = left_top1_idx[nt_idx].item()
        rc_idx = right_top1_idx[nt_idx].item()
        lp = left_top1_prob[nt_idx].item()
        rp = right_top1_prob[nt_idx].item()

        parts = []
        for tag, cidx in [("L", lc_idx), ("R", rc_idx)]:
            if cidx >= NT:
                t = cidx - NT
                pos = preterm_labels[t]
                ws = "/".join(w for w, _ in preterm_top_words[t][:2])
                parts.append(f"{tag}={pos}({ws})")
            else:
                parts.append(f"{tag}=NT{cidx}")
        return f"NT{nt_idx} [{' | '.join(parts)}]"

    # ── Interpretation helper ──
    def interpret_high_cos_pair(
        parent_nt: int, child_idx: int, direction: str, cos_val: float, prob: float,
    ) -> str:
        """Generate linguistic interpretation for a high-cosine pair."""
        if child_idx >= NT:
            t = child_idx - NT
            pos = preterm_labels[t]
            tw = [w for w, _ in preterm_top_words[t][:5]]
            top_word = tw[0]

            # Determiner phrase
            if pos == "DT":
                return (f"Determiner phrase: parent systematically selects "
                        f"'{top_word}' as {direction} child. "
                        f"High cos ({cos_val:.3f}) + high P ({prob:.3f}) = "
                        f"productive DP rule (e.g., 'the NP', 'a NP').")
            # Preposition/complementizer
            if pos == "IN/TO":
                if top_word == "to":
                    return (f"Infinitival/PP rule: parent selects 'to' as "
                            f"{direction} child (cos={cos_val:.3f}, P={prob:.3f}). "
                            f"This is a productive rule for to-infinitives or PPs.")
                if top_word in ("of", "in", "for", "on", "at", "by", "with", "from"):
                    return (f"PP rule: parent selects preposition '{top_word}' as "
                            f"{direction} child (cos={cos_val:.3f}, P={prob:.3f}). "
                            f"Productive prepositional phrase structure.")
                return (f"Prepositional/functional selection: '{top_word}' as "
                        f"{direction} child (cos={cos_val:.3f}, P={prob:.3f}).")
            # Coordinator
            if pos == "CC":
                return (f"Coordination rule: parent selects coordinator "
                        f"'{top_word}' (cos={cos_val:.3f}, P={prob:.3f}). "
                        f"Productive conjunction structure.")
            # Modal/auxiliary
            if pos in ("MD", "AUX"):
                return (f"Verbal auxiliary rule: parent selects "
                        f"'{top_word}' (cos={cos_val:.3f}, P={prob:.3f}). "
                        f"Systematic selection of function word in VP.")
            # Negation
            if pos == "RB-NEG":
                return (f"Negation rule: parent selects '{top_word}' "
                        f"(cos={cos_val:.3f}, P={prob:.3f}). "
                        f"Systematic negation attachment.")
            # Content word categories
            if all(w.endswith("ly") for w in tw[:3]):
                return (f"Adverb modifier: parent selects adverb "
                        f"'{top_word}' (cos={cos_val:.3f}, P={prob:.3f}). "
                        f"Content-word modification pattern.")
            # Proper nouns (geographic, person names)
            if top_word in ("new", "south", "east", "west", "north", "san", "los",
                            "wall", "hong"):
                return (f"Proper noun compound: parent selects "
                        f"'{top_word}' in multiword proper noun "
                        f"(cos={cos_val:.3f}, P={prob:.3f}). "
                        f"Lexically fixed compound (e.g., '{top_word} {tw[1]}').")
            if top_word in ("mr.", "mrs.", "ms.", "dr.", "rep.", "sen.", "jon"):
                return (f"Name title: parent selects title '{top_word}' "
                        f"(cos={cos_val:.3f}, P={prob:.3f}). Productive naming rule.")
            # Number
            if top_word == "n" or pos == "CD":
                return (f"Numeral rule: parent selects number "
                        f"(cos={cos_val:.3f}, P={prob:.3f}). "
                        f"Productive numeric expression.")
            # Generic content word
            return (f"Content word selection: '{top_word}' "
                    f"(cos={cos_val:.3f}, P={prob:.3f}). "
                    f"Semantically coherent lexical selection.")
        else:
            return (f"Recursive NT expansion: parent selects NT{child_idx} as "
                    f"{direction} child (cos={cos_val:.3f}, P={prob:.3f}). "
                    f"Productive recursive rule.")

    def interpret_low_cos_pair(
        parent_nt: int, child_idx: int, direction: str,
        cos_val: float, prob: float,
    ) -> str:
        """Generate linguistic interpretation for a low-cosine pair."""
        if prob < 0.01:
            return (f"Near-uniform distribution (P={prob:.4f}): the parent "
                    f"does not strongly prefer any single child. "
                    f"The HolE relation template cannot approximate an "
                    f"idiosyncratic or nearly flat distribution. "
                    f"cos={cos_val:.3f} reflects this uncertainty.")
        if child_idx >= NT:
            t = child_idx - NT
            tw = [w for w, _ in preterm_top_words[t][:5]]
            return (f"Exceptional lexical selection: '{tw[0]}' selected "
                    f"with moderate P={prob:.3f} but low cos={cos_val:.3f}. "
                    f"The parent-child embedding geometry deviates from "
                    f"the learned relation template, suggesting this rule "
                    f"is memorized rather than systematic.")
        return (f"Exceptional NT selection: NT{child_idx} selected "
                f"with P={prob:.3f} but low cos={cos_val:.3f}. "
                f"This recursive expansion is not well-captured by "
                f"the global relation vector.")

    # ══════════════════════════════════════════════════════════════════
    # Build report
    # ══════════════════════════════════════════════════════════════════
    lines = []

    def out(s: str = "") -> None:
        lines.append(s)
        print(s)

    out("=" * 80)
    out("  SP3 Qualitative Analysis: Concrete Examples of Relation Extraction")
    out("  Model: HN-PCFG with freq_cnorm, tau={:.4f}".format(tau))
    out("  Checkpoint: {}".format(args.checkpoint))
    out("=" * 80)

    # ── Overall statistics ──
    out("\n" + "=" * 80)
    out("  OVERALL STATISTICS")
    out("=" * 80)
    out(f"  cos(r_ext, v_left)  -- mean: {cos_left.mean():.4f}, "
        f"std: {cos_left.std():.4f}, median: {cos_left.median():.4f}")
    out(f"  cos(r_ext, v_right) -- mean: {cos_right.mean():.4f}, "
        f"std: {cos_right.std():.4f}, median: {cos_right.median():.4f}")
    out(f"  cos(r_ext_left, r_ext_right) cross -- mean: {cos_cross.mean():.4f}, "
        f"std: {cos_cross.std():.4f}")

    # NT vs T child breakdown
    left_is_t = (left_top1_idx >= NT)
    left_is_nt = ~left_is_t
    right_is_t = (right_top1_idx >= NT)
    right_is_nt = ~right_is_t
    out(f"\n  Left children:  {left_is_nt.sum().item()} NTs, {left_is_t.sum().item()} Ts")
    out(f"  Right children: {right_is_nt.sum().item()} NTs, {right_is_t.sum().item()} Ts")
    out(f"  cos(r_ext, v_left)  for NT children: {cos_left[left_is_nt].mean():.4f} "
        f"(n={left_is_nt.sum().item()})")
    out(f"  cos(r_ext, v_left)  for T children:  {cos_left[left_is_t].mean():.4f} "
        f"(n={left_is_t.sum().item()})")
    out(f"  cos(r_ext, v_right) for NT children: {cos_right[right_is_nt].mean():.4f} "
        f"(n={right_is_nt.sum().item()})")
    out(f"  cos(r_ext, v_right) for T children:  {cos_right[right_is_t].mean():.4f} "
        f"(n={right_is_t.sum().item()})")

    # ══════════════════════════════════════════════════════════════════
    # A. Top-20 HIGHEST cosine (v_left)
    # ══════════════════════════════════════════════════════════════════
    out("\n" + "=" * 80)
    out("  A. TOP-20 HIGHEST cos(r_ext, v_left) — MOST SYSTEMATIC LEFT RULES")
    out("=" * 80)
    sorted_left = cos_left.argsort(descending=True)
    for rank, nt_idx in enumerate(sorted_left[:20].tolist()):
        cos_val = cos_left[nt_idx].item()
        cidx = left_top1_idx[nt_idx].item()
        prob = left_top1_prob[nt_idx].item()
        out(f"\n  #{rank+1}  cos={cos_val:.4f}  P(child|parent)={prob:.4f}")
        out(f"    Parent: {describe_nt_role(nt_idx)}")
        out(f"    Left-child: {describe_child_detail(cidx)}")
        out(f"    >> {interpret_high_cos_pair(nt_idx, cidx, 'left', cos_val, prob)}")

    # ══════════════════════════════════════════════════════════════════
    # A'. Top-20 HIGHEST cosine (v_right)
    # ══════════════════════════════════════════════════════════════════
    out("\n" + "=" * 80)
    out("  A'. TOP-20 HIGHEST cos(r_ext, v_right) — MOST SYSTEMATIC RIGHT RULES")
    out("=" * 80)
    sorted_right = cos_right.argsort(descending=True)
    for rank, nt_idx in enumerate(sorted_right[:20].tolist()):
        cos_val = cos_right[nt_idx].item()
        cidx = right_top1_idx[nt_idx].item()
        prob = right_top1_prob[nt_idx].item()
        out(f"\n  #{rank+1}  cos={cos_val:.4f}  P(child|parent)={prob:.4f}")
        out(f"    Parent: {describe_nt_role(nt_idx)}")
        out(f"    Right-child: {describe_child_detail(cidx)}")
        out(f"    >> {interpret_high_cos_pair(nt_idx, cidx, 'right', cos_val, prob)}")

    # ══════════════════════════════════════════════════════════════════
    # B. Top-20 LOWEST cosine (v_left)
    # ══════════════════════════════════════════════════════════════════
    out("\n" + "=" * 80)
    out("  B. TOP-20 LOWEST cos(r_ext, v_left) — MOST EXCEPTIONAL LEFT RULES")
    out("=" * 80)
    sorted_left_asc = cos_left.argsort(descending=False)
    for rank, nt_idx in enumerate(sorted_left_asc[:20].tolist()):
        cos_val = cos_left[nt_idx].item()
        cidx = left_top1_idx[nt_idx].item()
        prob = left_top1_prob[nt_idx].item()
        out(f"\n  #{rank+1}  cos={cos_val:.4f}  P(child|parent)={prob:.4f}")
        out(f"    Parent: {describe_nt_role(nt_idx)}")
        out(f"    Left-child: {describe_child_detail(cidx)}")
        # Check if the top-1 is actually dominant
        top5p = left_top5_prob[nt_idx].tolist()
        entropy = -(torch.tensor(top5p) * torch.tensor(top5p).log()).sum().item()
        out(f"    Top-5 probs: {[f'{p:.3f}' for p in top5p]}  "
            f"(H={entropy:.3f})")
        out(f"    >> {interpret_low_cos_pair(nt_idx, cidx, 'left', cos_val, prob)}")

    # ══════════════════════════════════════════════════════════════════
    # B'. Top-20 LOWEST cosine (v_right)
    # ══════════════════════════════════════════════════════════════════
    out("\n" + "=" * 80)
    out("  B'. TOP-20 LOWEST cos(r_ext, v_right) — MOST EXCEPTIONAL RIGHT RULES")
    out("=" * 80)
    sorted_right_asc = cos_right.argsort(descending=False)
    for rank, nt_idx in enumerate(sorted_right_asc[:20].tolist()):
        cos_val = cos_right[nt_idx].item()
        cidx = right_top1_idx[nt_idx].item()
        prob = right_top1_prob[nt_idx].item()
        out(f"\n  #{rank+1}  cos={cos_val:.4f}  P(child|parent)={prob:.4f}")
        out(f"    Parent: {describe_nt_role(nt_idx)}")
        out(f"    Right-child: {describe_child_detail(cidx)}")
        top5p = right_top5_prob[nt_idx].tolist()
        entropy = -(torch.tensor(top5p) * torch.tensor(top5p).log()).sum().item()
        out(f"    Top-5 probs: {[f'{p:.3f}' for p in top5p]}  "
            f"(H={entropy:.3f})")
        out(f"    >> {interpret_low_cos_pair(nt_idx, cidx, 'right', cos_val, prob)}")

    # ══════════════════════════════════════════════════════════════════
    # C. Top-10 ROOT nonterminals
    # ══════════════════════════════════════════════════════════════════
    out("\n" + "=" * 80)
    out("  C. TOP-10 ROOT NONTERMINALS")
    out("=" * 80)
    root_topk_probs, root_topk_idx = root_dist.topk(10)
    for rank in range(10):
        nt_idx = root_topk_idx[rank].item()
        rp = root_topk_probs[rank].item()
        out(f"\n  #{rank+1}  P(root)={rp:.4f}")
        out(f"    {describe_nt_role(nt_idx)}")
        # Describe children in more detail
        lc_idx = left_top1_idx[nt_idx].item()
        rc_idx = right_top1_idx[nt_idx].item()
        out(f"    Left:  {describe_child_detail(lc_idx)}")
        out(f"    Right: {describe_child_detail(rc_idx)}")
        out(f"    cos(r_ext_left, v_left)={cos_left[nt_idx]:.4f}  "
            f"cos(r_ext_right, v_right)={cos_right[nt_idx]:.4f}")
        # S-like analysis
        r_desc = describe_child(rc_idx)
        l_desc = describe_child(lc_idx)
        if "NT" in l_desc and "NT" in r_desc:
            out("    >> S-like: both children are NTs (clausal structure)")
        elif any(p in l_desc for p in ["DT", "PRP", "PRP$"]):
            out("    >> Subject-initial: left child is a determiner/pronoun "
                "(nominal subject)")

    # ══════════════════════════════════════════════════════════════════
    # D. Systematic patterns analysis
    # ══════════════════════════════════════════════════════════════════
    out("\n" + "=" * 80)
    out("  D. SYSTEMATIC PATTERNS")
    out("=" * 80)

    # D1: POS distribution of left/right T children
    out("\n  D1. POS-like categories of preterminal children")
    out("  " + "-" * 60)

    left_t_pos = Counter()
    right_t_pos = Counter()
    for nt_idx in range(NT):
        lc = left_top1_idx[nt_idx].item()
        rc = right_top1_idx[nt_idx].item()
        if lc >= NT:
            left_t_pos[preterm_labels[lc - NT]] += 1
        if rc >= NT:
            right_t_pos[preterm_labels[rc - NT]] += 1

    out("\n  Left-child T categories (top-15):")
    for pos, count in left_t_pos.most_common(15):
        out(f"    {pos:>12}: {count:>5} parents")

    out("\n  Right-child T categories (top-15):")
    for pos, count in right_t_pos.most_common(15):
        out(f"    {pos:>12}: {count:>5} parents")

    # D2: Closed-class vs open-class split
    out("\n  D2. Closed-class vs open-class left children")
    out("  " + "-" * 60)
    closed_pos = {"DT", "IN/TO", "CC", "MD", "AUX", "RB-NEG", ",", ".", "``", "PRP", "PRP$"}
    n_closed_left = sum(
        1 for nt_idx in range(NT)
        if left_top1_idx[nt_idx].item() >= NT
        and preterm_labels[left_top1_idx[nt_idx].item() - NT] in closed_pos
    )
    n_open_left = left_is_t.sum().item() - n_closed_left
    out(f"  Closed-class left T children: {n_closed_left}")
    out(f"  Open-class left T children:   {n_open_left}")
    out(f"  NT left children:             {left_is_nt.sum().item()}")

    # Mean cos for each group
    closed_mask = torch.zeros(NT, dtype=torch.bool, device=device)
    open_mask = torch.zeros(NT, dtype=torch.bool, device=device)
    for nt_idx in range(NT):
        lc = left_top1_idx[nt_idx].item()
        if lc >= NT:
            if preterm_labels[lc - NT] in closed_pos:
                closed_mask[nt_idx] = True
            else:
                open_mask[nt_idx] = True

    if closed_mask.any():
        out(f"  Mean cos(r_ext, v_left) for closed-class left children: "
            f"{cos_left[closed_mask].mean():.4f}")
    if open_mask.any():
        out(f"  Mean cos(r_ext, v_left) for open-class left children:   "
            f"{cos_left[open_mask].mean():.4f}")
    if left_is_nt.any():
        out(f"  Mean cos(r_ext, v_left) for NT left children:           "
            f"{cos_left[left_is_nt].mean():.4f}")

    # D3: Parents that exclusively select NT vs T children
    out("\n  D3. NT-exclusive vs T-exclusive parents")
    out("  " + "-" * 60)
    both_nt = (left_top1_idx < NT) & (right_top1_idx < NT)
    both_t = (left_top1_idx >= NT) & (right_top1_idx >= NT)
    left_t_right_nt = (left_top1_idx >= NT) & (right_top1_idx < NT)
    left_nt_right_t = (left_top1_idx < NT) & (right_top1_idx >= NT)
    out(f"  Both children NT: {both_nt.sum().item()}")
    out(f"  Both children T:  {both_t.sum().item()}")
    out(f"  Left=T, Right=NT: {left_t_right_nt.sum().item()}")
    out(f"  Left=NT, Right=T: {left_nt_right_t.sum().item()}")

    # D4: Which preterminals are most "popular" as left children?
    out("\n  D4. Most popular left-child preterminals")
    out("  " + "-" * 60)
    left_preterm_counter = Counter()
    for nt_idx in range(NT):
        lc = left_top1_idx[nt_idx].item()
        if lc >= NT:
            left_preterm_counter[lc - NT] += 1
    out(f"  (Total distinct preterminals used as top-1 left child: "
        f"{len(left_preterm_counter)})")
    for t_idx, count in left_preterm_counter.most_common(15):
        pos = preterm_labels[t_idx]
        ws = "/".join(w for w, _ in preterm_top_words[t_idx][:3])
        out(f"    T{t_idx:>5} [{pos:>8}] ({ws:>30}): "
            f"selected by {count:>4} parents")

    # D5: Most popular right-child preterminals
    out("\n  D5. Most popular right-child preterminals")
    out("  " + "-" * 60)
    right_preterm_counter = Counter()
    for nt_idx in range(NT):
        rc = right_top1_idx[nt_idx].item()
        if rc >= NT:
            right_preterm_counter[rc - NT] += 1
    out(f"  (Total distinct preterminals used as top-1 right child: "
        f"{len(right_preterm_counter)})")
    for t_idx, count in right_preterm_counter.most_common(15):
        pos = preterm_labels[t_idx]
        ws = "/".join(w for w, _ in preterm_top_words[t_idx][:3])
        out(f"    T{t_idx:>5} [{pos:>8}] ({ws:>30}): "
            f"selected by {count:>4} parents")

    # D6: Shared left child analysis (many parents selecting the SAME child)
    out("\n  D6. Productive rules: parents sharing the same left child")
    out("  " + "-" * 60)
    out("  (If many parents select the same preterminal as left child,")
    out("   this indicates a productive phrase-structure pattern.)")
    for t_idx, count in left_preterm_counter.most_common(5):
        pos = preterm_labels[t_idx]
        ws = "/".join(w for w, _ in preterm_top_words[t_idx][:3])
        # Get the parents that select this
        parent_ids = [
            nt for nt in range(NT)
            if left_top1_idx[nt].item() == t_idx + NT
        ]
        cos_vals = cos_left[parent_ids]
        out(f"\n    T{t_idx} [{pos}] ({ws}): {count} parents")
        out(f"      cos(r_ext, v_left) among these parents: "
            f"mean={cos_vals.mean():.4f}, std={cos_vals.std():.4f}")
        # Show some example parents
        sorted_parents = sorted(parent_ids, key=lambda x: cos_left[x].item(),
                                reverse=True)
        for p in sorted_parents[:3]:
            rc = right_top1_idx[p].item()
            out(f"      Parent NT{p}: right={describe_child(rc)}, "
                f"cos={cos_left[p]:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # E. Cross-relation consistency
    # ══════════════════════════════════════════════════════════════════
    out("\n" + "=" * 80)
    out("  E. CROSS-RELATION CONSISTENCY")
    out("=" * 80)
    out(f"\n  cos(r_ext_left, r_ext_right) for the same parent:")
    out(f"    mean={cos_cross.mean():.4f}, std={cos_cross.std():.4f}, "
        f"median={cos_cross.median():.4f}")
    out(f"    min={cos_cross.min():.4f}, max={cos_cross.max():.4f}")

    out(f"\n  cos(v_left, v_right) = "
        f"{F.cosine_similarity(v_left.unsqueeze(0), v_right.unsqueeze(0)).item():.4f}")

    out("\n  Interpretation: If cos(r_ext_left, r_ext_right) is low, the model")
    out("  encodes genuinely different information in left vs right relations.")
    out("  This is expected: left-branching selects specifiers/determiners,")
    out("  while right-branching selects complements/modifiers.")

    # Show examples of most different and most similar
    cross_sorted = cos_cross.argsort()
    out("\n  Parents with MOST DIFFERENT left/right relations (lowest cross-cos):")
    for rank, nt_idx in enumerate(cross_sorted[:5].tolist()):
        lc = left_top1_idx[nt_idx].item()
        rc = right_top1_idx[nt_idx].item()
        out(f"    NT{nt_idx}: cross-cos={cos_cross[nt_idx]:.4f}")
        out(f"      L={describe_child(lc)}")
        out(f"      R={describe_child(rc)}")

    cross_sorted_desc = cos_cross.argsort(descending=True)
    out("\n  Parents with MOST SIMILAR left/right relations (highest cross-cos):")
    for rank, nt_idx in enumerate(cross_sorted_desc[:5].tolist()):
        lc = left_top1_idx[nt_idx].item()
        rc = right_top1_idx[nt_idx].item()
        out(f"    NT{nt_idx}: cross-cos={cos_cross[nt_idx]:.4f}")
        out(f"      L={describe_child(lc)}")
        out(f"      R={describe_child(rc)}")

    # ══════════════════════════════════════════════════════════════════
    # F. Linguistic Interpretation Summary
    # ══════════════════════════════════════════════════════════════════
    out("\n" + "=" * 80)
    out("  F. LINGUISTIC INTERPRETATION SUMMARY")
    out("=" * 80)

    # Compute fraction of high-cosine parents
    high_cos_thresh = 0.5
    n_high_left = (cos_left > high_cos_thresh).sum().item()
    n_high_right = (cos_right > high_cos_thresh).sum().item()
    out(f"\n  Parents with cos(r_ext, v) > {high_cos_thresh}:")
    out(f"    Left:  {n_high_left}/{NT} ({100*n_high_left/NT:.1f}%)")
    out(f"    Right: {n_high_right}/{NT} ({100*n_high_right/NT:.1f}%)")

    out("\n  Key findings:")

    # Find dominant pattern: what POS are high-cos parents selecting?
    high_cos_left_mask = cos_left > high_cos_thresh
    high_cos_pos = Counter()
    for nt_idx in range(NT):
        if high_cos_left_mask[nt_idx]:
            lc = left_top1_idx[nt_idx].item()
            if lc >= NT:
                high_cos_pos[preterm_labels[lc - NT]] += 1
            else:
                high_cos_pos["NT-child"] += 1

    out("\n  POS distribution of left children among HIGH-cosine parents:")
    for pos, cnt in high_cos_pos.most_common(10):
        out(f"    {pos:>12}: {cnt}")

    low_cos_left_mask = cos_left < 0.3
    low_cos_pos = Counter()
    for nt_idx in range(NT):
        if low_cos_left_mask[nt_idx]:
            lc = left_top1_idx[nt_idx].item()
            if lc >= NT:
                low_cos_pos[preterm_labels[lc - NT]] += 1
            else:
                low_cos_pos["NT-child"] += 1

    out("\n  POS distribution of left children among LOW-cosine parents (cos < 0.3):")
    for pos, cnt in low_cos_pos.most_common(10):
        out(f"    {pos:>12}: {cnt}")

    # Probability concentration vs cosine
    out("\n  Correlation between P(top-1 child) and cos(r_ext, v):")
    corr_left = torch.corrcoef(torch.stack([cos_left, left_top1_prob]))[0, 1].item()
    corr_right = torch.corrcoef(torch.stack([cos_right, right_top1_prob]))[0, 1].item()
    out(f"    Left:  r = {corr_left:.4f}")
    out(f"    Right: r = {corr_right:.4f}")
    out("    (Positive correlation means sharper distributions have higher cosine,")
    out("     confirming that the relation better captures concentrated selections.)")

    out("\n" + "=" * 80)
    out("  END OF REPORT")
    out("=" * 80)

    # ── Save report ──
    report_path = output_dir / "sp3_qualitative_analysis.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to: {report_path}")

    # ── Save structured data ──
    save_data = {
        "cos_left": cos_left.cpu(),
        "cos_right": cos_right.cpu(),
        "cos_cross": cos_cross.cpu(),
        "left_top1_idx": left_top1_idx.cpu(),
        "right_top1_idx": right_top1_idx.cpu(),
        "left_top1_prob": left_top1_prob.cpu(),
        "right_top1_prob": right_top1_prob.cpu(),
        "left_top5_idx": left_top5_idx.cpu(),
        "right_top5_idx": right_top5_idx.cpu(),
        "left_top5_prob": left_top5_prob.cpu(),
        "right_top5_prob": right_top5_prob.cpu(),
        "root_dist": root_dist.cpu(),
        "root_topk_idx": root_topk_idx.cpu(),
        "root_topk_probs": root_topk_probs.cpu(),
        "preterm_labels": preterm_labels,
        "preterm_top_words": preterm_top_words,
        "idx2word": idx2word,
        "tau": tau,
    }
    pkl_path = output_dir / "sp3_example_data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"Structured data saved to: {pkl_path}")


if __name__ == "__main__":
    main()
