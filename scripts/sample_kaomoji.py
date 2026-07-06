"""
Sample novel kaomoji from a trained HN-PCFG checkpoint.

The model defines a probabilistic context-free grammar with input-independent
binary rules (`root`, `left_m/p`, `right_m/p`) and unary T -> vocab
distributions. We sample a parse tree top-down:

  1. Sample a root NT index r ~ p(NT_root).
  2. For each NT cell, sample its left child and right child independently
     from p(child | parent), where the child can be either an NT (recurse)
     or a pre-terminal T (terminate).
  3. For each pre-terminal cell, sample one vocab token from p(vocab | T).

Concatenating the leaves (with the preprocessing-time escapes reversed:
`-LRB-` -> "(", `-RRB-` -> ")", `_` -> " ") yields a kaomoji-like string.

Truncation safeguards:
  - reject samples whose leaf count exceeds `max_leaves`
  - reject samples whose recursion depth exceeds `max_depth`
  - retry up to `max_attempts` times to gather `num_samples` raw samples

Novelty filter:
  - the generated string must not occur in any of the train/val/test
    raw-kaomoji files (data/raw/kaomoji-{train,val,test}.txt).

Usage:
  python scripts/sample_kaomoji.py \\
      --conf  config/formal/hnpcfg_kaomoji_main.yaml \\
      --ckpt  log/hnpcfg_kaomoji_main_seed0/HNPCFG*/best.pt \\
      --num-samples 100 \\
      --out log/analysis/kaomoji_novel_samples.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import yaml
from easydict import EasyDict as edict


_UNESCAPE = {"-LRB-": "(", "-RRB-": ")", "_": " "}


def unescape(tok: str) -> str:
    return _UNESCAPE.get(tok, tok)


@torch.no_grad()
def get_full_distributions(model):
    """Return root_logp (NT,), left/right (NT+T, NT), unary (T, V) log probs.

    Reproduces the relevant pieces of HN_PCFG.forward without going through
    a data batch, so the rules are batch-independent.
    """
    NT = model.NT
    nonterm_emb = model.rule_state_emb[:NT]
    term_emb = model.rule_state_emb[NT:]

    # Root: p(NT_root)
    root_logits = model.root_emb @ nonterm_emb.t()
    root_logp = (root_logits * model.log_tau_root.exp()).log_softmax(-1)

    # Unary: term_logits[T, V] (note the `.t()` in forward()).
    term_logits = model._hol_scores(
        model.v_term, term_emb, model.vocab_emb.T, model.log_tau_term
    ).t()
    unary_logp = term_logits.log_softmax(-1)  # (T, V)

    # Binary rules: scores[child=NT+T, parent=NT].
    left = model._hol_scores(
        model.v_left, nonterm_emb, model.rule_state_emb, model.log_tau_rule
    )
    right = model._hol_scores(
        model.v_right, nonterm_emb, model.rule_state_emb, model.log_tau_rule
    )
    left_logp = left.log_softmax(dim=-2)   # (NT+T, NT)
    right_logp = right.log_softmax(dim=-2)

    return root_logp, left_logp, right_logp, unary_logp


def sample_one(root_logp, left_logp, right_logp, unary_logp,
               idx2word, NT, max_leaves, max_depth):
    """Sample one tree and return its list of leaf tokens, or None on failure."""
    root_idx = torch.multinomial(root_logp.exp(), 1).item()

    leaves: list[str] = []
    # stack item: (kind, idx, depth). kind in {"NT", "T"}.
    stack: list[tuple[str, int, int]] = [("NT", root_idx, 0)]
    while stack:
        kind, idx, depth = stack.pop()
        if depth > max_depth:
            return None
        if kind == "NT":
            left_dist = left_logp[:, idx].exp()
            right_dist = right_logp[:, idx].exp()
            left_idx = torch.multinomial(left_dist, 1).item()
            right_idx = torch.multinomial(right_dist, 1).item()
            # push right first, then left, so left is expanded first (DFS, left-to-right)
            for child_idx in (right_idx, left_idx):
                if child_idx < NT:
                    stack.append(("NT", child_idx, depth + 1))
                else:
                    stack.append(("T", child_idx - NT, depth + 1))
        else:  # T leaf -> sample one vocab token
            v_dist = unary_logp[idx].exp()
            v_idx = torch.multinomial(v_dist, 1).item()
            tok = idx2word(v_idx)
            if tok is None:
                return None
            leaves.append(tok)
            if len(leaves) > max_leaves:
                return None
    return leaves


def leaves_to_string(leaves: list[str]) -> str:
    return "".join(unescape(t) for t in leaves)


def load_existing(raw_dir: Path) -> set[str]:
    seen: set[str] = set()
    for split in ("train", "val", "test"):
        p = raw_dir / f"kaomoji-{split}.txt"
        if not p.exists():
            print(f"[warn] {p} missing", file=sys.stderr)
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                seen.add(line.rstrip("\n"))
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=100,
                    help="number of novel samples to collect")
    ap.add_argument("--max-leaves", type=int, default=40)
    ap.add_argument("--max-depth", type=int, default=50)
    ap.add_argument("--max-attempts", type=int, default=10000,
                    help="upper bound on sampling attempts")
    ap.add_argument("--min-leaves", type=int, default=2)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    with args.conf.open() as f:
        cfg = edict(yaml.load(f, Loader=yaml.Loader))
    cfg.device = args.device if torch.cuda.is_available() else "cpu"

    from parser.helper.data_module import DataModule
    from parser.helper.util import get_model

    dataset = DataModule(cfg)
    model = get_model(cfg.model, dataset)
    state = torch.load(str(args.ckpt), map_location=cfg.device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(cfg.device).eval()

    word_vocab = dataset.word_vocab
    pad_idx = word_vocab.padding_idx
    unk_idx = word_vocab.unknown_idx

    def idx2word(i: int) -> str | None:
        if i == pad_idx or i == unk_idx:
            return None
        try:
            return word_vocab.to_word(i)
        except Exception:
            return None

    root_logp, left_logp, right_logp, unary_logp = get_full_distributions(model)

    existing = load_existing(args.raw_dir)
    print(f"[info] loaded {len(existing)} existing kaomoji strings from "
          f"{args.raw_dir}/kaomoji-(train|val|test).txt", file=sys.stderr)

    novel: list[tuple[str, list[str]]] = []  # (string, leaves)
    seen_novel: set[str] = set()
    n_attempts = 0
    n_truncated = 0
    n_in_dataset = 0
    n_dup_novel = 0
    while len(novel) < args.num_samples and n_attempts < args.max_attempts:
        n_attempts += 1
        leaves = sample_one(root_logp, left_logp, right_logp, unary_logp,
                            idx2word, model.NT, args.max_leaves, args.max_depth)
        if leaves is None or len(leaves) < args.min_leaves:
            n_truncated += 1
            continue
        s = leaves_to_string(leaves)
        if s in existing:
            n_in_dataset += 1
            continue
        if s in seen_novel:
            n_dup_novel += 1
            continue
        seen_novel.add(s)
        novel.append((s, leaves))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for s, _ in novel:
            f.write(s + "\n")

    print(f"[info] attempted {n_attempts} samples", file=sys.stderr)
    print(f"[info]   - truncated/too-short: {n_truncated}", file=sys.stderr)
    print(f"[info]   - in existing train/val/test: {n_in_dataset}", file=sys.stderr)
    print(f"[info]   - duplicates within novel set: {n_dup_novel}", file=sys.stderr)
    print(f"[info] wrote {len(novel)} novel kaomoji to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
