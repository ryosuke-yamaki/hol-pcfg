"""Assign PTB phrase / POS labels to HN-PCFG NT and PT indices.

For each gold span (s, e, label) in the eval split:
- width >= 2: argmax over span_marginals[b, s, e, :NT] gives the NT index
  to which `label` is voted.
- width == 1: argmax over unary[b, l, :T] gives the PT index to which the
  POS tag at position l is voted.

The argmax label per index (after support filter) is dumped to JSON.

This is the hol-pcfg variant: it consumes the `data/clean/<lang>-{val,test}.pickle`
format (raw `word` / `pos` / `gold_tree` fields) and loads raw state-dict `.pt`
checkpoints saved by `parser/cmds/train.py`. Word IDs are produced with the same
fastNLP vocabulary the training pipeline builds (parser/helper/data_module.py:
clean_word + Vocabulary(max_size=vocab_size).from_dataset over the train split),
so they stay aligned with the model's `vocab_emb`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from easydict import EasyDict as edict

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from parser.model.HN_PCFG import HN_PCFG  # noqa: E402


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_label(label: str | None) -> str | None:
    """Collapse multilingual phrase / POS variants onto their base category.

    - Strips SPMRL morphological encoding (everything from `##` onward).
    - Strips functional / case suffixes after the first `-`,
      e.g. `NP-SBJ` -> `NP`, `KON-CD` -> `KON`.
    """
    if not label:
        return None
    base = label.split("##", 1)[0].split("-", 1)[0]
    return base or None


class _StubDataset:
    """Minimal stand-in so HN_PCFG.__init__ can resolve `len(dataset.V)` and
    `dataset.device` without importing the full data pipeline."""

    def __init__(self, vocab_size: int, device: torch.device) -> None:
        self.V = list(range(vocab_size))
        self.device = device


def load_model(
    ckpt_path: str, NT: int, T: int, s_dim: int, device: torch.device
) -> HN_PCFG:
    """Load a raw state_dict (.pt) into a freshly-constructed HN_PCFG."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if any(k.startswith("model.") for k in sd.keys()):
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    vocab_size = sd["vocab_emb"].shape[1]

    args = edict(
        NT=NT, T=T, s_dim=s_dim,
        tau_root_init=1.0, tau_term_init=1.0, tau_rule_init=1.0,
        scoring_fn="hole", complex_normalization=True,
        learnable_temperature=True,
    )
    stub = _StubDataset(vocab_size=vocab_size, device=device)
    model = HN_PCFG(args, stub)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        # learnable taus and projection metadata can legitimately differ;
        # surface anything else as a warning rather than failing.
        print(f"[warn] state_dict mismatch  missing={missing}  unexpected={unexpected}")
    model.eval()
    model.to(device)
    return model


def _clean_word_fn(lowercase: bool = True, collapse_numbers: bool = True):
    """Replicate parser/helper/data_module.py:clean_word (lowercase + number
    collapse) so the vocabulary and word indexing match training exactly."""
    import re

    def process(w: str) -> str:
        if lowercase:
            w = w.lower()
        if collapse_numbers:
            w = re.sub(r"[0-9]{1,}([,.]?[0-9]*)*", "N", w)
        return w

    return lambda words: [process(w) for w in words]


def build_train_vocab(train_pickle: str, vocab_size: int):
    """Build the fastNLP Vocabulary exactly as parser/helper/data_module.py does
    (from the cleaned train `word` field) so word->id mapping matches the model.

    Returns (vocab, clean_fn).
    """
    from fastNLP.core.dataset import DataSet
    from fastNLP.core.vocabulary import Vocabulary

    clean = _clean_word_fn()
    train_data = pickle.load(open(train_pickle, "rb"))
    ds = DataSet()
    ds.add_field("word", train_data["word"])
    ds.apply_field(clean, "word", "word")
    vocab = Vocabulary(max_size=vocab_size)
    vocab.from_dataset(ds, field_name="word")
    return vocab, clean


def collate(
    batch_ids: list[list[int]], batch_lens: list[int], pad_id: int = 0
) -> torch.Tensor:
    L = max(batch_lens)
    out = torch.full((len(batch_ids), L), pad_id, dtype=torch.long)
    for b, ids in enumerate(batch_ids):
        out[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


def aggregate_labels(
    counter: dict[int, Counter], dim: int, min_support: int
) -> tuple[list[str | None], list[int]]:
    labels: list[str | None] = [None] * dim
    support: list[int] = [0] * dim
    for idx, c in counter.items():
        if not c:
            continue
        label, count = c.most_common(1)[0]
        if count >= min_support:
            labels[idx] = label
            support[idx] = count
        else:
            support[idx] = count
    return labels, support


def print_top_labels(
    labels: list[str | None], support: list[int], name: str, k: int = 20
) -> None:
    by_label: Counter = Counter()
    for lab, sup in zip(labels, support):
        if lab is not None and sup > 0:
            by_label[lab] += 1
    print(f"[{name}] top-{k} label -> #indices assigned to that label")
    for lab, n_idx in by_label.most_common(k):
        print(f"  {lab:>8s}: {n_idx}")


@torch.enable_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="HN-PCFG .pt state_dict")
    ap.add_argument(
        "--data_pickle", required=True,
        help="data/clean/<lang>-{val,test}.pickle (raw word/pos/gold_tree format)",
    )
    ap.add_argument(
        "--train_pickle", default=None,
        help="train split used to build the vocab (default: --data_pickle with the "
             "split replaced by 'train', e.g. english-val.pickle -> english-train.pickle)",
    )
    ap.add_argument(
        "--vocab_size", type=int, default=10000,
        help="Vocabulary max_size; must match the config used for training",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="english")
    ap.add_argument("--NT", type=int, default=4096)
    ap.add_argument("--T", type=int, default=8192)
    ap.add_argument("--s_dim", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_sentences", type=int, default=-1)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_support", type=int, default=5)
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)
    t0 = time.time()

    model = load_model(args.ckpt, args.NT, args.T, args.s_dim, device)
    print(
        f"[load] ckpt={args.ckpt}  NT={args.NT}  T={args.T}  "
        f"s_dim={args.s_dim}  device={device}  dt={time.time() - t0:.1f}s"
    )

    # Build the vocab the same way training does, then index this split's words.
    train_pickle = args.train_pickle
    if train_pickle is None:
        name = Path(args.data_pickle).name
        for split in ("-val.", "-test."):
            if split in name:
                name = name.replace(split, "-train.")
                break
        train_pickle = str(Path(args.data_pickle).with_name(name))
    vocab, clean = build_train_vocab(train_pickle, args.vocab_size)
    model_vocab = model.vocab_emb.shape[1]
    print(f"[vocab] built from {train_pickle}  size={len(vocab)}  model_vocab_emb={model_vocab}")
    if len(vocab) != model_vocab:
        print(
            "[warn] vocab size != model vocab_emb size; word IDs may be misaligned "
            "with the checkpoint -- check --vocab_size / --train_pickle."
        )

    with open(args.data_pickle, "rb") as f:
        D = pickle.load(f)
    words_raw = D["word"]
    pos_raw = D["pos"]
    trees_raw = D["gold_tree"]

    # `.pickle` stores bare word sequences (no BOS/EOS), matching what
    # data_module feeds the model; index them with the training vocab.
    ids_per_sent: list[list[int]] = []
    pos_per_sent: list[list[str]] = []
    spans_per_sent: list[list[tuple[int, int, str]]] = []
    for i in range(len(words_raw)):
        cleaned = clean(list(words_raw[i]))
        ids_per_sent.append([vocab.to_index(w) for w in cleaned])
        pos_per_sent.append(list(pos_raw[i]))
        spans_per_sent.append([(int(s), int(e), lab) for s, e, lab in trees_raw[i]])

    n_total = len(ids_per_sent)
    keep_idx = [i for i, ids in enumerate(ids_per_sent) if len(ids) >= 2]
    n_dropped = n_total - len(keep_idx)
    if args.max_sentences >= 0:
        keep_idx = keep_idx[: args.max_sentences]
    n_use = len(keep_idx)
    print(
        f"[data] n_sentences={n_total}  using={n_use}  "
        f"dropped_short={n_dropped}"
    )

    counter_nt: dict[int, Counter] = {}
    counter_pt: dict[int, Counter] = {}

    n_processed = 0
    n_gold_spans_nt = 0
    n_gold_spans_pt = 0
    for batch_start in range(0, n_use, args.batch_size):
        batch_end = min(batch_start + args.batch_size, n_use)
        batch_idx = [keep_idx[i] for i in range(batch_start, batch_end)]
        batch_ids = [ids_per_sent[i] for i in batch_idx]
        batch_lens = [len(ids) for ids in batch_ids]
        x = collate(batch_ids, batch_lens).to(device)
        lens = torch.tensor(batch_lens, dtype=torch.long, device=device)

        rules = model(input={"word": x})
        out = model.pcfg._inside(rules, lens, span_dist=True)
        span_marginals = out["span_marginals"].detach()  # (B, N, N, NT)
        assert span_marginals.shape[-1] == args.NT
        unary = rules["unary"].detach()  # (B, L, T)
        assert unary.shape[-1] == args.T

        nt_argmax = span_marginals.argmax(dim=-1).cpu().numpy()  # (B, N, N)
        # `unary` is log P(word_l | t), not the leaf posterior P(t | leaf).
        # Taking argmax_t gives the preterminal most prone to emit that word,
        # which is a first-order approximation of the MAP preterminal.
        pt_argmax = unary.argmax(dim=-1).cpu().numpy()  # (B, L)

        for bi, sid in enumerate(batch_idx):
            seqlen = batch_lens[bi]
            pos_seq = pos_per_sent[sid]
            for li in range(min(seqlen, len(pos_seq))):
                pt_idx = int(pt_argmax[bi, li])
                tag = normalize_label(pos_seq[li])
                if tag is None:
                    continue
                counter_pt.setdefault(pt_idx, Counter())[tag] += 1
                n_gold_spans_pt += 1
            for span in spans_per_sent[sid]:
                s, e, raw_label = span
                if e - s < 2 or e > seqlen:
                    continue
                label = normalize_label(raw_label)
                if label is None:
                    continue
                nt_idx = int(nt_argmax[bi, s, e])
                counter_nt.setdefault(nt_idx, Counter())[label] += 1
                n_gold_spans_nt += 1

        n_processed += len(batch_idx)
        if batch_start // args.batch_size % 25 == 0:
            print(
                f"[batch] {n_processed}/{n_use}  "
                f"nt_spans={n_gold_spans_nt}  pt_spans={n_gold_spans_pt}  "
                f"dt={time.time() - t0:.1f}s"
            )

    nt_label, nt_support = aggregate_labels(counter_nt, args.NT, args.min_support)
    pt_label, pt_support = aggregate_labels(counter_pt, args.T, args.min_support)
    n_assigned_nt = sum(lab is not None for lab in nt_label)
    n_assigned_pt = sum(lab is not None for lab in pt_label)
    print(
        f"[assign] NT assigned={n_assigned_nt}/{args.NT}  "
        f"PT assigned={n_assigned_pt}/{args.T}"
    )
    print_top_labels(nt_label, nt_support, "NT")
    print_top_labels(pt_label, pt_support, "PT")

    ckpt_id = Path(args.ckpt).parent.name
    payload: dict[str, Any] = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "ckpt_id": ckpt_id,
        "lang": args.lang,
        "split": Path(args.data_pickle).stem.split("-")[-1],
        "n_sentences_used": n_processed,
        "NT": args.NT,
        "T": args.T,
        "min_support": args.min_support,
        "nt_label": nt_label,
        "nt_support": nt_support,
        "pt_label": pt_label,
        "pt_support": pt_support,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"[done] wrote {out_path}  total={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
