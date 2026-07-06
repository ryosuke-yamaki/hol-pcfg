"""Run HN-PCFG inference on KTB JA test pickle and export bracketed pred trees
in the same order/filtering as the gold file, so Li et al.'s evaluate.py can
score them directly.
"""

import argparse
import pickle
import sys
import torch
import yaml
import os
from easydict import EasyDict as edict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parser.helper.util import get_model
from parser.helper.data_module import DataModule
from parser.helper.loader_wrapper import DataPrefetcher


def spans_to_bracketed(spans, words, label="S"):
    """Convert a binary-tree's full span list to a bracketed NLTK string.

    `spans` is the output of _cky_zero_order's backtrack: a list of (i, j)
    spans including length-1 spans and (0, N), forming a complete binary tree.
    """
    spans_set = {(int(i), int(j)) for (i, j) in spans}
    # Defensive: add leaves and root in case they were stripped somewhere
    n = len(words)
    for i in range(n):
        spans_set.add((i, i + 1))
    spans_set.add((0, n))

    def build(i, j):
        # escape NLTK reserved chars in word forms
        if j - i == 1:
            w = words[i].replace("(", "-LRB-").replace(")", "-RRB-")
            return f"({label} {w})"
        for k in range(i + 1, j):
            if (i, k) in spans_set and (k, j) in spans_set:
                return f"({label} {build(i, k)} {build(k, j)})"
        # No legal split (shouldn't happen if spans form a binary tree).
        # Fall back to a right-branching chain.
        right = build(i + 1, j) if j - (i + 1) > 0 else ""
        leaf = words[i].replace("(", "-LRB-").replace(")", "-RRB-")
        return f"({label} ({label} {leaf}) {right})"

    return build(0, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_from_dir", required=True,
                        help="dir containing best.pt + config.yaml")
    parser.add_argument("--pred_out", required=True,
                        help="output bracketed pred trees")
    parser.add_argument("--gold_out", required=True,
                        help="output gold trees filtered to match pred order")
    parser.add_argument("--decode_type", default="mbr")
    parser.add_argument("--device", default="0")
    parser.add_argument("--gold_text_file", required=True,
                        help="bracketed-tree gold file aligned with the test pickle "
                             "(one tree per line, same order as the test pickle)")
    args = parser.parse_args()

    cfg_path = Path(args.load_from_dir) / "config.yaml"
    with cfg_path.open() as f:
        cfg = edict(yaml.load(f, Loader=yaml.Loader))
    cfg.update({"conf": str(cfg_path),
                "load_from_dir": args.load_from_dir,
                "device": "cuda" if torch.cuda.is_available() else "cpu"})
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    dataset = DataModule(cfg)
    model = get_model(cfg.model, dataset)
    model.load_state_dict(torch.load(str(Path(args.load_from_dir) / "best.pt"),
                                     map_location=cfg.device))
    model.to(cfg.device).eval()

    # We need to keep track of which test sentences the loader actually yields
    # (length>=2 after the DataModule drop). Build a lookup table from the
    # word-sequence to its original raw-pickle index so we can map predictions
    # back to gold tree text lines.
    test_ds = dataset.test_dataset
    # IMPORTANT: read the raw word ids before any further indexing/sorting.
    raw_word_seqs = [tuple(w) for w in test_ds.get_field("word").content]

    test_loader = dataset.test_dataloader
    test_loader = DataPrefetcher(test_loader, device=cfg.device)

    predictions = {}  # raw_word_seq_tuple -> spans
    with torch.no_grad():
        for x, y in test_loader:
            result = model.evaluate(x, decode_type=args.decode_type, eval_dep=False)
            spans_batch = result["prediction"]
            word_batch = x["word"].cpu().tolist()
            seq_lens = x["seq_len"].cpu().tolist()
            for spans, wid_seq, L in zip(spans_batch, word_batch, seq_lens):
                key = tuple(wid_seq[:L])
                predictions[key] = spans

    # Load the gold file aligned with the pickle. The pickle's "word" field
    # holds the surface tokens; we need to write the surface tokens back into
    # the predicted tree so the eval script sees the same leaves as gold.
    test_pickle_path = Path(cfg.data.test_file)
    with test_pickle_path.open("rb") as f:
        raw_test = pickle.load(f)
    raw_words_per_sent = raw_test["word"]
    n_total = len(raw_words_per_sent)

    # Also load the matching gold bracketed text lines (parallel to pickle order).
    # Li et al.'s preprocess.py wrote one tree per line in the same order as the
    # pickle was built from. So we can read those lines and filter to len>=2.
    # Provided via the required --gold_text_file argument.
    gold_text_file = Path(args.gold_text_file)
    with gold_text_file.open("r", encoding="utf-8") as f:
        gold_lines = [ln.strip() for ln in f if ln.strip()]
    assert len(gold_lines) == n_total, (
        f"gold lines {len(gold_lines)} != pickle sents {n_total}"
    )

    # Map the surface-form word tuple to vocab-id tuple via the loader's vocab.
    word_vocab = dataset.word_vocab

    pred_writes = 0
    gold_writes = 0
    skipped_len1 = 0
    skipped_no_pred = 0
    with open(args.pred_out, "w", encoding="utf-8") as fpred, \
         open(args.gold_out, "w", encoding="utf-8") as fgold:
        for surf_words, gold_line in zip(raw_words_per_sent, gold_lines):
            n = len(surf_words)
            if n < 2:
                skipped_len1 += 1
                continue
            # surf_words is list of strings; convert via clean_word equivalent
            # used by data_module: lowercase + digit->N
            import re
            def clean(w):
                w = w.lower()
                w = re.sub("[0-9]{1,}([,.]?[0-9]*)*", "N", w)
                return w
            cleaned = [clean(w) for w in surf_words]
            wid_seq = tuple(word_vocab.to_index(w) for w in cleaned)
            spans = predictions.get(wid_seq)
            if spans is None:
                skipped_no_pred += 1
                continue
            tree_str = spans_to_bracketed(spans, list(surf_words), label="S")
            fpred.write(tree_str + "\n")
            fgold.write(gold_line + "\n")
            pred_writes += 1
            gold_writes += 1

    print(f"wrote {pred_writes} pred trees; gold {gold_writes}; "
          f"skipped len-1={skipped_len1}, missing-pred={skipped_no_pred}")
    print(f"pred file: {args.pred_out}")
    print(f"gold file: {args.gold_out}")


if __name__ == "__main__":
    main()
