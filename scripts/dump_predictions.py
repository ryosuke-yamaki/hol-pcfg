"""Dump per-sentence MBR predictions for a trained hol-pcfg checkpoint.

For each test sentence the script writes one JSONL line containing:
  - sent_id        : raw-pickle index (matches data/clean/english-test.pickle)
  - words          : cleaned word list (lowercased, numbers replaced by 'N')
  - length         : len(words)
  - gold_spans     : [[start, end, label], ...] from pickle gold_tree
  - pred_spans     : MBR-decoded spans as [[start, end], ...]
  - raw_words      : original-case word list from pickle
  - raw_pos        : POS tags aligned with raw_words

Span indices are 0-indexed token positions; hol-pcfg test data is already
punct-removed (data/clean), so cleaned/raw share the same span coords.

Usage:
    python scripts/dump_predictions.py \
        --conf runs/optuna_v2/hnpcfg-v2-hp/phase2/configs/rank1.yaml \
        --ckpt log/pipeline/rank1_seed12_20260410_084429_pid93379/best.pt \
        --out  analysis/error_analysis/HN-PCFG/predictions_test.jsonl \
        --data_test data/clean/english-test.pickle

Unlike the vendored `parsing_by_maxseminfo` harness, hol-pcfg uses raw torch
state_dict (no Lightning hyper_parameters), so the yaml config must be supplied
separately.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import yaml
from easydict import EasyDict as edict


# Mirror DataModule.setup() clean_word: lowercase + replace digits with 'N'.
_NUM_RE = re.compile(r"[0-9]{1,}([,.]?[0-9]*)*")


def _clean_word(w: str) -> str:
    return _NUM_RE.sub("N", w.lower())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dump MBR predictions to JSONL for a trained hol-pcfg ckpt."
    )
    p.add_argument("--conf", "-c", required=True,
                   help="Path to the yaml config used for training.")
    p.add_argument("--ckpt", required=True,
                   help="Path to the raw state_dict checkpoint (best.pt).")
    p.add_argument("--out", required=True, help="Output JSONL path.")
    p.add_argument("--data_test", default=None,
                   help="Override config's data.test_file (e.g. when the yaml "
                        "still references the old data/ptb-test.pickle path).")
    p.add_argument("--data_train", default=None,
                   help="Override data.train_file (DataModule loads all 3 "
                        "splits to build the vocab; train+val paths still need "
                        "to resolve).")
    p.add_argument("--data_val", default=None,
                   help="Override data.val_file.")
    p.add_argument("--data_dir", default=None,
                   help="Convenience: set train/val/test to "
                        "<dir>/english-{train,val,test}.pickle.")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.conf) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    cfg = edict(cfg)
    cfg.device = args.device if torch.cuda.is_available() else "cpu"

    if args.data_dir is not None:
        d = args.data_dir.rstrip("/")
        cfg.data.train_file = f"{d}/english-train.pickle"
        cfg.data.val_file = f"{d}/english-val.pickle"
        cfg.data.test_file = f"{d}/english-test.pickle"
    if args.data_train is not None:
        cfg.data.train_file = args.data_train
    if args.data_val is not None:
        cfg.data.val_file = args.data_val
    if args.data_test is not None:
        cfg.data.test_file = args.data_test

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] config loaded from {args.conf}")
    print(f"[info] test pickle:        {cfg.data.test_file}")

    from parser.helper.data_module import DataModule
    from parser.helper.util import get_model

    dataset = DataModule(cfg)
    model = get_model(cfg.model, dataset)
    state = torch.load(args.ckpt, map_location=cfg.device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print(f"[warn] missing keys: {missing}", file=sys.stderr)
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}", file=sys.stderr)
    model.eval()
    print(f"[info] checkpoint loaded from {args.ckpt}")

    # Reload raw pickle to get pos / raw_words / gold (DataModule mangles word
    # to indexed integers and doesn't carry pos through).
    with open(cfg.data.test_file, "rb") as f:
        raw = pickle.load(f)
    raw_words = raw["word"]
    raw_pos = raw["pos"]
    raw_gold = raw["gold_tree"]

    # DataModule drops seq_len==1 sentences. Map kept-idx → raw-idx.
    kept_to_raw = [i for i, w in enumerate(raw_words) if len(w) > 1]
    assert len(kept_to_raw) == len(dataset.test_dataset), (
        f"kept count mismatch: pickle keeps {len(kept_to_raw)} "
        f"but test_dataset has {len(dataset.test_dataset)}"
    )

    test_dataset = dataset.test_dataset
    test_loader = dataset.test_dataloader
    sampler = test_loader.batch_sampler  # ByLengthSampler

    # Walk sampler.total (deterministic, equal-length batches; no padding).
    n_written = 0
    n_batches = 0
    with open(out_path, "w") as fout, torch.no_grad():
        for batch_indices in sampler.total:
            lens = [test_dataset[i]["seq_len"] for i in batch_indices]
            assert max(lens) == min(lens), (
                "ByLengthSampler should yield equal-length batches"
            )

            word_ids = torch.tensor(
                [test_dataset[i]["word"] for i in batch_indices],
                device=cfg.device,
            )
            seq_len = torch.tensor(lens, device=cfg.device)
            x_target = {"word": word_ids, "seq_len": seq_len}

            outputs = model.evaluate(x_target, decode_type="mbr")
            preds = outputs["prediction"]

            for k, kept_idx in enumerate(batch_indices):
                raw_idx = kept_to_raw[kept_idx]
                w_list = list(raw_words[raw_idx])
                p_list = list(raw_pos[raw_idx])
                g_list = list(raw_gold[raw_idx])
                cleaned = [_clean_word(w) for w in w_list]

                record = {
                    "sent_id": int(raw_idx),
                    "words": cleaned,
                    "length": len(cleaned),
                    "gold_spans": [[int(s), int(e), str(lbl)] for s, e, lbl in g_list],
                    "pred_spans": [[int(s), int(e)] for s, e in preds[k]],
                    "raw_words": w_list,
                    "raw_pos": p_list,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1
            n_batches += 1

    print(f"[info] wrote {n_written} sentences across {n_batches} batches to {out_path}")


if __name__ == "__main__":
    main()
