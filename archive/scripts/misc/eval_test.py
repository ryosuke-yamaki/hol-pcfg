#!/usr/bin/env python3
"""Evaluate saved checkpoints on test data.

Usage:
    python scripts/eval_test.py \
        --config runs/optuna_v2/hnpcfg-v2-hp/phase2/configs/rank1.yaml \
        --checkpoint-dir log/pipeline/ \
        --pattern 'rank1_seed*_20260410*' \
        --device 0
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import torch
import yaml
from easydict import EasyDict as edict

from parser.helper.data_module import DataModule
from parser.helper.loader_wrapper import DataPrefetcher
from parser.helper.metric import UF1, LikelihoodMetric
from parser.helper.util import get_model


@torch.no_grad()
def evaluate_test(model, loader) -> dict:
    model.eval()
    metric_f1 = UF1()
    metric_ll = LikelihoodMetric()
    for x, y in loader:
        result = model.evaluate(x, decode_type='mbr')
        metric_f1(result['prediction'], y['gold_tree'])
        metric_ll(result['partition'], x['seq_len'])
    return {
        'sentence_f1': metric_f1.sentence_uf1,
        'corpus_f1': metric_f1.corpus_uf1,
        'likelihood': metric_ll.avg_likelihood.item(),
        'perplexity': metric_ll.perplexity.item(),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate on test data')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--pattern', required=True,
                        help='Glob pattern for checkpoint dirs')
    parser.add_argument('--device', default='0')
    parser.add_argument('--output', default=None,
                        help='JSON output path')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    with open(args.config) as f:
        cfg = edict(yaml.load(f, Loader=yaml.Loader))
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = DataModule(cfg)
    test_loader = dataset.test_dataloader

    # Find checkpoints
    dirs = sorted(glob.glob(os.path.join(args.checkpoint_dir, args.pattern)))
    checkpoints = []
    for d in dirs:
        pt = os.path.join(d, 'best.pt')
        if os.path.exists(pt):
            # Extract seed from dirname: rank1_seed3_20260410_..._pid12345
            m = re.search(r'seed(\d+)', os.path.basename(d))
            seed = int(m.group(1)) if m else -1
            checkpoints.append((seed, pt))

    checkpoints.sort(key=lambda x: x[0])
    print(f"Found {len(checkpoints)} checkpoints")
    print()

    results = []
    for seed, ckpt_path in checkpoints:
        model = get_model(cfg.model, dataset)
        state = torch.load(ckpt_path, map_location=cfg.device, weights_only=True)
        model.load_state_dict(state)

        test_auto = DataPrefetcher(test_loader, device=cfg.device)
        metrics = evaluate_test(model, test_auto)

        results.append({
            'seed': seed,
            'test_sentence_f1': metrics['sentence_f1'],
            'test_corpus_f1': metrics['corpus_f1'],
            'test_likelihood': metrics['likelihood'],
            'test_perplexity': metrics['perplexity'],
        })
        print(f"  seed {seed:>3}: SF1={metrics['sentence_f1']*100:.2f}%  "
              f"CF1={metrics['corpus_f1']*100:.2f}%  "
              f"LL={metrics['likelihood']:.2f}")

        del model, state
        torch.cuda.empty_cache()

    # Summary
    import numpy as np
    sf1s = [r['test_sentence_f1'] for r in results]
    mean = np.mean(sf1s) * 100
    std = np.std(sf1s, ddof=1) * 100
    collapse = sum(1 for x in sf1s if x < 0.55)

    print()
    print("=" * 55)
    print(f"TEST RESULT: {mean:.2f}% +/- {std:.2f}%  "
          f"({len(sf1s)} seeds, {collapse} collapse)")
    print(f"Target:      65.10% +/- 2.10% (SN-PCFG NT=4096)")
    print("=" * 55)

    if args.output:
        out = {
            'mean_test_f1': mean,
            'std_test_f1': std,
            'n_seeds': len(sf1s),
            'n_collapse': collapse,
            'individual': results,
        }
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
