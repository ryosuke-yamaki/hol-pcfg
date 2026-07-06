#!/usr/bin/env python3
"""Run a single HN-PCFG training and save results to JSON.

Used by run_pipeline.py for Phase 0 (baseline), Phase 2 (seed validation),
and Phase 3 (final validation).

Usage:
    python scripts/run_single_train.py \
        --config archive/configs/normalization_phases/hn_pcfg_nt4096_optuna_v2.yaml \
        --seed 42 --device 0 --result-path results/phase0/seed42.json
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

import wandb
from parser.helper.data_module import DataModule
from parser.helper.loader_wrapper import DataPrefetcher
from parser.helper.metric import LikelihoodMetric, Metric, UF1
from parser.helper.util import get_model, get_optimizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, optimizer, loader, clip: float) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, _ in loader:
        optimizer.zero_grad()
        loss = model.loss(x)
        loss.backward()
        if clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if hasattr(model, 'project_embeddings'):
            model.project_embeddings()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader) -> tuple:
    model.eval()
    metric_f1 = UF1()
    metric_ll = LikelihoodMetric()
    for x, y in loader:
        result = model.evaluate(x, decode_type='mbr')
        metric_f1(result['prediction'], y['gold_tree'])
        metric_ll(result['partition'], x['seq_len'])
    return metric_f1, metric_ll


def main():
    parser = argparse.ArgumentParser(
        description='Single HN-PCFG training run with JSON result output')
    parser.add_argument('--config', '-c', required=True,
                        help='YAML config path')
    parser.add_argument('--seed', '-s', type=int, required=True,
                        help='Random seed')
    parser.add_argument('--device', '-d', default=None,
                        help='GPU device ID (omit to use CUDA_VISIBLE_DEVICES '
                             'from environment, e.g. set by GPUPool)')
    parser.add_argument('--result-path', '-r', required=True,
                        help='Path to save JSON result')
    parser.add_argument('--wandb-name', default=None,
                        help='W&B run name override')
    parser.add_argument('--wandb-tags', nargs='*', default=None,
                        help='Additional W&B tags')
    cli = parser.parse_args()

    # Load config
    with open(cli.config) as f:
        yaml_cfg = yaml.load(f, Loader=yaml.Loader)
    args = edict(yaml_cfg)

    # Device: respect CUDA_VISIBLE_DEVICES set by parent (e.g. GPUPool).
    # --device flag is only applied when running standalone.
    if cli.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cli.device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(cli.seed)

    # Save dir — include PID to avoid collision when multiple processes
    # start within the same second (e.g. 16 parallel workers in Phase 2)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    config_stem = Path(cli.config).stem
    base_save_dir = args.get('save_dir', 'log')
    args.save_dir = (f"{base_save_dir}/pipeline/{config_stem}"
                     f"_seed{cli.seed}_{timestamp}_pid{os.getpid()}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Ensure result dir exists
    os.makedirs(Path(cli.result_path).parent, exist_ok=True)

    # W&B
    wandb_cfg = getattr(args, 'wandb', None)
    use_wandb = wandb_cfg is not None and getattr(wandb_cfg, 'enabled', True)
    if use_wandb:
        run_name = cli.wandb_name or f"{config_stem}-seed{cli.seed}"
        tags = list(getattr(wandb_cfg, 'tags', []) or [])
        if cli.wandb_tags:
            tags.extend(cli.wandb_tags)
        wandb.init(
            project=getattr(wandb_cfg, 'project', 'hol-pcfg'),
            entity=getattr(wandb_cfg, 'entity', None),
            name=run_name,
            group=getattr(wandb_cfg, 'group', None),
            tags=tags,
            config={
                'seed': cli.seed,
                'model': dict(args.model),
                'train': dict(args.train),
                'optimizer': dict(args.optimizer),
            },
            reinit=True,
        )

    result = {'seed': cli.seed, 'config': cli.config, 'status': 'failed'}

    try:
        dataset = DataModule(args)
        model = get_model(args.model, dataset)
        optimizer = get_optimizer(args.optimizer, model)

        train_arg = args.train
        eval_loader = dataset.val_dataloader

        best_ll = Metric()
        f1_at_best_ll = 0.0
        best_f1_overall = 0.0
        best_epoch = 1

        for epoch in range(1, train_arg.max_epoch + 1):
            if train_arg.curriculum:
                train_loader = dataset.train_dataloader(
                    max_len=min(train_arg.start_len + epoch - 1,
                                train_arg.max_len))
            else:
                train_loader = dataset.train_dataloader(
                    max_len=train_arg.max_len)

            train_auto = DataPrefetcher(train_loader, device=args.device)
            eval_auto = DataPrefetcher(eval_loader, device=args.device)

            avg_loss = train_one_epoch(model, optimizer, train_auto,
                                       train_arg.clip)

            dev_f1_metric, dev_ll = evaluate(model, eval_auto)
            val_sf1 = dev_f1_metric.sentence_uf1

            # Dual-track selection
            is_best_ll = dev_ll > best_ll
            if is_best_ll:
                best_ll = dev_ll
                best_epoch = epoch
                f1_at_best_ll = val_sf1
                torch.save(model.state_dict(),
                           f"{args.save_dir}/best.pt")

            if val_sf1 > best_f1_overall:
                best_f1_overall = val_sf1

            if use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train/loss': avg_loss,
                    'val/sentence_f1': val_sf1,
                    'val/corpus_f1': dev_f1_metric.corpus_uf1,
                    'val/likelihood': dev_ll.avg_likelihood.item(),
                    'val/perplexity': dev_ll.perplexity.item(),
                    'best/epoch': best_epoch,
                    'best/likelihood': best_ll.avg_likelihood.item(),
                    'best/f1_at_best_ll': f1_at_best_ll,
                    'best/f1_overall': best_f1_overall,
                }
                if hasattr(model, 'get_monitoring_metrics'):
                    log_dict.update(model.get_monitoring_metrics())
                wandb.log(log_dict)

            print(f"Epoch {epoch}/{train_arg.max_epoch} | "
                  f"loss={avg_loss:.2f} | SF1={val_sf1:.4f} | "
                  f"best_F1={max(f1_at_best_ll, best_f1_overall):.4f}")

            # Early stopping
            if (train_arg.patience > 0
                    and epoch - best_epoch >= train_arg.patience):
                print(f"Early stopping at epoch {epoch} "
                      f"(patience={train_arg.patience})")
                break

        # Collect monitoring metrics from final state
        tau_metrics = {}
        if hasattr(model, 'get_monitoring_metrics'):
            mon = model.get_monitoring_metrics()
            tau_metrics = {
                'tau_root': mon.get('monitor/tau_root', None),
                'tau_rule': mon.get('monitor/tau_rule', None),
                'tau_term': mon.get('monitor/tau_term', None),
            }

        best_f1 = max(f1_at_best_ll, best_f1_overall)
        result = {
            'seed': cli.seed,
            'config': cli.config,
            'status': 'completed',
            'best_f1': best_f1,
            'f1_at_best_ll': f1_at_best_ll,
            'best_f1_overall': best_f1_overall,
            'best_ll': best_ll.avg_likelihood.item(),
            'best_epoch': best_epoch,
            'total_epochs': epoch,
            **tau_metrics,
        }

        # Final test evaluation using the val-LL-best checkpoint
        best_ckpt_path = f"{args.save_dir}/best.pt"
        if os.path.exists(best_ckpt_path):
            state = torch.load(best_ckpt_path, map_location=args.device,
                               weights_only=True)
            model.load_state_dict(state)
            test_loader = dataset.test_dataloader
            test_auto = DataPrefetcher(test_loader, device=args.device)
            test_f1_metric, test_ll = evaluate(model, test_auto)
            test_sf1 = test_f1_metric.sentence_uf1
            test_cf1 = test_f1_metric.corpus_uf1
            test_ll_val = test_ll.avg_likelihood.item()
            test_ppl = test_ll.perplexity.item()

            result['test_sentence_f1'] = test_sf1
            result['test_corpus_f1'] = test_cf1
            result['test_likelihood'] = test_ll_val
            result['test_perplexity'] = test_ppl

            print(f"Test (best ckpt epoch {best_epoch}): "
                  f"SF1={test_sf1:.4f}  CF1={test_cf1:.4f}  "
                  f"LL={test_ll_val:.2f}  PPL={test_ppl:.2f}")

            if use_wandb:
                wandb.log({
                    'test/sentence_f1': test_sf1,
                    'test/corpus_f1': test_cf1,
                    'test/likelihood': test_ll_val,
                    'test/perplexity': test_ppl,
                })
                wandb.summary['test/sentence_f1'] = test_sf1
                wandb.summary['test/corpus_f1'] = test_cf1
                wandb.summary['test/likelihood'] = test_ll_val
                wandb.summary['test/perplexity'] = test_ppl
        else:
            print(f"WARNING: no best checkpoint at {best_ckpt_path}, "
                  f"skipping test evaluation")

    except Exception as e:
        result['error'] = str(e)
        print(f"Training failed: {e}")
    finally:
        if use_wandb:
            wandb.finish(quiet=True)
        if 'model' in locals():
            del model
        if 'optimizer' in locals():
            del optimizer
        torch.cuda.empty_cache()

    # Save result
    with open(cli.result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to {cli.result_path}")


if __name__ == '__main__':
    main()
