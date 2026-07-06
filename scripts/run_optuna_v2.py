#!/usr/bin/env python3
"""Optuna HP search worker for HN-PCFG v2.

Searches: lr, mu, batch_size, tau_root_init, tau_rule_init, tau_term_init.
Fixed: nu=0.999, patience=10, clip=3, relation_weight_decay=0.0.
Uses dual-track model selection (max of LL-best-F1 and overall-best-F1).
Training seed is fixed at 42 for all trials.

Usage (single worker):
    python scripts/run_optuna_v2.py --device 0 --worker-id 0

This script is typically launched by run_pipeline.py.
"""

import argparse
import copy
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
import optuna
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
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    t = tqdm(loader, total=int(len(loader)), position=0, leave=True,
             disable=True)  # Disable tqdm in worker mode
    for x, _ in t:
        optimizer.zero_grad()
        loss = model.loss(x)
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"NaN/Inf loss detected: {loss.item()}")
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
    """Evaluate model. Returns (UF1, LikelihoodMetric)."""
    model.eval()
    metric_f1 = UF1()
    metric_ll = LikelihoodMetric()
    for x, y in loader:
        result = model.evaluate(x, decode_type='mbr')
        metric_f1(result['prediction'], y['gold_tree'])
        metric_ll(result['partition'], x['seq_len'])
    return metric_f1, metric_ll


def objective(trial: optuna.Trial, base_config_path: str,
              device: str, training_seed: int) -> float:
    """Optuna objective function. Returns best val/sentence_f1 (dual-track)."""

    with open(base_config_path) as f:
        yaml_cfg = yaml.load(f, Loader=yaml.Loader)
    args = edict(yaml_cfg)

    # --- Suggest HPs ---
    lr = trial.suggest_float('lr', 5e-4, 1.5e-2, log=True)
    mu = trial.suggest_categorical('mu', [0.5, 0.75, 0.9])
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    tau_root_init = trial.suggest_float('tau_root_init', 1.0, 15.0, log=True)
    tau_rule_init = trial.suggest_float('tau_rule_init', 1.0, 15.0, log=True)
    tau_term_init = trial.suggest_float('tau_term_init', 1.0, 15.0, log=True)

    # Override config — searched params
    args.optimizer.lr = lr
    args.optimizer.mu = mu
    args.train.batch_size = batch_size
    args.model.tau_root_init = tau_root_init
    args.model.tau_rule_init = tau_rule_init
    args.model.tau_term_init = tau_term_init

    # Override config — fixed params
    args.optimizer.nu = 0.999
    args.optimizer.relation_weight_decay = 0.0
    args.train.patience = 10
    args.train.clip = 3

    # Device: CUDA_VISIBLE_DEVICES is set by the parent process (GPUPool
    # or launch script).  Do NOT override it here — the objective function
    # runs inside a worker process whose GPU was already assigned.
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(training_seed)

    # Save dir — trial.number is unique, but add PID as extra safety
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.save_dir = (f"log/optuna_v2/{trial.study.study_name}/"
                     f"trial{trial.number}_{timestamp}_pid{os.getpid()}")
    os.makedirs(args.save_dir, exist_ok=True)

    with open(f"{args.save_dir}/config.yaml", 'w') as f:
        yaml.dump(dict(args), f)

    # W&B
    wandb_cfg = getattr(args, 'wandb', None)
    use_wandb = wandb_cfg is not None and getattr(wandb_cfg, 'enabled', True)
    if use_wandb:
        wandb.init(
            project=getattr(wandb_cfg, 'project', 'hol-pcfg'),
            entity=getattr(wandb_cfg, 'entity', None),
            name=(f"optuna-v2-t{trial.number}"
                  f"-lr{lr:.1e}-mu{mu}-bs{batch_size}"
                  f"-tr{tau_root_init:.1f}-trl{tau_rule_init:.1f}"
                  f"-tt{tau_term_init:.1f}"),
            tags=['optuna-v2', trial.study.study_name],
            config={
                'trial_number': trial.number,
                'seed': training_seed,
                'model': dict(args.model),
                'train': dict(args.train),
                'optimizer': dict(args.optimizer),
            },
            reinit=True,
        )

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

            # Loss explosion guard
            if avg_loss > 500:
                raise ValueError(f"Loss explosion: {avg_loss:.1f}")

            dev_f1_metric, dev_ll = evaluate(model, eval_auto)
            val_sf1 = dev_f1_metric.sentence_uf1

            # --- Dual-track model selection ---
            # Track 1: LL-best (save checkpoint at best likelihood epoch)
            is_best_ll = dev_ll > best_ll
            if is_best_ll:
                best_ll = dev_ll
                best_epoch = epoch
                f1_at_best_ll = val_sf1
                torch.save(model.state_dict(),
                           f"{args.save_dir}/best.pt")

            # Track 2: overall best F1
            if val_sf1 > best_f1_overall:
                best_f1_overall = val_sf1

            # W&B logging
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

            # Optuna: report F1 and check pruning
            trial.report(val_sf1, epoch)
            if trial.should_prune():
                if use_wandb:
                    wandb.finish(quiet=True)
                raise optuna.TrialPruned()

            # Early stopping (patience-based, LL)
            if (train_arg.patience > 0
                    and epoch - best_epoch >= train_arg.patience):
                break

    except optuna.TrialPruned:
        raise
    except Exception as e:
        if use_wandb:
            wandb.finish(quiet=True)
        raise optuna.TrialPruned(f"Trial failed: {e}")
    finally:
        if use_wandb:
            wandb.finish(quiet=True)
        if 'model' in locals():
            del model
        if 'optimizer' in locals():
            del optimizer
        torch.cuda.empty_cache()

    # Dual-track: return the better of the two
    return max(f1_at_best_ll, best_f1_overall)


def main():
    parser = argparse.ArgumentParser(
        description='Optuna HP search worker for HN-PCFG v2')
    parser.add_argument('--device', '-d', default='0',
                        help='GPU device ID (CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--worker-id', '-w', type=int, default=0,
                        help='Worker ID for sampler seed differentiation')
    parser.add_argument('--study-name', default='hnpcfg-v2-hp',
                        help='Optuna study name')
    parser.add_argument('--base-config',
                        default='archive/configs/normalization_phases/hn_pcfg_nt4096_optuna_v2.yaml',
                        help='Base YAML config')
    parser.add_argument('--n-trials', type=int, default=7,
                        help='Number of trials per worker')
    parser.add_argument('--journal-path',
                        default='optuna_v2_journal.log',
                        help='Path to JournalStorage file')
    parser.add_argument('--training-seed', type=int, default=42,
                        help='Fixed training seed for all trials')
    parser.add_argument('--startup-delay', type=float, default=2.0,
                        help='Per-worker startup delay in seconds')
    args = parser.parse_args()

    # Set GPU for standalone use.  When launched by GPUPool (run_pipeline.py),
    # CUDA_VISIBLE_DEVICES is already set in the subprocess environment.
    # setdefault ensures we don't override GPUPool's assignment.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.device)

    # Stagger startup to let constant_liar see other workers' RUNNING trials
    delay = args.worker_id * args.startup_delay
    if delay > 0:
        print(f"[Worker {args.worker_id}] Waiting {delay:.0f}s startup delay")
        time.sleep(delay)

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(args.journal_path)
    )
    sampler = optuna.samplers.TPESampler(
        seed=42 + args.worker_id,
        constant_liar=True,
        n_startup_trials=20,
        multivariate=True,
        group=True,
    )
    pruner = optuna.pruners.PercentilePruner(
        percentile=33.0,
        n_startup_trials=15,
        n_warmup_steps=5,
        interval_steps=2,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction='maximize',
        load_if_exists=True,
    )

    print(f"[Worker {args.worker_id}] Study '{args.study_name}' "
          f"on GPU {args.device}, n_trials={args.n_trials}")
    print(f"[Worker {args.worker_id}] Existing trials: {len(study.trials)}")

    study.optimize(
        lambda trial: objective(trial, args.base_config, args.device,
                                args.training_seed),
        n_trials=args.n_trials,
    )

    print(f"[Worker {args.worker_id}] Done. "
          f"Total trials: {len(study.trials)}")


if __name__ == '__main__':
    main()
