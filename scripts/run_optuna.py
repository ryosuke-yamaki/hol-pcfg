#!/usr/bin/env python3
"""Optuna HP search for HN-PCFG (allproj cnorm + single tau).

Usage:
    # Single GPU
    python scripts/run_optuna.py --device 0 --study-name hn-pcfg-hp

    # Multi-GPU (4 processes sharing one study)
    bash scripts/launch_optuna_multigpu.sh hn-pcfg-hp
"""

import argparse
import copy
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta
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
    t = tqdm(loader, total=int(len(loader)), position=0, leave=True)
    for x, _ in t:
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
        t.set_postfix(loss=loss.item())
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


def objective(trial: optuna.Trial, base_config_path: str, device: str) -> float:
    """Optuna objective function. Returns best val/sentence_f1."""

    # Load base config
    with open(base_config_path) as f:
        yaml_cfg = yaml.load(f, Loader=yaml.Loader)
    args = edict(yaml_cfg)

    # --- Suggest HPs ---
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    mu = trial.suggest_categorical('mu', [0.5, 0.75, 0.9, 0.95])
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    nu = trial.suggest_categorical('nu', [0.99, 0.999])

    # Override config
    args.optimizer.lr = lr
    args.optimizer.mu = mu
    args.optimizer.nu = nu
    args.train.batch_size = batch_size
    args.train.patience = 8
    args.optimizer.relation_weight_decay = 0.0

    # Device and seed
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42 + trial.number
    set_seed(seed)

    # Save dir
    timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    args.save_dir = f"log/optuna/{trial.study.study_name}/trial{trial.number}_{timestamp}"
    os.makedirs(args.save_dir, exist_ok=True)

    # Save config for reproducibility
    with open(f"{args.save_dir}/config.yaml", 'w') as f:
        yaml.dump(dict(args), f)

    # W&B
    wandb_cfg = getattr(args, 'wandb', None)
    use_wandb = wandb_cfg is not None and getattr(wandb_cfg, 'enabled', True)
    if use_wandb:
        wandb.init(
            project=getattr(wandb_cfg, 'project', 'hol-pcfg'),
            entity=getattr(wandb_cfg, 'entity', None),
            name=f"optuna-t{trial.number}-lr{lr:.1e}-mu{mu}-bs{batch_size}-nu{nu}",
            tags=['optuna', trial.study.study_name],
            config={
                'trial_number': trial.number,
                'seed': seed,
                'model': dict(args.model),
                'train': dict(args.train),
                'optimizer': dict(args.optimizer),
            },
            reinit=True,
        )

    try:
        # Build model and optimizer
        dataset = DataModule(args)
        model = get_model(args.model, dataset)
        optimizer = get_optimizer(args.optimizer, model)

        train_arg = args.train
        eval_loader = dataset.val_dataloader

        best_ll = Metric()
        best_f1 = 0.0
        best_epoch = 1

        for epoch in range(1, train_arg.max_epoch + 1):
            # Train
            if train_arg.curriculum:
                train_loader = dataset.train_dataloader(
                    max_len=min(train_arg.start_len + epoch - 1, train_arg.max_len))
            else:
                train_loader = dataset.train_dataloader(max_len=train_arg.max_len)

            train_loader_auto = DataPrefetcher(train_loader, device=args.device)
            eval_loader_auto = DataPrefetcher(eval_loader, device=args.device)

            avg_loss = train_one_epoch(model, optimizer, train_loader_auto, train_arg.clip)

            # Evaluate
            dev_f1_metric, dev_ll = evaluate(model, eval_loader_auto)
            val_sf1 = dev_f1_metric.sentence_uf1

            # Model selection: likelihood-based (same as baseline)
            is_best = dev_ll > best_ll
            if is_best:
                best_ll = dev_ll
                best_epoch = epoch
                best_f1 = val_sf1
                torch.save(model.state_dict(), f"{args.save_dir}/best.pt")

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
                }
                if hasattr(model, 'get_monitoring_metrics'):
                    log_dict.update(model.get_monitoring_metrics())
                wandb.log(log_dict)

            # Optuna: report and prune
            trial.report(val_sf1, epoch)
            if trial.should_prune():
                if use_wandb:
                    wandb.finish(quiet=True)
                raise optuna.TrialPruned()

            # Early stopping (patience-based, likelihood)
            if train_arg.patience > 0 and epoch - best_epoch >= train_arg.patience:
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
        # Free GPU memory
        if 'model' in locals():
            del model
        if 'optimizer' in locals():
            del optimizer
        torch.cuda.empty_cache()

    return best_f1


def main():
    parser = argparse.ArgumentParser(description='Optuna HP search for HN-PCFG')
    parser.add_argument('--device', '-d', default='0', help='GPU device ID')
    parser.add_argument('--study-name', default='hn-pcfg-hp', help='Optuna study name')
    parser.add_argument('--base-config', default='archive/configs/normalization_phases/hn_pcfg_allproj_cnorm_tau.yaml',
                        help='Base YAML config to optimize')
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Number of trials (None = run until stopped)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds (None = no timeout)')
    parser.add_argument('--journal-path', default='optuna_journal.log',
                        help='Path to JournalStorage file')
    parser.add_argument('--startup-delay', type=float, default=10.0,
                        help='Per-device startup delay in seconds (device_id * delay)')
    cli_args = parser.parse_args()

    # Stagger worker startup to avoid simultaneous ask() during startup trials
    device_id = int(cli_args.device)
    delay = device_id * cli_args.startup_delay
    if delay > 0:
        print(f"Waiting {delay:.0f}s (device {device_id} startup delay)...")
        time.sleep(delay)

    # Create study
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(cli_args.journal_path)
    )
    sampler = optuna.samplers.TPESampler(
        seed=42 + device_id,
        constant_liar=True,
        n_startup_trials=10,
    )
    pruner = optuna.pruners.PercentilePruner(
        percentile=40.0,
        n_startup_trials=10,
        n_warmup_steps=8,
    )

    study = optuna.create_study(
        study_name=cli_args.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction='maximize',
        load_if_exists=True,
    )

    print(f"Study '{cli_args.study_name}' on GPU {cli_args.device}")
    print(f"Existing trials: {len(study.trials)}")

    study.optimize(
        lambda trial: objective(trial, cli_args.base_config, cli_args.device),
        n_trials=cli_args.n_trials,
        timeout=cli_args.timeout,
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best val SF1: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")
    print("=" * 60)


if __name__ == '__main__':
    main()
