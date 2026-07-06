#!/usr/bin/env python3
"""Phase 0 baseline: HN-PCFG (refactored, phase-only) + Sem-Info at 100k steps.

Runs the rank1 baseline HP (lr=1e-3, mu=0.9, tau_*_init=7, num_samples=2,
rl_warmup=5k, maxent=-0.01) on the post-refactor HN-PCFG model for a single
seed. Used as the Phase 0 reference for the Optuna v3 pipeline.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_phase0_baseline.py --seed 1 --device 0
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from parsing_by_maxseminfo import parser as seminfo_parser
sys.modules['parser'] = seminfo_parser

import lightning.pytorch as L
import torch
import yaml
from easydict import EasyDict as edict
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from parser.lightning_wrapper.LitNPCFG import LitXNPCFGFCReward
from parser.lightning_wrapper.scheduler import WarmupScheduler
from parser.helper.pas_grammar_data_helper import DataModuleForPASCtrlPCFGReward


BASE_CONFIG = "config/seminfo/hnpcfg_nt1024_t2048_rank1_seminfo.yaml"
WANDB_PROJECT = "hol-pcfg"
# W&B entity is not hardcoded: defaults to the WANDB_ENTITY env var (None -> use
# the logged-in default entity). Override with --wandb_entity.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
STUDY_NAME = "hnpcfg-rank1-seminfo-v3"  # matches Optuna v3 study for downstream aggregation


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--wandb_entity', default=None,
                   help='W&B entity (default: WANDB_ENTITY env var, else the '
                        'logged-in default entity).')
    cli = p.parse_args()
    if cli.wandb_entity is not None:
        globals()['WANDB_ENTITY'] = cli.wandb_entity

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cli.device)
    L.seed_everything(cli.seed, workers=True)

    with open(BASE_CONFIG) as f:
        args = edict(yaml.load(f, Loader=yaml.Loader))
    args.langstr = getattr(args, 'langstr', 'english')

    device = torch.device('cuda')

    # Data
    dst = DataModuleForPASCtrlPCFGReward(
        hparams=args, langstr=args.langstr, use_cache=True,
        max_size=10000, merge_pas_data=False,
        pas_subsample=getattr(args, 'preprocessing_pas_subsample_count', 0),
        flag_use_pos_unks=getattr(args.experimental, 'flag_use_pos_unks', False),
    )
    train_dl, _ = dst.train_dataloader(
        args.langstr, max_len=getattr(args, 'max_length', 40), min_len=3,
        device=device,
        pas_subsample_count=args.experimental.pas_subsample_count,
        flag_curriculum_learning=getattr(args.experimental, 'flag_curriculum_learning', False),
        add_sentence_level_span=getattr(args.experimental, 'add_sentence_level_span', False),
        min_span_reward=args.experimental.min_span_reward,
        mode_reward=args.experimental.mode_reward,
        supervised_mode=getattr(args.experimental, 'supervised_mode', False),
    )
    val_dl, _ = dst.dev_full_dataloader(
        args.langstr, max_len=100000, min_len=2, device=device,
        min_span_reward=args.experimental.min_span_reward,
        mode_reward=args.experimental.mode_reward,
    )
    test_dl, _ = dst.test_dataloader(
        args.langstr, max_len=1000000, min_len=2, device=device,
    )

    # Model
    basemodel = args.model.model_name.split("-")[0]
    model = LitXNPCFGFCReward(
        basemodel, args.model, dst.word_vocab.vocab_size,
        args.experimental, args.optimizer, args.langstr,
    )

    # Callbacks
    ts = time.strftime("%m%d_%H%M%S")
    ckpt_dir = f"ckpt/phase0_baseline/seed{cli.seed}_{ts}"
    os.makedirs(ckpt_dir, exist_ok=True)

    early_stop = EarlyStopping(
        monitor="val/sentence_f1", min_delta=0.002,
        patience=args.train.patience, verbose=False, mode="max",
    )
    ckpt_callback = ModelCheckpoint(
        save_top_k=1, monitor="val/sentence_f1", mode="max",
        dirpath=ckpt_dir, filename="best",
    )
    rl_scheduler = WarmupScheduler(
        warmup_steps=args.experimental.rl_warmup_steps,
        coeff_name="rl_coeff",
        initial_coeff=args.experimental.rl_initial_coeff,
        start_step=args.experimental.rl_start_step,
        target_coeff=args.experimental.rl_target_coeff,
    )
    maxent_scheduler = WarmupScheduler(
        warmup_steps=1, coeff_name="maxent_coeff",
        initial_coeff=args.experimental.maxent_initial_coeff,
        start_step=0,
        target_coeff=args.experimental.maxent_target_coeff,
    )

    wandb_logger = WandbLogger(
        project=WANDB_PROJECT, entity=WANDB_ENTITY,
        name=f"phase0-rank1-seminfo-seed{cli.seed}",
        log_model=False,
        tags=['phase0-v3', STUDY_NAME, 'rank1-seminfo', f'seed{cli.seed}'],
        config={
            'seed': cli.seed,
            'phase': 'phase0',
            'study_name': STUDY_NAME,
            'base_config': BASE_CONFIG,
            'model': dict(args.model),
            'optimizer': dict(args.optimizer),
            'experimental': dict(args.experimental),
        },
    )

    trainer = L.Trainer(
        max_steps=args.train.max_steps, min_steps=args.train.min_steps,
        val_check_interval=2000,
        check_val_every_n_epoch=None,
        gradient_clip_val=args.train.clip,
        gradient_clip_algorithm="norm",
        callbacks=[early_stop, ckpt_callback, rl_scheduler, maxent_scheduler],
        logger=[wandb_logger],
        inference_mode=False, log_every_n_steps=10,
        accelerator="gpu", devices=1,
        enable_progress_bar=False,
    )

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_val_sf1 = ckpt_callback.best_model_score
    print(f"\nBest val SF1: {best_val_sf1:.4f}")

    test_results = trainer.test(model, dataloaders=test_dl)
    test_sf1 = test_results[0].get('test/sentence_f1', 0.0) if test_results else 0.0
    print(f"Test SF1: {test_sf1:.4f}")

    wandb_logger.experiment.finish(quiet=True)


if __name__ == '__main__':
    main()
