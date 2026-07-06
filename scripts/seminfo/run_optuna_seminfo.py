#!/usr/bin/env python3
"""Optuna HP search v3 for post-refactor HN-PCFG + Sem-Info.

Pipeline:
  Phase 0 (separate runner): rank1-seminfo x 5 seeds baseline (scripts/run_phase0_baseline.py)
  Phase 1: TPE exploration, 200 trials, fixed training seed=42 (this script)
  Phase 2: Top-5 x 5 seeds (1-5) for config selection (this script --phase2-only)
  Phase 3: Best x 5 new seeds (6-10) for final validation (this script --phase3-only)

Key differences vs v2:
  * 3 independent taus (tau_root_init / tau_rule_init / tau_term_init) with log-uniform [1.0, 20.0]
  * num_samples restricted to [2, 4] (drop 8 to avoid OOM-biased pruning)
  * Per-worker TPE seed = 42 + device_id*4 + worker_subid (prevents intra-GPU TPE collision)
  * OOM guard around trainer.fit -> TrialPruned
  * Phase 3 (Best x 5 new seeds) added
  * --smoke-test for 5k-step sanity check

Usage:
    # Single worker
    python scripts/run_optuna_seminfo.py --device 0 --worker-subid 0

    # Multi-GPU multi-worker
    bash scripts/launch_optuna_4gpu.sh
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import optuna
import yaml
from easydict import EasyDict as edict

# Add repo root (which contains the parsing_by_maxseminfo package) to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from parsing_by_maxseminfo import parser as seminfo_parser
sys.modules['parser'] = seminfo_parser

import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from parser.lightning_wrapper.LitNPCFG import LitXNPCFGFCReward
from parser.lightning_wrapper.scheduler import WarmupScheduler
from parser.helper.pas_grammar_data_helper import DataModuleForPASCtrlPCFGReward


# ============================================================
# Constants
# ============================================================
BASE_CONFIG = "config/seminfo/hnpcfg_nt1024_t2048_rank1_seminfo.yaml"
STUDY_NAME = "hnpcfg-rank1-seminfo-v3"
JOURNAL_PATH = f"optuna_{STUDY_NAME}_journal.log"
WANDB_PROJECT = "hol-pcfg"
# W&B entity is not hardcoded: defaults to the WANDB_ENTITY env var (None -> use
# the logged-in default entity). Override with --wandb_entity.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")

# Training schedule
MAX_STEPS = 100000
MIN_STEPS = 30000
VAL_CHECK_INTERVAL = 2000

# Pipeline sizes
PHASE1_N_TRIALS = 200
PHASE2_TOP_K = 5
PHASE2_SEEDS = [1, 2, 3, 4, 5]
PHASE3_SEEDS = [6, 7, 8, 9, 10]

TRAIN_SEED_PHASE1 = 42  # fixed across all Phase 1 trials so TPE sees HP signal

WORKERS_PER_GPU = 4  # default worker layout; per-worker TPE seed = 42 + device*4 + subid


# ============================================================
# Helpers
# ============================================================
def load_base_config(path: str) -> edict:
    with open(path) as f:
        return edict(yaml.load(f, Loader=yaml.Loader))


def build_data_module(args):
    return DataModuleForPASCtrlPCFGReward(
        hparams=args,
        langstr=args.langstr,
        use_cache=True,
        max_size=10000,
        merge_pas_data=False,
        pas_subsample=getattr(args, 'preprocessing_pas_subsample_count', 0),
        flag_use_pos_unks=getattr(args.experimental, 'flag_use_pos_unks', False),
    )


def build_dataloaders(dst, args, device):
    train_dl, _ = dst.train_dataloader(
        args.langstr,
        max_len=getattr(args, 'max_length', 40),
        min_len=3,
        device=device,
        pas_subsample_count=args.experimental.pas_subsample_count,
        flag_curriculum_learning=getattr(args.experimental, 'flag_curriculum_learning', False),
        add_sentence_level_span=getattr(args.experimental, 'add_sentence_level_span', False),
        min_span_reward=args.experimental.min_span_reward,
        mode_reward=getattr(args.experimental, 'mode_reward', 'log_tfidf'),
        supervised_mode=getattr(args.experimental, 'supervised_mode', False),
    )
    val_dl, _ = dst.dev_full_dataloader(
        args.langstr, max_len=100000, min_len=2, device=device,
        min_span_reward=args.experimental.min_span_reward,
        mode_reward=getattr(args.experimental, 'mode_reward', 'log_tfidf'),
    )
    test_dl, _ = dst.test_dataloader(
        args.langstr, max_len=1000000, min_len=2, device=device,
    )
    return train_dl, val_dl, test_dl


def apply_hp_overrides(args: edict, hp: dict) -> None:
    """Write HP values into the edict config (post-refactor safe).

    The post-refactor HN_PCFG.py reads args.model.{tau_root_init, tau_rule_init,
    tau_term_init}. It does NOT read args.model.tau_init, so we never set it.
    """
    args.optimizer.lr = hp['lr']
    args.optimizer.mu = hp['mu']
    args.model.tau_root_init = hp['tau_root_init']
    args.model.tau_rule_init = hp['tau_rule_init']
    args.model.tau_term_init = hp['tau_term_init']
    args.experimental.rl_warmup_steps = hp['rl_warmup_steps']
    args.experimental.maxent_initial_coeff = hp['maxent_coeff']
    args.experimental.maxent_target_coeff = hp['maxent_coeff']
    args.experimental.num_samples = hp['num_samples']
    # Lock Sem-Info training regime (even if base config is switched later)
    args.experimental.mode = 'rl'
    args.experimental.mode_reward = 'log_tfidf'
    args.experimental.sample_mode = 'crf'


def _is_oom(exc: BaseException) -> bool:
    """Detect CUDA OOM across PyTorch versions (2.x has torch.cuda.OutOfMemoryError)."""
    if isinstance(exc, getattr(torch.cuda, 'OutOfMemoryError', ())):  # type: ignore[arg-type]
        return True
    msg = str(exc).lower()
    return 'out of memory' in msg or 'cuda' in msg and 'oom' in msg


# ============================================================
# Single-trial training (used by Phase 2 and Phase 3)
# ============================================================
def run_single_trial(args, ckpt_dir: str, remark: str, device_id: int,
                     wandb_tags=None, seed=None,
                     max_steps: int = MAX_STEPS, min_steps: int = MIN_STEPS,
                     wandb_project: str = WANDB_PROJECT,
                     extra_wandb_config: dict | None = None):
    """Run a Sem-Info training run. Returns (best_val_sf1, test_sf1)."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    if seed is not None:
        L.seed_everything(seed, workers=True)
    device = torch.device('cuda')

    dst = build_data_module(args)
    train_dl, val_dl, test_dl = build_dataloaders(dst, args, device)

    basemodel = args.model.model_name.split("-")[0]
    model = LitXNPCFGFCReward(
        basemodel, args.model, dst.word_vocab.vocab_size,
        args.experimental, args.optimizer, args.langstr,
    )

    early_stop = EarlyStopping(
        monitor="val/sentence_f1", min_delta=0.002,
        patience=args.train.patience, verbose=False, mode="max",
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_callback = ModelCheckpoint(
        save_top_k=1, monitor="val/sentence_f1", mode="max",
        dirpath=ckpt_dir, filename="best",
    )
    rl_scheduler = WarmupScheduler(
        warmup_steps=getattr(args.experimental, 'rl_warmup_steps', 5000),
        coeff_name="rl_coeff",
        initial_coeff=getattr(args.experimental, 'rl_initial_coeff', 0.0),
        start_step=getattr(args.experimental, 'rl_start_step', 0),
        target_coeff=getattr(args.experimental, 'rl_target_coeff', 1.0),
    )
    maxent_scheduler = WarmupScheduler(
        warmup_steps=getattr(args.experimental, 'maxent_warmup_steps', 1),
        coeff_name="maxent_coeff",
        initial_coeff=getattr(args.experimental, 'maxent_initial_coeff', -0.01),
        start_step=getattr(args.experimental, 'maxent_start_step', 0),
        target_coeff=getattr(args.experimental, 'maxent_target_coeff', -0.01),
    )

    wandb_config = {
        'model': dict(args.model),
        'optimizer': dict(args.optimizer),
        'experimental': dict(args.experimental),
        'base_config': BASE_CONFIG,
        'study_name': STUDY_NAME,
    }
    if extra_wandb_config:
        wandb_config.update(extra_wandb_config)

    wandb_logger = WandbLogger(
        project=wandb_project, entity=WANDB_ENTITY,
        name=remark, log_model=False,
        tags=wandb_tags or [],
        config=wandb_config,
    )

    trainer = L.Trainer(
        max_steps=max_steps, min_steps=min_steps,
        val_check_interval=VAL_CHECK_INTERVAL,
        check_val_every_n_epoch=None,
        gradient_clip_val=args.train.clip,
        gradient_clip_algorithm="norm",
        callbacks=[early_stop, ckpt_callback, rl_scheduler, maxent_scheduler],
        logger=[wandb_logger],
        inference_mode=False, log_every_n_steps=10,
        accelerator="gpu", devices=1,
        enable_progress_bar=False,
    )

    try:
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    except RuntimeError as e:
        if _is_oom(e):
            print(f"[OOM] {remark}: {e}", flush=True)
            wandb_logger.experiment.finish(quiet=True)
            del model, trainer
            torch.cuda.empty_cache()
            raise
        raise

    best_val_sf1 = ckpt_callback.best_model_score
    best_val_sf1 = best_val_sf1.item() if best_val_sf1 is not None else 0.0

    test_results = trainer.test(model, dataloaders=test_dl)
    test_sf1 = test_results[0].get('test/sentence_f1', 0.0) if test_results else 0.0

    wandb_logger.experiment.finish(quiet=True)
    del model, trainer
    torch.cuda.empty_cache()
    return best_val_sf1, test_sf1


# ============================================================
# Optuna pruning callback
# ============================================================
class OptunaPruningCallback(L.Callback):
    def __init__(self, trial: optuna.Trial, monitor: str = "val/sentence_f1"):
        self.trial = trial
        self.monitor = monitor
        self.step_count = 0

    def on_validation_end(self, trainer, pl_module):
        val_sf1 = trainer.callback_metrics.get(self.monitor)
        if val_sf1 is not None:
            self.step_count += 1
            self.trial.report(val_sf1.item(), self.step_count)
            if self.trial.should_prune():
                raise optuna.TrialPruned()


# ============================================================
# Phase 1: objective (TPE exploration)
# ============================================================
def objective(trial: optuna.Trial, device_id: int,
              max_steps: int = MAX_STEPS, min_steps: int = MIN_STEPS,
              wandb_project: str = WANDB_PROJECT) -> float:
    args = load_base_config(BASE_CONFIG)
    args.langstr = getattr(args, 'langstr', 'english')

    # HP suggestion (8D)
    hp = {
        'lr':              trial.suggest_float('lr', 3e-4, 1e-2, log=True),
        'tau_root_init':   trial.suggest_float('tau_root_init', 1.0, 20.0, log=True),
        'tau_rule_init':   trial.suggest_float('tau_rule_init', 1.0, 20.0, log=True),
        'tau_term_init':   trial.suggest_float('tau_term_init', 1.0, 20.0, log=True),
        'mu':              trial.suggest_categorical('mu', [0.5, 0.75, 0.9, 0.95]),
        'num_samples':     trial.suggest_categorical('num_samples', [2, 4]),
        'rl_warmup_steps': trial.suggest_categorical('rl_warmup_steps', [1000, 3000, 5000, 8000]),
        'maxent_coeff':    trial.suggest_categorical('maxent_coeff', [-0.001, -0.005, -0.01, -0.02]),
    }
    apply_hp_overrides(args, hp)

    # Phase 1: training seed is fixed so TPE sees HP signal, not seed noise
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    L.seed_everything(TRAIN_SEED_PHASE1, workers=True)
    device = torch.device('cuda')

    ts = time.strftime("%m%d_%H%M%S")
    ckpt_dir = f"ckpt/optuna/{STUDY_NAME}/trial{trial.number:04d}_{ts}_gpu{device_id}"
    remark = (f"optuna-t{trial.number}-lr{hp['lr']:.1e}-mu{hp['mu']}-"
              f"tr{hp['tau_root_init']:.1f}-tu{hp['tau_rule_init']:.1f}-"
              f"tt{hp['tau_term_init']:.1f}-ns{hp['num_samples']}-"
              f"rw{hp['rl_warmup_steps']}-me{hp['maxent_coeff']}")

    dst = build_data_module(args)
    train_dl, val_dl, _ = build_dataloaders(dst, args, device)

    basemodel = args.model.model_name.split("-")[0]
    model = LitXNPCFGFCReward(
        basemodel, args.model, dst.word_vocab.vocab_size,
        args.experimental, args.optimizer, args.langstr,
    )

    early_stop = EarlyStopping(
        monitor="val/sentence_f1", min_delta=0.002,
        patience=args.train.patience, verbose=False, mode="max",
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_callback = ModelCheckpoint(
        save_top_k=1, monitor="val/sentence_f1", mode="max",
        dirpath=ckpt_dir, filename="best",
    )
    rl_scheduler = WarmupScheduler(
        warmup_steps=hp['rl_warmup_steps'], coeff_name="rl_coeff",
        initial_coeff=0.0, start_step=0, target_coeff=1.0,
    )
    maxent_scheduler = WarmupScheduler(
        warmup_steps=1, coeff_name="maxent_coeff",
        initial_coeff=hp['maxent_coeff'], start_step=0,
        target_coeff=hp['maxent_coeff'],
    )
    optuna_cb = OptunaPruningCallback(trial)

    wandb_logger = WandbLogger(
        project=wandb_project, entity=WANDB_ENTITY,
        name=remark, log_model=False,
        tags=['optuna-v3', STUDY_NAME, 'phase1', f'gpu{device_id}', f'trial{trial.number}'],
        config={
            'phase': 'phase1',
            'trial': trial.number,
            'train_seed': TRAIN_SEED_PHASE1,
            'hp': hp,
            'study_name': STUDY_NAME,
            'base_config': BASE_CONFIG,
            'model': dict(args.model),
            'experimental': dict(args.experimental),
        },
    )

    trainer = L.Trainer(
        max_steps=max_steps, min_steps=min_steps,
        val_check_interval=VAL_CHECK_INTERVAL,
        check_val_every_n_epoch=None,
        gradient_clip_val=args.train.clip,
        gradient_clip_algorithm="norm",
        callbacks=[early_stop, ckpt_callback, rl_scheduler,
                   maxent_scheduler, optuna_cb],
        logger=[wandb_logger],
        inference_mode=False, log_every_n_steps=10,
        accelerator="gpu", devices=1,
        enable_progress_bar=False,
    )

    try:
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    except optuna.TrialPruned:
        wandb_logger.experiment.finish(quiet=True)
        del model, trainer
        torch.cuda.empty_cache()
        raise
    except RuntimeError as e:
        if _is_oom(e):
            print(f"[OOM] trial {trial.number}: pruning. {e}", flush=True)
            wandb_logger.experiment.finish(quiet=True)
            del model, trainer
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        raise

    best_val_sf1 = ckpt_callback.best_model_score
    result = best_val_sf1.item() if best_val_sf1 is not None else 0.0

    wandb_logger.experiment.finish(quiet=True)
    del model, trainer
    torch.cuda.empty_cache()
    return result


# ============================================================
# Phase 2: Top-K x 5 seeds verification
# ============================================================
def run_phase2(study: optuna.Study, device_id: int,
               worker_id: int = 0, n_workers: int = 1):
    print("\n" + "=" * 60)
    print(f"PHASE 2: Seed Verification (worker {worker_id}/{n_workers})")
    print("=" * 60)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    top_trials = completed[:PHASE2_TOP_K]

    print(f"Top-{PHASE2_TOP_K} trials from Phase 1:")
    for i, t in enumerate(top_trials):
        print(f"  Rank {i+1}: Trial #{t.number}, SF1={t.value:.4f}, params={t.params}")

    all_jobs = [(rank, seed, trial)
                for rank, trial in enumerate(top_trials)
                for seed in PHASE2_SEEDS]
    my_jobs = [j for i, j in enumerate(all_jobs) if i % n_workers == worker_id]
    print(f"Total jobs: {len(all_jobs)}, this worker: {len(my_jobs)}")

    results = {}
    for rank, seed, trial in my_jobs:
        hp = trial.params
        args = load_base_config(BASE_CONFIG)
        args.langstr = getattr(args, 'langstr', 'english')
        apply_hp_overrides(args, hp)

        ts = time.strftime("%m%d_%H%M%S")
        ckpt_dir = f"ckpt/optuna/{STUDY_NAME}/phase2_rank{rank+1}_seed{seed}_{ts}"
        remark = f"phase2-rank{rank+1}-seed{seed}"

        print(f"\n  [Rank {rank+1}, Seed {seed}] Starting...", flush=True)
        val_sf1, test_sf1 = run_single_trial(
            args, ckpt_dir, remark, device_id,
            wandb_tags=['optuna-v3', STUDY_NAME, 'phase2', f'rank{rank+1}', f'seed{seed}'],
            seed=seed,
            extra_wandb_config={'phase': 'phase2', 'rank': rank + 1,
                                'trial_number': trial.number, 'hp': hp},
        )

        key = f"rank{rank+1}_trial{trial.number}"
        if key not in results:
            results[key] = {'params': hp, 'phase1_sf1': trial.value, 'phase2': []}
        results[key]['phase2'].append({'seed': seed, 'val_sf1': val_sf1, 'test_sf1': test_sf1})
        print(f"  [Rank {rank+1}, Seed {seed}] val_SF1={val_sf1:.4f}, test_SF1={test_sf1:.4f}", flush=True)

    results_path = f"logs/optuna_{STUDY_NAME}_phase2_worker{worker_id}.json"
    os.makedirs("logs", exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWorker {worker_id} results saved to: {results_path}")
    return results


# ============================================================
# Phase 3: Best config x 5 new seeds
# ============================================================
def _pick_best_config_from_phase2(study_name: str) -> dict:
    """Aggregate Phase 2 worker JSONs -> pick config with max mean(test_sf1)."""
    log_dir = Path("logs")
    pattern = f"optuna_{study_name}_phase2_worker*.json"
    files = sorted(log_dir.glob(pattern))
    if not files:
        raise RuntimeError(
            f"No Phase 2 worker results matching {pattern}. "
            "Run Phase 2 (and aggregate_phase2.py) first.")

    merged = {}
    for f in files:
        data = json.loads(f.read_text())
        for key, val in data.items():
            if key not in merged:
                merged[key] = val
            else:
                merged[key]['phase2'].extend(val['phase2'])

    def mean_test(d):
        tests = [r['test_sf1'] for r in d['phase2']]
        return sum(tests) / max(len(tests), 1)

    best_key = max(merged, key=lambda k: mean_test(merged[k]))
    print(f"[Phase 3] Best config: {best_key}, mean_test_sf1={mean_test(merged[best_key]):.4f}")
    return {'key': best_key, **merged[best_key]}


def run_phase3(device_id: int, worker_id: int = 0, n_workers: int = 1):
    print("\n" + "=" * 60)
    print(f"PHASE 3: Final Validation (worker {worker_id}/{n_workers})")
    print("=" * 60)

    best = _pick_best_config_from_phase2(STUDY_NAME)
    hp = best['params']

    all_jobs = [(seed,) for seed in PHASE3_SEEDS]
    my_jobs = [j for i, j in enumerate(all_jobs) if i % n_workers == worker_id]
    print(f"Total jobs: {len(all_jobs)}, this worker: {len(my_jobs)}")
    print(f"Best HP: {hp}")

    results = {'params': hp, 'source_key': best['key'],
               'phase2_mean_test_sf1': None, 'phase3': []}
    for (seed,) in my_jobs:
        args = load_base_config(BASE_CONFIG)
        args.langstr = getattr(args, 'langstr', 'english')
        apply_hp_overrides(args, hp)

        ts = time.strftime("%m%d_%H%M%S")
        ckpt_dir = f"ckpt/optuna/{STUDY_NAME}/phase3_seed{seed}_{ts}"
        remark = f"phase3-best-seed{seed}"

        print(f"\n  [Phase 3, Seed {seed}] Starting...", flush=True)
        val_sf1, test_sf1 = run_single_trial(
            args, ckpt_dir, remark, device_id,
            wandb_tags=['optuna-v3', STUDY_NAME, 'phase3', 'best', f'seed{seed}'],
            seed=seed,
            extra_wandb_config={'phase': 'phase3', 'hp': hp,
                                'source_key': best['key']},
        )
        results['phase3'].append({'seed': seed, 'val_sf1': val_sf1, 'test_sf1': test_sf1})
        print(f"  [Phase 3, Seed {seed}] val_SF1={val_sf1:.4f}, test_SF1={test_sf1:.4f}", flush=True)

    results_path = f"logs/optuna_{STUDY_NAME}_phase3_worker{worker_id}.json"
    os.makedirs("logs", exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWorker {worker_id} Phase 3 results saved to: {results_path}")
    return results


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(description='Optuna v3 for post-refactor HN-PCFG + Sem-Info')
    p.add_argument('--device', '-d', type=int, default=0, help='GPU device ID')
    p.add_argument('--worker-subid', type=int, default=0,
                   help='Per-GPU worker sub-id (0..WORKERS_PER_GPU-1). '
                        'TPE seed = 42 + device*4 + worker_subid.')
    p.add_argument('--n-trials', type=int, default=PHASE1_N_TRIALS,
                   help='Trials per worker invocation')
    p.add_argument('--study-name', default=STUDY_NAME)
    p.add_argument('--journal-path', default=JOURNAL_PATH)
    p.add_argument('--wandb_entity', default=None,
                   help='W&B entity (default: WANDB_ENTITY env var, else the '
                        'logged-in default entity).')
    p.add_argument('--startup-delay', type=float, default=5.0,
                   help='Per-worker startup delay (legacy; launch script handles scheduling)')
    p.add_argument('--phase2-only', action='store_true')
    p.add_argument('--skip-phase2', action='store_true')
    p.add_argument('--phase2-worker-id', type=int, default=0)
    p.add_argument('--phase2-n-workers', type=int, default=1)
    p.add_argument('--phase3-only', action='store_true')
    p.add_argument('--skip-phase3', action='store_true')
    p.add_argument('--phase3-worker-id', type=int, default=0)
    p.add_argument('--phase3-n-workers', type=int, default=1)
    p.add_argument('--smoke-test', action='store_true',
                   help='5k-step sanity check: MAX_STEPS=5000, MIN_STEPS=2000, '
                        'n_trials=1, project=hol-pcfg-smoke')
    cli = p.parse_args()

    if cli.wandb_entity is not None:
        globals()['WANDB_ENTITY'] = cli.wandb_entity

    device_id = cli.device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

    # Smoke test overrides
    if cli.smoke_test:
        globals()['MAX_STEPS'] = 5000
        globals()['MIN_STEPS'] = 2000
        wandb_project = "hol-pcfg-smoke"
        n_trials = 1
        print(f"[SMOKE TEST] MAX_STEPS=5000, MIN_STEPS=2000, n_trials=1, "
              f"wandb_project={wandb_project}")
    else:
        wandb_project = WANDB_PROJECT
        n_trials = cli.n_trials

    # Per-worker TPE seed (fix intra-GPU TPE state collision)
    worker_id_global = device_id * WORKERS_PER_GPU + cli.worker_subid
    tpe_seed = 42 + worker_id_global
    print(f"[GPU {device_id} / sub {cli.worker_subid}] TPE seed = {tpe_seed}")

    # Only Phase 1 needs sampler/pruner; Phase 2/3 just read study params
    study_kwargs = {
        'study_name': cli.study_name,
        'storage': optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(cli.journal_path)
        ),
        'direction': 'maximize',
        'load_if_exists': True,
    }

    if not cli.phase2_only and not cli.phase3_only:
        study_kwargs['sampler'] = optuna.samplers.TPESampler(
            seed=tpe_seed,
            constant_liar=True,
            n_startup_trials=20,
            multivariate=True,
            group=True,
        )
        study_kwargs['pruner'] = optuna.pruners.PercentilePruner(
            percentile=25.0,
            n_startup_trials=20,
            n_warmup_steps=10,  # 10 * 2000 = 20k training steps before pruning kicks in
        )

    # Stagger worker startup a bit to avoid directory name collisions
    if not cli.phase2_only and not cli.phase3_only:
        delay = worker_id_global * cli.startup_delay
        if delay > 0:
            print(f"[GPU {device_id} / sub {cli.worker_subid}] "
                  f"Waiting {delay:.0f}s startup delay...")
            time.sleep(delay)

    study = optuna.create_study(**study_kwargs)
    print(f"[GPU {device_id} / sub {cli.worker_subid}] "
          f"Study '{cli.study_name}', existing trials: {len(study.trials)}")

    # Phase 1
    if not cli.phase2_only and not cli.phase3_only:
        print(f"\n[GPU {device_id} / sub {cli.worker_subid}] "
              f"Starting Phase 1: {n_trials} trials")
        study.optimize(
            lambda trial: objective(trial, device_id,
                                    max_steps=MAX_STEPS,
                                    min_steps=MIN_STEPS,
                                    wandb_project=wandb_project),
            n_trials=n_trials,
        )
        if device_id == 0 and cli.worker_subid == 0:
            completed = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE]
            pruned = [t for t in study.trials
                      if t.state == optuna.trial.TrialState.PRUNED]
            print(f"\nPhase 1 running total: {len(completed)} completed, {len(pruned)} pruned")
            if study.best_trial is not None:
                print(f"Best trial so far: #{study.best_trial.number}")
                print(f"Best SF1: {study.best_trial.value:.4f}")
                print(f"Best params: {study.best_trial.params}")

    # Phase 2 (can be reached from --phase2-only OR as chained step after Phase 1)
    if not cli.skip_phase2 and (cli.phase2_only or
                                (not cli.phase3_only and device_id == 0 and cli.worker_subid == 0)):
        run_phase2(study, device_id,
                   worker_id=cli.phase2_worker_id,
                   n_workers=cli.phase2_n_workers)

    # Phase 3
    if cli.phase3_only and not cli.skip_phase3:
        run_phase3(device_id,
                   worker_id=cli.phase3_worker_id,
                   n_workers=cli.phase3_n_workers)


if __name__ == '__main__':
    main()
