#!/usr/bin/env python3
"""Automated HP search pipeline for HN-PCFG.

Runs all phases sequentially:
  Phase 0: Baseline validation (8 seeds with current best HP)
  Phase 1: Optuna HP search (100 trials, 16 parallel workers)
  Phase 2: Seed validation (top-5 configs × 8 seeds)
  Phase 3: Final validation (best config × 16 seeds)

Usage:
    python scripts/run_pipeline.py \\
        --study-name hnpcfg-v2-hp \\
        --num-gpus 4 --procs-per-gpu 4

    # Resume from a specific phase:
    python scripts/run_pipeline.py --start-phase 2 ...
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from easydict import EasyDict as edict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PHASE0_SEEDS = [1, 2, 3, 4, 5, 42, 106, 256]
PHASE1_TARGET_TRIALS = 100
PHASE1_EXTENSION_THRESHOLD = 25  # Extend if completed < this
PHASE1_EXTENSION_AMOUNT = 50
PHASE2_TOP_K = 5
PHASE2_SEEDS = list(range(1, 9))   # 8 seeds
PHASE3_SEEDS = list(range(1, 17))  # 16 seeds
BASE_CONFIG = 'archive/configs/normalization_phases/hn_pcfg_nt4096_optuna_v2.yaml'

POLL_INTERVAL = 15  # seconds
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


# ---------------------------------------------------------------------------
# GPU Pool: manages parallel subprocesses across GPUs
# ---------------------------------------------------------------------------
class GPUPool:
    """Manages parallel subprocess execution across multiple GPUs.

    Slots are numbered 0..(num_gpus * procs_per_gpu - 1).
    Slot i is assigned to GPU (i // procs_per_gpu).
    """

    def __init__(self, num_gpus: int, procs_per_gpu: int):
        self.num_gpus = num_gpus
        self.procs_per_gpu = procs_per_gpu
        self.max_parallel = num_gpus * procs_per_gpu
        self._slots: dict[int, tuple[subprocess.Popen, str, Path]] = {}

    def gpu_for_slot(self, slot: int) -> int:
        return slot // self.procs_per_gpu

    def _find_free_slot(self) -> int:
        for i in range(self.max_parallel):
            if i not in self._slots:
                return i
        return -1

    def _poll_completed(self) -> list[tuple[str, int]]:
        """Poll active processes. Returns list of (task_key, returncode)."""
        completed = []
        for slot in list(self._slots.keys()):
            proc, task_key, _ = self._slots[slot]
            ret = proc.poll()
            if ret is not None:
                completed.append((task_key, ret))
                del self._slots[slot]
        return completed

    def run_tasks(
        self,
        tasks: list[tuple[list[str], str]],
        log_dir: str | Path,
    ) -> dict[str, int]:
        """Run tasks in parallel across GPUs.

        Args:
            tasks: list of (command, task_key) pairs.
                   command is a list of strings for subprocess.
                   task_key is a unique string identifier.
            log_dir: directory for stdout/stderr logs.

        Returns:
            dict mapping task_key -> returncode (0 = success).
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        queue = list(tasks)
        results: dict[str, int] = {}
        total = len(queue)

        print(f"  Scheduling {total} tasks across "
              f"{self.num_gpus} GPUs × {self.procs_per_gpu} procs "
              f"= {self.max_parallel} slots")

        while queue or self._slots:
            # Fill free slots from queue
            submitted = 0
            while queue:
                slot = self._find_free_slot()
                if slot < 0:
                    break
                cmd, task_key = queue.pop(0)
                gpu = self.gpu_for_slot(slot)
                env = dict(os.environ)
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                env['PYTHONPATH'] = PROJECT_ROOT
                log_path = log_dir / f"{task_key}.log"
                log_f = open(log_path, 'w')
                proc = subprocess.Popen(
                    cmd, env=env, cwd=PROJECT_ROOT,
                    stdout=log_f, stderr=subprocess.STDOUT,
                )
                self._slots[slot] = (proc, task_key, log_path)
                submitted += 1

            if submitted > 0:
                done = len(results)
                active = len(self._slots)
                remaining = len(queue)
                print(f"  [{done}/{total} done, {active} active, "
                      f"{remaining} queued]")

            # Poll for completions
            for task_key, ret in self._poll_completed():
                status = "OK" if ret == 0 else f"FAIL(rc={ret})"
                results[task_key] = ret
                done = len(results)
                print(f"  [{done}/{total}] {task_key}: {status}")

            if self._slots:
                time.sleep(POLL_INTERVAL)

        n_ok = sum(1 for v in results.values() if v == 0)
        n_fail = len(results) - n_ok
        print(f"  All tasks complete: {n_ok} succeeded, {n_fail} failed")
        return results

    def terminate_all(self):
        """Kill all active processes (for cleanup on error)."""
        for slot in list(self._slots.keys()):
            proc, task_key, _ = self._slots[slot]
            proc.terminate()
            print(f"  Terminated: {task_key}")
        self._slots.clear()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self, args):
        self.study_name = args.study_name
        self.base_config = args.base_config
        self.num_gpus = args.num_gpus
        self.procs_per_gpu = args.procs_per_gpu
        self.max_parallel = args.num_gpus * args.procs_per_gpu
        self.output_dir = Path(args.output_dir) / self.study_name
        self.start_phase = args.start_phase
        self.pool = GPUPool(args.num_gpus, args.procs_per_gpu)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = str(self.output_dir / 'optuna_journal.log')

    # -----------------------------------------------------------------------
    # Main entry
    # -----------------------------------------------------------------------
    def run(self):
        print("=" * 70)
        print(f"HN-PCFG HP Search Pipeline: {self.study_name}")
        print(f"  GPUs: {self.num_gpus} × {self.procs_per_gpu} = "
              f"{self.max_parallel} parallel")
        print(f"  Output: {self.output_dir}")
        print(f"  Base config: {self.base_config}")
        print("=" * 70)

        try:
            if self.start_phase <= 0:
                self._run_phase0()
            if self.start_phase <= 1:
                self._run_phase1()
            if self.start_phase <= 2:
                self._run_phase2()
            if self.start_phase <= 3:
                self._run_phase3()
        except KeyboardInterrupt:
            print("\n\nPipeline interrupted. Cleaning up...")
            self.pool.terminate_all()
            sys.exit(1)

        print("\n" + "=" * 70)
        print("Pipeline complete!")
        print("=" * 70)

    # -----------------------------------------------------------------------
    # Phase 0: Baseline validation
    # -----------------------------------------------------------------------
    def _run_phase0(self):
        phase_dir = self.output_dir / 'phase0'
        result_dir = phase_dir / 'results'
        log_dir = phase_dir / 'logs'

        print(f"\n{'='*70}")
        print("Phase 0: Baseline validation")
        print(f"  Seeds: {PHASE0_SEEDS}")
        print(f"{'='*70}")

        tasks = []
        for seed in PHASE0_SEEDS:
            result_path = result_dir / f"seed{seed}.json"
            if result_path.exists():
                print(f"  Skipping seed {seed} (result exists)")
                continue
            task_key = f"phase0_seed{seed}"
            cmd = [
                sys.executable, 'scripts/run_single_train.py',
                '--config', self.base_config,
                '--seed', str(seed),
                '--result-path', str(result_path),
                '--wandb-name',
                f'{self.study_name}-phase0-seed{seed}',
                '--wandb-tags', 'phase0', self.study_name,
            ]
            tasks.append((cmd, task_key))

        if tasks:
            self.pool.run_tasks(tasks, log_dir)

        # Summarize
        self._summarize_phase('Phase 0', result_dir)

    # -----------------------------------------------------------------------
    # Phase 1: Optuna HP search
    # -----------------------------------------------------------------------
    def _run_phase1(self):
        log_dir = self.output_dir / 'phase1' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print("Phase 1: Optuna HP search")
        print(f"  Target trials: {PHASE1_TARGET_TRIALS}")
        print(f"  Workers: {self.max_parallel}")
        print(f"  Journal: {self.journal_path}")
        print(f"{'='*70}")

        trials_per_worker = math.ceil(
            PHASE1_TARGET_TRIALS / self.max_parallel)

        self._launch_optuna_workers(trials_per_worker, log_dir, round_num=1)

        # Check extension criteria
        n_completed = self._count_completed_trials()
        print(f"\n  Phase 1 result: {n_completed} completed trials")

        if n_completed < PHASE1_EXTENSION_THRESHOLD:
            print(f"  Completed < {PHASE1_EXTENSION_THRESHOLD}: "
                  f"extending by {PHASE1_EXTENSION_AMOUNT} trials")
            ext_per_worker = math.ceil(
                PHASE1_EXTENSION_AMOUNT / self.max_parallel)
            self._launch_optuna_workers(ext_per_worker, log_dir, round_num=2)
            n_completed = self._count_completed_trials()
            print(f"  After extension: {n_completed} completed trials")

        # Print top-5 summary
        self._summarize_optuna()

    def _launch_optuna_workers(self, trials_per_worker: int,
                                log_dir: Path, round_num: int):
        """Launch Optuna workers in parallel."""
        tasks = []
        for worker_id in range(self.max_parallel):
            gpu_id = worker_id // self.procs_per_gpu
            task_key = f"optuna_r{round_num}_w{worker_id}"
            cmd = [
                sys.executable, 'scripts/run_optuna_v2.py',
                '--device', str(gpu_id),
                '--worker-id', str(worker_id),
                '--study-name', self.study_name,
                '--base-config', self.base_config,
                '--n-trials', str(trials_per_worker),
                '--journal-path', self.journal_path,
                '--training-seed', '42',
                '--startup-delay', '2.0',
            ]
            tasks.append((cmd, task_key))

        self.pool.run_tasks(tasks, log_dir)

    def _count_completed_trials(self) -> int:
        import optuna
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(self.journal_path)
        )
        study = optuna.load_study(
            study_name=self.study_name, storage=storage)
        return len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])

    def _summarize_optuna(self):
        import optuna
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(self.journal_path)
        )
        study = optuna.load_study(
            study_name=self.study_name, storage=storage)

        complete = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in study.trials
                  if t.state == optuna.trial.TrialState.PRUNED]

        print(f"\n  Trials: {len(study.trials)} total, "
              f"{len(complete)} completed, {len(pruned)} pruned")

        if complete:
            sorted_trials = sorted(complete, key=lambda t: t.value,
                                   reverse=True)
            print(f"\n  Top-5 configs:")
            for i, t in enumerate(sorted_trials[:5]):
                p = t.params
                print(f"    Rank {i+1} (trial #{t.number}): "
                      f"F1={t.value:.4f} | "
                      f"lr={p['lr']:.2e} mu={p['mu']} bs={p['batch_size']} "
                      f"tr={p['tau_root_init']:.2f} "
                      f"trl={p['tau_rule_init']:.2f} "
                      f"tt={p['tau_term_init']:.2f}")

    # -----------------------------------------------------------------------
    # Phase 2: Seed validation
    # -----------------------------------------------------------------------
    def _run_phase2(self):
        phase_dir = self.output_dir / 'phase2'
        config_dir = phase_dir / 'configs'
        result_dir = phase_dir / 'results'
        log_dir = phase_dir / 'logs'
        config_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print("Phase 2: Seed validation")
        print(f"  Top-{PHASE2_TOP_K} × {len(PHASE2_SEEDS)} seeds")
        print(f"{'='*70}")

        # Extract top-K configs from Optuna study
        top_configs = self._extract_top_k(PHASE2_TOP_K)
        if not top_configs:
            print("  ERROR: No completed trials in Phase 1!")
            sys.exit(1)

        # Generate config files for each rank
        config_paths = {}
        for rank, (trial_num, params, value) in enumerate(top_configs, 1):
            config_path = config_dir / f"rank{rank}.yaml"
            self._generate_config(params, config_path,
                                  run_name_prefix=f'{self.study_name}-p2-rank{rank}')
            config_paths[rank] = config_path
            print(f"  Rank {rank} (trial #{trial_num}, F1={value:.4f}): "
                  f"{config_path}")

        # Create tasks: rank × seed
        tasks = []
        for rank, config_path in config_paths.items():
            for seed in PHASE2_SEEDS:
                result_path = result_dir / f"rank{rank}_seed{seed}.json"
                if result_path.exists():
                    print(f"  Skipping rank{rank}/seed{seed} (result exists)")
                    continue
                task_key = f"p2_rank{rank}_seed{seed}"
                cmd = [
                    sys.executable, 'scripts/run_single_train.py',
                    '--config', str(config_path),
                    '--seed', str(seed),
                    '--result-path', str(result_path),
                    '--wandb-name',
                    f'{self.study_name}-p2-rank{rank}-seed{seed}',
                    '--wandb-tags', 'phase2', self.study_name,
                    f'rank{rank}',
                ]
                tasks.append((cmd, task_key))

        if tasks:
            self.pool.run_tasks(tasks, log_dir)

        # Summarize and select best
        best_rank = self._summarize_phase2(result_dir)
        return best_rank

    def _extract_top_k(self, k: int) -> list[tuple[int, dict, float]]:
        """Extract top-K trial configs from Optuna study."""
        import optuna
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(self.journal_path)
        )
        study = optuna.load_study(
            study_name=self.study_name, storage=storage)
        complete = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            return []
        sorted_trials = sorted(complete, key=lambda t: t.value,
                               reverse=True)
        return [
            (t.number, dict(t.params), t.value)
            for t in sorted_trials[:k]
        ]

    def _generate_config(self, params: dict, output_path: Path,
                         run_name_prefix: str = ''):
        """Generate a YAML config file from Optuna params."""
        with open(self.base_config) as f:
            cfg = yaml.load(f, Loader=yaml.Loader)

        cfg['optimizer']['lr'] = params['lr']
        cfg['optimizer']['mu'] = params['mu']
        cfg['optimizer']['nu'] = 0.999
        cfg['optimizer']['relation_weight_decay'] = 0.0
        cfg['train']['batch_size'] = params['batch_size']
        cfg['train']['patience'] = 10
        cfg['train']['clip'] = 3
        cfg['model']['tau_root_init'] = params['tau_root_init']
        cfg['model']['tau_rule_init'] = params['tau_rule_init']
        cfg['model']['tau_term_init'] = params['tau_term_init']

        if run_name_prefix:
            cfg['wandb']['run_name'] = run_name_prefix

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

    def _summarize_phase2(self, result_dir: Path) -> int:
        """Summarize Phase 2 results. Returns best rank."""
        print(f"\n  Phase 2 Results:")
        print(f"  {'Rank':<6} {'Mean F1':>10} {'Std':>8} "
              f"{'Mean-0.5*Std':>14} {'Collapse':>10}")
        print(f"  {'-'*52}")

        rank_stats = {}
        for rank in range(1, PHASE2_TOP_K + 1):
            f1s = []
            for seed in PHASE2_SEEDS:
                rp = result_dir / f"rank{rank}_seed{seed}.json"
                if rp.exists():
                    with open(rp) as f:
                        r = json.load(f)
                    if r.get('status') == 'completed':
                        f1s.append(r['best_f1'])
            if f1s:
                mean_f1 = np.mean(f1s)
                std_f1 = np.std(f1s, ddof=1) if len(f1s) > 1 else 0
                score = mean_f1 - 0.5 * std_f1
                n_collapse = sum(1 for x in f1s if x < 0.55)
                rank_stats[rank] = {
                    'mean': mean_f1, 'std': std_f1,
                    'score': score, 'collapse': n_collapse,
                    'n': len(f1s),
                }
                print(f"  Rank {rank:<4} {mean_f1*100:>9.2f}% "
                      f"{std_f1*100:>7.2f}% {score*100:>13.2f}% "
                      f"{n_collapse:>6}/{len(f1s)}")

        if not rank_stats:
            print("  No results available!")
            return 1

        # Select best: highest score with collapse < 2
        valid = {k: v for k, v in rank_stats.items() if v['collapse'] < 2}
        if not valid:
            print("  WARNING: All configs have >= 2 collapses. "
                  "Using best score regardless.")
            valid = rank_stats

        best_rank = max(valid, key=lambda k: valid[k]['score'])
        best = valid[best_rank]
        print(f"\n  Best: Rank {best_rank} "
              f"(mean={best['mean']*100:.2f}%, std={best['std']*100:.2f}%)")

        # Save best rank info
        info_path = self.output_dir / 'phase2_best.json'
        with open(info_path, 'w') as f:
            json.dump({'best_rank': best_rank, 'stats': rank_stats}, f,
                      indent=2, default=str)
        return best_rank

    # -----------------------------------------------------------------------
    # Phase 3: Final validation
    # -----------------------------------------------------------------------
    def _run_phase3(self):
        phase_dir = self.output_dir / 'phase3'
        result_dir = phase_dir / 'results'
        log_dir = phase_dir / 'logs'

        print(f"\n{'='*70}")
        print("Phase 3: Final validation")
        print(f"  {len(PHASE3_SEEDS)} seeds")
        print(f"{'='*70}")

        # Load best config from Phase 2
        best_info_path = self.output_dir / 'phase2_best.json'
        if not best_info_path.exists():
            print("  ERROR: Phase 2 results not found!")
            sys.exit(1)

        with open(best_info_path) as f:
            best_info = json.load(f)
        best_rank = best_info['best_rank']

        config_path = (self.output_dir / 'phase2' / 'configs'
                       / f'rank{best_rank}.yaml')
        print(f"  Using Phase 2 best: Rank {best_rank} ({config_path})")

        # Create tasks
        tasks = []
        for seed in PHASE3_SEEDS:
            result_path = result_dir / f"seed{seed}.json"
            if result_path.exists():
                print(f"  Skipping seed {seed} (result exists)")
                continue
            task_key = f"p3_seed{seed}"
            cmd = [
                sys.executable, 'scripts/run_single_train.py',
                '--config', str(config_path),
                '--seed', str(seed),
                '--result-path', str(result_path),
                '--wandb-name',
                f'{self.study_name}-p3-best-seed{seed}',
                '--wandb-tags', 'phase3', self.study_name, 'final',
            ]
            tasks.append((cmd, task_key))

        if tasks:
            self.pool.run_tasks(tasks, log_dir)

        # Final summary
        self._summarize_final(result_dir)

    def _summarize_final(self, result_dir: Path):
        """Print final results for Phase 3."""
        f1s = []
        for seed in PHASE3_SEEDS:
            rp = result_dir / f"seed{seed}.json"
            if rp.exists():
                with open(rp) as f:
                    r = json.load(f)
                if r.get('status') == 'completed':
                    f1s.append(r['best_f1'])
                    print(f"    seed {seed}: F1={r['best_f1']*100:.2f}%")

        if f1s:
            mean_f1 = np.mean(f1s) * 100
            std_f1 = np.std(f1s, ddof=1) * 100 if len(f1s) > 1 else 0
            n_collapse = sum(1 for x in f1s if x < 0.55)

            print(f"\n  {'='*50}")
            print(f"  FINAL RESULT: {mean_f1:.2f} +/- {std_f1:.2f} "
                  f"({len(f1s)} seeds)")
            print(f"  Collapse: {n_collapse}/{len(f1s)}")
            print(f"  Target:   65.10 +/- 2.10 (SN-PCFG)")

            if mean_f1 >= 63.0 and std_f1 <= 3.0:
                print(f"  Status:   MINIMUM TARGET MET")
            if mean_f1 >= 65.0:
                print(f"  Status:   TARGET MET")
            print(f"  {'='*50}")

            final_path = self.output_dir / 'final_result.json'
            with open(final_path, 'w') as f:
                json.dump({
                    'mean_f1': mean_f1, 'std_f1': std_f1,
                    'n_seeds': len(f1s), 'n_collapse': n_collapse,
                    'individual_f1s': f1s,
                }, f, indent=2)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _summarize_phase(self, phase_name: str, result_dir: Path):
        """Print summary statistics for a set of result JSON files."""
        f1s = []
        for rp in sorted(result_dir.glob('*.json')):
            with open(rp) as f:
                r = json.load(f)
            if r.get('status') == 'completed':
                f1s.append(r['best_f1'])
                print(f"    seed {r['seed']}: F1={r['best_f1']*100:.2f}%")

        if f1s:
            mean_f1 = np.mean(f1s) * 100
            std_f1 = np.std(f1s, ddof=1) * 100 if len(f1s) > 1 else 0
            print(f"\n  {phase_name} Summary: "
                  f"{mean_f1:.2f} +/- {std_f1:.2f} ({len(f1s)} seeds)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='HN-PCFG HP search pipeline (all phases)')
    parser.add_argument('--study-name', default='hnpcfg-v2-hp',
                        help='Study name (used for output dir and W&B)')
    parser.add_argument('--base-config', default=BASE_CONFIG,
                        help='Base YAML config path')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='Number of GPUs')
    parser.add_argument('--procs-per-gpu', type=int, default=4,
                        help='Max processes per GPU')
    parser.add_argument('--output-dir', default='runs/optuna_v2',
                        help='Output directory')
    parser.add_argument('--start-phase', type=int, default=0,
                        choices=[0, 1, 2, 3],
                        help='Phase to start from (for resuming)')
    args = parser.parse_args()

    pipeline = Pipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()
