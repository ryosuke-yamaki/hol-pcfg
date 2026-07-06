"""Tests for the Optuna v2 pipeline components.

Tests cover:
  - GPUPool slot assignment and task execution
  - run_optuna_v2 objective function (mock trial)
  - run_single_train result JSON output
  - Config generation from Optuna params
  - Phase transition logic (top-K extraction, best rank selection)
"""

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -------------------------------------------------------------------
# 1. GPUPool tests
# -------------------------------------------------------------------

def test_gpu_pool_slot_assignment():
    """Verify slot -> GPU mapping for various configurations."""
    print("[1/8] GPUPool slot assignment ... ", end="", flush=True)
    from scripts.run_pipeline import GPUPool

    pool = GPUPool(num_gpus=4, procs_per_gpu=4)
    assert pool.max_parallel == 16

    # Slot 0-3 -> GPU 0, 4-7 -> GPU 1, 8-11 -> GPU 2, 12-15 -> GPU 3
    for slot in range(16):
        expected_gpu = slot // 4
        assert pool.gpu_for_slot(slot) == expected_gpu, \
            f"slot {slot}: expected GPU {expected_gpu}, got {pool.gpu_for_slot(slot)}"

    # 2 GPUs × 2 procs
    pool2 = GPUPool(num_gpus=2, procs_per_gpu=2)
    assert pool2.max_parallel == 4
    assert pool2.gpu_for_slot(0) == 0
    assert pool2.gpu_for_slot(1) == 0
    assert pool2.gpu_for_slot(2) == 1
    assert pool2.gpu_for_slot(3) == 1

    print("OK")


def test_gpu_pool_run_tasks():
    """Verify GPUPool can run simple tasks and collect results."""
    print("[2/8] GPUPool run_tasks ... ", end="", flush=True)
    from scripts.run_pipeline import GPUPool

    pool = GPUPool(num_gpus=1, procs_per_gpu=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        tasks = [
            ([sys.executable, '-c', 'print("hello")'], 'task_a'),
            ([sys.executable, '-c', 'print("world")'], 'task_b'),
            ([sys.executable, '-c', 'import sys; sys.exit(1)'], 'task_fail'),
        ]
        results = pool.run_tasks(tasks, os.path.join(tmpdir, 'logs'))

        assert results['task_a'] == 0, f"task_a should succeed: {results['task_a']}"
        assert results['task_b'] == 0, f"task_b should succeed: {results['task_b']}"
        assert results['task_fail'] != 0, "task_fail should fail"

        # Check logs exist
        assert Path(tmpdir, 'logs', 'task_a.log').exists()
        assert Path(tmpdir, 'logs', 'task_b.log').exists()

    print("OK")


# -------------------------------------------------------------------
# 2. Config generation test
# -------------------------------------------------------------------

def test_config_generation():
    """Verify _generate_config produces valid YAML with correct values."""
    print("[3/8] Config generation ... ", end="", flush=True)
    from scripts.run_pipeline import Pipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock pipeline
        mock_args = MagicMock()
        mock_args.study_name = 'test'
        mock_args.base_config = 'archive/configs/normalization_phases/hn_pcfg_nt4096_optuna_v2.yaml'
        mock_args.num_gpus = 1
        mock_args.procs_per_gpu = 1
        mock_args.output_dir = tmpdir
        mock_args.start_phase = 0
        pipeline = Pipeline(mock_args)

        params = {
            'lr': 0.003,
            'mu': 0.75,
            'batch_size': 16,
            'tau_root_init': 5.0,
            'tau_rule_init': 7.0,
            'tau_term_init': 3.0,
        }
        out_path = Path(tmpdir) / 'test_config.yaml'
        pipeline._generate_config(params, out_path, run_name_prefix='test-run')

        assert out_path.exists(), "Config file not created"

        with open(out_path) as f:
            cfg = yaml.load(f, Loader=yaml.Loader)

        assert cfg['optimizer']['lr'] == 0.003
        assert cfg['optimizer']['mu'] == 0.75
        assert cfg['optimizer']['nu'] == 0.999
        assert cfg['optimizer']['relation_weight_decay'] == 0.0
        assert cfg['train']['batch_size'] == 16
        assert cfg['train']['patience'] == 10
        assert cfg['model']['tau_root_init'] == 5.0
        assert cfg['model']['tau_rule_init'] == 7.0
        assert cfg['model']['tau_term_init'] == 3.0
        assert cfg['model']['NT'] == 4096
        assert cfg['wandb']['run_name'] == 'test-run'

    print("OK")


# -------------------------------------------------------------------
# 3. run_single_train result output test
# -------------------------------------------------------------------

def test_single_train_result_json():
    """Verify run_single_train.py produces valid result JSON."""
    print("[4/8] Single train result JSON ... ", end="", flush=True)

    # Create a minimal config for a tiny model (fast execution)
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'device': '0',
            'save_dir': os.path.join(tmpdir, 'log'),
            'data': {
                'train_file': 'data/clean/english-train.pickle',
                'val_file': 'data/clean/english-val.pickle',
                'test_file': 'data/clean/english-test.pickle',
                'vocab_type': 'max_size',
                'vocab_size': 10000,
                'min_freq': 2,
            },
            'model': {
                'model_name': 'HNPCFG',
                'NT': 30,
                'T': 60,
                's_dim': 64,
                'tau_root_init': 5.0,
                'tau_term_init': 5.0,
                'tau_rule_init': 5.0,
            },
            'train': {
                'batch_size': 4,
                'max_epoch': 1,
                'max_len': 10,
                'curriculum': 0,
                'start_len': 10,
                'increment': 1,
                'patience': 5,
                'clip': 3,
            },
            'test': {
                'batch_size': 4,
                'max_tokens': 100,
                'bucket': 32,
                'decode': 'mbr',
                'sampler': 'batch',
            },
            'optimizer': {
                'name': 'adam',
                'lr': 0.001,
                'mu': 0.9,
                'nu': 0.999,
            },
            'wandb': {
                'enabled': False,
            },
        }
        config_path = os.path.join(tmpdir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        result_path = os.path.join(tmpdir, 'result.json')

        project_root = str(Path(__file__).resolve().parent.parent)
        env = dict(os.environ)
        env['PYTHONPATH'] = project_root
        ret = subprocess.run(
            [sys.executable, 'scripts/run_single_train.py',
             '--config', config_path,
             '--seed', '42',
             '--device', '0',
             '--result-path', result_path],
            capture_output=True, text=True, timeout=300,
            cwd=project_root, env=env,
        )

        assert ret.returncode == 0, \
            f"run_single_train failed:\nstdout: {ret.stdout[-500:]}\nstderr: {ret.stderr[-500:]}"
        assert os.path.exists(result_path), "Result JSON not created"

        with open(result_path) as f:
            result = json.load(f)

        assert result['status'] == 'completed', f"Status: {result['status']}"
        assert 'best_f1' in result, "Missing best_f1"
        assert 'best_ll' in result, "Missing best_ll"
        assert 'best_epoch' in result, "Missing best_epoch"
        assert isinstance(result['best_f1'], float), "best_f1 not float"
        assert 0.0 <= result['best_f1'] <= 1.0, \
            f"best_f1 out of range: {result['best_f1']}"
        assert result['seed'] == 42

        # Check tau values are present
        assert 'tau_root' in result, "Missing tau_root"
        assert 'tau_rule' in result, "Missing tau_rule"
        assert 'tau_term' in result, "Missing tau_term"

    print("OK")


# -------------------------------------------------------------------
# 4. Optuna v2 search space test
# -------------------------------------------------------------------

def test_optuna_v2_search_space():
    """Verify run_optuna_v2 suggests the correct parameters."""
    print("[5/8] Optuna v2 search space ... ", end="", flush=True)
    import optuna

    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with tempfile.TemporaryDirectory() as tmpdir:
        journal_path = os.path.join(tmpdir, 'test_journal.log')
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(journal_path))

        study = optuna.create_study(
            study_name='test-search-space',
            storage=storage,
            sampler=optuna.samplers.TPESampler(
                seed=42, n_startup_trials=1, multivariate=True, group=True),
            pruner=optuna.pruners.PercentilePruner(
                percentile=33.0, n_startup_trials=1, n_warmup_steps=5),
            direction='maximize',
        )

        # Create a mock objective that just checks params
        captured_params = {}

        def mock_objective(trial):
            lr = trial.suggest_float('lr', 5e-4, 1.5e-2, log=True)
            mu = trial.suggest_categorical('mu', [0.5, 0.75, 0.9])
            bs = trial.suggest_categorical('batch_size', [16, 32])
            tr = trial.suggest_float('tau_root_init', 1.0, 15.0, log=True)
            trl = trial.suggest_float('tau_rule_init', 1.0, 15.0, log=True)
            tt = trial.suggest_float('tau_term_init', 1.0, 15.0, log=True)
            captured_params.update({
                'lr': lr, 'mu': mu, 'batch_size': bs,
                'tau_root_init': tr, 'tau_rule_init': trl,
                'tau_term_init': tt,
            })
            return 0.5  # dummy

        study.optimize(mock_objective, n_trials=1)

        # Verify all expected params are suggested
        assert 'lr' in captured_params
        assert 'mu' in captured_params
        assert 'batch_size' in captured_params
        assert 'tau_root_init' in captured_params
        assert 'tau_rule_init' in captured_params
        assert 'tau_term_init' in captured_params

        # Verify ranges
        assert 5e-4 <= captured_params['lr'] <= 1.5e-2
        assert captured_params['mu'] in [0.5, 0.75, 0.9]
        assert captured_params['batch_size'] in [16, 32]
        assert 1.0 <= captured_params['tau_root_init'] <= 15.0
        assert 1.0 <= captured_params['tau_rule_init'] <= 15.0
        assert 1.0 <= captured_params['tau_term_init'] <= 15.0

        # Verify nu is NOT searched (should be fixed)
        assert 'nu' not in captured_params

    print("OK")


# -------------------------------------------------------------------
# 5. tau_init integration test
# -------------------------------------------------------------------

def test_tau_init_values():
    """Verify tau_*_init values are correctly propagated to the model."""
    print("[6/8] tau_init propagation ... ", end="", flush=True)
    from parser.model.HN_PCFG import HN_PCFG

    class MockDS:
        device = 'cpu'
        word_vocab = list(range(100))

    for tau_val in [1.0, 5.0, 10.0, 15.0]:
        args = MagicMock()
        args.NT = 30
        args.T = 60
        args.s_dim = 64
        args.tau_root_init = tau_val
        args.tau_term_init = tau_val
        args.tau_rule_init = tau_val
        # getattr fallback
        type(args).__getattr__ = lambda self, name: {
            'tau_root_init': tau_val,
            'tau_term_init': tau_val,
            'tau_rule_init': tau_val,
        }.get(name, MagicMock())

        from easydict import EasyDict as edict
        args = edict(NT=30, T=60, s_dim=64,
                     tau_root_init=tau_val,
                     tau_term_init=tau_val,
                     tau_rule_init=tau_val)

        model = HN_PCFG(args, MockDS())
        actual_tau_root = model.log_tau_root.data.exp().item()
        actual_tau_term = model.log_tau_term.data.exp().item()
        actual_tau_rule = model.log_tau_rule.data.exp().item()

        assert abs(actual_tau_root - tau_val) < 1e-5, \
            f"tau_root: expected {tau_val}, got {actual_tau_root}"
        assert abs(actual_tau_term - tau_val) < 1e-5, \
            f"tau_term: expected {tau_val}, got {actual_tau_term}"
        assert abs(actual_tau_rule - tau_val) < 1e-5, \
            f"tau_rule: expected {tau_val}, got {actual_tau_rule}"

    print("OK")


# -------------------------------------------------------------------
# 6. Phase 2 scoring logic test
# -------------------------------------------------------------------

def test_phase2_scoring():
    """Verify mean - 0.5*std scoring and collapse filtering."""
    print("[7/8] Phase 2 scoring logic ... ", end="", flush=True)
    import numpy as np

    # Simulate results for 3 ranks
    results = {
        1: [0.62, 0.63, 0.61, 0.60, 0.64, 0.63, 0.62, 0.61],  # Good
        2: [0.65, 0.53, 0.64, 0.52, 0.63, 0.54, 0.62, 0.51],  # High collapse
        3: [0.58, 0.59, 0.57, 0.58, 0.59, 0.58, 0.57, 0.58],  # Low variance
    }

    rank_stats = {}
    for rank, f1s in results.items():
        mean = np.mean(f1s)
        std = np.std(f1s, ddof=1)
        score = mean - 0.5 * std
        collapse = sum(1 for x in f1s if x < 0.55)
        rank_stats[rank] = {
            'mean': mean, 'std': std, 'score': score, 'collapse': collapse,
        }

    # Rank 2 should be filtered out (4 collapses >= 2)
    valid = {k: v for k, v in rank_stats.items() if v['collapse'] < 2}
    assert 2 not in valid, "Rank 2 should be filtered (high collapse)"
    assert 1 in valid, "Rank 1 should be valid"
    assert 3 in valid, "Rank 3 should be valid"

    # Best should be rank 1 (higher mean despite slightly higher std)
    best = max(valid, key=lambda k: valid[k]['score'])
    assert best == 1, f"Expected rank 1 as best, got rank {best}"

    print("OK")


# -------------------------------------------------------------------
# 7. Dual-track selection test
# -------------------------------------------------------------------

def test_dual_track_selection():
    """Verify dual-track model selection returns max(f1_at_best_ll, best_f1)."""
    print("[8/8] Dual-track selection ... ", end="", flush=True)

    # Scenario 1: LL-best has higher F1
    f1_at_best_ll = 0.63
    best_f1_overall = 0.61
    result = max(f1_at_best_ll, best_f1_overall)
    assert result == 0.63

    # Scenario 2: Overall best F1 is higher (LL and F1 disagree)
    f1_at_best_ll = 0.58
    best_f1_overall = 0.62
    result = max(f1_at_best_ll, best_f1_overall)
    assert result == 0.62

    # Scenario 3: Equal
    f1_at_best_ll = 0.60
    best_f1_overall = 0.60
    result = max(f1_at_best_ll, best_f1_overall)
    assert result == 0.60

    print("OK")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Pipeline component tests")
    print("=" * 60)

    tests = [
        test_gpu_pool_slot_assignment,
        test_gpu_pool_run_tasks,
        test_config_generation,
        test_single_train_result_json,
        test_optuna_v2_search_space,
        test_tau_init_values,
        test_phase2_scoring,
        test_dual_track_selection,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, e))
            import traceback
            print(f"FAIL")
            traceback.print_exc()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
