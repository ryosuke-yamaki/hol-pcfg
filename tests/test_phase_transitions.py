"""Integration tests for automatic phase transitions.

Tests the full Phase 0 → 1 → 2 → 3 pipeline logic:
  - Phase 0 results aggregation
  - Phase 1 → 2: Optuna top-K extraction and config generation
  - Phase 2 → 3: best rank selection from multi-seed results
  - Phase 3: final result aggregation
  - Save-dir collision safety under concurrent launches
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optuna
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pipeline import Pipeline, PHASE2_TOP_K, PHASE2_SEEDS, PHASE3_SEEDS


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def make_pipeline(tmpdir: str, study_name: str = 'test-pipeline'):
    """Create a Pipeline instance with minimal args."""
    from unittest.mock import MagicMock
    args = MagicMock()
    args.study_name = study_name
    args.base_config = 'archive/configs/normalization_phases/hn_pcfg_nt4096_optuna_v2.yaml'
    args.num_gpus = 1
    args.procs_per_gpu = 1
    args.output_dir = tmpdir
    args.start_phase = 0
    return Pipeline(args)


def create_mock_result(path: Path, seed: int, f1: float,
                       status: str = 'completed'):
    """Create a mock result JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        'seed': seed,
        'config': 'test.yaml',
        'status': status,
        'best_f1': f1,
        'f1_at_best_ll': f1 - 0.005,
        'best_f1_overall': f1,
        'best_ll': -113.0,
        'best_epoch': 15,
        'total_epochs': 20,
        'tau_root': 14.0,
        'tau_rule': 21.0,
        'tau_term': 13.5,
    }
    with open(path, 'w') as f:
        json.dump(result, f)


def create_mock_optuna_study(journal_path: str, study_name: str,
                             n_completed: int = 10, n_pruned: int = 5):
    """Create a mock Optuna study with completed and pruned trials."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(journal_path))

    study = optuna.create_study(
        study_name=study_name, storage=storage,
        direction='maximize', load_if_exists=True,
    )

    # Add completed trials with varying F1 values
    for i in range(n_completed):
        f1 = 0.55 + i * 0.01 + np.random.uniform(-0.005, 0.005)
        study.add_trial(
            optuna.trial.create_trial(
                params={
                    'lr': 0.001 + i * 0.0005,
                    'mu': [0.5, 0.75, 0.9][i % 3],
                    'batch_size': [16, 32][i % 2],
                    'tau_root_init': 3.0 + i * 0.5,
                    'tau_rule_init': 4.0 + i * 0.3,
                    'tau_term_init': 5.0 + i * 0.4,
                },
                distributions={
                    'lr': optuna.distributions.FloatDistribution(5e-4, 1.5e-2, log=True),
                    'mu': optuna.distributions.CategoricalDistribution([0.5, 0.75, 0.9]),
                    'batch_size': optuna.distributions.CategoricalDistribution([16, 32]),
                    'tau_root_init': optuna.distributions.FloatDistribution(1.0, 15.0, log=True),
                    'tau_rule_init': optuna.distributions.FloatDistribution(1.0, 15.0, log=True),
                    'tau_term_init': optuna.distributions.FloatDistribution(1.0, 15.0, log=True),
                },
                values=[f1],
                state=optuna.trial.TrialState.COMPLETE,
            )
        )

    # Add pruned trials
    for i in range(n_pruned):
        study.add_trial(
            optuna.trial.create_trial(
                params={
                    'lr': 0.008 + i * 0.0005,
                    'mu': 0.9,
                    'batch_size': 32,
                    'tau_root_init': 1.0,
                    'tau_rule_init': 1.0,
                    'tau_term_init': 1.0,
                },
                distributions={
                    'lr': optuna.distributions.FloatDistribution(5e-4, 1.5e-2, log=True),
                    'mu': optuna.distributions.CategoricalDistribution([0.5, 0.75, 0.9]),
                    'batch_size': optuna.distributions.CategoricalDistribution([16, 32]),
                    'tau_root_init': optuna.distributions.FloatDistribution(1.0, 15.0, log=True),
                    'tau_rule_init': optuna.distributions.FloatDistribution(1.0, 15.0, log=True),
                    'tau_term_init': optuna.distributions.FloatDistribution(1.0, 15.0, log=True),
                },
                values=[None],
                state=optuna.trial.TrialState.PRUNED,
            )
        )

    return study


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_phase0_to_phase1_transition():
    """Verify Phase 0 results are aggregated correctly before Phase 1."""
    print("[1/6] Phase 0 → 1 transition ... ", end="", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = make_pipeline(tmpdir)
        result_dir = pipeline.output_dir / 'phase0' / 'results'

        # Create mock Phase 0 results (8 seeds)
        f1_values = [0.58, 0.63, 0.54, 0.53, 0.62, 0.60, 0.59, 0.61]
        for i, (seed, f1) in enumerate(zip([1,2,3,4,5,42,106,256], f1_values)):
            create_mock_result(result_dir / f"seed{seed}.json", seed, f1)

        # Verify summarization works (captures output)
        pipeline._summarize_phase('Phase 0', result_dir)

        # Verify all results are readable
        f1s = []
        for rp in sorted(result_dir.glob('*.json')):
            with open(rp) as f:
                r = json.load(f)
            f1s.append(r['best_f1'])

        assert len(f1s) == 8, f"Expected 8 results, got {len(f1s)}"
        mean_f1 = np.mean(f1s)
        assert 0.50 < mean_f1 < 0.70, f"Mean F1 out of range: {mean_f1}"

    print("OK")


def test_phase1_to_phase2_transition():
    """Verify top-K extraction from Optuna study and config generation."""
    print("[2/6] Phase 1 → 2 transition ... ", end="", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = make_pipeline(tmpdir, study_name='test-p1-p2')

        # Create mock Optuna study
        create_mock_optuna_study(
            pipeline.journal_path, pipeline.study_name,
            n_completed=10, n_pruned=5)

        # Test top-K extraction
        top_configs = pipeline._extract_top_k(PHASE2_TOP_K)
        assert len(top_configs) == PHASE2_TOP_K, \
            f"Expected {PHASE2_TOP_K} configs, got {len(top_configs)}"

        # Verify configs are sorted by value (descending)
        values = [v for _, _, v in top_configs]
        for i in range(len(values) - 1):
            assert values[i] >= values[i+1], \
                f"Configs not sorted: {values[i]} < {values[i+1]}"

        # Verify config generation
        config_dir = pipeline.output_dir / 'phase2' / 'configs'
        config_dir.mkdir(parents=True, exist_ok=True)
        for rank, (trial_num, params, value) in enumerate(top_configs, 1):
            config_path = config_dir / f"rank{rank}.yaml"
            pipeline._generate_config(params, config_path,
                                      run_name_prefix=f'test-rank{rank}')

            # Verify generated config
            with open(config_path) as f:
                cfg = yaml.load(f, Loader=yaml.Loader)

            assert cfg['optimizer']['lr'] == params['lr']
            assert cfg['optimizer']['mu'] == params['mu']
            assert cfg['train']['batch_size'] == params['batch_size']
            assert cfg['model']['tau_root_init'] == params['tau_root_init']
            assert cfg['model']['tau_rule_init'] == params['tau_rule_init']
            assert cfg['model']['tau_term_init'] == params['tau_term_init']
            # Fixed params
            assert cfg['optimizer']['nu'] == 0.999
            assert cfg['train']['patience'] == 10

        # Verify all 5 configs exist
        for rank in range(1, PHASE2_TOP_K + 1):
            assert (config_dir / f"rank{rank}.yaml").exists(), \
                f"rank{rank}.yaml not created"

    print("OK")


def test_phase2_to_phase3_transition():
    """Verify best rank selection from Phase 2 multi-seed results."""
    print("[3/6] Phase 2 → 3 transition ... ", end="", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = make_pipeline(tmpdir, study_name='test-p2-p3')
        result_dir = pipeline.output_dir / 'phase2' / 'results'

        # Create mock Phase 2 results: 5 ranks × 8 seeds
        rank_f1s = {
            1: [0.62, 0.63, 0.61, 0.60, 0.64, 0.63, 0.62, 0.61],  # Good
            2: [0.65, 0.53, 0.64, 0.52, 0.63, 0.54, 0.62, 0.51],  # Collapse
            3: [0.58, 0.59, 0.57, 0.58, 0.59, 0.58, 0.57, 0.58],  # Low var
            4: [0.60, 0.61, 0.59, 0.60, 0.62, 0.61, 0.60, 0.59],  # Medium
            5: [0.56, 0.57, 0.55, 0.56, 0.57, 0.56, 0.55, 0.56],  # Low
        }
        for rank, f1s in rank_f1s.items():
            for seed, f1 in zip(PHASE2_SEEDS, f1s):
                create_mock_result(
                    result_dir / f"rank{rank}_seed{seed}.json", seed, f1)

        # Run phase 2 summary
        best_rank = pipeline._summarize_phase2(result_dir)

        # Rank 2 has 4 collapses (<0.55) → should be excluded
        # Rank 1 has highest mean-0.5*std among valid ranks
        assert best_rank == 1, f"Expected best_rank=1, got {best_rank}"

        # Verify phase2_best.json is written
        best_info_path = pipeline.output_dir / 'phase2_best.json'
        assert best_info_path.exists(), "phase2_best.json not created"

        with open(best_info_path) as f:
            best_info = json.load(f)
        assert best_info['best_rank'] == 1

    print("OK")


def test_phase3_final_aggregation():
    """Verify Phase 3 final result aggregation."""
    print("[4/6] Phase 3 final aggregation ... ", end="", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = make_pipeline(tmpdir, study_name='test-p3')
        result_dir = pipeline.output_dir / 'phase3' / 'results'

        # Create mock Phase 3 results: 16 seeds
        np.random.seed(42)
        for seed in PHASE3_SEEDS:
            f1 = 0.63 + np.random.normal(0, 0.02)
            create_mock_result(
                result_dir / f"seed{seed}.json", seed, f1)

        # Run final summary
        pipeline._summarize_final(result_dir)

        # Verify final_result.json
        final_path = pipeline.output_dir / 'final_result.json'
        assert final_path.exists(), "final_result.json not created"

        with open(final_path) as f:
            final = json.load(f)
        assert final['n_seeds'] == 16
        assert 55.0 < final['mean_f1'] < 70.0
        assert 0.0 <= final['std_f1'] < 10.0

    print("OK")


def test_full_phase_chain():
    """End-to-end: Phase 0 → mock Phase 1 → Phase 2 → Phase 3."""
    print("[5/6] Full phase chain (mocked) ... ", end="", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = make_pipeline(tmpdir, study_name='test-full-chain')

        # --- Phase 0: create results ---
        result_dir_p0 = pipeline.output_dir / 'phase0' / 'results'
        for seed in [1,2,3,4,5,42,106,256]:
            create_mock_result(
                result_dir_p0 / f"seed{seed}.json", seed,
                0.58 + np.random.uniform(-0.03, 0.03))

        # --- Phase 1: create mock Optuna study ---
        study = create_mock_optuna_study(
            pipeline.journal_path, pipeline.study_name,
            n_completed=15, n_pruned=10)

        # --- Phase 1 → 2: extract top-K and generate configs ---
        top_configs = pipeline._extract_top_k(PHASE2_TOP_K)
        assert len(top_configs) == PHASE2_TOP_K

        config_dir = pipeline.output_dir / 'phase2' / 'configs'
        for rank, (trial_num, params, value) in enumerate(top_configs, 1):
            config_path = config_dir / f"rank{rank}.yaml"
            pipeline._generate_config(params, config_path)

        # --- Phase 2: create mock results ---
        result_dir_p2 = pipeline.output_dir / 'phase2' / 'results'
        np.random.seed(123)
        for rank in range(1, PHASE2_TOP_K + 1):
            base_f1 = 0.60 + rank * 0.005
            for seed in PHASE2_SEEDS:
                f1 = base_f1 + np.random.normal(0, 0.015)
                create_mock_result(
                    result_dir_p2 / f"rank{rank}_seed{seed}.json", seed, f1)

        # --- Phase 2 → 3: select best ---
        best_rank = pipeline._summarize_phase2(result_dir_p2)
        assert 1 <= best_rank <= PHASE2_TOP_K

        # Verify phase2_best.json exists and is used
        best_info_path = pipeline.output_dir / 'phase2_best.json'
        assert best_info_path.exists()
        with open(best_info_path) as f:
            best_info = json.load(f)
        assert best_info['best_rank'] == best_rank

        # Verify the best config file exists
        best_config = config_dir / f"rank{best_rank}.yaml"
        assert best_config.exists(), f"Best config not found: {best_config}"

        # --- Phase 3: create mock results ---
        result_dir_p3 = pipeline.output_dir / 'phase3' / 'results'
        for seed in PHASE3_SEEDS:
            f1 = 0.63 + np.random.normal(0, 0.02)
            create_mock_result(
                result_dir_p3 / f"seed{seed}.json", seed, f1)

        pipeline._summarize_final(result_dir_p3)

        final_path = pipeline.output_dir / 'final_result.json'
        assert final_path.exists()

    print("OK")


def test_savedir_collision_safety():
    """Verify that concurrent processes get distinct save directories."""
    print("[6/6] Save-dir collision safety ... ", end="", flush=True)

    project_root = str(Path(__file__).resolve().parent.parent)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal config
        config = {
            'device': '0',
            'save_dir': os.path.join(tmpdir, 'log'),
            'data': {
                'train_file': 'data/clean/english-train.pickle',
                'val_file': 'data/clean/english-val.pickle',
                'test_file': 'data/clean/english-test.pickle',
                'vocab_type': 'max_size', 'vocab_size': 10000, 'min_freq': 2,
            },
            'model': {
                'model_name': 'HNPCFG', 'NT': 30, 'T': 60, 's_dim': 64,
            },
            'train': {
                'batch_size': 4, 'max_epoch': 1, 'max_len': 10,
                'curriculum': 0, 'start_len': 10, 'increment': 1,
                'patience': 5, 'clip': 3,
            },
            'test': {
                'batch_size': 4, 'max_tokens': 100, 'bucket': 32,
                'decode': 'mbr', 'sampler': 'batch',
            },
            'optimizer': {'name': 'adam', 'lr': 0.001, 'mu': 0.9, 'nu': 0.999},
            'wandb': {'enabled': False},
        }
        config_path = os.path.join(tmpdir, 'test_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Launch 4 concurrent processes with the SAME config and seed
        env = dict(os.environ)
        env['PYTHONPATH'] = project_root
        procs = []
        result_paths = []
        for i in range(4):
            rp = os.path.join(tmpdir, f'result_{i}.json')
            result_paths.append(rp)
            proc = subprocess.Popen(
                [sys.executable, 'scripts/run_single_train.py',
                 '--config', config_path,
                 '--seed', '42',  # Same seed for all!
                 '--device', '0',
                 '--result-path', rp],
                env=env, cwd=project_root,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            procs.append(proc)

        # Wait for all to complete
        for proc in procs:
            ret = proc.wait()
            assert ret == 0, f"Process exited with code {ret}"

        # Verify all 4 produced results (no crash from dir collision)
        for rp in result_paths:
            assert os.path.exists(rp), f"Result not created: {rp}"
            with open(rp) as f:
                r = json.load(f)
            assert r['status'] == 'completed', f"Status: {r['status']}"

        # Verify the save_dirs are all different.
        # run_single_train uses config's save_dir as base:
        #   {save_dir}/pipeline/{config_stem}_seed{seed}_{ts}_pid{pid}
        save_dirs = set()
        log_base = os.path.join(tmpdir, 'log', 'pipeline')
        assert os.path.exists(log_base), \
            f"Pipeline log dir not created: {log_base}"
        for d in os.listdir(log_base):
            save_dirs.add(d)
        # 4 processes should have created 4 distinct directories (different PIDs)
        assert len(save_dirs) == 4, \
            f"Expected 4 distinct save_dirs, got {len(save_dirs)}: {save_dirs}"

    print("OK")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Phase transition integration tests")
    print("=" * 60)

    tests = [
        test_phase0_to_phase1_transition,
        test_phase1_to_phase2_transition,
        test_phase2_to_phase3_transition,
        test_phase3_final_aggregation,
        test_full_phase_chain,
        test_savedir_collision_safety,
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
