#!/usr/bin/env python3
"""Aggregate Phase 2 or Phase 3 worker JSONs into a single summary file.

Usage:
    python scripts/aggregate_phase2.py [STUDY_NAME] [--phase {2,3}]

Phase 2 (default): merges Top-5 x N seeds into ranked summary with mean/std.
Phase 3: merges Best x N seeds into a single config summary.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def aggregate_phase2(study_name: str, log_dir: str = "logs") -> dict:
    pattern = f"optuna_{study_name}_phase2_worker*.json"
    files = sorted(Path(log_dir).glob(pattern))
    if not files:
        print(f"No worker result files found matching {log_dir}/{pattern}")
        sys.exit(1)

    merged: dict = {}
    print(f"Found {len(files)} Phase 2 worker files:")
    for f in files:
        print(f"  {f}")
        data = json.loads(f.read_text())
        for key, val in data.items():
            if key not in merged:
                merged[key] = val
            else:
                merged[key]['phase2'].extend(val['phase2'])

    for key in merged:
        merged[key]['phase2'].sort(key=lambda r: r['seed'])

    print("\n" + "=" * 60)
    print("PHASE 2 RESULTS SUMMARY")
    print("=" * 60)
    for key in sorted(merged.keys()):
        data = merged[key]
        val_scores = [r['val_sf1'] for r in data['phase2']]
        test_scores = [r['test_sf1'] for r in data['phase2']]
        print(f"\n{key} (Phase1 SF1={data['phase1_sf1']:.4f}):")
        print(f"  Val  SF1: {np.mean(val_scores):.4f} +/- {np.std(val_scores):.4f} "
              f"[{np.min(val_scores):.4f}, {np.max(val_scores):.4f}]")
        print(f"  Test SF1: {np.mean(test_scores):.4f} +/- {np.std(test_scores):.4f} "
              f"[{np.min(test_scores):.4f}, {np.max(test_scores):.4f}]")
        print(f"  Seeds: {[r['seed'] for r in data['phase2']]}")
        print(f"  Params: {data['params']}")

    out_path = f"{log_dir}/optuna_{study_name}_phase2_results.json"
    Path(out_path).write_text(json.dumps(merged, indent=2, default=str))
    print(f"\nMerged results saved to: {out_path}")
    return merged


def aggregate_phase3(study_name: str, log_dir: str = "logs") -> dict:
    pattern = f"optuna_{study_name}_phase3_worker*.json"
    files = sorted(Path(log_dir).glob(pattern))
    if not files:
        print(f"No worker result files found matching {log_dir}/{pattern}")
        sys.exit(1)

    # Phase 3 worker JSONs each have {'params': ..., 'source_key': ..., 'phase3': [...]}
    # All workers share the same params (Best config from Phase 2); merge phase3 entries.
    merged = {'params': None, 'source_key': None, 'phase3': []}
    print(f"Found {len(files)} Phase 3 worker files:")
    for f in files:
        print(f"  {f}")
        data = json.loads(f.read_text())
        if merged['params'] is None:
            merged['params'] = data.get('params')
            merged['source_key'] = data.get('source_key')
        merged['phase3'].extend(data.get('phase3', []))
    merged['phase3'].sort(key=lambda r: r['seed'])

    val_scores = [r['val_sf1'] for r in merged['phase3']]
    test_scores = [r['test_sf1'] for r in merged['phase3']]
    print("\n" + "=" * 60)
    print("PHASE 3 RESULTS SUMMARY")
    print("=" * 60)
    print(f"Source (Phase 2 best): {merged['source_key']}")
    print(f"Params: {merged['params']}")
    print(f"Seeds: {[r['seed'] for r in merged['phase3']]}")
    print(f"Val  SF1: {np.mean(val_scores):.4f} +/- {np.std(val_scores):.4f} "
          f"[{np.min(val_scores):.4f}, {np.max(val_scores):.4f}]")
    print(f"Test SF1: {np.mean(test_scores):.4f} +/- {np.std(test_scores):.4f} "
          f"[{np.min(test_scores):.4f}, {np.max(test_scores):.4f}]")

    out_path = f"{log_dir}/optuna_{study_name}_phase3_results.json"
    Path(out_path).write_text(json.dumps(merged, indent=2, default=str))
    print(f"\nMerged results saved to: {out_path}")
    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument('study_name', nargs='?', default='hnpcfg-rank1-seminfo-v3')
    p.add_argument('--phase', type=int, choices=[2, 3], default=2)
    p.add_argument('--log-dir', default='logs')
    cli = p.parse_args()

    if cli.phase == 2:
        aggregate_phase2(cli.study_name, cli.log_dir)
    else:
        aggregate_phase3(cli.study_name, cli.log_dir)


if __name__ == '__main__':
    main()
