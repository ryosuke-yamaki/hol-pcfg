"""Re-evaluate v3 Optuna trial192 runs using best.ckpt (val/sentence_f1-best).

Scope: Phase 2 rank2 (trial192, seeds 1-5) and Phase 3 (trial192 HPs, seeds 6-10).
The original pipeline reported test/sentence_f1 computed from end-of-training
weights (trainer.test with ckpt_path=None), so we re-run trainer.test with
ckpt_path=<best.ckpt> to obtain the best-ckpt test metrics.

Authoritative phase2 rank2 dirs were resolved by matching worker Optuna-init
timestamps in logs/optuna_hnpcfg-rank1-seminfo-v3/phase2_w{5..9}.log
(17:35:41 / 46 / 52 / 57 / 36:02) against ckpt dir names.

Usage:
    python scripts/eval_best_v3_trial192.py --gpu 3
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

BASE_CONFIG = "config/pas-grammar/english-ew-reward-tbtok-idf/hnpcfg_nt1024_t2048_rank1_seminfo.yaml"

CKPT_ROOT = Path("ckpt/optuna/hnpcfg-rank1-seminfo-v3")

# Phase 2 rank2 (trial192), seeds 1-5.
# Each (rank=2, seed) has TWO sibling dirs in this study (a duplicated launch
# overlap during phase2). The earlier-timestamp dirs are the completed runs
# whose best_model_score matches the corresponding worker JSON / W&B run.
# Mapping verified by full-precision val_sf1 match against
# logs/optuna_hnpcfg-rank1-seminfo-v3_phase2_worker{5,6,7,8,9}.json.
PHASE2_RANK2 = {
    1: "phase2_rank2_seed1_0417_173520",  # worker5, val_sf1=0.6851522923
    2: "phase2_rank2_seed2_0417_173525",  # worker6, val_sf1=0.6913194656
    3: "phase2_rank2_seed3_0417_173530",  # worker7, val_sf1=0.6813588142
    4: "phase2_rank2_seed4_0417_173535",  # worker8, val_sf1=0.6821818352
    5: "phase2_rank2_seed5_0417_173539",  # worker9 = W&B run n7e2qm8t, val_sf1=0.6918381453
}

# Phase 3 (trial192 HPs applied to new seeds 6-10) — single dir per seed.
PHASE3 = {
    6:  "phase3_seed6_0419_021927",
    7:  "phase3_seed7_0419_021932",
    8:  "phase3_seed8_0419_021937",
    9:  "phase3_seed9_0419_021942",
    10: "phase3_seed10_0419_021947",
}

LOG_DIR = Path("logs/optuna_hnpcfg-rank1-seminfo-v3_eval_best_trial192")
OUT_JSON = Path("logs/optuna_hnpcfg-rank1-seminfo-v3_eval_best_trial192.json")


def parse_test_metrics(text: str) -> dict:
    m = re.search(r"\[\{[^\[\]]*'test/[^\[\]]*\}\]", text)
    if not m:
        return {}
    try:
        parsed = ast.literal_eval(m.group(0))
        return parsed[0] if isinstance(parsed, list) and parsed else {}
    except (ValueError, SyntaxError):
        return {}


def parse_val_metrics(text: str) -> dict:
    m = re.search(r"\[\{[^\[\]]*'val/[^\[\]]*\}\]", text)
    if not m:
        return {}
    try:
        parsed = ast.literal_eval(m.group(0))
        return parsed[0] if isinstance(parsed, list) and parsed else {}
    except (ValueError, SyntaxError):
        return {}


def run_eval(phase: str, seed: int, run_dir: Path, gpu: int) -> dict:
    best_path = run_dir / "best.ckpt"
    if not best_path.exists():
        return {"phase": phase, "seed": seed, "run_dir": str(run_dir),
                "error": f"missing {best_path}"}

    tag = f"{phase}_seed{seed}"
    log_file = LOG_DIR / f"{tag}.log"
    dummy_ckpt_dir = Path(f"/tmp/eval_v3_{tag}_{os.getpid()}")
    dummy_ckpt_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "parsing_by_maxseminfo.train",
        "--conf", BASE_CONFIG,
        "--langstr", "english",
        "--ckpt_dir", str(dummy_ckpt_dir),
        "--remark", f"eval-v3-trial192-{tag}",
        "--wandb_project", "hol-pcfg",
        "--wandb_entity", "ryosuke-yamaki",
        "--eval_ckpt", str(best_path),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "WANDB_MODE": "disabled"}
    with open(log_file, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    combined = log_file.read_text()
    test_metrics = parse_test_metrics(combined)
    val_metrics = parse_val_metrics(combined)
    return {
        "phase": phase,
        "seed": seed,
        "run_dir": str(run_dir),
        "best_ckpt": str(best_path),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "returncode": proc.returncode,
        "log_file": str(log_file),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = (
        [("phase2_rank2", s, CKPT_ROOT / PHASE2_RANK2[s]) for s in sorted(PHASE2_RANK2)]
        + [("phase3", s, CKPT_ROOT / PHASE3[s]) for s in sorted(PHASE3)]
    )
    print(f"Evaluating {len(jobs)} ckpts on GPU {args.gpu}")

    results = []
    for phase, seed, run_dir in jobs:
        print(f"[{phase} seed={seed}] {run_dir.name}", flush=True)
        res = run_eval(phase, seed, run_dir, args.gpu)
        tm = res.get("test_metrics") or {}
        vm = res.get("val_metrics") or {}
        print(
            f"  -> val_sf1={vm.get('val/sentence_f1')} test_sf1={tm.get('test/sentence_f1')} "
            f"(rc={res.get('returncode')})",
            flush=True,
        )
        results.append(res)
        OUT_JSON.write_text(json.dumps(results, indent=2))

    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
