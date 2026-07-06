"""Re-evaluate multilingual_n7e2qm8t runs using the val/sentence_f1 best checkpoint.

For each run dir under ckpt/multilingual_n7e2qm8t/ whose name matches
<lang>_seed<N>_<ts>, this script:
  1. Reads `best_model_path` / `best_model_score` from checkpoint metadata
     (written by Lightning's ModelCheckpoint callback).
  2. Launches `python -m parsing_by_maxseminfo.train --eval_ckpt <path>` on the
     configured GPU, parses stderr for test metrics.
  3. Aggregates results into JSON at logs/multilingual_n7e2qm8t/eval_best_<tag>.json.

Usage:
    python scripts/eval_best_ckpts.py --seeds 1 2 3 4 5 --gpu 3 --tag seeds1_5
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

import torch

# ---------------------------------------------------------------------------
# Config mapping (mirrors scripts/run_multilingual_n7e2qm8t_hp.sh)
# ---------------------------------------------------------------------------
CONFIGS = {
    "chinese": "config/pas-grammar/chinese-ew-reward-tbtok-idf/hnpcfg_nt1024_t2048_n7e2qm8t_zh.yaml",
    "french":  "config/pas-grammar/french-ew-reward-tbtok-idf/hnpcfg_nt1024_t2048_n7e2qm8t_fr.yaml",
    "german":  "config/pas-grammar/german-ew-reward-tbtok-idf-vocab30k/hnpcfg_nt1024_t2048_n7e2qm8t_de.yaml",
}

CKPT_ROOT = Path("ckpt/multilingual_n7e2qm8t")
LOG_ROOT = Path("logs/multilingual_n7e2qm8t")

RUN_RE = re.compile(r"^(chinese|french|german)_seed(\d+)_(\d{4}_\d{6})$")


def find_best_ckpt(run_dir: Path) -> tuple[str, float]:
    """Read best_model_path / best_model_score from ModelCheckpoint callback state."""
    ckpt_sub = run_dir / "ckpt-sf1_val"
    any_ckpt = next(ckpt_sub.glob("*.ckpt"))
    state = torch.load(any_ckpt, map_location="cpu", weights_only=False)
    cbs = state.get("callbacks", {})
    mc_key = next((k for k in cbs if "ModelCheckpoint" in str(k)), None)
    if mc_key is None:
        raise RuntimeError(f"ModelCheckpoint callback not found in {any_ckpt}")
    mc_state = cbs[mc_key]
    best_path = mc_state["best_model_path"]
    best_score = mc_state["best_model_score"]
    if hasattr(best_score, "item"):
        best_score = float(best_score.item())
    return best_path, float(best_score)


def parse_test_metrics(stderr_text: str) -> dict:
    """Parse trainer.test stderr output for test/* metrics."""
    # The list-dict appears on the same line as "Eval-only mode with ckpt=...:"
    m = re.search(r"\[\{[^\[\]]*'test/[^\[\]]*\}\]", stderr_text)
    if not m:
        return {}
    try:
        parsed = ast.literal_eval(m.group(0))
        return parsed[0] if isinstance(parsed, list) and parsed else {}
    except (ValueError, SyntaxError):
        return {}


def discover_runs(seeds: list[int], langs: list[str] | None = None) -> list[dict]:
    runs = []
    target_langs = langs if langs else list(CONFIGS.keys())
    for lang in target_langs:
        for seed in seeds:
            matches = sorted(CKPT_ROOT.glob(f"{lang}_seed{seed}_*"))
            if not matches:
                print(f"[warn] no ckpt dir for {lang} seed={seed}", file=sys.stderr)
                continue
            run_dir = matches[-1]  # latest timestamp
            m = RUN_RE.match(run_dir.name)
            if not m:
                continue
            runs.append({"lang": lang, "seed": seed, "run_dir": str(run_dir)})
    return runs


def run_eval(run: dict, gpu: int, log_dir: Path) -> dict:
    lang = run["lang"]
    seed = run["seed"]
    run_dir = Path(run["run_dir"])
    conf = CONFIGS[lang]

    best_path, best_score = find_best_ckpt(run_dir)
    print(f"[{lang} seed={seed}] best_model_score={best_score:.4f} path={best_path}")

    log_file = log_dir / f"eval_best_{lang}_seed{seed}.log"
    cmd = [
        "python", "-m", "parsing_by_maxseminfo.train",
        "--conf", conf,
        "--langstr", lang,
        "--ckpt_dir", str(run_dir),  # required, unused in eval
        "--remark", f"eval-best-{lang}-seed{seed}",
        "--wandb_project", "hol-pcfg",
        "--wandb_entity", "ryosuke-yamaki",
        "--eval_ckpt", best_path,
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "WANDB_MODE": "disabled"}
    with open(log_file, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    with open(log_file) as f:
        combined = f.read()
    metrics = parse_test_metrics(combined)
    return {
        "lang": lang,
        "seed": seed,
        "run_dir": str(run_dir),
        "best_ckpt": best_path,
        "val_sent_f1_best": best_score,
        "test_metrics": metrics,
        "returncode": proc.returncode,
        "log_file": str(log_file),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--tag", type=str, default="seeds1_5")
    ap.add_argument("--langs", type=str, nargs="+", default=None,
                    help="Subset of languages to evaluate (default: all).")
    args = ap.parse_args()

    log_dir = LOG_ROOT
    log_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.seeds, args.langs)
    labels = [f"{r['lang']}-{r['seed']}" for r in runs]
    print(f"Discovered {len(runs)} runs: {labels}")

    results = []
    for r in runs:
        try:
            res = run_eval(r, args.gpu, log_dir)
        except Exception as e:
            res = {**r, "error": repr(e)}
        results.append(res)
        # incremental dump
        out_path = log_dir / f"eval_best_{args.tag}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  -> {res.get('test_metrics') or res.get('error')}")

    print(f"\nSaved aggregated results to {out_path}")


if __name__ == "__main__":
    main()
