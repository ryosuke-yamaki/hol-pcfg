#!/usr/bin/env python3
"""Faithful single-run reproducer for the HN-PCFG (SemInfo) English group.

Given a source W&B run id, pull its exact HP / seed / rank / trial / phase from
the run config + tags and re-train via run_optuna_seminfo.run_single_trial (the
exact fn used by both native Phase 2 and Phase 3). Training inputs (HP, seed,
data, code path) are identical; only the unused Optuna-journal lookup is skipped
(the committed journal is an empty stub).

  python repro_driver.py --device 1 --run-id g2djnx6f
"""
import argparse
import time
import sys
from pathlib import Path
import importlib.util

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location(
    "ros", str(Path(__file__).resolve().parent / "run_optuna_seminfo.py")
)
ros = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ros)

# Keys consumed by ros.apply_hp_overrides (the Optuna search space).
HP_KEYS = ["lr", "mu", "tau_root_init", "tau_rule_init", "tau_term_init",
           "num_samples", "rl_warmup_steps", "maxent_coeff"]


def _tag_suffix(tags, prefix):
    for t in tags:
        if t.startswith(prefix) and t[len(prefix):].isdigit():
            return int(t[len(prefix):])
    return None


def fetch_source(run_id: str):
    """Read the source run's HP/phase/seed/rank/trial from W&B."""
    import wandb
    api = wandb.Api()
    entity = ros.WANDB_ENTITY or api.default_entity
    if entity is None:
        raise RuntimeError(
            "could not resolve a W&B entity for the source-run lookup: "
            "set WANDB_ENTITY or pass --wandb_entity"
        )
    run = api.run(f"{entity}/{ros.WANDB_PROJECT}/{run_id}")
    cfg = run.config
    hp = {k: cfg["hp"][k] for k in HP_KEYS}
    phase = cfg.get("phase") or next(t for t in run.tags if t.startswith("phase"))
    seed = _tag_suffix(run.tags, "seed")
    rank = cfg.get("rank") or _tag_suffix(run.tags, "rank")
    trial = cfg.get("trial_number")
    if seed is None:
        raise ValueError(f"could not infer seed from tags {run.tags}")
    return run.name, hp, phase, seed, rank, trial


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--run-id", required=True,
                   help="source W&B run id to reproduce (HP/seed/rank/trial/phase pulled from it)")
    p.add_argument("--wandb_entity", default=None,
                   help="W&B entity for both the source-run lookup and the repro "
                        "run (default: WANDB_ENTITY env var, else the logged-in "
                        "default entity).")
    a = p.parse_args()

    if a.wandb_entity is not None:
        ros.WANDB_ENTITY = a.wandb_entity

    src_name, HP, phase, seed, rank, trial = fetch_source(a.run_id)
    print(f"SOURCE {a.run_id} name={src_name} phase={phase} seed={seed} "
          f"rank={rank} trial={trial}\n  HP={HP}", flush=True)

    args = ros.load_base_config(ros.BASE_CONFIG)
    args.langstr = "english"
    ros.apply_hp_overrides(args, HP)

    ts = time.strftime("%m%d_%H%M%S")
    if phase == "phase3":
        ckpt_dir = f"ckpt/optuna/{ros.STUDY_NAME}/phase3_seed{seed}_{ts}"
        remark = f"phase3-best-seed{seed}"
        tags = ["optuna-v3", ros.STUDY_NAME, "phase3", "best", f"seed{seed}"]
        extra = {"phase": "phase3", "hp": HP, "repro_of": a.run_id}
    else:
        ckpt_dir = f"ckpt/optuna/{ros.STUDY_NAME}/phase2_rank{rank}_seed{seed}_{ts}"
        remark = f"phase2-rank{rank}-seed{seed}"
        tags = ["optuna-v3", ros.STUDY_NAME, "phase2", f"rank{rank}", f"seed{seed}"]
        extra = {"phase": "phase2", "rank": rank, "trial_number": trial,
                 "hp": HP, "repro_of": a.run_id}

    print(f"REPRO START {remark} device={a.device} seed={seed} ckpt={ckpt_dir}", flush=True)
    val_sf1, test_sf1 = ros.run_single_trial(
        args, ckpt_dir, remark, a.device,
        wandb_tags=tags, seed=seed, extra_wandb_config=extra,
    )
    print(f"REPRO DONE {remark} val_SF1={val_sf1:.4f} test_SF1={test_sf1:.4f}", flush=True)


if __name__ == "__main__":
    main()
