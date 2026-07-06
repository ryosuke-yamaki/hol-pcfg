#!/usr/bin/env python3
"""Launch the NT x s_dim SemInfo sweep for HN-PCFG and SN-PCFG.

Sweep:
    NT    in {128, 256, 512, 1024, 2048}
    s_dim in {64, 128, 256, 512, 1024}
    seed  in {0, 1, 2, 3}
    model in {HN-PCFG, SN-PCFG}
  => 5 * 5 * 4 * 2 = 200 runs

T is fixed to 2 * NT throughout, matching every existing config.

HN-PCFG hyperparameters mirror W&B run n7e2qm8t (Optuna v3 phase2 rank2 seed5
best HPs for English SemInfo). SN-PCFG uses the project's default config and
gets SemInfo behavior via the runtime CLI overrides:
    --set_training_mode rl --set_mode_reward log_tfidf
(matching the convention from scripts/run_multiseed_baseline.sh).

Seed is set via PL_GLOBAL_SEED and PYTHONHASHSEED env vars; the training
entry point (parsing_by_maxseminfo.train) uses Lightning's seed_everything.

Scheduling:
    - N_GPUS x procs_per_gpu workers in total. Each worker is a thread,
      pinned to a GPU (worker i -> gpus[i // procs_per_gpu]).
    - Jobs are sorted by estimated cost (NT * s_dim) ascending and dealt
      round-robin to all worker queues so smaller jobs finish first and
      total wall-time per worker is balanced.
    - Each worker runs its queue sequentially on its pinned GPU.
    - A completed-status result JSON acts as the skip marker; failed-status
      JSONs do NOT cause skip (unlike the buggy hol-pcfg launcher), so OOM
      retries are automatic on the next launcher invocation.
    - A ps-based check also prevents duplicate launches when multiple
      supplementary launchers run side-by-side.

Usage:
    python scripts/launch_seminfo_nt_sdim_sweep.py
    python scripts/launch_seminfo_nt_sdim_sweep.py --gpus 0,1,2,3 --procs-per-gpu 2
    python scripts/launch_seminfo_nt_sdim_sweep.py --dry-run
    python scripts/launch_seminfo_nt_sdim_sweep.py --models hn     # only HN-PCFG
    python scripts/launch_seminfo_nt_sdim_sweep.py --nt 128 --sdim 64  # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NT_VALUES = [128, 256, 512, 1024, 2048]
SDIM_VALUES = [64, 128, 256, 512, 1024]
SEEDS = [0, 1, 2, 3]

MODEL_SPECS = {
    "hn": {
        "model_name": "HNPCFG-FixedCostReward",
        "template": PROJECT_ROOT / "config/sweeps/hn_pcfg_seminfo_nt_sdim_base.yaml",
        "tag_prefix": "hnpcfg",
        # HN-PCFG config already has mode=rl, mode_reward=log_tfidf, so no
        # CLI override needed.
        "extra_args": [],
    },
    "sn": {
        "model_name": "SNPCFG-FixedCostReward",
        "template": PROJECT_ROOT / "config/sweeps/sn_pcfg_seminfo_nt_sdim_base.yaml",
        "tag_prefix": "snpcfg",
        # SN-PCFG config defaults to mode=nll; override to rl + log_tfidf
        # at runtime, matching scripts/run_multiseed_baseline.sh.
        "extra_args": ["--set_training_mode", "rl", "--set_mode_reward", "log_tfidf"],
    },
}

SWEEP_TAG = "seminfo-nt-sdim-sweep"
WANDB_PROJECT = "hol-pcfg"
# W&B entity is not hardcoded: defaults to the WANDB_ENTITY env var (None -> use
# the logged-in default entity). Override with --wandb_entity.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_GROUP = "seminfo-nt-sdim-sweep"

CONFIG_OUT_DIR = PROJECT_ROOT / "config/sweeps/seminfo_nt_sdim"
RESULT_ROOT = PROJECT_ROOT / "results/seminfo_nt_sdim_sweep"
LOG_ROOT = PROJECT_ROOT / "logs/seminfo_nt_sdim_sweep"
CKPT_ROOT = PROJECT_ROOT / "ckpt/seminfo_nt_sdim_sweep"


@dataclass(frozen=True)
class Job:
    model_key: str          # "hn" or "sn"
    nt: int
    sdim: int
    seed: int
    config_path: Path
    result_path: Path
    log_path: Path
    ckpt_dir: Path
    remark: str

    @property
    def cost(self) -> int:
        return self.nt * self.sdim


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def render_config(template_path: Path, nt: int, sdim: int, out_path: Path) -> None:
    with template_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["NT"] = nt
    cfg["model"]["T"] = 2 * nt
    cfg["model"]["s_dim"] = sdim
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def build_jobs(model_keys: list[str], nts: list[int], sdims: list[int],
               seeds: list[int]) -> list[Job]:
    jobs: list[Job] = []
    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mk in model_keys:
        spec = MODEL_SPECS[mk]
        result_dir = RESULT_ROOT / mk
        log_dir = LOG_ROOT
        result_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        for nt in nts:
            for sdim in sdims:
                cfg_path = CONFIG_OUT_DIR / f"{mk}_NT{nt}_sdim{sdim}.yaml"
                render_config(spec["template"], nt, sdim, cfg_path)
                for seed in seeds:
                    result_path = (result_dir
                                   / f"NT{nt}_sdim{sdim}_seed{seed}.json")
                    log_path = (log_dir
                                / f"{mk}_NT{nt}_sdim{sdim}_seed{seed}.log")
                    ckpt_dir = (CKPT_ROOT / mk
                                / f"NT{nt}_sdim{sdim}_seed{seed}")
                    remark = (f"{spec['tag_prefix']}-nt{nt}-sdim{sdim}"
                              f"-seed{seed}")
                    jobs.append(Job(
                        model_key=mk, nt=nt, sdim=sdim, seed=seed,
                        config_path=cfg_path, result_path=result_path,
                        log_path=log_path, ckpt_dir=ckpt_dir, remark=remark,
                    ))
    return jobs


def assign_jobs(jobs: list[Job], n_workers: int) -> list[list[Job]]:
    """Sort by cost ascending, deal round-robin to n_workers queues."""
    sorted_jobs = sorted(jobs, key=lambda j: (j.cost, j.model_key, j.seed))
    queues: list[list[Job]] = [[] for _ in range(n_workers)]
    for i, job in enumerate(sorted_jobs):
        queues[i % n_workers].append(job)
    return queues


def _is_job_completed(job: Job) -> bool:
    """True iff result JSON exists AND status == 'completed'.

    This is the key fix vs the hol-pcfg launcher: failed/incomplete result
    JSONs don't cause skip, so an OOM-failed job is retried automatically
    on the next launcher invocation.
    """
    if not job.result_path.exists():
        return False
    try:
        with job.result_path.open() as f:
            d = json.load(f)
        return d.get("status") == "completed"
    except Exception:
        return False


def _is_job_running_elsewhere(job: Job) -> bool:
    """True if any other process is currently training the same job.

    Matches on the per-combination config path and the seed-controlling env
    var (PL_GLOBAL_SEED=N) since the training CLI itself doesn't carry a
    --seed flag.
    """
    needle_cfg = str(job.config_path)
    needle_remark = f"--remark {job.remark}"
    try:
        out = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    except Exception:
        return False
    for line in out.splitlines():
        if ('parsing_by_maxseminfo.train' in line
                and needle_cfg in line
                and needle_remark in line):
            return True
    return False


# Patterns for extracting test metrics from training stdout
_METRIC_PATTERNS = {
    "test_sentence_f1": [
        re.compile(r"test/sentence_f1[^0-9\-]+([-+]?\d+\.\d+)"),
        re.compile(r"'test/sentence_f1':\s*([-+]?\d+\.\d+)"),
    ],
    "test_corpus_f1": [
        re.compile(r"test/corpus_f1[^0-9\-]+([-+]?\d+\.\d+)"),
        re.compile(r"'test/corpus_f1':\s*([-+]?\d+\.\d+)"),
    ],
    "test_avg_ll": [
        re.compile(r"test/avg_ll[^0-9\-]+([-+]?\d+\.\d+)"),
        re.compile(r"'test/avg_ll':\s*([-+]?\d+\.\d+)"),
    ],
    "test_avg_ppl": [
        re.compile(r"test/avg_ppl[^0-9\-]+([-+]?\d+\.\d+)"),
        re.compile(r"'test/avg_ppl':\s*([-+]?\d+\.\d+)"),
    ],
}


def _parse_metrics_from_log(log_path: Path) -> dict:
    """Extract last-seen test metrics from a training log."""
    if not log_path.exists():
        return {}
    metrics: dict = {}
    text = log_path.read_text(errors="replace")
    for key, patterns in _METRIC_PATTERNS.items():
        for pat in patterns:
            matches = pat.findall(text)
            if matches:
                try:
                    metrics[key] = float(matches[-1])
                    break
                except ValueError:
                    pass
    return metrics


def _write_result_json(job: Job, status: str, rc: int | None,
                       elapsed_s: float, extra: dict | None = None) -> None:
    metrics = _parse_metrics_from_log(job.log_path)
    payload = {
        "model": MODEL_SPECS[job.model_key]["model_name"],
        "model_key": job.model_key,
        "NT": job.nt,
        "T": 2 * job.nt,
        "s_dim": job.sdim,
        "seed": job.seed,
        "remark": job.remark,
        "config": str(job.config_path.relative_to(PROJECT_ROOT)),
        "status": status,
        "rc": rc,
        "elapsed_s": elapsed_s,
        **metrics,
    }
    if extra:
        payload.update(extra)
    job.result_path.parent.mkdir(parents=True, exist_ok=True)
    with job.result_path.open("w") as f:
        json.dump(payload, f, indent=2)


def run_one(job: Job, gpu: int) -> int:
    """Run one training job on the given GPU. Returns process return code."""
    spec = MODEL_SPECS[job.model_key]

    job.ckpt_dir.mkdir(parents=True, exist_ok=True)

    tags = [
        SWEEP_TAG,
        spec["tag_prefix"],
        f"nt:{job.nt}",
        f"sdim:{job.sdim}",
        f"seed:{job.seed}",
    ]

    cmd = [
        sys.executable, "-m", "parsing_by_maxseminfo.train",
        "-c", str(job.config_path),
        "--ckpt_dir", str(job.ckpt_dir),
        "--remark", job.remark,
        "--langstr", "english",
        "--wandb_project", WANDB_PROJECT,
        "--ngpu", "1",
    ]
    if WANDB_ENTITY:
        cmd += ["--wandb_entity", WANDB_ENTITY]
    for tag in tags:
        cmd += ["--wandb_tags", tag]
    cmd += list(spec["extra_args"])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH") else str(PROJECT_ROOT)
    )
    env["PL_GLOBAL_SEED"] = str(job.seed)
    env["PYTHONHASHSEED"] = str(job.seed)
    env["WANDB_RUN_GROUP"] = WANDB_GROUP

    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    with job.log_path.open("w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              env=env, cwd=str(PROJECT_ROOT))
    return proc.returncode


def worker(worker_id: int, gpu: int, job_queue: list[Job],
           stats: dict, stats_lock: threading.Lock,
           stop_event: threading.Event) -> None:
    log(f"[worker {worker_id} gpu {gpu}] starting "
        f"({len(job_queue)} jobs assigned)")
    for idx, job in enumerate(job_queue, start=1):
        if stop_event.is_set():
            log(f"[worker {worker_id} gpu {gpu}] stop signal -- aborting")
            return
        prefix = (f"[worker {worker_id} gpu {gpu}] "
                  f"({idx}/{len(job_queue)}) "
                  f"{job.model_key} NT={job.nt} sdim={job.sdim} "
                  f"seed={job.seed}")
        if _is_job_completed(job):
            log(f"{prefix} SKIP (completed)")
            with stats_lock:
                stats["skipped"] += 1
            continue
        if _is_job_running_elsewhere(job):
            log(f"{prefix} SKIP (already running elsewhere)")
            with stats_lock:
                stats["skipped"] += 1
            continue
        log(f"{prefix} START")
        start = time.time()
        try:
            rc = run_one(job, gpu)
        except Exception as e:
            elapsed = time.time() - start
            log(f"{prefix} EXCEPTION: {e}")
            _write_result_json(job, status="exception", rc=None,
                               elapsed_s=elapsed, extra={"error": str(e)})
            with stats_lock:
                stats["failed"] += 1
            continue
        elapsed = time.time() - start
        hours = elapsed / 3600.0
        if rc != 0:
            log(f"{prefix} FAILED rc={rc} elapsed={elapsed:.0f}s "
                f"({hours:.2f}h) log={job.log_path}")
            _write_result_json(job, status="failed", rc=rc, elapsed_s=elapsed)
            with stats_lock:
                stats["failed"] += 1
        else:
            log(f"{prefix} DONE   elapsed={elapsed:.0f}s ({hours:.2f}h)")
            _write_result_json(job, status="completed", rc=rc,
                               elapsed_s=elapsed)
            with stats_lock:
                stats["completed"] += 1
    log(f"[worker {worker_id} gpu {gpu}] queue exhausted")


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3",
                    help="Comma-separated GPU IDs (default: 0,1,2,3)")
    ap.add_argument("--procs-per-gpu", type=int, default=2,
                    help="Concurrent jobs per GPU (default: 2)")
    ap.add_argument("--models", default="hn,sn",
                    help="Comma-separated subset of {hn,sn} (default: hn,sn)")
    ap.add_argument("--nt", type=parse_int_list, default=None,
                    help="Override NT values (comma-separated)")
    ap.add_argument("--sdim", type=parse_int_list, default=None,
                    help="Override s_dim values (comma-separated)")
    ap.add_argument("--seeds", type=parse_int_list, default=None,
                    help="Override seeds (comma-separated)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the job plan and exit")
    ap.add_argument("--wandb_entity", default=None,
                    help="W&B entity passed to the training subprocesses "
                         "(default: WANDB_ENTITY env var, else the logged-in "
                         "default entity).")
    args = ap.parse_args()

    if args.wandb_entity is not None:
        globals()["WANDB_ENTITY"] = args.wandb_entity

    gpus = parse_int_list(args.gpus)
    procs_per_gpu = args.procs_per_gpu
    if procs_per_gpu < 1:
        ap.error("--procs-per-gpu must be >= 1")
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    for mk in model_keys:
        if mk not in MODEL_SPECS:
            ap.error(f"unknown model key '{mk}'")
    nts = args.nt or NT_VALUES
    sdims = args.sdim or SDIM_VALUES
    seeds = args.seeds or SEEDS

    n_workers = len(gpus) * procs_per_gpu
    worker_gpus = [gpus[i // procs_per_gpu] for i in range(n_workers)]

    jobs = build_jobs(model_keys, nts, sdims, seeds)
    queues = assign_jobs(jobs, n_workers)

    log(f"Project root: {PROJECT_ROOT}")
    log(f"GPUs: {gpus} (procs-per-gpu={procs_per_gpu}, "
        f"workers={n_workers})")
    log(f"Models: {model_keys}")
    log(f"NT values: {nts}")
    log(f"s_dim values: {sdims}")
    log(f"Seeds: {seeds}")
    log(f"Total jobs: {len(jobs)}")
    for i, q in enumerate(queues):
        total_cost = sum(j.cost for j in q)
        log(f"  queue {i} (gpu {worker_gpus[i]}): {len(q)} jobs, "
            f"sum(NT*sdim)={total_cost}")

    if args.dry_run:
        for q_idx, q in enumerate(queues):
            log(f"--- queue {q_idx} (gpu {worker_gpus[q_idx]}) ---")
            for j in q:
                log(f"  {j.model_key} NT={j.nt} sdim={j.sdim} "
                    f"seed={j.seed} -> "
                    f"{j.result_path.relative_to(PROJECT_ROOT)}")
        return 0

    stats = {"completed": 0, "failed": 0, "skipped": 0}
    stats_lock = threading.Lock()
    stop_event = threading.Event()

    threads: list[threading.Thread] = []
    for i, gpu in enumerate(worker_gpus):
        t = threading.Thread(
            target=worker,
            args=(i, gpu, queues[i], stats, stats_lock, stop_event),
            name=f"worker-{i}-gpu{gpu}",
            daemon=False,
        )
        t.start()
        threads.append(t)

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        log("KeyboardInterrupt -- signalling workers to stop after current jobs")
        stop_event.set()
        for t in threads:
            t.join()

    log(f"All workers finished. completed={stats['completed']} "
        f"failed={stats['failed']} skipped={stats['skipped']} "
        f"total={len(jobs)}")
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
