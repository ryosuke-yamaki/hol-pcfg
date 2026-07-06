#!/usr/bin/env python3
"""Launch the NT x s_dim sweep for HN-PCFG and SN-PCFG.

Sweep:
    NT    in {512, 1024, 2048, 4096, 8192}
    s_dim in {64, 128, 256, 512, 1024}
    seed  in {0, 1, 2, 3}
    model in {HNPCFG, SNPCFG}
  => 5 * 5 * 4 * 2 = 200 runs

T is fixed to 2 * NT throughout, matching every existing config.

HN-PCFG hyperparameters mirror the eji18kkl run (v2-hp Optuna phase3 best HPs).
SN-PCFG uses the project's default SN-PCFG hyperparameters.

Scheduling:
    - N_GPUS x procs_per_gpu workers in total. Each worker is a thread,
      pinned to a GPU (worker i -> gpus[i // procs_per_gpu]).
    - Jobs are sorted by estimated cost (NT * s_dim) ascending and dealt
      round-robin to all worker queues so smaller jobs finish first and
      total wall-time is balanced.
    - Each worker runs its queue sequentially on its pinned GPU.
    - Result JSON existing => skip (resumable).

Usage:
    python scripts/launch_nt_sdim_sweep.py
    python scripts/launch_nt_sdim_sweep.py --gpus 0,1,2,3 --procs-per-gpu 2
    python scripts/launch_nt_sdim_sweep.py --dry-run
    python scripts/launch_nt_sdim_sweep.py --models hn          # only HN-PCFG
    python scripts/launch_nt_sdim_sweep.py --nt 512 --sdim 64   # smoke test
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NT_VALUES = [512, 1024, 2048, 4096, 8192]
SDIM_VALUES = [64, 128, 256, 512, 1024]
SEEDS = [0, 1, 2, 3]

MODEL_SPECS = {
    "hn": {
        "model_name": "HNPCFG",
        "template": PROJECT_ROOT / "config/sweeps/hn_pcfg_nt_sdim_base.yaml",
        "tag_prefix": "hnpcfg",
    },
    "sn": {
        "model_name": "SNPCFG",
        "template": PROJECT_ROOT / "config/sweeps/sn_pcfg_nt_sdim_base.yaml",
        "tag_prefix": "snpcfg",
    },
}

SWEEP_TAG = "nt-sdim-sweep"
WANDB_GROUP = "nt-sdim-sweep"

CONFIG_OUT_DIR = PROJECT_ROOT / "config/sweeps/nt_sdim"
RESULT_ROOT = PROJECT_ROOT / "results/nt_sdim_sweep"
LOG_ROOT = PROJECT_ROOT / "logs/nt_sdim_sweep"


@dataclass(frozen=True)
class Job:
    model_key: str        # "hn" or "sn"
    nt: int
    sdim: int
    seed: int
    config_path: Path
    result_path: Path
    log_path: Path
    run_name: str

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
                    run_name = (f"{spec['tag_prefix']}-nt{nt}-sdim{sdim}"
                                f"-seed{seed}")
                    jobs.append(Job(
                        model_key=mk, nt=nt, sdim=sdim, seed=seed,
                        config_path=cfg_path, result_path=result_path,
                        log_path=log_path, run_name=run_name,
                    ))
    return jobs


def assign_jobs(jobs: list[Job], n_workers: int) -> list[list[Job]]:
    """Sort by cost ascending, deal round-robin to n_workers queues."""
    sorted_jobs = sorted(jobs, key=lambda j: (j.cost, j.model_key, j.seed))
    queues: list[list[Job]] = [[] for _ in range(n_workers)]
    for i, job in enumerate(sorted_jobs):
        queues[i % n_workers].append(job)
    return queues


def run_one(job: Job, gpu: int) -> int:
    """Run one job on the given GPU. Returns process return code."""
    spec = MODEL_SPECS[job.model_key]
    extra_tags = [
        SWEEP_TAG,
        spec["tag_prefix"],
        f"nt:{job.nt}",
        f"sdim:{job.sdim}",
        f"seed:{job.seed}",
    ]
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_single_train.py"),
        "--config", str(job.config_path),
        "--seed", str(job.seed),
        "--result-path", str(job.result_path),
        "--wandb-name", job.run_name,
        "--wandb-tags", *extra_tags,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH") else str(PROJECT_ROOT)
    )
    env["WANDB_RUN_GROUP"] = WANDB_GROUP

    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    with job.log_path.open("w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              env=env, cwd=str(PROJECT_ROOT))
    return proc.returncode


def _is_job_running_elsewhere(job: Job) -> bool:
    """Return True if any process is already running this exact job.

    Detects orphan training processes from a previously-killed master, so a
    supplementary launcher started afterwards does not duplicate work.
    Matches on the per-combination config path plus the seed (these uniquely
    identify the job).
    """
    needle_cfg = str(job.config_path)
    needle_seed = f"--seed {job.seed} "
    try:
        out = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    except Exception:
        return False
    for line in out.splitlines():
        if ('run_single_train.py' in line
                and needle_cfg in line
                and needle_seed in line):
            return True
    return False


def worker(worker_id: int, gpu: int, job_queue: list[Job],
           stats: dict, stats_lock: threading.Lock,
           stop_event: threading.Event) -> None:
    log(f"[worker {worker_id} gpu {gpu}] starting "
        f"({len(job_queue)} jobs assigned)")
    for idx, job in enumerate(job_queue, start=1):
        if stop_event.is_set():
            log(f"[worker {worker_id} gpu {gpu}] stop signal — aborting")
            return
        prefix = (f"[worker {worker_id} gpu {gpu}] "
                  f"({idx}/{len(job_queue)}) "
                  f"{job.model_key} NT={job.nt} sdim={job.sdim} "
                  f"seed={job.seed}")
        if job.result_path.exists():
            log(f"{prefix} SKIP (result exists)")
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
            log(f"{prefix} EXCEPTION: {e}")
            with stats_lock:
                stats["failed"] += 1
            continue
        elapsed = time.time() - start
        hours = elapsed / 3600.0
        if rc != 0:
            log(f"{prefix} FAILED rc={rc} elapsed={elapsed:.0f}s "
                f"({hours:.2f}h) log={job.log_path}")
            with stats_lock:
                stats["failed"] += 1
        else:
            log(f"{prefix} DONE   elapsed={elapsed:.0f}s ({hours:.2f}h)")
            with stats_lock:
                stats["completed"] += 1
    log(f"[worker {worker_id} gpu {gpu}] queue exhausted")


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3",
                    help="Comma-separated GPU IDs (default: 0,1,2,3)")
    ap.add_argument("--procs-per-gpu", type=int, default=1,
                    help="Concurrent jobs per GPU (default: 1)")
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
    args = ap.parse_args()

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
                    f"seed={j.seed} -> {j.result_path.relative_to(PROJECT_ROOT)}")
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
        log("KeyboardInterrupt — signalling workers to stop after current jobs")
        stop_event.set()
        for t in threads:
            t.join()

    log(f"All workers finished. completed={stats['completed']} "
        f"failed={stats['failed']} skipped={stats['skipped']} "
        f"total={len(jobs)}")
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
