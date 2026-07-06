#!/usr/bin/env python3
"""Launch multilingual PCFG training across the 6 SPMRL languages.

Job space:
    model in {hn, sn, sc, npcfg}
        -> config/multilingual/{hnpcfg,snpcfg,scpcfg,npcfg}_<lang>.yaml
    lang  in {basque, hebrew, hungarian, korean, polish, swedish}
    seed  in {0, 1, 2, 3}

--models defaults to {hn, sn, sc} => 3 * 6 * 4 = 72 runs. N-PCFG is opt-in
(--models npcfg => 6 * 4 = 24 runs): its inside chart peaks ~16-20 GiB at
batch=8, so it runs on a separate A100 80GB host.

Scheduling:
    - len(gpus) * procs_per_gpu worker threads. Worker i is pinned to
      gpus[i // procs_per_gpu] through CUDA_VISIBLE_DEVICES.
    - Every job is pushed into ONE shared queue, longest-estimated-cost first
      (LPT heuristic). Idle workers pull the next job, so the ~5x spread in
      per-run time across languages stays load-balanced without static
      partitioning.
    - A run counts as done only when its result JSON exists AND has
      status == 'completed'. run_single_train.py writes the JSON even when
      training raises (status stays 'failed'), so an exists-only check would
      permanently skip crashed/OOM runs. Failed/partial JSONs are retried on
      the next launch (resumable).

Usage:
    python scripts/launch_multilingual_pcfg.py                 # 72 runs, GPU 2&3
    python scripts/launch_multilingual_pcfg.py --dry-run
    python scripts/launch_multilingual_pcfg.py --models hn,sn
    python scripts/launch_multilingual_pcfg.py --models npcfg --gpus 0,1,2
    python scripts/launch_multilingual_pcfg.py --langs korean --models sc
"""
from __future__ import annotations

import argparse
import json
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

LANGS = ["basque", "hebrew", "hungarian", "korean", "polish", "swedish"]
SEEDS = [0, 1, 2, 3]

# epoch_sec: wall-clock per epoch on PTB (2409 train batches). HN/SN are
# measured from train logs; SC is estimated (~0.55x SN, NT halved). Used only
# to order jobs longest-first and to print an ETA -- the real run length is
# decided by the config (max_epoch + early stopping).
MODEL_SPECS = {
    "hn": {"config_prefix": "hnpcfg", "epoch_sec": 269.0},
    "sn": {"config_prefix": "snpcfg", "epoch_sec": 260.0},
    "sc": {"config_prefix": "scpcfg", "epoch_sec": 150.0},
    "npcfg": {"config_prefix": "npcfg", "epoch_sec": 200.0},
}

# Train batches per corpus / PTB's 2409 batches; per-epoch time ~ this ratio.
LANG_BATCH_RATIO = {
    "basque": 0.205, "hebrew": 0.125, "hungarian": 0.211,
    "korean": 0.602, "polish": 0.176, "swedish": 0.132,
}

EPOCHS_ESTIMATE = 26          # avg epochs incl. early stopping (ETA only)
PER_GPU_THROUGHPUT_2X = 1.18  # combined throughput of 2 co-located runs vs 1

CONFIG_DIR = PROJECT_ROOT / "config/multilingual"
RESULT_ROOT = PROJECT_ROOT / "results/multilingual_pcfg"
LOG_ROOT = PROJECT_ROOT / "logs/multilingual_pcfg"
WANDB_GROUP = "multilingual-pcfg"


@dataclass(frozen=True)
class Job:
    model_key: str
    lang: str
    seed: int
    config_path: Path
    result_path: Path
    log_path: Path
    run_name: str

    @property
    def cost(self) -> float:
        """Estimated wall-clock seconds per epoch for this (model, lang)."""
        return MODEL_SPECS[self.model_key]["epoch_sec"] * LANG_BATCH_RATIO[self.lang]

    @property
    def label(self) -> str:
        return f"{self.model_key}/{self.lang}/seed{self.seed}"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def is_completed(result_path: Path) -> bool:
    """True only if the result JSON exists with status == 'completed'.

    run_single_train.py always writes the JSON, even on failure (status stays
    'failed'), so an exists-only check would skip crashed/OOM runs forever.
    """
    if not result_path.exists():
        return False
    try:
        with result_path.open() as f:
            return json.load(f).get("status") == "completed"
    except (json.JSONDecodeError, OSError):
        return False


def build_jobs(model_keys: list[str], langs: list[str],
               seeds: list[int]) -> list[Job]:
    jobs: list[Job] = []
    for mk in model_keys:
        prefix = MODEL_SPECS[mk]["config_prefix"]
        (RESULT_ROOT / mk).mkdir(parents=True, exist_ok=True)
        for lang in langs:
            cfg = CONFIG_DIR / f"{prefix}_{lang}.yaml"
            if not cfg.exists():
                raise FileNotFoundError(f"config not found: {cfg}")
            with cfg.open() as f:
                base_run_name = ((yaml.safe_load(f) or {}).get("wandb", {})
                                 .get("run_name", f"{prefix}-{lang}"))
            for seed in seeds:
                jobs.append(Job(
                    model_key=mk, lang=lang, seed=seed,
                    config_path=cfg,
                    result_path=RESULT_ROOT / mk / f"{lang}_seed{seed}.json",
                    log_path=LOG_ROOT / f"{prefix}_{lang}_seed{seed}.log",
                    run_name=f"{base_run_name}-seed{seed}",
                ))
    return jobs


def is_running_elsewhere(job: Job) -> bool:
    """True if some process is already training this exact (config, seed).

    Guards against duplicating work when the launcher is started twice.
    """
    needle_cfg = str(job.config_path)
    needle_seed = f"--seed {job.seed} "
    try:
        out = subprocess.check_output(["ps", "-eo", "args"], text=True)
    except Exception:
        return False
    for line in out.splitlines():
        if ("run_single_train.py" in line and needle_cfg in line
                and needle_seed in line):
            return True
    return False


def run_one(job: Job, gpu: int) -> int:
    """Run one training job on the given GPU; return the process exit code."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_single_train.py"),
        "--config", str(job.config_path),
        "--seed", str(job.seed),
        "--result-path", str(job.result_path),
        "--wandb-name", job.run_name,
        "--wandb-tags", WANDB_GROUP, f"seed:{job.seed}",
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


def worker(wid: int, gpu: int, job_q: "queue.Queue[Job]",
           stats: dict, lock: threading.Lock, stop: threading.Event,
           total: int) -> None:
    tag = f"[w{wid} gpu{gpu}]"
    log(f"{tag} started")
    while not stop.is_set():
        try:
            job = job_q.get_nowait()
        except queue.Empty:
            break
        prefix = f"{tag} {job.label}"
        if is_completed(job.result_path):
            log(f"{prefix} SKIP (already completed)")
            with lock:
                stats["skipped"] += 1
            continue
        if is_running_elsewhere(job):
            log(f"{prefix} SKIP (running elsewhere)")
            with lock:
                stats["skipped"] += 1
            continue
        with lock:
            settled = stats["completed"] + stats["failed"] + stats["skipped"]
        log(f"{prefix} START ({settled}/{total} settled) "
            f"~{job.cost * EPOCHS_ESTIMATE / 60:.0f}min est")
        start = time.time()
        try:
            rc = run_one(job, gpu)
        except Exception as e:
            log(f"{prefix} EXCEPTION {e}")
            with lock:
                stats["failed"] += 1
            continue
        elapsed = time.time() - start
        ok = rc == 0 and is_completed(job.result_path)
        with lock:
            stats["completed" if ok else "failed"] += 1
        verdict = "DONE" if ok else f"FAILED rc={rc}"
        log(f"{prefix} {verdict} elapsed={elapsed / 60:.1f}min "
            f"log={job.log_path.relative_to(PROJECT_ROOT)}")
    log(f"{tag} queue exhausted")


def parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default="2,3", help="GPU IDs (default: 2,3)")
    ap.add_argument("--procs-per-gpu", type=int, default=2,
                    help="Concurrent runs per GPU (default: 2)")
    ap.add_argument("--models", default="hn,sn,sc",
                    help="Subset of {hn,sn,sc,npcfg} (default: hn,sn,sc)")
    ap.add_argument("--langs", default=",".join(LANGS),
                    help="Subset of languages (default: all 6)")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)),
                    help="Seeds (default: 0,1,2,3)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the job plan and ETA, then exit")
    args = ap.parse_args()

    gpus = [int(x) for x in parse_csv(args.gpus)]
    ppg = args.procs_per_gpu
    if ppg < 1:
        ap.error("--procs-per-gpu must be >= 1")
    model_keys = parse_csv(args.models)
    for mk in model_keys:
        if mk not in MODEL_SPECS:
            ap.error(f"unknown model '{mk}' "
                     f"(choose from {','.join(MODEL_SPECS)})")
    langs = parse_csv(args.langs)
    for lng in langs:
        if lng not in LANG_BATCH_RATIO:
            ap.error(f"unknown language '{lng}'")
    seeds = [int(x) for x in parse_csv(args.seeds)]

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(model_keys, langs, seeds)
    jobs.sort(key=lambda j: j.cost, reverse=True)  # LPT: longest first

    n_workers = len(gpus) * ppg
    worker_gpus = [gpus[i // ppg] for i in range(n_workers)]

    total_gpu_sec = sum(j.cost for j in jobs) * EPOCHS_ESTIMATE
    per_gpu_tput = PER_GPU_THROUGHPUT_2X if ppg >= 2 else 1.0
    eta_h = total_gpu_sec / (len(gpus) * per_gpu_tput) / 3600

    log(f"Project root : {PROJECT_ROOT}")
    log(f"GPUs         : {gpus}  (procs/gpu={ppg}, workers={n_workers})")
    log(f"Models       : {model_keys}")
    log(f"Languages    : {langs}")
    log(f"Seeds        : {seeds}")
    log(f"Total jobs   : {len(jobs)}")
    log(f"Est GPU-time : {total_gpu_sec / 3600:.1f} GPU-h "
        f"(@ ~{EPOCHS_ESTIMATE} epochs/run)")
    log(f"Est wall-time: ~{eta_h:.1f} h "
        f"({ppg} runs/GPU give ~{per_gpu_tput:.2f}x per GPU, not {ppg}x "
        f"-- training is GPU-bound)")

    if args.dry_run:
        for j in jobs:
            log(f"  {j.label:26s} ~{j.cost * EPOCHS_ESTIMATE / 60:5.0f}min  "
                f"wandb={j.run_name}")
        return 0

    job_q: "queue.Queue[Job]" = queue.Queue()
    for j in jobs:
        job_q.put(j)

    stats = {"completed": 0, "failed": 0, "skipped": 0}
    lock = threading.Lock()
    stop = threading.Event()
    threads: list[threading.Thread] = []
    for i, gpu in enumerate(worker_gpus):
        t = threading.Thread(
            target=worker,
            args=(i, gpu, job_q, stats, lock, stop, len(jobs)),
            name=f"w{i}-gpu{gpu}", daemon=False)
        t.start()
        threads.append(t)

    t0 = time.time()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        log("KeyboardInterrupt -- workers stop after their current run")
        stop.set()
        for t in threads:
            t.join()

    log(f"Finished in {(time.time() - t0) / 3600:.2f} h | "
        f"completed={stats['completed']} failed={stats['failed']} "
        f"skipped={stats['skipped']} total={len(jobs)}")
    if stats["failed"]:
        log("Re-run the same command to retry failed/incomplete jobs.")
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
