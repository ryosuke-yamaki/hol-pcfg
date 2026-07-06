# SemInfo paper-run launchers (frozen)

Baked-schedule shell launchers that reproduced the SemInfo (max-semantic-information)
paper numbers with the vendored `parsing_by_maxseminfo/` package. Kept verbatim for
provenance; **not maintained** and not meant to run as-is from this repo.

They are frozen exactly as they ran in the private `hol-pcfg-seminfo` repo, so they:

- assume the private `hol-pcfg-seminfo` repo root as CWD (one, `run_seminfo_nt_sweep.sh`,
  even hard-codes `cd /workspace/hol-pcfg-seminfo`) and reference config paths under
  `config/pas-grammar/*` (preserved here under `archive/configs/seminfo_ablations/`),
- bake in seed schedules and a fixed multi-GPU layout,
- pass a hard-coded W&B entity (`--wandb_entity ryosuke-yamaki`),
- reference checkpoints / study journals that are not shipped with this repo.

The **living, de-personalized** equivalents are under `scripts/seminfo/`
(`launch_seminfo_nt_sdim_sweep.py`, `run_optuna_seminfo.py`, `repro_driver.py`,
`run_phase0_baseline.py`, `aggregate_phase2.py`) — use those to reproduce, and pass
`--wandb_entity` (or set `WANDB_ENTITY`).

| File | What it launched |
|------|------------------|
| `launch_optuna_4gpu.sh`, `launch_phase2_4gpu.sh`, `launch_phase3_4gpu.sh`, `launch_full_pipeline.sh`, `launch_phase0_5seeds.sh` | Optuna v3 pipeline (Phase 0 baseline, Phase 1 TPE search, Phase 2/3 seed verification) across 4 GPUs. |
| `watch_and_launch_phase2.sh`, `watch_phase1_then_phase23.sh` | Poll-and-chain wrappers that start later Optuna phases once earlier ones finish. |
| `run_seminfo_nt_sweep.sh`, `run_table_ablation_seminfo_parallel.sh`, `run_vterm_delta_ablation.sh` | Hol-PCFG NT scan, scoring/normalization table-ablation grid, and vterm-delta ablation. |
| `run_multiseed_baseline.sh` | SN-PCFG + Hol-PCFG multi-seed SemInfo baselines. |
| `run_paper_grid_3seeds.sh`, `run_paper_smaller_5seeds.sh`, `run_paper_variants_5seeds.sh` | English paper grid / smaller-NT / scoring-variant runs. |
| `run_multilingual_paper_10seeds.sh`, `run_multilingual_paper_v2_10seeds.sh`, `run_multilingual_n7e2qm8t_hp.sh` | Hol-PCFG multilingual (en/zh/fr/de) paper runs. |
