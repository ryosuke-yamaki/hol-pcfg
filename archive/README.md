# Archive

Frozen, study-specific scripts and configs that reproduced past results but are **not**
reusable tooling. Kept under version control for provenance; they are not maintained and
may reference paths/checkpoints that no longer exist in this repo.

## `models/`

| File | What it was |
|------|-------------|
| `HC_PCFG.py` | HC-PCFG (Holographic Compound PCFG): HolE scoring from Hol-PCFG + the compound VAE from SC-PCFG, with `relation`/`parent` z-injection and `additive`/`phase_rotation` methods. Was a fork contribution dispatched via `model_name='HCPCFG'`; de-registered from `parser/model/__init__.py` and `parser/helper/util.py:get_model`, so it is no longer selectable. Kept for provenance. Pairs with `configs/hcpcfg_variants/` and `scripts/hcpcfg_diagnostics/`; its `from parser.*` imports still resolve from the repo root if revived. |

## `scripts/`

| Subdir | What it was |
|--------|-------------|
| `normalization_phases/` | The `run_phase{05,1,2,3,3_5,4,4_5}.sh` ladder (plus the `run_phase4_5_fg.sh` foreground variant) + `run_norm_experiments.sh`: the normalization / cnorm phase-transition ablation, each with a hard-coded `config/simplepcfg/*` list. |
| `paper_runs/` | Paper-table/figure launchers (v2hp per-language, table ablation, multilingual, full baseline, formal-nt1024) with baked-in seed schedules and GPU layout. |
| `sp_verification/` | SP1–SP4 HolE-algebra verification suite + shared `sp_utils.py` + `label_nonterminals.py`. Expects a checkpoint under `log/` that is not shipped. |
| `hcpcfg_diagnostics/` | HC-PCFG diagnostic sequences (`run_hcpcfg_diagnostics.sh`, `train_sequential.sh`, `phase0_diagnostics.py`). Drives the archived model `models/HC_PCFG.py`. |
| `seminfo_terminal/` | SemInfo / terminal-HolE analysis. The SemInfo training package is now vendored in-repo as `parsing_by_maxseminfo/`, but these scripts still need that study's checkpoints (`/workspace/hol-pcfg-seminfo/ckpt/...`) and `data/seminfo/`, neither of which is shipped here. |
| `seminfo_c0_sid1193/` | Frozen C0 phase-landscape figure scripts from the `hol-pcfg-seminfo` study: the `render_*_sid1193.py` torus / relation / parse-tree renderers (hardcode `REPO='/workspace/hol-pcfg-seminfo'`, import `parsing_by_maxseminfo`, default to unshipped SemInfo Optuna checkpoints), plus `fdr_torus_viz.py`, `residual_phase_heatmap.py`, and the checkpoint-loading `compute_fdr_scores.py` FDR axis selector. The imported training package is vendored in-repo as `parsing_by_maxseminfo/`. |
| `seminfo_sp4/` | SP4 HolE-algebra systematicity verification + A1 tangent-PCA / B1 phase figures from the `hol-pcfg-seminfo` study; SemInfo-trained counterpart to `sp_verification/`. Every analysis script is `--ckpt`-driven and needs the study's checkpoints, which are not shipped. |
| `seminfo_evals/` | One-off SemInfo re-eval launchers (`eval_best_v3_trial192.py`, `eval_best_ckpts.py`) with hardcoded `ckpt/optuna/...` and `ckpt/multilingual_...` roots; both shell out to `python -m parsing_by_maxseminfo.train --eval_ckpt`. |
| `seminfo_paper_runs/` | Frozen baked-schedule shell launchers for the SemInfo paper runs (Optuna v3 pipeline, NT / ablation grids, multilingual, multiseed baselines) against the vendored `parsing_by_maxseminfo/` harness. Assume the private `hol-pcfg-seminfo` repo root as CWD and bake in seed schedules, a fixed multi-GPU layout, and the `ryosuke-yamaki` W&B entity; reference checkpoints not shipped here. The living, de-personalized ports are under `scripts/seminfo/`. See the subdir `README.md`. |
| `optuna_phase2/` | `run_optuna_phase2.sh` — seed verification for the top-5 Optuna phase-2 configs. The script is frozen with its old `config/simplepcfg/*` paths; the preserved configs now live under `archive/configs/optuna_phase2/`. |
| `misc/` | `eval_test.py` — scratch eval helper; all paths are required CLI args, and its docstring usage example points at one specific finished run (`log/pipeline/`, pattern `rank1_seed*_20260410*`). |

## `configs/`

| Subdir | What it was |
|--------|-------------|
| `optuna_phase2/` | `optuna_phase2_rank{1,2,3,4,5}.yaml` — all configs from the Optuna phase-2 study. |
| `normalization_phases/` | The PTB norm-mode ablation Hol-PCFG configs (34) driven by `archive/scripts/normalization_phases/run_phase*.sh`: families `unit_sphere` / `freq_cnorm`(allproj) / `ecnorm` / `max_norm` / `normless` / `fixedscale` / scale_c (`cnorm_us_c_*`) / plain `tau`/`wd*`/`nt*` / holeterm-optuna. NOTE: these ablate `projection_mode`/`use_cnorm`/`scale_c`/`use_multi_tau`/`max_norm`/`fixed_scale`, which the current `parser/model/HN_PCFG.py` no longer reads (removed in `46e238b`), so they are not reproducible against HEAD. The two PTB Hol-PCFG base configs used by the living Optuna tooling (`hn_pcfg_nt4096_optuna_v2`, `hn_pcfg_allproj_cnorm_tau`) also live here — `run_pipeline.py`/`run_optuna.py`/`run_optuna_v2.py`/`run_single_train.py` and the pipeline tests reference them at this archived path. |
| `hn_english_variants/` | English Hol-PCFG v2hp scale/HP variants (`nt4096_v2hp_en`, `nt8192_v2hp_en`, `nt4096_v2hp_en_sdim1024`). Used in the `HN-PCFG (LL) paper-v1-*-English` W&B groups, not the per-language canonical. The kept English canonical is `config/multilingual/hnpcfg_english.yaml` (W&B group `HN-PCFG (LL) - English`). |
| `hn_english_ablations/` | English Hol-PCFG scoring/cnorm/tau ablation-table configs (`p3_eji18kkl_en_abl_{conv,full,hadamard,no_cnorm,no_tau}`). Not used in any `HN-PCFG (LL)` group (`_abl_full` duplicates the kept p3 canonical). |
| `hn_japanese_variants/` | Non-canonical Japanese Hol-PCFG configs: `hn_pcfg_nt4096_v2hp_ja` (v2hp HP variant, no W&B runs found), `hn_pcfg_ja_char_smoke` (tiny NT=16 smoke), and `hn_pcfg_p3_eji18kkl_ja_variantfdb` (Li et al. 2020 variantFDB protocol; appendix/future, data not generated). The canonical Japanese configs live in `config/multilingual/{hnpcfg,snpcfg,scpcfg,npcfg}_japanese.yaml`, with the char-level KTB variant at `config/multilingual/hnpcfg_japanese_char.yaml`. |
| `hcpcfg_variants/` | HC-PCFG (Holographic Compound PCFG) configs from the HC-PCFG diagnostic study: `hc_pcfg_{relation,parent,relation_phaserot,parent_phaserot,beta_zero_relation,beta_zero_parent,z_zero}`. Pairs with `archive/scripts/hcpcfg_diagnostics/` and the archived model `archive/models/HC_PCFG.py`. No canonical HC config is kept in `config/`, and `model_name='HCPCFG'` is no longer registered in `get_model`. |
| `seminfo_ablations/` | SemInfo (`parsing_by_maxseminfo/`) ablation / variant / multilingual configs, copied as-is from `config/pas-grammar/{english,chinese,french,german}-ew-reward-tbtok-idf/`: Hol-PCFG scoring/normalization ablations (`hnpcfg_n7e2qm8t_en_abl_*`), NT×s_dim paper variants (`*_rank1_sdim*_seminfo_paper`), `100k` / `holeterm` / `vterm_delta` variants, and per-language (zh/fr/de) paper configs plus `snpcfg/cpcfg/npcfg/scpcfg/tnpcfg` baselines. Referenced by `archive/scripts/seminfo_paper_runs/`. Data paths are relative (`data/english/...`); the `hnpcfg_nt1024_t2048_rank1_seminfo_paper*` variants point at a `data/seminfo/...` tree that was never populated. The canonical English Hol-PCFG config kept live is `config/seminfo/hnpcfg_nt1024_t2048_rank1_seminfo.yaml`; the sweep templates are `config/sweeps/{hn,sn}_pcfg_seminfo_nt_sdim_base.yaml`. |
