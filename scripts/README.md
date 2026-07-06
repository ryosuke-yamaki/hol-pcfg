# Scripts

Reusable tooling. Each script has a module docstring with its full usage; this is just a
map. Frozen, one-off experiment launchers live under `../archive/` instead.

## Data preparation

| Script | Purpose |
|--------|---------|
| `download_formal_data.sh` | Download the Lample symbolic-math dataset (license/citation notes inline). |
| `split_symmath.py`, `preprocess_symmath_infix.py` | Split the Lample math (`prim_fwd` -> `symmath-*.prefix`) and build the infix `symmath_infix` data. |
| `preprocess_kaomoji.py`, `sample_kaomoji.py` | Build/sample the kaomoji data. |
| `prepare_ktb_data.sh`, `prepare_ktb_char_data.sh`, `preprocess_ktb_char.py` | Keyaki Treebank (KTB) word/char-level data. |

## Training launchers

| Script | Purpose |
|--------|---------|
| `launch_multilingual_pcfg.py` | Multi-GPU SPMRL grid (hn/sn/sc × 6 langs × seeds; npcfg opt-in). Drives `config/multilingual/`. |
| `launch_nt_sdim_sweep.py` | NT × s_dim sweep (200 runs) for Hol-PCFG / SN-PCFG; generates `config/sweeps/nt_sdim/`. |
| `launch_optuna_multigpu.sh` | Multi-GPU wrapper around the legacy v1 `run_optuna.py`. |

## Optuna HP search

| Script | Purpose |
|--------|---------|
| `run_pipeline.py` | End-to-end HP-search pipeline (phases 0–3). |
| `run_optuna.py`, `run_optuna_v2.py` | Optuna studies (v2 is current, driven by `run_pipeline.py`; v1 is legacy). |
| `run_single_train.py` | Single training run writing a result JSON. Invoked by the pipeline **and** by `tests/`. |

## Analysis / figures

| Script | Purpose |
|--------|---------|
| `dump_predictions.py`, `verify_f1_from_jsonl.py` | Dump predictions to JSONL and recompute F1. |
| `analyze_error.py`, `render_b1_pdf.py` | Hol-PCFG vs SN-PCFG error analysis + tree PDFs (`render_b1_pdf` imports `analyze_error`). |
| `render_parse_pdf.py` | Per-sentence parse-tree visualizer (referenced by `parser/model/HN_PCFG.py`). |
| `baseline_f1_formal.py` | Right-branching / random F1 baselines. |
| `analyze_kaomoji_nts.py` | Kaomoji nonterminal analysis. |
| `predict_and_export_trees.py`, `export_char_morpheme_trees.py`, `eval_phrase_projection.py` | KTB char-level prediction/export/phrase-projection eval. |
| `c0_phase_landscape/` | C0 phase-landscape analysis (heatmaps, PCA); fully argparse-driven. |

## SemInfo (`seminfo/`)

Living, de-personalized ports of the SemInfo launchers. They drive the vendored
`../parsing_by_maxseminfo/` harness (a different entry point than the native families —
see the top-level README "Experiment families"). The frozen paper-run shells stay under
`../archive/scripts/seminfo_paper_runs/`.

| Script | Purpose |
|--------|---------|
| `launch_seminfo_nt_sdim_sweep.py` | NT × s_dim SemInfo sweep (200 runs) for Hol-PCFG / SN-PCFG; generates `config/sweeps/seminfo_nt_sdim/`. |
| `run_optuna_seminfo.py` | Optuna v3 HP search (phases 1–3) for the post-refactor Hol-PCFG + SemInfo. |
| `repro_driver.py` | Faithful single-run reproducer: re-train a given W&B run id via `run_optuna_seminfo.run_single_trial`. |
| `run_phase0_baseline.py` | Phase 0 rank1-SemInfo baseline (single seed) reference for the Optuna pipeline. |
| `aggregate_phase2.py` | Merge Phase 2/3 worker result JSONs into a ranked summary. |
| `train_seeded.py` | Seeded, bit-reproducible wrapper around `parsing_by_maxseminfo.train` (`--seed N` + passthrough). |
