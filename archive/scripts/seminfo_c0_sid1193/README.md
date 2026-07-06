# seminfo_c0_sid1193

Frozen, as-run snapshot of the C0 phase-landscape figure scripts from the
separate `hol-pcfg-seminfo` study. They rendered the paper's torus / phase-
landscape figures (a combined panel, the direct-arrow relation geometry, and
the `sid1193` example parse tree) and selected the FDR torus axes behind them.

Provenance only — not maintained, and not runnable here as-is:

- The `render_{combined,direct_arrows,parse_tree}_sid1193.py` renderers hardcode
  `REPO = Path('/workspace/hol-pcfg-seminfo')`, `sys.path`-insert
  `parsing_by_maxseminfo/`, and
  `from parsing_by_maxseminfo.parser.model.HN_PCFG import HNPCFGFixedCostReward`.
  Their `--ckpt` defaults point at April/June-2026 SemInfo Optuna checkpoints
  (`ckpt/optuna/hnpcfg-rank1-seminfo-v3/...`) that are **not** shipped publicly.
- `fdr_torus_viz.py` is the shared phase/FDR helper the renderers and
  `compute_fdr_scores.py` import; `residual_phase_heatmap.py` is a standalone
  `--ckpt`-driven residual-phase heatmap.
- `compute_fdr_scores.py` is the circular-FDR torus-axis (k*, l*) selector. It
  lives here rather than in `scripts/c0_phase_landscape/` because it loads a
  checkpoint (`load_nt_emb` → `torch.load(--ckpt)`) and imports `fdr_torus_viz`,
  so it is not checkpoint-free first-party tooling.

The training package these import is being vendored into this repo separately as
`parsing_by_maxseminfo/`; the scripts here stay frozen exactly as they were run
for the study.
