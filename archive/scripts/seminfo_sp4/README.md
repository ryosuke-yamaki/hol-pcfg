# seminfo_sp4

Frozen, as-run snapshot of the SP4 relation-extraction suite from the separate
`hol-pcfg-seminfo` study — the HolE-algebra systematicity verification (SP4)
plus its A1 tangent-space and B1 phase-geometry figures. This is the SemInfo-
trained counterpart to `archive/scripts/sp_verification/` (the SP1–SP4 suite in
this repo).

Provenance only — not maintained. Every analysis script takes a `--ckpt` and
loads it via `torch.load`; they need the study's SemInfo checkpoints, which are
**not** shipped publicly. `aggregate_sp4.py` only collates the per-run JSON they
write.

| File | What it did |
|------|-------------|
| `run_sp4.py` | SP4 HolE-algebra systematicity driver (writes per-relation JSON metrics). |
| `aggregate_sp4.py` | Aggregates the per-run SP4 JSON into a summary table (pandas). |
| `a1_tangent_pca.py`, `a1_viz.py` | A1 tangent-space PCA figures of the relation geometry. |
| `b1a_phase_heatmap.py`, `b1b_phase_scatter.py`, `b1c_phase_dispersion.py` | B1 phase-geometry figures (heatmap / scatter / dispersion). |
| `pca_viz.py` | PCA embedding scatter of the NT / T / relation vectors. |
