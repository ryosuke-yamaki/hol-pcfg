# seminfo_evals

Frozen, one-off re-evaluation launchers from the separate `hol-pcfg-seminfo`
study. `eval_best_v3_trial192.py` re-runs `trainer.test` with
`ckpt_path=<best.ckpt>` for the v3 Optuna trial-192 runs under the hardcoded
`CKPT_ROOT = ckpt/optuna/hnpcfg-rank1-seminfo-v3`, and `eval_best_ckpts.py`
picks the best `sentence_f1` checkpoint per multilingual run under
`CKPT_ROOT = ckpt/multilingual_n7e2qm8t` and re-evaluates it. Both shell out to
`python -m parsing_by_maxseminfo.train --eval_ckpt ...` and reference checkpoint
directories that are **not** shipped publicly; kept for provenance only.
