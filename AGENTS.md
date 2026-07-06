# AGENTS.md

Operational brief for coding agents working in this repo. Keep it thin — it points
to the human docs rather than restating them.

## Setup & run

- Docker is the supported environment (see README "Setup"). torch / triton / CUDA come
  from the `nvcr.io` base image; `docker/requirements.txt` adds the rest.
- Train: `python train.py --conf config/<family>/<config>.yaml --device 0 [--seed 0]`
- Evaluate: `python evaluate.py --load_from_dir log/<run_dir> --decode_type mbr [--device 0]`

## Verify (run after any code change)

```bash
python -m pytest tests/test_hn_pcfg_scoring.py tests/test_hn_pcfg_smoke.py \
  tests/test_seminfo_hn_scoring.py tests/test_seminfo_hn_smoke.py
```

CPU-only (~7s), no data or GPU needed. The SemInfo pair (`test_seminfo_hn_*`)
additionally needs the SemInfo deps from `docker/requirements.txt` (`triton` /
`sympy` importable — satisfied in the supported container). The integration tests
(`tests/test_pipeline_components.py`, `tests/test_phase_transitions.py`) require
`optuna` + CUDA + the PTB pickles at
`data/clean/english-{train,val,test}.pickle` and are **not** part of the
default check.

## Conventions

- Branch off `develop`; name branches `fix/` `feature/` `refactor/`; PRs target `develop`.
- Write code comments, docstrings, commit messages, and error messages in **English**.
- Add type hints to new functions/methods. The style target is ruff / black, but
  no repo-local formatter config is checked in; match surrounding style and avoid
  broad formatting-only rewrites unless requested.
- Model dispatch is by `model.model_name` → `parser/helper/util.py:get_model`. Adding a
  model = 3 code touchpoints + a config — see [`docs/adding-a-model.md`](docs/adding-a-model.md).
- One-off, study-specific launchers/configs belong under `archive/`, **not** `scripts/`
  or `config/` (see [`archive/README.md`](archive/README.md)).
- `archive/` is mostly frozen provenance. Do not move or delete archived base configs
  still referenced by live tooling/tests unless you update those references too.
- `fastNLP/` is a vendored, locally patched third-party copy. Do **not** replace it
  with a PyPI install; see [`fastNLP/LOCAL_PATCHES.md`](fastNLP/LOCAL_PATCHES.md).
- `parsing_by_maxseminfo/` is a vendored copy kept **byte-faithful** for reproducibility
  (third-party SemInfo harness + our Hol-PCFG head). Do **not** refactor it; see
  [`parsing_by_maxseminfo/LOCAL_PATCHES.md`](parsing_by_maxseminfo/LOCAL_PATCHES.md).
- Never commit generated outputs: `data/ log/ logs/ results/ runs/ wandb/` and the
  generated `config/sweeps/nt_sdim/` are gitignored — commit only the `*_base.yaml`
  sweep templates. For the full generated-artifact list, check [`.gitignore`](.gitignore).

## Gotchas

- `train.py` catches all exceptions, prints the traceback, deletes the run directory,
  and then **exits 0** — a failed training run still returns success. Do not trust the
  exit code: check stderr / the traceback and whether `<save_dir>/best.pt` was produced.

## Pointers

- [`README.md`](README.md) — repo layout, `model_name` dispatch, train/eval, config schema, experiment families
- [`docs/adding-a-model.md`](docs/adding-a-model.md) — wiring a new model
- [`archive/README.md`](archive/README.md) — what is frozen and why
- [`parsing_by_maxseminfo/LOCAL_PATCHES.md`](parsing_by_maxseminfo/LOCAL_PATCHES.md) — vendored SemInfo harness: what is trimmed/patched and the first-party Hol-PCFG head
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — third-party provenance & attribution (TN-PCFG, SemInfo, fastNLP, Supar)
