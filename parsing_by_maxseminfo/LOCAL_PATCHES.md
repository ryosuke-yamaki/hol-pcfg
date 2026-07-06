# parsing_by_maxseminfo — vendored copy with local patches

`parsing_by_maxseminfo/` is a **vendored, trimmed** copy of the reference code for

> Junjie Chen, Xiangheng He, Yusuke Miyao, Danushka Bollegala.
> *Improving Unsupervised Constituency Parsing via Maximizing Semantic Information.*
> ICLR 2025. OpenReview: [`qyU5s4fzLg`](https://openreview.net/forum?id=qyU5s4fzLg).

Upstream repository:
<https://github.com/junjiechen-chris/Improving-Unsupervised-Constituency-Parsing-via-Maximizing-Semantic-Information>
(package `parsing_by_maxseminfo`, `version = 0.1.0`, author Junjie Chen). The upstream
code is itself derived from TN-PCFG / Simple-PCFG (Yang et al.); see the repo-root
`THIRD_PARTY_NOTICES.md` for the full third-party lineage that this vendored tree
inherits.

## Why vendored (and why the parser core is duplicated)

The SemInfo training path (RL with a log-TF·IDF reward, CRF sampling, on
`HNPCFG-FixedCostReward` / `SNPCFG-FixedCostReward`) is bound to a reward-aware CKY
engine (`MySPCFGFaster` and siblings) and a deeply Lightning-coupled harness
(`parser/lightning_wrapper/`, RL/maxent warmup callbacks, best-`val/sentence_f1`
checkpoint + `trainer.test(ckpt_path="best")`) that hol-pcfg's own `parser/` does not
provide. Rather than refactor those reward classes onto hol-pcfg's parser core (a
behavior-risking rewrite), we vendor the upstream package **unchanged in package name
and internal layout** so the published SemInfo numbers reproduce bit-for-bit. This is
the same treatment `fastNLP/` gets (see `fastNLP/LOCAL_PATCHES.md`). The resulting
deliberate duplication of the PCFG/CKY core between `parser/` and
`parsing_by_maxseminfo/parser/` is intentional and documented here; a future
parity-gated core merge is possible but explicitly out of scope for this vendoring.

Entry point is unchanged: `python -m parsing_by_maxseminfo.train -c <config> --ckpt_dir <dir> ...`.

## Do not replace with a PyPI install

This package is not published on PyPI. It carries a first-party Hol-PCFG model
(see below) and is trimmed of upstream modules that are dead on the shipped training
path; a PyPI install would not match.

## Trimmed modules (removed vs upstream)

Removed because they are unused on the shipped train/eval path (verified by import
closure — `python -c "import parsing_by_maxseminfo.train"` — and by the tests):

- `parser/cmds/` — the non-Lightning CLI entry points (the shipped entry is the
  top-level `train.py` + `parser/lightning_wrapper/`).
- `parser/helper/lm_datasets/` — `torchtext`-based LM datasets (drops the `torchtext`
  dependency, which is not otherwise needed).
- `parser/helper/loader_wrapper.py`
- `utils/prep.py`, `utils/spanoverlap.py`

`__pycache__/`, `*.pyc`, and the repo-root `openai_key` credential file were never copied.

### Restored after the trim

- `parser/helper/util.py` — restored **verbatim**. It was on the initial drop list but
  is genuinely imported on the train path
  (`parser/helper/pas_grammar_data_helper.py:18: from .util import SpanScorer`). No
  import site was edited.

## Local modifications (vs upstream)

The **only** in-package code change is replacing the `openai_key` JSON credential file
with the `OPENAI_API_KEY` environment variable (offline preprocessing only; never
touched at train time):

- `preprocess/augmenting.py` (~L12): reads `os.environ.get("OPENAI_API_KEY")` and raises
  `RuntimeError` if it is unset, instead of `open("openai_key")`.
- `preprocess/downloading_from_openai.py` (~L25): same change.

The old `openai_key` JSON also carried an `organization` field, which the env replacement
intentionally drops — the OpenAI SDK falls back to the `OPENAI_ORG_ID` env var or the
account's default organization.

## First-party additions (ours, not upstream)

- `parser/model/HN_PCFG.py` — the **Hol-PCFG** (holographic-embedding) model:
  `HNPCFGPairwise` (HolE/hadamard/conv scoring over phase-normalized complex
  embeddings) and `HNPCFGFixedCostReward` (its SemInfo reward head, whose `loss` is
  borrowed from `SNPCFGFixedCostReward`). This is our contribution added onto the
  upstream package.
- `tests/test_seminfo_hn_scoring.py`, `tests/test_seminfo_hn_smoke.py` (in the repo
  `tests/` dir) — cover the above; they import `parsing_by_maxseminfo.parser.model.HN_PCFG`.

When syncing with upstream, re-apply the `OPENAI_API_KEY` change and re-check the import
closure and the two SemInfo Hol-PCFG tests.
