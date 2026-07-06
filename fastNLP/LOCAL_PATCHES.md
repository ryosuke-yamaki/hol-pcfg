# fastNLP — vendored copy with local patches

`fastNLP/` is a **vendored** copy of [fastNLP](https://github.com/fastnlp/fastNLP)
(`__version__ = '0.5.6'`). It is imported by the data loader
(`parser/helper/data_module.py`), one active analysis script
(`scripts/c0_phase_landscape/build_symbol_labels.py`), and a few archived scripts under
`archive/scripts/sp_verification/`, using a small surface of `fastNLP.core` (`DataSet`,
`DataSetIter`, `Vocabulary`, `BucketSampler`, `ConstantTokenNumSampler`) — the loader uses
the full set; the analysis scripts use only `DataSet` / `Vocabulary`.

## Do not replace with a PyPI install

Version `0.5.6` is **not published on PyPI** (PyPI has 0.5.5 / 0.6.0 / 1.0.1), and this
copy carries local behavioral patches that the published releases lack. Running
`pip install fastNLP` would silently revert them and change training/eval behavior.

## Local modifications (vs upstream)

`fastNLP/core/vocabulary.py`:

1. **`idx2word` setter fix** (~L121): assigns `self._idx2word = value`.
   Upstream 0.5.5/0.6.0 buggily assign `self._word2idx = value` in this setter.

2. **`construct_vocab` field-shape handling** (~L367-373): the nested-field branch
   `# or not _is_iterable(field[0])` and the `RuntimeError("Only support field with
   2 dimensions.")` guard are commented out, relaxing how non-flat fields are indexed.

When syncing with upstream or upgrading, re-apply these patches and add a regression
check on `Vocabulary` indexing before relying on the result.
