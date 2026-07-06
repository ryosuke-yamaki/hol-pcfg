"""Build CHARACTER-level KTB Japanese pickles for HN-PCFG.

Mirrors preprocessing.py but tokenizes each surface morpheme into its
characters so the unsupervised parser must induce morphological segmentation
on its own. Li et al. placeholder tokens ([num], [unk], -LRB-, -RRB-, ...) are
kept atomic (one leaf each), never split into characters.

Each output pickle keeps the usual {'word', 'pos', 'gold_tree'} contract that
parser/helper/data_module.py expects, plus two EXTRA keys used only for the
fair phrase-structure comparison (DataModule ignores unknown keys):

  - morph_offsets : per-sentence cumulative char offsets of morpheme boundaries
                    (off[i] = #chars before morpheme i; len == num_morphemes + 1)
  - morph_gold    : per-sentence gold spans in MORPHEME-index space (identical to
                    the morpheme baseline gold), so char predictions can be
                    projected back to morpheme granularity in
                    scripts/eval_phrase_projection.py.

`gold_tree` itself is the morpheme phrase structure RE-INDEXED to char offsets
(phrase-structure-only, as decided for this experiment); it is used by the
in-training UF1 as a monitoring signal only.

Usage (mirrors preprocessing.py):
    python scripts/preprocess_ktb_char.py \
        --train_file <ktb_len40_nopunct/train> \
        --val_file   <ktb_len40_nopunct/dev> \
        --test_file  <ktb_len40_nopunct/test> \
        --cache_path data/clean/japanese-ktb-char-
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

from nltk import Tree

# preprocessing.py lives at the repo root; mirror dump_predictions.py's path fix.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preprocessing import factorize  # noqa: E402


_ESCAPE = {'(': '-LRB-', ')': '-RRB-'}
# Li et al. placeholders kept atomic: bracketed ([num], [unk]) or PTB-dashed
# (-LRB-, -RRB-, -NONE-, ...). Genuine Japanese morphemes never match these.
_ATOMIC_RE = re.compile(r'^\[[^\]]+\]$|^-[A-Za-z]+-$')


def _escape_char(ch: str) -> str:
    if ch in _ESCAPE:
        return _ESCAPE[ch]
    if ch.isspace():
        return '_'
    return ch


def char_units(morph: str) -> list[str]:
    """Expand one morpheme into character leaves; atomic placeholders stay whole."""
    if _ATOMIC_RE.match(morph):
        return [morph]
    return [_escape_char(c) for c in morph]


def build_char_record(tree: Tree):
    """Return (word, pos, char_gold, off, morph_gold) for one bracketed tree.

    char_gold re-indexes the morpheme phrase spans to char offsets; off is the
    morpheme-boundary char-offset array; morph_gold is the untouched
    morpheme-index gold (== baseline gold) used for projection.
    """
    leaves = tree.pos()                       # [(morpheme, pos), ...]
    morphs = [m for m, _ in leaves]
    poss = [p for _, p in leaves]
    morph_gold = factorize(tree)              # spans over morpheme indices (incl. root)

    units = [char_units(m) for m in morphs]
    off = [0]
    for u in units:
        off.append(off[-1] + len(u))

    word = tuple(u for us in units for u in us)
    pos = tuple(poss[i] for i, us in enumerate(units) for _ in us)
    char_gold = [[off[a], off[b], label] for a, b, label in morph_gold]

    return word, pos, char_gold, off, morph_gold


def create_dataset(file_name: str) -> dict:
    word_array, pos_array = [], []
    gold_trees, morph_offsets, morph_golds = [], [], []
    n_skipped = 0
    with open(file_name, 'r') as f:
        for line in f:
            tree = Tree.fromstring(line)
            word, pos, char_gold, off, morph_gold = build_char_record(tree)
            # Single-morpheme trees have empty gold (factorize returns []). The
            # morpheme baseline drops these via the seq_len==1 rule; at char
            # level seq_len can exceed 1, which would crash UF1 (max of empty
            # gold). Drop them here to stay consistent with the baseline set.
            if not char_gold:
                n_skipped += 1
                continue
            word_array.append(word)
            pos_array.append(pos)
            gold_trees.append(char_gold)
            morph_offsets.append(off)
            morph_golds.append(morph_gold)

    if n_skipped:
        print(f"  [skip] {n_skipped} single-morpheme sentence(s) dropped (empty gold)")

    return {'word': word_array,
            'pos': pos_array,
            'gold_tree': gold_trees,
            'morph_offsets': morph_offsets,
            'morph_gold': morph_golds}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='preprocess KTB bracketed files into character-level pickles.'
    )
    parser.add_argument('--train_file', default='data/ktb-train.txt')
    parser.add_argument('--val_file', default='data/ktb-valid.txt')
    parser.add_argument('--test_file', default='data/ktb-test.txt')
    parser.add_argument('--cache_path', default='data/')
    args = parser.parse_args()

    for split, path in [('train', args.train_file),
                        ('val', args.val_file),
                        ('test', args.test_file)]:
        result = create_dataset(path)
        n_sents = len(result['word'])
        n_chars = sum(len(w) for w in result['word'])
        avg = n_chars / max(n_sents, 1)
        print(f"[{split}] {n_sents} sents, {n_chars} char-tokens (avg {avg:.1f}/sent)")
        with open(args.cache_path + f"{split}.pickle", "wb") as f:
            pickle.dump(result, f)
