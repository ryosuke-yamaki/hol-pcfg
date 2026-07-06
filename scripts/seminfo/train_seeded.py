#!/usr/bin/env python3
"""Seeded, reproducible wrapper around ``parsing_by_maxseminfo.train``.

Why this exists
---------------
The vendored ``parsing_by_maxseminfo/train.py`` entry point is kept **verbatim**
(see ``parsing_by_maxseminfo/LOCAL_PATCHES.md``) and performs **no** seeding of
its own -- it never calls ``lightning.seed_everything``. The ``PL_GLOBAL_SEED``
environment variable that the sweep launcher exports does **not** seed weight
initialization or the Monte-Carlo (``torch.multinomial``) CRF tree sampling, so
bare ``python -m parsing_by_maxseminfo.train ...`` runs are **not** run-to-run
reproducible (identical commands diverge from the first logged step). That
matches how the paper's sweep runs were produced and is left intact.

This wrapper closes the gap without touching the vendored code: it calls
``lightning.seed_everything(seed, workers=True)`` *before* handing control to the
unchanged training entry point. Under a fixed seed on matched hardware/stack the
resulting runs are bit-reproducible (verified bit-identical across runs, with
``PYTHONHASHSEED=0`` exported as below). Python's hash randomization cannot be
disabled from inside a running interpreter, so export ``PYTHONHASHSEED`` in the
shell as well for strict bit-reproduction.

Usage
-----
Identical to ``python -m parsing_by_maxseminfo.train ...`` with a leading
``--seed``; every other argument is passed straight through::

    PYTHONHASHSEED=0 python scripts/seminfo/train_seeded.py --seed 0 \\
        -c config/seminfo/hnpcfg_nt1024_t2048_rank1_seminfo.yaml \\
        --ckpt_dir log/seminfo/en-seed0 --langstr english --ngpu 1
"""

import argparse
import runpy
import sys
from pathlib import Path

import lightning as L

# Make the vendored `parsing_by_maxseminfo` package importable when this script
# is run by path (`python scripts/seminfo/train_seeded.py ...`): the `-m` form
# picks it up from CWD, but a by-path run puts only this file's dir on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    """Seed the RNGs, then execute ``parsing_by_maxseminfo.train`` as ``__main__``."""
    parser = argparse.ArgumentParser(
        description=(
            "Seeded wrapper around `python -m parsing_by_maxseminfo.train`. "
            "Consumes --seed and passes every other argument through unchanged."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Global RNG seed (lightning.seed_everything, workers=True).",
    )
    args, passthrough = parser.parse_known_args()

    L.seed_everything(args.seed, workers=True)
    print(
        f"[train_seeded] lightning.seed_everything({args.seed}, workers=True) called",
        flush=True,
    )

    # Rewrite argv so the vendored entry point's argparse sees only the
    # passthrough args; runpy(alter_sys=True) fixes up argv[0] to train.py.
    sys.argv = [sys.argv[0], *passthrough]
    runpy.run_module("parsing_by_maxseminfo.train", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
