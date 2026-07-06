"""Aggregate per-checkpoint SP4 JSONs into a long-format summary.

Reads results/sp4/<lang>.json for a fixed set of languages and emits:
  - results/sp4/summary.parquet  (long-format: lang, condition, metric, value)
  - stdout comparison table grouped by condition
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


LANGS = ["english", "french", "german", "chinese"]
CONDITIONS = ["rule_left", "rule_right", "terminal"]
METRICS = [
    "mean_cos", "std_cos",
    "mean_cos_shuf", "std_cos_shuf",
    "cohen_d",
    "R", "R_shuf", "theta_deg",
    "norm_preserve_max_dev",
]


def main() -> None:
    base = Path("results/sp4")
    rows = []
    wide_rows = []
    for lang in LANGS:
        path = base / f"{lang}.json"
        data = json.loads(path.read_text())
        for cond in CONDITIONS:
            cvals = data["conditions"][cond]
            wide_row = {"lang": lang, "condition": cond, **{m: cvals[m] for m in METRICS}}
            wide_rows.append(wide_row)
            for m in METRICS:
                rows.append({
                    "lang": lang, "condition": cond,
                    "metric": m, "value": cvals[m],
                })

    long_df = pd.DataFrame(rows)
    wide_df = pd.DataFrame(wide_rows)

    out_parquet = base / "summary.parquet"
    long_df.to_parquet(out_parquet, index=False)
    print(f"Wrote {out_parquet} ({len(long_df)} rows)")

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print()
    for cond in CONDITIONS:
        print(f"=== {cond} ===")
        sub = wide_df[wide_df["condition"] == cond].set_index("lang")[
            ["mean_cos", "mean_cos_shuf", "cohen_d",
             "R", "R_shuf", "theta_deg"]
        ]
        print(sub.to_string())
        print()


if __name__ == "__main__":
    main()
