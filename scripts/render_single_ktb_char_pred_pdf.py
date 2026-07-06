#!/usr/bin/env python3
"""Render validation raw_idx=1459 (KTB char-level HN-PCFG parse) as PDF + SVG.

Model-free renderer: it reads the cached parse in
``analysis/ktb_char/data/val_1459.parse.json`` and draws the predicted tree. It
does not import torch or the HN-PCFG model, so re-rendering only needs
matplotlib and the cached JSON.

Edit the style with env vars and re-run, e.g.:

    TREE_FONT_SCALE=1.6 SHOW_LABELS=0 python scripts/render_single_ktb_char_pred_pdf.py

Env overrides (with defaults):
  PARSE_JSON      = analysis/ktb_char/data/val_1459.parse.json
  KTB_FONT        = ~/.fonts/unifont.ttf
  TREE_FONT_SCALE = 1.3
  TREE_X_STEP     = 1.25     # horizontal spacing between leaves
  TREE_Y_STEP     = 1.35     # vertical spacing between tree levels
  TITLE_MODE      = none     # "none" or "text"
  SHOW_LABELS     = 1        # 1 keep NT labels, 0 clean tree (leaves only)
  SHOW_PT         = 0        # 1 also show PT=... labels at leaves
  SHOW_CAPTION    = 0        # 1 show the "HN-PCFG predicted (char)" caption
  OMIT_SUFFIX_TEXT =          # if set, drop this exact suffix from the rendered leaves
  OUT_PDF         = analysis/ktb_char/val_1459_pred_tree_seed0.pdf
  OUT_SVG         = analysis/ktb_char/val_1459_pred_tree_seed0.svg
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARSE_JSON = Path(
    os.environ.get(
        "PARSE_JSON",
        ROOT / "analysis" / "ktb_char" / "data" / "val_1459.parse.json",
    )
)
FONT = Path(os.environ.get("KTB_FONT", Path.home() / ".fonts" / "unifont.ttf"))
OUT_PDF = Path(
    os.environ.get(
        "OUT_PDF",
        ROOT / "analysis" / "ktb_char" / "val_1459_pred_tree_seed0.pdf",
    )
)
OUT_SVG = Path(
    os.environ.get(
        "OUT_SVG",
        ROOT / "analysis" / "ktb_char" / "val_1459_pred_tree_seed0.svg",
    )
)
TREE_FONT_SCALE = float(os.environ.get("TREE_FONT_SCALE", "1.3"))
TREE_X_STEP = float(os.environ.get("TREE_X_STEP", "1.25"))
TREE_Y_STEP = float(os.environ.get("TREE_Y_STEP", "1.35"))
TITLE_MODE = os.environ.get("TITLE_MODE", "none")
SHOW_LABELS = os.environ.get("SHOW_LABELS", "1") != "0"
SHOW_PT = os.environ.get("SHOW_PT", "0") != "0"
SHOW_CAPTION = os.environ.get("SHOW_CAPTION", "0") != "0"
OMIT_SUFFIX_TEXT = os.environ.get("OMIT_SUFFIX_TEXT", "")

_LEAF_DISPLAY_MAP = {"-LRB-": "(", "-RRB-": ")"}


def _display_leaf(token: str) -> str:
    return _LEAF_DISPLAY_MAP.get(token, token)


def build_binary_tree(spans, n):
    """Map each non-trivial internal span -> (left, right); leaves -> None."""
    span_set = set(spans) | {(i, i + 1) for i in range(n)} | {(0, n)}
    children: dict = {}
    for i, j in sorted(span_set, key=lambda x: x[1] - x[0]):
        if j - i == 1:
            children[(i, j)] = None
            continue
        split = None
        for k in range(i + 1, j):
            if (i, k) in span_set and (k, j) in span_set:
                split = k
                break
        if split is None:
            split = i + 1
            span_set.add((i, split))
            span_set.add((split, j))
            children.setdefault((i, split), None)
            if (split, j) not in children and j - split == 1:
                children[(split, j)] = None
        children[(i, j)] = ((i, split), (split, j))
    return children


def layout_tree(children, root):
    """Recursively compute (x, y) for each node. y=0 at root, grows downward."""
    pos: dict = {}

    def recurse(span, depth):
        child = children.get(span)
        if child is None:
            x = (span[0] + span[1]) / 2.0
            pos[span] = (x, depth)
            return x
        left_x = recurse(child[0], depth + 1)
        right_x = recurse(child[1], depth + 1)
        x = (left_x + right_x) / 2.0
        pos[span] = (x, depth)
        return x

    recurse(root, 0)
    return pos


def crop_pdf_to_content(path: Path, pad: int = 2) -> None:
    """Crop a PDF to its ink bounding box via Ghostscript (no-op if gs missing)."""
    if shutil.which("gs") is None:
        print("[warn] ghostscript not found; PDF left uncropped", file=sys.stderr)
        return
    bbox = subprocess.run(
        ["gs", "-q", "-dBATCH", "-dNOPAUSE", "-sDEVICE=bbox", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(
        r"%%BoundingBox:\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)",
        bbox.stderr,
    )
    if not match:
        return
    x0, y0, x1, y1 = (int(v) for v in match.groups())
    x0 -= pad
    y0 -= pad
    x1 += pad
    y1 += pad
    width, height = x1 - x0, y1 - y0
    tmp = Path(str(path) + ".tmp")
    subprocess.run(
        [
            "gs",
            "-q",
            "-o",
            str(tmp),
            "-sDEVICE=pdfwrite",
            f"-dDEVICEWIDTHPOINTS={width}",
            f"-dDEVICEHEIGHTPOINTS={height}",
            "-dFIXEDMEDIA",
            "-c",
            f"<</PageOffset [{-x0} {-y0}]>> setpagedevice",
            "-f",
            str(path),
        ],
        capture_output=True,
        check=False,
    )
    if tmp.exists():
        shutil.move(str(tmp), str(path))


def font_prop(size):
    if FONT.is_file():
        return font_manager.FontProperties(fname=str(FONT), size=size)
    return font_manager.FontProperties(family="monospace", size=size)


def main() -> int:
    rec = json.loads(PARSE_JSON.read_text(encoding="utf-8"))
    leaves = rec["leaf_words"]
    raw_text = rec.get("raw_text", "")
    pred_spans = [(int(i), int(j)) for i, j in rec["pred_spans"]]
    nt_labels = {(int(i), int(j)): int(v) for i, j, v in rec.get("nt_labels", [])}
    pt_labels = [int(x) for x in rec.get("pt_labels", [])]

    if OMIT_SUFFIX_TEXT:
        rendered_text = "".join(leaves)
        if not rendered_text.endswith(OMIT_SUFFIX_TEXT):
            raise ValueError(
                f"OMIT_SUFFIX_TEXT={OMIT_SUFFIX_TEXT!r} is not a suffix of "
                f"the rendered sentence {rendered_text!r}"
            )
        cut = len(rendered_text) - len(OMIT_SUFFIX_TEXT)
        leaves = leaves[:cut]
        raw_text = rendered_text[:cut]
        pred_spans = [(i, j) for i, j in pred_spans if j <= cut]
        nt_labels = {span: nt for span, nt in nt_labels.items() if span[1] <= cut}
        pt_labels = pt_labels[:cut]

    n = len(leaves)

    if FONT.is_file():
        font_manager.fontManager.addfont(str(FONT))
    else:
        print(f"[warn] font not found at {FONT}; glyphs may render as boxes", file=sys.stderr)
    matplotlib.rcParams["svg.fonttype"] = "path"

    scale = TREE_FONT_SCALE
    children = build_binary_tree(pred_spans, n)
    pos = layout_tree(children, (0, n))
    pos = {
        span: (x * TREE_X_STEP, y * TREE_Y_STEP)
        for span, (x, y) in pos.items()
    }
    max_depth = max(p[1] for p in pos.values())

    fig_w = max(6.0, 0.55 * n * TREE_X_STEP + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 1.5 + 0.55 * max_depth))

    for span, child in children.items():
        if child is None:
            continue
        x0, y0 = pos[span]
        for c in child:
            x1, y1 = pos[c]
            ax.plot([x0, x1], [-y0, -y1], color="black", linewidth=1.0, zorder=1)

    if SHOW_LABELS:
        for span, (x, y) in pos.items():
            if children.get(span) is None:
                continue
            nt_id = nt_labels.get(span)
            label = f"NT={nt_id}" if nt_id is not None else "X"
            ax.text(
                x,
                -y + 0.05,
                label,
                ha="center",
                va="bottom",
                fontsize=12 * scale,
                color="black",
                zorder=3,
                bbox={
                    "boxstyle": "round,pad=0.08",
                    "facecolor": "white",
                    "edgecolor": "none",
                },
            )

    pt_y = -(max_depth + 0.15)
    leaf_y = -(max_depth + 0.3)
    leaf_kwargs = {
        "ha": "center",
        "va": "top",
        "fontsize": 14 * scale,
        "fontproperties": font_prop(14 * scale),
    }
    for i, token in enumerate(leaves):
        if SHOW_PT and i < len(pt_labels):
            ax.text(
                i + 0.5,
                pt_y,
                f"PT={pt_labels[i]}",
                ha="center",
                va="top",
                fontsize=10 * scale,
                color="darkgreen",
                family="monospace",
            )
        ax.text((i + 0.5) * TREE_X_STEP, leaf_y, _display_leaf(token), **leaf_kwargs)

    y_top = 0.45
    ax.set_xlim(-0.5 * TREE_X_STEP, (n + 0.5) * TREE_X_STEP)
    ax.set_ylim(leaf_y - 0.7, y_top)
    ax.set_axis_off()
    if SHOW_CAPTION:
        ax.text(
            -0.4,
            y_top,
            "HN-PCFG predicted (char)",
            fontsize=11 * scale,
            color="gray",
            ha="left",
            va="top",
        )
    if TITLE_MODE == "text" and raw_text:
        ax.set_title(raw_text, loc="center", fontsize=34, fontproperties=font_prop(34), pad=6)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.05)
    crop_pdf_to_content(OUT_PDF)
    plt.close(fig)
    print(
        f"[info] rendered #{rec.get('raw_idx')} (scale={scale}, x_step={TREE_X_STEP}, y_step={TREE_Y_STEP}, "
        f"labels={'on' if SHOW_LABELS else 'off'}, title={TITLE_MODE}, "
        f"omit_suffix={OMIT_SUFFIX_TEXT!r}) to:\n"
        f"  {OUT_PDF}\n  {OUT_SVG}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
