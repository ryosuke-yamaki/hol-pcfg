"""Torus visualization helpers used by the sid=1193 renderers.

This module only exposes the small set of helpers consumed by

    render_direct_arrows_sid1193.py
    render_parse_tree_sid1193.py
    render_combined_sid1193.py

The selected (k*, l*) FDR scores live in
``results/c0_phase_landscape/english/n7e2qm8t/label/
fdr_scores_gold_english_phase2_rank3_seed4_0417_173559.json`` and the
renderers just read them; the original FDR-pipeline / pseudo-clustering /
relation-conditioned plotting code that produced that JSON has been
removed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from cossin_pca_viz import phrase_group  # noqa: E402


def compute_phases(w: torch.Tensor) -> np.ndarray:
    """Interior FFT phases (drops k=0 and k=d/2 which are real for real input)."""
    X = torch.fft.rfft(w, dim=-1)
    return torch.angle(X[..., 1:-1]).cpu().numpy().astype(np.float32)


def wrap_to_pi(theta: np.ndarray) -> np.ndarray:
    """Wrap angles into [-pi, pi)."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def trim_png_whitespace(path: Path, threshold: int = 250, pad: int = 4) -> None:
    """Crop a saved PNG to the bbox of its non-white content.

    matplotlib's ``bbox_inches="tight"`` uses the 3D axes' figure-coordinate
    bbox, not the rendered scene's pixel bbox, so 3D plots often retain
    significant whitespace around the actual donut. Silently no-ops if PIL
    is missing.
    """
    try:
        from PIL import Image
    except ImportError:
        return
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    mask = np.any(arr < threshold, axis=-1)
    if not mask.any():
        return
    ys, xs = np.where(mask)
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, arr.shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, arr.shape[1])
    img.crop((x0, y0, x1, y1)).save(path)


def make_groups(
    nt_label: list[str | None],
    nt_support: list[int],
    min_support: int,
    phrase_groups: list[str],
) -> np.ndarray:
    eff = [
        lab if (lab is not None and sup >= min_support) else None
        for lab, sup in zip(nt_label, nt_support)
    ]
    return np.array([phrase_group(lab, phrase_groups) for lab in eff])


def _torus_xyz(
    phi_k: np.ndarray, phi_l: np.ndarray, R_major: float, r_minor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = (R_major + r_minor * np.cos(phi_l)) * np.cos(phi_k)
    y = (R_major + r_minor * np.cos(phi_l)) * np.sin(phi_k)
    z = r_minor * np.sin(phi_l)
    return x, y, z


def _draw_donut_surface(
    ax_3d, R_major: float, r_minor: float, draw_wireframe: bool = True,
) -> None:
    if not draw_wireframe:
        return
    u = np.linspace(-np.pi, np.pi, 60)
    v = np.linspace(-np.pi, np.pi, 60)
    U, V = np.meshgrid(u, v)
    Xs = (R_major + r_minor * np.cos(V)) * np.cos(U)
    Ys = (R_major + r_minor * np.cos(V)) * np.sin(U)
    Zs = r_minor * np.sin(V)
    ax_3d.plot_surface(
        Xs, Ys, Zs,
        color="#b8c5d6", alpha=0.12, linewidth=0,
        antialiased=True, shade=True,
    )
    ax_3d.plot_wireframe(
        Xs, Ys, Zs,
        rstride=4, cstride=4,
        color="#4a4a4a", linewidth=0.7, alpha=0.40,
    )
