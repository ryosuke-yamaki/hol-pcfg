"""Shared utilities for SP1/SP2/SP3/SP4 verification scripts.

Common functions extracted from SP3 analysis scripts, plus new functions
for inverse relations (SP1) and relation composition (SP2).
"""

from pathlib import Path

import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = (
    "log/hn_pcfg_allproj_cnorm_tau/HNPCFG2026-03-31-12_43_40/best.pt"
)
NT = 4096
T = 8192
S_DIM = 512


# ──────────────────────────────────────────────────────────────────────
# Core math operations
# ──────────────────────────────────────────────────────────────────────
def circular_correlation(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    """Circular correlation a ⋆ b = IFFT(conj(FFT(a)) * FFT(b)).

    Used for relation extraction: r_ext = e_A ⋆ e_B.
    """
    a_f = torch.fft.rfft(a, dim=-1)
    b_f = torch.fft.rfft(b, dim=-1)
    return torch.fft.irfft(a_f.conj() * b_f, n=n, dim=-1)


def circular_convolution(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    """Circular convolution a * b = IFFT(FFT(a) * FFT(b)).

    Used for template computation and relation composition.
    Note: differs from correlation by the absence of conj().
    """
    a_f = torch.fft.rfft(a, dim=-1)
    b_f = torch.fft.rfft(b, dim=-1)
    return torch.fft.irfft(a_f * b_f, n=n, dim=-1)


def compute_inverse_relation(v: torch.Tensor, s_dim: int) -> torch.Tensor:
    """Compute inverse relation v⁻¹ under cnorm (|FFT(v)[k]|=1).

    v⁻¹ = IFFT(conj(FFT(v))), since 1/FFT(v)[k] = conj(FFT(v)[k])
    when |FFT(v)[k]| = 1.

    Returns:
        Tensor with same shape as v.
    """
    v_f = torch.fft.rfft(v, dim=-1)
    return torch.fft.irfft(v_f.conj(), n=s_dim, dim=-1)


# ──────────────────────────────────────────────────────────────────────
# HolE scoring
# ──────────────────────────────────────────────────────────────────────
def compute_rule_probs(
    v: torch.Tensor,
    nt_emb: torch.Tensor,
    all_emb: torch.Tensor,
    tau: float,
    s_dim: int,
) -> torch.Tensor:
    """Compute P(child | parent, relation) via HolE scoring + softmax.

    Args:
        v: (s_dim,) relation vector
        nt_emb: (NT, s_dim) nonterminal embeddings
        all_emb: (NT+T, s_dim) all entity embeddings
        tau: temperature scalar
        s_dim: embedding dimension

    Returns:
        (NT+T, NT) probability matrix, softmax over children (dim=0)
    """
    v_f = torch.fft.rfft(v.unsqueeze(0), dim=-1)          # (1, F)
    parent_f = torch.fft.rfft(nt_emb, dim=-1)             # (NT, F)
    template = torch.fft.irfft(
        v_f.unsqueeze(1) * parent_f.unsqueeze(0),
        n=s_dim, dim=-1
    )                                                       # (1, NT, s_dim)
    scores = torch.einsum("cs, rps -> rcp", all_emb, template)  # (1, C, NT)
    scores = scores.squeeze(0) * tau                        # (C, NT)
    return scores.softmax(dim=0)


def compute_raw_scores(
    v: torch.Tensor,
    nt_emb: torch.Tensor,
    all_emb: torch.Tensor,
    s_dim: int,
) -> torch.Tensor:
    """Compute raw HolE scores WITHOUT temperature or softmax.

    Returns:
        (NT+T, NT) raw score matrix: S[b, a] = e_b^T circonv(v, e_a)
    """
    v_f = torch.fft.rfft(v.unsqueeze(0), dim=-1)
    parent_f = torch.fft.rfft(nt_emb, dim=-1)
    template = torch.fft.irfft(
        v_f.unsqueeze(1) * parent_f.unsqueeze(0),
        n=s_dim, dim=-1
    )
    scores = torch.einsum("cs, rps -> rcp", all_emb, template)
    return scores.squeeze(0)


# ──────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────────────
def load_checkpoint(
    path: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
) -> dict:
    """Load checkpoint and extract key tensors.

    Returns dict with: emb, nt_emb, all_emb, v_left, v_right, tau, ckpt
    """
    dev = torch.device(device)
    ckpt = torch.load(path, map_location=dev, weights_only=True)

    emb = ckpt["rule_state_emb"]                    # (NT+T, s_dim)
    v_left = ckpt["v_left"]
    v_right = ckpt["v_right"]
    if v_left.dim() == 2:
        v_left = v_left.squeeze(0)
    if v_right.dim() == 2:
        v_right = v_right.squeeze(0)
    tau = ckpt["log_tau"].exp().item()

    return {
        "emb": emb,
        "nt_emb": emb[:NT],
        "all_emb": emb,
        "v_left": v_left,
        "v_right": v_right,
        "tau": tau,
        "ckpt": ckpt,
    }


# ──────────────────────────────────────────────────────────────────────
# Normalization verification
# ──────────────────────────────────────────────────────────────────────
def verify_normalization(ckpt: dict, s_dim: int = S_DIM) -> dict:
    """Verify freq_cnorm and relation projection cnorm conditions.

    Returns dict with verification metrics.
    """
    emb = ckpt["rule_state_emb"]
    v_left = ckpt["v_left"]
    v_right = ckpt["v_right"]
    if v_left.dim() == 2:
        v_left = v_left.squeeze(0)
    if v_right.dim() == 2:
        v_right = v_right.squeeze(0)

    results = {}

    # Entity: |FFT(e)[k]| = 1
    emb_f = torch.fft.rfft(emb, dim=-1)
    emb_mag = emb_f.abs()
    results["entity_fft_mag_mean"] = emb_mag.mean().item()
    results["entity_fft_mag_std"] = emb_mag.std().item()
    results["entity_max_deviation"] = (emb_mag - 1.0).abs().max().item()

    # Relations: |FFT(v)[k]| = 1
    for name, v in [("v_left", v_left), ("v_right", v_right)]:
        v_f = torch.fft.rfft(v.unsqueeze(0), dim=-1)
        v_mag = v_f.abs()
        results[f"{name}_fft_mag_mean"] = v_mag.mean().item()
        results[f"{name}_max_deviation"] = (v_mag - 1.0).abs().max().item()

    # Auto-correlation: e ⋆ e ≈ δ
    sample_idx = torch.randint(0, emb.shape[0], (100,))
    sample = emb[sample_idx]
    auto_corr = circular_correlation(sample, sample, s_dim)
    peak = auto_corr[:, 0].mean().item()
    off_peak = auto_corr[:, 1:].abs().mean().item()
    results["autocorr_peak"] = peak
    results["autocorr_off_peak"] = off_peak
    results["autocorr_ratio"] = peak / max(off_peak, 1e-12)

    return results


def print_normalization(results: dict) -> None:
    """Pretty-print normalization verification results."""
    print("\n" + "=" * 60)
    print("  Normalization Verification")
    print("=" * 60)
    print(f"\n  Entity (freq_cnorm):")
    print(f"    |FFT(e)[k]| mean={results['entity_fft_mag_mean']:.8f}  "
          f"std={results['entity_fft_mag_std']:.8f}")
    print(f"    max deviation from 1: {results['entity_max_deviation']:.2e}")
    for name in ["v_left", "v_right"]:
        print(f"\n  {name} (projection cnorm):")
        print(f"    |FFT(v)[k]| mean={results[f'{name}_fft_mag_mean']:.8f}")
        print(f"    max deviation from 1: {results[f'{name}_max_deviation']:.2e}")
    print(f"\n  Auto-correlation e ⋆ e:")
    print(f"    peak={results['autocorr_peak']:.6f}  "
          f"off-peak={results['autocorr_off_peak']:.6f}  "
          f"ratio={results['autocorr_ratio']:.1f}")


# ──────────────────────────────────────────────────────────────────────
# NT label loading
# ──────────────────────────────────────────────────────────────────────
def load_nt_labels(path: str = "results/sp3/nt_labels.pkl") -> dict | None:
    """Load NT/T labels from pickle, returning None if not found."""
    p = Path(path)
    if not p.exists():
        print(f"  Warning: {path} not found. Skipping label-based analysis.")
        return None
    import pickle
    with open(p, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────
# Matplotlib setup
# ──────────────────────────────────────────────────────────────────────
def setup_matplotlib() -> None:
    """Configure matplotlib for ACL-style publication figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9,
        "font.family": "sans-serif",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "svg",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def save_svg(fig, output_dir: Path, name: str) -> None:
    """Save figure as SVG and print confirmation."""
    path = output_dir / f"{name}.svg"
    fig.savefig(path, format="svg")
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────
def _self_test() -> None:
    """Run self-tests for all utility functions."""
    print("Running sp_utils self-tests...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test 1: circular_convolution(v, v_inv) ≈ δ
    print("\n  Test 1: circonv(v, v_inv) ≈ δ")
    v = torch.randn(S_DIM, device=device)
    # Apply cnorm to v
    v_f = torch.fft.rfft(v)
    v_f = v_f / v_f.abs().clamp(min=1e-12)
    v = torch.fft.irfft(v_f, n=S_DIM)

    v_inv = compute_inverse_relation(v, S_DIM)
    delta = circular_convolution(v, v_inv, S_DIM)
    peak = delta[0].item()
    off_peak_max = delta[1:].abs().max().item()
    print(f"    peak={peak:.6f}  off_peak_max={off_peak_max:.2e}")
    assert abs(peak - 1.0) < 1e-5, f"Peak should be ~1.0, got {peak}"
    assert off_peak_max < 1e-5, f"Off-peak should be ~0, got {off_peak_max}"
    print("    PASSED")

    # Test 2: double inverse = identity
    print("\n  Test 2: inverse(inverse(v)) ≈ v")
    v_inv_inv = compute_inverse_relation(v_inv, S_DIM)
    max_diff = (v - v_inv_inv).abs().max().item()
    print(f"    max |v - inv(inv(v))| = {max_diff:.2e}")
    assert max_diff < 1e-5, f"Double inverse should recover v, diff={max_diff}"
    print("    PASSED")

    # Test 3: correlation vs convolution relationship
    # a ⋆ b = circonv(conj_reverse(a), b) where conj_reverse flips and conjugates
    # For real signals: a ⋆ b = circonv(a_reversed, b) where a_reversed[k] = a[-k]
    print("\n  Test 3: correlation = convolution with reversed first arg")
    a = torch.randn(S_DIM, device=device)
    b = torch.randn(S_DIM, device=device)
    corr = circular_correlation(a, b, S_DIM)
    a_rev = torch.roll(a.flip(0), 1)  # reverse and shift
    conv_rev = circular_convolution(a_rev, b, S_DIM)
    max_diff = (corr - conv_rev).abs().max().item()
    print(f"    max |corr(a,b) - conv(rev(a), b)| = {max_diff:.2e}")
    assert max_diff < 1e-4, f"Should match, diff={max_diff}"
    print("    PASSED")

    # Test 4: forward score = inverse score (algebraic identity)
    print("\n  Test 4: e_B^T circonv(v, e_A) = e_A^T circonv(v_inv, e_B)")
    e_A = torch.randn(S_DIM, device=device)
    e_B = torch.randn(S_DIM, device=device)
    # Apply cnorm
    for e in [e_A, e_B]:
        ef = torch.fft.rfft(e)
        ef = ef / ef.abs().clamp(min=1e-12)
        e.data = torch.fft.irfft(ef, n=S_DIM)

    fwd = e_B @ circular_convolution(v, e_A, S_DIM)
    inv = e_A @ circular_convolution(v_inv, e_B, S_DIM)
    diff = abs(fwd.item() - inv.item())
    print(f"    forward={fwd.item():.6f}  inverse={inv.item():.6f}  diff={diff:.2e}")
    assert diff < 1e-4, f"Scores should match, diff={diff}"
    print("    PASSED")

    # Test 5: load checkpoint
    print("\n  Test 5: load checkpoint")
    ckpt_path = Path(DEFAULT_CHECKPOINT)
    if ckpt_path.exists():
        data = load_checkpoint(str(ckpt_path), device)
        print(f"    emb: {data['emb'].shape}, v_left: {data['v_left'].shape}, tau={data['tau']:.4f}")
        norm_results = verify_normalization(data["ckpt"])
        print_normalization(norm_results)
        print("    PASSED")
    else:
        print(f"    Skipped (checkpoint not found at {ckpt_path})")

    print("\n  All tests passed!")


if __name__ == "__main__":
    _self_test()
