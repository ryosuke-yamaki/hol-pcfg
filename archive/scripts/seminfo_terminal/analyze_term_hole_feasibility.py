"""Analyze feasibility of replacing term_mlp with HolE scoring.

Compare score magnitudes and softmax distributions for terminal scoring:
1. Current term_mlp (baseline)
2. HolE without freq_cnorm on vocab_emb
3. HolE with freq_cnorm on vocab_emb
4. HolE with various tau values

Uses a trained HN-PCFG checkpoint (allproj-cnorm-tau).
"""

import sys
import math
import torch
import numpy as np

sys.path.insert(0, "/workspace/hol-pcfg")


def freq_cnorm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply frequency-domain circular normalization: |FFT(x)[k]| = 1."""
    x_f = torch.fft.rfft(x, dim=dim)
    x_f = x_f / x_f.abs().clamp(min=1e-12)
    return torch.fft.irfft(x_f, n=x.shape[dim], dim=dim)


def compute_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute entropy of softmax(logits) along dim."""
    p = logits.softmax(dim=dim)
    log_p = logits.log_softmax(dim=dim)
    return -(p * log_p).sum(dim=dim)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt_path = "log/hn_pcfg_allproj_cnorm_tau/HNPCFG2026-03-31-12_43_40/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt

    # Extract model parameters
    rule_state_emb = state["rule_state_emb"]  # (NT+T, s_dim)
    vocab_emb = state["vocab_emb"]            # (s_dim, V)
    v_left = state["v_left"]                  # (R, s_dim)
    v_right = state["v_right"]                # (R, s_dim)

    # Get model dimensions from config or checkpoint
    NT = 4096
    T = 8192
    s_dim = rule_state_emb.shape[1]
    V = vocab_emb.shape[1]

    print(f"Model: NT={NT}, T={T}, s_dim={s_dim}, V={V}")
    print(f"rule_state_emb: {rule_state_emb.shape}")
    print(f"vocab_emb: {vocab_emb.shape}")
    print(f"v_left: {v_left.shape}")
    print()

    # Extract term embeddings (already freq_cnorm'd in checkpoint)
    term_emb = rule_state_emb[NT:]  # (T, s_dim)

    # Verify freq_cnorm on term_emb
    term_f = torch.fft.rfft(term_emb, dim=-1)
    term_f_mag = term_f.abs()
    print("=== Verification: term_emb freq_cnorm ===")
    print(f"  |FFT(term_emb)[k]| mean: {term_f_mag.mean():.6f}")
    print(f"  |FFT(term_emb)[k]| max deviation from 1: {(term_f_mag - 1).abs().max():.2e}")
    print(f"  ||term_emb|| mean: {term_emb.norm(dim=-1).mean():.4f}")
    print()

    # Check vocab_emb statistics
    vocab_emb_t = vocab_emb.T  # (V, s_dim) for analysis
    vocab_f = torch.fft.rfft(vocab_emb_t, dim=-1)
    vocab_f_mag = vocab_f.abs()
    print("=== vocab_emb statistics (raw, no cnorm) ===")
    print(f"  ||vocab_emb_w|| mean: {vocab_emb_t.norm(dim=-1).mean():.4f}")
    print(f"  ||vocab_emb_w|| std:  {vocab_emb_t.norm(dim=-1).std():.4f}")
    print(f"  ||vocab_emb_w|| min:  {vocab_emb_t.norm(dim=-1).min():.4f}")
    print(f"  ||vocab_emb_w|| max:  {vocab_emb_t.norm(dim=-1).max():.4f}")
    print(f"  |FFT(vocab_emb)[k]| mean: {vocab_f_mag.mean():.4f}")
    print(f"  |FFT(vocab_emb)[k]| std:  {vocab_f_mag.std():.4f}")
    print()

    # Use v_left as proxy for v_term (both are relation vectors)
    v_term = v_left.squeeze(0) if v_left.dim() > 1 else v_left  # (s_dim,)
    v_f = torch.fft.rfft(v_term, dim=-1)
    v_f_mag = v_f.abs()
    print(f"=== v_left (proxy for v_term) ===")
    print(f"  ||v_left||: {v_term.norm():.4f}")
    print(f"  |FFT(v_left)[k]| mean: {v_f_mag.mean():.6f}")
    print(f"  |FFT(v_left)[k]| max deviation from 1: {(v_f_mag - 1).abs().max():.2e}")
    print()

    # Also create a random v_term (Xavier init, then cnorm'd)
    bnd = math.sqrt(6.0 / (1 + s_dim))
    v_term_random = torch.empty(s_dim, device=device).uniform_(-bnd, bnd)
    v_term_random = freq_cnorm(v_term_random)

    # =========================================================
    # Experiment 1: Current term_mlp baseline
    # =========================================================
    print("=" * 70)
    print("=== Experiment 1: Current term_mlp (baseline) ===")

    # Load term_mlp weights and reconstruct
    from parser.modules.res import ResLayer
    import torch.nn as nn

    term_mlp = nn.Sequential(
        nn.Linear(s_dim, s_dim),
        ResLayer(s_dim, s_dim),
        ResLayer(s_dim, s_dim),
        ResLayer(s_dim, s_dim),
    ).to(device)

    # Load term_mlp state dict
    term_mlp_state = {
        k.replace("term_mlp.", ""): v
        for k, v in state.items()
        if k.startswith("term_mlp.")
    }
    term_mlp.load_state_dict(term_mlp_state)
    term_mlp.eval()

    with torch.no_grad():
        mlp_output = term_mlp(term_emb) + term_emb  # (T, s_dim)
        logits_mlp = mlp_output @ vocab_emb          # (T, V)

    # Check if tau_term exists
    tau_term = 1.0
    if "log_tau_term" in state:
        tau_term = state["log_tau_term"].exp().item()
    elif "log_tau" in state:
        tau_term = state["log_tau"].exp().item()
    print(f"  tau_term: {tau_term:.4f}")

    logits_mlp_scaled = logits_mlp * tau_term

    entropy_mlp = compute_entropy(logits_mlp_scaled, dim=-1)
    max_entropy = math.log(V)

    print(f"  ||mlp_output|| mean: {mlp_output.norm(dim=-1).mean():.4f}")
    print(f"  |FFT(mlp_output)[k]| mean: {torch.fft.rfft(mlp_output, dim=-1).abs().mean():.4f}")
    print(f"  logits (before tau): mean={logits_mlp.mean():.4f}, std={logits_mlp.std():.4f}")
    print(f"  logits (before tau): min={logits_mlp.min():.4f}, max={logits_mlp.max():.4f}")
    print(f"  logits (after tau):  mean={logits_mlp_scaled.mean():.4f}, std={logits_mlp_scaled.std():.4f}")
    print(f"  logits (after tau):  min={logits_mlp_scaled.min():.4f}, max={logits_mlp_scaled.max():.4f}")
    print(f"  softmax entropy: mean={entropy_mlp.mean():.4f}, std={entropy_mlp.std():.4f}")
    print(f"  softmax entropy: min={entropy_mlp.min():.4f}, max={entropy_mlp.max():.4f}")
    print(f"  max possible entropy (uniform over V={V}): {max_entropy:.4f}")
    print(f"  entropy ratio (mean/max): {entropy_mlp.mean() / max_entropy:.4f}")
    print(f"  P(top-1) mean: {logits_mlp_scaled.softmax(-1).max(-1).values.mean():.4f}")
    print(f"  P(top-1) > 0.9 ratio: {(logits_mlp_scaled.softmax(-1).max(-1).values > 0.9).float().mean():.4f}")
    print()

    # =========================================================
    # Experiment 2: HolE without freq_cnorm on vocab_emb
    # =========================================================
    print("=" * 70)
    print("=== Experiment 2: HolE terminal (vocab_emb RAW, no cnorm) ===")

    with torch.no_grad():
        # circonv(v_term, term_emb) using learned v_left as proxy
        template_raw = torch.fft.irfft(
            v_f.unsqueeze(0) * term_f, n=s_dim, dim=-1
        )  # (T, s_dim)

        logits_hole_raw = template_raw @ vocab_emb  # (T, V)

    print(f"  ||template|| mean: {template_raw.norm(dim=-1).mean():.4f}")
    print(f"  |FFT(template)[k]| mean: {torch.fft.rfft(template_raw, dim=-1).abs().mean():.6f}")
    print(f"  logits (no tau): mean={logits_hole_raw.mean():.4f}, std={logits_hole_raw.std():.4f}")
    print(f"  logits (no tau): min={logits_hole_raw.min():.4f}, max={logits_hole_raw.max():.4f}")

    # Try various tau values
    for tau_val in [1.0, 5.0, 10.0, 20.0, tau_term]:
        logits_scaled = logits_hole_raw * tau_val
        ent = compute_entropy(logits_scaled, dim=-1)
        p_top1 = logits_scaled.softmax(-1).max(-1).values
        print(f"  tau={tau_val:6.1f}: logits std={logits_scaled.std():.2f}, "
              f"entropy={ent.mean():.4f} ({ent.mean()/max_entropy:.3f}), "
              f"P(top1)={p_top1.mean():.4f}, "
              f"P(top1)>0.9: {(p_top1>0.9).float().mean():.3f}")
    print()

    # =========================================================
    # Experiment 3: HolE with freq_cnorm on vocab_emb
    # =========================================================
    print("=" * 70)
    print("=== Experiment 3: HolE terminal (vocab_emb with freq_cnorm) ===")

    with torch.no_grad():
        # Apply freq_cnorm to vocab_emb
        vocab_emb_cnorm = freq_cnorm(vocab_emb_t, dim=-1).T  # (s_dim, V)

        # Verify cnorm
        vc_f = torch.fft.rfft(vocab_emb_cnorm.T, dim=-1)
        print(f"  |FFT(vocab_cnorm)[k]| mean: {vc_f.abs().mean():.6f}")
        print(f"  |FFT(vocab_cnorm)[k]| max dev: {(vc_f.abs() - 1).abs().max():.2e}")
        print(f"  ||vocab_cnorm_w|| mean: {vocab_emb_cnorm.T.norm(dim=-1).mean():.4f}")

        logits_hole_cnorm = template_raw @ vocab_emb_cnorm  # (T, V)

    print(f"  logits (no tau): mean={logits_hole_cnorm.mean():.4f}, std={logits_hole_cnorm.std():.4f}")
    print(f"  logits (no tau): min={logits_hole_cnorm.min():.4f}, max={logits_hole_cnorm.max():.4f}")

    for tau_val in [1.0, 5.0, 10.0, 20.0, tau_term]:
        logits_scaled = logits_hole_cnorm * tau_val
        ent = compute_entropy(logits_scaled, dim=-1)
        p_top1 = logits_scaled.softmax(-1).max(-1).values
        print(f"  tau={tau_val:6.1f}: logits std={logits_scaled.std():.2f}, "
              f"entropy={ent.mean():.4f} ({ent.mean()/max_entropy:.3f}), "
              f"P(top1)={p_top1.mean():.4f}, "
              f"P(top1)>0.9: {(p_top1>0.9).float().mean():.3f}")
    print()

    # =========================================================
    # Experiment 4: Comparison of score magnitudes
    # =========================================================
    print("=" * 70)
    print("=== Experiment 4: Score magnitude comparison ===")
    print()

    # Rule scoring for comparison
    with torch.no_grad():
        nonterm_emb = rule_state_emb[:NT]
        all_emb = rule_state_emb
        rule_template = torch.fft.irfft(
            v_f.unsqueeze(0) * torch.fft.rfft(nonterm_emb, dim=-1),
            n=s_dim, dim=-1
        )  # (NT, s_dim)
        rule_scores = all_emb @ rule_template.T  # (NT+T, NT)

    print(f"  Rule scores (before tau):   mean={rule_scores.mean():.4f}, "
          f"std={rule_scores.std():.4f}, "
          f"min={rule_scores.min():.4f}, max={rule_scores.max():.4f}")
    print(f"  Term HolE raw (before tau): mean={logits_hole_raw.mean():.4f}, "
          f"std={logits_hole_raw.std():.4f}, "
          f"min={logits_hole_raw.min():.4f}, max={logits_hole_raw.max():.4f}")
    print(f"  Term HolE cnorm (no tau):   mean={logits_hole_cnorm.mean():.4f}, "
          f"std={logits_hole_cnorm.std():.4f}, "
          f"min={logits_hole_cnorm.min():.4f}, max={logits_hole_cnorm.max():.4f}")
    print(f"  Term MLP (before tau):      mean={logits_mlp.mean():.4f}, "
          f"std={logits_mlp.std():.4f}, "
          f"min={logits_mlp.min():.4f}, max={logits_mlp.max():.4f}")
    print()

    # Compare the theoretical score std for each case
    # Under freq_cnorm: ||e|| = 1, score = (1/d) Σ w[k] cos(phase_diff) → std ≈ 1/sqrt(d)
    theory_std_cnorm = 1.0 / math.sqrt(s_dim)
    print(f"  Theoretical score std (both cnorm): {theory_std_cnorm:.4f}")
    print(f"  Actual rule score std:              {rule_scores.std():.4f}")
    print(f"  Actual term HolE cnorm score std:   {logits_hole_cnorm.std():.4f}")
    print(f"  Actual term HolE raw score std:     {logits_hole_raw.std():.4f}")
    print()

    # =========================================================
    # Experiment 5: Norm amplification analysis
    # =========================================================
    print("=" * 70)
    print("=== Experiment 5: Norm amplification (the core concern) ===")
    print()

    with torch.no_grad():
        # template = circonv(v_term, term_emb): both cnorm'd → ||template|| = 1
        print(f"  ||circonv(v_term, term_emb)|| mean: {template_raw.norm(dim=-1).mean():.4f}")
        print(f"  ||vocab_emb_w|| (raw) mean:         {vocab_emb_t.norm(dim=-1).mean():.4f}")
        print(f"  ||vocab_emb_w|| (cnorm) mean:       {vocab_emb_cnorm.T.norm(dim=-1).mean():.4f}")
        print()

        # Score = template^T @ vocab_emb_w
        # |score| ≈ ||template|| * ||vocab_emb_w|| * cos(angle)
        # With raw vocab: ||template||=1, ||vocab||≈X → scores amplified by X
        # With cnorm vocab: ||template||=1, ||vocab||=1 → scores bounded

        # Show the amplification factor
        raw_norm = vocab_emb_t.norm(dim=-1)
        print(f"  Amplification factor (||vocab_raw|| / ||vocab_cnorm||):")
        print(f"    mean: {(raw_norm / 1.0).mean():.4f}")
        print(f"    std:  {(raw_norm / 1.0).std():.4f}")
        print(f"    max:  {(raw_norm / 1.0).max():.4f}")
        print()

        # What tau is needed to match MLP entropy with each HolE variant?
        target_entropy = entropy_mlp.mean().item()
        print(f"  Target entropy (MLP baseline): {target_entropy:.4f}")
        print()

        # Binary search for tau that matches MLP entropy
        for label, logits_base in [
            ("HolE raw vocab", logits_hole_raw),
            ("HolE cnorm vocab", logits_hole_cnorm),
        ]:
            lo, hi = 0.01, 1000.0
            for _ in range(50):
                mid = (lo + hi) / 2
                ent = compute_entropy(logits_base * mid, dim=-1).mean().item()
                if ent > target_entropy:
                    lo = mid
                else:
                    hi = mid
            matching_tau = (lo + hi) / 2
            ent_check = compute_entropy(logits_base * matching_tau, dim=-1).mean().item()
            print(f"  {label}: tau={matching_tau:.2f} gives entropy={ent_check:.4f} "
                  f"(target={target_entropy:.4f})")

    print()
    print("=" * 70)
    print("=== Summary ===")
    print()
    print("Key question: Does HolE terminal scoring produce over-sharp softmax")
    print("distributions that would prevent learning?")
    print()
    print(f"  MLP baseline:       logits std={logits_mlp.std():.4f} (before tau={tau_term:.1f})")
    print(f"  HolE + raw vocab:   logits std={logits_hole_raw.std():.4f} (before tau)")
    print(f"  HolE + cnorm vocab: logits std={logits_hole_cnorm.std():.4f} (before tau)")
    print(f"  Rule HolE scores:   logits std={rule_scores.std():.4f} (before tau)")


if __name__ == "__main__":
    main()
