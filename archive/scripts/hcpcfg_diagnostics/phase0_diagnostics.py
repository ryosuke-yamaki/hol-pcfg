#!/usr/bin/env python3
"""Phase 0 Diagnostics: Embedding Collapse (0-A) + scale_c Convergence (0-B).

Usage:
    python scripts/phase0_diagnostics.py
"""

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


# ============================================================
# Phase 0-A: Embedding Collapse Diagnosis
# ============================================================

def analyse_embedding(ckpt_path: str, label: str) -> dict:
    """Analyse rule_state_emb from a checkpoint for angular collapse."""
    state = torch.load(ckpt_path, map_location="cpu")
    emb = state["rule_state_emb"]  # (NT+T, d)
    NT = 4096  # standard config

    nt_emb = emb[:NT]       # nonterminal embeddings
    t_emb = emb[NT:]        # terminal embeddings
    all_emb = emb

    results = {"label": label, "path": ckpt_path}

    for name, e in [("NT", nt_emb), ("T", t_emb), ("ALL", all_emb)]:
        norms = e.norm(dim=-1)
        e_normed = F.normalize(e, dim=-1)

        # Cosine similarity statistics (sample 2000 random pairs for efficiency)
        n = e_normed.shape[0]
        if n > 2000:
            idx = torch.randperm(n)[:2000]
            e_sample = e_normed[idx]
        else:
            e_sample = e_normed
        cos_mat = e_sample @ e_sample.t()

        # Mask diagonal
        mask = ~torch.eye(cos_mat.shape[0], dtype=torch.bool)
        cos_offdiag = cos_mat[mask]

        results[f"{name}/norm_mean"] = norms.mean().item()
        results[f"{name}/norm_std"] = norms.std().item()
        results[f"{name}/cos_mean"] = cos_offdiag.mean().item()
        results[f"{name}/cos_std"] = cos_offdiag.std().item()
        results[f"{name}/cos_abs_mean"] = cos_offdiag.abs().mean().item()
        results[f"{name}/cos_max"] = cos_offdiag.max().item()
        results[f"{name}/cos_min"] = cos_offdiag.min().item()
        results[f"{name}/cos_sq_mean"] = cos_offdiag.pow(2).mean().item()

        # Effective dimensionality via singular values
        if n <= 4096:
            _, S, _ = torch.svd(e_normed)
        else:
            # For large T, sample
            _, S, _ = torch.svd(e_sample)
        S_norm = S / S.sum()
        entropy = -(S_norm * S_norm.clamp(min=1e-12).log()).sum().item()
        max_entropy = math.log(min(e_sample.shape[0], e_sample.shape[1]))
        results[f"{name}/sv_entropy"] = entropy
        results[f"{name}/sv_max_entropy"] = max_entropy
        results[f"{name}/effective_dim_ratio"] = entropy / max_entropy if max_entropy > 0 else 0

        # 90% energy rank
        cumsum = (S ** 2).cumsum(0) / (S ** 2).sum()
        rank90 = (cumsum < 0.9).sum().item() + 1
        results[f"{name}/rank_90pct"] = rank90

    # v_left / v_right norms and spectral info
    for vname in ["v_left", "v_right"]:
        if vname in state:
            v = state[vname]
            results[f"{vname}/norm"] = v.norm().item()
            v_f = torch.fft.rfft(v)
            mag = v_f.abs()
            log_mag = torch.log(mag + 1e-8)
            results[f"{vname}/spectral_logvar"] = log_mag.var().item()
            results[f"{vname}/spectral_max"] = mag.max().item()
            results[f"{vname}/spectral_min"] = mag.min().item()

    # scale_c / tau
    if "scale_c" in state:
        results["scale_c"] = state["scale_c"].item()
    if "log_tau" in state:
        results["tau"] = math.exp(state["log_tau"].item())

    return results


def run_phase0a():
    """Phase 0-A: Compare embedding structure across model types."""
    print("=" * 70)
    print("Phase 0-A: Embedding Collapse Diagnosis")
    print("=" * 70)

    # Collect checkpoints by model type
    model_groups = {
        "SN-PCFG": sorted(Path("log/simple_npcfg_nt4096_t8192_curriculum0").glob("*/best.pt")),
        "HN us+tau": sorted(Path("log/hn_pcfg_unit_sphere_tau").glob("*/best.pt")),
        "HN us+c": sorted(Path("log/hn_pcfg_unit_sphere").glob("*/best.pt")),
        "HN cnorm+us+c": sorted(Path("log/hn_pcfg_unitsphere_cnorm_scale").glob("*/best.pt")),
        "HN cnorm+us+c rulesonly": sorted(Path("log/hn_pcfg_cnorm_us_c_rulesonly").glob("*/best.pt")),
    }

    all_results = {}
    for group_name, ckpts in model_groups.items():
        if not ckpts:
            continue
        group_results = []
        for ckpt in ckpts:
            res = analyse_embedding(str(ckpt), f"{group_name}")
            group_results.append(res)
        all_results[group_name] = group_results

    # Print comparison table
    metrics_to_compare = [
        ("NT/cos_mean", "NT cos(θ) mean"),
        ("NT/cos_std", "NT cos(θ) std"),
        ("NT/cos_sq_mean", "NT cos²(θ) mean"),
        ("NT/cos_abs_mean", "NT |cos(θ)| mean"),
        ("NT/rank_90pct", "NT 90% rank"),
        ("NT/effective_dim_ratio", "NT eff. dim ratio"),
    ]

    print("\n" + "-" * 90)
    header = f"{'Model':<28}" + "".join(f"{'mean':>10}{'std':>8}" for _, _ in metrics_to_compare[:3])
    print(f"{'':28}" + f"{'cos mean':>18}{'cos² mean':>18}{'90% rank':>18}")
    print("-" * 90)

    for group_name, results in all_results.items():
        n = len(results)
        line = f"{group_name:<28}"
        for metric_key, _ in [("NT/cos_mean", ""), ("NT/cos_sq_mean", ""), ("NT/rank_90pct", "")]:
            vals = [r[metric_key] for r in results]
            mean_v = sum(vals) / len(vals)
            if len(vals) > 1:
                std_v = (sum((x - mean_v) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
            else:
                std_v = 0.0
            line += f"{mean_v:>10.4f}{std_v:>8.4f}"
        line += f"  (n={n})"
        print(line)

    print("-" * 90)

    # Detailed per-model output
    print("\n\nDetailed Results per Model Type:")
    for group_name, results in all_results.items():
        print(f"\n--- {group_name} ({len(results)} runs) ---")
        for metric_key, metric_label in metrics_to_compare:
            vals = [r[metric_key] for r in results]
            mean_v = sum(vals) / len(vals)
            if len(vals) > 1:
                std_v = (sum((x - mean_v) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
            else:
                std_v = 0.0
            print(f"  {metric_label:<25}: {mean_v:.4f} ± {std_v:.4f}")

        # v norms and spectral
        for vname in ["v_left", "v_right"]:
            key = f"{vname}/norm"
            if key in results[0]:
                vals = [r[key] for r in results]
                mean_v = sum(vals) / len(vals)
                print(f"  {vname} norm              : {mean_v:.3f}")
            key_sv = f"{vname}/spectral_logvar"
            if key_sv in results[0]:
                vals = [r[key_sv] for r in results]
                mean_v = sum(vals) / len(vals)
                print(f"  {vname} spectral logvar   : {mean_v:.4f}")

        # scale_c / tau
        if "scale_c" in results[0]:
            vals = [r["scale_c"] for r in results]
            mean_v = sum(vals) / len(vals)
            std_v = (sum((x - mean_v) ** 2 for x in vals) / max(len(vals) - 1, 1)) ** 0.5
            print(f"  scale_c                 : {mean_v:.3f} ± {std_v:.3f}")
        if "tau" in results[0]:
            vals = [r["tau"] for r in results]
            mean_v = sum(vals) / len(vals)
            std_v = (sum((x - mean_v) ** 2 for x in vals) / max(len(vals) - 1, 1)) ** 0.5
            print(f"  tau                     : {mean_v:.2f} ± {std_v:.2f}")

    # Collapse diagnosis verdict
    print("\n" + "=" * 70)
    print("Phase 0-A VERDICT: Embedding Collapse Diagnosis")
    print("=" * 70)

    sn_results = all_results.get("SN-PCFG", [])
    hn_tau_results = all_results.get("HN us+tau", [])

    if sn_results and hn_tau_results:
        sn_cos_sq = sum(r["NT/cos_sq_mean"] for r in sn_results) / len(sn_results)
        hn_cos_sq = sum(r["NT/cos_sq_mean"] for r in hn_tau_results) / len(hn_tau_results)
        sn_rank90 = sum(r["NT/rank_90pct"] for r in sn_results) / len(sn_results)
        hn_rank90 = sum(r["NT/rank_90pct"] for r in hn_tau_results) / len(hn_tau_results)

        print(f"  SN-PCFG NT cos²(θ) mean : {sn_cos_sq:.6f}")
        print(f"  HN us+tau NT cos²(θ) mean: {hn_cos_sq:.6f}")
        print(f"  Ratio (HN/SN)           : {hn_cos_sq/sn_cos_sq:.2f}x")
        print()
        print(f"  SN-PCFG NT 90% rank     : {sn_rank90:.0f}")
        print(f"  HN us+tau NT 90% rank   : {hn_rank90:.0f}")
        print()

        if hn_cos_sq > sn_cos_sq * 1.5:
            print("  >> COLLAPSE DETECTED: HN-PCFG cos²(θ) is >1.5x SN-PCFG.")
            print("  >> RECOMMENDATION: Adopt Cosine Diversity Loss.")
        elif hn_cos_sq > sn_cos_sq * 1.2:
            print("  >> MILD COLLAPSE: HN-PCFG cos²(θ) is 1.2-1.5x SN-PCFG.")
            print("  >> RECOMMENDATION: Consider Cosine Diversity Loss (low priority).")
        else:
            print("  >> NO COLLAPSE: HN-PCFG cos²(θ) is within 1.2x of SN-PCFG.")
            print("  >> RECOMMENDATION: Do NOT adopt Cosine Diversity Loss.")
    else:
        print("  Insufficient data for comparison.")

    return all_results


# ============================================================
# Phase 0-B: scale_c Convergence Analysis
# ============================================================

def run_phase0b():
    """Phase 0-B: Analyse scale_c convergence from W&B logs and checkpoints."""
    print("\n\n" + "=" * 70)
    print("Phase 0-B: scale_c Convergence Analysis")
    print("=" * 70)

    # Collect scale_c from checkpoints (more reliable than W&B summary)
    scale_c_groups = defaultdict(list)

    # us+c (cnorm OFF)
    for ckpt in sorted(Path("log/hn_pcfg_unit_sphere").glob("*/best.pt")):
        state = torch.load(str(ckpt), map_location="cpu")
        if "scale_c" in state:
            scale_c_groups["us+c (cnorm OFF)"].append(state["scale_c"].item())

    # cnorm+us+c
    for ckpt in sorted(Path("log/hn_pcfg_unitsphere_cnorm_scale").glob("*/best.pt")):
        state = torch.load(str(ckpt), map_location="cpu")
        if "scale_c" in state:
            scale_c_groups["cnorm+us+c"].append(state["scale_c"].item())

    # cnorm+us+c rulesonly
    for ckpt in sorted(Path("log/hn_pcfg_cnorm_us_c_rulesonly").glob("*/best.pt")):
        state = torch.load(str(ckpt), map_location="cpu")
        if "scale_c" in state:
            scale_c_groups["cnorm+us+c rulesonly"].append(state["scale_c"].item())

    # cnorm+us+c cinit4
    for ckpt in sorted(Path("log/hn_pcfg_cnorm_us_c_cinit4").glob("*/best.pt")):
        state = torch.load(str(ckpt), map_location="cpu")
        if "scale_c" in state:
            scale_c_groups["cnorm+us+c c_init=4"].append(state["scale_c"].item())

    # cnorm+us+c lr_c
    for ckpt in sorted(Path("log/hn_pcfg_cnorm_us_c_lrc").glob("*/best.pt")):
        state = torch.load(str(ckpt), map_location="cpu")
        if "scale_c" in state:
            scale_c_groups["cnorm+us+c lr_c=0.0005"].append(state["scale_c"].item())

    # normless1 + c
    for ckpt in sorted(Path("log/hn_pcfg_normless1_scale").glob("*/best.pt")):
        state = torch.load(str(ckpt), map_location="cpu")
        if "scale_c" in state:
            scale_c_groups["normless1+c"].append(state["scale_c"].item())

    # tau models
    tau_groups = defaultdict(list)
    for ckpt in sorted(Path("log/hn_pcfg_unit_sphere_tau").glob("*/best.pt")):
        state = torch.load(str(ckpt), map_location="cpu")
        if "log_tau" in state:
            tau_groups["us+tau"].append(math.exp(state["log_tau"].item()))

    for ckpt in sorted(Path("log/hn_pcfg_unitsphere_cnorm_tau").glob("*/best.pt")):
        state = torch.load(str(ckpt), map_location="cpu")
        if "log_tau" in state:
            tau_groups["cnorm+us+tau"].append(math.exp(state["log_tau"].item()))

    for ckpt in sorted(Path("log/hn_pcfg_tau").glob("*/best.pt")):
        state = torch.load(str(ckpt), map_location="cpu")
        if "log_tau" in state:
            tau_groups["normless1+tau"].append(math.exp(state["log_tau"].item()))

    # Print results
    print("\nscale_c convergence values:")
    print("-" * 70)
    print(f"{'Config':<30}{'n':>4}{'mean':>10}{'std':>10}{'CV':>8}{'values'}")
    print("-" * 70)

    for config, vals in sorted(scale_c_groups.items()):
        n = len(vals)
        mean_v = sum(vals) / n
        std_v = (sum((x - mean_v) ** 2 for x in vals) / max(n - 1, 1)) ** 0.5
        cv = std_v / mean_v if mean_v != 0 else float("inf")
        vals_str = ", ".join(f"{v:.3f}" for v in vals)
        print(f"{config:<30}{n:>4}{mean_v:>10.3f}{std_v:>10.3f}{cv:>8.3f}  [{vals_str}]")

    print("\ntau convergence values:")
    print("-" * 70)
    for config, vals in sorted(tau_groups.items()):
        n = len(vals)
        mean_v = sum(vals) / n
        std_v = (sum((x - mean_v) ** 2 for x in vals) / max(n - 1, 1)) ** 0.5
        cv = std_v / mean_v if mean_v != 0 else float("inf")
        vals_str = ", ".join(f"{v:.2f}" for v in vals)
        print(f"{config:<30}{n:>4}{mean_v:>10.2f}{std_v:>10.2f}{cv:>8.3f}  [{vals_str}]")

    # Verdict
    print("\n" + "=" * 70)
    print("Phase 0-B VERDICT: scale_c Convergence")
    print("=" * 70)

    uc_vals = scale_c_groups.get("us+c (cnorm OFF)", [])
    cnorm_vals = scale_c_groups.get("cnorm+us+c", [])

    if uc_vals:
        mean_v = sum(uc_vals) / len(uc_vals)
        std_v = (sum((x - mean_v) ** 2 for x in uc_vals) / max(len(uc_vals) - 1, 1)) ** 0.5
        cv = std_v / mean_v
        print(f"\n  us+c: scale_c = {mean_v:.3f} ± {std_v:.3f} (CV={cv:.3f}, n={len(uc_vals)})")
        if cv < 0.3:
            print(f"  >> STABLE (CV < 0.3): scale_c converges reliably to ~{mean_v:.2f}")
        else:
            print(f"  >> UNSTABLE (CV >= 0.3): scale_c convergence is unreliable")

    if cnorm_vals:
        mean_v = sum(cnorm_vals) / len(cnorm_vals)
        std_v = (sum((x - mean_v) ** 2 for x in cnorm_vals) / max(len(cnorm_vals) - 1, 1)) ** 0.5
        cv = std_v / mean_v
        print(f"\n  cnorm+us+c: scale_c = {mean_v:.3f} ± {std_v:.3f} (CV={cv:.3f}, n={len(cnorm_vals)})")
        if cv < 0.3:
            print(f"  >> STABLE (CV < 0.3): scale_c converges reliably to ~{mean_v:.2f}")
        else:
            print(f"  >> UNSTABLE (CV >= 0.3): scale_c convergence is unreliable")

    print(f"\n  sqrt(d) = sqrt(512) = {math.sqrt(512):.2f}")
    print()
    if uc_vals:
        mean_uc = sum(uc_vals) / len(uc_vals)
        cv_uc = (sum((x - mean_uc) ** 2 for x in uc_vals) / max(len(uc_vals) - 1, 1)) ** 0.5 / mean_uc
        if cv_uc < 0.3:
            print("  >> RECOMMENDATION: scale_c is stable. Keep current unit_sphere + learnable scale.")
            print("  >> Do NOT switch to max_norm projection.")
        else:
            print("  >> RECOMMENDATION: scale_c is unstable. Consider max_norm=sqrt(d) as alternative.")


if __name__ == "__main__":
    results_0a = run_phase0a()
    run_phase0b()
    print("\n\nDiagnostics complete.")
