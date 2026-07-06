"""SP2 Verification: Relation Composition via Circular Convolution.

Verifies that HolE relations compose via circular convolution:
    v_ll = circonv(v_left, v_left) queries "left child's left child" in one step.
Under cnorm, |FFT(v_comp)[k]| = 1 is preserved.

CRITICAL: Composed score != Marginal probability.
  Composed:  score(C|A, v_ll) = <circonv(v_ll, e_A), e_C>   (single inner product)
  Marginal:  P(C|A) = sum_B P(B|A) * P(C|B)                 (matrix product)
The script compares both to sequential argmax.

Usage:
    python scripts/analyze_sp2_relation_composition.py [--checkpoint PATH]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from sp_utils import (
    NT, T, S_DIM, DEFAULT_CHECKPOINT,
    circular_convolution, load_checkpoint, setup_matplotlib, save_svg,
    compute_rule_probs, compute_raw_scores,
)

import numpy as np
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="results/sp2")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--n-parents", type=int, default=200,
                   help="Number of parents to sample for detailed analysis")
    p.add_argument("--max-hop", type=int, default=5)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Section 1: Compose relations and verify cnorm
# ──────────────────────────────────────────────────────────────────────
def compose_relations(
    v_left: torch.Tensor, v_right: torch.Tensor, s_dim: int
) -> dict[str, torch.Tensor]:
    """Compose all 4 two-hop relations via circular convolution."""
    compositions = {
        "ll": (v_left, v_left),
        "rr": (v_right, v_right),
        "lr": (v_left, v_right),
        "rl": (v_right, v_left),
    }
    result = {}
    for name, (v1, v2) in compositions.items():
        # circonv(v1, v2): apply v2 first, then v1
        # v_ll = circonv(v_left, v_left) → "left of left"
        result[name] = circular_convolution(v1, v2, s_dim)
    return result


def verify_cnorm_preservation(
    v_composed: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Verify |FFT(v_comp)[k]| = 1 for composed relations."""
    results = {}
    for name, v in v_composed.items():
        v_f = torch.fft.rfft(v)
        mag = v_f.abs()
        max_dev = (mag - 1.0).abs().max().item()
        mean_mag = mag.mean().item()
        results[name] = {"max_deviation": max_dev, "mean_mag": mean_mag}
    return results


# ──────────────────────────────────────────────────────────────────────
# Section 2: 3-way comparison
# ──────────────────────────────────────────────────────────────────────
def compute_composed_topk(
    v_comp: torch.Tensor,
    parent_idx: int,
    nt_emb: torch.Tensor,
    all_emb: torch.Tensor,
    tau: float,
    s_dim: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Composed 1-hop retrieval: score(C|A, v_comp) with softmax."""
    e_A = nt_emb[parent_idx]  # (s_dim,)
    template = circular_convolution(v_comp, e_A, s_dim)  # (s_dim,)
    scores = all_emb @ template  # (NT+T,)
    probs = (scores * tau).softmax(dim=0)
    return probs.topk(k)


def compute_sequential_topk(
    P_first: torch.Tensor,
    P_second: torch.Tensor,
    parent_idx: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Sequential argmax: top-1 NT child via P_first, then top-k via P_second.

    Returns (topk_values, topk_indices, intermediate_idx).
    """
    # Step 1: best NT child of parent via first relation
    first_probs = P_first[:NT, parent_idx]  # (NT,) -- restrict to NT children
    intermediate_idx = first_probs.argmax().item()

    # Step 2: top-k children of intermediate via second relation
    second_probs = P_second[:, intermediate_idx]  # (NT+T,)
    vals, idxs = second_probs.topk(k)
    return vals, idxs, intermediate_idx


def compute_marginal_topk(
    P_2hop: torch.Tensor,
    parent_idx: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k from 2-hop marginal distribution."""
    col = P_2hop[:, parent_idx]  # (NT+T,)
    return col.topk(k)


def jaccard_at_k(set_a: torch.Tensor, set_b: torch.Tensor) -> float:
    """Jaccard similarity between two index sets."""
    a = set(set_a.cpu().tolist())
    b = set(set_b.cpu().tolist())
    if len(a) == 0 and len(b) == 0:
        return 1.0
    return len(a & b) / len(a | b)


def rank_correlation(
    indices_a: torch.Tensor,
    values_a: torch.Tensor,
    indices_b: torch.Tensor,
    values_b: torch.Tensor,
) -> float:
    """Spearman rank correlation over the union of two top-k sets."""
    a_set = set(indices_a.cpu().tolist())
    b_set = set(indices_b.cpu().tolist())
    union = sorted(a_set | b_set)
    if len(union) < 3:
        return float("nan")

    # Build rank maps (lower rank = higher prob)
    a_rank = {idx: rank for rank, idx in enumerate(indices_a.cpu().tolist())}
    b_rank = {idx: rank for rank, idx in enumerate(indices_b.cpu().tolist())}

    # Items not in a list get rank = k (worst)
    k = len(indices_a)
    ranks_a = [a_rank.get(idx, k) for idx in union]
    ranks_b = [b_rank.get(idx, k) for idx in union]

    corr, _ = spearmanr(ranks_a, ranks_b)
    return corr


def run_3way_comparison(
    v_composed: dict[str, torch.Tensor],
    data: dict,
    args: argparse.Namespace,
) -> dict:
    """Run 3-way comparison for all 4 composition types."""
    device = data["v_left"].device
    nt_emb = data["nt_emb"]
    all_emb = data["all_emb"]
    tau = data["tau"]

    # Map composition type to (first_relation, second_relation)
    comp_relations = {
        "ll": ("left", "left"),
        "rr": ("right", "right"),
        "lr": ("left", "right"),
        "rl": ("right", "left"),
    }

    # Precompute probability matrices for left and right
    print("\n  Computing P_left...")
    P_left = compute_rule_probs(data["v_left"], nt_emb, all_emb, tau, S_DIM)
    print(f"    P_left shape: {P_left.shape}, sum check col 0: {P_left[:, 0].sum():.6f}")
    print("  Computing P_right...")
    P_right = compute_rule_probs(data["v_right"], nt_emb, all_emb, tau, S_DIM)
    print(f"    P_right shape: {P_right.shape}")

    P_map = {"left": P_left, "right": P_right}

    results = {}

    for comp_name, (first_rel, second_rel) in comp_relations.items():
        print(f"\n  === Composition: {comp_name} ({first_rel} then {second_rel}) ===")
        v_comp = v_composed[comp_name]
        P_first = P_map[first_rel]
        P_second = P_map[second_rel]

        # Compute 2-hop marginal: P_2hop = P_second @ P_first[:NT, :]
        # P_2hop[c, a] = sum_b P_second[c, b] * P_first[b, a]  for b in NTs
        print(f"    Computing 2-hop marginal ({comp_name})...")
        P_first_nt = P_first[:NT, :]  # (NT, NT)
        # Use float32 for matmul accuracy
        P_2hop = P_second.float() @ P_first_nt.float()  # (NT+T, NT)
        P_2hop = P_2hop.to(P_left.dtype)
        print(f"    P_2hop shape: {P_2hop.shape}, sum col 0: {P_2hop[:, 0].sum():.6f}")
        print(f"    (sum < 1 because some 1st-hop probability goes to terminals)")

        # Find parents whose top-1 child (via first relation) is NT
        top1_child = P_first.argmax(dim=0)  # (NT,)
        nt_parents_mask = top1_child < NT
        nt_parent_indices = nt_parents_mask.nonzero(as_tuple=True)[0]
        n_nt_parents = nt_parent_indices.shape[0]
        print(f"    Parents with top-1 {first_rel} child being NT: {n_nt_parents}/{NT} "
              f"({100*n_nt_parents/NT:.1f}%)")

        # Sample parents
        n_sample = min(args.n_parents, n_nt_parents)
        if n_sample == 0:
            print(f"    WARNING: No parents with NT top-1 child for {comp_name}")
            results[comp_name] = {"skipped": True}
            continue

        perm = torch.randperm(n_nt_parents, device=device)[:n_sample]
        sampled_parents = nt_parent_indices[perm]

        jaccards = {"comp_vs_seq": [], "comp_vs_marg": [], "seq_vs_marg": []}
        spearmans = {"comp_vs_seq": [], "comp_vs_marg": [], "seq_vs_marg": []}
        examples = []

        for i, pidx in enumerate(sampled_parents):
            pidx = pidx.item()

            # (a) Composed
            comp_vals, comp_idxs = compute_composed_topk(
                v_comp, pidx, nt_emb, all_emb, tau, S_DIM, args.top_k)

            # (b) Sequential argmax
            seq_vals, seq_idxs, inter_idx = compute_sequential_topk(
                P_first, P_second, pidx, args.top_k)

            # (c) Full marginal
            marg_vals, marg_idxs = compute_marginal_topk(P_2hop, pidx, args.top_k)

            # Jaccard@k
            jaccards["comp_vs_seq"].append(jaccard_at_k(comp_idxs, seq_idxs))
            jaccards["comp_vs_marg"].append(jaccard_at_k(comp_idxs, marg_idxs))
            jaccards["seq_vs_marg"].append(jaccard_at_k(seq_idxs, marg_idxs))

            # Spearman
            spearmans["comp_vs_seq"].append(
                rank_correlation(comp_idxs, comp_vals, seq_idxs, seq_vals))
            spearmans["comp_vs_marg"].append(
                rank_correlation(comp_idxs, comp_vals, marg_idxs, marg_vals))
            spearmans["seq_vs_marg"].append(
                rank_correlation(seq_idxs, seq_vals, marg_idxs, marg_vals))

            # Save some examples
            if len(examples) < 5:
                examples.append({
                    "parent": pidx,
                    "intermediate": inter_idx,
                    "composed_top5": comp_idxs[:5].cpu().tolist(),
                    "sequential_top5": seq_idxs[:5].cpu().tolist(),
                    "marginal_top5": marg_idxs[:5].cpu().tolist(),
                })

        # Aggregate
        def agg(vals):
            valid = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if not valid:
                return {"mean": float("nan"), "std": float("nan"), "n": 0}
            return {"mean": float(np.mean(valid)), "std": float(np.std(valid)),
                    "n": len(valid)}

        comp_result = {
            "n_nt_parents": n_nt_parents,
            "n_sampled": n_sample,
            "jaccard": {k: agg(v) for k, v in jaccards.items()},
            "spearman": {k: agg(v) for k, v in spearmans.items()},
            "examples": examples,
        }

        print(f"    Jaccard@{args.top_k}:")
        for pair, stats in comp_result["jaccard"].items():
            print(f"      {pair}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"    Spearman:")
        for pair, stats in comp_result["spearman"].items():
            print(f"      {pair}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        results[comp_name] = comp_result

        # Free marginal matrix
        del P_2hop

    # Free prob matrices
    del P_left, P_right
    torch.cuda.empty_cache()

    return results


# ──────────────────────────────────────────────────────────────────────
# Section 3: Effective sample size
# ──────────────────────────────────────────────────────────────────────
def analyze_effective_sample_size(data: dict) -> dict:
    """Count parents whose top-1 left/right child is NT vs T."""
    nt_emb = data["nt_emb"]
    all_emb = data["all_emb"]
    tau = data["tau"]

    results = {}
    for rel_name, v in [("left", data["v_left"]), ("right", data["v_right"])]:
        print(f"\n  Relation: {rel_name}")
        P = compute_rule_probs(v, nt_emb, all_emb, tau, S_DIM)
        top1 = P.argmax(dim=0)  # (NT,)
        n_nt = (top1 < NT).sum().item()
        n_t = (top1 >= NT).sum().item()
        print(f"    Top-1 child is NT: {n_nt}/{NT} ({100*n_nt/NT:.1f}%)")
        print(f"    Top-1 child is T:  {n_t}/{NT} ({100*n_t/NT:.1f}%)")

        # Also: first NT in ranked list (for all parents)
        # Find the rank of the best NT child
        nt_probs = P[:NT, :]  # (NT, NT) -- prob of NT children
        best_nt_prob = nt_probs.max(dim=0).values  # (NT,)
        overall_best_prob = P.max(dim=0).values
        ratio = (best_nt_prob / overall_best_prob.clamp(min=1e-20))
        print(f"    Best NT prob / best overall prob: "
              f"mean={ratio.mean():.4f}, median={ratio.median():.4f}")

        results[rel_name] = {
            "top1_nt": n_nt,
            "top1_t": n_t,
            "nt_frac": n_nt / NT,
            "best_nt_vs_overall_mean": ratio.mean().item(),
            "best_nt_vs_overall_median": ratio.median().item(),
        }
        del P
    torch.cuda.empty_cache()
    return results


# ──────────────────────────────────────────────────────────────────────
# Section 4: n-hop convergence
# ──────────────────────────────────────────────────────────────────────
def analyze_nhop_convergence(data: dict, args: argparse.Namespace) -> dict:
    """Track n-hop composition for n=1..max_hop."""
    v_left = data["v_left"]
    nt_emb = data["nt_emb"]
    all_emb = data["all_emb"]
    tau = data["tau"]
    device = v_left.device

    n_rep = 100  # representative parents
    torch.manual_seed(42)
    parent_indices = torch.randperm(NT, device=device)[:n_rep]

    # Track top-1 at each hop
    prev_top1 = None
    hop_results = []

    v_n = v_left.clone()  # n=1

    for n in range(1, args.max_hop + 1):
        if n > 1:
            v_n = circular_convolution(v_left, v_n, S_DIM)

        # cnorm check
        v_f = torch.fft.rfft(v_n)
        mag = v_f.abs()
        max_dev = (mag - 1.0).abs().max().item()
        mean_dev = (mag - 1.0).abs().mean().item()

        # Top-1 for each representative parent
        # Batch computation: template = circonv(v_n, e_A) for all parents
        parents_emb = nt_emb[parent_indices]  # (n_rep, s_dim)
        v_f_batch = torch.fft.rfft(v_n.unsqueeze(0), dim=-1)  # (1, F)
        p_f = torch.fft.rfft(parents_emb, dim=-1)  # (n_rep, F)
        templates = torch.fft.irfft(v_f_batch * p_f, n=S_DIM, dim=-1)  # (n_rep, s_dim)
        scores = templates @ all_emb.t()  # (n_rep, NT+T)
        top1 = scores.argmax(dim=1)  # (n_rep,)

        # Fraction changed from previous hop
        if prev_top1 is not None:
            frac_changed = (top1 != prev_top1).float().mean().item()
        else:
            frac_changed = 1.0

        print(f"    n={n}: cnorm max_dev={max_dev:.2e}, mean_dev={mean_dev:.2e}, "
              f"frac_changed={frac_changed:.4f}")

        hop_results.append({
            "n": n,
            "cnorm_max_dev": max_dev,
            "cnorm_mean_dev": mean_dev,
            "frac_changed": frac_changed,
            "top1_examples": top1[:10].cpu().tolist(),
        })
        prev_top1 = top1.clone()

    return {"hops": hop_results}


# ──────────────────────────────────────────────────────────────────────
# Section 5: Cross-composition
# ──────────────────────────────────────────────────────────────────────
def analyze_cross_composition(
    v_composed: dict[str, torch.Tensor],
    data: dict,
    args: argparse.Namespace,
) -> dict:
    """Compare cross-composition (lr, rl) with sequential retrieval."""
    device = data["v_left"].device
    nt_emb = data["nt_emb"]
    all_emb = data["all_emb"]
    tau = data["tau"]

    # We only need P_left and P_right for sequential
    P_left = compute_rule_probs(data["v_left"], nt_emb, all_emb, tau, S_DIM)
    P_right = compute_rule_probs(data["v_right"], nt_emb, all_emb, tau, S_DIM)

    cross_configs = {
        "lr": {"first": P_left, "second": P_right,
               "desc": "left child's right child"},
        "rl": {"first": P_right, "second": P_left,
               "desc": "right child's left child"},
    }

    results = {}
    for comp_name, cfg in cross_configs.items():
        print(f"\n  Cross-composition: {comp_name} ({cfg['desc']})")
        v_comp = v_composed[comp_name]
        P_first = cfg["first"]
        P_second = cfg["second"]

        # Parents whose top-1 child via first relation is NT
        top1 = P_first.argmax(dim=0)
        nt_mask = top1 < NT
        nt_parents = nt_mask.nonzero(as_tuple=True)[0]
        n_sample = min(args.n_parents, nt_parents.shape[0])

        perm = torch.randperm(nt_parents.shape[0], device=device)[:n_sample]
        sampled = nt_parents[perm]

        jaccards = []
        examples = []
        for pidx in sampled:
            pidx = pidx.item()
            # Composed
            comp_vals, comp_idxs = compute_composed_topk(
                v_comp, pidx, nt_emb, all_emb, tau, S_DIM, args.top_k)
            # Sequential
            seq_vals, seq_idxs, inter = compute_sequential_topk(
                P_first, P_second, pidx, args.top_k)

            jaccards.append(jaccard_at_k(comp_idxs, seq_idxs))

            if len(examples) < 3:
                examples.append({
                    "parent": pidx, "intermediate": inter,
                    "composed_top5": comp_idxs[:5].cpu().tolist(),
                    "sequential_top5": seq_idxs[:5].cpu().tolist(),
                })

        mean_j = float(np.mean(jaccards))
        std_j = float(np.std(jaccards))
        print(f"    Jaccard@{args.top_k}: {mean_j:.4f} ± {std_j:.4f} (n={n_sample})")

        results[comp_name] = {
            "desc": cfg["desc"],
            "n_sampled": n_sample,
            "jaccard_mean": mean_j,
            "jaccard_std": std_j,
            "examples": examples,
        }

    del P_left, P_right
    torch.cuda.empty_cache()
    return results


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────
def plot_3way_comparison(comparison_results: dict, output_dir: Path, top_k: int) -> None:
    """Grouped bar chart: Jaccard@k for each composition type."""
    import matplotlib.pyplot as plt
    setup_matplotlib()

    comp_types = [k for k in ["ll", "rr", "lr", "rl"]
                  if k in comparison_results and not comparison_results[k].get("skipped")]
    if not comp_types:
        print("  WARNING: No composition types to plot.")
        return

    pairs = ["comp_vs_seq", "comp_vs_marg", "seq_vs_marg"]
    pair_labels = ["Composed vs Sequential", "Composed vs Marginal", "Sequential vs Marginal"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    x = np.arange(len(comp_types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (pair, label, color) in enumerate(zip(pairs, pair_labels, colors)):
        means = [comparison_results[ct]["jaccard"][pair]["mean"] for ct in comp_types]
        stds = [comparison_results[ct]["jaccard"][pair]["std"] for ct in comp_types]
        ax.bar(x + i * width, means, width, yerr=stds, label=label,
               color=color, alpha=0.85, capsize=3)

    ax.set_xlabel("Composition type")
    ax.set_ylabel(f"Jaccard@{top_k}")
    ax.set_title(f"SP2: 3-Way Comparison of 2-Hop Retrieval (Jaccard@{top_k})")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"v_{ct}" for ct in comp_types])
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_svg(fig, output_dir, "sp2_3way_comparison")
    plt.close(fig)


def plot_nhop_convergence(nhop_results: dict, output_dir: Path) -> None:
    """Line plots: fraction changed and cnorm deviation vs n."""
    import matplotlib.pyplot as plt
    setup_matplotlib()

    hops = nhop_results["hops"]
    ns = [h["n"] for h in hops]
    frac_changed = [h["frac_changed"] for h in hops]
    cnorm_dev = [h["cnorm_max_dev"] for h in hops]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    ax1.plot(ns, frac_changed, "o-", color="#2196F3", linewidth=2, markersize=6)
    ax1.set_xlabel("Number of hops (n)")
    ax1.set_ylabel("Fraction of parents with changed top-1")
    ax1.set_title("Top-1 Stability Across Hops")
    ax1.set_xticks(ns)
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax2.bar(ns, cnorm_dev, color="#FF5722", alpha=0.85)
    ax2.set_xlabel("Number of hops (n)")
    ax2.set_ylabel("Max |FFT magnitude| - 1")
    ax2.set_title("cnorm Deviation After n Compositions")
    ax2.set_xticks(ns)
    ax2.ticklabel_format(axis="y", style="scientific", scilimits=(-2, 2))

    fig.suptitle("SP2: n-Hop Convergence Analysis", fontsize=11, y=1.02)
    fig.tight_layout()
    save_svg(fig, output_dir, "sp2_nhop_convergence")
    plt.close(fig)


def plot_cnorm_preservation(nhop_results: dict, output_dir: Path) -> None:
    """Bar chart: max FFT magnitude deviation for n=1..max_hop."""
    import matplotlib.pyplot as plt
    setup_matplotlib()

    hops = nhop_results["hops"]
    ns = [h["n"] for h in hops]
    devs = [h["cnorm_max_dev"] for h in hops]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(ns, devs, color="#9C27B0", alpha=0.85)
    ax.set_xlabel("Number of compositions (n)")
    ax.set_ylabel("Max |FFT(v^n)[k]| - 1")
    ax.set_title("SP2: cnorm Preservation Under Composition")
    ax.set_xticks(ns)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(-2, 2))

    # Annotate values
    for bar, dev in zip(bars, devs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{dev:.1e}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    save_svg(fig, output_dir, "sp2_cnorm_preservation")
    plt.close(fig)


def write_examples(
    comparison_results: dict, cross_results: dict, output_dir: Path
) -> None:
    """Write concrete 2-hop chain examples to text file."""
    path = output_dir / "sp2_examples.txt"
    with open(path, "w") as f:
        f.write("SP2: Relation Composition — Concrete 2-Hop Chain Examples\n")
        f.write("=" * 70 + "\n\n")
        f.write("NOTE: Composed score != Marginal probability.\n")
        f.write("  Composed:  score(C|A, v_ll) = <circonv(v_ll, e_A), e_C>\n")
        f.write("  Marginal:  P(C|A) = sum_B P(B|A) * P(C|B)\n")
        f.write("These are fundamentally different operations.\n\n")

        for comp_name in ["ll", "rr", "lr", "rl"]:
            if comp_name not in comparison_results:
                continue
            cr = comparison_results[comp_name]
            if cr.get("skipped"):
                continue
            f.write(f"\n{'─'*70}\n")
            desc_map = {"ll": "left-of-left", "rr": "right-of-right",
                        "lr": "right-of-left", "rl": "left-of-right"}
            f.write(f"Composition: v_{comp_name} = circonv(v_{comp_name[0]}, v_{comp_name[1]}) "
                    f"— {desc_map[comp_name]}\n")
            f.write(f"  Jaccard@10: comp_vs_seq={cr['jaccard']['comp_vs_seq']['mean']:.4f}, "
                    f"comp_vs_marg={cr['jaccard']['comp_vs_marg']['mean']:.4f}, "
                    f"seq_vs_marg={cr['jaccard']['seq_vs_marg']['mean']:.4f}\n\n")
            for ex in cr.get("examples", []):
                f.write(f"  Parent NT_{ex['parent']} → Intermediate NT_{ex['intermediate']}\n")
                f.write(f"    Composed top-5:   {ex['composed_top5']}\n")
                f.write(f"    Sequential top-5: {ex['sequential_top5']}\n")
                f.write(f"    Marginal top-5:   {ex['marginal_top5']}\n\n")

        if cross_results:
            f.write(f"\n{'='*70}\n")
            f.write("Cross-Composition Examples\n")
            for comp_name, cr in cross_results.items():
                f.write(f"\n  v_{comp_name} — {cr['desc']}\n")
                f.write(f"  Jaccard@10: {cr['jaccard_mean']:.4f} ± {cr['jaccard_std']:.4f}\n")
                for ex in cr.get("examples", []):
                    f.write(f"    Parent NT_{ex['parent']} → Intermediate NT_{ex['intermediate']}\n")
                    f.write(f"      Composed top-5:   {ex['composed_top5']}\n")
                    f.write(f"      Sequential top-5: {ex['sequential_top5']}\n\n")

    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SP2: Relation Composition via Circular Convolution")
    print("=" * 70)

    # Load model
    print(f"\n  Loading checkpoint: {args.checkpoint}")
    data = load_checkpoint(args.checkpoint, args.device)
    print(f"  tau = {data['tau']:.4f}")
    print(f"  v_left shape: {data['v_left'].shape}")

    all_results = {}

    # ── Section 1: Compose & verify cnorm ──
    print("\n" + "─" * 70)
    print("  Section 1: Compose Relations & Verify cnorm")
    print("─" * 70)
    v_composed = compose_relations(data["v_left"], data["v_right"], S_DIM)
    cnorm_results = verify_cnorm_preservation(v_composed)
    for name, res in cnorm_results.items():
        print(f"    v_{name}: max_dev={res['max_deviation']:.2e}, "
              f"mean_mag={res['mean_mag']:.8f}")
    all_results["cnorm_verification"] = cnorm_results

    # ── Section 2: 3-way comparison ──
    print("\n" + "─" * 70)
    print("  Section 2: 3-Way Comparison (Composed vs Sequential vs Marginal)")
    print("─" * 70)
    comparison_results = run_3way_comparison(v_composed, data, args)
    all_results["3way_comparison"] = comparison_results

    # ── Section 3: Effective sample size ──
    print("\n" + "─" * 70)
    print("  Section 3: Effective Sample Size")
    print("─" * 70)
    ess_results = analyze_effective_sample_size(data)
    all_results["effective_sample_size"] = ess_results

    # ── Section 4: n-hop convergence ──
    print("\n" + "─" * 70)
    print("  Section 4: n-Hop Convergence (n=1..{})".format(args.max_hop))
    print("─" * 70)
    nhop_results = analyze_nhop_convergence(data, args)
    all_results["nhop_convergence"] = nhop_results

    # ── Section 5: Cross-composition ──
    print("\n" + "─" * 70)
    print("  Section 5: Cross-Composition (lr, rl)")
    print("─" * 70)
    cross_results = analyze_cross_composition(v_composed, data, args)
    all_results["cross_composition"] = cross_results

    # ── Figures ──
    print("\n" + "─" * 70)
    print("  Generating Figures")
    print("─" * 70)
    plot_3way_comparison(comparison_results, output_dir, args.top_k)
    plot_nhop_convergence(nhop_results, output_dir)
    plot_cnorm_preservation(nhop_results, output_dir)
    write_examples(comparison_results, cross_results, output_dir)

    # ── Save JSON ──
    json_path = output_dir / "sp2_results.json"

    def to_serializable(obj):
        if isinstance(obj, (torch.Tensor,)):
            return obj.cpu().tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    import copy
    serializable = json.loads(json.dumps(all_results, default=to_serializable))
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2, default=to_serializable)
    print(f"\n  Saved: {json_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SP2 Summary")
    print("=" * 70)
    print("\n  1. cnorm Preservation:")
    for name, res in cnorm_results.items():
        print(f"     v_{name}: max deviation = {res['max_deviation']:.2e}")
    print("\n  2. 3-Way Comparison (Jaccard@{} means):".format(args.top_k))
    for ct in ["ll", "rr", "lr", "rl"]:
        if ct in comparison_results and not comparison_results[ct].get("skipped"):
            cr = comparison_results[ct]
            j = cr["jaccard"]
            print(f"     v_{ct}: comp_vs_seq={j['comp_vs_seq']['mean']:.4f}  "
                  f"comp_vs_marg={j['comp_vs_marg']['mean']:.4f}  "
                  f"seq_vs_marg={j['seq_vs_marg']['mean']:.4f}")
    print("\n  3. Effective Sample Size:")
    for rel, res in ess_results.items():
        print(f"     {rel}: {res['top1_nt']}/{NT} parents have NT top-1 child "
              f"({100*res['nt_frac']:.1f}%)")
    print("\n  4. n-Hop Convergence:")
    for h in nhop_results["hops"]:
        print(f"     n={h['n']}: cnorm_dev={h['cnorm_max_dev']:.2e}, "
              f"frac_changed={h['frac_changed']:.4f}")
    print("\n  NOTE: Composed score and marginal probability are fundamentally")
    print("  different operations. The PCFG inside algorithm does not use")
    print("  multi-hop composition — it is a single-hop scoring mechanism.")
    print("  Composition is an emergent algebraic property of the HolE framework.")


if __name__ == "__main__":
    main()
