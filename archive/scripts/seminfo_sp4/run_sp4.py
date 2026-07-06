"""SP4 Relation Extraction verification for HN-PCFG checkpoints.

Loads a Lightning checkpoint, extracts relation vectors via circular
correlation for top-1 (parent, child) pairs in three conditions
(rule_left, rule_right, terminal), and computes:

  Level-0 : phase-only manifold / Parseval sanity
  Level-1 : mean cos(r_ext, v) vs shuffled baseline
  Level-2 : Cohen's d between main and shuffled distributions
  Level-3 : circular-stats concentration (R, theta_deg vs v)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def circcorr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Circular correlation r = IFFT(conj(FFT(x)) * FFT(y)) along last dim."""
    s_dim = x.shape[-1]
    xf = torch.fft.rfft(x, dim=-1)
    yf = torch.fft.rfft(y, dim=-1)
    return torch.fft.irfft(xf.conj() * yf, n=s_dim, dim=-1)


def circconv(v: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Circular convolution: template[i] = IFFT(FFT(v) * FFT(source[i]))."""
    s_dim = source.shape[-1]
    vf = torch.fft.rfft(v, dim=-1)
    sf = torch.fft.rfft(source, dim=-1)
    return torch.fft.irfft(vf.unsqueeze(0) * sf, n=s_dim, dim=-1)


def cnorm_max_dev(x: torch.Tensor) -> float:
    """max_k | |FFT(x)[k]| - 1 |  (phase-only manifold deviation)."""
    return (torch.fft.rfft(x, dim=-1).abs() - 1.0).abs().max().item()


def cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1, eps=eps)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _concentration(unit_vectors: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Mean resultant length R and mean direction mu of a set of unit vectors."""
    mu = unit_vectors.mean(dim=0)
    return mu.norm().item(), mu


def compute_condition(
    parents: torch.Tensor,         # (P, d)
    candidates: torch.Tensor,      # (C, d)
    v: torch.Tensor,               # (d,)
    num_shuffles: int,
    rng: torch.Generator,
) -> dict:
    """SP4 metrics for one condition: top-1 child per parent + baselines."""
    templates = circconv(v, parents)                # (P, d)
    scores = candidates @ templates.t()             # (C, P)
    top_idx = scores.argmax(dim=-2)                 # (P,)
    top_children = candidates[top_idx]              # (P, d)

    # Main
    r_main = circcorr(parents, top_children)        # (P, d)
    cos_main = cos_sim(r_main, v.unsqueeze(0))      # (P,)
    mean_cos = cos_main.mean().item()
    std_cos = cos_main.std().item()

    u_main = F.normalize(r_main, dim=-1)
    R_main, mu_main = _concentration(u_main)
    v_unit = v / v.norm().clamp(min=1e-12)
    mu_unit = mu_main / mu_main.norm().clamp(min=1e-12)
    theta_deg = math.degrees(
        math.acos(torch.clamp(torch.dot(mu_unit, v_unit), -1.0, 1.0).item())
    )

    v_norm = v.norm().item()
    r_norms = r_main.norm(dim=-1)
    norm_preserve_max_dev = (
        (r_norms - v_norm).abs() / max(v_norm, 1e-12)
    ).max().item()

    # Shuffled baselines
    shuf_mean_cos, shuf_std_cos, shuf_R = [], [], []
    for _ in range(num_shuffles):
        perm = torch.randperm(top_children.shape[0], generator=rng)
        shuf_children = top_children[perm]
        r_shuf = circcorr(parents, shuf_children)
        cos_shuf = cos_sim(r_shuf, v.unsqueeze(0))
        shuf_mean_cos.append(cos_shuf.mean().item())
        shuf_std_cos.append(cos_shuf.std().item())
        R_s, _ = _concentration(F.normalize(r_shuf, dim=-1))
        shuf_R.append(R_s)

    mean_cos_shuf = float(torch.tensor(shuf_mean_cos).mean())
    std_cos_shuf = float(torch.tensor(shuf_std_cos).mean())
    R_shuf = float(torch.tensor(shuf_R).mean())

    pooled = math.sqrt((std_cos ** 2 + std_cos_shuf ** 2) / 2.0) + 1e-12
    cohen_d = (mean_cos - mean_cos_shuf) / pooled

    return {
        "mean_cos": mean_cos,
        "std_cos": std_cos,
        "mean_cos_shuf": mean_cos_shuf,
        "std_cos_shuf": std_cos_shuf,
        "cohen_d": cohen_d,
        "R": R_main,
        "R_shuf": R_shuf,
        "theta_deg": theta_deg,
        "norm_preserve_max_dev": norm_preserve_max_dev,
    }


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_model_state(ckpt_path: str) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    return {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--wandb_id", default=None)
    ap.add_argument("--NT", type=int, default=1024)
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--num_shuffles", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    sd = load_model_state(args.ckpt)
    device = torch.device(args.device)

    rule_state_emb = sd["rule_state_emb"].to(device).float()
    v_left = sd["v_left"].to(device).float()
    v_right = sd["v_right"].to(device).float()
    v_term = sd["v_term"].to(device).float()
    vocab_emb = sd["vocab_emb"].to(device).float()        # (d, V)

    NT = args.NT
    T_count = rule_state_emb.shape[0] - NT
    nt_emb = rule_state_emb[:NT]
    t_emb = rule_state_emb[NT:]
    vocab = vocab_emb.t()                                 # (V, d)

    sanity = {
        "cnorm_rule_state_emb_max_dev": cnorm_max_dev(rule_state_emb),
        "cnorm_v_left_max_dev": cnorm_max_dev(v_left),
        "cnorm_v_right_max_dev": cnorm_max_dev(v_right),
        "cnorm_v_term_max_dev": cnorm_max_dev(v_term),
        "cnorm_vocab_emb_max_dev": cnorm_max_dev(vocab),
    }

    rng = torch.Generator()
    rng.manual_seed(args.seed)

    conditions = {
        "rule_left": compute_condition(
            nt_emb, rule_state_emb, v_left, args.num_shuffles, rng),
        "rule_right": compute_condition(
            nt_emb, rule_state_emb, v_right, args.num_shuffles, rng),
        "terminal": compute_condition(
            t_emb, vocab, v_term, args.num_shuffles, rng),
    }

    result = {
        "ckpt": args.ckpt,
        "lang": args.lang,
        "wandb_id": args.wandb_id,
        "NT": NT,
        "T": T_count,
        "V": vocab.shape[0],
        "s_dim": rule_state_emb.shape[-1],
        "num_shuffles": args.num_shuffles,
        "seed": args.seed,
        "sanity": sanity,
        "conditions": conditions,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")

    for name, c in conditions.items():
        print(
            f"  [{name:10s}]  mean_cos={c['mean_cos']:.4f}  "
            f"shuf={c['mean_cos_shuf']:.4f}  d={c['cohen_d']:.2f}  "
            f"R={c['R']:.3f}  R_shuf={c['R_shuf']:.3f}  "
            f"theta={c['theta_deg']:.1f}deg"
        )


if __name__ == "__main__":
    main()
