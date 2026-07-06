"""Numerical verification of HN_PCFG scoring functions.

Checks that:
- 'hole' branch computes <v, a ⋆ b> via the corr-conv identity (current behavior).
- 'hadamard' branch computes <v, a ⊙ b>.
- 'conv' branch computes <v, a * b> via the <v, a*b> = <b, a ⋆ v> identity.

Each computed score is compared against a direct loop reference (definition).
"""
import math

import torch

from parser.model.HN_PCFG import HN_PCFG


def _direct_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a.numel()
    out = torch.zeros(d, dtype=a.dtype)
    for k in range(d):
        s = 0.0
        for i in range(d):
            s = s + a[i] * b[(k + i) % d]
        out[k] = s
    return out


def _direct_conv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a.numel()
    out = torch.zeros(d, dtype=a.dtype)
    for n in range(d):
        s = 0.0
        for m in range(d):
            s = s + a[m] * b[(n - m) % d]
        out[n] = s
    return out


def _ref_hole(v, a, b):
    # <v, a ⋆ b>
    return torch.dot(v, _direct_corr(a, b))


def _ref_hadamard(v, a, b):
    return torch.dot(v, a * b)


def _ref_conv(v, a, b):
    # <v, a * b>
    return torch.dot(v, _direct_conv(a, b))


def _scores(model, v, source, target):
    log_tau = torch.tensor(0.0)
    return model._hol_scores(v, source, target, log_tau)


def _build_model(scoring_fn, s_dim):
    class _DummyArgs:
        pass

    class _DummyDataset:
        device = torch.device('cpu')
        V = list(range(7))

    args = _DummyArgs()
    args.NT = 3
    args.T = 2
    args.s_dim = s_dim
    args.scoring_fn = scoring_fn
    args.complex_normalization = True
    args.learnable_temperature = True
    args.tau_root_init = 1.0
    args.tau_rule_init = 1.0
    args.tau_term_init = 1.0
    return HN_PCFG(args, _DummyDataset())


def _check_variant(scoring_fn, ref_fn, atol=1e-5):
    torch.manual_seed(0)
    s_dim = 8
    model = _build_model(scoring_fn, s_dim).double()
    n_source, n_target = 4, 5
    v = torch.randn(s_dim, dtype=torch.float64)
    source = torch.randn(n_source, s_dim, dtype=torch.float64)
    target = torch.randn(n_target, s_dim, dtype=torch.float64)
    scores = _scores(model, v, source, target)
    assert scores.shape == (n_target, n_source), scores.shape
    for t in range(n_target):
        for s in range(n_source):
            expected = ref_fn(v, source[s], target[t])
            actual = scores[t, s]
            assert torch.allclose(actual, expected, atol=atol), (
                f"{scoring_fn}: mismatch at (t={t}, s={s}) "
                f"actual={actual.item()} expected={expected.item()}"
            )


def test_hole():
    _check_variant('hole', _ref_hole)


def test_hadamard():
    _check_variant('hadamard', _ref_hadamard)


def test_conv():
    _check_variant('conv', _ref_conv)


if __name__ == '__main__':
    test_hole()
    print('hole OK')
    test_hadamard()
    print('hadamard OK')
    test_conv()
    print('conv OK')
