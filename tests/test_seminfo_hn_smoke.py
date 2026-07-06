"""Smoke test: build HNPCFGPairwise/FixedCostReward with each ablation variant.

Mirrors hol-pcfg/tests/test_hn_pcfg_smoke.py.
"""
import torch

from parsing_by_maxseminfo.parser.model.HN_PCFG import (
    HNPCFGPairwise,
    HNPCFGFixedCostReward,
)


def _make_args(**kwargs):
    class _Args:
        pass

    a = _Args()
    a.NT = 8
    a.T = 4
    a.s_dim = 16
    a.tau_root_init = 1.5
    a.tau_rule_init = 2.0
    a.tau_term_init = 1.2
    for k, v in kwargs.items():
        setattr(a, k, v)
    return a


def _fake_input():
    seq_len = torch.tensor([5, 4])
    word = torch.randint(0, 50, (2, 5))
    return {'word': word, 'seq_len': seq_len}


def _build_and_forward(model_cls, model_args):
    model = model_cls(model_args, vocab_size=50)
    rules = model(_fake_input())
    assert set(rules.keys()) == {
        'unary', 'root', 'left_m', 'right_m', 'left_p', 'right_p', 'kl'
    }
    return model


def test_default():
    model = _build_and_forward(HNPCFGPairwise, _make_args())
    assert isinstance(model.log_tau_root, torch.nn.Parameter)
    assert model.log_tau_root.requires_grad
    assert model.scoring_fn == 'hole'
    assert model.complex_normalization is True


def test_hadamard():
    model = _build_and_forward(HNPCFGPairwise, _make_args(scoring_fn='hadamard'))
    assert model.scoring_fn == 'hadamard'


def test_conv():
    model = _build_and_forward(HNPCFGPairwise, _make_args(scoring_fn='conv'))
    assert model.scoring_fn == 'conv'


def test_no_cnorm_does_not_project():
    model = _build_and_forward(HNPCFGPairwise, _make_args(complex_normalization=False))
    assert model.complex_normalization is False
    pre = model.rule_state_emb.data.clone()
    model.rule_state_emb.data.mul_(3.0)
    model.project_embeddings()
    assert torch.allclose(model.rule_state_emb.data, pre.mul(3.0))


def test_learnable_temp_false_freezes_tau():
    model = _build_and_forward(HNPCFGPairwise, _make_args(learnable_temperature=False))
    for name in ('log_tau_root', 'log_tau_rule', 'log_tau_term'):
        buf = getattr(model, name)
        assert not isinstance(buf, torch.nn.Parameter), name
        assert torch.equal(buf, torch.zeros_like(buf)), name


def test_fixed_cost_reward_inherits_switches():
    """HNPCFGFixedCostReward subclass should pick up the same switches."""
    model = HNPCFGFixedCostReward(
        _make_args(scoring_fn='hadamard', learnable_temperature=False),
        vocab_size=50,
    )
    assert model.scoring_fn == 'hadamard'
    assert model.learnable_temperature is False


if __name__ == '__main__':
    torch.manual_seed(0)
    test_default()
    print('default OK')
    torch.manual_seed(0)
    test_hadamard()
    print('hadamard OK')
    torch.manual_seed(0)
    test_conv()
    print('conv OK')
    torch.manual_seed(0)
    test_no_cnorm_does_not_project()
    print('no_cnorm OK')
    torch.manual_seed(0)
    test_learnable_temp_false_freezes_tau()
    print('no_tau OK')
    torch.manual_seed(0)
    test_fixed_cost_reward_inherits_switches()
    print('fixed_cost_reward OK')
