"""Small activation/rank checks for the offline plasticity diagnostic."""
import pytest
import torch

from rl.encode import Caps, Encoder
from rl.engine import FakeEngine
from rl.model import ModelCfg, Policy
from rl.ppo import to_torch
from tools.plasticity import Diagnostics, StateBatches, feature_rank, value_forward


def test_feature_rank():
    identity = feature_rank(torch.eye(8))
    rank_one = feature_rank(torch.ones(8, 8))
    assert identity["srank"] == 8
    assert identity["entropy_rank"] == pytest.approx(8)
    assert rank_one["srank"] == 1
    assert rank_one["entropy_rank"] == pytest.approx(1, abs=1e-5)
    assert feature_rank(torch.zeros(8, 8))["srank"] == 0


def test_zero_unit_increases_dormancy():
    torch.manual_seed(0)
    model = Policy(ModelCfg(d=16, heads=2, layers=1, ff_mult=2)).eval()
    enc = Encoder(Caps(max_tok=48, max_ent=8, max_cand=32, max_cent=4))
    obs = enc.alloc(4)
    engine = FakeEngine()
    try:
        r = engine.reset(seed="plasticity_test")
        for i in range(4):
            enc.encode(r, {k: v[i, ...] for k, v in obs.items()})
            r = engine.step(0)
    finally:
        engine.close()
    states = StateBatches(to_torch(obs, "cpu"), "cpu")
    combat = torch.tensor([False, True, False, True])
    before = Diagnostics(model, states, combat).run(2)
    with torch.no_grad():
        first = model.blocks[0].mlp[0]
        first.weight[0].zero_()
        first.bias[0].zero_()
    after = Diagnostics(model, states, combat).run(2)
    b = before["dormant"]["layers"]["blocks.0.mlp"]
    a = after["dormant"]["layers"]["blocks.0.mlp"]
    for threshold in ("0.025", "0.1"):
        assert a["fraction"][threshold] > b["fraction"][threshold]
    assert a["scores"][0] == 0
    assert a["n_samples"] == int(obs["n_tok"].sum())
    batch = states.get(torch.arange(4))
    with torch.no_grad():
        torch.testing.assert_close(value_forward(model, batch), model(batch)[1])
