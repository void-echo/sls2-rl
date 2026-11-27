"""Frozen policy gradients, independent run sampling, and path geometry."""
import numpy as np
import pytest
import torch

from rl.encode import Caps, Encoder
from rl.engine import FakeEngine
from rl.model import ModelCfg, Policy
from rl.ppo import to_torch
from tools.grad_snr import episode_pair, fit_noise_ratio, gradient_cosines, policy_gradient
from tools.path_straightness import distances, parameter_groups, path_straightness


def test_episode_pair_never_splits_runs():
    rng = np.random.default_rng(12)
    # Interleaved collection order; one run can span multiple rollout chunks.
    episodes = np.tile(np.arange(30), 17)
    valid = np.ones(len(episodes), bool)
    valid[::11] = False
    for k in (48, 150, 1000):
        a, b = episode_pair(episodes, valid, k, rng)
        assert not set(episodes[a]) & set(episodes[b])
        assert valid[a].all() and valid[b].all()
        for ix in (a, b):
            np.testing.assert_array_equal(np.sort(ix), np.flatnonzero(valid & np.isin(episodes, episodes[ix])))
        if k == 1000:
            assert len(a) + len(b) == valid.sum()
        else:
            assert k <= len(a) < k + 17 and k <= len(b) < k + 17


def test_policy_gradient_microbatches_and_identical_sample_cosine():
    torch.manual_seed(8)
    model = Policy(ModelCfg(d=16, heads=2, layers=1, ff_mult=2))
    enc = Encoder(Caps(max_tok=48, max_ent=8, max_cand=32, max_cent=4))
    obs = enc.alloc(7)
    engine = FakeEngine()
    try:
        r = engine.reset(seed="noise-gradient")
        for i in range(7):
            enc.encode(r, {key: value[i, ...] for key, value in obs.items()})
            r = engine.step(0)
    finally:
        engine.close()
    obs = to_torch(obs, "cpu")
    idx, adv, actions = np.arange(7), np.linspace(-1, 2, 7, dtype=np.float32), np.zeros(7, np.int64)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    micro = policy_gradient(model, [(0, 7, obs)], idx, adv, actions, 2)
    same = policy_gradient(model, [(0, 7, obs)], idx, adv, actions, 2)
    assert gradient_cosines(micro, same) == pytest.approx(dict(all=1, no_table=1), abs=1e-10)
    # Independent reference: a single dense forward, with the explicit objective.
    model.zero_grad(set_to_none=True)
    logits, _ = model(obs)
    (torch.as_tensor(adv) * logits.log_softmax(-1)[:, 0]).mean().backward()
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
        if p.grad is not None:
            torch.testing.assert_close(micro[name], p.grad, rtol=2e-4, atol=2e-6)
        if name.startswith("value."):
            assert p.grad is None


def test_noise_fit_recovers_known_c_and_no_signal_boundary():
    sizes = np.array([512, 2048, 8192], dtype=float)
    assert fit_noise_ratio(sizes, sizes / (sizes + 3210))["c"] == pytest.approx(3210, rel=1e-6)
    assert fit_noise_ratio(sizes, [-0.1, -0.03, -0.02])["status"] == "infinite_c"


def test_straight_line_path():
    direction = torch.arange(1, 129, dtype=torch.float64)
    states = [{"blocks.0.weight": (i * i + 3 * i) * direction} for i in range(20)]
    result = path_straightness(states, [1, 2, 4, 8])
    for row in result["trunk"]:
        assert row["mean"] == pytest.approx(1, abs=1e-12)
        assert row["n_valid"] == row["n_windows"] == 20 - row["window"]
    print("straight line: " + ", ".join(f"w={r['window']}: {r['mean']:.6f}" for r in result["trunk"]))


def test_random_walk_path():
    rng = np.random.default_rng(0)
    walk = np.vstack([np.zeros(1024), rng.normal(size=(256, 1024)).cumsum(axis=0)])
    states = [{"blocks.0.weight": torch.from_numpy(theta)} for theta in walk]
    rows = path_straightness(states, [1, 2, 4, 8])["all"]
    for row in rows:
        assert row["mean"] == pytest.approx(1 / np.sqrt(row["window"]), rel=0.03)
    print("random walk: " + ", ".join(f"w={r['window']}: {r['mean']:.6f} (baseline {r['random_walk_baseline']:.6f})" for r in rows))


def test_module_distances_and_stationary_path():
    assert parameter_groups("table") == ["all"]
    assert "trunk" in parameter_groups("blocks.0.attn.q.weight")
    assert "candidate" in parameter_groups("cand_blocks.0.self_attn.q.weight")
    assert "value" in parameter_groups("value.1.weight")
    a = {"table": torch.zeros(1), "logit.0.weight": torch.zeros(1)}
    b = {"table": torch.tensor([3.]), "logit.0.weight": torch.tensor([4.])}
    d = distances(a, b)
    assert d["all"] == 5 and d["no_table"] == d["logit"] == 4
    assert path_straightness([a, a], [1])["all"][0]["mean"] is None
