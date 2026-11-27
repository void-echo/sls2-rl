import contextlib
import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import rl.evaluate as ev
import rl.train as tr
from rl.actors import ActorPool, SharedObs, _Worker
from rl.config import Config, parse_args
from rl.encode import Caps, Encoder, HASH_ROWS, candidate_type, field_spec, k2
from rl.ppo import Rollout, ValueNorm, ppo_update, to_torch
from rl.protocol import decision_id
from rl.tests.test_force import tiny_config


TASTE_STATS = {"taste_ep_frac", "taste_win", "main_win", "taste_steps", "taste_kl", "taste_changed"}
TASTE_DECISIONS = {decision_id("card_reward"), decision_id("shop")}


def test_taste_config():
    cfg = Config.from_dict({})
    assert (cfg.taste_frac, cfg.taste_beta) == (0.0, 0.0)
    assert not cfg.taste_enabled
    cfg = parse_args(["--taste_frac", ".5", "--taste_beta", "2", "--explore_eps", ".2"])
    assert (cfg.taste_frac, cfg.taste_beta, cfg.explore_eps) == (.5, 2., .2)
    assert cfg.taste_enabled and Config.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize("flags", [
    ["--force_ids", "CARD.X", "--force_frac", ".5"],
    ["--async_update"],
    ["--group_k", "2"],
    ["--branch_decisions", "rest_site"],
])
def test_taste_rejects_incompatible_modes(flags):
    with pytest.raises(ValueError, match="taste cannot be combined"):
        parse_args(["--taste_frac", "1", "--taste_beta", "2", *flags])
    assert not parse_args(["--taste_frac", "1", *flags]).taste_enabled
    assert not parse_args(["--taste_beta", "2", *flags]).taste_enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_shares_card_identity_rows(tmp_path, enabled):
    cfg = tiny_config(tmp_path, taste_frac=1., taste_beta=float(enabled))
    shm = SharedObs(cfg.caps, 1, path=str(tmp_path / "shm"), taste=cfg.taste_enabled)
    worker = _Worker(0, [0], None, shm.path, dict(cfg.to_dict(), n_envs=1), SimpleNamespace(value=0), None, None)
    reward_card = dict(k="reward_card", card=dict(id="CARD.X"))
    buy_card = dict(k="buy_card", card=dict(id="CARD.Y"))
    key_x, key_y = (k2("card.id", name) for name in ("CARD.X", "CARD.Y"))
    candidates = [reward_card, buy_card, dict(k="buy_card", card={}), dict(k="buy_relic", id="CARD.Y")]
    try:
        assert set(shm.arrays) == set(field_spec(cfg.caps)) | ({"cand_card_rows"} if enabled else set())
        for decision, cands, hits in (
            ("card_reward", candidates, {0: key_x}),
            ("shop", candidates, {1: key_y}),
            ("shop", [dict(k="leave")], {}),
            ("combat_play", candidates, {}),
        ):
            response = dict(decision=decision, cands=cands)
            worker.encode(0, response)
            for name, data in Encoder(cfg.caps).encode(response).items():
                np.testing.assert_array_equal(shm.arrays[name][0], data)
            if enabled:
                expected = np.full((cfg.caps.max_cand, 2), -1, np.int32)
                for j, key in hits.items():
                    expected[j] = ((key >> 16) & 0xFFFF, key & 0xFFFF)
                actual = shm.arrays["cand_card_rows"]
                assert actual.dtype == np.int32
                np.testing.assert_array_equal(actual[0], expected)
                gathered = shm.gather([0])["cand_card_rows"]
                np.testing.assert_array_equal(gathered[0], expected[:gathered.shape[1]])
    finally:
        worker.shutdown()
        worker.shm.close()
        shm.close()


class CardPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        rng = torch.Generator().manual_seed(17)
        self.table = torch.nn.Parameter(torch.randn(HASH_ROWS, 3, generator=rng))

    def forward(self, obs):
        assert "cand_card_rows" not in obs
        types = obs["cand_type"]
        cards = (types == candidate_type("reward_card")) | (types == candidate_type("buy_card"))
        legal = torch.arange(types.shape[1])[None] < obs["n_cand"][:, None]
        return (100. * cards).masked_fill(~legal, -torch.inf), torch.full((len(types),), .25)


@pytest.mark.parametrize("beta", [50., 1e-4])
def test_fake_collection_uses_taste_sampling_and_unbiased_logp(tmp_path, monkeypatch, beta):
    cfg = tiny_config(tmp_path, taste_frac=1., taste_beta=beta)
    pool = ActorPool(cfg, seed_list=[f"force_{i}" for i in range(2000)])
    model = CardPolicy()
    device = torch.device("cpu")
    col = tr.Collector(cfg, pool, model, ValueNorm(), device, contextlib.nullcontext, err_log=False)
    buf = Rollout(cfg.caps, cfg.rollout_steps, 1, device)
    real_act, real_rand = tr.act_batch, torch.rand_like
    records, seen = [], set()
    direction = None
    real_handle = col.handle

    def handle(events, b):
        nonlocal direction
        for event in events:
            real_handle([event], b)
            if event[0] == "ep_start":
                direction = col.taste[0]
                assert direction.shape == (3,)
                assert np.linalg.norm(direction) == pytest.approx(1.)
            elif event[0] == "ep":
                assert event[1]["taste"] == 1 and not col.taste

    def act(model, obs, amp_ctx, **kwargs):
        if not kwargs.get("return_logits"):
            return real_act(model, obs, amp_ctx, **kwargs)
        assert col.taste[0] is direction
        decision = int(obs["decision"][0])
        target = decision in TASTE_DECISIONS
        n = int(obs["n_cand"][0])
        rows = pool.shm.arrays["cand_card_rows"][0, :n].copy()

        def sample_uniform(logits):
            nonlocal noise
            u = torch.full_like(logits, .5) if beta == 50. and target else real_rand(logits)
            u = u.clamp(1e-10, 1 - 1e-7)
            noise = torch.log(-torch.log(u))[0, :n]
            return u

        noise = None
        with monkeypatch.context() as patch:
            patch.setattr(torch, "rand_like", sample_uniform)
            actions, logp, value, batch_logits = real_act(model, obs, amp_ctx, **kwargs)
        logits = batch_logits[0, :n]
        expected = logits.clone()
        valid = rows[:, 0] >= 0
        if target:
            with torch.no_grad():
                embeds = model.table[rows[valid, 0]] + model.table[rows[valid, 1]]
                scores = (embeds @ torch.from_numpy(direction)) / embeds.norm(dim=-1)
                expected[valid] += beta * scores
            seen.add((decision, bool(valid.any())))
            if beta == 50. and valid.any():
                best = np.flatnonzero(valid)[scores.argmax()]
                assert rows[actions.item()].tolist() == rows[best].tolist()
        action = actions.item()
        assert action == (expected - noise).argmax()
        assert logp.item() == pytest.approx(logits.log_softmax(-1)[action].item())
        p = torch.distributions.Categorical(logits=expected.double())
        q = torch.distributions.Categorical(logits=logits.double())
        kl = torch.distributions.kl_divergence(p, q).item()
        records.append((action, logp.item(), target, kl, expected.argmax().item() != logits.argmax().item()))
        return actions, logp, value, batch_logits

    col.handle = handle
    monkeypatch.setattr(tr, "act_batch", act)
    torch.manual_seed(17)
    try:
        for _ in range(4):
            records.clear()
            _, stats = col.collect(buf)
            actions, logps, targets, kls, changed = map(np.array, zip(*records))
            np.testing.assert_array_equal(buf.act[:buf.n], actions)
            np.testing.assert_array_equal(buf.logp[:buf.n], logps.astype(np.float32))
            np.testing.assert_array_equal(buf.explored[:buf.n], targets)
            assert stats["taste_steps"] == targets.sum()
            if targets.any():
                assert stats["taste_changed"] == pytest.approx(changed[targets].mean())
                assert stats["taste_kl"] == pytest.approx(kls[targets].mean(), abs=1e-7, rel=2e-6)
                if beta < 1.:
                    assert stats["taste_kl"] < 1e-6
        assert not col.errors
        assert {(decision, True) for decision in TASTE_DECISIONS} <= seen
        assert (decision_id("shop"), False) in seen
    finally:
        pool.close()


def test_disabled_training_logs_no_taste_columns(tmp_path, monkeypatch):
    def pool(cfg, **kwargs):
        actors = ActorPool(cfg, **kwargs)
        assert set(actors.shm.arrays) == set(field_spec(cfg.caps))
        return actors

    monkeypatch.setattr(tr, "ActorPool", pool)
    cfg = tiny_config(tmp_path, total_updates=1, max_episode_steps=2)
    assert tr.train(cfg)["update"] == 1
    with (tmp_path / "progress.csv").open() as f:
        assert TASTE_STATS.isdisjoint(next(csv.DictReader(f)))
    with (tmp_path / "episodes.csv").open() as f:
        assert "taste" not in next(csv.DictReader(f))


def test_training_logs_taste_fraction_and_win_rates(tmp_path, monkeypatch):
    def pool(cfg, **kwargs):
        return ActorPool(cfg, seed_list=[f"force_logs_{i}" for i in range(2000)], **kwargs)

    monkeypatch.setattr(tr, "ActorPool", pool)
    cfg = tiny_config(tmp_path, total_updates=3, epochs=1, rollout_steps=512, ascension=35, taste_frac=.5, taste_beta=2.)
    assert tr.train(cfg)["update"] == 3
    with (tmp_path / "progress.csv").open() as f:
        rows = list(csv.DictReader(f))
    with (tmp_path / "episodes.csv").open() as f:
        episodes = [e for e in csv.DictReader(f) if not e["aborted"]]
    assert .35 < np.mean([int(e["taste"]) for e in episodes]) < .65
    assert TASTE_STATS <= rows[0].keys()
    for row in rows:
        done = [e for e in episodes if e["update"] == row["update"]]
        assert float(row["taste_ep_frac"]) == np.mean([int(e["taste"]) for e in done])
        for flag, key in [("1", "taste_win"), ("0", "main_win")]:
            group = [e for e in done if e["taste"] == flag]
            expected = np.mean([e["victory"] == "True" for e in group]) if group else ""
            assert (float(row[key]) if row[key] else "") == expected
        assert 0 <= int(row["taste_steps"]) <= cfg.rollout_steps


def test_taste_episodes_do_not_enter_curriculum():
    c = tr.AscCurriculum(Config(asc_curriculum=True, asc_window=2, asc_threshold=1.), 1)
    assert not c.record(dict(asc=1, victory=True, taste=1))
    assert not c.windows
    assert not c.record(dict(asc=1, victory=True, taste=0))
    assert c.record(dict(asc=1, victory=True, taste=0))


def test_ppo_skips_explored_policy_gradients_and_keeps_value_gradients():
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(3, 2))
            self.values = torch.nn.Parameter(torch.zeros(3))

        def forward(self, obs):
            rows = obs["decision"].long()
            return self.logits[rows], self.values[rows]

    cfg = Config(taste_frac=1., taste_beta=2., epochs=1, minibatch=3, micro_batch=3,
                 value_norm=False, ent_coef=0., max_grad_norm=10.)
    caps = Caps(max_tok=8, max_cand=2)
    encoded = Encoder(caps).encode(dict(decision="card_reward", cands=[dict(k="skip")] * 2))
    obs = {k: np.repeat(v[None], 3, axis=0) for k, v in encoded.items()}
    obs["decision"][:] = [0, 1, 2]
    buf = Rollout(caps, 3, 1, torch.device("cpu"))
    buf.add([0] * 3, to_torch(obs, "cpu"), [0] * 3, -np.log(2) + np.array([0., 1., -1.]), [0.] * 3)
    buf.explored[1:] = True
    model = Policy()
    stats = ppo_update(model, torch.optim.SGD(model.parameters(), lr=.1), buf, np.array([1., 2., -2.], np.float32),
                       np.ones(3, np.float32), np.ones(3, bool), ValueNorm(), cfg, np.random.default_rng(17),
                       contextlib.nullcontext)
    assert model.logits.grad[0].abs().sum() > 0
    assert (model.logits.grad[1:] == 0).all()
    assert (model.values.grad != 0).all()
    assert stats["approx_kl"] == stats["clipfrac"] == 0.
    assert stats["explored_frac"] == pytest.approx(2 / 3)


def test_evaluation_disables_taste(tmp_path, monkeypatch):
    cfg = tiny_config(tmp_path, taste_frac=1., taste_beta=50.)

    def pool(eval_cfg, seed_list):
        assert eval_cfg.taste_frac == 0. and not eval_cfg.taste_enabled
        actors = ActorPool(eval_cfg, seed_list)
        assert "cand_card_rows" not in actors.shm.arrays
        return actors

    monkeypatch.setattr(ev, "ActorPool", pool)
    model = CardPolicy()
    seeds = [("Ironclad", f"taste_eval_{i}") for i in range(4)]
    actual = ev.run_eval(cfg, model, seeds, progress=False)
    assert cfg.taste_enabled
    expected = ev.run_eval(tiny_config(tmp_path), model, seeds, progress=False)
    assert actual == expected and all("taste" not in e for e in actual)
