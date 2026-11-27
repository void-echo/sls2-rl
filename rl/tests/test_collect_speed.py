import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import rl.train as tr
from rl.bench import make_obs
from rl.config import Config, parse_args
from rl.encode import Caps
from rl.ppo import Rollout


SCALARS = ("act", "logp", "val", "rew", "done", "trunc", "cut", "bootstrap", "has_res",
           "forced", "combat", "explored")


def snapshot(buf):
    return ({k: v[:buf.n].cpu().numpy().tobytes() for k, v in buf.obs.items()},
            {k: getattr(buf, k)[:buf.n].tobytes() for k in SCALARS},
            copy.deepcopy((buf.env_idx, buf.groups, buf.branches, buf.branch_extra)))


def test_rollout_handoff_prefix_and_metadata():
    caps = Caps(max_tok=8, max_ent=2, max_cand=8, max_cent=2)
    src, dst = [Rollout(caps, 9, 3, torch.device("cpu"), track_combat=True) for _ in range(2)]
    src.n = 5
    for v in src.obs.values():
        v[:5] = torch.arange(v[:5].numel()).reshape(v[:5].shape).to(v.dtype)
    for k in SCALARS:
        getattr(src, k)[:5] = np.arange(5)
        getattr(dst, k)[5:] = 7
    src.env_idx = [[0, 3], [1, 4], [2]]
    src.groups = [dict(trials=[[0, 3]], wins=[True], boots=[0.5])]
    src.branches = [dict(trials=[[1, 4]], boots=[0.2])]
    src.branch_extra = 2
    expected = snapshot(src)
    gae = src.gae(np.zeros(3), .99, .95)
    dst.copy_from(src)
    assert snapshot(dst) == expected
    for a, b in zip(gae, dst.gae(np.zeros(3), .99, .95)):
        assert a.tobytes() == b.tobytes()
    assert np.all(dst.act[5:] == 7)
    assert torch.count_nonzero(dst.obs["tok_key"][5:]) == 0
    src.reset()
    src.act[:] = 0
    assert snapshot(dst) == expected


def test_split_first_ppo_input_matches_two_learner_buffers(tmp_path, monkeypatch):
    caps = Caps(max_tok=64, max_ent=16, max_cand=64, max_cent=8)
    arrays = make_obs(64, caps, inflate=0)
    captured = []

    class Pool:
        def __init__(self, cfg, resume_steps=0):
            self.n_envs = cfg.n_envs
            self.env_steps = SimpleNamespace(value=0)
            self.shm = self
            self.steps = np.zeros(self.n_envs, np.int64)
            self.events = [("obs", e) for e in range(self.n_envs)]

        def gather(self, envs):
            idx = [(self.steps[e] * self.n_envs + e) % 64 for e in envs]
            return {k: v[idx].copy() for k, v in arrays.items()}

        def act(self, envs, actions):
            for e, a in zip(envs, actions):
                self.steps[e] += 1
                self.events.append(("res", e, float(a) / 10, self.steps[e] % 7 == 0, False, True))

        def poll(self, timeout):
            events, self.events = self.events, []
            return events

        def close(self):
            pass

    def update(model, opt, buf, adv, ret, mask, *args, **kwargs):
        captured.append((snapshot(buf), tuple(x.tobytes() for x in (adv, ret, mask))))
        return dict(approx_kl=0., approx_kl_first=0., v_loss=0.)

    monkeypatch.setattr(tr, "ActorPool", Pool)
    monkeypatch.setattr(tr, "ppo_update", update)
    monkeypatch.setattr(tr, "save_ckpt", lambda *args: None)
    for split in (False, True):
        cfg = parse_args(["--preset", "tiny", "--async_update", "true", "--async_split_buffers", str(split),
                          "--rollout_steps", "32", "--total_updates", "2", "--log_every", "999",
                          "--lam_build", ".5", "--run_dir", str(tmp_path / str(split))])
        cfg.caps = caps
        # Exercise both layouts on CPU; device selection itself is checked separately.
        monkeypatch.setattr(tr, "_split_buffers", lambda *args: split)
        assert tr.train(cfg)["update"] == 2
    assert captured[0] == captured[2]
    assert captured[1] == captured[3]


def test_speed_flags_default_off_and_split_device_selection(monkeypatch):
    cfg = Config()
    assert not cfg.async_split_buffers and not cfg.collect_fast
    assert not Config.from_dict({}).collect_fast
    cfg = parse_args(["--async_update", "true", "--async_split_buffers", "true", "--collect_fast", "true"])
    assert cfg.async_split_buffers and cfg.collect_fast
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert not tr._split_buffers(cfg, torch.device("cuda"), torch.device("cuda:0"))
    assert tr._split_buffers(cfg, torch.device("cuda:0"), torch.device("cuda:1"))
    assert not tr._split_buffers(cfg, torch.device("cpu"), torch.device("cpu:0"))
    cfg.async_update = False
    assert not tr._split_buffers(cfg, torch.device("cuda:0"), torch.device("cuda:1"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require a GPU")
@pytest.mark.parametrize("amp", [False, True])
def test_fast_policy_matches_and_refreshes_weights(amp, monkeypatch):
    from rl.collect_fast import FastPolicy
    from rl.model import ModelCfg, Policy
    from rl.ppo import to_torch

    dev = torch.device("cuda:0")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", amp)
    caps = Caps(max_tok=64, max_ent=16, max_cand=64, max_cent=8)
    torch.manual_seed(7)
    model = Policy(ModelCfg(d=64, layers=1, heads=4)).to(dev).eval().requires_grad_(False)
    fast = FastPolicy(model, caps, 64, dev, amp)
    batch = make_obs(37, caps, inflate=0)
    cfg = Config(amp=amp)
    ctx = tr.amp_context(cfg, dev)
    for revision in range(2):
        if revision:
            model.value[-1].bias.add_(.25)
            model.table.add_(.002)
        for n in (19, 37, 19):
            b = {k: v[:n] for k, v in batch.items()}
            old = to_torch(b, dev)
            new = fast.prepare(b)
            assert all(torch.equal(old[k], new[k]) for k in old)
            state = torch.cuda.get_rng_state(dev)
            expected = tr.act_batch(model, old, ctx, return_logits=True)
            torch.cuda.set_rng_state(state, dev)
            actual = tr.act_batch(fast, new, ctx, return_logits=True)
            assert torch.equal(actual[0], expected[0])
            for a, e in zip(actual[1:], expected[1:]):
                torch.testing.assert_close(a, e, rtol=0, atol=1e-2 if amp else 1e-6)
