import contextlib
import csv
import signal
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import rl.ppo as ppo
import rl.train as tr
from rl.config import Config, parse_args
from rl.encode import Caps


def _rows(path, name="progress.csv"):
    with (path / name).open() as f:
        return list(csv.DictReader(f))


def test_tiny_six_updates_and_resume(tmp_path):
    args = ["--preset", "tiny", "--run_dir", str(tmp_path), "--async_update",
            "--ckpt_every", "2", "--lr_kl_target", "0.002"]
    state = tr.train(parse_args(args + ["--total_updates", "6"]))
    rows = _rows(tmp_path)
    assert state["update"] == 6 and state["env_steps"] == 6 * 512
    assert [int(r["update"]) for r in rows] == list(range(1, 7))
    assert {"approx_kl", "approx_kl_first", "update_wait", "kl_ctrl"} <= rows[0].keys()
    losses = [float(r["v_loss"]) for r in rows]
    assert min(losses[1:]) < losses[0]
    ema = None
    for row in rows:
        kl = float(row["kl_ctrl"])
        assert kl >= 1e-8 and float(row["update_wait"]) >= 0
        ema = kl if ema is None else .9 * ema + .1 * kl
    assert state["kl_ema"] == pytest.approx(ema)
    ck = torch.load(tmp_path / "ckpt/latest.pt", weights_only=False)
    assert ck["update"] == 6 and ck["env_steps"] == 6 * 512
    assert ck["lr_now"] == state["lr_now"] and ck["kl_ema"] == state["kl_ema"]
    resumed = tr.train(parse_args(args + ["--total_updates", "8", "--resume", "auto"]))
    rows = _rows(tmp_path)
    assert resumed["update"] == 8 and resumed["env_steps"] == 8 * 512
    assert [int(r["update"]) for r in rows] == list(range(1, 9))
    assert float(rows[6]["lr"]) == state["lr_now"]


def test_async_fakeproc(tmp_path):
    cfg = parse_args(["--preset", "tiny", "--run_dir", str(tmp_path), "--async_update", "true",
                      "--engine", "fakeproc", "--fake_args=--step_ms 1", "--total_updates", "2"])
    assert tr.train(cfg)["update"] == 2
    assert len(_rows(tmp_path)) == 2


@pytest.mark.parametrize("kl_target", [0, .002])
def test_sync_log_and_kl_control(tmp_path, monkeypatch, kl_target):
    inputs, stats = [], []
    real_step, real_update = tr.kl_lr_step, tr.ppo_update

    def step(lr, ema, kl, cfg):
        inputs.append(kl)
        return real_step(lr, ema, kl, cfg)

    def update(*args, **kwargs):
        result = real_update(*args, **kwargs)
        stats.append(result.copy())
        return result

    monkeypatch.setattr(tr, "kl_lr_step", step)
    monkeypatch.setattr(tr, "ppo_update", update)
    cfg = parse_args(["--preset", "tiny", "--run_dir", str(tmp_path), "--async_update", "false",
                      "--total_updates", "1", "--lr_kl_target", str(kl_target)])
    assert tr.train(cfg)["update"] == 1
    row = _rows(tmp_path)[0]
    assert {"update_wait", "kl_ctrl"}.isdisjoint(row)
    assert "approx_kl_first" in stats[0]
    if kl_target:
        assert "approx_kl_first" in row
        assert inputs == [stats[0]["approx_kl"]]
    else:
        assert not inputs


def test_async_config():
    assert not Config().async_update and not Config.from_dict({}).async_update
    assert parse_args(["--async_update"]).async_update
    for flags in (["--group_k", "2"], ["--branch_decisions", "rest_site"], ["--explore_eps", ".1"]):
        with pytest.raises(ValueError, match="async_update cannot be combined"):
            parse_args(["--async_update", *flags])


def test_first_kl_precedes_optimizer_step_and_weights_microbatches(monkeypatch):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(0.))

        def forward(self, batch):
            x = batch["decision"].float() * .2 + self.p
            return torch.stack((x, -self.p.expand_as(x)), -1), self.p.expand_as(x)

    buf = ppo.Rollout(Caps(max_tok=8, max_ent=1, max_cand=2, max_cent=1), 5, 1, torch.device("cpu"))
    buf.n = 5
    buf.obs["n_tok"][:] = 1
    buf.obs["n_cand"][:] = 2
    buf.obs["decision"][:] = torch.arange(5)
    model = Toy()
    logits, _ = model(buf.obs)
    shifts = np.array([.2, -.4, .8, -.1, .5], np.float32)
    buf.logp[:] = logits.log_softmax(-1)[:, 0].detach().numpy() - shifts
    first = np.array([4, 0, 3])
    monkeypatch.setattr(ppo, "bucketed_minibatches", lambda *args: [first, np.array([1, 2])])
    cfg = Config(epochs=2, minibatch=3, micro_batch=2, value_norm=False)
    stats = ppo.ppo_update(model, torch.optim.SGD(model.parameters(), lr=.1), buf,
                           np.arange(5, dtype=np.float32), np.zeros(5, np.float32), np.ones(5, bool),
                           ppo.ValueNorm(), cfg, np.random.default_rng(0), contextlib.nullcontext)
    expected = (np.expm1(shifts[first]) - shifts[first]).mean()
    assert stats["approx_kl_first"] == pytest.approx(expected, abs=1e-7)
    assert stats["approx_kl"] != pytest.approx(stats["approx_kl_first"], abs=1e-5)


@pytest.mark.parametrize("ending", ["total_updates", "keyboard", "sigterm", "sigint"])
def test_overlap_refresh_and_consistent_shutdown(tmp_path, monkeypatch, ending):
    entered, release, active = threading.Event(), threading.Event(), threading.Event()
    collected, updated, saved = [], [], []
    holder = {}
    main_thread = threading.get_ident()

    class Pool:
        def __init__(self, cfg, resume_steps=0):
            self.env_steps = SimpleNamespace(value=resume_steps)
            self.closed = False
            holder["pool"] = self

        def close(self):
            self.closed = True

    class Buffer:
        def __init__(self, *args):
            self.n = 2

        def gae(self, *args):
            return None, None, None

    def collect(self, buf):
        holder["col"] = self
        collected.append(buf)
        number = len(collected)
        if number == 2:
            assert entered.wait(5)
            assert active.is_set() and not self.model.training
            assert self.model is not self.updater.model and self.vnorm is not self.updater.vnorm
            old = {k: v.clone() for k, v in self.model.state_dict().items()}
            assert any(not torch.equal(v, self.updater.model.state_dict()[k]) for k, v in old.items())
            with self.amp_ctx():
                assert all(torch.equal(v, self.model.state_dict()[k]) for k, v in old.items())
                assert self.vnorm.w == 0
            release.set()
            if ending == "keyboard":
                raise KeyboardInterrupt
            if ending in ("sigterm", "sigint"):
                sig = signal.SIGTERM if ending == "sigterm" else signal.SIGINT
                signal.getsignal(sig)(sig, None)
                with self.amp_ctx():
                    pytest.fail("collection continued after stop")
            self.updater.future.result(timeout=5)
            with self.amp_ctx():
                assert all(torch.equal(v, self.updater.model.state_dict()[k])
                           for k, v in self.model.state_dict().items())
                assert self.vnorm.state_dict() == self.updater.vnorm.state_dict()
        self.episodes.append(dict(seed=str(number), victory=False, floor=1, ret=1., steps=2))
        return np.zeros(1, np.float32), dict(dec_counts=np.ones(1), env_sps=1.)

    def update(model, opt, buf, adv, ret, mask, vnorm, cfg, rng, amp_ctx):
        assert threading.get_ident() != main_thread
        assert not active.is_set()
        active.set()
        updated.append(buf)
        opt.zero_grad()
        model(torch.ones(1, 1)).sum().backward()
        opt.step()
        vnorm.update([len(updated)])
        entered.set()
        assert release.wait(5)
        active.clear()
        return dict(approx_kl=.3, approx_kl_first=.1, v_loss=1.)

    real_save = tr.save_ckpt

    def save(path, model, opt, vnorm, cfg, state, keep, rng):
        assert not active.is_set() and holder["col"].updater.future.done()
        assert state["update"] == len(updated)
        assert all(int(s["step"]) == state["update"] for s in opt.state.values())
        assert vnorm.state_dict() == holder["col"].vnorm.state_dict()
        saved.append(state["update"])
        return real_save(path, model, opt, vnorm, cfg, state, keep, rng)

    monkeypatch.setattr(tr, "Policy", lambda cfg: torch.nn.Linear(1, 1))
    monkeypatch.setattr(tr, "ActorPool", Pool)
    monkeypatch.setattr(tr, "Rollout", Buffer)
    monkeypatch.setattr(tr._AsyncCollector, "collect", collect)
    monkeypatch.setattr(tr, "ppo_update", update)
    monkeypatch.setattr(tr, "save_ckpt", save)
    cfg = Config(run_dir=str(tmp_path), device="cpu", async_update=True, total_updates=3,
                 ckpt_every=1, log_every=999, lr_kl_target=.002)
    state = tr.train(cfg)
    n = 3 if ending == "total_updates" else 1
    assert state["update"] == n and state["env_steps"] == 2 * n
    assert holder["pool"].closed and not active.is_set()
    assert saved == list(range(1, n + 1)) + [n]
    assert len({id(buf) for buf in collected}) == 2
    assert updated == collected[:n]
    if n == 3:
        assert collected[0] is collected[2] and collected[1] is collected[3]
    rows = _rows(tmp_path)
    assert [int(r["update"]) for r in rows] == list(range(1, n + 1))
    assert all(float(r["kl_ctrl"]) == pytest.approx(.2) for r in rows)
    assert [r["seed"] for r in _rows(tmp_path, "episodes.csv")] == [str(i) for i in range(1, n + 1)]
    ck = torch.load(tmp_path / "ckpt/latest.pt", weights_only=False)
    assert ck["update"] == n and ck["env_steps"] == 2 * n


def test_infer_device_needs_async_and_runs_when_equal(tmp_path):
    with pytest.raises(ValueError):
        Config(infer_device="cpu")
    st = tr.train(parse_args(["--preset", "tiny", "--run_dir", str(tmp_path), "--async_update", "true", "--infer_device", "cpu",
                              "--total_updates", "3"]))
    assert st["update"] == 3
