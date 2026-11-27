import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import rl.train as tr
from rl.config import Config, parse_args


def controlled_train(monkeypatch, path, updates, draws, resume=''):
    class Pool:
        def __init__(self, cfg, resume_steps=0):
            self.env_steps = SimpleNamespace(value=resume_steps)
        def close(self):
            pass
    class Buffer:
        def __init__(self, *args):
            self.n = 1
        def gae(self, *args):
            return None, None, None
    class Collector:
        def __init__(self, *args):
            self.episodes = []
        def collect(self, buf):
            return None, {}
    def update(model, opt, buf, adv, ret, mask, vn, cfg, rng, amp):
        cuda = torch.rand((), device='cuda').item() if torch.cuda.is_available() else None
        draws.append((torch.rand(()).item(), int(rng.integers(2**31)), cuda))
        opt.zero_grad()
        model(torch.ones(1, 1)).sum().backward()
        opt.step()
        return {}
    with monkeypatch.context() as m:
        m.setattr(tr, 'Policy', lambda cfg: torch.nn.Linear(1, 1))
        m.setattr(tr, 'ActorPool', Pool)
        m.setattr(tr, 'Rollout', Buffer)
        m.setattr(tr, 'Collector', Collector)
        m.setattr(tr, 'ppo_update', update)
        return tr.train(Config(run_dir=str(path), device='cpu', total_updates=updates, log_every=999,
                               ckpt_every=999, resume=resume))


def test_learner_rng_continues_and_old_checkpoint_loads(tmp_path, monkeypatch):
    continuous, split = [], []
    controlled_train(monkeypatch, tmp_path / 'continuous', 2, continuous)
    controlled_train(monkeypatch, tmp_path / 'split', 1, split)
    controlled_train(monkeypatch, tmp_path / 'split', 2, split, 'auto')
    assert continuous == split
    ckpt = tmp_path / 'split/ckpt/latest.pt'
    ck = torch.load(ckpt, weights_only=False)
    assert 'torch_rng_state' in ck and 'numpy_rng_state' in ck
    if torch.cuda.is_available():
        assert 'cuda_rng_state' in ck
    for field in ('torch_rng_state', 'numpy_rng_state', 'cuda_rng_state'):
        ck.pop(field, None)
    torch.save(ck, ckpt)
    assert controlled_train(monkeypatch, tmp_path / 'split', 3, [], 'auto')['update'] == 3


def test_tiny_three_updates_then_resume_five(tmp_path, monkeypatch):
    args = ['--preset', 'tiny', '--run_dir', str(tmp_path), '--floor_anneal_steps', '100000']
    real_collect, real_update, real_pool = tr.Collector.collect, tr.ppo_update, tr.ActorPool
    starts, shared, numpy_states = [], [], []
    def pool(cfg, seed_list=None, resume_steps=0):
        p = real_pool(cfg, seed_list=seed_list, resume_steps=resume_steps)
        shared.append(p.env_steps)
        assert p.env_steps.value == resume_steps
        return p
    def collect(self, buf):
        starts.append((self.pool.env_steps.value, torch.get_rng_state().clone()))
        return real_collect(self, buf)
    def update(*args):
        numpy_states.append(args[8].bit_generator.state)
        return real_update(*args)
    monkeypatch.setattr(tr, 'ActorPool', pool)
    monkeypatch.setattr(tr.Collector, 'collect', collect)
    monkeypatch.setattr(tr, 'ppo_update', update)
    first = tr.train(parse_args(args + ['--total_updates', '3']))
    ck = torch.load(tmp_path / 'ckpt/latest.pt', weights_only=False)
    assert first['env_steps'] == shared[0].value == 1536
    resumed = tr.train(parse_args(args + ['--total_updates', '5', '--resume', 'auto']))
    assert resumed['update'] == 5 and resumed['env_steps'] == shared[1].value == 2560
    assert [n for n, _ in starts] == [0, 512, 1024, 1536, 2048]
    assert torch.equal(starts[3][1], ck['torch_rng_state'])
    assert numpy_states[3] == ck['numpy_rng_state']
    with (tmp_path / 'progress.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r['update']) for r in rows] == [1, 2, 3, 4, 5]
    assert [int(r['env_steps']) for r in rows] == [512, 1024, 1536, 2048, 2560]
    with (tmp_path / 'episodes.csv').open() as f:
        episodes = list(csv.DictReader(f))
    assert episodes and all(None not in r for r in rows + episodes)
    for e in episodes:
        step = (1 - float(e['floor_c']) / .01) * 100000
        assert step == pytest.approx(round(step / 512) * 512, abs=1e-8)
        assert 0 <= step <= int(e['env_steps'])
    assert any(0 < float(e['floor_c']) < .01 for e in episodes)
