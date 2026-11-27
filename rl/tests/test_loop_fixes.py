import contextlib
import csv
import json
import math
import multiprocessing as mp
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl.actors import _Worker
from rl.config import Config
from rl.encode import Caps
from rl.evaluate import summarize
from rl.ppo import Rollout, ValueNorm, masked_logp_entropy, ppo_update
from rl.train import CSVLog
import rl.eval_watch as watch


def test_workers_share_global_reward_schedule():
    steps = mp.get_context('spawn').Value('q', 1000)
    workers = [_Worker.__new__(_Worker) for _ in range(2)]
    for w, local in zip(workers, [900, 100]):
        w.n_trans, w.env_steps = local, steps
        w.cfg = dict(floor_reward=.01, floor_anneal_steps=2000, n_workers=2, resume_steps=500,
                     shaping='switch', shaping_switch_steps=1500)
        assert w.episode_floor_reward() == pytest.approx(.005)
        assert w.episode_shaping() == 'per_floor'
    steps.value = 1500
    for w in workers:
        assert w.episode_floor_reward() == pytest.approx(.0025)
        assert w.episode_shaping() == 'potential'
    steps.value = 2**33
    assert workers[0].episode_floor_reward() == 0


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, b):
        x = b['decision'].float() * .2 + self.p
        return torch.stack((x, self.p.expand_as(x)), -1), self.p.expand_as(x)


def test_ppo_metrics_and_epoch_kl_are_sample_weighted():
    b = Rollout(Caps(max_tok=8, max_ent=1, max_cand=2, max_cent=1), 6, 1, torch.device('cpu'))
    b.n = 6
    b.obs['n_tok'][:] = torch.arange(1, 7)
    b.obs['n_cand'][:] = 2
    b.obs['decision'][:] = torch.arange(6)
    m = Toy()
    logits, v = m(b.obs)
    logp, ent = masked_logp_entropy(logits, torch.zeros(6, dtype=torch.int64), torch.ones(6, 2, dtype=torch.bool))
    b.logp[:] = logp.detach().numpy()
    b.logp[4] -= np.log(2)
    ret = np.array([0, 0, 0, 0, 10, 0], np.float32)
    adv = np.arange(6, dtype=np.float32)
    a = torch.tensor((adv[:5] - adv[:5].mean()) / (adv[:5].std() + 1e-8))
    ratio = (logp[:5].detach() - torch.tensor(b.logp[:5])).exp()
    expected = dict(pg_loss=-torch.min(ratio * a, ratio.clamp(.8, 1.2) * a).mean().item(),
                    v_loss=10., entropy=ent[:5].mean().item(), approx_kl=(1 - np.log(2)) / 5, clipfrac=.2)
    cfg = Config(epochs=3, minibatch=4, micro_batch=3, value_norm=False, target_kl=.1)
    out = ppo_update(m, torch.optim.SGD(m.parameters(), lr=0), b, adv, ret,
                     np.array([1, 1, 1, 1, 1, 0], bool), ValueNorm(), cfg,
                     np.random.default_rng(0), contextlib.nullcontext)
    for key, value in expected.items():
        assert out[key] == pytest.approx(value, abs=1e-6)
    assert out['epochs_done'] == 3 and out['kl_stop'] == 0


def test_csv_resume_uses_existing_header(tmp_path, capsys):
    p = tmp_path / 'episodes.csv'
    p.write_text('update,cstep,tstep,round,victory,aborted\n1,0,0,,False,\n')
    log = CSVLog(p)
    log.write(dict(aborted='reset_timeout', update=2, extra=42))
    log.write(dict(update=3, tstep=4, extra=43))
    log.close()
    with p.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[1]['aborted'] == 'reset_timeout' and rows[1]['tstep'] == ''
    assert rows[2]['tstep'] == '4' and rows[2]['aborted'] == ''
    assert list(rows[1]) == ['update', 'cstep', 'tstep', 'round', 'victory', 'aborted']
    assert capsys.readouterr().out.count('dropping fields extra') == 1


@pytest.mark.parametrize('limit,reason', [('max_episode_steps', 'episode_step_limit'),
                                         ('max_combat_steps', 'combat_step_limit'),
                                         ('max_turn_steps', 'turn_step_limit')])
def test_step_limit_emits_abort_with_response_info(limit, reason):
    w = _Worker.__new__(_Worker)
    w.cfg = Config(**{limit: 1}).to_dict()
    w.state, w.ep = {0: w.ACT}, {0: dict(env=0, steps=0, ret=0., floor=1, act=1)}
    w.events = []
    next_runs = []
    w.next_run = next_runs.append
    r = dict(type='rl', done=False, cands=[dict(k='option')], info=dict(total_floor=2, act=1),
             obs=dict(combat=dict(round=1)))
    w.handle(0, r)
    assert w.events[0][0] == 'err' and reason in w.events[0][2]
    assert str(r['info']) in w.events[0][2]
    assert w.events[1][0] == 'res' and w.events[1][4] is True
    assert w.events[2][1]['aborted'] == reason and next_runs == [0]


@pytest.mark.parametrize('eps', [[], [dict(aborted='timeout', victory=None, floor=0)]])
def test_empty_evaluation_has_complete_schema(eps):
    s = summarize(eps)
    assert s['n'] == 0 and s['aborted'] == len(eps)
    for key in ('floor_mean', 'floor_median', 'floor_q25', 'floor_q75', 'steps_mean'):
        assert math.isnan(s[key])
    assert s['floor_hist'] == {}


@pytest.mark.parametrize('failures', [1, 3])
def test_watch_retries_failure_and_records_all_aborted(tmp_path, monkeypatch, capsys, failures):
    arch = tmp_path / 'eval_ckpts'
    arch.mkdir()
    for u in (100, 200):
        (arch / f'ckpt_{u:06d}.pt').touch()
    calls = []
    now = [0.]
    monkeypatch.setattr(watch.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(watch.time, 'sleep', lambda n: now.__setitem__(0, now[0] + n))
    monkeypatch.setattr(watch, 'training_alive', lambda name: False)
    def run(cmd, **kwargs):
        u = int(cmd[cmd.index('--ckpt') + 1].rsplit('_', 1)[1][:-3])
        calls.append(u)
        if u == 100 and calls.count(100) <= failures:
            return SimpleNamespace(returncode=1, stdout='', stderr='failed')
        out = cmd[cmd.index('--out') + 1]
        from pathlib import Path
        p = Path(out)
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(dict(summary=dict(all=summarize([dict(aborted='timeout')])))))
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(watch.subprocess, 'run', run)
    watch.main(['--run_dir', str(tmp_path), '--engine_dir', 'unused', '--poll', '1'])
    assert calls[:2] == [100, 200]
    assert calls.count(100) == min(failures + 1, 3)
    with (tmp_path / 'eval.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert {int(r['update']) for r in rows} == ({100, 200} if failures == 1 else {200})
    assert rows[0]['n'] == '0' and rows[0]['aborted'] == '1' and rows[0]['floor_mean'] == 'nan'
    assert ('after 3 failures' in capsys.readouterr().out) == (failures == 3)


def test_training_alive_has_name_boundary(monkeypatch):
    def run(cmd, **kwargs):
        pattern = cmd[-1].replace('[[:space:]]', r'\s')
        return SimpleNamespace(returncode=0 if re.search(pattern, 'python -m rl.train --name run_other') else 1)
    monkeypatch.setattr(watch.subprocess, 'run', run)
    assert not watch.training_alive('run')
    assert watch.training_alive('run_other')
