import contextlib
import csv
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict

import numpy as np
import pytest
import torch

from rl.actors import ActorPool
from rl.config import Config, REPO, parse_args
from rl.encode import TT, Caps, Encoder, candidate_type
from rl.evaluate import run_eval
from rl.model import ModelCfg, Policy
from rl.ppo import Rollout, ValueNorm, sliced, to_torch
from rl.tests.test_core import RS
from rl.train import Collector, episode_stats, train


EXPLORE_STATS = {'explore_ep_frac', 'explore_replaced', 'explore_build_decisions',
                 'w_win_rate_main', 'w_floor_mean_main'}


class SkewPolicy(torch.nn.Module):
    def forward(self, obs):
        n = obs['n_cand'].long()
        c = torch.arange(obs['cand_type'].shape[1], device=n.device)
        logits = (-2. * c).expand(len(n), -1).masked_fill(c[None] >= n[:, None], -torch.inf)
        return logits, torch.full((len(n),), .25, device=n.device)


def _collector(frac=1., seed=17):
    cfg = Config(explore_eps=1., explore_frac=frac, seed=seed)
    return Collector(cfg, None, None, None, 'cpu', contextlib.nullcontext, err_log=False)


def test_explore_config():
    assert Config().explore_eps == Config.from_dict({}).explore_eps == 0.
    assert Config().explore_frac == .5
    cfg = parse_args(['--explore_eps', '.05', '--explore_frac', '.7', '--lam_build', '.95'])
    assert (cfg.explore_eps, cfg.explore_frac, cfg.lam_build) == (.05, .7, .95)
    for flags in (['--group_k', '2'], ['--branch_decisions', 'rest_site']):
        with pytest.raises(ValueError, match='explore_eps cannot be combined'):
            parse_args(['--explore_eps', '.05', *flags])
        assert parse_args(['--explore_eps', '0', *flags]).explore_eps == 0.


def test_candidate_types_are_stable_and_cover_unknown_strings():
    kinds = sorted({c['k'] for r in RS for c in r['cands']})
    ids = [candidate_type(k) for k in kinds]
    assert len(ids) == len(set(ids)) and all(0 < i <= 32767 for i in ids)
    code = 'from rl.encode import candidate_type; import json, sys; print([candidate_type(k) for k in json.loads(sys.argv[1])])'
    other = subprocess.run([sys.executable, '-c', code, json.dumps(kinds)], cwd=REPO,
                           env={**os.environ, 'PYTHONHASHSEED': '123'}, check=True, capture_output=True, text=True)
    assert json.loads(other.stdout) == ids
    enc = Encoder()
    for r in RS:
        obs = enc.encode(r)
        n = len(r['cands'])
        np.testing.assert_array_equal(obs['cand_type'][:n], [candidate_type(c['k']) for c in r['cands']])
        assert not obs['cand_type'][n:].any()
    r = dict(RS[0], cands=[dict(k='new_candidate_kind'), dict(k='new_candidate_kind')])
    obs = enc.encode(r)
    assert obs['cand_type'][0] == obs['cand_type'][1] == candidate_type('new_candidate_kind')


def test_candidate_types_do_not_affect_model_outputs():
    enc = Encoder(Caps(max_tok=64, max_ent=12, max_cand=64))
    batch = enc.alloc(4)
    for i, r in enumerate(RS[:4]):
        enc.encode(r, {k: v[i, ...] for k, v in batch.items()})
    obs = sliced(to_torch(batch, 'cpu'), np.arange(4))
    model = Policy(ModelCfg(d=32, layers=1, heads=4, cand_layers=1)).eval()
    with torch.no_grad():
        expected = model({k: v for k, v in obs.items() if k != 'cand_type'})
        actual = model(obs)
        obs['cand_type'].fill_(32767)
        changed = model(obs)
    for a, b, c in zip(expected, actual, changed):
        assert torch.equal(a, b) and torch.equal(a, c)


def test_fake_shop_sampling_is_uniform_over_types_and_within_type():
    r = max((r for r in RS if r['decision'] == 'shop'), key=lambda r: len({c['k'] for c in r['cands']}))
    obs = Encoder().encode(r)
    n = 4000
    batch = {k: np.repeat(obs[k][None], n, axis=0) for k in ('tok_type', 'cand_type', 'n_cand')}
    col = _collector()
    col.handle([('ep_start', e) for e in range(n)], None)
    col.explore_replaced = col.explore_build_decisions = 0
    actions = torch.zeros(n, dtype=torch.long)
    logits = -2. * torch.arange(len(r['cands'])).expand(n, -1)
    original = logits.clone()
    logp = torch.zeros(n)
    col.explore_actions(range(n), batch, actions, logp, logits)
    assert col.explore_replaced == col.explore_build_decisions == n
    assert torch.equal(logits, original)
    assert torch.equal(logp, logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(1))
    counts = Counter(r['cands'][a]['k'] for a in actions.tolist())
    assert len(counts) >= 5
    for kind, count in counts.items():
        assert .8 * n / len(counts) < count < 1.2 * n / len(counts)
        members = [i for i, c in enumerate(r['cands']) if c['k'] == kind]
        for i in members:
            assert .65 * count / len(members) < int((actions == i).sum()) < 1.35 * count / len(members)


@pytest.mark.parametrize('frac', [0., .5, 1.])
def test_fake_collection_replacement_logp_and_episode_flags(tmp_path, frac):
    cfg = parse_args(['--preset', 'tiny', '--n_workers', '1', '--envs_per_worker', '1',
                      '--rollout_steps', '256', '--explore_eps', '1', '--explore_frac', str(frac),
                      '--lam_build', '.95', '--seed', '17', '--run_dir', str(tmp_path)])
    cfg.caps = Caps(max_tok=96, max_ent=12, max_cand=128)
    pool = ActorPool(cfg, seed_list=[f'explore_{i}' for i in range(2000)])
    col = Collector(cfg, pool, SkewPolicy(), ValueNorm(), 'cpu', contextlib.nullcontext, err_log=False)
    buf = Rollout(cfg.caps, cfg.rollout_steps, 1, 'cpu', track_combat=True)
    handle, explore_actions = col.handle, col.explore_actions
    starts, flags, seen = Counter(), {}, defaultdict(set)
    records = []

    def handle_events(events, b):
        for ev in events:
            if ev[0] == 'ep_start':
                starts[ev[1]] += 1
            handle([ev], b)
            if ev[0] == 'ep_start':
                flags[ev[1], starts[ev[1]]] = col.explore[ev[1]]
            elif ev[0] == 'ep':
                e = ev[1]['env']
                assert ev[1]['explore'] == flags[e, starts[e]]

    def replace(envs, obs, actions, logp, logits):
        before, before_logp = actions.clone(), logp.clone()
        original = logits.clone()
        combat = np.any(obs['tok_type'] == TT['COMBAT'], axis=1)
        eligible = np.array([not combat[j] and col.explore[e] for j, e in enumerate(envs)])
        n_before = col.explore_replaced
        for e in envs:
            key = (e, starts[e])
            seen[key].add(col.explore[e])
            assert col.explore[e] == flags[key]
        explore_actions(envs, obs, actions, logp, logits)
        assert col.explore_replaced - n_before == int(eligible.sum())
        assert torch.equal(logits, original)
        assert torch.equal(actions[~eligible], before[~eligible])
        assert torch.equal(logp[~eligible], before_logp[~eligible])
        expected = original.log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
        records.extend(zip(actions.tolist(), expected.tolist(), eligible.tolist()))

    col.handle, col.explore_actions = handle_events, replace
    build, combat, replaced = 0, 0, 0
    try:
        for _ in range(4):
            records.clear()
            boot, stats = col.collect(buf)
            actions, logps, eligible = map(np.array, zip(*records))
            np.testing.assert_array_equal(buf.act[:buf.n], actions)
            np.testing.assert_array_equal(buf.logp[:buf.n], logps.astype(np.float32))
            assert np.all(buf.act[:buf.n] < buf.obs['n_cand'][:buf.n].numpy())
            assert not np.any(eligible & buf.combat[:buf.n])
            assert stats['explore_replaced'] == stats['explore_build_decisions'] == int(eligible.sum())
            if frac == 1.:
                np.testing.assert_array_equal(eligible, ~buf.combat[:buf.n])
            build += int((~buf.combat[:buf.n]).sum())
            combat += int(buf.combat[:buf.n].sum())
            replaced += stats['explore_replaced']
            assert buf.gae_build(boot, cfg.gamma, cfg.lam, cfg.lam_build)[2].all()
        assert not col.errors and build > 0 and combat > 0
        assert all(len(v) == 1 for v in seen.values())
        if frac == 0.:
            assert replaced == 0
        else:
            assert replaced > 0
    finally:
        pool.close()


def test_explore_fraction_uses_collector_seed():
    def flags():
        col = _collector(frac=.5)
        out = []
        for _ in range(1000):
            col.handle([('ep_start', 0)], None)
            flag = col.explore[0]
            for _ in range(3):
                col.handle([('obs', 0), ('res', 0, 0., False, False, True)], None)
                assert col.explore[0] == flag
            col.handle([('ep', dict(env=0))], None)
            assert col.episodes[-1]['explore'] == int(flag)
            out.append(flag)
        return out
    first = flags()
    random.seed(1234)
    np.random.seed(5678)
    torch.manual_seed(9012)
    assert flags() == first
    assert .45 < np.mean(first) < .55


@pytest.mark.parametrize('resume', [False, True])
def test_training_logs_exploration_and_main_window(tmp_path, resume):
    cfg = parse_args(['--preset', 'tiny', '--n_workers', '1', '--envs_per_worker', '1',
                      '--rollout_steps', '256', '--total_updates', '3', '--explore_eps', '.2',
                      '--lam_build', '.95', '--run_dir', str(tmp_path)])
    if resume:
        cfg.explore_eps, cfg.total_updates = 0., 1
        assert train(cfg)['update'] == 1
        cfg.explore_eps, cfg.total_updates, cfg.resume = .2, 3, 'auto'
    assert train(cfg)['update'] == 3
    with (tmp_path / 'progress.csv').open() as f:
        rows = list(csv.DictReader(f))
    with (tmp_path / 'episodes.csv').open() as f:
        eps = list(csv.DictReader(f))
    assert EXPLORE_STATS <= rows[0].keys() and 'explore' in eps[0]
    window = []
    for row in rows[int(resume):]:
        current = [e for e in eps if e['update'] == row['update']]
        assert all(e['explore'] in ('0', '1') for e in current)
        window = (window + current)[-200:]
        done = [e for e in current if not e['aborted']]
        expected_frac = np.mean([int(e['explore']) for e in done]) if done else ''
        if done:
            assert float(row['explore_ep_frac']) == expected_frac
        else:
            assert row['explore_ep_frac'] == ''
        main = [dict(e, victory=e['victory'] == 'True', floor=int(e['floor']), ret=float(e['ret']),
                     steps=int(e['steps'])) for e in window if e['explore'] == '0']
        stats = episode_stats(main)
        for key in ('win_rate', 'floor_mean'):
            expected = stats[key]
            assert (float(row[f'w_{key}_main']) if expected != '' else row[f'w_{key}_main']) == expected
        assert 0 <= int(row['explore_replaced']) <= int(row['explore_build_decisions']) <= 256


def test_evaluation_disables_exploration(monkeypatch):
    import rl.evaluate as ev
    cfg = Config(explore_eps=1., explore_frac=1., n_workers=1, envs_per_worker=1)
    class Pool:
        def __init__(self, cfg, seed_list):
            assert cfg.explore_eps == 0.
        def close(self):
            pass
    monkeypatch.setattr(ev, 'ActorPool', Pool)
    assert run_eval(cfg, None, [], progress=False) == []
    assert cfg.explore_eps == 1.


BASELINE_TRAIN = '''
import itertools
import random
import sys
from types import SimpleNamespace
import torch
import rl.ppo as ppo
import rl.train as tr
from rl.config import parse_args

torch.set_num_threads(1)
random.seed(17)
ticks = itertools.count()
tr.time = ppo.time = SimpleNamespace(time=lambda: next(ticks) * .001)
cfg = parse_args(['--preset', 'tiny', '--seed', '17', '--run_dir', sys.argv[1],
                  '--n_workers', '1', '--envs_per_worker', '1', '--rollout_steps', '128',
                  '--total_updates', '3'])
assert getattr(cfg, 'explore_eps', 0.) == 0.
assert tr.train(cfg)['update'] == 3
'''


def test_disabled_training_matches_7c798dd_bytes():
    os.makedirs(Path(REPO) / 'runs', exist_ok=True)
    with tempfile.TemporaryDirectory(dir=Path(REPO) / 'runs', prefix='explore_baseline_') as root:
        root = Path(root)
        baseline = root / 'baseline'
        archive = subprocess.run(['git', 'archive', '7c798dd', 'rl'], cwd=REPO, check=True, capture_output=True)
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(baseline, filter='data')
        for source, run in [(baseline, root / 'before'), (Path(REPO), root / 'after')]:
            env = {**os.environ, 'PYTHONPATH': str(source), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            subprocess.run([sys.executable, '-c', BASELINE_TRAIN, str(run)], cwd=source, env=env,
                           check=True, capture_output=True, text=True, timeout=120)
        before = (root / 'before/progress.csv').read_bytes()
        after = (root / 'after/progress.csv').read_bytes()
        assert before == after
        assert EXPLORE_STATS.isdisjoint(next(csv.DictReader(io.StringIO(after.decode()))))
        with (root / 'after/episodes.csv').open() as f:
            assert 'explore' not in next(csv.reader(f))
