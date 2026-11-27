import contextlib
import csv
import io
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from rl.actors import ActorPool, SharedObs, _Worker
from rl.config import Config, REPO, parse_args
from rl.evaluate import run_eval
from rl.tests.test_explore import BASELINE_TRAIN
from rl.train import AscCurriculum, Collector, train


def test_asc_config():
    cfg = Config.from_dict({})
    assert not cfg.asc_curriculum
    assert (cfg.asc_start, cfg.asc_max, cfg.asc_threshold, cfg.asc_window, cfg.asc_frontier_frac) == (1, 10, .6, 300, .5)
    cfg = parse_args(['--asc_curriculum', 'true', '--asc_start', '0', '--asc_max', '8',
                      '--asc_threshold', '.7', '--asc_window', '20', '--asc_frontier_frac', '.4',
                      '--explore_eps', '.2', '--lam_build', '.95'])
    assert cfg.asc_curriculum and (cfg.asc_start, cfg.asc_max, cfg.asc_threshold, cfg.asc_window,
                                   cfg.asc_frontier_frac, cfg.explore_eps, cfg.lam_build) == (0, 8, .7, 20, .4, .2, .95)
    for flags in (['--group_k', '2'], ['--branch_decisions', 'rest_site']):
        with pytest.raises(ValueError, match='asc_curriculum cannot be combined'):
            parse_args(['--asc_curriculum', 'true', *flags])
        assert not parse_args(['--asc_curriculum', 'false', *flags]).asc_curriculum


def test_frontier_uses_full_recent_non_aborted_window_and_stops_at_max():
    cfg = Config(asc_curriculum=True, asc_window=5, asc_threshold=.6, asc_max=3)
    c = AscCurriculum(cfg, 1)
    assert c.stats() == {'asc_frontier': 1}
    assert not c.record(dict(asc=1, victory=None, aborted='timeout'))
    assert c.stats()['w_win_a1'] == ''
    for win in (False, False, False, True):
        assert not c.record(dict(asc=1, victory=win))
    assert len(c.windows[1]) == 4
    assert not c.record(dict(asc=1, victory=True))
    assert c.stats()['w_win_a1'] == .4
    for _ in range(5):
        assert not c.record(dict(asc=0, victory=True))
    assert c.frontier == 1
    assert c.record(dict(asc=1, victory=True))
    assert c.frontier == 2 and c.stats()['w_win_a1'] == .6
    for _ in range(10):
        assert not c.record(dict(asc=1, victory=True))
    for _ in range(4):
        assert not c.record(dict(asc=2, victory=True))
    assert c.record(dict(asc=2, victory=True))
    for _ in range(10):
        assert not c.record(dict(asc=3, victory=True))
    assert c.frontier == 3 and len(c.windows[3]) == 5


@pytest.mark.parametrize('frontier,frac', [(0, .5), (1, 0.), (5, .35), (10, .5), (4, 1.)])
def test_actor_level_distribution_and_seed(tmp_path, frontier, frac):
    cfg = Config(asc_curriculum=True, asc_frontier_frac=frac, seed=17, n_workers=1, envs_per_worker=1,
                 engine='fake', run_dir=str(tmp_path))
    d = dict(cfg.to_dict(), n_envs=1)
    shm = SharedObs(cfg.caps, 1)
    def draw(wid):
        w = _Worker(wid, [0], None, shm.path, d, SimpleNamespace(value=0), None, None)
        w.asc_frontier = frontier
        try:
            return [w.episode_ascension() for _ in range(30000)]
        finally:
            w.shutdown()
            w.shm.close()
    try:
        levels = draw(0)
        random.seed(1234)
        assert draw(0) == levels
        counts = Counter(levels)
        if frontier == 0:
            assert counts == {0: len(levels)}
        else:
            assert counts[frontier] / len(levels) == pytest.approx(frac, abs=.015)
            for level in range(frontier):
                assert counts[level] / len(levels) == pytest.approx((1 - frac) / frontier, abs=.01)
            if 0 < frac < 1:
                assert draw(1) != levels
    finally:
        shm.close()


def test_frontier_message_reaches_all_actors_and_only_changes_new_episodes(tmp_path):
    cfg = parse_args(['--preset', 'tiny', '--run_dir', str(tmp_path), '--envs_per_worker', '1',
                      '--asc_curriculum', 'true', '--asc_frontier_frac', '1', '--max_episode_steps', '1'])
    pool = ActorPool(cfg, asc_frontier=3)
    def wait_for(kind):
        events, deadline = [], time.monotonic() + 20
        while len([e for e in events if e[0] == kind]) < cfg.n_envs and time.monotonic() < deadline:
            events.extend(pool.poll(.1))
        found = [e for e in events if e[0] == kind]
        assert len(found) == cfg.n_envs
        return found
    try:
        wait_for('obs')
        pool.set_asc_frontier(6)
        pool.act(range(cfg.n_envs), [0] * cfg.n_envs)
        assert {e[1]['asc'] for e in wait_for('ep')} == {3}
        pool.act(range(cfg.n_envs), [0] * cfg.n_envs)
        assert {e[1]['asc'] for e in wait_for('ep')} == {6}
    finally:
        pool.close()


def test_collector_broadcasts_on_promotion():
    sent = []
    cfg = Config(asc_curriculum=True, asc_window=2, asc_threshold=.5)
    pool = SimpleNamespace(asc_frontier=1, set_asc_frontier=sent.append)
    col = Collector(cfg, pool, None, None, 'cpu', contextlib.nullcontext, err_log=False)
    col.handle([('ep', dict(asc=1, victory=False)), ('ep', dict(asc=1, victory=True)),
                ('ep', dict(asc=1, victory=True))], None)
    assert sent == [2] and col.curriculum.frontier == 2


def test_training_logs_levels_and_restores_frontier_from_new_and_old_checkpoints(tmp_path, monkeypatch):
    import rl.train as tr
    def pool(cfg, **kwargs):
        return ActorPool(cfg, seed_list=[f'asc_{i}' for i in range(1000)], **kwargs)
    monkeypatch.setattr(tr, 'ActorPool', pool)
    args = ['--preset', 'tiny', '--n_workers', '1', '--envs_per_worker', '1', '--rollout_steps', '512',
            '--run_dir', str(tmp_path), '--asc_curriculum', 'true', '--asc_frontier_frac', '1',
            '--asc_window', '1', '--asc_threshold', '0', '--asc_max', '2',
            '--explore_eps', '.2', '--lam_build', '.95']
    assert train(parse_args(args + ['--total_updates', '1']))['asc_frontier'] == 2
    ckpt = tmp_path / 'ckpt/latest.pt'
    assert torch.load(ckpt, weights_only=False)['asc_frontier'] == 2
    resume = args + ['--asc_window', '1000', '--resume', 'auto']
    assert train(parse_args(resume + ['--total_updates', '2']))['asc_frontier'] == 2
    ck = torch.load(ckpt, weights_only=False)
    ck.pop('asc_frontier')
    torch.save(ck, ckpt)
    assert train(parse_args(resume + ['--total_updates', '3', '--asc_start', '3', '--asc_max', '3']))['asc_frontier'] == 3
    with (tmp_path / 'episodes.csv').open() as f:
        eps = list(csv.DictReader(f))
    assert {int(e['asc']) for e in eps if e['update'] == '2'} == {2}
    assert {int(e['asc']) for e in eps if e['update'] == '3'} == {3}
    with (tmp_path / 'progress.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r['asc_frontier']) for r in rows] == [2, 2, 3]
    assert {'w_win_a1', 'w_win_a2', 'w_win_a3', 'explore_ep_frac', 'build_frac'} <= rows[0].keys()
    assert rows[0]['w_win_a3'] == ''
    for row in rows:
        for level in range(1, 4):
            done = [e for e in eps if e['update'] == row['update'] and e['asc'] == str(level) and not e['aborted']]
            window = done[-(1 if row['update'] == '1' else 1000):]
            expected = sum(e['victory'] == 'True' for e in window) / len(window) if window else ''
            actual = row[f'w_win_a{level}']
            assert (float(actual) if actual else '') == expected


def test_evaluation_disables_curriculum(monkeypatch):
    import rl.evaluate as ev
    cfg = Config(asc_curriculum=True, ascension=10)
    class Pool:
        def __init__(self, cfg, seed_list):
            assert not cfg.asc_curriculum and cfg.ascension == 10
        def close(self):
            pass
    monkeypatch.setattr(ev, 'ActorPool', Pool)
    assert run_eval(cfg, None, [], progress=False) == []
    assert cfg.asc_curriculum


def test_disabled_training_matches_1567681_bytes():
    os.makedirs(Path(REPO) / 'runs', exist_ok=True)
    with tempfile.TemporaryDirectory(dir=Path(REPO) / 'runs', prefix='asc_baseline_') as root:
        root = Path(root)
        baseline = root / 'baseline'
        archive = subprocess.run(['git', 'archive', '1567681', 'rl'], cwd=REPO, check=True, capture_output=True)
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(baseline, filter='data')
        for source, run in [(baseline, root / 'before'), (Path(REPO), root / 'after')]:
            env = {**os.environ, 'PYTHONPATH': str(source), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            script = BASELINE_TRAIN + '\nassert not getattr(cfg, "asc_curriculum", False)\n'
            subprocess.run([sys.executable, '-c', script, str(run)], cwd=source, env=env,
                           check=True, capture_output=True, text=True, timeout=120)
        for file in ('progress.csv', 'episodes.csv'):
            assert (root / 'before' / file).read_bytes() == (root / 'after' / file).read_bytes()
        with (root / 'after/progress.csv').open() as f:
            fields = next(csv.reader(f))
        assert 'asc_frontier' not in fields and not any(k.startswith('w_win_a') for k in fields)
        assert 'asc_frontier' not in torch.load(root / 'after/ckpt/latest.pt', weights_only=False)
