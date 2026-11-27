import csv
import io
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import tempfile

import numpy as np
import pytest

from rl.config import Config, REPO, parse_args
from rl.encode import TT, Caps, Encoder
from rl.engine import FakeEngine
from rl.ppo import Rollout, to_torch
from rl.train import train


BUILD_STATS = {'build_frac', 'adv_std_combat', 'adv_std_build', 'adv_std_build_std'}


def _random_buffer(seed, boundaries=True):
    rng = np.random.default_rng(seed)
    b = Rollout(Caps(max_tok=4, max_ent=1, max_cand=2, max_cent=1), 240, 6, 'cpu')
    b.n = 240
    envs = rng.choice(5, b.n, p=[.02, .08, .15, .25, .5])
    b.env_idx = [np.flatnonzero(envs == e).tolist() for e in range(6)]
    b.rew[:] = rng.normal(size=b.n)
    b.val[:] = rng.normal(size=b.n)
    b.done[:] = rng.random(b.n) < .1
    b.combat[:] = rng.random(b.n) < .7
    b.forced[:] = rng.random(b.n) < .05
    if boundaries:
        b.trunc[:] = rng.random(b.n) < .08
        b.cut[:] = rng.random(b.n) < .12
        b.bootstrap[:] = rng.normal(size=b.n)
    return b, rng.normal(size=6).astype(np.float32), rng


def _forward_gae(b, boot, gamma, lams):
    adv = np.zeros(b.n)
    for e, ix in enumerate(b.env_idx):
        for start, i in enumerate(ix):
            weight = 1.
            for t in range(start, len(ix)):
                j = ix[t]
                if b.trunc[j]:
                    break
                nv = b.bootstrap[j] if b.cut[j] else b.val[ix[t + 1]] if t + 1 < len(ix) else boot[e]
                delta = float(b.rew[j]) - float(b.val[j])
                if not b.done[j]:
                    delta += gamma * float(nv)
                adv[i] += weight * delta
                if b.done[j] or b.cut[j]:
                    break
                weight *= gamma * lams[j]
    return adv


@pytest.mark.parametrize('seed', range(10))
def test_constant_lambda_array_is_identical(seed):
    b, boot, _ = _random_buffer(seed)
    scalar = b.gae(boot, .999, .95)
    array = b.gae(boot, .999, np.full(b.n, .95))
    for actual, expected in zip(array, scalar):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('seed', range(20))
def test_transition_lambdas_match_forward_sum(seed):
    b, boot, rng = _random_buffer(seed)
    gamma = .999
    lams = rng.choice([0., .3, .95, 1.], b.n)
    adv, ret, mask = b.gae(boot, gamma, lams)
    expected = _forward_gae(b, boot, gamma, lams)
    np.testing.assert_allclose(adv, expected, rtol=0, atol=1e-5)
    np.testing.assert_allclose(ret, expected + b.val[:b.n], rtol=0, atol=1e-5)
    np.testing.assert_array_equal(mask, ~b.trunc[:b.n] & ~b.forced[:b.n])


def _macro_gae(b, boot, gamma, lam):
    out = np.zeros(b.n)
    for e, ix in enumerate(b.env_idx):
        ends = [t + 1 for t, i in enumerate(ix) if b.done[i]]
        if ix and (not ends or ends[-1] != len(ix)):
            ends.append(len(ix))
        start = 0
        for end in ends:
            builds = [t for t in range(start, end) if not b.combat[ix[t]]]
            rewards, discounts, values = [], [], []
            for j, t in enumerate(builds):
                nxt = builds[j + 1] if j + 1 < len(builds) else end
                rewards.append(sum(gamma**k * float(b.rew[ix[s]]) for k, s in enumerate(range(t, nxt))))
                discounts.append(gamma**(nxt - t))
                values.append(float(b.val[ix[t]]))
            nv = 0. if b.done[ix[end - 1]] else float(boot[e])
            na = 0.
            for j in reversed(range(len(builds))):
                delta = rewards[j] + discounts[j] * nv - values[j]
                na = delta + discounts[j] * lam * na
                out[ix[builds[j]]] = na
                nv = values[j]
            start = end
    return out


@pytest.mark.parametrize('seed', range(20))
def test_build_gae_matches_explicit_macro_steps(seed):
    b, boot, rng = _random_buffer(seed, boundaries=False)
    gamma, lam_build = .999, float(rng.choice([0., .5, .95, 1.]))
    expected = _macro_gae(b, boot, gamma, lam_build)
    standard = b.gae(boot, gamma, .95)
    adv, ret, mask, stats = b.gae_build(boot, gamma, .95, lam_build)
    build = ~b.combat[:b.n]
    np.testing.assert_allclose(adv[build], expected[build], rtol=0, atol=1e-5)
    np.testing.assert_allclose(ret[build], expected[build] + b.val[:b.n][build], rtol=0, atol=1e-5)
    np.testing.assert_array_equal(adv[~build], standard[0][~build])
    np.testing.assert_array_equal(ret[~build], standard[1][~build])
    np.testing.assert_array_equal(mask, standard[2])
    assert stats == dict(build_frac=float(build.mean()), adv_std_combat=float(adv[~build].std()),
                         adv_std_build=float(adv[build].std()), adv_std_build_std=float(standard[0][build].std()))


def test_build_merge_preserves_boundaries_and_mask():
    b, boot, _ = _random_buffer(71)
    lams = np.ones(b.n)
    for ix in b.env_idx:
        for i, nxt in zip(ix, ix[1:]):
            lams[i] = .9 if not b.combat[nxt] else 1.
    expected = _forward_gae(b, boot, .99, lams)
    standard = b.gae(boot, .99, .95)
    adv, ret, mask, _ = b.gae_build(boot, .99, .95, .9)
    build = ~b.combat[:b.n]
    np.testing.assert_allclose(adv[build], expected[build], rtol=0, atol=1e-5)
    np.testing.assert_allclose(ret[build], expected[build] + b.val[:b.n][build], rtol=0, atol=1e-5)
    np.testing.assert_array_equal(adv[~build], standard[0][~build])
    np.testing.assert_array_equal(ret[~build], standard[1][~build])
    np.testing.assert_array_equal(mask, standard[2])


@pytest.mark.parametrize('combat', [False, True])
def test_build_stats_with_one_decision_class(combat):
    b, boot, _ = _random_buffer(0)
    b.combat[:] = combat
    _, _, _, stats = b.gae_build(boot, .99, .95, .95)
    assert all(np.isfinite(v) for v in stats.values())
    assert stats['build_frac'] == float(not combat)
    assert stats['adv_std_build' if combat else 'adv_std_combat'] == 0.


@pytest.mark.parametrize('enc_rev', [0, 1])
def test_combat_token_matches_fake_engine_observation(enc_rev):
    engine, rng = FakeEngine(), random.Random(0)
    caps = Caps(max_tok=8, max_ent=4, max_cand=16, max_cent=4, enc_rev=enc_rev)
    enc = Encoder(caps)
    b = Rollout(caps, 1, 1, 'cpu', track_combat=True)
    seen = set()
    try:
        for ep in range(8):
            r = engine.reset(character=['Ironclad', 'Defect', 'Necrobinder'][ep % 3], seed=f'combat-{ep}')
            while True:
                expected = r['obs']['combat'] is not None
                obs = enc.encode(r)
                assert bool(np.any(obs['tok_type'] == TT['COMBAT'])) == expected
                b.reset()
                b.add([0], to_torch({k: v[None] for k, v in obs.items()}, 'cpu'), [0], [0.], [0.])
                assert b.combat[0] == expected
                seen.add(expected)
                if r['done']:
                    break
                r = engine.step(rng.randrange(len(r['cands'])))
        assert seen == {False, True}
        b.grow(2)
        assert b.combat[0] == expected and len(b.combat) == 2
    finally:
        engine.close()


def test_lam_build_config():
    assert Config().lam_build is None
    assert Config.from_dict({}).lam_build is None
    assert parse_args(['--lam_build', '0.95']).lam_build == .95
    for flags in (['--group_k', '2'], ['--branch_decisions', 'rest_site']):
        with pytest.raises(ValueError, match='lam_build cannot be combined'):
            parse_args(['--lam_build', '0.95', *flags])


@pytest.mark.parametrize('resume', [False, True])
def test_enabled_training_logs_build_stats(resume):
    os.makedirs(Path(REPO) / 'runs', exist_ok=True)
    with tempfile.TemporaryDirectory(dir=Path(REPO) / 'runs', prefix='mgae_test_') as run:
        cfg = parse_args(['--preset', 'tiny', '--run_dir', run, '--lam_build', '0.95',
                          '--n_workers', '1', '--envs_per_worker', '1', '--rollout_steps', '128',
                          '--total_updates', '2'])
        if resume:
            cfg.lam_build, cfg.total_updates = None, 1
            assert train(cfg)['update'] == 1
            cfg.lam_build, cfg.total_updates, cfg.resume = .95, 2, 'auto'
        assert train(cfg)['update'] == 2
        with open(Path(run) / 'progress.csv') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2 and BUILD_STATS <= rows[0].keys()
        for row in rows[1:] if resume else rows:
            assert 0 < float(row['build_frac']) < 1
            assert all(np.isfinite(float(row[k])) and float(row[k]) >= 0 for k in BUILD_STATS)


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
assert getattr(cfg, 'lam_build', None) is None
assert tr.train(cfg)['update'] == 3
'''


def test_disabled_training_matches_baseline_bytes():
    os.makedirs(Path(REPO) / 'runs', exist_ok=True)
    with tempfile.TemporaryDirectory(dir=Path(REPO) / 'runs', prefix='mgae_baseline_') as root:
        root = Path(root)
        baseline = root / 'baseline'
        archive = subprocess.run(['git', 'archive', '23daecc', 'rl'], cwd=REPO, check=True, capture_output=True)
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(baseline, filter='data')
        for source, run in [(baseline, root / 'before'), (Path(REPO), root / 'after')]:
            env = {**os.environ, 'PYTHONPATH': str(source), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            subprocess.run([sys.executable, '-c', BASELINE_TRAIN, str(run)], cwd=source, env=env,
                           check=True, capture_output=True, text=True, timeout=120)
        before = (root / 'before/progress.csv').read_bytes()
        after = (root / 'after/progress.csv').read_bytes()
        assert before == after
        assert BUILD_STATS.isdisjoint(next(csv.DictReader(io.StringIO(after.decode()))))
