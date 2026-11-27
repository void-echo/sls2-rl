import contextlib
import copy
import csv
import itertools
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl.actors import ActorPool, SharedObs, _Worker
from rl.config import Config, parse_args
from rl.encode import Caps, Encoder, candidate_type, field_spec
from rl.evaluate import run_eval
from rl.fake_engine import CARD_POOL
from rl.force import force_choice, force_matches, parse_force_list
from rl.ppo import Rollout, ValueNorm, ppo_update, to_torch
from rl.protocol import decision_id
from rl.tests.test_core import RS
from rl.train import AscCurriculum, Collector, train


FORCE_STATS = {'force_ep_frac', 'force_win', 'main_win', 'force_replaced'}
FORCE_VARS = 'Damage,Block,StrengthPower'


def tiny_config(path, **kwargs):
    cfg = parse_args(['--preset', 'tiny', '--n_workers', '1', '--envs_per_worker', '1',
                      '--rollout_steps', '256', '--seed', '17', '--run_dir', str(path)])
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    cfg.caps = Caps(max_tok=96, max_ent=12, max_cand=128)
    return cfg


def test_force_config():
    cfg = Config.from_dict({})
    assert (cfg.force_ids, cfg.force_vars, cfg.force_frac, cfg.force_prob) == ('', '', 0., 1.)
    assert not cfg.force_enabled
    cfg = parse_args(['--force_ids', 'CARD.FORGOTTEN_RITUAL', '--force_vars', 'Damage,Block',
                      '--force_frac', '.5', '--force_prob', '.7', '--explore_eps', '.2'])
    assert cfg.force_enabled and cfg.explore_eps == .2
    assert (cfg.force_ids, cfg.force_vars, cfg.force_frac, cfg.force_prob) == (
        'CARD.FORGOTTEN_RITUAL', 'Damage,Block', .5, .7)
    assert Config.from_dict(cfg.to_dict()) == cfg
    assert not Config(force_ids='CARD.X').force_enabled
    assert not Config(force_frac=1.).force_enabled
    assert not Config(force_frac=1., force_ids=' , ', force_vars=' ').force_enabled


@pytest.mark.parametrize('flags', [['--async_update'], ['--group_k', '2'], ['--branch_decisions', 'rest_site']])
def test_force_rejects_incompatible_modes_only_when_enabled(flags):
    with pytest.raises(ValueError, match='force cannot be combined'):
        parse_args(['--force_vars', 'Damage', '--force_frac', '1', *flags])
    assert not parse_args(['--force_ids', 'CARD.X', *flags]).force_enabled
    assert not parse_args(['--force_frac', '1', *flags]).force_enabled


@pytest.mark.parametrize('decision', ['card_reward', 'shop', 'card_select', 'combat_play'])
def test_shared_matching_and_highest_probability_first_tie(decision):
    response = dict(decision=decision, cands=[
        dict(k='reward_card', card=dict(id='CARD.X')),
        dict(k='buy_card', card=dict(id='CARD.Y', v={'Damage': 0})),
        dict(k='buy_card', card=dict(id='CARD.Z', pv={'Damage': 20})),
        dict(k='pick', card=dict(id='CARD.X', v={'Damage': 3})),
        dict(k='buy_relic', id='CARD.X'), dict(k='skip'),
    ])
    matches = force_matches(response, parse_force_list(' CARD.X, ,'), parse_force_list(' Damage '))
    assert matches == ([True, True, False, False, False, False]
                       if decision in ('card_reward', 'shop') else [False] * 6)
    assert force_choice(matches, [.1, .3, .5, .1, 0, 0]) == (1 if any(matches) else None)
    assert force_choice(matches, [.2] * 6) == (0 if any(matches) else None)


@pytest.mark.parametrize('enabled', [False, True])
def test_worker_encodes_optional_matches_and_clears_stale_rows(tmp_path, monkeypatch, enabled):
    cfg = tiny_config(tmp_path, force_frac=float(enabled), force_vars=FORCE_VARS)
    shm = SharedObs(cfg.caps, 1, force=cfg.force_enabled)
    worker = _Worker(0, [0], None, shm.path, dict(cfg.to_dict(), n_envs=1),
                     SimpleNamespace(value=0), None, None)
    if not enabled:
        def unexpected_match(*args):
            pytest.fail('disabled force must not match candidates')
        monkeypatch.setattr('rl.actors.force_matches', unexpected_match)
    try:
        responses = [max((r for r in RS if r['decision'] == kind), key=lambda r: len(r['cands']))
                     for kind in ('shop', 'card_reward', 'combat_play')]
        for response in responses:
            worker.encode(0, response)
            expected = Encoder(cfg.caps).encode(response)
            for key, value in expected.items():
                np.testing.assert_array_equal(shm.arrays[key][0], value)
            assert set(shm.arrays) == set(field_spec(cfg.caps)) | ({'cand_force'} if enabled else set())
            if enabled:
                matches = force_matches(response, set(), parse_force_list(FORCE_VARS))
                actual = shm.gather([0])['cand_force'][0]
                np.testing.assert_array_equal(actual[:len(matches)], matches)
                assert not actual[len(matches):].any()
                assert not shm.arrays['cand_force'][0, len(matches):].any()
    finally:
        worker.shutdown()
        worker.shm.close()
        shm.close()


class AvoidCardsPolicy(torch.nn.Module):
    def forward(self, obs):
        n = obs['n_cand'].long()
        c = torch.arange(obs['cand_type'].shape[1], device=n.device)
        logits = (-2. * c).expand(len(n), -1).clone()
        cards = (obs['cand_type'] == candidate_type('reward_card')) | (obs['cand_type'] == candidate_type('buy_card'))
        logits = (logits - 100 * cards).masked_fill(c[None] >= n[:, None], -torch.inf)
        return logits, torch.full((len(n),), .25, device=n.device)


@pytest.mark.parametrize('selector,explore', [('ids', 0.), ('vars', 0.), ('vars', 1.)])
def test_fake_collection_forces_rewards_and_shops_with_policy_logp(tmp_path, selector, explore):
    cfg = tiny_config(tmp_path, force_frac=1., explore_eps=explore, explore_frac=1.)
    if selector == 'ids':
        cfg.force_ids = ','.join(c['id'] for c in CARD_POOL)
    else:
        cfg.force_vars = FORCE_VARS
    pool = ActorPool(cfg, seed_list=[f'force_{i}' for i in range(2000)])
    col = Collector(cfg, pool, AvoidCardsPolicy(), ValueNorm(), 'cpu', contextlib.nullcontext, err_log=False)
    buf = Rollout(cfg.caps, cfg.rollout_steps, 1, torch.device('cpu'))
    original_force, original_handle = col.force_actions, col.handle
    records, decisions, counts = [], set(), {0: 0}

    def handle(events, b):
        for ev in events:
            original_handle([ev], b)
            if ev[0] == 'ep_start':
                counts[ev[1]] = 0
                assert col.force[ev[1]]
            elif ev[0] == 'ep':
                assert ev[1]['force'] == 1 and ev[1]['n_forced'] == counts[ev[1]['env']]

    def force(envs, matches, actions, logp, logits):
        before, old_logp = actions.clone(), logp.clone()
        old_logits = logits.clone()
        replaced, locked = original_force(envs, matches, actions, logp, logits)
        expected = before.clone()
        for j in range(len(envs)):
            best = force_choice(matches[j], logits[j].softmax(-1))
            if best is not None:
                expected[j] = best
                decisions.add(int(pool.shm.arrays['decision'][envs[j]]))
        assert locked == np.flatnonzero(matches.any(axis=1)).tolist()
        assert replaced == torch.nonzero(expected != before).flatten().tolist()
        assert torch.equal(logits, old_logits) and torch.equal(actions, expected)
        torch.testing.assert_close(logp, logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(1), rtol=0, atol=0)
        for j, e in enumerate(envs):
            counts[e] += int(j in replaced)
            records.append((int(actions[j]), float(logp[j]), j in locked, j in replaced))
            if j not in locked:
                assert logp[j] == old_logp[j]
        return replaced, locked

    col.force_actions, col.handle = force, handle
    torch.manual_seed(17)
    try:
        for _ in range(4):
            records.clear()
            _, stats = col.collect(buf)
            actions, logps, locked, replaced = map(np.array, zip(*records))
            np.testing.assert_array_equal(buf.act[:buf.n][locked], actions[locked])
            np.testing.assert_array_equal(buf.logp[:buf.n][locked], logps[locked].astype(np.float32))
            np.testing.assert_array_equal(buf.explored[:buf.n][locked], replaced[locked])
            if not explore:
                np.testing.assert_array_equal(buf.act[:buf.n], actions)
                np.testing.assert_array_equal(buf.explored[:buf.n], replaced)
            assert stats['force_replaced'] == replaced.sum()
        assert not col.errors
        assert decisions == {decision_id('card_reward'), decision_id('shop')}
        assert sum(e['n_forced'] for e in col.episodes) > 0
    finally:
        pool.close()


@pytest.mark.parametrize('prob', [0., .3, 1.])
def test_force_probability_and_already_sampled_action(prob):
    cfg = Config(force_vars='Damage', force_frac=1., force_prob=prob, seed=17)
    col = Collector(cfg, None, None, None, 'cpu', contextlib.nullcontext, err_log=False)
    n = 3000
    col.handle([('ep_start', e) for e in range(n)], None)
    col.force_replaced = 0
    actions = torch.arange(n) % 3
    before = actions.clone()
    logits = torch.tensor([0., 2., 2.]).expand(n, -1)
    logp = logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
    replaced, locked = col.force_actions(range(n), np.tile([False, True, True], (n, 1)), actions, logp, logits)
    assert set(replaced) == set(locked) - set(torch.nonzero(before == 1).flatten().tolist())
    assert (actions[locked] == 1).all()
    assert col.force_replaced == sum(col.n_forced.values()) == len(replaced)
    torch.testing.assert_close(logp, logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(1), rtol=0, atol=0)
    assert len(locked) / n == pytest.approx(prob, abs=.03)


def test_disabled_training_has_identical_csv_bytes_and_no_new_columns(tmp_path, monkeypatch):
    import rl.ppo as ppo
    import rl.train as tr
    outputs = []
    for i, settings in enumerate(({}, dict(force_ids='CARD.X'), dict(force_frac=1.))):
        random.seed(17)
        ticks = itertools.count()
        clock = SimpleNamespace(time=lambda: next(ticks) * .001, strftime=lambda fmt: '00:00:00')
        monkeypatch.setattr(tr, 'time', clock)
        monkeypatch.setattr(ppo, 'time', clock)
        cfg = tiny_config(tmp_path / str(i), total_updates=1, max_episode_steps=2, **settings)
        assert not cfg.force_enabled and train(cfg)['update'] == 1
        outputs.append(tuple((tmp_path / str(i) / name).read_bytes() for name in ('progress.csv', 'episodes.csv')))
        with (tmp_path / str(i) / 'progress.csv').open() as f:
            assert FORCE_STATS.isdisjoint(next(csv.DictReader(f)))
        with (tmp_path / str(i) / 'episodes.csv').open() as f:
            assert {'force', 'n_forced'}.isdisjoint(next(csv.DictReader(f)))
    assert outputs[0] == outputs[1] == outputs[2]


@pytest.mark.parametrize('resume', [False, True])
def test_training_logs_episode_fraction_and_split_win_rates(tmp_path, monkeypatch, resume):
    import rl.train as tr
    def pool(cfg, **kwargs):
        return ActorPool(cfg, seed_list=[f'force_logs_{i}' for i in range(2000)], **kwargs)
    monkeypatch.setattr(tr, 'ActorPool', pool)
    cfg = tiny_config(tmp_path, total_updates=3, epochs=1, ascension=35, force_frac=.5, force_vars=FORCE_VARS)
    cfg.rollout_steps = 512
    if resume:
        cfg.force_frac, cfg.total_updates = 0., 1
        assert train(cfg)['update'] == 1
        cfg.force_frac, cfg.total_updates, cfg.resume = .5, 3, 'auto'
    assert train(cfg)['update'] == 3
    with (tmp_path / 'progress.csv').open() as f:
        rows = list(csv.DictReader(f))
    with (tmp_path / 'episodes.csv').open() as f:
        episodes = list(csv.DictReader(f))
    current = [e for e in episodes if e['force']]
    assert .35 < np.mean([int(e['force']) for e in current]) < .65
    assert FORCE_STATS <= rows[0].keys()
    for row in rows[int(resume):]:
        done = [e for e in episodes if e['update'] == row['update'] and not e['aborted']]
        assert done
        assert float(row['force_ep_frac']) == np.mean([int(e['force']) for e in done])
        for flag, key in [('1', 'force_win'), ('0', 'main_win')]:
            group = [e for e in done if e['force'] == flag]
            expected = np.mean([e['victory'] == 'True' for e in group]) if group else ''
            assert (float(row[key]) if row[key] else '') == expected
        assert 0 <= int(row['force_replaced']) <= cfg.rollout_steps
        assert all(int(e['n_forced']) == 0 for e in done if e['force'] == '0')


def test_forced_episodes_do_not_enter_curriculum_window():
    c = AscCurriculum(Config(asc_curriculum=True, asc_window=2, asc_threshold=1.), 1)
    assert not c.record(dict(asc=1, victory=True, force=1))
    assert not c.windows
    assert not c.record(dict(asc=1, victory=True, force=0))
    assert not c.record(dict(asc=1, victory=False, force=1))
    assert c.record(dict(asc=1, victory=True, force=0))
    assert list(c.windows[1]) == [True, True]


def test_force_alone_uses_self_imitation_and_finite_ratios():
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(3))
        def forward(self, obs):
            return self.logits.expand(len(obs['n_cand']), -1), self.logits.sum().expand(len(obs['n_cand'])) * 0
    cfg = Config(force_frac=1., force_vars='Damage', epochs=1, minibatch=3, micro_batch=3,
                 value_norm=False, ent_coef=0., vf_coef=0.)
    caps = Caps(max_tok=8, max_cand=3)
    response = dict(decision='card_reward', cands=[dict(k='skip')] * 3)
    obs = Encoder(caps).encode(response)
    obs = to_torch({k: np.repeat(v[None], 3, axis=0) for k, v in obs.items()}, 'cpu')
    buf = Rollout(caps, 3, 1, torch.device('cpu'))
    buf.add([0] * 3, obs, [1] * 3, [-np.log(3), -np.log(3) + 100, -np.log(3) - 100], [0.] * 3)
    buf.explored[1:] = True
    model = Policy()
    adv = np.array([0., -2., 2.], np.float32)
    stats = ppo_update(model, torch.optim.SGD(model.parameters(), lr=0.), buf, adv, np.zeros(3, np.float32),
                       np.ones(3, bool), ValueNorm(), cfg, np.random.default_rng(17), contextlib.nullcontext)
    assert stats['pg_loss'] == pytest.approx(-(1 + cfg.clip) * (adv / adv.std())[2] / 3)
    assert stats['approx_kl'] == stats['clipfrac'] == 0.
    assert stats['explored_frac'] == pytest.approx(2 / 3)
    assert np.isfinite(stats['explored_ratio_mean']) and stats['explored_ratio_mean'] > 1e5


def test_evaluation_disables_force_without_changing_checkpoint_config(tmp_path, monkeypatch):
    import rl.evaluate as ev
    cfg = tiny_config(tmp_path, force_frac=1., force_vars=FORCE_VARS, explore_eps=1.)
    before = copy.deepcopy(cfg.to_dict())
    real_pool = ev.ActorPool
    def pool(eval_cfg, seed_list):
        assert not eval_cfg.force_enabled and eval_cfg.explore_eps == 0.
        result = real_pool(eval_cfg, seed_list)
        assert 'cand_force' not in result.shm.arrays
        return result
    monkeypatch.setattr(ev, 'ActorPool', pool)
    seeds = [('Ironclad', f'force_eval_{i}') for i in range(8)]
    actual = run_eval(cfg, AvoidCardsPolicy(), seeds, progress=False)
    cfg.force_frac = 0.
    expected = run_eval(cfg, AvoidCardsPolicy(), seeds, progress=False)
    assert actual == expected and all('force' not in e and 'n_forced' not in e for e in actual)
    cfg.force_frac = 1.
    assert cfg.to_dict() == before


def test_force_learn_false_gives_forced_steps_no_policy_gradient():
    """With force_learn off a replaced step must not move the policy whatever its advantage; with it on, a positive one does."""
    import torch as T
    from rl import ppo as P

    def surrogate(force_learn, adv):
        cfg = Config(force_ids='CARD.X', force_frac=1., force_learn=force_learn)
        ex = T.tensor([True, False])
        A = T.tensor([adv, adv])
        drop = cfg.force_enabled and not cfg.force_learn
        return T.where(ex if drop else ex & (A < 0), T.zeros_like(A), A)

    assert surrogate(False, 1.).tolist() == [0., 1.] and surrogate(False, -1.).tolist() == [0., -1.]
    assert surrogate(True, 1.).tolist() == [1., 1.] and surrogate(True, -1.).tolist() == [0., -1.]
    src = open(P.__file__).read()
    assert 'ex if drop_replaced else ex & (A < 0)' in src and 'drop_replaced = cfg.force_enabled and not cfg.force_learn' in src


def test_rewards_screen_item_with_a_listed_card_must_be_opened():
    response = dict(decision='rewards', cands=[
        dict(k='reward', idx=0, rt='Potion', id='POTION.X'),
        dict(k='reward', idx=1, rt='Card', n=3, cards=[dict(id='CARD.A', v={'Block': 5}), dict(id='CARD.X', v={})]),
        dict(k='reward', idx=2, rt='Card', n=3, cards=[dict(id='CARD.A', v={'Block': 5})]),
        dict(k='reward', idx=3, rt='Card', n=3),
        dict(k='proceed')])
    assert force_matches(response, {'CARD.X'}, set()) == [False, True, False, False, False]
    assert force_matches(response, set(), {'Block'}) == [False, True, True, False, False]
    assert force_matches(dict(response, decision='event_choice'), {'CARD.X'}, set()) == [False] * 5


def test_force_subset_draws_one_set_per_game(tmp_path):
    cfg = tiny_config(tmp_path, force_frac=1., force_ids=','.join(c['id'] for c in CARD_POOL), force_subset=.5)
    shm = SharedObs(cfg.caps, 1, force=cfg.force_enabled)
    worker = _Worker(0, [0], None, shm.path, dict(cfg.to_dict(), n_envs=1), SimpleNamespace(value=0), None, None)
    try:
        sets = []
        for _ in range(30):
            worker.draw_force_subset(0)
            active = worker.force_active[0]
            assert active <= worker.force_ids
            response = max((r for r in RS if r['decision'] == 'card_reward'), key=lambda r: len(r['cands']))
            worker.encode(0, response)
            expected = force_matches(response, active, set())
            np.testing.assert_array_equal(shm.arrays['cand_force'][0, :len(expected)], expected)
            sets.append(frozenset(active))
        assert len(set(sets)) > 5
        mean = np.mean([len(s) for s in sets]) / len(worker.force_ids)
        assert .25 < mean < .75
    finally:
        worker.shutdown(); worker.shm.close(); shm.close()
