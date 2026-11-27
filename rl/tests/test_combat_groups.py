import contextlib
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl.actors import ActorPool, _Worker
from rl.config import Config, parse_args
from rl.encode import Caps
from rl.engine import FakeEngine
from rl.ppo import Rollout, ValueNorm
from rl.train import Collector, act_batch


@pytest.mark.parametrize('room', ['Unknown', 'Elite'])
def test_fake_room_load_replays_rng_and_state(room):
    engine = FakeEngine()
    try:
        r = engine.reset(seed='combat-save')
        assert r['info']['save'] is None
        r = engine.step(0)
        node = r['cands'][0]['node']
        engine.server.game.map_nodes[node]['t'] = room
        r = copy.deepcopy(engine.step(0))
        entry = copy.deepcopy(r)
        save = r['info']['save']
        assert set(save) == {'id', 'kind', 'floor', 'room'} and save['kind'] == 'room'
        actions, responses = [], []
        while not r['done'] and r['info']['save']['id'] == save['id']:
            actions.append(len(r['cands']) - 1)
            r = copy.deepcopy(engine.step(actions[-1]))
            responses.append(r)
        initial = copy.deepcopy(engine.request({'cmd': 'rl_load', 'id': save['id']}))
        assert initial == entry
        replay = [copy.deepcopy(engine.step(a)) for a in actions]
        # A later room can receive a new save ID; game observations and RNG must still match.
        for actual, expected in zip(replay, responses):
            actual['info'].pop('save')
            expected['info'].pop('save')
            assert actual == expected
        assert engine.request({'cmd': 'rl_load', 'id': -1})['type'] == 'error'
        assert engine.request({'cmd': 'rl_load', 'id': save['id']}) == initial
    finally:
        engine.close()


def _buffer(cap=20, n_envs=1):
    return Rollout(Caps(max_tok=8, max_ent=2, max_cand=4, max_cent=2), cap, n_envs, 'cpu')


def test_cut_gae_matches_stepwise_reference():
    b = _buffer(9, 2)
    b.n = 9
    b.env_idx = [[0, 2, 3, 5, 8], [1, 4, 6, 7]]
    b.rew[:9] = [.2, -.1, .5, 20, .3, -.4, .8, .9, .7]
    b.val[:9] = [.4, .1, -.2, 10, .8, .3, -.1, .5, .6]
    b.done[[5, 7]] = True
    b.trunc[6] = True
    b.cut[[2, 4, 5]] = True
    b.bootstrap[[2, 4, 5]] = [1.7, -.8, 999]
    boot, gamma, lam = [.9, 8], .93, .81
    expected = np.zeros(9)
    for e, ix in enumerate(b.env_idx):
        nv, na = boot[e], 0.
        for i in reversed(ix):
            if b.done[i]:
                nv, na = 0., 0.
            elif b.cut[i]:
                nv, na = b.bootstrap[i], 0.
            expected[i] = 0. if b.trunc[i] else b.rew[i] + gamma * nv - b.val[i] + gamma * lam * na
            nv, na = b.val[i], expected[i]
    adv, ret, mask = b.gae(boot, gamma, lam)
    np.testing.assert_allclose(adv, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ret, expected + b.val[:9], rtol=1e-6, atol=1e-6)
    assert not mask[6] and mask.sum() == 8
    before = adv.copy()
    b.rew[3] += 1000
    after, _, _ = b.gae(boot, gamma, lam)
    np.testing.assert_array_equal(before[[0, 2]], after[[0, 2]])


@pytest.mark.parametrize('alpha', [0., .5, 1.])
def test_leave_one_out_advantages_keep_value_targets(alpha):
    b = _buffer(8)
    b.n, b.env_idx = 8, [list(range(8))]
    b.rew[:8] = [1., .3, -.2, .7, .1, .8, -.5, .2]
    b.val[:8] = [.2, .3, .1, .9, -.1, .4, .6, .2]
    b.cut[[1, 4]] = True
    b.bootstrap[[1, 4]] = [2., 0.]
    b.done[4] = True
    b.groups = [dict(trials=[[0, 1], [2, 3, 4], [5, 6]], boots=[2., 0., -.4], wins=[True, False, True])]
    gamma = .9
    adv, ret, mask = b.gae([.6], gamma, .8)
    original_ret = ret.copy()
    expected, outcomes = adv.copy(), []
    for ix, boot in zip(b.groups[0]['trials'], b.groups[0]['boots']):
        outcomes.append(sum(gamma ** t * b.rew[i] for t, i in enumerate(ix)) + gamma ** len(ix) * boot)
    for t, ix in enumerate(b.groups[0]['trials']):
        baseline = np.mean([o for j, o in enumerate(outcomes) if j != t])
        expected[ix] = alpha * adv[ix] + (1 - alpha) * (outcomes[t] - baseline)
    mixed, stats = b.group_advantages(adv, gamma, alpha)
    np.testing.assert_allclose(mixed, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(ret, original_ret)
    np.testing.assert_array_equal(adv + b.val[:8], ret)
    assert mixed[7] == adv[7] and mask.all()
    assert stats['group_count'] == 1
    assert stats['group_win_rate'] == pytest.approx(2 / 3)
    assert stats['group_outcome_std'] == pytest.approx(np.std(outcomes))
    assert stats['group_extra_frac'] == 5 / 8


def _worker(k=3):
    w = _Worker.__new__(_Worker)
    w.cfg = Config(group_k=k, restart_every_episode=True).to_dict()
    ep = dict(env=0, seed='test', steps=10, ret=.1, floor=2, act=1, cstep=1, tstep=1, round=1)
    w.ep, w.state = {0: ep.copy()}, {0: w.ACT}
    w.groups = {0: dict(id=7, ep=ep.copy(), trial=0, last=False)}
    w.events, w.deadline = [], {}
    sent, restarts, encoded = [], [], []
    w.eng = {0: SimpleNamespace(send=sent.append)}
    w.next_run = restarts.append
    w.encode = lambda e, r: encoded.append(copy.deepcopy(r))
    return w, sent, restarts, encoded


def test_intermediate_deaths_only_record_last_episode():
    w, sent, restarts, _ = _worker()
    death = dict(type='rl', done=True, victory=False, info=dict(total_floor=2, act=1), cands=[])
    entry = dict(type='rl', done=False, cands=[dict(k='end_turn')])
    for trial in range(3):
        w.state[0] = w.ACT
        w.handle(0, death)
        eps = [ev[1] for ev in w.events if ev[0] == 'ep']
        if trial < 2:
            assert not eps and not restarts and w.state[0] == w.LOAD
            assert w.ep[0]['steps'] == 10
            w.handle(0, entry)
        else:
            assert len(eps) == 1 and eps[0]['steps'] == 11
            assert eps[0]['ret'] == .1 and restarts == [0]
    assert sent == [{'cmd': 'rl_load', 'id': 7}] * 2


@pytest.mark.parametrize('trial,repeat', [(0, True), (2, False)])
def test_winning_end_waits_for_value_before_reloading(trial, repeat):
    w, sent, restarts, encoded = _worker()
    w.groups[0]['trial'] = trial
    end = dict(type='rl', done=False, cands=[dict(k='skip')], info=dict(total_floor=2, act=1), obs=dict(combat=None))
    w.handle(0, end)
    assert w.state[0] == w.VALUE and not sent and not restarts
    assert encoded == [end]
    assert w.events[-1] == ('group_end', 0, True, repeat, False)
    w.value_done(0)
    assert w.state[0] == (w.LOAD if repeat else w.IDLE)
    assert sent == ([{'cmd': 'rl_load', 'id': 7}] if repeat else [])


class EndTurnPolicy(torch.nn.Module):
    def forward(self, obs):
        n = obs['n_cand'].long()
        logits = torch.full((len(n), int(n.max())), -100., device=n.device)
        logits.scatter_(1, (n - 1)[:, None], 100.)
        return logits, torch.full((len(n),), .25, device=n.device)


def test_value_only_inference_does_not_sample():
    state = torch.get_rng_state()
    a, lp, v = act_batch(EndTurnPolicy(), {'n_cand': torch.tensor([2, 4])}, contextlib.nullcontext, n_actions=0)
    assert a.numel() == lp.numel() == 0 and len(v) == 2
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize('k', [4, 100])
def test_rollout_boundary_finishes_group_with_identical_starts(tmp_path, k):
    cfg = parse_args(['--preset', 'tiny', '--n_workers', '1', '--envs_per_worker', '1', '--rollout_steps', '3',
                      '--group_k', str(k), '--group_rooms', 'Monster,Elite,Boss', '--run_dir', str(tmp_path)])
    pool = ActorPool(cfg, seed_list=['boundary', 'next'])
    col = Collector(cfg, pool, EndTurnPolicy(), ValueNorm(), 'cpu', contextlib.nullcontext, err_log=False)
    b = Rollout(cfg.caps, cfg.rollout_steps, 1, 'cpu')
    try:
        boot, _ = col.collect(b)
        assert not col.errors and b.has_res[:b.n].all()
        assert b.n > cfg.rollout_steps and len(b.groups) == 1
        g = b.groups[0]
        if k == 4:
            assert len(g['trials']) == k
        else:
            assert 1 < len(g['trials']) < k
            assert b.n >= cfg.rollout_steps + 256 - 1
        assert g['wins'] == [False] * len(g['trials'])
        assert len(col.episodes) == 1 and col.episodes[0]['seed'] == 'boundary'
        first = [ix[0] for ix in g['trials']]
        for v in b.obs.values():
            for i in first[1:]:
                assert torch.equal(v[first[0]], v[i])
        assert b.done[[ix[-1] for ix in g['trials']]].all()
        assert b.cut[[ix[-1] for ix in g['trials'][:-1]]].all()
        # Only the final attempt contributes to the episode's counters.
        assert col.episodes[0]['steps'] == 2 + len(g['trials'][-1])
        assert not any(g['trials'][0] for g in col.groups.values())
        assert b.gae(boot, cfg.gamma, cfg.lam)[2].all()
    finally:
        pool.close()


def test_group_step_limit_cancels_before_next_episode():
    w, _, restarts, _ = _worker()
    w.cfg['max_episode_steps'] = 11
    w.handle(0, dict(type='rl', done=False, cands=[dict(k='end_turn')],
                     info=dict(total_floor=2, act=1), obs=dict(combat=dict(round=1))))
    assert not w.groups and ('group_cancel', 0) in w.events
    assert restarts == [0]
    assert [ev[1]['aborted'] for ev in w.events if ev[0] == 'ep'] == ['episode_step_limit']


def test_evaluation_disables_checkpoint_groups(tmp_path):
    from rl.evaluate import run_eval
    cfg = parse_args(['--preset', 'tiny', '--n_workers', '1', '--envs_per_worker', '1',
                      '--group_k', '4', '--group_rooms', 'Monster,Elite,Boss', '--run_dir', str(tmp_path)])
    eps = run_eval(cfg, EndTurnPolicy(), [('Ironclad', 'eval-group')], progress=False, stall_seconds=3)
    assert cfg.group_k == 4
    assert len(eps) == 1 and not eps[0]['aborted']
