import contextlib
import copy
import csv
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl.actors import ActorPool, _Worker
from rl.config import Config, parse_args
from rl.encode import Caps, Encoder
from rl.fake_engine import FakeGame
from rl.ppo import Rollout, ValueNorm, branch_cross_entropy, ppo_update
from rl.protocol import decision_id
from rl.train import CSVLog, Collector


class FirstPolicy(torch.nn.Module):
    def forward(self, b):
        n = b['n_cand'].long()
        logits = torch.full((len(n), int(n.max())), -100., device=n.device)
        logits[:, 0] = 100.
        return logits, torch.full((len(n),), .25, device=n.device)


def _config(path, **kw):
    cfg = parse_args(['--preset', 'tiny', '--n_workers', '1', '--envs_per_worker', '1',
                      '--engine', 'fakeproc', '--fake_args=--rest_test --log_every 0',
                      '--branch_decisions', 'rest_site', '--rollout_steps', '1', '--run_dir', str(path)])
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


@pytest.mark.parametrize('cap', [1, 2, 600])
def test_boundary_identical_entries_masks_bootstraps_and_episode_counts(tmp_path, cap):
    cfg = _config(tmp_path, branch_max_steps=cap)
    pool = ActorPool(cfg, seed_list=['branch', 'next'])
    col = Collector(cfg, pool, FirstPolicy(), ValueNorm(), 'cpu', contextlib.nullcontext, err_log=False)
    b = Rollout(cfg.caps, 1, 1, torch.device('cpu'))
    try:
        boot, _ = col.collect(b)
        assert not col.errors and not col.branches and not col.groups and not b.groups
        assert b.n > cfg.rollout_steps and b.has_res[:b.n].all()
        assert len(b.branches) == 1
        branch = b.branches[0]
        side, main = branch['trials']
        assert branch['actions'] == [1, 0]
        for values in b.obs.values():
            assert torch.equal(values[side[0]], values[main[0]])
        assert b.act[side[0]] == 1 and b.act[main[0]] == 0
        # The sampled main action has one entry; only the side's imposed first step is masked.
        assert len(np.flatnonzero(b.obs['decision'][:b.n].numpy() == b.obs['decision'][main[0]].item())) == 2
        assert b.forced[:b.n].sum() == 1 and b.forced[side[0]] and not b.forced[main[0]]
        assert b.cut[side[-1]] and not b.cut[main[-1]]
        assert len(side) == len(main) == min(cap, 4)
        expected_boot = 0. if cap == 600 else .25
        assert branch['boots'] == [expected_boot] * 2
        assert b.bootstrap[side[-1]] == expected_boot
        adv, ret, mask = b.gae(boot, cfg.gamma, cfg.lam)
        assert np.array_equal(mask, ~b.forced[:b.n])
        expected = np.zeros(b.n)
        for ix, value in zip(branch['trials'], branch['boots']):
            gae = 0.
            for i in reversed(ix):
                if b.done[i]:
                    value, gae = 0., 0.
                gae = b.rew[i] + cfg.gamma * value - b.val[i] + cfg.gamma * cfg.lam * gae
                expected[i] = gae
                value = b.val[i]
        np.testing.assert_allclose(adv, expected, atol=1e-6)
        np.testing.assert_allclose(ret, expected + b.val[:b.n], atol=1e-6)
        _, _, _, stats = b.branch_targets(cfg.gamma, 1., .1)
        for action, ix, value in zip(branch['actions'], branch['trials'], branch['boots']):
            q = sum(cfg.gamma**t * b.rew[i] for t, i in enumerate(ix)) + cfg.gamma**len(ix) * value
            assert stats['branch_q_' + ['HEAL', 'SMITH'][action]] == pytest.approx(q)
        assert b.branch_extra == len(side)
        assert len(col.episodes) == (1 if cap == 600 else 0)
        if col.episodes:
            ep = col.episodes[0]
            assert ep['seed'] == 'branch' and not ep['victory'] and ep['steps'] == len(main)
            assert ep['ret'] == pytest.approx(sum(b.rew[main]))
    finally:
        pool.close()


def _worker():
    w = _Worker.__new__(_Worker)
    w.cfg = Config(branch_decisions='rest_site').to_dict()
    w.enc = Encoder()
    slot = w.enc.alloc()
    w.shm = SimpleNamespace(slot=lambda e: slot)
    w.groups, w.seen_save, w.last = {}, {}, {}
    w.branches, w.branch_offers = {}, {}
    w.branch_decisions, w.branch_rng = {'rest_site'}, random.Random(0)
    w.group_rooms = {'Boss', 'Elite'}
    w.ep = {0: dict(env=0, steps=10, ret=.2, floor=2, act=1)}
    w.events, w.deadline, w.state = [], {}, {0: w.ACT}
    sent, restarts = [], []
    w.eng = {0: SimpleNamespace(send=sent.append)}
    w.next_run = restarts.append
    entry = copy.deepcopy(FakeGame(rest_test=True).start())
    entry['info'].update(total_floor=2, act=1)
    entry['info']['save']['room'] = 'RestSite'
    return w, entry, sent, restarts


@pytest.mark.parametrize('change', ['none', 'one_option', 'wrong_kind', 'later_decision', 'group', 'prob_zero', 'wrong_room'])
def test_only_first_eligible_room_decision_offers_branch(change):
    w, r, _, _ = _worker()
    if change == 'one_option':
        r['cands'][1]['k'] = 'leave'
    elif change == 'wrong_kind':
        r['info']['save']['kind'] = 'combat'
    elif change == 'later_decision':
        initial = copy.deepcopy(r)
        initial['decision'] = 'unknown'
        w.encode(0, initial)
    elif change == 'group':
        w.groups[0] = {}
    elif change == 'prob_zero':
        w.cfg['branch_prob'] = 0.
    elif change == 'wrong_room':
        r['info']['save']['room'] = 'Event'
    w.encode(0, r)
    offers = [ev for ev in w.events if ev[0] == 'branch_offer']
    assert len(offers) == (1 if change == 'none' else 0)
    w.encode(0, r)
    assert len([ev for ev in w.events if ev[0] == 'branch_offer']) == len(offers)


def test_boss_exit_bootstraps_before_load_and_main_continues():
    w, entry, sent, restarts = _worker()
    w.encode(0, entry)
    w.start_branch(0, 0)
    assert sent == [dict(cmd='rl_load', id=1)]
    w.handle(0, entry)
    assert ('branch_trial', 0, 1, False) in w.events
    boss = copy.deepcopy(entry)
    boss['decision'] = 'combat_play'
    boss['obs']['g']['room'] = 'Boss'
    boss['obs']['combat'] = dict(round=1)
    boss['info']['save'] = dict(id=2, kind='room', room='Boss')
    end = copy.deepcopy(boss)
    end['obs']['combat'], end['decision'] = None, 'card_reward'
    for main in [False, True]:
        w.state[0] = w.ACT
        w.handle(0, boss)
        assert w.state[0] == w.IDLE and not w.groups
        w.state[0] = w.ACT
        w.handle(0, end)
        assert w.state[0] == w.VALUE
        assert w.events[-1] == ('branch_end', 0, main, False)
        assert len(sent) == (1 if not main else 2)
        w.value_done(0)
        if not main:
            assert w.state[0] == w.LOAD and w.ep[0]['steps'] == 10
            w.handle(0, entry)
        else:
            assert w.state[0] == w.IDLE and not w.branches
    assert not restarts and not any(ev[0] == 'ep' for ev in w.events)
    assert w.ep[0]['steps'] == 12


def test_side_failure_restores_sampled_main_without_episode():
    w, entry, sent, _ = _worker()
    w.encode(0, entry)
    w.start_branch(0, 0)
    w.handle(0, entry)
    w.state[0] = w.ACT
    w.fail(0, 'injected error')
    assert w.state[0] == w.LOAD and len(sent) == 2
    assert ('branch_cancel', 0, True) in w.events
    assert not any(ev[0] == 'ep' for ev in w.events)
    w.handle(0, entry)
    assert ('branch_trial', 0, 0, True) in w.events


@pytest.mark.parametrize('room', ['Elite', 'Boss'])
@pytest.mark.parametrize('main', [False, True])
def test_only_main_branch_starts_combat_groups(room, main):
    w, r, _, _ = _worker()
    w.cfg['group_k'] = 3
    w.branches[0] = dict(main=main, boss=False)
    r['obs']['combat'] = dict(round=1)
    r['obs']['g']['room'] = room
    r['info']['save']['room'] = room
    w.encode(0, r)
    assert bool(w.groups) == main
    assert w.events == ([('group_start', 0)] if main else [])
    assert not w.branch_offers


@pytest.mark.parametrize('main', [False, True])
def test_active_branch_does_not_offer_nested_rest_branch(main):
    w, r, _, _ = _worker()
    w.cfg['group_k'] = 3
    w.branches[0] = dict(main=main, boss=False)
    w.encode(0, r)
    assert not w.branch_offers and not w.events


@pytest.mark.parametrize('terminal', [False, True])
def test_main_boss_finishes_group_before_branch(terminal):
    w, entry, sent, restarts = _worker()
    w.cfg['group_k'] = 3
    w.encode(0, entry)
    w.start_branch(0, 0)
    w.next_branch(0)
    w.handle(0, entry)
    boss = copy.deepcopy(entry)
    boss['obs']['combat'] = dict(round=1)
    boss['obs']['g']['room'] = 'Boss'
    boss['info']['save'] = dict(id=2, kind='room', room='Boss')
    w.state[0] = w.ACT
    w.handle(0, boss)
    end = copy.deepcopy(boss)
    end['obs']['combat'] = None
    if terminal:
        end.update(done=True, victory=False, cands=[])
    for trial in range(3):
        w.events.clear()
        w.state[0] = w.ACT
        w.handle(0, end)
        assert w.events[1] == ('group_end', 0, not terminal, trial < 2, terminal)
        if trial < 2:
            assert not any(ev[0] in ('branch_end', 'ep') for ev in w.events)
            assert 0 in w.branches and not restarts
            if not terminal:
                w.value_done(0)
            assert w.state[0] == w.LOAD and w.ep[0]['steps'] == 11
            w.handle(0, boss)
            assert not any(ev[0] == 'branch_trial' for ev in w.events)
        elif terminal:
            assert w.events[2] == ('branch_end', 0, True, True)
            assert not w.groups and not w.branches and restarts == [0]
        else:
            assert w.state[0] == w.VALUE and 0 in w.groups
            assert not any(ev[0] == 'branch_end' for ev in w.events)
            w.value_done(0)
            assert not w.groups and w.state[0] == w.VALUE
            assert w.events[-1] == ('branch_end', 0, True, False)
            w.value_done(0)
            assert not w.branches and w.state[0] == w.IDLE
    assert sent[-2:] == [dict(cmd='rl_load', id=2)] * 2


class RestPolicy(FirstPolicy):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, obs):
        logits, value = super().forward(obs)
        if self.action == 1:
            rest = obs['decision'] == decision_id('rest_site')
            if rest.any():
                logits[rest, 0], logits[rest, 1] = -100., 100.
        return logits, value


@pytest.mark.parametrize('action,cap,k', [(0, 600, 3), (1, 600, 3), (0, 3, 3), (0, 600, 1000)])
def test_main_branch_outcome_uses_only_final_combat_path(tmp_path, action, cap, k):
    cfg = _config(tmp_path, branch_max_steps=cap, group_k=k)
    pool = ActorPool(cfg, seed_list=['combined', 'next'])
    col = Collector(cfg, pool, RestPolicy(action), ValueNorm(), 'cpu', contextlib.nullcontext, err_log=False)
    b = Rollout(cfg.caps, 1, 1, torch.device('cpu'))
    try:
        col.collect(b)
        assert not col.errors and not col.groups and not col.branches
        assert b.has_res[:b.n].all() and len(b.branches) == len(b.groups) == 1
        branch, group = b.branches[0], b.groups[0]
        side, main = branch['trials']
        if k == 3:
            assert len(group['trials']) == k
        else:
            assert 1 < len(group['trials']) < k and b.n >= 256
        prefix = list(range(len(side), group['trials'][0][0]))
        expected = prefix + group['trials'][-1][:max(0, cap - len(prefix))]
        assert main == expected and len(main) == min(cap, 4)
        dropped = [i for ix in group['trials'][:-1] for i in ix]
        assert not set(side) & {i for ix in group['trials'] for i in ix}
        assert not set(main) & set(dropped) and b.cut[dropped[-1]]
        assert group['wins'] == [bool(action)] * len(group['trials'])
        assert group['boots'] == [0.] * len(group['trials'])
        assert b.forced[:b.n].sum() == 1 and b.forced[side[0]]
        assert len(col.episodes) == 1 and col.episodes[0]['steps'] == 4
        assert col.episodes[0]['ret'] == pytest.approx(sum(b.rew[prefix + group['trials'][-1]]))
        assert branch['boots'][-1] == (0. if cap == 600 else .25)
        b.rew[:b.n] = np.arange(b.n) * .03 - .1
        gamma = .9
        q = branch['boots'][-1]
        for i in reversed(expected):
            q = float(b.rew[i]) + gamma * q
        key = 'branch_q_' + ['HEAL', 'SMITH'][action]
        stats = b.branch_targets(gamma, 1., .1)[3]
        assert stats[key] == pytest.approx(q)
        b.rew[dropped] += 1000
        assert b.branch_targets(gamma, 1., .1)[3][key] == stats[key]
    finally:
        pool.close()


def test_cancelled_group_attaches_only_current_trial_to_main():
    cfg = Config(group_k=3, branch_decisions='rest_site')
    col = Collector(cfg, None, None, None, 'cpu', contextlib.nullcontext, err_log=False)
    b = Rollout(Caps(max_tok=8, max_ent=1, max_cand=4, max_cent=1), 7, 1, 'cpu')
    b.n, b.env_idx = 6, [list(range(6))]
    b.cut[3], b.bootstrap[3] = True, .7
    col.branches[0] = dict(trials=[[0, 1]], boots=[])
    col.groups[0] = dict(trials=[[2, 3], [4, 5]], wins=[True], boots=[.7])
    col.pending[0] = 5
    col.handle([('res', 0, .2, False, True, False), ('group_cancel', 0)], b)
    assert col.branches[0]['trials'][-1] == [0, 1, 4, 5]
    assert not col.groups and not b.groups and b.trunc[5]
    assert b.cut[3] and b.bootstrap[3] == pytest.approx(.7)


def test_relink_sides_preserves_main_gae_and_counts_main_once():
    cfg = Config(branch_decisions='rest_site')
    col = Collector(cfg, None, None, None, 'cpu', contextlib.nullcontext, err_log=False)
    b = Rollout(Caps(max_tok=8, max_ent=1, max_cand=4, max_cent=1), 8, 1, 'cpu')
    b.n, b.env_idx = 4, [[0, 1, 2, 3]]
    b.rew[:7] = [.2, .1, 10, 20, .3, .4, .5]
    b.val[:7] = np.arange(7) / 10
    col.branches[0] = dict(trials=[[2, 3]], boots=[])
    col.finish_branch(0, .7, False, b)
    assert b.env_idx[0] == [2, 3, 0, 1]
    b.env_idx[0].extend([4, 5, 6])
    b.n = 7
    b.done[6] = True
    adv, _, _ = b.gae([0.], .9, .8)
    expected, nv, na = {}, 0., 0.
    for i in reversed([0, 1, 4, 5, 6]):
        expected[i] = b.rew[i] + .9 * nv - b.val[i] + .9 * .8 * na
        nv, na = b.val[i], expected[i]
    np.testing.assert_allclose(adv[[0, 1, 4, 5, 6]], [expected[i] for i in [0, 1, 4, 5, 6]], atol=1e-6)
    assert len(b.env_idx[0]) == len(set(b.env_idx[0])) == 7
    b.rew[2:4] += 1000
    np.testing.assert_array_equal(adv[[0, 1, 4, 5, 6]], b.gae([0.], .9, .8)[0][[0, 1, 4, 5, 6]])


def test_targets_and_subset_cross_entropy_match_reference():
    b = Rollout(Caps(max_tok=8, max_ent=1, max_cand=4, max_cent=1), 6, 1, 'cpu')
    b.n = 6
    b.rew[:6] = [.2, .3, -.5, 1., .1, .4]
    branch = dict(options=[0, 2, 3], keys=['HEAL', 'SMITH', 'OTHER'], actions=[0, 3, 2],
                  trials=[[0, 1], [2], [3, 4, 5]], boots=[.7, -.2, 0.],
                  logits=np.array([-1., 50., -6., -2.], np.float32))
    b.branches = [branch]
    gamma, std, temp = .9, .03, .7
    idx, targets, choices, stats = b.branch_targets(gamma, std, temp)
    q = {a: sum(gamma**t * b.rew[i] for t, i in enumerate(ix)) + gamma**len(ix)*boot
         for a, ix, boot in zip(branch['actions'], branch['trials'], branch['boots'])}
    values = np.array([q[a] for a in branch['options']])
    scores = branch['logits'][branch['options']] + np.clip((values-values.mean())/std, -5, 5)/temp
    expected = np.exp(scores-scores.max()); expected /= expected.sum()
    assert idx == [3] and targets[0][1] == 0 and not choices[0][1]
    np.testing.assert_allclose(targets[0][branch['options']], expected, atol=1e-7)
    assert stats['branch_q_SMITH'] == pytest.approx(q[2])
    logits = torch.tensor([[.7, 100., -1.2, 2.]], requires_grad=True)
    ce = branch_cross_entropy(logits, torch.tensor(np.array(targets)), torch.tensor(np.array(choices))).mean()
    ref = -(torch.tensor(expected) * logits[0, branch['options']].log_softmax(-1)).sum()
    torch.testing.assert_close(ce, ref.float())
    ce.backward()
    assert logits.grad[0, 1] == 0


@pytest.mark.parametrize('micro', [1, 4])
def test_ppo_masks_forced_step_and_accumulates_mean_branch_ce(micro):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(6, 3))
            self.v = torch.nn.Parameter(torch.zeros(6))
        def forward(self, obs):
            i = obs['decision'].long()
            return self.p[i], self.v[i]
    b = Rollout(Caps(max_tok=8, max_ent=1, max_cand=3, max_cent=1), 6, 1, torch.device('cpu'))
    b.n, b.env_idx = 6, [list(range(6))]
    b.obs['decision'][:] = torch.arange(6)
    b.obs['n_tok'][:], b.obs['n_cand'][:] = 1, 3
    b.logp[:] = -np.log(3)
    b.forced[0] = True
    b.branches = [dict(options=[0, 2], keys=['HEAL', 'SMITH'], actions=[2, 0],
                       trials=[[0], [i]], boots=[0., 0.], logits=np.zeros(3)) for i in [1, 4]]
    b.rew[0] = 1.
    adv, ret, mask = b.gae([0.], .9, .8)
    cfg = Config(branch_decisions='rest_site', value_norm=False, epochs=1, minibatch=6,
                 micro_batch=micro, ent_coef=0., vf_coef=0., max_grad_norm=1e6, branch_coef=1.7)
    model = Toy()
    idx, targets, choices, _ = b.branch_targets(cfg.gamma, 1., cfg.branch_temp)
    ref = branch_cross_entropy(model.p[idx], torch.tensor(np.array(targets)), torch.tensor(np.array(choices))).mean()*cfg.branch_coef
    expected = torch.autograd.grad(ref, model.p)[0]
    ppo_update(model, torch.optim.SGD(model.parameters(), lr=0), b, np.zeros(6, np.float32),
               ret, mask, ValueNorm(), cfg, np.random.default_rng(0), contextlib.nullcontext)
    torch.testing.assert_close(model.p.grad, expected)
    assert not model.p.grad[0].any() and not model.v.grad[0]


def test_branch_csv_adds_late_option_columns(tmp_path, capsys):
    path = tmp_path / 'progress.csv'
    log = CSVLog(path, expand=True)
    log.write(dict(update=1, branch_count=0))
    log.write(dict(update=2, branch_count=1, branch_q_SMITH=.9))
    log.close()
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['branch_q_SMITH'] == '' and rows[1]['branch_q_SMITH'] == '0.9'
    assert not capsys.readouterr().out


def test_evaluation_disables_branches(tmp_path):
    from rl.evaluate import run_eval
    cfg = _config(tmp_path)
    eps = run_eval(cfg, FirstPolicy(), [('Ironclad', 'eval-branch')], progress=False, stall_seconds=3)
    assert cfg.branch_decisions == 'rest_site'
    assert len(eps) == 1 and not eps[0]['aborted'] and eps[0]['steps'] == 4
