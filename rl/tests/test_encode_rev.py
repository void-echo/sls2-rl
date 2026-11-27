import copy

import numpy as np
import torch

from rl.encode import Caps, Encoder, N_TOKEN_TYPES, TT, creature_ents, k1, k2
from rl.model import ModelCfg, Policy
from rl.ppo import sliced, to_torch


def keys(a):
    return {int(x) & 0xffffffff for x in a if x}


def test_power_cards_keep_instance_owner():
    creature = dict(id='PLAYER', pw=[dict(id='NIGHTMARE', n=1, cards=['BASH']),
                                     dict(id='NIGHTMARE', n=3, cards=['DEFEND'])])
    other = copy.deepcopy(creature)
    other['pw'][0]['cards'], other['pw'][1]['cards'] = other['pw'][1]['cards'], other['pw'][0]['cards']
    enc = Encoder(Caps(enc_rev=1, max_ent=40))
    a = enc.encode(dict(obs=dict(combat=dict(player=creature))))
    b = enc.encode(dict(obs=dict(combat=dict(player=other))))
    player = np.flatnonzero(a['tok_type'] == TT['PLAYER'])[0]
    assert k2('pw.card', 'NIGHTMARE/BASH') in keys(a['tok_key'][player])
    assert k2('pw.card', 'NIGHTMARE#1/DEFEND') in keys(a['tok_key'][player])
    assert keys(a['tok_key'][player]) != keys(b['tok_key'][player])


def test_room_history_type_id_order_and_caps():
    rooms = [dict(t='Monster', id=f'FIGHT{i}') for i in range(8)] + [dict(t='Elite', id='A'), dict(t='Rest')]
    r = dict(obs=dict(g=dict(act_rooms=rooms), map=dict(nodes=[dict(r=0, c=0, t='Monster')]),
                      deck=[dict(id='STRIKE')]))
    enc = Encoder(Caps(enc_rev=1))
    a = enc.encode(r)
    tokens = np.flatnonzero(a['tok_type'] == TT['ROOM'])
    assert len(tokens) == 10 and TT['ROOM'] == 20 and N_TOKEN_TYPES == 24
    assert keys(a['tok_key'][tokens[0]]) == {k2('room.t', 'Rest'), k1('room.ago')}
    assert k2('room.id', 'A') in keys(a['tok_key'][tokens[1]])
    assert a['tok_val'][tokens[0], 1, 1] == np.float16(.1)
    assert a['tok_val'][tokens[1], 2, 1] == np.float16(.2)
    assert k1('g.act_rooms.len') in keys(a['tok_key'][0])
    assert k2('g.act_rooms', 'FIGHT0') not in keys(a['tok_key'][0])
    assert np.flatnonzero(a['tok_type'] == TT['MAPNODE'])[-1] < tokens[0]
    assert tokens[-1] < np.flatnonzero(a['tok_type'] == TT['DECK'])[0]
    for change in ('ninth', 'type', 'order'):
        b = copy.deepcopy(r)
        history = b['obs']['g']['act_rooms']
        if change == 'ninth':
            history[8]['id'] = 'B'
        elif change == 'type':
            history[0]['t'] = 'Elite'
        else:
            history.reverse()
        out = enc.encode(b)
        assert any(a[k].tobytes() != out[k].tobytes() for k in a)
    limited = Encoder(Caps(enc_rev=1, rooms=2))
    assert (limited.encode(r)['tok_type'] == TT['ROOM']).sum() == 2
    assert limited.trunc['rooms'] == 1
    r['obs']['combat'] = {}
    assert not (enc.encode(r)['tok_type'] == TT['ROOM']).any()
    assert (Encoder(Caps(enc_rev=1, rooms_in_combat=True)).encode(r)['tok_type'] == TT['ROOM']).sum() == 10
    assert not (Encoder(Caps(rooms_in_combat=True)).encode(r)['tok_type'] == TT['ROOM']).any()
    assert Caps.from_dict({}).enc_rev == 0


def test_inline_cards_deduplicate_full_face_and_preserve_refs():
    strike = dict(id='STRIKE', cost=1, v=dict(Damage=6), d=dict(count=1))
    cards = [copy.deepcopy(strike) for _ in range(96)] + [dict(id='BASH', cost=2), dict(id='DEFEND', cost=1)]
    r = dict(decision='card_select', obs=dict(deck=cards),
             cands=[dict(k='pick', src=i, card=c) for i, c in enumerate(cards)])
    enc = Encoder(Caps(enc_rev=1, max_tok=192, max_ent=40, max_cent=12, pile_counts=True))
    a = enc.encode(r)
    assert (a['tok_type'] == TT['OFFER_CARD']).sum() == 3
    assert (a['cand_ref'][:96, 0] == a['cand_ref'][0, 0]).all()
    assert a['cand_ref'][96, 0] >= 0 and a['cand_ref'][97, 0] >= 0
    assert a['cand_ref'][96, 0] != a['cand_ref'][97, 0]
    assert not enc.trunc
    torch.manual_seed(3)
    model = Policy(ModelCfg(d=32, layers=1, heads=4)).eval()
    batch = to_torch({k: v[None] for k, v in a.items()}, 'cpu')
    with torch.no_grad():
        logits, _ = model(sliced(batch, torch.tensor([0])))
    assert abs(float(logits[0, 96] - logits[0, 97])) > 1e-6
    reordered = dict(d=dict(count=1), v=dict(Damage=6), cost=1, id='STRIKE')
    changed = dict(strike, d=dict(count=2))
    r['cands'] = [dict(k='pick', card=strike), dict(k='bundle', cards=[reordered, changed, strike])]
    b = enc.encode(r)
    assert (b['tok_type'] == TT['OFFER_CARD']).sum() == 2
    assert b['cand_ref'][0, 0] == b['cand_ref'][1, 2] == b['cand_ref'][1, 4]
    assert b['cand_ref'][1, 3] != b['cand_ref'][0, 0]


def test_candidate_relic_state_matches_owned_relic_keys():
    state = dict(counter=3, Mode='On')
    r = dict(obs=dict(relics=[dict(id='RELIC.A', d=state)]),
             cands=[dict(k='option', relic='RELIC.A', relic_d=state),
                    dict(k='option', relic='RELIC.B', relic_d=state)])
    a = Encoder(Caps(enc_rev=1)).encode(r)
    owned = np.flatnonzero(a['tok_type'] == TT['RELIC'])[0]
    expected = {k2('relic.d', 'RELIC.A/counter'), k2('relic.d', 'RELIC.A/Mode=On')}
    assert expected <= keys(a['tok_key'][owned])
    assert expected <= keys(a['cand_key'][0])
    assert not expected.intersection(keys(a['cand_key'][1]))
    assert k2('relic.d', 'RELIC.B/counter') in keys(a['cand_key'][1])


def test_power_amounts_precede_move_and_state_entries():
    e = dict(id='GIANT', move='A', then='B', prev=['C', 'D', 'E'], done=dict(A=3), d=dict(count=2),
             pw=[dict(id=f'POWER{i}', n=i) for i in range(5)])
    entries = creature_ents(e, enc_rev=1)
    indices = {key: i for i, (key, _, _) in enumerate(entries)}
    assert max(indices[k2('pw', f'POWER{i}')] for i in range(5)) < indices[k2('cr.move', 'GIANT/A')]
    assert max(indices[k2('pw', f'POWER{i}')] for i in range(5)) < indices[k2('cr.d', 'GIANT/count')]
    a = Encoder(Caps(enc_rev=1, max_ent=11)).encode(dict(obs=dict(combat=dict(enemies=[e]))))
    token = np.flatnonzero(a['tok_type'] == TT['ENEMY'])[0]
    assert {k2('pw', f'POWER{i}') for i in range(5)} <= keys(a['tok_key'][token])


def test_rev2_arrows_status_counts_and_impaled_enemies():
    combat = dict(player=dict(id='PLAYER', hp=70), enemies=[dict(id='A', hp=10), dict(id='B', hp=20)],
                  arrows_available=1, arrows=[dict(slot=0, status='Available', destroy_after_impaled=False, tgt=-1),
                                              dict(slot=1, status='Impaled', destroy_after_impaled=True, tgt=1),
                                              dict(slot=2, status='Impaled', destroy_after_impaled=False, tgt=1)])
    r = dict(obs=dict(combat=combat))
    a = Encoder(Caps(enc_rev=1)).encode(r)
    b = Encoder(Caps(enc_rev=2)).encode(r)
    cb_a = np.flatnonzero(a['tok_type'] == TT['COMBAT'])[0]
    cb_b = np.flatnonzero(b['tok_type'] == TT['COMBAT'])[0]
    assert k1('cb.arrows.len') in keys(a['tok_key'][cb_a]) and k2('cb.arrow', 'Impaled') not in keys(a['tok_key'][cb_a])
    kb = keys(b['tok_key'][cb_b])
    assert {k2('cb.arrow', 'Available'), k2('cb.arrow', 'Impaled'), k2('cb.arrow_destroy', 'Impaled')} <= kb
    en = np.flatnonzero(b['tok_type'] == TT['ENEMY'])
    assert k1('cr.arrows_in') not in keys(b['tok_key'][en[0]]) and k1('cr.arrows_in') in keys(b['tok_key'][en[1]])
    en_a = np.flatnonzero(a['tok_type'] == TT['ENEMY'])
    assert all(k1('cr.arrows_in') not in keys(a['tok_key'][t]) for t in en_a)
    # without arrows, rev 2 encodes exactly like rev 1
    plain = dict(obs=dict(combat=dict(player=dict(id='PLAYER', hp=70), enemies=[dict(id='A', hp=10)])))
    pa, pb = Encoder(Caps(enc_rev=1)).encode(plain), Encoder(Caps(enc_rev=2)).encode(plain)
    for k in pa:
        assert np.array_equal(pa[k], pb[k]), k
