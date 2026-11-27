"""Unit tests: protocol/fake engine, encoder, engine subprocess handling, GAE, model invariances.
Run from the repo root:  python -m pytest -q rl/tests"""
import collections
import copy
import random
import sys

import numpy as np
import pytest
import torch

from rl.encode import TT, Caps, Encoder, hkey
from rl.engine import EngineDead, EngineProcess, FakeEngine, make_engine
from rl.fake_engine import FakeServer
from rl.model import ModelCfg, Policy
from rl.ppo import Rollout, sliced, to_torch
from rl.protocol import DECISIONS, check_response


def random_responses(n_eps=40, seed=0, chars=("Ironclad", "Defect", "Necrobinder")):
    srv, rng, out = FakeServer(), random.Random(seed), []
    for ep in range(n_eps):
        R = srv.handle({"cmd": "rl_start", "character": chars[ep % len(chars)], "seed": f"t{ep}"})
        while True:
            out.append(R)
            if R["done"]:
                break
            R = srv.handle({"cmd": "rl_act", "i": rng.randrange(len(R["cands"]))})
    return out


RS = random_responses()


def test_fake_schema_and_coverage():
    kinds = collections.Counter(R["decision"] for R in RS)
    for R in RS:
        assert check_response(R) == [], check_response(R)
    missing = set(DECISIONS) - set(kinds)
    assert not missing, f"fake never produced {missing}"
    assert any(len(R["cands"]) > 30 for R in RS)  # crystal sphere / big combat


def test_encoder_shapes_and_refs():
    enc = Encoder()
    for R in RS:
        if R["done"]:
            continue
        o = enc.encode(R)
        n, c = int(o["n_tok"]), int(o["n_cand"])
        assert o["tok_type"][0] == TT["GLOBAL"] and (o["tok_type"][:n] > 0).all() and (o["tok_type"][n:] == 0).all()
        assert c == len(R["cands"])
        assert np.isfinite(o["tok_val"].astype(np.float32)).all()
        for j, cd in enumerate(R["cands"]):
            ref = o["cand_ref"][j]
            if cd["k"] == "play":
                assert o["tok_type"][ref[0]] == TT["HAND"]
                assert (ref[1] >= 0) == (cd["tgt"] >= 0)
                if cd["tgt"] >= 0:
                    assert o["tok_type"][ref[1]] == TT["ENEMY"]
            if cd["k"] == "map":
                assert o["tok_type"][ref[0]] == TT["MAPNODE"]
            if cd["k"] in ("reward_card", "pick", "buy_card"):
                assert o["tok_type"][ref[0]] == TT["OFFER_CARD"]
            if cd["k"] in ("potion", "discard_potion"):
                assert o["tok_type"][ref[0]] == TT["POTION"]
            if cd["k"] == "bundle":
                assert (o["tok_type"][ref[2:2 + len(cd["cards"])]] == TT["OFFER_CARD"]).all()
    assert not enc.trunc


def test_encoder_robust_to_unknown_content_and_caps():
    R = copy.deepcopy(next(R for R in RS if R["decision"] == "combat_play"))
    R["decision"] = "brand_new_kind"
    R["obs"]["combat"]["hand"][0]["new_field"] = {"x": 3, "y": "z"}
    R["obs"]["combat"]["enemies"][0]["pw"] += [{"id": f"POWER.P{i}", "n": i} for i in range(40)]
    R["obs"]["deck"] = R["obs"]["deck"] * 20
    R["obs"]["combat"]["draw"] = R["obs"]["combat"]["draw"] * 20
    R["cands"] = R["cands"] + [{"k": "mystery", "foo": 1.5, "bar": "baz"}] * 300
    enc = Encoder(Caps(deck_in_combat=True, deck=80, draw=64))  # group caps that fit within max_tok together
    o = enc.encode(R)
    assert int(o["n_cand"]) == enc.c.max_cand
    assert (o["tok_type"] == TT["DECK"]).sum() == enc.c.deck and (o["tok_type"] == TT["DRAW"]).sum() == enc.c.draw
    assert enc.trunc["cands"] and enc.trunc["ent"] and enc.trunc["deck"] and enc.trunc["draw"]
    enc = Encoder(Caps(deck_in_combat=True, max_tok=100))
    o = enc.encode(R)
    assert int(o["n_tok"]) == 100 and enc.trunc["tokens"]
    # referenced entities (hand, enemies) were added before the truncated piles
    assert o["tok_type"][o["cand_ref"][0][0]] == TT["HAND"]


def test_hash_stable():
    # fixed across processes and Python versions (blake2b, not hash())
    assert hkey("card.id=CARD.BASH") == hkey("card.id=CARD.BASH")
    assert hkey("a") != hkey("b")
    h = hkey("card.id=CARD.BASH")
    assert h >> 16 and h & 0xFFFF


def test_fake_engine_inprocess_interface():
    eng = FakeEngine(p_error=0.0)
    eng.start()
    assert eng.poll()["type"] == "ready"
    R = eng.request({"cmd": "rl_start", "character": "Silent", "seed": "x"})
    assert R["type"] == "rl" and R["cands"]
    R = eng.step(0)
    assert R["type"] == "rl"


def test_engine_process_subprocess_logs_errors_crash_timeout():
    eng = make_engine("fakeproc", fake_args=["--log_every", "2"])
    eng.start()
    assert eng.wait_ready(30)["type"] == "ready"
    R = eng.request({"cmd": "rl_start", "seed": "a"}, timeout=30)
    for _ in range(20):  # log lines interleaved with JSON must be skipped
        R = eng.step(len(R["cands"]) - 1, timeout=30)
        if R["done"]:
            break
    assert R["type"] == "rl" and eng.log_tail
    assert eng.request({"cmd": "rl_act", "i": 99999}, timeout=30)["type"] == "error"
    eng.close()
    crash = make_engine("fakeproc", fake_args=["--p_crash", "1.0"])
    crash.start()
    crash.wait_ready(30)
    with pytest.raises(EngineDead):
        crash.request({"cmd": "rl_start", "seed": "a"}, timeout=30)
    crash.close()
    hang = make_engine("fakeproc", fake_args=["--p_hang", "1.0"])
    hang.start()
    hang.wait_ready(30)
    with pytest.raises(EngineDead, match="timeout"):
        hang.request({"cmd": "rl_start", "seed": "a"}, timeout=1.0)
    hang.close()
    assert not hang.alive


def _naive_gae(r, v, done, trunc, boot, g, lam):
    """Reference: walk back with explicit episode/segment handling."""
    n = len(r)
    adv = np.zeros(n)
    nv, na = boot, 0.0
    for t in reversed(range(n)):
        if trunc[t]:
            adv[t], nv, na = 0.0, v[t], 0.0
            continue
        if done[t]:
            nv, na = 0.0, 0.0
        d = r[t] + g * nv - v[t]
        adv[t] = d + g * lam * na
        nv, na = v[t], adv[t]
    return adv


def test_gae_matches_reference():
    rng = np.random.default_rng(0)
    n_env, caps = 3, Caps(max_tok=4, max_ent=2, max_cand=2, max_cent=2, max_ref=2)
    buf = Rollout(caps, 300, n_env, "cpu")
    seqs = {e: [] for e in range(n_env)}
    dummy = {k: torch.zeros((1,) + tuple(v.shape[1:]), dtype=v.dtype) for k, v in buf.obs.items()}
    for step in range(300):
        e = int(rng.integers(n_env))
        i = buf.add([e], dummy, [0], [0.0], [float(rng.normal())])[0]
        d, tr = rng.random() < 0.05, rng.random() < 0.03
        buf.set_result(i, float(rng.normal()), d, tr and not d)
        seqs[e].append(i)
    boot = rng.normal(size=n_env).astype(np.float32)
    adv, ret, mask = buf.gae(boot, 0.99, 0.9)
    for e, ix in seqs.items():
        ref = _naive_gae(buf.rew[ix], buf.val[ix], buf.done[ix], buf.trunc[ix], boot[e], 0.99, 0.9)
        np.testing.assert_allclose(adv[ix], ref, rtol=1e-4, atol=1e-5)
    assert (mask == ~buf.trunc[:300]).all()


def _batch(n=16):
    enc = Encoder()
    arrs = enc.alloc(n)
    obs = [R for R in RS if not R["done"]][:n]
    for i, R in enumerate(obs):
        enc.encode(R, {k: v[i, ...] for k, v in arrs.items()})
    return arrs


def test_model_masks_and_permutation_invariance():
    torch.manual_seed(0)
    m = Policy(ModelCfg(d=32, layers=2, heads=4)).eval()
    arrs = _batch()
    b = sliced(to_torch(arrs, "cpu"), torch.arange(16))
    with torch.no_grad():
        l1, v1 = m(b)
    # garbage in padding must not matter
    b2 = {k: x.clone() for k, x in b.items()}
    n = b2["n_tok"].long()
    pad = torch.arange(b2["tok_key"].shape[1])[None] >= n[:, None]
    b2["tok_key"][pad] = 12345
    b2["tok_type"][pad] = 3
    with torch.no_grad():
        l2, v2 = m(b2)
    assert torch.allclose(l1, l2, atol=1e-4) and torch.allclose(v1, v2, atol=1e-4)
    # permuting non-global tokens (and remapping candidate refs) leaves outputs unchanged
    i = 0
    nt = int(b["n_tok"][i])
    perm = torch.cat([torch.tensor([0]), 1 + torch.randperm(nt - 1)])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(nt)
    b3 = {k: x[i:i + 1].clone() for k, x in b.items()}
    for k in ("tok_type", "tok_key", "tok_val"):
        b3[k][0, :nt] = b[k][i, perm]
    ref = b3["cand_ref"].long()
    b3["cand_ref"] = torch.where(ref >= 0, inv[ref.clamp(min=0)], ref).to(torch.int16)
    with torch.no_grad():
        l3, v3 = m(b3)
    c = int(b["n_cand"][i])
    assert torch.allclose(l1[i, :c], l3[0, :c], atol=1e-4) and torch.allclose(v1[i], v3[0], atol=1e-4)


def test_model_backward_finite():
    m = Policy(ModelCfg(d=32, layers=1, heads=4))
    b = sliced(to_torch(_batch(), "cpu"), torch.arange(16))
    logits, v = m(b)
    cm = torch.arange(logits.shape[1])[None] < b["n_cand"].long()[:, None]
    (torch.log_softmax(logits, -1).masked_fill(~cm, 0).sum() + v.sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_map_path_stats_match_brute_force():
    from rl.encode import _path_stats
    rng = random.Random(3)
    for trial in range(30):
        rows = [rng.randint(1, 4) for _ in range(rng.randint(2, 7))]
        ids, types, edges = [], [], []
        for r, n in enumerate(rows):
            ids.append([len(types) + j for j in range(n)])
            types.extend(rng.choice(["Monster", "Elite", "RestSite", "Shop", "Unknown"]) for _ in range(n))
        for r in range(len(rows) - 1):
            for a in ids[r]:
                for b in rng.sample(ids[r + 1], rng.randint(1, len(ids[r + 1]))):
                    edges.append([a, b])
        pmax, pmin, dist, npaths = _path_stats(types, edges)
        children = collections.defaultdict(list)
        for a, b in edges:
            children[a].append(b)

        def paths(i):  # every path from i to a sink, excluding i
            if not children[i]:
                return [[]]
            return [[c] + p for c in children[i] for p in paths(c)]
        for i in range(len(types)):
            ps = paths(i)
            assert npaths[i] == len(ps)
            for t in set(types):
                counts = [sum(types[j] == t for j in p) for p in ps]
                assert pmax[i].get(t, 0) == max(counts)
                assert pmin[i].get(t, 0) == min(counts)
                firsts = [next((k + 1 for k, j in enumerate(p) if types[j] == t), None) for p in ps]
                reach = [f for f in firsts if f is not None]
                assert dist[i].get(t) == (min(reach) if reach else None)


def test_encoder_map_paths_only_adds_entries():
    Rs = [R for R in random_responses(6) if R["decision"] == "map_select"][:20]
    assert Rs
    off, on = Encoder(Caps(max_ent=48)), Encoder(Caps(max_ent=48, map_paths=True))
    for R in Rs:
        a, b = off.encode(R), on.encode(R)
        assert (a["tok_type"] == b["tok_type"]).all()
        mapnode = a["tok_type"] == TT["MAPNODE"]
        other = ~mapnode
        assert (a["tok_key"][other] == b["tok_key"][other]).all()
        assert ((b["tok_key"][mapnode] != 0).sum(-1) >= (a["tok_key"][mapnode] != 0).sum(-1)).all()


def test_enemy_move_fields_are_keyed_by_monster():
    from rl.encode import creature_ents, k2
    e = {"id": "MONSTER.A", "hp": 10, "max_hp": 10, "block": 0, "alive": True, "move": "BITE", "then": "ROAR",
         "prev": ["ROAR", "BITE"], "intents": [{"t": "StatusCard", "dmg": 0, "hits": 0, "n": 2}],
         "pw": [{"id": "POWER.SWIPE_POWER", "n": 1, "v": {"Gold": 5}, "d": {"x": 1}, "cards": ["CARD.BASH"]}],
         "d": {"isAwake": 1}}
    keys = {k for k, _, _ in creature_ents(e)}
    for k in (k2("cr.move", "MONSTER.A/BITE"), k2("cr.then", "MONSTER.A/ROAR"), k2("cr.prev1", "MONSTER.A/ROAR"),
              k2("cr.prev2", "MONSTER.A/BITE"), k2("pw.v", "POWER.SWIPE_POWER/Gold"), k2("pw.d", "POWER.SWIPE_POWER/x"),
              k2("pw.card", "CARD.BASH"), k2("cr.d", "MONSTER.A/isAwake")):
        assert k in keys
    assert k2("cr.move", "BITE") not in keys


def test_boss_reward_on_act_change_only():
    from rl.actors import progress_reward
    cfg = {"floor_reward": 0.0, "boss_reward": 0.2, "gamma": 0.999}
    ep = {"floor": 1, "act": 1, "shaping": "per_floor"}
    assert progress_reward(cfg, ep, {"act": 1, "total_floor": 5}, False) == 0.0
    assert abs(progress_reward(cfg, ep, {"act": 2, "total_floor": 18}, False) - 0.2) < 1e-9
    assert progress_reward(cfg, ep, {"act": 2, "total_floor": 19}, False) == 0.0
    assert progress_reward(cfg, ep, {}, False) == 0.0  # missing info keeps the act
    cfg = {"floor_reward": 0.01, "gamma": 0.999}       # old configs have no boss_reward
    ep = {"floor": 1, "act": 1, "shaping": "per_floor"}
    assert abs(progress_reward(cfg, ep, {"act": 2, "total_floor": 3}, False) - 0.02) < 1e-9


def test_pile_counts_multiset():
    R = copy.deepcopy(next(R for R in RS if R["decision"] == "combat_play"))
    card = R["obs"]["combat"]["hand"][0]
    other = dict(card, up=card.get("up", 0) + 1)
    R["obs"]["combat"]["draw"] = [card] * 80 + [other] * 3
    enc = Encoder(Caps(pile_counts=True))
    o = enc.encode(R)
    assert (o["tok_type"] == TT["DRAW"]).sum() == 2 and not enc.trunc
    cnt = hkey("card.count")
    vals = sorted(float(o["tok_val"][i][list(o["tok_key"][i]).index(cnt)][1]) for i in np.nonzero(o["tok_type"] == TT["DRAW"])[0])
    assert np.allclose(vals, [3 / 5.0, 5.0], atol=1e-3)  # linear channel: count / scale, clipped at 5 (80 -> 5; symlog keeps the size)
    # the order inside a pile does not matter
    R2 = copy.deepcopy(R)
    R2["obs"]["combat"]["draw"] = [other] * 3 + [card] * 80
    o2 = enc.encode(R2)
    assert sorted(map(tuple, o["tok_key"][o["tok_type"] == TT["DRAW"]])) == sorted(map(tuple, o2["tok_key"][o2["tok_type"] == TT["DRAW"]]))


def test_floor_reward_anneal_per_episode():
    from rl.actors import progress_reward
    cfg = {"floor_reward": 0.01, "boss_reward": 0.2, "gamma": 0.999}
    ep = {"floor": 1, "act": 1, "shaping": "per_floor", "floor_c": 0.0025}  # a quarter of the way down
    assert abs(progress_reward(cfg, ep, {"act": 1, "total_floor": 3}, False) - 0.005) < 1e-12
    ep = {"floor": 1, "act": 1, "shaping": "per_floor", "floor_c": 0.0}
    assert abs(progress_reward(cfg, ep, {"act": 2, "total_floor": 18}, False) - 0.2) < 1e-12


def test_shaping_fade():
    from types import SimpleNamespace
    from rl.actors import _Worker, progress_reward
    fade = lambda steps, **c: _Worker.shaping_fade(SimpleNamespace(cfg=c, env_steps=SimpleNamespace(value=steps)))
    assert fade(10) == 1.0  # off by default
    c = dict(shaping_fade_start=100, shaping_fade_steps=100, shaping_fade_to=0.0)
    assert fade(50, **c) == 1.0 and abs(fade(150, **c) - 0.5) < 1e-12 and fade(500, **c) == 0.0
    assert abs(fade(500, **{**c, "shaping_fade_to": 0.25}) - 0.25) < 1e-12
    # the episode's coefficients win over the configured rewards
    cfg = dict(floor_reward=0.01, boss_reward=0.2, gamma=1.0, shaping="per_floor")
    ep = dict(floor=5, act=1, floor_c=0.0, boss_c=0.0)
    assert progress_reward(cfg, ep, {"act": 2, "total_floor": 18}, False) == 0.0
    ep = dict(floor=5, act=1, floor_c=0.005, boss_c=0.1)
    assert abs(progress_reward(cfg, ep, {"act": 2, "total_floor": 7}, False) - 0.11) < 1e-12
