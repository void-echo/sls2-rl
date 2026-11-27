#!/usr/bin/env python3
"""Play runs on the headless v1 engine (a private copy, see headless.py) and record, per run:

  <out>/<run_id>/headless.jsonl   one line per decision point: {step, decision, obs, cands, info}
  <out>/<run_id>/actions.jsonl    the action log replayed by the client: {step, decision, i, kind_pos, cand}
  <out>/<run_id>/run.json         run spec (character, ascension, seed, policy, end, decision counts)

The action log is the sequence of chosen candidate descriptors (`cands[i]` dicts) plus the decision kind, which
the ParityOracle mod matches against candidates it computes from the real client's state.

Policies are test drivers for coverage, not agents:
  random   biased random legal choice (plays cards more than ending the turn, takes rewards more than skipping,
           uses / discards potions now and then in every decision)
  greedy   a simple survival heuristic (block when attacked, hit the weakest enemy, rest when hurt, avoid elites
           when low)
  search   greedy outside combat; each player turn is the best of --rollouts randomized greedy rollouts, played out in
           scratch engines from a replay of the action prefix (the engine is deterministic), so runs reach acts 2-3
  model    the trained agent (--ckpt), argmax: records a run it plays so the client can show it
  nav      random, but exercising the navigation rules (engine/RL_PROTOCOL.md, "No navigation cycles"): opens card
           and removal rewards, rerolls, uses potions on those sub-screens (none in combat), then backs out or takes;
           prefers Cook, then Smith, and the merchant's removal (using potions inside their selections)
--play-first CARD.X plays that card whenever it is playable (any policy), e.g. Forbidden Grimoire for its removal
reward; --play-first POTION.X uses that potion in combat whenever it can (Ambergris for an extra turn).

Usage: record.py --out tmp/parity/runs/v1a --runs 10 [--characters Ironclad,Silent] [--ascension 0,10] [--seeds a,b,c]
                 [--policy random|greedy|search|nav|model --ckpt X] [--prefer-rooms Shop,Treasure] [--hoard-potions] [--max-steps 400]
                 [--setup '<json list of scenario commands>'] [--play-first CARD.X|POTION.X]
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from headless import Engine, DEFAULT_ENGINE  # noqa: E402

CHARACTERS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder", "Fool"]
POTION_KINDS = ("potion", "discard_potion")


def _split(cands):
    own = [i for i, c in enumerate(cands) if c["k"] not in POTION_KINDS]
    use = [i for i, c in enumerate(cands) if c["k"] == "potion"]
    disc = [i for i, c in enumerate(cands) if c["k"] == "discard_potion"]
    return own, use, disc


def _of(cands, idx, *kinds):
    return [i for i in idx if cands[i]["k"] in kinds]


def _map_pref(r, rng, idx, prefer):
    cands, nodes = r["cands"], r["obs"]["map"]["nodes"]
    good = [i for i in _of(cands, idx, "map") if 0 <= cands[i]["node"] < len(nodes) and nodes[cands[i]["node"]]["t"] in prefer]
    return rng.choice(good) if good else None


def choose_random(r, rng, spec):
    cands, dec = r["cands"], r["decision"]
    own, use, disc = _split(cands)
    hoard = spec.get("hoard_potions")
    if dec == "combat_play":
        plays = _of(cands, own, "play")
        ends = _of(cands, own, "end_turn")
        x = rng.random()
        if plays and x < 0.80:
            return rng.choice(plays)
        if use and not hoard and x < 0.88:
            return rng.choice(use)
        if disc and not hoard and x < 0.90:
            return rng.choice(disc)
        return ends[0] if ends else rng.choice(range(len(cands)))
    # outside combat play: the belt is usable on every screen; exercise it rarely
    if not hoard and use and rng.random() < 0.04:
        return rng.choice(use)
    if not hoard and disc and rng.random() < 0.02:
        return rng.choice(disc)
    if not own:
        return rng.choice(range(len(cands)))
    if dec == "map_select" and spec.get("prefer_rooms"):
        i = _map_pref(r, rng, own, spec["prefer_rooms"])
        if i is not None and rng.random() < 0.9:
            return i
    if dec == "rewards":
        takes = _of(cands, own, "reward")
        if takes and rng.random() < 0.85:
            return rng.choice(takes)
        pro = _of(cands, own, "proceed")
        return pro[0] if pro else rng.choice(own)
    if dec == "card_reward":
        cards = _of(cands, own, "reward_card")
        alts = _of(cands, own, "reroll", "sacrifice", "alt")
        x = rng.random()
        if alts and x < 0.25:
            return rng.choice(alts)
        if cards and x < 0.80:
            return rng.choice(cards)
        return rng.choice(own)
    if dec == "shop":
        buys = [i for i in own if cands[i]["k"] != "leave"]
        leave = _of(cands, own, "leave")
        if buys and rng.random() < 0.55:
            return rng.choice(buys)
        return leave[0] if leave else rng.choice(own)
    if dec == "treasure":
        take = _of(cands, own, "open", "treasure")
        if take and rng.random() < 0.85:
            return rng.choice(take)
    return rng.choice(own)


def _incoming(obs):
    cb = obs.get("combat") or {}
    tot = 0
    for e in cb.get("enemies", []):
        for it in e.get("intents", []):
            if it.get("t", "").startswith("Attack") or it.get("dmg", 0) > 0:
                tot += it.get("dmg", 0) * max(1, it.get("hits", 1))
    return tot


def card_value(c):
    """Rough worth of a card for the greedy driver's deck building (higher is better)."""
    if c["type"] in ("Curse", "Status"):
        return -10
    if c["rar"] == "Basic":
        return 1 + (3 if c["id"].endswith("BASH") else 0) - (0 if c["up"] == 0 else -2)
    v = c["v"]
    cost = max(1, c["cost"]) if not c["x"] else 2
    dmg = v.get("Damage", v.get("CalculatedDamage", 0)) * max(1, v.get("Repeat", 1))
    val = (dmg + 0.9 * v.get("Block", 0)) / cost
    if c["type"] == "Power":
        val = max(val, 9)
    val += {"Rare": 3, "Uncommon": 1.5}.get(c["rar"], 0) + 2 * c["up"]
    return val


def choose_greedy(r, rng, spec):
    cands, dec, obs = r["cands"], r["decision"], r["obs"]
    own, use, disc = _split(cands)
    if dec != "combat_play" and spec.get("explore", 0) > 0 and rng.random() < spec["explore"]:
        return rng.choice(range(len(cands)))   # coverage: any legal input, potions included
    g = obs["g"]
    hp_frac = g["hp"] / max(1, g["max_hp"])
    if dec == "combat_play":
        cb = obs["combat"]
        hand = cb["hand"]
        enemies = cb["enemies"]
        plays = _of(cands, own, "play")
        incoming = _incoming(obs)
        block = cb["player"]["block"]
        if use and not spec.get("hoard_potions") and (hp_frac < 0.4 or (g["room"] in ("Elite", "Boss") and cb["round"] == 1)) and rng.random() < 0.7:
            return rng.choice(use)

        def card(i):
            return hand[cands[i]["hand"]]

        blockers = [i for i in plays if "Block" in card(i)["v"] and card(i)["type"] == "Skill"]
        if blockers and incoming > block:
            return max(blockers, key=lambda i: card(i)["v"].get("Block", 0))
        powers = [i for i in plays if card(i)["type"] == "Power"]
        if powers:
            return rng.choice(powers)
        attacks = [i for i in plays if card(i)["type"] == "Attack"]
        if attacks:
            def score(i):
                t = cands[i]["tgt"]
                ehp = enemies[t]["hp"] + enemies[t]["block"] if 0 <= t < len(enemies) else 999
                return (card(i)["pv"].get("Damage", card(i)["v"].get("Damage", 0)), -ehp)
            return max(attacks, key=score)
        others = [i for i in plays if card(i)["type"] not in ("Curse", "Status")]
        if others:
            return rng.choice(others)
        ends = _of(cands, own, "end_turn")
        return ends[0] if ends else rng.choice(range(len(cands)))
    if not own:
        return rng.choice(range(len(cands)))
    if dec == "map_select":
        nodes = obs["map"]["nodes"]
        maps = _of(cands, own, "map")
        if hp_frac < 0.5:
            order = ["RestSite", "Shop", "Treasure", "Unknown", "Monster", "Elite"]
        elif hp_frac < 0.75:
            order = ["Monster", "Unknown", "RestSite", "Treasure", "Shop", "Elite"]
        else:
            order = ["Elite", "Monster", "Unknown", "Treasure", "RestSite", "Shop"]
        if spec.get("prefer_rooms"):
            order = list(spec["prefer_rooms"]) + order
        def rank(i):
            t = nodes[cands[i]["node"]]["t"] if 0 <= cands[i]["node"] < len(nodes) else "?"
            return order.index(t) if t in order else len(order)
        best = min(rank(i) for i in maps)
        return rng.choice([i for i in maps if rank(i) == best])
    if dec == "rest_site":
        rests = _of(cands, own, "rest")
        keys = {i: cands[i]["key"].upper() for i in rests}
        heal = [i for i in rests if "HEAL" in keys[i] or keys[i] == "REST"]
        smith = [i for i in rests if "SMITH" in keys[i]]
        if heal and hp_frac < 0.6:
            return heal[0]
        if smith:
            return smith[0]
        if rests:
            return rng.choice(rests)
        return own[0]
    if dec == "rewards":
        takes = _of(cands, own, "reward")
        if takes:
            return takes[0]
        return _of(cands, own, "proceed")[0] if _of(cands, own, "proceed") else own[0]
    if dec == "card_reward":
        cards = _of(cands, own, "reward_card")
        alts = _of(cands, own, "reroll", "sacrifice", "alt")
        if alts and spec.get("explore", 0) > 0 and rng.random() < 0.5:
            return rng.choice(alts)
        if cards:
            best = max(cards, key=lambda i: card_value(cands[i]["card"]) + rng.random())
            if card_value(cands[best]["card"]) >= 6 or len(obs["deck"]) < 14:
                return best
        skip = _of(cands, own, "skip")
        return skip[0] if skip else rng.choice(own)
    if dec == "shop":
        rem = _of(cands, own, "remove")
        if rem:
            return rem[0]
        buys = [i for i in own if cands[i]["k"] in ("buy_card", "buy_relic", "buy_potion")]
        if buys and rng.random() < 0.4:
            return rng.choice(buys)
        leave = _of(cands, own, "leave")
        return leave[0] if leave else rng.choice(own)
    if dec == "treasure":
        take = _of(cands, own, "open", "treasure")
        if take:
            return take[0]
    if dec == "card_select":
        picks = _of(cands, own, "pick")
        conf = _of(cands, own, "confirm")
        ctx = obs["ctx"]
        sel = ctx.get("sel")
        if picks and sel in ("deck_remove", "deck_transform"):
            return min(picks, key=lambda i: (card_value(cands[i]["card"]), rng.random()))
        if picks and sel in ("deck_upgrade", "hand_upgrade"):
            return max(picks, key=lambda i: (card_value(cands[i]["card"]), rng.random()))
        if conf and len(ctx.get("picked", [])) >= ctx.get("min", 0) and rng.random() < 0.5:
            return conf[0]
        if picks:
            return rng.choice(picks)
    return rng.choice(own)


def choose_nav(r, rng, spec):
    """Coverage driver for the navigation rules: every back-out the engine still offers (card reward Skip, removal
    cancel) is taken often, both before and after something changed on the sub-screen (Driftwood's REROLL, a potion),
    so declines, held-back rewards and their release (a reward taken, a potion on the rewards screen) all occur."""
    cands, dec = r["cands"], r["decision"]
    own, use, disc = _split(cands)
    if dec == "combat_play":   # potions are kept for the reward and selection screens
        return choose_random(r, rng, dict(spec, hoard_potions=True))
    if dec == "rewards":
        opens = [i for i in own if cands[i]["k"] == "reward" and cands[i].get("rt") in ("Card", "RemoveCard")]
        if use and rng.random() < 0.15:
            return rng.choice(use)
        if opens and rng.random() < 0.75:
            return rng.choice(opens)
        takes = _of(cands, own, "reward")
        if takes and rng.random() < 0.7:
            return rng.choice(takes)
        pro = _of(cands, own, "proceed")
        return pro[0] if pro else rng.choice(own or list(range(len(cands))))
    if dec == "card_reward":
        rer = _of(cands, own, "reroll")
        if rer and rng.random() < 0.7:
            return rer[0]
        if use and rng.random() < 0.3:
            return rng.choice(use)
        skip = _of(cands, own, "skip")
        if skip and rng.random() < 0.5:
            return skip[0]
        cards = _of(cands, own, "reward_card")
        return rng.choice(cards or own or list(range(len(cands))))
    if dec == "card_select":
        if use and rng.random() < 0.25:
            return rng.choice(use)
        skip = _of(cands, own, "skip")
        if skip and rng.random() < 0.5:
            return skip[0]
        return choose_random(r, rng, spec)
    if dec == "rest_site":
        cook = [i for i in own if cands[i]["k"] == "rest" and cands[i]["key"].upper() == "COOK"]
        smith = [i for i in own if cands[i]["k"] == "rest" and cands[i]["key"].upper() == "SMITH"]
        if (cook or smith) and rng.random() < 0.85:
            return (cook or smith)[0]
    if dec == "shop":
        rem = _of(cands, own, "remove")
        if rem and rng.random() < 0.85:
            return rem[0]
    return choose_random(r, rng, spec)


def with_play_first(choose):
    """spec["play_first"]: play that card id (CARD.X) or use that potion (POTION.X) in combat whenever a candidate for
    it exists (scenario coverage knob)."""
    def f(r, rng, spec):
        want = spec.get("play_first")
        if want and r["decision"] == "combat_play":
            if want.startswith("POTION."):
                slots = {p["slot"]: p["id"] for p in r["obs"]["potions"]}
                mine = [i for i, c in enumerate(r["cands"]) if c["k"] == "potion" and slots.get(c["slot"]) == want]
            else:
                hand = r["obs"]["combat"]["hand"]
                mine = [i for i, c in enumerate(r["cands"]) if c["k"] == "play" and hand[c["hand"]]["id"] == want]
            if mine:
                return mine[0]
        return choose(r, rng, spec)
    return f


def choose_greedy_eps(r, rng, spec, eps=0.15):
    """Greedy with some random combat moves: the rollout policy of the search driver."""
    if r["decision"] == "combat_play" and rng.random() < eps:
        own = [i for i, c in enumerate(r["cands"]) if c["k"] in ("play", "end_turn")]
        if own:
            return rng.choice(own)
    return choose_greedy(r, rng, spec)


_MODELS = {}


def choose_model(r, rng, spec):
    """The trained agent itself (checkpoint spec["ckpt"], argmax), so the client can replay a run it played."""
    m = _MODELS.get(spec["ckpt"])
    if m is None:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import torch
        from rl.encode import Encoder
        from rl.ppo import to_torch
        from rl.train import load_policy
        model, _, cfg, _ = load_policy(spec["ckpt"], torch.device("cpu"))
        model.eval()
        m = _MODELS[spec["ckpt"]] = (model, Encoder(cfg.caps), to_torch, torch)
    model, enc, to_torch, torch = m
    batch = enc.alloc(1)
    o = enc.encode(r)
    for k in batch:
        batch[k][0] = o[k]
    with torch.no_grad():
        logits, _ = model(to_torch(batch, torch.device("cpu")))
    return int(logits[0, : len(r["cands"])].float().argmax())


def choose_actions(r, rng, spec):
    """Replay a logged game: spec["actions"] is the list of candidate indices (rl.evaluate's episode "actions" or a
    line of a training run's actions/*.jsonl.gz). No model, so the replay is exact on any machine."""
    i = spec["_next"] = spec.get("_next", -1) + 1
    if i >= len(spec["actions"]):
        raise RuntimeError(f"{spec['id']}: the action log ends at step {i} but the game goes on")
    return spec["actions"][i]


POLICIES = {"random": choose_random, "greedy": choose_greedy, "search": choose_greedy, "nav": choose_nav,
            "model": choose_model, "actions": choose_actions}


class TurnSearch:
    """Search test driver (coverage only, not an agent). The engine is deterministic given seed + actions, so a
    state can be re-created by replaying the action prefix. At the first decision of each player turn, K scratch
    engines (in parallel threads) replay the prefix and play the rest of the turn with a randomized greedy policy
    up to the next turn (or the end of combat); the main run then follows the best rollout (alive first, then
    combat won, then HP minus remaining enemy HP)."""

    def __init__(self, engine_dir, log_dir, k):
        from concurrent.futures import ThreadPoolExecutor
        self.engines = [Engine(engine_dir, log_path=os.path.join(log_dir, f"search_{j}.log")) for j in range(k)]
        self.pool = ThreadPoolExecutor(max_workers=k)

    def _rollout(self, eng, spec, prefix, seed, eps):
        r = start_with_setup(eng, spec)
        for i in prefix:
            r = eng.act(i)
        if r.get("type") != "rl" or r["decision"] != "combat_play":
            return None, None
        rr = random.Random(seed)
        roll, ended = [], False
        while r.get("type") == "rl" and not r["done"] and len(roll) < 300:
            if r["obs"].get("combat") is None:
                break
            if ended and r["decision"] == "combat_play":
                break
            i = choose_greedy_eps(r, rr, spec, eps)
            if r["cands"][i]["k"] == "end_turn":
                ended = True
            roll.append(i)
            r = eng.act(i)
        if r.get("type") != "rl":
            return None, None
        g = r["obs"]["g"]
        if r["done"]:
            return roll, (0, 0, 0)
        cb = r["obs"].get("combat")
        if cb is None:
            return roll, (1, 1, g["hp"])
        foes = sum(e["hp"] + e["block"] for e in cb["enemies"])
        return roll, (1, 0, 2 * g["hp"] + cb["player"]["block"] - foes)

    def plan(self, spec, prefix, rng, room=None):
        seeds = [rng.random() for _ in self.engines]
        rounds = [seeds]
        if room in ("Boss", "Elite") and spec.get("boss_rounds", 0) > 0:
            # extra rollouts for hard fights, seeded separately so the trajectory before them does not change
            extra = random.Random(len(prefix) * 7919 + 17)
            rounds += [[extra.random() for _ in self.engines] for _ in range(spec["boss_rounds"])]
        best, best_score = None, None
        for rnd, sd in enumerate(rounds):
            futs = [self.pool.submit(self._rollout, e, spec, list(prefix), sd[j], 0.0 if (j == 0 and rnd == 0) else 0.35)
                    for j, e in enumerate(self.engines)]
            for f in futs:
                roll, score = f.result()
                if roll is not None and (best_score is None or score > best_score):
                    best, best_score = roll, score
        return best

    def close(self):
        self.pool.shutdown()
        for e in self.engines:
            e.close()


def start_with_setup(eng, spec):
    """rl_start, then the run's scenario setup commands (engine/RL_PROTOCOL.md scenario commands; the mod performs
    the same game calls at the same point in the client)."""
    r = eng.start(spec["character"], spec["ascension"], spec["seed"])
    for cmd in spec.get("setup") or []:
        r = eng.send(cmd)
    return r


def kind_pos(cands, i):
    k = cands[i]["k"]
    return sum(1 for c in cands[:i] if c["k"] == k)


def record_run(eng, spec, out_dir, search=None):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(spec["policy_seed"])
    choose = with_play_first(POLICIES[spec.get("policy", "random")])
    taken, plan = [], []
    hl = open(os.path.join(out_dir, "headless.jsonl"), "w")
    al = open(os.path.join(out_dir, "actions.jsonl"), "w")
    r = start_with_setup(eng, spec)
    end = "max_steps"
    step = 0
    decisions = {}
    for step in range(spec["max_steps"] + 1):
        if r.get("type") != "rl":
            hl.write(json.dumps({"step": step, "error": r.get("message", "")[:4000]}) + "\n")
            end = "engine_error"
            break
        hl.write(json.dumps({"step": step, "decision": r["decision"], "obs": r["obs"], "cands": r["cands"],
                             "info": r["info"]}) + "\n")
        decisions[r["decision"]] = decisions.get(r["decision"], 0) + 1
        if r["done"]:
            end = "victory" if r["victory"] else ("abandoned" if r["info"].get("abandoned") else "game_over")
            break
        if step == spec["max_steps"]:
            break
        combat_now = r["obs"].get("combat") is not None
        if search is not None and combat_now and not plan and r["decision"] == "combat_play":
            plan = search.plan(spec, taken, rng, r["obs"]["g"]["room"]) or []
        if plan and combat_now:
            i = plan.pop(0)
            if i >= len(r["cands"]):  # the replay did not follow the rollout: should never happen (determinism)
                print(f"{spec['id']}: rollout diverged at step {step}")
                plan, i = [], choose(r, rng, spec)
        else:
            plan = []
            i = choose(r, rng, spec)
        taken.append(i)
        al.write(json.dumps({"step": step, "decision": r["decision"], "i": i, "kind_pos": kind_pos(r["cands"], i),
                             "cand": r["cands"][i]}) + "\n")
        r = eng.act(i)
    hl.close()
    al.close()
    last = r.get("info", {}) if r.get("type") == "rl" else {}
    spec = dict(spec, end=end, steps=step, decisions=decisions, act=last.get("act"), floor=last.get("total_floor"))
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(spec, f, indent=1)
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--characters", default=",".join(CHARACTERS))
    ap.add_argument("--ascension", default="0", help="comma list, cycled per run (e.g. 0,10)")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed-prefix", default="par")
    ap.add_argument("--start", type=int, default=0, help="first run number (seed suffix)")
    ap.add_argument("--seeds", default="", help="comma list of exact seed strings (e.g. eval_62,eval_123 from rl.evaluate); "
                    "overrides --runs/--seed-prefix/--start")
    ap.add_argument("--policy", default="random", choices=sorted(POLICIES))
    ap.add_argument("--prefer-rooms", default="", help="comma list of map node types, e.g. Shop,RestSite,Unknown,Treasure")
    ap.add_argument("--hoard-potions", action="store_true", help="never use or discard potions (fills the belt)")
    ap.add_argument("--explore", type=float, default=0.0, help="greedy/search: chance of a uniformly random legal input outside combat")
    ap.add_argument("--setup", default="", help="JSON list of scenario commands sent after rl_start, e.g. "
                    '\'[{"cmd":"obtain_relics","ids":["DRIFTWOOD"]}]\'')
    ap.add_argument("--play-first", default="", help="card id played whenever playable (CARD.FORBIDDEN_GRIMOIRE) or potion used in combat (POTION.AMBERGRIS)")
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--fresh", action="store_true", help="new engine process per run")
    ap.add_argument("--rollouts", type=int, default=8, help="search policy: parallel rollouts per player turn")
    ap.add_argument("--boss-rounds", type=int, default=0, help="search policy: extra rollout rounds on elite/boss turns")
    ap.add_argument("--actions-from", default="", help="actions policy: an rl.evaluate / eval_variant JSON whose episodes carry "
                    "\"actions\"; the runs are the --seeds found in it (character taken from the file)")
    ap.add_argument("--ckpt", default="", help="model policy: checkpoint of the trained agent (its engine must be the one "
                    "it was trained on)")
    a = ap.parse_args()
    chars = a.characters.split(",")
    ascs = [int(x) for x in a.ascension.split(",")]
    prefer = [x for x in a.prefer_rooms.split(",") if x]
    os.makedirs(a.out, exist_ok=True)
    eng = None
    seeds = [x for x in a.seeds.split(",") if x]
    logged = {}
    if a.actions_from:
        logged = {e["seed"]: e for e in json.load(open(a.actions_from))["episodes"] if e.get("actions")}
        a.policy = "actions"
        missing = [x for x in seeds if x not in logged]
        if missing:
            raise SystemExit(f"no logged actions for {missing}")
    for k in range(len(seeds) if seeds else a.runs):
        n = a.start + k
        ch = chars[k % len(chars)]
        if logged:
            ch = logged[seeds[k]]["char"]
        asc = ascs[k % len(ascs)]
        seed = seeds[k] if seeds else f"{a.seed_prefix}{n:03d}"
        rid = f"{seed}_{ch}_A{asc}"
        spec = {"id": rid, "character": ch, "ascension": asc, "seed": seed,
                "policy": a.policy, "policy_seed": 1000 + n, "max_steps": a.max_steps, "prefer_rooms": prefer,
                "hoard_potions": a.hoard_potions, "boss_rounds": a.boss_rounds, "explore": a.explore,
                "setup": json.loads(a.setup) if a.setup else [], "play_first": a.play_first or None,
                "ckpt": os.path.abspath(a.ckpt) if a.ckpt else None}
        if logged:
            spec["actions"] = logged[seed]["actions"]
            spec["max_steps"] = max(spec["max_steps"], len(spec["actions"]) + 5)
        if eng is None or a.fresh:
            if eng is not None:
                eng.close()
            eng = Engine(a.engine, log_path=os.path.join(a.out, f"engine_{n:03d}.log"))
        search = TurnSearch(a.engine, a.out, a.rollouts) if a.policy == "search" else None
        try:
            res = record_run(eng, spec, os.path.join(a.out, rid), search)
        except Exception as ex:  # engine died: restart it for the next run
            print(f"{rid}: engine failure {ex}")
            eng.close()
            eng = None
            continue
        finally:
            if search is not None:
                search.close()
        print(f"{rid}: {res['end']} after {res['steps']} steps, act {res['act']} floor {res['floor']}; {res['decisions']}")
    if eng is not None:
        eng.close()


if __name__ == "__main__":
    main()
