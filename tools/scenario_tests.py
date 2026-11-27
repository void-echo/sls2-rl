#!/usr/bin/env python3
"""Scenario tests for content that random play rarely reaches (acts 2-3 bosses, the A10 double boss, the victory path)
and for v1 decision semantics (including the navigation rules: no action only leads back to an identical earlier decision). Each test builds its scenario with the debug commands (dbg_enter_act, rl_enter_room,
set_player, obtain_relics, dbg_procure_potion) in a fresh engine and prints PASS/FAIL with evidence.

  python3 tools/scenario_tests.py            # all
  python3 tools/scenario_tests.py a10_double_boss_and_victory kaiser_crab
"""
import os
import random
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_runner import Engine, terminal_problem  # noqa: E402

TESTS = {}


def test(fn):
    TESTS[fn.__name__] = fn
    return fn


class S:
    def __init__(self, eng, char="Ironclad", asc=0, seed="scen"):
        self.eng = eng
        self.rng = random.Random(seed)
        self.r = self.send({"cmd": "rl_start", "character": char, "ascension": asc, "seed": seed})
        self.seen = []

    def send(self, cmd):
        r = self.eng.send(cmd, 120)
        if r.get("type") == "error":
            raise RuntimeError("engine error: " + r.get("message", "")[:800])
        if r.get("type") == "rl":
            self.r = r
        return r

    def act(self, i):
        return self.send({"cmd": "rl_act", "i": i})

    def find(self, pred):
        return [i for i, c in enumerate(self.r["cands"]) if pred(c)]

    def step(self):
        """One policy step: plays attacks/cards, takes rewards, otherwise the first candidate."""
        r, cands = self.r, self.r["cands"]
        d = r["decision"]
        self.seen.append(d)
        if d == "combat_play":
            plays = self.find(lambda c: c["k"] == "play")
            if plays and self.rng.random() < 0.9:
                return self.act(self.rng.choice(plays))
            return self.act(self.find(lambda c: c["k"] == "end_turn")[0])
        if d == "rewards":
            ks = self.find(lambda c: c["k"] == "reward" and c.get("rt") != "Card")
            return self.act(ks[0] if ks else self.find(lambda c: c["k"] == "proceed")[0])
        if d == "card_select":
            ks = self.find(lambda c: c["k"] in ("pick", "confirm"))
            return self.act(ks[0])
        ks = self.find(lambda c: c["k"] not in ("potion", "discard_potion"))
        return self.act(ks[0] if ks else 0)

    def until(self, pred, cap=4000):
        for _ in range(cap):
            if self.r["done"] or pred(self.r):
                return self.r
            self.step()
        raise RuntimeError(f"step cap at {self.r['decision']}")

    def map(self):
        self.until(lambda r: r["decision"] == "map_select")


def at_map(r):
    return r["decision"] == "map_select"


# Ironclad build that kills every boss/elite under the test policy (AoE for Decimillipede, burst for the Insatiable's sandpit)
STRONG = {"hp": 99999, "max_hp": 99999, "deck": ["BLUDGEON"] * 4 + ["WHIRLWIND"] * 4,
          "relics": ["BURNING_BLOOD", "LANTERN", "ICE_CREAM", "AKABEKO", "VAJRA", "BRIMSTONE", "SWORD_OF_JADE"]}


def boss_fight(s, at="boss"):
    s.send({"cmd": "set_player", **STRONG})
    s.send({"cmd": "rl_enter_room", "type": "Boss", "at": at})
    enemies = sorted({e["id"] for e in s.r["obs"]["combat"]["enemies"]}) if s.r["obs"].get("combat") else []
    s.until(lambda r: r["obs"].get("combat") is None)
    return enemies


@test
def act2_bosses(eng_factory):
    """Every act 2 boss (Hive: Knowledge Demon, Kaiser Crab, The Insatiable) is fought to the end and leads to act 3."""
    out = []
    for enc in ("KNOWLEDGE_DEMON_BOSS", "KAISER_CRAB_BOSS", "THE_INSATIABLE_BOSS"):
        eng = eng_factory()
        s = S(eng, seed="a2_" + enc)
        s.map()
        s.send({"cmd": "dbg_enter_act", "act": 2})
        s.map()
        s.send({"cmd": "set_player", **STRONG})
        s.send({"cmd": "rl_enter_room", "type": "Boss", "encounter": enc, "at": "boss"})
        first = s.r["decision"]
        rounds, choices = 0, []
        while not s.r["done"] and s.r["obs"].get("combat") is not None:
            if s.r["decision"] == "card_select":
                choices.append((s.r["obs"]["ctx"].get("min"), any(c["k"] in ("skip", "confirm") for c in s.r["cands"])))
            rounds = max(rounds, s.r["obs"]["combat"]["round"])
            s.step()
        dead = s.r["done"]
        if not dead:
            s.until(lambda r: r["obs"]["g"]["act"] == 3 or at_map(r))
        out.append((enc, first, rounds, dead, s.r["obs"]["g"]["act"], choices[:3]))
        eng.close()
    ok = all(f == "combat_play" and not d and a == 3 for _, f, _, d, a, _ in out)
    kd_choices = [c for e, *_, c in out if e == "KNOWLEDGE_DEMON_BOSS"][0]
    ok = ok and all(m == 1 and not skip for m, skip in kd_choices)
    return ("PASS" if ok else "FAIL"), f"(boss, first decision, rounds, died, act after, card_select (min, skippable)): {out}"


@test
def act3_bosses_and_elites(eng_factory):
    """Act 3 (Glory) bosses and the content the review flagged (Test Subject, Soul Nexus, Decimillipede, Phantasmal Gardeners)."""
    out = []
    for typ, enc, act in (("Boss", "TEST_SUBJECT_BOSS", 3), ("Boss", "QUEEN_BOSS", 3), ("Boss", "AEONGLASS_BOSS", 3),
                          ("Elite", "SOUL_NEXUS_ELITE", 3), ("Elite", "DECIMILLIPEDE_ELITE", 2),
                          ("Elite", "PHANTASMAL_GARDENERS_ELITE", 1)):
        eng = eng_factory()
        s = S(eng, seed="a3_" + enc)
        s.map()
        if act > 1:
            s.send({"cmd": "dbg_enter_act", "act": act})
            s.map()
        s.send({"cmd": "set_player", **STRONG})
        try:
            s.send({"cmd": "rl_enter_room", "type": typ, "encounter": enc})
        except RuntimeError as e:
            out.append((enc, "enter failed: " + str(e)[:80]))
            eng.close()
            continue
        try:
            s.until(lambda r: r["obs"].get("combat") is None)
            out.append((enc, "ok" if not s.r["done"] else "died", s.r["decision"]))
        except RuntimeError as e:
            out.append((enc, "ERR " + str(e)[:200]))
        eng.close()
    ok = all(o[1] == "ok" for o in out)
    return ("PASS" if ok else "FAIL"), str(out)


@test
def a10_double_boss_and_victory(eng_factory):
    """A10 (DoubleBoss): act 3 has a second boss after the first; then The Architect and WinRun -> victory."""
    eng = eng_factory()
    s = S(eng, asc=10, seed="a10db")
    s.map()
    s.send({"cmd": "dbg_enter_act", "act": 3})
    s.map()
    info = eng.send({"cmd": "dbg_info"})
    e1 = boss_fight(s, "boss")
    after1 = s.r["decision"]
    s.until(at_map)
    nodes = s.r["obs"]["map"]["nodes"]
    offered = [nodes[c["node"]]["t"] for c in s.r["cands"] if c["k"] == "map"]
    s.act(s.find(lambda c: c["k"] == "map")[0])
    e2 = sorted({e["id"] for e in s.r["obs"]["combat"]["enemies"]}) if s.r["obs"].get("combat") else []
    s.until(lambda r: r["obs"].get("combat") is None)
    s.until(lambda r: False, cap=200)
    r = s.r
    ok = (info["second_boss"] and offered == ["Boss"] and e2 and e2 != e1 and r["done"] and r["victory"] is True
          and terminal_problem(r) is None and r["obs"]["g"]["act"] == 3)
    eng.close()
    return ("PASS" if ok else "FAIL"), (f"second boss on map: {info['second_boss']}; boss 1 {e1} -> {after1}; offered after it {offered}; "
                                        f"boss 2 {e2}; end: done={r['done']} victory={r['victory']} act={r['obs']['g']['act']} "
                                        f"event={r['obs']['ctx'].get('event')} decisions tail {s.seen[-4:]}")


@test
def a0_single_boss_victory(eng_factory):
    """A0: one act 3 boss, then The Architect -> victory (no second boss node)."""
    eng = eng_factory()
    s = S(eng, asc=0, seed="a0win")
    s.map()
    s.send({"cmd": "dbg_enter_act", "act": 3})
    s.map()
    info = eng.send({"cmd": "dbg_info"})
    boss_fight(s, "boss")
    s.until(lambda r: False, cap=200)
    r = s.r
    eng.close()
    ok = not info["second_boss"] and r["done"] and r["victory"] is True and terminal_problem(r) is None
    return ("PASS" if ok else "FAIL"), f"second boss: {info['second_boss']}; done={r['done']} victory={r['victory']} decisions tail {s.seen[-4:]}"


@test
def lords_parasol_forced_removal(eng_factory):
    """Lord's Parasol buys the whole shop and forces one card removal that cannot be cancelled."""
    eng = eng_factory()
    s = S(eng, seed="parasol")
    s.map()
    s.send({"cmd": "obtain_relics", "ids": ["LORDS_PARASOL"]})
    s.map()
    s.send({"cmd": "rl_enter_room", "type": "Shop"})
    selects = []
    while s.r["decision"] != "shop":
        if s.r["decision"] == "card_select":
            selects.append((s.r["obs"]["ctx"].get("sel"), s.r["obs"]["ctx"].get("cancelable", False),
                            sorted({c["k"] for c in s.r["cands"]})))
        s.step()
    buy = s.find(lambda c: c["k"].startswith("buy_") or c["k"] == "remove")
    eng.close()
    last = selects[-1] if selects else None
    ok = last is not None and last[0] == "deck_remove" and not last[1] and "skip" not in last[2] and not buy
    return ("PASS" if ok else "FAIL"), f"card selects {selects}; purchasable left in shop: {len(buy)}"


@test
def fake_merchant_foul_potion_fight(eng_factory):
    """Fake Merchant: buy a fake relic, then throw a Foul Potion -> FakeMerchantEventEncounter -> Fake Merchant's Rug."""
    eng = eng_factory()
    s = S(eng, seed="fakem")
    s.map()
    s.send({"cmd": "set_player", "gold": 500, "potions": ["FOUL_POTION"]})
    s.send({"cmd": "dbg_enter_act", "act": 2})
    s.map()
    s.send({"cmd": "rl_enter_room", "type": "Event", "event": "FAKE_MERCHANT"})
    d0 = s.r["decision"]
    s.act(s.find(lambda c: c["k"] == "buy_relic")[0])
    bought = s.r["obs"]["relics"][-1]["id"]
    s.act(s.find(lambda c: c["k"] == "potion")[0])
    fight = s.r["obs"].get("combat") is not None
    s.send({"cmd": "set_player", "hp": 9999, "max_hp": 9999})
    rewards = []
    while not at_map(s.r) and not s.r["done"]:
        if s.r["decision"] == "rewards":
            rewards = [c.get("id") for c in s.r["cands"] if c["k"] == "reward"]
        s.step()
    relics = [x["id"] for x in s.r["obs"]["relics"]]
    eng.close()
    ok = d0 == "fake_merchant" and fight and "RELIC.FAKE_MERCHANTS_RUG" in relics and bought.startswith("RELIC.FAKE_")
    return ("PASS" if ok else "FAIL"), f"first {d0}; bought {bought}; fight started {fight}; reward relics {rewards}; rug owned {'RELIC.FAKE_MERCHANTS_RUG' in relics}"


@test
def foul_potion_at_shop(eng_factory):
    eng = eng_factory()
    s = S(eng, seed="foulshop")
    s.map()
    s.send({"cmd": "set_player", "gold": 0, "potions": ["FOUL_POTION"]})
    s.send({"cmd": "rl_enter_room", "type": "Shop"})
    g0 = s.r["obs"]["g"]["gold"]
    s.act(s.find(lambda c: c["k"] == "potion")[0])
    g1 = s.r["obs"]["g"]["gold"]
    eng.close()
    return ("PASS" if g1 - g0 == 100 and s.r["decision"] == "shop" else "FAIL"), f"gold {g0} -> {g1} after throwing Foul Potion at the merchant"


@test
def treasure_open_and_skip(eng_factory):
    """Chest: opening is the only way forward (gold before the relic); the relic can be skipped."""
    eng = eng_factory()
    s = S(eng, seed="chest")
    s.map()
    s.send({"cmd": "rl_enter_room", "type": "Treasure"})
    k0 = [c["k"] for c in s.r["cands"]]
    g0 = s.r["obs"]["g"]["gold"]
    s.act(s.find(lambda c: c["k"] == "open")[0])
    k1 = sorted({c["k"] for c in s.r["cands"]})
    g1 = s.r["obs"]["g"]["gold"]
    n_rel = len(s.r["obs"]["relics"])
    s.act(s.find(lambda c: c["k"] == "skip")[0])
    eng.close()
    ok = k0 == ["open"] or set(k0) <= {"open", "potion", "discard_potion"}
    ok = ok and "treasure" in k1 and "skip" in k1 and g1 > g0 and at_map(s.r) and len(s.r["obs"]["relics"]) == n_rel
    return ("PASS" if ok else "FAIL"), f"before opening {k0}; after {k1}; gold {g0}->{g1}; after skip: {s.r['decision']}"


def ks(r):
    return sorted({c["k"] for c in r["cands"]})


def win_to_rewards(s, deck=None, relics=None, potions=None, encounter=None, play_first=None):
    """Strong build into a Monster room, fight until the rewards screen. play_first: card id played whenever playable
    (before anything else; until it has been played the turn is ended without attacking)."""
    setup = dict(STRONG, deck=deck or STRONG["deck"], relics=relics or STRONG["relics"])
    if potions is not None:
        setup["potions"] = potions
    s.send({"cmd": "set_player", **setup})
    room = {"cmd": "rl_enter_room", "type": "Monster"}
    if encounter:
        room["encounter"] = encounter
    s.send(room)
    played = play_first is None
    for _ in range(3000):
        r = s.r
        if r["done"] or r["decision"] == "rewards":
            return r
        if r["decision"] == "combat_play" and not played:
            hand = [c["id"] for c in r["obs"]["combat"]["hand"]]
            mine = [i for i, c in enumerate(r["cands"]) if c["k"] == "play" and hand[c["hand"]] == play_first]
            if mine:
                s.act(mine[0])
                played = True
            else:
                s.act(s.find(lambda c: c["k"] == "end_turn")[0])
            continue
        s.step()
    raise RuntimeError(f"no rewards screen, at {s.r['decision']}")


def card_reward_cand(r):
    return [c for c in r["cands"] if c["k"] == "reward" and c.get("rt") == "Card"]


@test
def card_reward_skip_declines(eng_factory):
    """Rewards screen: the card reward shows its cards and alternatives inline; Skip on the opened card reward (nothing
    changed) declines it: it is no longer offered (re-opening would only show the identical screen again), and it is
    left behind at proceed like an untaken reward (SkipLocalRewardsSet -> CardReward.OnSkipped)."""
    eng = eng_factory()
    s = S(eng, seed="crskip")
    s.map()
    win_to_rewards(s)
    cand = card_reward_cand(s.r)
    inline = [c["id"] for c in cand[0]["cards"]] if cand else None
    alts = cand[0].get("alts") if cand else None
    s.act(s.r["cands"].index(cand[0]))
    shown = [c["card"]["id"] for c in s.r["cands"] if c["k"] == "reward_card"]
    ctx_alts = s.r["obs"]["ctx"].get("alts")
    s.act(s.find(lambda c: c["k"] == "skip")[0])
    after = (s.r["decision"], [c.get("rt") or c["k"] for c in s.r["cands"]])
    deck_n = len(s.r["obs"]["deck"])
    s.act(s.find(lambda c: c["k"] == "proceed")[0])
    eng.close()
    ok = (cand and inline == shown and len(shown) == 3 and alts == ctx_alts == ["Skip"] and after[0] == "rewards"
          and "Card" not in after[1] and at_map(s.r) and len(s.r["obs"]["deck"]) == deck_n)
    return ("PASS" if ok else "FAIL"), f"inline {inline} alts {alts}; opened {shown} alts {ctx_alts}; after skip {after}; after proceed {s.r['decision']}"


@test
def card_reward_reroll_or_potion_then_skip(eng_factory):
    """Driftwood: after REROLL (or a potion) on the card reward screen, Skip is a plain back-out: the reward stays with
    the rerolled cards but is held back until the game changes on the rewards screen (re-opening at once would
    reproduce the screen just left); a reward taken or a potion used releases it. A skip with nothing changed declines."""
    eng = eng_factory()
    s = S(eng, seed="crroll")
    s.map()
    win_to_rewards(s, relics=STRONG["relics"] + ["DRIFTWOOD"], potions=["FRUIT_JUICE", "FRUIT_JUICE"])
    log = []
    cand = card_reward_cand(s.r)
    log.append(("alts", cand[0].get("alts") if cand else None))
    s.act(s.r["cands"].index(cand[0]))
    s.act(s.find(lambda c: c["k"] == "reroll")[0])
    rerolled = [c["card"]["id"] for c in s.r["cands"] if c["k"] == "reward_card"]
    s.act(s.find(lambda c: c["k"] == "skip")[0])
    held1 = (s.r["decision"], not card_reward_cand(s.r))
    s.act(s.find(lambda c: c["k"] == "reward" and c.get("rt") == "Gold")[0])
    back = card_reward_cand(s.r)
    released1 = (bool(back), [c["id"] for c in back[0]["cards"]] if back else None, back[0].get("alts") if back else None)
    s.act(s.r["cands"].index(back[0]))
    s.act(s.find(lambda c: c["k"] == "potion")[0])
    s.act(s.find(lambda c: c["k"] == "skip")[0])
    held2 = (s.r["decision"], not card_reward_cand(s.r))
    s.act(s.find(lambda c: c["k"] == "potion")[0])
    released2 = bool(card_reward_cand(s.r))
    s.act(s.r["cands"].index(card_reward_cand(s.r)[0]))
    s.act(s.find(lambda c: c["k"] == "skip")[0])
    declined = (s.r["decision"], [c.get("rt") or c["k"] for c in s.r["cands"] if c["k"] in ("reward", "proceed")])
    eng.close()
    ok = (log[0][1] == ["Skip", "REROLL"] and held1 == ("rewards", True) and released1 == (True, rerolled, ["Skip"])
          and held2 == ("rewards", True) and released2 and declined[0] == "rewards" and "Card" not in declined[1] and "proceed" in declined[1])
    return ("PASS" if ok else "FAIL"), (f"{log}; rerolled {rerolled}; after reroll+skip (decision, card held) {held1}; after gold "
                                        f"(offered, cards, alts) {released1}; after potion+skip {held2}; after a potion on the "
                                        f"rewards screen offered again {released2}; plain skip -> {declined}")


@test
def removal_reward_skip_declines(eng_factory):
    """Forbidden Grimoire's card removal reward: cancelling its selection declines the reward (no longer offered)."""
    eng = eng_factory()
    s = S(eng, seed="grim")
    s.map()
    win_to_rewards(s, deck=["FORBIDDEN_GRIMOIRE"] + ["STRIKE_IRONCLAD"] * 4 + ["BLUDGEON"] * 2, play_first="CARD.FORBIDDEN_GRIMOIRE")
    rem = s.find(lambda c: c["k"] == "reward" and c.get("rt") == "RemoveCard")
    deck0 = len(s.r["obs"]["deck"])
    s.act(rem[0])
    sel = (s.r["decision"], s.r["obs"]["ctx"].get("sel"), ks(s.r))
    s.act(s.find(lambda c: c["k"] == "skip")[0])
    after = (s.r["decision"], [c.get("rt") or c["k"] for c in s.r["cands"]], len(s.r["obs"]["deck"]))
    eng.close()
    ok = rem and sel[:2] == ("card_select", "deck_remove") and "skip" in sel[2] and after[0] == "rewards" and "RemoveCard" not in after[1] and after[2] == deck0
    return ("PASS" if ok else "FAIL"), f"removal reward offered {bool(rem)}; opened {sel}; after skip {after}"


@test
def smith_has_no_cancel(eng_factory):
    """Smith: cancelling would return to the identical rest site, so the upgrade selection has no skip; picking upgrades."""
    eng = eng_factory()
    s = S(eng, seed="smith")
    s.map()
    s.send({"cmd": "rl_enter_room", "type": "Rest"})
    s.act(s.find(lambda c: c.get("key") == "SMITH")[0])
    sel = (s.r["decision"], s.r["obs"]["ctx"].get("cancelable"), ks(s.r))
    up0 = sum(c["up"] for c in s.r["obs"]["deck"])
    s.act(s.find(lambda c: c["k"] == "pick")[0])
    after = (s.r["decision"], sorted(c.get("key") or c["k"] for c in s.r["cands"]), sum(c["up"] for c in s.r["obs"]["deck"]) - up0)
    eng.close()
    # the rest site has no option left after a successful one (ShouldDisableRemainingRestSiteOptions): the map opens
    ok = sel[:2] == ("card_select", True) and "skip" not in sel[2] and after[0] == "map_select" and after[2] == 1
    return ("PASS" if ok else "FAIL"), f"Smith -> {sel}; pick -> {after}"


@test
def cook_has_no_cancel(eng_factory):
    """Meat Cleaver's Cook: no cancel on its two-card removal (it would return to the identical rest site)."""
    eng = eng_factory()
    s = S(eng, seed="cook")
    s.map()
    s.send({"cmd": "set_player", "relics": ["BURNING_BLOOD", "MEAT_CLEAVER"]})
    s.send({"cmd": "rl_enter_room", "type": "Rest"})
    keys = sorted(c.get("key") or c["k"] for c in s.r["cands"])
    s.act(s.find(lambda c: c.get("key") == "COOK")[0])
    sel = (s.r["decision"], s.r["obs"]["ctx"].get("cancelable"), s.r["obs"]["ctx"].get("max"), ks(s.r))
    deck0, mhp0 = len(s.r["obs"]["deck"]), s.r["obs"]["g"]["max_hp"]
    s.act(s.find(lambda c: c["k"] == "pick")[0])
    mid = ks(s.r)
    s.act(s.find(lambda c: c["k"] == "pick")[0])
    after = (s.r["decision"], deck0 - len(s.r["obs"]["deck"]), s.r["obs"]["g"]["max_hp"] - mhp0)
    eng.close()
    ok = "COOK" in keys and sel[:3] == ("card_select", True, 2) and "skip" not in sel[3] and "skip" not in mid and after == ("map_select", 2, 5)
    return ("PASS" if ok else "FAIL"), f"rest options {keys}; Cook -> {sel}; after one pick {mid}; after two (decision, removed, max hp) {after}"


@test
def shop_removal_has_no_cancel(eng_factory):
    """Merchant card removal: no cancel (it would return to the identical shop); the shop offers everything else directly."""
    eng = eng_factory()
    s = S(eng, seed="shoprem")
    s.map()
    s.send({"cmd": "set_player", "gold": 300})
    s.send({"cmd": "rl_enter_room", "type": "Shop"})
    rem = s.find(lambda c: c["k"] == "remove")
    price = s.r["cands"][rem[0]]["price"] if rem else None
    g0, deck0 = s.r["obs"]["g"]["gold"], len(s.r["obs"]["deck"])
    s.act(rem[0])
    sel = (s.r["decision"], s.r["obs"]["ctx"].get("sel"), s.r["obs"]["ctx"].get("cancelable"), ks(s.r))
    s.act(s.find(lambda c: c["k"] == "pick")[0])
    after = (s.r["decision"], "remove" in ks(s.r), g0 - s.r["obs"]["g"]["gold"], deck0 - len(s.r["obs"]["deck"]))
    eng.close()
    ok = rem and sel[:3] == ("card_select", "deck_remove", True) and "skip" not in sel[3] and after == ("shop", False, price, 1)
    return ("PASS" if ok else "FAIL"), f"remove at {price}; opened {sel}; after pick (decision, remove offered, gold spent, removed) {after}"


@test
def shop_potion_not_offered_with_sozu(eng_factory):
    """Sozu: a merchant potion purchase fails (TryToProcure NotAllowed) and changes nothing, so no buy_potion is
    offered; a potion reward is still offered, because PotionReward.OnSelect takes it (the potion is lost)."""
    eng = eng_factory()
    s = S(eng, seed="sozu")
    s.map()
    s.send({"cmd": "set_player", "gold": 500, "relics": ["BURNING_BLOOD", "SOZU"]})
    s.send({"cmd": "rl_enter_room", "type": "Shop"})
    shop = sorted({c["k"] for c in s.r["cands"]})
    s.act(s.find(lambda c: c["k"] == "leave")[0])
    s.map()
    win_to_rewards(s, relics=STRONG["relics"] + ["SOZU", "WHITE_BEAST_STATUE"], potions=[])
    pot = s.find(lambda c: c["k"] == "reward" and c.get("rt") == "Potion")
    taken = None
    if pot:
        n = len(s.r["cands"])
        s.act(pot[0])
        taken = (s.r["decision"], len(s.r["cands"]) < n, [p["id"] for p in s.r["obs"]["potions"]])
    eng.close()
    ok = "buy_potion" not in shop and "buy_card" in shop and pot and taken[0] == "rewards" and taken[1] and taken[2] == []
    return ("PASS" if ok else "FAIL"), f"shop candidate kinds {shop}; potion reward offered {bool(pot)}, after taking it {taken}"


@test
def autoplay_repeats_are_distinct(eng_factory):
    """Duplicator: Begone is played twice; transforming Minion Strike into Minion Strike leaves the hand unchanged, so
    only combat.played (cards started this turn, replays and autoplays included) tells the two selections apart
    (found by tools/cycle_check.py with Decisions, Decisions autoplaying Begone)."""
    eng = eng_factory()
    s = S(eng, seed="dupbegone")
    s.map()
    s.send({"cmd": "set_player", "hp": 500, "max_hp": 500, "relics": [], "potions": ["DUPLICATOR"],
            "deck": ["BEGONE", "MINION_STRIKE", "DEFEND_IRONCLAD", "DEFEND_IRONCLAD", "DEFEND_IRONCLAD"]})
    s.send({"cmd": "rl_enter_room", "type": "Monster", "encounter": "SHRINKER_BEETLE_WEAK"})
    s.act(s.find(lambda c: c["k"] == "potion")[0])
    hand = [c["id"] for c in s.r["obs"]["combat"]["hand"]]
    s.act(s.find(lambda c: c["k"] == "play" and hand[c["hand"]] == "CARD.BEGONE")[0])
    seen = []
    while s.r["decision"] == "card_select" and s.r["obs"]["ctx"].get("source") == "CARD.BEGONE" and len(seen) < 5:
        seen.append((s.r["obs"]["combat"]["played"], sorted(c["id"] for c in s.r["obs"]["combat"]["hand"])))
        s.act(s.find(lambda c: c["k"] == "pick" and c["card"]["id"] == "CARD.MINION_STRIKE")[0])
    eng.close()
    ok = len(seen) == 2 and seen[0][0] != seen[1][0] and seen[0][1] == seen[1][1]
    return ("PASS" if ok else "FAIL"), f"Begone selections (played, hand): {seen}"


@test
def act1_variants(eng_factory):
    """ActModel.GetRandomList: act 1 is Overgrowth or Underdocks depending on the seed."""
    eng = eng_factory()
    acts = set()
    for i in range(16):
        r = eng.send({"cmd": "rl_start", "character": "Silent", "seed": f"actvar{i}"})
        acts.add(r["obs"]["g"]["act_id"])
    eng.close()
    return ("PASS" if {"ACT.OVERGROWTH", "ACT.UNDERDOCKS"} <= acts else "FAIL"), f"act 1 variants over 16 seeds: {sorted(acts)}"


@test
def trial_abandon_flag(eng_factory):
    eng = eng_factory()
    s = S(eng, seed="trial")
    s.map()
    s.send({"cmd": "dbg_enter_act", "act": 3})
    s.map()
    s.send({"cmd": "rl_enter_room", "type": "Event", "event": "TRIAL"})
    s.act(s.find(lambda c: (c.get("key") or "").endswith("REJECT"))[0])
    s.act(s.find(lambda c: (c.get("key") or "").endswith("DOUBLE_DOWN"))[0])
    r = s.r
    eng.close()
    ok = r["done"] and r["victory"] is False and r["info"]["abandoned"] is True and r["info"]["hp"] == 0
    return ("PASS" if ok else "FAIL"), f"done={r['done']} victory={r['victory']} abandoned={r['info'].get('abandoned')} hp={r['info']['hp']}"


@test
def potion_slots_with_gap(eng_factory):
    """Slots are the real belt indices: after using the middle potion, discarding the last one discards that one."""
    eng = eng_factory()
    s = S(eng, seed="slots")
    s.map()
    s.send({"cmd": "set_player", "potions": ["BLOOD_POTION", "FRUIT_JUICE", "BLOCK_POTION"], "hp": 20})
    s.act(s.find(lambda c: c["k"] == "potion" and c["slot"] == 1)[0])
    slots1 = [(p["id"], p["slot"]) for p in s.r["obs"]["potions"]]
    s.act(s.find(lambda c: c["k"] == "discard_potion" and c["slot"] == 2)[0])
    slots2 = [(p["id"], p["slot"]) for p in s.r["obs"]["potions"]]
    eng.close()
    ok = slots1 == [("POTION.BLOOD_POTION", 0), ("POTION.BLOCK_POTION", 2)] and slots2 == [("POTION.BLOOD_POTION", 0)]
    return ("PASS" if ok else "FAIL"), f"after drinking slot 1: {slots1}; after discarding slot 2: {slots2}"


@test
def hook_choice_then_combat_end(eng_factory):
    """An end-of-turn hook choice whose hook action finishes as the last enemy dies (the orphaned hook action).

    Stampede's AfterAutoPostPlayPhaseEntered autoplays Headbutt; its discard-pile choice makes the end-of-turn
    context a GenericHookGameAction. Howl From Beyond (put into the exhaust pile with Burning Pact, a later listener
    of the same hook) then kills the beetle, so the turn loop, resumed inline from the hook action's CompletionTask,
    ends the combat. The client's EndCombatInternal yields at the run save before SetCombatState(NotInCombat), so the
    hook action is popped first; before the fix ActionQueueSet.CombatEnded cancelled the Finished action and PopAction
    threw "Tried to pop action GenericHookGameAction ... didn't find it in any queue"."""
    eng = eng_factory()
    s = S(eng, seed="orph")
    s.map()
    s.send({"cmd": "set_player", "hp": 500, "max_hp": 500, "relics": [],
            "deck": ["STAMPEDE", "HEADBUTT", "HOWL_FROM_BEYOND", "BURNING_PACT", "DEFEND_IRONCLAD"]})
    s.send({"cmd": "rl_enter_room", "type": "Monster", "encounter": "SHRINKER_BEETLE_WEAK"})
    for _ in range(200):
        r = s.r
        if r["done"] or r["obs"].get("combat") is None:
            break
        cands, hand = r["cands"], [c["id"] for c in r["obs"]["combat"]["hand"]]
        if r["decision"] == "card_select":
            picks = [i for i, c in enumerate(cands) if c["k"] == "pick"]
            howl = [i for i in picks if cands[i]["card"]["id"] == "CARD.HOWL_FROM_BEYOND"]
            s.act((howl or picks or [0])[0])
            continue
        play = {hand[c["hand"]]: i for i, c in enumerate(cands) if c["k"] == "play"}
        k = play.get("CARD.STAMPEDE")
        if k is None and "CARD.HOWL_FROM_BEYOND" in hand:
            k = play.get("CARD.BURNING_PACT")
        s.act(k if k is not None else s.find(lambda c: c["k"] == "end_turn")[0])
    recent = eng.send({"cmd": "dbg_info"})["recent"]
    eng.close()
    hook = [i for i, e in enumerate(recent) if "GenericHook" in e and "STAMPEDE_POWER>HEADBUTT" in e]
    finishing = [i for i in hook if "finishing:" in recent[i]]
    popped = [i for i in hook if "popped:" in recent[i]]
    won = [i for i, e in enumerate(recent) if "combat won" in e]
    cancelled = [e for e in recent if "cancelled:" in e and "GenericHook" in e]
    # Precondition: the killing end of turn really ran through the hook action (choice first, kill after).
    staged = bool(finishing and won and finishing[-1] < won[-1]
                  and not any("turn started" in e for e in recent[finishing[-1]:won[-1]]))
    ok = (staged and s.r["decision"] == "rewards" and not s.r["done"] and popped and popped[-1] < won[-1]
          and not cancelled)
    tail = [e.split("] ", 1)[-1][:70] for e in recent[(finishing[-1] if finishing else -6):][:6]]
    return ("PASS" if ok else "FAIL"), (f"scenario staged={staged}; decision after the kill: {s.r['decision']}; "
                                        f"hook action popped before combat won: {bool(popped and won and popped[-1] < won[-1])}; "
                                        f"cancelled hook actions: {len(cancelled)}; events: {tail}")



def play_named(s, card_id, target=0):
    hand = s.r["obs"]["combat"]["hand"]
    choices = s.find(lambda c: c["k"] == "play" and hand[c["hand"]]["id"] == "CARD." + card_id
                     and c.get("tgt", -1) in (-1, target))
    assert choices, (card_id, s.r["decision"], hand)
    s.act(choices[0])


@test
def fool_arrows_and_melodies(eng_factory):
    s = S(eng_factory(), char="Fool", seed="fool_arrows")
    s.map()
    s.send({"cmd": "set_player", "hp": 500, "max_hp": 500, "relics": ["SONG_BOW", "LANTERN"],
            "deck": ["FOOLISH_SHOT", "WAH_KI", "STRIKE_FOOL", "CHO_READY", "DEFEND_FOOL"]})
    s.send({"cmd": "rl_enter_room", "type": "Elite", "encounter": "KNIGHTS_ELITE"})
    assert s.r["obs"]["combat"]["arrows_available"] == 1
    play_named(s, "FOOLISH_SHOT")
    c = s.r["obs"]["combat"]
    assert c["arrows_available"] == 0 and c["arrows"][0]["status"] == "Away" and c["arrows"][0]["tgt"] == 0
    assert c["turn"]["arrows_shot"] == 1
    play_named(s, "WAH_KI")
    c = s.r["obs"]["combat"]
    impaled = next(p for p in c["enemies"][0]["pw"] if p["id"] == "POWER.IMPALED_POWER")
    assert impaled["v"]["HitCounter"] == 1
    assert c["turn"]["wah"] and c["turn"]["ki"] and not c["turn"]["cho"]
    play_named(s, "STRIKE_FOOL")
    c = s.r["obs"]["combat"]
    assert c["arrows_available"] == 1 and c["arrows"][0]["status"] == "Retrieved" and c["arrows"][0]["tgt"] == -1
    play_named(s, "CHO_READY")
    c = s.r["obs"]["combat"]
    assert c["turn"]["cho"] and c["turn"]["melody_cards"] == 2
    return "PASS", "arrow impales target, returns after three hits; Wah/Cho/Ki and melody-card count observed"


@test
def fool_recall_choice(eng_factory):
    s = S(eng_factory(), char="Fool", seed="fool_recall")
    s.map()
    s.send({"cmd": "set_player", "hp": 500, "max_hp": 500, "relics": ["SONG_BOW"],
            "deck": ["FOOLISH_SHOT", "RECALL", "DEFEND_FOOL", "DEFEND_FOOL", "DEFEND_FOOL"]})
    s.send({"cmd": "rl_enter_room", "type": "Elite", "encounter": "KNIGHTS_ELITE"})
    play_named(s, "DEFEND_FOOL")
    play_named(s, "FOOLISH_SHOT")
    play_named(s, "RECALL")
    assert s.r["decision"] == "card_select" and s.r["obs"]["ctx"]["sel"] == "combat_pile"
    assert s.r["obs"]["combat"]["arrows_available"] == 1
    s.act(s.find(lambda c: c["k"] == "pick" and c["card"]["id"] == "CARD.FOOLISH_SHOT")[0])
    assert s.r["decision"] == "combat_play" and s.r["obs"]["combat"]["draw_top"][0]["id"] == "CARD.FOOLISH_SHOT"
    return "PASS", "Recall exposes discard selection and preserves the known draw-pile top"



@test
def new_pets_are_observed(eng_factory):
    s = S(eng_factory(), seed="new_pets")
    s.map()
    s.send({"cmd": "obtain_relics", "ids": ["WUPPY", "BYRDINE"]})
    s.send({"cmd": "rl_enter_room", "type": "Monster", "encounter": "SHRINKER_BEETLE_WEAK"})
    allies = {a["id"]: a for a in s.r["obs"]["combat"]["allies"]}
    assert {"MONSTER.WUPPY", "MONSTER.BYRDINE_MONSTER"} <= allies.keys(), allies
    assert allies["MONSTER.WUPPY"]["max_hp"] == 60
    return "PASS", "Wuppy and Byrdine appear with HP and powers in allies"


def main():
    args = sys.argv[1:]
    engine_dir = None
    if "--engine" in args:  # test another engine build than ./engine
        k = args.index("--engine")
        engine_dir = args[k + 1]
        del args[k:k + 2]
    names = args or list(TESTS)
    engines = []

    def factory():
        e = Engine(engine_dir, timeout=120)
        engines.append(e)
        return e
    fails = 0
    for n in names:
        try:
            st, msg = TESTS[n](factory)
        except Exception as e:
            st, msg = "ERR", f"{type(e).__name__}: {str(e)[:600]}"
            traceback.print_exc(limit=1)
        if st != "PASS":
            fails += 1
        print(f"{st:4s} {n}: {msg}", flush=True)
        for e in engines:
            try:
                e.p.kill()
            except Exception:
                pass
        engines.clear()
    print(f"{len(names) - fails}/{len(names)} PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
