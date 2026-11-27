"""A fake StS2 engine that speaks the RL protocol (engine/RL_PROTOCOL.md).

`FakeGame` is a small but coherent card game: real-looking combat (energy, draw pile, block,
intents, Strength/Vulnerable), an act map, card rewards, shops, events, rest sites, treasure,
multi-select card screens, bundles and a crystal-sphere minigame. Content IDs are fake but keep
the real naming scheme. It is learnable (good card play and pathing win more often than random),
so an end-to-end training run on it is a meaningful pipeline check.

Run as a JSON-lines subprocess (to exercise EngineProcess with pipes, log noise, crashes, hangs):
    python -m rl.fake_engine [--p_error 0.001] [--p_crash 0.0] [--p_hang 0.0] [--step_ms 0]
"""
import argparse
import base64
import copy
import json
import pickle
import random
import sys
import time

# ---------------------------------------------------------------- content (fixed across games)

_POOL_RNG = random.Random(12345)
KEYWORDS = ("Exhaust", "Ethereal", "Innate", "Retain")
RARITIES = ("Common", "Uncommon", "Rare")


def _make_pool():
    pool = []
    for i in range(36):
        typ = ("Attack", "Skill", "Power")[i % 3]
        rar = RARITIES[_POOL_RNG.randrange(3)]
        cost = _POOL_RNG.choice((0, 1, 1, 1, 2, 2, 3))
        quality = _POOL_RNG.random()  # hidden: decides how strong the card is
        if typ == "Attack":
            tgt = "AllEnemies" if _POOL_RNG.random() < 0.25 else "AnyEnemy"
            v = {"Damage": int(3 + 6 * cost * (0.5 + quality) + 3 * quality)}
            if _POOL_RNG.random() < 0.3:
                v["VulnerablePower"] = 1 + _POOL_RNG.randrange(2)
        elif typ == "Skill":
            tgt = "Self"
            v = {"Block": int(3 + 5 * cost * (0.5 + quality) + 3 * quality)}
            if _POOL_RNG.random() < 0.3:
                v["Cards"] = 1 + _POOL_RNG.randrange(2)
        else:
            tgt = "None"
            v = {"StrengthPower": 1 + int(2 * quality)}
        kw = [KEYWORDS[_POOL_RNG.randrange(4)]] if _POOL_RNG.random() < 0.2 else []
        pool.append(dict(id=f"CARD.FAKE_{typ.upper()}_{i:02d}", cost=cost, type=typ, rar=rar, tgt=tgt, v=v, kw=kw))
    return pool


CARD_POOL = _make_pool()
STARTER = {
    "Ironclad": [("CARD.STRIKE_IRONCLAD", 5), ("CARD.DEFEND_IRONCLAD", 4), ("CARD.BASH", 1)],
    "Silent": [("CARD.STRIKE_SILENT", 5), ("CARD.DEFEND_SILENT", 5), ("CARD.NEUTRALIZE", 1)],
    "Defect": [("CARD.STRIKE_DEFECT", 4), ("CARD.DEFEND_DEFECT", 4), ("CARD.ZAP", 1), ("CARD.DUALCAST", 1)],
    "Regent": [("CARD.STRIKE_REGENT", 4), ("CARD.DEFEND_REGENT", 4), ("CARD.FALLING_STAR", 1)],
    "Necrobinder": [("CARD.STRIKE_NECROBINDER", 4), ("CARD.DEFEND_NECROBINDER", 4), ("CARD.POKE", 1)],
}
RELICS = [f"RELIC.FAKE_{i:02d}" for i in range(20)]
POTIONS = {"POTION.FAKE_FIRE": "AnyEnemy", "POTION.FAKE_BLOCK": "Self", "POTION.FAKE_HEAL": "Self"}
MONSTERS = {  # id: (hp, base attack)
    "MONSTER.FAKE_SLIME_S": (11, 4), "MONSTER.FAKE_SLIME_M": (24, 7), "MONSTER.FAKE_LOUSE": (14, 6),
    "MONSTER.FAKE_CULTIST": (44, 6), "MONSTER.FAKE_JAW_WORM": (38, 9),
}
ELITES = {"MONSTER.FAKE_NOB": (70, 12), "MONSTER.FAKE_SENTRY": (36, 8)}
BOSSES = {"MONSTER.FAKE_GUARDIAN": (130, 14), "MONSTER.FAKE_HEXAGHOST": (120, 11)}
NODE_TYPES = ("Monster", "Monster", "Monster", "Elite", "Rest", "Shop", "Event", "Treasure", "Unknown")


def _starter_card(cid):
    if "STRIKE" in cid:
        return dict(id=cid, cost=1, type="Attack", rar="Basic", tgt="AnyEnemy", v={"Damage": 6}, kw=[])
    if "DEFEND" in cid:
        return dict(id=cid, cost=1, type="Skill", rar="Basic", tgt="Self", v={"Block": 5}, kw=[])
    return dict(id=cid, cost=2, type="Attack", rar="Basic", tgt="AnyEnemy", v={"Damage": 8, "VulnerablePower": 2}, kw=[])


def _card(base, up=0):
    c = dict(base)
    c.update(up=up, star=0, ench=None, ench_n=0, aff=None, aff_n=0)
    c["v"] = {k: (v + (v // 3 + 1) * up if k in ("Damage", "Block") else v) for k, v in base["v"].items()}
    c["kw"] = list(base.get("kw", []))
    return c


def _pw(lst, pid):
    for p in lst:
        if p["id"] == pid:
            return p["n"]
    return 0


def _add_pw(lst, pid, n):
    for p in lst:
        if p["id"] == pid:
            p["n"] += n
            return
    lst.append({"id": pid, "n": n})


class FakeGame:
    """One run of the fake game. `start()` and `step(i)` return protocol responses R."""

    def __init__(self, n_acts=2, rows=7, max_steps=3000, rest_test=False):
        self.n_acts, self.rows, self.max_steps = n_acts, rows, max_steps
        self.rest_test = rest_test

    # ------------------------------------------------------------ run setup
    def start(self, character="Ironclad", ascension=0, seed="0"):
        self.rng = random.Random(f"fake-{seed}-{character}-{ascension}")
        self.char, self.asc = character, ascension
        self.hp = self.max_hp = 80 - 2 * ascension
        self.gold, self.act, self.floor, self.total_floor, self.steps = 99, 1, 0, 0, 0
        self.deck = [_card(_starter_card(cid)) for cid, n in STARTER.get(character, STARTER["Ironclad"]) for _ in range(n)]
        self.relics = [{"id": "RELIC.FAKE_STARTER", "v": {"Heal": 6}}]
        self.potions = {}
        self.potion_slots = 3
        self.combat = None
        self.ctx = {}
        self.done, self.victory = False, None
        self.saves, self.save, self.save_id = {}, None, 0
        self._new_map()
        self._neow()
        if self.rest_test:
            self.deck = [_card(_starter_card("CARD.STRIKE_IRONCLAD"))]
            self._do_map({"node": 0})
        return self._response()

    def _new_map(self):
        if self.rest_test:
            self.map_nodes = [dict(c=3, r=1, t="Rest"), dict(c=3, r=2, t="Boss")]
            self.map_edges, self.map_cur, self.first_row = [(0, 1)], -1, [0]
            return
        rng, rows = self.rng, self.rows
        nodes, by_row = [], []
        for r in range(rows):
            if r == rows - 1:
                cols = [3]
            else:
                cols = sorted(rng.sample(range(7), rng.randint(2, 4)))
            row = []
            for c in cols:
                if r == 0:
                    t = "Monster"
                elif r == rows - 1:
                    t = "Boss"
                elif r == rows - 2:
                    t = "Rest"
                else:
                    t = rng.choice(NODE_TYPES)
                row.append(len(nodes))
                nodes.append({"c": c, "r": r + 1, "t": t})
            by_row.append(row)
        edges = set()
        for r in range(rows - 1):
            a, b = by_row[r], by_row[r + 1]
            for i in a:
                edges.add((i, min(b, key=lambda j: abs(nodes[j]["c"] - nodes[i]["c"]))))
            for j in b:
                edges.add((min(a, key=lambda i: abs(nodes[j]["c"] - nodes[i]["c"])), j))
            for _ in range(rng.randint(0, 2)):
                edges.add((rng.choice(a), rng.choice(b)))
        self.map_nodes, self.map_edges, self.map_cur = nodes, sorted(edges), -1
        self.first_row = by_row[0]

    def _neow(self):
        opts = [
            {"k": "option", "key": "NEOW.OPT.GOLD", "v": {"Gold": 100}},
            {"k": "option", "key": "NEOW.OPT.MAX_HP", "v": {"MaxHp": 8}},
            {"k": "option", "key": "NEOW.OPT.RELIC", "v": {}},
        ]
        self._set("event_choice", opts, ctx={"event": "EVENT.NEOW"})
        self.on_option = self._neow_choice

    def _neow_choice(self, c):
        if c["key"] == "NEOW.OPT.GOLD":
            self.gold += 100
        elif c["key"] == "NEOW.OPT.MAX_HP":
            self.max_hp += 8
            self.hp += 8
        else:
            self.relics.append({"id": self.rng.choice(RELICS), "v": {}})
        self._to_map()

    # ------------------------------------------------------------ decision plumbing
    def _set(self, decision, cands, ctx=None):
        self.decision, self.cands, self.ctx = decision, cands, ctx or {}

    def _to_map(self):
        if self.map_cur < 0:
            nxt = list(self.first_row)
        else:
            nxt = [j for i, j in self.map_edges if i == self.map_cur]
        if not nxt:  # end of act map without boss node should not happen
            return self._end(False)
        self._set("map_select", [{"k": "map", "node": j} for j in nxt])

    def _end(self, victory):
        self.done, self.victory = True, victory
        self._set("game_over", [])

    def step(self, i):
        if self.done:
            raise ValueError("game is over")
        c = self.cands[i]
        self.steps += 1
        k = c["k"]
        getattr(self, "_do_" + k)(c)
        if not self.done and self.hp <= 0:
            self._end(False)
        if not self.done and self.steps >= self.max_steps:
            self._end(False)
        return self._response()

    # ------------------------------------------------------------ map / rooms
    def _do_map(self, c):
        self.map_cur = c["node"]
        self.floor = self.map_nodes[c["node"]]["r"]
        self.total_floor += 1
        self.save_id += 1
        state = {k: v for k, v in self.__dict__.items() if k not in ("saves", "save", "save_id")}
        state = copy.deepcopy(state, {id(self): self})
        self._enter_room()
        self.save = dict(id=self.save_id, kind="room", floor=self.total_floor, room=self.room)
        self.saves[self.save_id] = (self.save.copy(), state)

    def load(self, save_id):
        meta, state = self.saves[save_id]
        self.__dict__ = dict(copy.deepcopy(state, {id(self): self}), saves=self.saves,
                             save=meta.copy(), save_id=self.save_id)
        self._enter_room()
        return self._response()

    def _enter_room(self):
        t = self.map_nodes[self.map_cur]["t"]
        if t == "Unknown":
            t = self.rng.choice(("Monster", "Event", "Event", "Shop", "Treasure"))
        self.room = t
        if t in ("Monster", "Elite", "Boss"):
            self._start_combat(t)
        elif t == "Rest":
            self._set("rest_site", [{"k": "rest", "key": "HEAL"}, {"k": "rest", "key": "SMITH"}])
        elif t == "Shop":
            self._open_shop()
        elif t == "Event":
            self._event()
        elif t == "Treasure":
            self._set("treasure", [{"k": "treasure", "id": self.rng.choice(RELICS)}, {"k": "leave"}])

    def _do_proceed(self, c):
        self._to_map()

    def _do_leave(self, c):
        self.shop = None
        self._to_map()

    def _do_skip(self, c):
        if self.decision == "card_select":
            return self._finish_select()
        self._after_reward()

    def _do_treasure(self, c):
        self.relics.append({"id": c["id"], "v": {"Counter": self.rng.randrange(4)}})
        self._to_map()

    def _do_rest(self, c):
        if c["key"] == "HEAL":
            self.hp = min(self.max_hp, self.hp + int(0.3 * self.max_hp))
            self._to_map()
        else:
            opts = [j for j, cd in enumerate(self.deck) if cd["up"] == 0]
            if not opts:
                return self._to_map()
            self._card_select(opts, 1, 1, "upgrade", then=self._to_map)

    # ------------------------------------------------------------ multi-select card screens
    def _card_select(self, deck_idx, mn, mx, reason, then):
        self.sel = {"opts": deck_idx, "picked": [], "min": mn, "max": mx, "reason": reason, "then": then}
        self._select_cands()

    def _select_cands(self):
        s = self.sel
        cands = [{"k": "pick", "card": self.deck[j], "src": n} for n, j in enumerate(s["opts"]) if n not in s["picked"]]
        if len(s["picked"]) >= s["min"]:
            cands.append({"k": "confirm"})
        ctx = {"min": s["min"], "max": s["max"], "reason": s["reason"],
               "picked": [self.deck[s["opts"][n]] for n in s["picked"]]}
        self._set("card_select", cands, ctx)

    def _do_pick(self, c):
        self.sel["picked"].append(c["src"])
        if len(self.sel["picked"]) >= self.sel["max"]:
            return self._finish_select()
        self._select_cands()

    def _do_confirm(self, c):
        self._finish_select()

    def _finish_select(self):
        s = self.sel
        idx = sorted((s["opts"][n] for n in s["picked"]), reverse=True)
        for j in idx:
            if s["reason"] == "upgrade":
                base = next((b for b in CARD_POOL if b["id"] == self.deck[j]["id"]), None) or _starter_card(self.deck[j]["id"])
                self.deck[j] = _card(base, up=1)
            elif s["reason"] == "remove":
                self.deck.pop(j)
            elif s["reason"] == "transform":
                self.deck[j] = _card(self.rng.choice(CARD_POOL))
        self.sel = None
        s["then"]()

    # ------------------------------------------------------------ shop / events / rewards
    def _open_shop(self):
        rng = self.rng
        self.shop = {
            "cards": [(_card(rng.choice(CARD_POOL)), rng.randint(45, 160)) for _ in range(5)],
            "relics": [(rng.choice(RELICS), rng.randint(140, 300)) for _ in range(2)],
            "potions": [(rng.choice(list(POTIONS)), rng.randint(40, 80)) for _ in range(2)],
            "remove": 75,
        }
        self._shop_cands()

    def _shop_cands(self):
        s, g = self.shop, self.gold
        cands = [{"k": "buy_card", "card": c, "price": p} for c, p in s["cards"] if p <= g]
        cands += [{"k": "buy_relic", "id": r, "price": p} for r, p in s["relics"] if p <= g]
        if len(self.potions) < self.potion_slots:
            cands += [{"k": "buy_potion", "id": r, "price": p} for r, p in s["potions"] if p <= g]
        if s["remove"] is not None and s["remove"] <= g and len(self.deck) > 1:
            cands.append({"k": "remove", "price": s["remove"]})
        cands.append({"k": "leave"})
        self._set("shop", cands)

    def _do_buy_card(self, c):
        self.gold -= c["price"]
        self.deck.append(c["card"])
        self.shop["cards"] = [x for x in self.shop["cards"] if x[0] is not c["card"]]
        self._shop_cands()

    def _do_buy_relic(self, c):
        self.gold -= c["price"]
        self.relics.append({"id": c["id"], "v": {}})
        self.shop["relics"] = [x for x in self.shop["relics"] if x[0] != c["id"]]
        self._shop_cands()

    def _do_buy_potion(self, c):
        self.gold -= c["price"]
        self._gain_potion(c["id"])
        self.shop["potions"] = [x for x in self.shop["potions"] if x[0] != c["id"]]
        self._shop_cands()

    def _do_remove(self, c):
        self.gold -= c["price"]
        self.shop["remove"] = None
        self._card_select(list(range(len(self.deck))), 1, 1, "remove", then=self._shop_cands)

    def _gain_potion(self, pid):
        for s in range(self.potion_slots):
            if s not in self.potions:
                self.potions[s] = pid
                return

    def _event(self):
        rng = self.rng
        eid = rng.randrange(8)
        opts = []
        for j in range(rng.randint(2, 3)):
            v = {"Gold": rng.choice((0, 0, 25, 50, 100)), "HpLoss": rng.choice((0, 0, 5, 10, 15))}
            opts.append({"k": "option", "key": f"EVENT.FAKE_{eid}.OPT_{j}", "v": v})
        self._set("event_choice", opts, ctx={"event": f"EVENT.FAKE_{eid}"})
        self.on_option = self._event_choice

    def _event_choice(self, c):
        self.gold += c["v"].get("Gold", 0)
        self.hp -= c["v"].get("HpLoss", 0)
        roll = self.rng.random()
        if roll < 0.2 and len(self.deck) > 2:
            n = min(len(self.deck), 5)
            opts = self.rng.sample(range(len(self.deck)), n)
            self._card_select(opts, 0, 2, self.rng.choice(("remove", "transform")), then=self._to_map)
        elif roll < 0.35:
            packs = [[_card(self.rng.choice(CARD_POOL)) for _ in range(3)] for _ in range(self.rng.randint(2, 3))]
            self._set("bundle_select", [{"k": "bundle", "cards": p} for p in packs] + [{"k": "skip"}])
        elif roll < 0.45:
            self.sphere = {"left": 3, "revealed": set(), "w": 5}
            self._sphere_cands()
        elif roll < 0.55:
            self._set("unknown", [{"k": "proceed"}])
        else:
            self._to_map()

    def _do_option(self, c):
        self.on_option(c)

    def _do_bundle(self, c):
        self.deck.extend(c["cards"])
        self._to_map()

    def _sphere_cands(self):
        s = self.sphere
        cells = [(x, y) for x in range(s["w"]) for y in range(s["w"]) if (x, y) not in s["revealed"]]
        cands = [{"k": "sphere", "tool": t, "x": x, "y": y} for t in ("small", "big") for x, y in cells]
        self._set("crystal_sphere", cands, ctx={"tools_left": s["left"], "revealed": len(s["revealed"])})

    def _do_sphere(self, c):
        s = self.sphere
        r = 0 if c["tool"] == "small" else 1
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                s["revealed"].add((c["x"] + dx, c["y"] + dy))
        s["left"] -= 1 if c["tool"] == "small" else 2
        if (c["x"] + c["y"]) % 4 == 0:
            self.gold += 15
        if s["left"] <= 0:
            self._set("unknown", [{"k": "proceed"}])
        else:
            self._sphere_cands()

    def _reward(self):
        offers = [_card(self.rng.choice(CARD_POOL)) for _ in range(3)]
        self._set("card_reward", [{"k": "reward_card", "card": c} for c in offers] + [{"k": "skip"}])

    def _do_reward_card(self, c):
        self.deck.append(c["card"])
        self._after_reward()

    def _after_reward(self):
        if self.room == "Boss":
            if self.act >= self.n_acts:
                return self._end(True)
            self.act += 1
            self.floor = 0
            self.hp = min(self.max_hp, self.hp + int(0.6 * (self.max_hp - self.hp)))
            self._new_map()
        self._to_map()

    # ------------------------------------------------------------ combat
    def _start_combat(self, room):
        rng = self.rng
        scale = 1.0 + 0.35 * (self.act - 1) + 0.04 * self.asc
        if room == "Monster":
            ids = [rng.choice(list(MONSTERS)) for _ in range(rng.choice((1, 1, 2, 3)))]
            table = MONSTERS
        elif room == "Elite":
            ids, table = [rng.choice(list(ELITES))], ELITES
        else:
            ids, table = [rng.choice(list(BOSSES))], BOSSES
        enemies = []
        for mid in ids:
            hp, atk = table[mid]
            hp = int(hp * scale * rng.uniform(0.9, 1.1))
            enemies.append({"id": mid, "hp": hp, "max_hp": hp, "block": 0, "alive": True, "pw": [],
                            "atk": int(atk * scale)})
        if self.rest_test:
            enemies = [dict(id="MONSTER.FAKE_BOSS", hp=7, max_hp=7, block=0, alive=True, pw=[], atk=100)]
        draw = [dict(c) for c in self.deck]
        rng.shuffle(draw)
        self.combat = {"round": 0, "energy": 3, "max_energy": 3, "hand": [], "draw": draw, "discard": [],
                       "exhaust": [], "block": 0, "pw": [], "enemies": enemies,
                       "allies": [{"id": "MONSTER.OSTY", "hp": 20, "max_hp": 20, "block": 0, "alive": True, "pw": []}]
                       if self.char == "Necrobinder" else [],
                       "orbs": [{"id": "ORB.LIGHTNING", "passive": 3, "evoke": 8}] if self.char == "Defect" else []}
        self._new_round()

    def _draw(self, n):
        cb = self.combat
        for _ in range(n):
            if not cb["draw"]:
                cb["draw"], cb["discard"] = cb["discard"], []
                self.rng.shuffle(cb["draw"])
            if not cb["draw"] or len(cb["hand"]) >= 10:
                return
            cb["hand"].append(cb["draw"].pop())

    def _new_round(self):
        cb = self.combat
        cb["round"] += 1
        cb["energy"] = cb["max_energy"]
        cb["block"] = 0
        for e in cb["enemies"]:
            e["block"] = 0
            roll = self.rng.random()
            st = _pw(e["pw"], "POWER.STRENGTH_POWER")
            if roll < 0.65:
                hits = 1 if self.rng.random() < 0.7 else 2
                dmg = max(1, (e["atk"] + st) // hits)
                e["intents"] = [{"t": "Attack", "dmg": dmg, "hits": hits}]
            elif roll < 0.85:
                e["intents"] = [{"t": "Defend"}]
            else:
                e["intents"] = [{"t": "Buff"}]
            if self.rest_test:
                e["intents"] = [{"t": "Attack", "dmg": 100, "hits": 1}]
        self._draw(5)
        self._combat_cands()

    def _combat_cands(self):
        cb = self.combat
        cands = []
        n_en = len(cb["enemies"])
        for h, c in enumerate(cb["hand"]):
            if not self._playable(c):
                continue
            if c["tgt"] == "AnyEnemy":
                cands += [{"k": "play", "hand": h, "tgt": t} for t in range(n_en)]
            else:
                cands.append({"k": "play", "hand": h, "tgt": -1})
        for s, pid in sorted(self.potions.items()):
            if POTIONS[pid] == "AnyEnemy":
                cands += [{"k": "potion", "slot": s, "tgt": t} for t in range(n_en)]
            else:
                cands.append({"k": "potion", "slot": s, "tgt": -1})
            cands.append({"k": "discard_potion", "slot": s})
        if not self.rest_test or not cands:
            cands.append({"k": "end_turn"})
        self._set("combat_play", cands)

    def _playable(self, c):
        return c["type"] in ("Attack", "Skill", "Power") and c["cost"] <= self.combat["energy"]

    def _hit(self, e, dmg):
        if _pw(e["pw"], "POWER.VULNERABLE_POWER") > 0:
            dmg = int(dmg * 1.5)
        absorbed = min(e["block"], dmg)
        e["block"] -= absorbed
        e["hp"] -= dmg - absorbed
        if e["hp"] <= 0:
            e["hp"], e["alive"] = 0, False

    def _do_play(self, c):
        cb = self.combat
        card = cb["hand"].pop(c["hand"])
        cb["energy"] -= card["cost"]
        st = _pw(cb["pw"], "POWER.STRENGTH_POWER")
        v = card["v"]
        targets = cb["enemies"] if card["tgt"] == "AllEnemies" else ([cb["enemies"][c["tgt"]]] if c["tgt"] >= 0 else [])
        for e in targets:
            if "Damage" in v:
                self._hit(e, v["Damage"] + st)
            if "VulnerablePower" in v:
                _add_pw(e["pw"], "POWER.VULNERABLE_POWER", v["VulnerablePower"])
        if "Block" in v:
            cb["block"] += v["Block"]
        if "StrengthPower" in v:
            _add_pw(cb["pw"], "POWER.STRENGTH_POWER", v["StrengthPower"])
        if "Cards" in v:
            self._draw(v["Cards"])
        if card["type"] != "Power":
            (cb["exhaust"] if "Exhaust" in card["kw"] else cb["discard"]).append(card)
        cb["enemies"] = [e for e in cb["enemies"] if e["alive"]]
        self._after_action()

    def _do_potion(self, c):
        cb = self.combat
        pid = self.potions.pop(c["slot"])
        if pid == "POTION.FAKE_FIRE":
            self._hit(cb["enemies"][c["tgt"]], 20)
        elif pid == "POTION.FAKE_BLOCK":
            cb["block"] += 12
        else:
            self.hp = min(self.max_hp, self.hp + 15)
        cb["enemies"] = [e for e in cb["enemies"] if e["alive"]]
        self._after_action()

    def _do_discard_potion(self, c):
        self.potions.pop(c["slot"])
        self._combat_cands()

    def _after_action(self):
        if not self.combat["enemies"]:
            return self._win_combat()
        self._combat_cands()

    def _do_end_turn(self, c):
        cb = self.combat
        for e in cb["enemies"]:
            for it in e["intents"]:
                if it["t"] == "Attack":
                    for _ in range(it["hits"]):
                        dmg = it["dmg"]
                        absorbed = min(cb["block"], dmg)
                        cb["block"] -= absorbed
                        self.hp -= dmg - absorbed
                elif it["t"] == "Defend":
                    e["block"] += 8
                else:
                    _add_pw(e["pw"], "POWER.STRENGTH_POWER", 2)
            for p in e["pw"]:
                if p["id"] == "POWER.VULNERABLE_POWER":
                    p["n"] -= 1
            e["pw"] = [p for p in e["pw"] if p["n"] != 0]
        if self.hp <= 0:
            return
        keep = [c for c in cb["hand"] if "Retain" in c["kw"]]
        for c in cb["hand"]:
            if "Retain" not in c["kw"]:
                (cb["exhaust"] if "Ethereal" in c["kw"] else cb["discard"]).append(c)
        cb["hand"] = keep
        self._new_round()

    def _win_combat(self):
        self.combat = None
        if self.rest_test:
            return self._end(True)
        self.gold += self.rng.randint(10, 25) * (3 if self.room == "Elite" else 1)
        if self.room == "Elite" or self.rng.random() < 0.3:
            self._gain_potion(self.rng.choice(list(POTIONS)))
        if self.room == "Elite":
            self.relics.append({"id": self.rng.choice(RELICS), "v": {}})
        self._reward()

    # ------------------------------------------------------------ observation
    def _hand_card(self, c):
        st = _pw(self.combat["pw"], "POWER.STRENGTH_POWER")
        out = dict(c, playable=self._playable(c))
        out["pv"] = {k: (v + st if k == "Damage" else v) for k, v in c["v"].items()}
        return out

    def _creature(self, e, intents=False):
        out = {k: e[k] for k in ("id", "hp", "max_hp", "block", "alive")}
        out["pw"] = [dict(p) for p in e["pw"]]
        if intents:
            out["intents"] = [dict(i) for i in e["intents"]]
        return out

    def _response(self):
        cb = self.combat
        combat = None
        if cb is not None:
            player = {"id": f"CHARACTER.{self.char.upper()}", "hp": self.hp, "max_hp": self.max_hp,
                      "block": cb["block"], "alive": True, "pw": [dict(p) for p in cb["pw"]]}
            combat = {"round": cb["round"], "energy": cb["energy"], "max_energy": cb["max_energy"],
                      "stars": 0, "hand": [self._hand_card(c) for c in cb["hand"]],
                      "draw": sorted(cb["draw"], key=lambda c: c["id"]), "discard": list(cb["discard"]),
                      "exhaust": list(cb["exhaust"]), "player": player,
                      "allies": [self._creature(a) for a in cb["allies"]],
                      "enemies": [self._creature(e, True) for e in cb["enemies"]],
                      "orbs": list(cb["orbs"]), "orb_slots": 3 if self.char == "Defect" else 0}
        g = {"act": self.act, "floor": self.floor, "total_floor": self.total_floor, "asc": self.asc,
             "gold": self.gold, "hp": max(self.hp, 0), "max_hp": self.max_hp, "char": self.char,
             "potion_slots": self.potion_slots, "room": getattr(self, "room", "Map"),
             "boss": "ENCOUNTER.FAKE_BOSS"}
        obs = {"g": g, "deck": list(self.deck), "relics": [dict(r) for r in self.relics],
               "potions": [{"id": p, "slot": s} for s, p in sorted(self.potions.items())],
               "combat": combat,
               "map": {"nodes": self.map_nodes, "edges": [list(e) for e in self.map_edges], "cur": self.map_cur},
               "ctx": self.ctx}
        info = {"act": self.act, "floor": self.floor, "total_floor": self.total_floor, "hp": max(self.hp, 0),
                "max_hp": self.max_hp, "gold": self.gold, "steps": self.steps, "warn": None,
                "save": self.save}
        return {"type": "rl", "done": self.done, "victory": self.victory, "decision": self.decision,
                "obs": obs, "cands": self.cands if not self.done else [], "info": info}


class FakeServer:
    """Protocol state machine around FakeGame (shared by the subprocess and in-process fakes)."""

    def __init__(self, p_error=0.0, seed=None, **game_kw):
        self.game = None
        self.broken = False
        self.p_error = p_error
        self.rng = random.Random(seed)
        self.game_kw = game_kw

    def handle(self, cmd):
        c = cmd.get("cmd")
        if c == "rl_start":
            self.game = FakeGame(**self.game_kw)
            self.broken = False
            return self.game.start(cmd.get("character", "Ironclad"), cmd.get("ascension", 0), str(cmd.get("seed", "0")))
        if c == "rl_act":
            if self.broken or self.game is None or self.game.done:
                return {"type": "error", "message": "no active run"}
            if self.rng.random() < self.p_error:
                self.broken = True
                return {"type": "error", "message": "FakeEngine injected error"}
            i = cmd.get("i")
            if not isinstance(i, int) or not 0 <= i < len(self.game.cands):
                return {"type": "error", "message": f"bad candidate index {i}"}
            return self.game.step(i)
        if c == "rl_save_get":
            if self.game is None or cmd.get("id") not in self.game.saves:
                return {"type": "error", "message": f"no save {cmd.get('id')}"}
            meta, state = self.game.saves[cmd["id"]]
            return {"type": "save", **meta, "data": base64.b64encode(pickle.dumps((meta, state))).decode()}
        if c == "rl_load" and cmd.get("data"):
            meta, state = pickle.loads(base64.b64decode(cmd["data"]))
            self.game = FakeGame(**self.game_kw)
            self.game.saves, self.game.save, self.game.save_id = {meta["id"]: (meta, state)}, None, meta["id"]
            self.broken = False
            return self.game.load(meta["id"])
        if c == "rl_load":
            save_id = cmd.get("id")
            if self.game is None or save_id not in self.game.saves:
                return {"type": "error", "message": f"no room save {save_id}"}
            self.broken = False
            return self.game.load(save_id)
        return {"type": "error", "message": f"unknown cmd {c}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p_error", type=float, default=0.0)
    ap.add_argument("--p_crash", type=float, default=0.0)
    ap.add_argument("--p_hang", type=float, default=0.0)
    ap.add_argument("--step_ms", type=float, default=0.0)
    ap.add_argument("--start_ms", type=float, default=0.0)
    ap.add_argument("--log_every", type=int, default=25, help="print a non-JSON log line every N commands")
    ap.add_argument("--rest_test", action="store_true")
    a = ap.parse_args()
    rng = random.Random()
    server = FakeServer(p_error=a.p_error, rest_test=a.rest_test)
    out = sys.stdout
    out.write("[fake] booting\n")
    out.write(json.dumps({"type": "ready", "version": "fake-0"}) + "\n")
    out.flush()
    n = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        cmd = json.loads(line)
        if cmd.get("cmd") == "quit":
            break
        n += 1
        if a.log_every and n % a.log_every == 0:
            out.write(f"[fake] handled {n} commands\n")
        if rng.random() < a.p_crash:
            sys.exit(3)
        if rng.random() < a.p_hang:
            time.sleep(3600)
        delay = a.start_ms if cmd.get("cmd") == "rl_start" else a.step_ms
        if delay:
            time.sleep(delay / 1000)
        out.write(json.dumps(server.handle(cmd), separators=(",", ":")) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
