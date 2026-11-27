"""Response R -> fixed-shape numpy arrays for the set/pointer model.

Every observed entity becomes a token. A token is a short list of *entries*; an entry is a hashed key
plus an optional number:
  * categorical field  ->  key = hash("card.id=CARD.BASH"), no number
  * numeric field      ->  key = hash("card.cost"),          number = 2
  * keyed number       ->  key = hash("pw=POWER.STRENGTH_POWER"), number = 3  (vars, powers, intents)
Keys are namespaced by field, hashed by two independent 16-bit hashes (one blake2b digest split in two)
and packed into one int32; the model sums both rows of a 2^16-row embedding table. Numbers are stored as
(sign*log1p|x|, clip(x/scale, -5, 5)). Fields the encoder does not know are flattened generically with the
same rules, so new game content or new protocol fields never break encoding.

Candidates are encoded the same way (their own entries) plus up to MAX_REF pointers into the token list:
  ref[0] primary entity (hand card / potion / map node / offered card / offered relic-or-potion)
  ref[1] target creature (enemy)
  ref[2:] members (cards of a bundle)
Inline cards in candidates become OFFER_CARD tokens encoded with the same card encoder as deck/hand cards.

Caps (see `Caps`) bound every group; tokens are added in priority order (globals, creatures, potions, hand,
offers, picked, relics, map, piles, deck) so referenced entities are never the ones truncated.
Truncation events are counted in `Encoder.trunc`.
"""
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, fields
from functools import lru_cache

import numpy as np

from .protocol import decision_id

# token types (0 = padding)
TT = dict(PAD=0, GLOBAL=1, COMBAT=2, CTX=3, DECK=4, HAND=5, DRAW=6, DISCARD=7, EXHAUST=8, OFFER_CARD=9,
          PICKED=10, RELIC=11, POTION=12, PLAYER=13, ALLY=14, ENEMY=15, ORB=16, MAPNODE=17, OFFER_ITEM=18, DRAW_TOP=19,
          ROOM=20)
N_TOKEN_TYPES = 24
HASH_ROWS = 1 << 16
LIN_CLIP = 5.0
NAN = float("nan")


@dataclass
class Caps:
    max_tok: int = 256     # tokens per observation
    max_ent: int = 24      # entries per token
    max_cand: int = 256    # candidates (crystal sphere can have 2 x 121)
    max_cent: int = 8      # entries per candidate
    max_ref: int = 8       # pointers per candidate (2 + up to 6 bundle members)
    deck: int = 80
    hand: int = 12
    draw: int = 128         # pile caps only bound each group; the total is max_tok
    discard: int = 128
    exhaust: int = 128
    offer: int = 96
    picked: int = 24
    relics: int = 48
    potions: int = 8
    allies: int = 4
    enemies: int = 8
    orbs: int = 12
    map: int = 100            # real act maps have ~70 nodes, all reachable at the start of an act
    deck_in_combat: bool = False   # master deck is ~ the union of the combat piles; skip it to save compute
    map_in_combat: bool = False
    map_paths: bool = False        # per map node: per room type, max/min count on one path onward and steps to the nearest
    pile_counts: bool = False      # deck and combat piles as multisets: one token per distinct card with its count
    enc_rev: int = 0
    rooms: int = 24
    rooms_in_combat: bool = False

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


def field_spec(c):
    """name -> (shape, dtype) of one encoded observation."""
    return {
        "tok_type": ((c.max_tok,), np.uint8),
        "tok_key": ((c.max_tok, c.max_ent), np.int32),
        "tok_val": ((c.max_tok, c.max_ent, 2), np.float16),
        "cand_key": ((c.max_cand, c.max_cent), np.int32),
        "cand_val": ((c.max_cand, c.max_cent, 2), np.float16),
        "cand_ref": ((c.max_cand, c.max_ref), np.int16),
        "cand_type": ((c.max_cand,), np.int16),
        "n_tok": ((), np.int16),
        "n_cand": ((), np.int16),
        "decision": ((), np.int8),
    }


@lru_cache(maxsize=None)
def hkey(s):
    d = hashlib.blake2b(s.encode(), digest_size=4, person=b"sls2rl-v0").digest()
    h1 = (d[0] | d[1] << 8) or 1  # row 0 is reserved for padding
    h2 = (d[2] | d[3] << 8) or 1
    return (h1 << 16) | h2


@lru_cache(maxsize=None)
def k2(a, b):
    return hkey(f"{a}={b}")


@lru_cache(maxsize=None)
def k1(a):
    return hkey(a)


def candidate_type(k):
    return k2("cand.k", k) % 32767 + 1


SCALES = {"hp": 100, "max_hp": 100, "gold": 100, "price": 100, "Gold": 100, "HpLoss": 20, "MaxHp": 20,
          "block": 20, "floor": 20, "total_floor": 20, "act": 4, "asc": 10, "energy": 5, "max_energy": 5,
          "round": 10, "cost": 3, "up": 1, "star": 3, "potion_slots": 5, "steps": 1000}


def _num(v):
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    return float(v)


def generic(ents, prefix, obj, skip=()):
    """Flatten unknown fields: str -> categorical, number/bool -> numeric, dict -> keyed, list -> bag."""
    for k, v in obj.items():
        if v is None or k in skip:
            continue
        if isinstance(v, (bool, int, float)):
            ents.append((k1(f"{prefix}.{k}"), _num(v), SCALES.get(k, 10.0)))
        elif isinstance(v, str):
            ents.append((k2(f"{prefix}.{k}", v), NAN, 1.0))
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (bool, int, float)):
                    ents.append((k2(f"{prefix}.{k}", kk), _num(vv), SCALES.get(kk, 10.0)))
                elif isinstance(vv, str):
                    ents.append((k2(f"{prefix}.{k}.{kk}", vv), NAN, 1.0))
        elif isinstance(v, list):
            for x in v[:8]:
                if isinstance(x, str):
                    ents.append((k2(f"{prefix}.{k}", x), NAN, 1.0))
                elif isinstance(x, dict) and isinstance(x.get("id"), str):
                    n = x.get("n", 1)
                    ents.append((k2(f"{prefix}.{k}", x["id"]), _num(n) if isinstance(n, (int, float)) else 1.0, 5.0))
            ents.append((k1(f"{prefix}.{k}.len"), float(len(v)), 10.0))


def own_state_ents(prefix, owner, d):
    """A model's own state (OwnState in the engine: fields its class declares) is keyed by the owner's id, since the
    same field name means different things in different classes. Text values (enums) become categorical keys."""
    ents = []
    for name, x in (d or {}).items():
        if isinstance(x, bool) or isinstance(x, (int, float)):
            ents.append((k2(prefix, f"{owner}/{name}"), _num(x), 10.0))
        elif isinstance(x, str):
            ents.append((k2(prefix, f"{owner}/{name}={x}"), NAN, 1.0))
    return ents


CARD_KNOWN = frozenset(("id", "type", "rar", "tgt", "ench", "ench_n", "aff", "aff_n", "up", "cost", "star",
                        "kw", "v", "pv", "playable", "d", "ench_d"))


def card_ents(c):
    ents = [(k2("card.id", c.get("id")), NAN, 1.0)]
    for f in ("type", "rar", "tgt"):
        x = c.get(f)
        if x is not None:
            ents.append((k2("card." + f, x), NAN, 1.0))
    for f, s in (("cost", 3.0), ("up", 1.0), ("star", 3.0)):
        x = c.get(f)
        if x is not None:
            ents.append((k1("card." + f), _num(x), s))
    for f in ("ench", "aff"):
        x = c.get(f)
        if x is not None:
            ents.append((k2("card." + f, x), NAN, 1.0))
            ents.append((k2("card." + f + "_n", x), _num(c.get(f + "_n") or 0), 5.0))
    if "playable" in c:
        ents.append((k1("card.playable"), 1.0 if c["playable"] else 0.0, 1.0))
    for x in c.get("kw") or ():
        ents.append((k2("card.kw", x), NAN, 1.0))
    for name, x in (c.get("v") or {}).items():
        if isinstance(x, (int, float)):
            ents.append((k2("card.v", name), _num(x), 10.0))
    for name, x in (c.get("pv") or {}).items():
        if isinstance(x, (int, float)):
            ents.append((k2("card.pv", name), _num(x), 10.0))
    ents.extend(own_state_ents("card.d", c.get("id"), c.get("d")))
    if c.get("ench_d"):
        ents.extend(own_state_ents("card.ench_d", c.get("ench"), c.get("ench_d")))
    if len(c) > len(CARD_KNOWN.intersection(c)):
        generic(ents, "card", c, CARD_KNOWN)
    return ents


RELIC_KNOWN = frozenset(("id", "v", "cnt", "status", "d", "cards"))
CREATURE_KNOWN = frozenset(("id", "hp", "max_hp", "block", "alive", "pw", "intents", "move", "then", "prev", "d", "done"))


def creature_ents(e, enc_rev=0):
    hp, mhp = _num(e.get("hp") or 0), _num(e.get("max_hp") or 0)
    ents = [(k2("cr.id", e.get("id")), NAN, 1.0), (k1("cr.hp"), hp, 100.0), (k1("cr.max_hp"), mhp, 100.0),
            (k1("cr.hp_frac"), hp / mhp if mhp > 0 else 0.0, 1.0),
            (k1("cr.block"), _num(e.get("block") or 0), 20.0),
            (k1("cr.alive"), 1.0 if e.get("alive", True) else 0.0, 1.0)]
    intents = e.get("intents")
    if intents:
        total = 0.0
        for it in intents:
            t = it.get("t")
            dmg, hits = it.get("dmg"), it.get("hits")
            ents.append((k2("intent", t), _num(dmg) if dmg is not None else 1.0, 10.0))
            if hits is not None:
                ents.append((k2("intent_hits", t), _num(hits), 3.0))
            if dmg is not None:
                total += _num(dmg) * (_num(hits) if hits is not None else 1.0)
            if len(it) > 3:
                generic(ents, f"intent.{t}", it, ("t", "dmg", "hits"))
        ents.append((k1("cr.intent_total"), total, 20.0))
    # move ids are only unique per monster: key them by monster and move (newest performed move first)
    cid = e.get("id")
    moves = []
    if isinstance(e.get("move"), str):
        moves.append((k2("cr.move", f"{cid}/{e['move']}"), NAN, 1.0))
    if isinstance(e.get("then"), str):
        moves.append((k2("cr.then", f"{cid}/{e['then']}"), NAN, 1.0))
    for i, mv in enumerate((e.get("prev") or [])[:3]):
        if isinstance(mv, str):
            moves.append((k2(f"cr.prev{i + 1}", f"{cid}/{mv}"), NAN, 1.0))
    moves.extend(own_state_ents("cr.d", cid, e.get("d")))
    for mv, n in (e.get("done") or {}).items():  # every move performed this combat, counted
        if isinstance(n, (int, float)):
            moves.append((k2("cr.done", f"{cid}/{mv}"), _num(n), 5.0))
    if enc_rev < 1:
        ents.extend(moves)
    pws = e.get("pw") or ()
    # a second instance of the same power (two The Bombs) gets its own keys, so amounts and vars stay paired
    seen, labels = Counter(), []
    for p in pws:
        pid = p.get("id")
        labels.append(pid if not seen[pid] else f"{pid}#{seen[pid]}")
        seen[pid] += 1
    for p, pid in zip(pws, labels):
        n = p.get("n", 1)
        ents.append((k2("pw", pid), _num(n) if n is not None else 1.0, 5.0))
    if enc_rev >= 1:
        ents.extend(moves)
    # a power's icon number, tooltip vars, private counters and held cards come after all powers
    for p, pid in zip(pws, labels):
        if isinstance(p.get("dn"), (int, float)):
            ents.append((k2("pw.dn", pid), _num(p["dn"]), 5.0))
        for name, x in (p.get("v") or {}).items():
            if isinstance(x, (int, float)):
                ents.append((k2("pw.v", f"{pid}/{name}"), _num(x), 10.0))
        ents.extend(own_state_ents("pw.d", pid, p.get("d")))
        for c in (p.get("cards") or ())[:4]:
            if isinstance(c, str):
                ents.append((k2("pw.card", f"{pid}/{c}" if enc_rev >= 1 else c), NAN, 1.0))
    if len(e) > len(CREATURE_KNOWN.intersection(e)):
        generic(ents, "cr", e, CREATURE_KNOWN)
    return ents


def _path_stats(types, edges):
    """Per node, over every path from it to the end of the map (the node itself excluded): per room type the
    largest and smallest count on one path and the steps to the nearest one, plus the number of paths.
    Returns (pmax, pmin, dist, npaths) lists of dicts / ints; a type missing from dist is unreachable."""
    n = len(types)
    children = [[] for _ in range(n)]
    for a, b in edges:
        if 0 <= a < n and 0 <= b < n and a != b:
            children[a].append(b)
    kinds = sorted({t for t in types if t is not None}, key=str)
    memo = [None] * n

    def go(i, depth=0):
        if memo[i] is None:
            memo[i] = ({}, {}, {}, 1)  # guards against cycles
            if children[i] and depth < 64:
                pmax, pmin, dist, npaths = {}, {}, {}, 0
                for ch in children[i]:
                    cmax, cmin, cdist, cn = go(ch, depth + 1)
                    npaths += cn
                    for t in kinds:
                        own = 1 if types[ch] == t else 0
                        hi, lo = cmax.get(t, 0) + own, cmin.get(t, 0) + own
                        pmax[t] = max(pmax.get(t, 0), hi)
                        pmin[t] = min(pmin[t], lo) if t in pmin else lo
                        d = 1 if own else (1 + cdist[t] if t in cdist else None)
                        if d is not None and (t not in dist or d < dist[t]):
                            dist[t] = d
                memo[i] = (pmax, pmin, dist, npaths)
        return memo[i]

    stats = [go(i) for i in range(n)]
    return [s[0] for s in stats], [s[1] for s in stats], [s[2] for s in stats], [s[3] for s in stats]


def _descendants(n_nodes, edges):
    """Bitset of all nodes reachable from each node (excluding itself)."""
    children = [[] for _ in range(n_nodes)]
    for a, b in edges:
        if 0 <= a < n_nodes and 0 <= b < n_nodes and a != b:
            children[a].append(b)
    memo = [None] * n_nodes

    def go(i, depth=0):
        if memo[i] is None:
            memo[i] = 0  # guards against cycles
            m = 0
            if depth < 64:
                for ch in children[i]:
                    m |= (1 << ch) | go(ch, depth + 1)
            memo[i] = m
        return memo[i]

    return [go(i) for i in range(n_nodes)]


def sphere_board_ents(cells):
    """Crystal Sphere board summary (only what the screen shows): hidden cells left, revealed cells per item."""
    seen = Counter((c.get("item") or "empty") for c in cells if not c.get("hidden"))
    ents = [(k1("sphere.hidden"), float(sum(1 for c in cells if c.get("hidden"))), 50.0)]
    for item, n in sorted(seen.items()):
        ents.append((k2("sphere.seen", item), float(n), 20.0))
    return ents


def sphere_cand_ents(cd, grid):
    """For one Crystal Sphere click: how many hidden cells it reveals (the big tool clears the 3x3 around the cell)
    and what the revealed cells within two steps of it show (items cover several adjacent cells)."""
    x, y = cd.get("x"), cd.get("y")
    if not isinstance(x, int) or not isinstance(y, int):
        return []
    r = 1 if cd.get("tool") == "big" else 0
    area = [grid.get((x + i, y + j)) for i in range(-r, r + 1) for j in range(-r, r + 1)]
    near = Counter()
    for i in range(-2, 3):
        for j in range(-2, 3):
            c = grid.get((x + i, y + j))
            if (i or j) and c is not None and not c.get("hidden"):
                near[c.get("item") or "empty"] += 1
    ents = [(k1("sphere.area_hidden"), float(sum(1 for c in area if c is not None and c.get("hidden"))), 5.0)]
    for item, n in sorted(near.items()):
        ents.append((k2("sphere.near", item), float(n), 5.0))
    return ents


class Encoder:
    def __init__(self, caps=None):
        self.c = caps if isinstance(caps, Caps) else Caps.from_dict(caps)
        self.spec = field_spec(self.c)
        self.trunc = Counter()

    def alloc(self, n=None):
        lead = () if n is None else (n,)
        return {k: np.zeros(lead + shape, dt) for k, (shape, dt) in self.spec.items()}

    def encode(self, R, out=None):
        c = self.c
        if out is None:
            out = self.alloc()
        else:
            for a in out.values():
                a[...] = 0
        obs = R.get("obs") or {}
        g = obs.get("g") or {}
        combat = obs.get("combat")
        cands = R.get("cands") or []
        if len(cands) > c.max_cand:
            self.trunc["cands"] += 1
            cands = cands[: c.max_cand]
        deck = obs.get("deck") or []
        relics = obs.get("relics") or []
        potions = obs.get("potions") or []
        ctx = obs.get("ctx") or {}
        decision = R.get("decision") or "unknown"

        types, toks = [], []

        def add(tt, ents):
            if len(types) >= c.max_tok:
                self.trunc["tokens"] += 1
                return -1
            types.append(tt)
            toks.append(ents)
            return len(types) - 1

        def group(items, cap, name):
            if len(items) > cap:
                self.trunc[name] += 1
                return items[:cap]
            return items

        # globals
        ents = [(k2("decision", decision), NAN, 1.0), (k1("n_deck"), float(len(deck)), 10.0),
                (k1("n_relics"), float(len(relics)), 10.0), (k1("n_potions"), float(len(potions)), 5.0),
                (k1("n_cands"), float(len(R.get("cands") or ())), 10.0)]
        generic(ents, "g", g, ("act_rooms",) if c.enc_rev >= 1 else ())
        if c.enc_rev >= 1 and g.get("act_rooms") is not None:
            ents.append((k1("g.act_rooms.len"), float(len(g["act_rooms"])), 10.0))
        add(TT["GLOBAL"], ents)
        ctx_cards = []
        cells = ctx.get("cells") if isinstance(ctx.get("cells"), list) else None
        grid = {(c.get("x"), c.get("y")): c for c in cells if isinstance(c, dict)} if cells else None
        ents = [(k2("ctx.decision", decision), NAN, 1.0)]
        for k, v in ctx.items():
            if k == "cells" and grid is not None:
                ents.extend(sphere_board_ents(cells))
            elif k == "unavailable":  # shelf items: their own tokens below
                ents.append((k1("ctx.unavailable.len"), float(len(v or ())), 10.0))
            elif isinstance(v, list) and v and isinstance(v[0], dict) and "type" in v[0] and "id" in v[0]:
                ctx_cards.extend(v)
                ents.append((k1(f"ctx.{k}.len"), float(len(v)), 10.0))
            elif v is not None:
                generic(ents, "ctx", {k: v})
        add(TT["CTX"], ents)

        hand_tok, enemy_tok, potion_tok, node_tok = [], [], {}, {}
        if combat is not None:
            ents = []
            for pile in ("hand", "draw", "discard", "exhaust"):
                ents.append((k1(f"cb.n_{pile}"), float(len(combat.get(pile) or ())), 10.0))
            generic(ents, "cb", combat, ("hand", "draw", "discard", "exhaust", "player", "allies", "enemies", "orbs",
                                         "draw_top"))
            impaled = {}
            if c.enc_rev >= 2:
                # enc_rev 2 (v0.113.1, the Fool): each arrow's status as a count per status, and how many arrows sit
                # in each enemy (generic() keeps only the list length: arrow dicts carry no "id")
                for a in combat.get("arrows") or ():
                    if not isinstance(a, dict):
                        continue
                    st = str(a.get("status"))
                    ents.append((k2("cb.arrow", st), NAN, 1.0))
                    if a.get("destroy_after_impaled"):
                        ents.append((k2("cb.arrow_destroy", st), NAN, 1.0))
                    t = a.get("tgt")
                    if isinstance(t, int) and t >= 0:
                        impaled[t] = impaled.get(t, 0) + 1
            add(TT["COMBAT"], ents)
            if combat.get("player"):
                add(TT["PLAYER"], creature_ents(combat["player"], c.enc_rev))
            for i, e in enumerate(group(combat.get("enemies") or [], c.enemies, "enemies")):
                ents = creature_ents(e, c.enc_rev)
                if impaled.get(i):
                    ents.append((k1("cr.arrows_in"), float(impaled[i]), 3.0))
                enemy_tok.append(add(TT["ENEMY"], ents))
            for a in group(combat.get("allies") or [], c.allies, "allies"):
                add(TT["ALLY"], creature_ents(a, c.enc_rev))
            for o in group(combat.get("orbs") or [], c.orbs, "orbs"):
                ents = [(k2("orb.id", o.get("id")), NAN, 1.0)]
                generic(ents, "orb", o, ("id",))
                add(TT["ORB"], ents)

        for p in group(potions, c.potions, "potions"):
            ents = [(k2("potion.id", p.get("id")), NAN, 1.0)]
            generic(ents, "potion", p, ("id",))
            potion_tok[p.get("slot")] = add(TT["POTION"], ents)

        if combat is not None:
            for i, card in enumerate(group(combat.get("hand") or [], c.hand, "hand")):
                ents = card_ents(card)
                ents.append((k1("hand.pos"), float(i), 10.0))
                hand_tok.append(add(TT["HAND"], ents))

        # offers referenced by candidates (inline cards / items)
        primary = [-1] * len(cands)
        members = [None] * len(cands)
        n_offer = 0
        offer_cards = {}

        def offer_card(card):
            nonlocal n_offer
            key = json.dumps(card, sort_keys=True) if c.enc_rev >= 1 else None
            if key is not None and key in offer_cards:
                return offer_cards[key]
            if n_offer >= c.offer:
                self.trunc["offer"] += 1
                return -1
            token = add(TT["OFFER_CARD"], card_ents(card))
            n_offer += 1
            if key is not None:
                offer_cards[key] = token
            return token

        for j, cd in enumerate(cands):
            if "card" in cd and isinstance(cd["card"], dict):
                primary[j] = offer_card(cd["card"])
            if isinstance(cd.get("cards"), list):
                ms = []
                for card in cd["cards"][: c.max_ref - 2]:
                    if c.enc_rev >= 1 or n_offer < c.offer:
                        ms.append(offer_card(card))
                    else:
                        self.trunc["offer"] += 1
                members[j] = ms
            if isinstance(cd.get("id"), str) and primary[j] < 0 and n_offer < c.offer:
                primary[j] = add(TT["OFFER_ITEM"], [(k2("item.id", cd["id"]), NAN, 1.0), (k2("item.k", cd.get("k")), NAN, 1.0)])
                n_offer += 1

        for card in group(ctx_cards, c.picked, "picked"):
            add(TT["PICKED"], card_ents(card))

        # shop shelf items that are not candidates (too expensive, or a potion the belt cannot take)
        for it in (ctx.get("unavailable") or [])[:24]:
            if not isinstance(it, dict):
                continue
            tail = [(k1("shop.unavailable"), 1.0, 1.0), (k1("shop.price"), _num(it.get("price") or 0), 100.0)]
            if isinstance(it.get("card"), dict):
                add(TT["OFFER_CARD"], card_ents(it["card"]) + tail)
            elif isinstance(it.get("id"), str):
                add(TT["OFFER_ITEM"], [(k2("item.id", it["id"]), NAN, 1.0), (k2("item.k", it.get("k")), NAN, 1.0)] + tail)

        for r in group(relics, c.relics, "relics"):
            rid = r.get("id")
            ents = [(k2("relic.id", rid), NAN, 1.0)]
            for name, x in (r.get("v") or {}).items():
                if isinstance(x, (int, float)):
                    ents.append((k2("relic.v", name), _num(x), 10.0))
            if isinstance(r.get("cnt"), (int, float)):  # the number on the relic bar
                ents.append((k2("relic.cnt", rid), _num(r["cnt"]), 10.0))
            if isinstance(r.get("status"), str):
                ents.append((k2("relic.status", r["status"]), NAN, 1.0))
            ents.extend(own_state_ents("relic.d", rid, r.get("d")))
            for cid in (r.get("cards") or ())[:12]:
                if isinstance(cid, str):
                    ents.append((k2("relic.card", cid), NAN, 1.0))
            if len(r) > len(RELIC_KNOWN.intersection(r)):
                generic(ents, "relic", r, RELIC_KNOWN)
            add(TT["RELIC"], ents)

        if combat is None or c.map_in_combat:
            node_tok = self._map_tokens(obs.get("map") or {}, cands, add)

        if c.enc_rev >= 1 and (combat is None or c.rooms_in_combat):
            for ago, room in enumerate(group(list(reversed(g.get("act_rooms") or [])), c.rooms, "rooms"), 1):
                ents = [(k2("room.t", room.get("t")), NAN, 1.0)]
                if room.get("id") is not None:
                    ents.append((k2("room.id", room["id"]), NAN, 1.0))
                ents.append((k1("room.ago"), float(ago), 10.0))
                add(TT["ROOM"], ents)

        if combat is not None:
            # the known top of the draw pile, in order (0 = the next card drawn); the pile itself is a multiset
            for i, card in enumerate((combat.get("draw_top") or [])[: c.hand]):
                ents = card_ents(card)
                ents.append((k1("draw.top"), float(i), 5.0))
                add(TT["DRAW_TOP"], ents)
            for pile, tt in (("draw", "DRAW"), ("discard", "DISCARD"), ("exhaust", "EXHAUST")):
                self._pile_tokens(combat.get(pile) or [], getattr(c, pile), pile, TT[tt], add, group)
        if combat is None or c.deck_in_combat:
            self._pile_tokens(deck, c.deck, "deck", TT["DECK"], add, group)

        # candidates
        cand_ents, refs = [], np.full((len(cands), c.max_ref), -1, np.int16)
        for j, cd in enumerate(cands):
            k = cd.get("k")
            out["cand_type"][j] = candidate_type(k)
            ents = [(k2("cand.k", k), NAN, 1.0)]
            for f, v in cd.items():
                if f in ("k", "card", "cards", "src"):
                    continue
                if f == "hand":
                    refs[j, 0] = hand_tok[v] if isinstance(v, int) and 0 <= v < len(hand_tok) else -1
                elif f == "tgt":
                    if isinstance(v, int) and 0 <= v < len(enemy_tok):
                        refs[j, 1] = enemy_tok[v]
                elif f == "node":
                    refs[j, 0] = node_tok.get(v, -1)
                elif f == "slot" and k in ("potion", "discard_potion"):
                    refs[j, 0] = potion_tok.get(v, -1)
                elif f == "id":
                    ents.append((k2("cand.id", v), NAN, 1.0))
                elif f == "relic_d" and c.enc_rev >= 1:
                    ents.extend(own_state_ents("relic.d", cd.get("relic"), v))
                else:
                    generic(ents, "cand", {f: v})
            if k == "sphere" and grid is not None:
                ents.extend(sphere_cand_ents(cd, grid))
            if primary[j] >= 0:
                refs[j, 0] = primary[j]
            if members[j]:
                refs[j, 2: 2 + len(members[j])] = members[j]
            cand_ents.append(ents)

        n_tok = len(types)
        out["tok_type"][:n_tok] = types
        self._fill(toks, c.max_ent, out["tok_key"], out["tok_val"], "ent")
        n_cand = len(cands)
        if n_cand:
            self._fill(cand_ents, c.max_cent, out["cand_key"], out["cand_val"], "cent")
            out["cand_ref"][:n_cand] = refs
        out["n_tok"][...] = n_tok
        out["n_cand"][...] = n_cand
        out["decision"][...] = decision_id(decision)
        return out

    def _pile_tokens(self, cards, cap, name, tt, add, group):
        """A pile's cards. With caps.pile_counts the pile is a multiset (the draw pile's order is hidden, and the
        piles' order is not encoded anyway): cards identical in everything the observation shows share one token
        with their count, so a deck of 80 cloned Perfected Strikes is one token, whatever the deck size."""
        if not self.c.pile_counts:
            for card in group(cards, cap, name):
                add(tt, card_ents(card))
            return
        counts, first = Counter(), {}
        for card in cards:
            key = json.dumps(card, sort_keys=True)
            counts[key] += 1
            first.setdefault(key, card)
        for key in group(list(first), cap, name):
            ents = card_ents(first[key])
            ents.append((k1("card.count"), float(counts[key]), 5.0))
            add(tt, ents)

    def _map_tokens(self, m, cands, add):
        nodes = m.get("nodes") or []
        if not nodes:
            return {}
        n = len(nodes)
        cur = m.get("cur", -1)
        cur = cur if isinstance(cur, int) and 0 <= cur < n else -1
        desc = _descendants(n, m.get("edges") or [])
        legal = {cd["node"] for cd in cands if cd.get("k") == "map" and isinstance(cd.get("node"), int)}
        if cur >= 0:
            keep = desc[cur] | (1 << cur)
            for j in legal:
                if 0 <= j < n:
                    keep |= (1 << j) | desc[j]
            idx = [i for i in range(n) if keep >> i & 1]
        else:
            idx = list(range(n))
        idx.sort(key=lambda i: (nodes[i].get("r", 0), nodes[i].get("c", 0)))
        if len(idx) > self.c.map:
            self.trunc["map"] += 1
            idx = [i for i in idx if i in legal or i == cur] + [i for i in idx if i not in legal and i != cur]
            idx = idx[: self.c.map]
        type_mask = Counter()
        for i, nd in enumerate(nodes):
            type_mask[nd.get("t")] |= 1 << i
        cur_r = nodes[cur].get("r", 0) if cur >= 0 else 0
        if self.c.map_paths:
            pmax, pmin, dist, npaths = _path_stats([nd.get("t") for nd in nodes], m.get("edges") or [])
        out = {}
        for i in idx:
            nd = nodes[i]
            r = nd.get("r", 0)
            ents = [(k2("node.t", nd.get("t")), NAN, 1.0), (k1("node.r"), _num(r), 10.0),
                    (k1("node.c"), _num(nd.get("c", 0)), 5.0), (k1("node.is_cur"), float(i == cur), 1.0),
                    (k1("node.is_next"), float(i in legal), 1.0), (k1("node.dr"), _num(r - cur_r), 5.0)]
            d = desc[i]
            for t, mask in type_mask.items():
                cnt = (d & mask).bit_count()
                if cnt:
                    ents.append((k2("node.down", t), float(cnt), 5.0))
            if self.c.map_paths:
                # what the paths onward from this node offer: per room type the most and the fewest on one path
                # (fewest only when every path has one) and the steps to the nearest; and how many paths there are
                ents.append((k1("node.npaths"), float(npaths[i]), 100.0))
                for t in sorted(pmax[i], key=str):
                    if pmax[i][t]:
                        ents.append((k2("node.pmax", t), float(pmax[i][t]), 5.0))
                    if pmin[i].get(t):
                        ents.append((k2("node.pmin", t), float(pmin[i][t]), 5.0))
                    if t in dist[i]:
                        ents.append((k2("node.dist", t), float(dist[i][t]), 5.0))
            if len(nd) > 3:
                generic(ents, "node", nd, ("c", "r", "t"))
            out[i] = add(TT["MAPNODE"], ents)
        return out

    def _fill(self, lists, cap, key_out, val_out, name):
        ti, ei, flat = [], [], []
        for t, ents in enumerate(lists):
            if len(ents) > cap:
                self.trunc[name] += 1
                ents = ents[:cap]
            L = len(ents)
            flat.extend(ents)
            ti.extend([t] * L)
            ei.extend(range(L))
        if not flat:
            return
        a = np.array(flat, dtype=np.float64)
        v = a[:, 1]
        nan = np.isnan(v)
        v = np.where(nan, 0.0, v)
        key_out[ti, ei] = a[:, 0].astype(np.uint32).view(np.int32)
        val_out[ti, ei, 0] = np.sign(v) * np.log1p(np.abs(v))
        val_out[ti, ei, 1] = np.clip(v / a[:, 2], -LIN_CLIP, LIN_CLIP)
