"""Constants of the engine RL protocol (engine/RL_PROTOCOL.md, protocol 1 = harness v1) and a loose schema checker.

The checker is used by the tests on FakeEngine output and by `python -m rl.engine --check`
on the real engine, so protocol drift shows up as readable messages instead of encoder crashes.
"""

# v0 decision kinds (FakeEngine covers all of them; their ids are stable).
DECISIONS = (
    "combat_play", "map_select", "card_reward", "card_select", "bundle_select", "event_choice",
    "rest_site", "shop", "treasure", "crystal_sphere", "unknown", "game_over",
)
# Decision kinds added by harness v1 (ids follow the v0 ones). The v1 engine never emits "unknown".
V1_DECISIONS = ("rewards", "fake_merchant")
ALL_DECISIONS = DECISIONS + V1_DECISIONS
DECISION_ID = {d: i for i, d in enumerate(ALL_DECISIONS)}
OTHER_DECISION = len(ALL_DECISIONS)  # id for kinds the engine adds later
N_DECISION_IDS = len(ALL_DECISIONS) + 1

CAND_KINDS = (
    "play", "potion", "discard_potion", "end_turn", "map", "reward_card", "skip", "pick", "confirm",
    "bundle", "option", "rest", "buy_card", "buy_relic", "buy_potion", "remove", "treasure", "leave",
    "proceed", "sphere",
    # harness v1
    "reward", "reroll", "sacrifice", "alt", "open",
)
REWARD_TYPES = ("Gold", "Potion", "Relic", "Card", "SpecialCard", "RemoveCard", "Linked")

CHARACTERS = ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder")


def decision_id(kind):
    return DECISION_ID.get(kind, OTHER_DECISION)


def decision_name(i):
    return ALL_DECISIONS[i] if i < len(ALL_DECISIONS) else "other"


def _check_card(c, where, out):
    if not isinstance(c, dict) or not isinstance(c.get("id"), str):
        out.append(f"{where}: card without string id: {str(c)[:80]}")
        return
    for k in ("v", "pv"):
        if k in c and c[k] is not None and not isinstance(c[k], dict):
            out.append(f"{where}: card.{k} is not a dict")


def _check_creature(c, where, out):
    if not isinstance(c, dict):
        out.append(f"{where}: creature is not a dict")
        return
    for k in ("hp", "max_hp"):
        if not isinstance(c.get(k), (int, float)):
            out.append(f"{where}: creature.{k} missing or not numeric")
    for p in c.get("pw") or []:
        if not isinstance(p, dict) or "id" not in p:
            out.append(f"{where}: bad power entry {str(p)[:60]}")


def check_response(R):
    """Return a list of human-readable problems (empty list = looks valid)."""
    out = []
    if not isinstance(R, dict):
        return ["response is not a JSON object"]
    t = R.get("type")
    if t == "error":
        return [] if "message" in R else ["error response without message"]
    if t != "rl":
        return [f"unexpected type {t!r}"]
    for k in ("done", "decision", "obs", "cands", "info"):
        if k not in R:
            out.append(f"missing key {k}")
    if out:
        return out
    cands, obs, done = R["cands"], R["obs"], R["done"]
    if not isinstance(cands, list):
        return ["cands is not a list"]
    info = R.get("info") or {}
    if done:
        if cands:
            out.append("done but cands not empty")
        if not isinstance(R.get("victory"), bool):
            out.append("done but victory is not a bool")
        elif R["victory"] is False and info.get("hp", 0) > 0 and not info.get("abandoned"):
            # a loss is a death by the game's rules (hp 0) or an abandon (info.abandoned)
            out.append(f"defeat reported with hp {info.get('hp')} and abandoned={info.get('abandoned')}")
    elif not cands:
        out.append("not done but no cands")
    if R.get("decision") == "unknown" and info.get("abandoned") is not None:
        out.append("v1 engine reported decision 'unknown'")
    if not isinstance(obs, dict):
        return out + ["obs is not a dict"]
    for k in ("g", "deck", "relics", "potions", "map"):
        if k not in obs:
            out.append(f"obs missing {k}")
    for i, c in enumerate(obs.get("deck") or []):
        _check_card(c, f"deck[{i}]", out)
    combat = obs.get("combat")
    hand, enemies = [], []
    if combat is not None:
        hand = combat.get("hand") or []
        enemies = combat.get("enemies") or []
        for pile in ("hand", "draw", "discard", "exhaust"):
            for i, c in enumerate(combat.get(pile) or []):
                _check_card(c, f"{pile}[{i}]", out)
        if "player" in combat:
            _check_creature(combat["player"], "player", out)
        for i, e in enumerate(enemies):
            _check_creature(e, f"enemy[{i}]", out)
    nodes = (obs.get("map") or {}).get("nodes") or []
    slots = {p.get("slot") for p in obs.get("potions") or [] if isinstance(p, dict)}
    for j, c in enumerate(cands):
        if not isinstance(c, dict) or not isinstance(c.get("k"), str):
            out.append(f"cand[{j}] has no string k")
            continue
        if "hand" in c and not (0 <= c["hand"] < len(hand)):
            out.append(f"cand[{j}] {c['k']}: hand index {c['hand']} out of range ({len(hand)})")
        if "tgt" in c and c["tgt"] is not None and c["tgt"] != -1 and not (0 <= c["tgt"] < len(enemies)):
            out.append(f"cand[{j}] {c['k']}: tgt {c['tgt']} out of range ({len(enemies)})")
        if "node" in c and not (0 <= c["node"] < len(nodes)):
            out.append(f"cand[{j}] map: node {c['node']} out of range ({len(nodes)})")
        if c["k"] in ("potion", "discard_potion") and c.get("slot") not in slots:
            out.append(f"cand[{j}] {c['k']}: slot {c.get('slot')} not among filled slots {sorted(slots)}")
        if c["k"] == "reward" and not isinstance(c.get("idx"), int):
            out.append(f"cand[{j}] reward without int idx")
        # a card reward shows its cards inline (protocol: no navigation cycles; opening it is never needed to look)
        if c["k"] == "reward" and c.get("rt") == "Card" and not isinstance(c.get("cards"), list):
            out.append(f"cand[{j}] card reward without inline cards")
        if "alts" in c and not (isinstance(c["alts"], list) and all(isinstance(a, str) for a in c["alts"])):
            out.append(f"cand[{j}] {c['k']}: alts is not a list of strings")
        if "card" in c:
            _check_card(c["card"], f"cand[{j}].card", out)
        for i, cc in enumerate(c.get("cards") or []):
            _check_card(cc, f"cand[{j}].cards[{i}]", out)
    return out
