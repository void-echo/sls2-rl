"""Where do the games that reach the last boss short of HP lose it? From tools/death_fork.py output (nodes.jsonl: every combat node
with entry/exit state, results.jsonl: the lost fights with 8 replays at the original HP (b) and 8 at full HP (c)).
Groups: win; lost at the act-3 boss with too little HP (full-HP replays win more than half, original-HP replays do not);
lost there with a deck that loses even at full HP; lost there although the original state wins more than half; lost earlier.
python3 hp_timeline.py <death_fork dir>"""
import json, sys, collections, statistics as st
d = sys.argv[1]
nodes = collections.defaultdict(list)
for l in open(f"{d}/nodes.jsonl"):
    n = json.loads(l); n.pop("actions", None); nodes[n["seed"]].append(n)
runs = {json.loads(l)["seed"]: json.loads(l) for l in open(f"{d}/runs.jsonl")}
cls = {}
for l in open(f"{d}/results.jsonl"):
    r = json.loads(l)
    wb = sum(x["won"] for x in r["b"]) / max(len(r["b"]), 1); wc = sum(x["won"] for x in r["c"]) / max(len(r["c"]), 1)
    kind = "unsteady" if wb > 0.5 else "lowhp" if wc > 0.5 else "deck"
    last = r["act"] == 3 and r["room"] == "Boss"
    cls[r["seed"]] = (kind if last else "earlier", r)
groups = collections.defaultdict(list)
for s, ns in nodes.items():
    ns.sort(key=lambda n: n["total_floor"])
    g = "win" if runs[s]["victory"] else ("a3boss_" + cls[s][0] if s in cls and cls[s][0] != "earlier" else "earlier")
    groups[g].append(ns)
def m(xs):
    xs = [x for x in xs if x is not None]
    return f"{st.mean(xs):6.1f}" if xs else "   -  "
def first(ns, f):
    return next((n for n in ns if f(n)), None)
order = ["win", "a3boss_lowhp", "a3boss_deck", "a3boss_unsteady", "earlier"]
print("games:", {g: len(groups[g]) for g in order})
rows = []
def row(name, fn):
    print(f"{name:44s}" + " ".join(f"{m([fn(ns) for ns in groups[g]]):>9s}" for g in order))
print(f"{'':44s}" + " ".join(f"{g.replace('a3boss_', 'a3:'):>9s}" for g in order))
for act in (1, 2, 3):
    row(f"act {act}: elites fought", lambda ns, a=act: sum(n["act"] == a and n["room"] == "Elite" for n in ns) if any(n["act"] == a for n in ns) else None)
    row(f"act {act}: HP lost in elites (net, total)", lambda ns, a=act: sum(n["entry"]["hp"] - n["exit"]["hp"] for n in ns if n["act"] == a and n["room"] == "Elite" and n["won"]) if any(n["act"] == a for n in ns) else None)
    row(f"act {act}: HP lost in normal fights (net, total)", lambda ns, a=act: sum(n["entry"]["hp"] - n["exit"]["hp"] for n in ns if n["act"] == a and n["room"] == "Monster" and n["won"]) if any(n["act"] == a for n in ns) else None)
    row(f"act {act}: potions used before the boss", lambda ns, a=act: sum(n["potions_used"] for n in ns if n["act"] == a and n["room"] != "Boss") if any(n["act"] == a for n in ns) else None)
    b = lambda ns, a=act: first(ns, lambda n: n["act"] == a and n["room"] == "Boss")
    row(f"act {act} boss: HP at the door", lambda ns: b(ns) and b(ns)["entry"]["hp"])
    row(f"act {act} boss: HP at the door / max HP (%)", lambda ns: b(ns) and 100 * b(ns)["entry"]["hp"] / b(ns)["entry"]["max_hp"])
    row(f"act {act} boss: potions at the door", lambda ns: b(ns) and len(b(ns)["entry"]["potions"]))
    row(f"act {act} boss: deck size / relics", lambda ns: b(ns) and b(ns)["entry"]["deck_size"])
    row(f"act {act} boss: relics", lambda ns: b(ns) and len(b(ns)["entry"]["relics"]))
    row(f"act {act} boss: value at the door", lambda ns: b(ns) and b(ns)["entry"]["value"])
    if act < 3:
        row(f"act {act} boss: HP lost (winners of it)", lambda ns: b(ns) and b(ns)["won"] and b(ns)["entry"]["hp"] - b(ns)["exit"]["hp"] or None)
b1 = lambda ns: first(ns, lambda n: n["act"] == 3 and n["room"] == "Boss")
b2 = lambda ns: (lambda bs: bs[1] if len(bs) > 1 else None)([n for n in ns if n["act"] == 3 and n["room"] == "Boss"])
row("act 3 boss 1: HP lost in it (survivors)", lambda ns: b1(ns) and b1(ns)["won"] and (b1(ns)["entry"]["hp"] - b1(ns)["exit"]["hp"]) or None)
row("act 3 boss 1: potions used in it", lambda ns: b1(ns) and b1(ns)["potions_used"])
row("act 3 boss 2: HP at the door", lambda ns: b2(ns) and b2(ns)["entry"]["hp"])
row("act 3 boss 2: HP at the door / max HP (%)", lambda ns: b2(ns) and 100 * b2(ns)["entry"]["hp"] / b2(ns)["entry"]["max_hp"])
row("act 3 boss 2: potions at the door", lambda ns: b2(ns) and len(b2(ns)["entry"]["potions"]))
for g in order[1:4]:
    died = collections.Counter("boss1" if b2(ns) is None else "boss2" for ns in groups[g])
    enc = collections.Counter((b2(ns) or b1(ns))["encounter"].split(".")[-1] for ns in groups[g])
    print(g, "died in", dict(died), "| fatal encounter", dict(enc.most_common(6)))
print("winners: act-3 boss pairs", dict(collections.Counter(b1(ns)["encounter"].split(".")[-1] + "+" + b2(ns)["encounter"].split(".")[-1] for ns in groups["win"] if b2(ns)).most_common(8)))
