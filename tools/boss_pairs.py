"""Act-3 boss pairs in the 11-20 death diagnosis (Ironclad 69.3%, 400 A10 games): how often each pair is reached, won,
and where the losses fall. The pair is fixed by the seed, so it is a property of the game, not of the play.
python3 boss_pairs.py <death_fork dir>"""
import json, sys, collections
d = sys.argv[1]
nodes = collections.defaultdict(list)
for l in open(f"{d}/nodes.jsonl"):
    n = json.loads(l); n.pop("actions", None); nodes[n["seed"]].append(n)
runs = {json.loads(l)["seed"]: json.loads(l) for l in open(f"{d}/runs.jsonl")}
res = {}
for l in open(f"{d}/results.jsonl"):
    r = json.loads(l)
    res[r["seed"]] = (sum(x["won"] for x in r["b"]) / len(r["b"]), sum(x["won"] for x in r["c"]) / len(r["c"]), r)
short = lambda e: e.split(".")[-1].replace("_BOSS", "").title()
pair = collections.defaultdict(lambda: [0, 0, 0, 0]); boss = collections.defaultdict(lambda: [0, 0, 0, 0])
for s, ns in nodes.items():
    b3 = [n for n in sorted(ns, key=lambda n: n["total_floor"]) if n["act"] == 3 and n["room"] == "Boss"]
    if not b3: continue
    win = runs[s]["victory"]
    first = short(b3[0]["encounter"])
    if len(b3) == 2: key = first + "+" + short(b3[1]["encounter"])
    else: key = first + "+?"
    p = pair[key]; p[0] += 1; p[1] += win
    for i, n in enumerate(b3):
        b = boss[(short(n["encounter"]), i + 1)]; b[0] += 1; b[1] += n["won"]
        if not n["won"] and s in res: b[2] += res[s][1] > 0.5; b[3] += res[s][1] <= 0.5
print("games reaching the act-3 bosses, by pair (second boss unknown when the first one killed):")
for k, v in sorted(pair.items(), key=lambda x: -x[1][0]): print(f"  {k:28s} n={v[0]:3d} won {v[1]:3d} ({100*v[1]/v[0]:.0f}%)")
print("each act-3 boss fight: fought, won, lost (of which a full-HP replay wins more than half / not)")
for k, v in sorted(boss.items()): print(f"  {k[0]:13s} as boss {k[1]}: fought {v[0]:3d}, won {100*v[1]/v[0]:5.1f}%, lost {v[0]-v[1]:2d} (HP {v[2]}, deck {v[3]})")
