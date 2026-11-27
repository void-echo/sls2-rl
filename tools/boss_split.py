"""Second vs first act-3 boss at A10 from tools/death_fork.py output: entry HP, potions, deck, relics, encounter, classes."""
import json, collections, statistics as st, sys
D = sys.argv[1].rstrip("/") + "/"  # a tools/death_fork.py output directory
nodes = [json.loads(l) for l in open(D + "nodes.jsonl")]
res = {(r["seed"], r["total_floor"]): r for r in map(json.loads, open(D + "results.jsonl"))}
b = [n for n in nodes if n["room"] == "Boss" and n["act"] == 3]
hp = lambda n: n["entry"]["hp"] / max(n["entry"]["max_hp"], 1)
for fl, name in [(15, "first boss (floor 48)"), (16, "second boss (floor 49)")]:
    ns = [n for n in b if n["floor"] == fl]
    w = [n for n in ns if n["won"]]; l = [n for n in ns if not n["won"]]
    print(f"\n== {name}: {len(ns)} reached, won {len(w)} ({100 * len(w) / max(len(ns), 1):.1f}%)")
    print(f"   entry HP%: winners {100 * st.mean(map(hp, w)):.0f} losers {100 * st.mean(map(hp, l)):.0f} | abs HP win {st.mean(n['entry']['hp'] for n in w):.0f} lose {st.mean(n['entry']['hp'] for n in l):.0f} | max HP win {st.mean(n['entry']['max_hp'] for n in w):.0f} lose {st.mean(n['entry']['max_hp'] for n in l):.0f}")
    print(f"   value: winners {st.mean(n['entry']['value'] for n in w):.3f} losers {st.mean(n['entry']['value'] for n in l):.3f} | deck size win {st.mean(n['entry']['deck_size'] for n in w):.1f} lose {st.mean(n['entry']['deck_size'] for n in l):.1f} | relics win {st.mean(len(n['entry']['relics']) for n in w):.1f} lose {st.mean(len(n['entry']['relics']) for n in l):.1f}")
    pot = lambda n: len(n['entry']['potions'])
    print(f"   potions at entry: win {st.mean(map(pot, w)):.2f} lose {st.mean(map(pot, l)):.2f} | used in fight win {st.mean(n['potions_used'] for n in w):.2f} lose {st.mean(n['potions_used'] for n in l):.2f} | combat steps win {st.mean(n['combat_steps'] for n in w):.0f} lose {st.mean(n['combat_steps'] for n in l):.0f}")
    encs = sorted({n['encounter'] for n in ns})
    print("   by encounter (reached, win%):", {e.split('.')[-1]: (sum(n['encounter'] == e for n in ns), round(100 * sum(n['won'] for n in ns if n['encounter'] == e) / max(1, sum(n['encounter'] == e for n in ns)))) for e in encs})
    bins = [(0, .4), (.4, .6), (.6, .8), (.8, .95), (.95, 1.01)]
    print("   win% by entry HP (n, win%):", {f"{int(a * 100)}-{int(c * 100)}": (sum(a <= hp(n) < c for n in ns), round(100 * sum(n['won'] for n in ns if a <= hp(n) < c) / max(1, sum(a <= hp(n) < c for n in ns)))) for a, c in bins})
    print("   win% by potions at entry (n, win%):", {k: (sum(pot(n) == k for n in ns), round(100 * sum(n['won'] for n in ns if pot(n) == k) / max(1, sum(pot(n) == k for n in ns)))) for k in sorted({pot(n) for n in ns})})
    cls = collections.Counter(); br = []; cr = []
    for n in l:
        r = res.get((n["seed"], n["total_floor"]))
        if not r: continue
        x = sum(t["won"] for t in r["b"]) / max(len(r["b"]), 1); y = sum(t["won"] for t in r["c"]) / max(len(r["c"]), 1)
        br.append(x); cr.append(y)
        cls["unstable" if x >= .5 else ("hp" if y >= .5 else "state")] += 1
    if br:
        print(f"   failures replayed n={len(br)}: {dict(cls)}; sampled win at own HP {100 * st.mean(br):.1f}%, at full HP {100 * st.mean(cr):.1f}%")
# what happens between the bosses
pairs = collections.defaultdict(dict)
for n in b:
    pairs[n["seed"]][n["floor"]] = n
gap = [(p[15]["exit"]["hp"], p[16]["entry"]["hp"], p[15]["exit"]["max_hp"]) for p in pairs.values() if 15 in p and 16 in p and p[15]["exit"]]
print(f"\nbetween the bosses, n={len(gap)}: HP leaving boss 1 {st.mean(g[0] for g in gap):.1f}, entering boss 2 {st.mean(g[1] for g in gap):.1f}, max HP {st.mean(g[2] for g in gap):.1f}")
print("   potions leaving boss 1 -> entering boss 2:", round(st.mean(len(p[15]['exit']['potions']) if 'potions' in p[15]['exit'] else 0 for p in pairs.values() if 15 in p and 16 in p and p[15]['exit']), 2), round(st.mean(len(p[16]['entry']['potions']) for p in pairs.values() if 15 in p and 16 in p), 2))
same = [(p[15]['encounter'].split('.')[-1], p[16]['encounter'].split('.')[-1], p[16]['won']) for p in pairs.values() if 15 in p and 16 in p]
c = collections.defaultdict(list)
for a, d, x in same: c[(a, d)].append(x)
print("   boss order (n, second-boss win%):", {f"{a}->{d}": (len(v), round(100 * sum(v) / len(v))) for (a, d), v in sorted(c.items())})
