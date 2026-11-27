"""Per-card drift of the identity embedding (table rows of k2('card.id', id)) between two checkpoints, with offer
counts from a trace to separate 'changed by the patch' from 'seen often'."""
import json, gzip, sys, collections, torch
sys.path.insert(0, ".")
from rl.encode import k2
from rl.analyze_prefs import name
def tab(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return ck["model"]["table"].float()
old, new = tab(sys.argv[1]), tab(sys.argv[2])
ids = [c for c in json.load(open("engine/content_ids_v0.113.1.json"))["cards"]] if isinstance(json.load(open("engine/content_ids_v0.113.1.json")), dict) else []
# offers / picks / plays in the new model's trace
off = collections.Counter(); seen = collections.Counter()
with gzip.open(sys.argv[3], "rt") as f:
    for line in f:
        r = json.loads(line)
        if r.get("type") != "step": continue
        if r["decision"] == "card_reward":
            for c in r["cands"]:
                if c.get("card"): off[c["card"]["id"]] += 1
        cb = r["state"].get("combat")
        if cb:
            for c in cb["hand"]: seen[c["id"]] += 1
rows = []
for cid in ids:
    k = k2("card.id", cid); a, b = (k >> 16) & 0xFFFF, k & 0xFFFF
    e0 = old[a] + old[b]; e1 = new[a] + new[b]
    d = (e1 - e0).norm().item(); rel = d / max(e0.norm().item(), 1e-6)
    rows.append((d, rel, cid, off[cid], seen[cid]))
rows.sort(reverse=True)
active = [r for r in rows if r[4] >= 200]
print(f"{len(ids)} card ids; {len(active)} appear in hand >= 200 times in the new model's 300 A10 games")
print("top drift among cards the Ironclad actually sees:")
for i, (d, rel, cid, o, s) in enumerate(active[:15]):
    print(f"  {i+1:2d}. {name(cid)[:28]:28s} drift {d:.3f} (rel {rel:.2f}) offered {o:4d} in-hand {s:6d}")
ranks = {r[2]: i + 1 for i, r in enumerate(active)}
for cid in ("CARD.FORGOTTEN_RITUAL", "CARD.EXPECT_A_FIGHT", "CARD.STRIKE_IRONCLAD", "CARD.BASH"):
    r = next((x for x in rows if x[2] == cid), None)
    print(f"  {cid}: drift {r[0]:.3f} rel {r[1]:.2f}, rank {ranks.get(cid, '-')}/{len(active)} among seen cards")
import statistics as st
json.dump({"old": sys.argv[1], "new": sys.argv[2], "trace": sys.argv[3], "rows": [{"card": cid, "name": name(cid), "drift": d, "rel": rel, "offered": o, "in_hand": s} for d, rel, cid, o, s in active]}, open(sys.argv[4], "w"), indent=0)
print("median drift of seen cards", round(st.median(r[0] for r in active), 3))
