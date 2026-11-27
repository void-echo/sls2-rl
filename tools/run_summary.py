"""Readable summary of one recorded run (parity/record.py output, e.g. a model run recorded for a watch session): the route, HP entering elites and bosses, card
picks, upgrades, shops, rest choices, the final deck and relics. Usage: python3 summarize.py <run_dir> [...]"""
import collections, json, sys


def short(x):
    return (x or "").split(".")[-1].replace("_", " ").title()


def main(d):
    L = [json.loads(l) for l in open(f"{d}/headless.jsonl")]
    A = {a["step"]: a for a in (json.loads(l) for l in open(f"{d}/actions.jsonl"))}
    run = json.load(open(f"{d}/run.json"))
    print(f"== {run['id']}: {run['end']} after {run['steps']} steps (checkpoint {(run.get('ckpt') or 'replayed from an action log').split('/')[-1]})")
    rooms, last_room, picks, upgrades, rests, shop_buys, removes = [], None, [], [], [], [], []
    for x in L:
        g = x["obs"]["g"]
        key = (g["total_floor"], g["room"])
        if key != last_room and g["room"] in ("Elite", "Boss", "Monster"):
            enemies = [short(e.get("id")) for e in (x["obs"].get("combat") or {}).get("enemies", [])] if x["obs"].get("combat") else []
            rooms.append((g["act"], g["total_floor"], g["room"], g["hp"], g["max_hp"], enemies))
        last_room = key
        a = A.get(x["step"])
        if not a:
            continue
        c = a["cand"]
        if x["decision"] == "card_reward":
            picks.append((g["total_floor"], short(c.get("card", {}).get("id") if isinstance(c.get("card"), dict) else c.get("id") or c.get("k"))))
        if x["decision"] == "rest_site":
            rests.append((g["total_floor"], c.get("key") or c.get("k"), round(100 * g["hp"] / max(g["max_hp"], 1))))
        if x["decision"] == "shop" and c.get("k") not in ("leave", "proceed"):
            shop_buys.append((g["total_floor"], c.get("k"), short(c.get("id") or (c.get("card") or {}).get("id") if isinstance(c.get("card"), dict) else c.get("id"))))
    end = L[-2]["obs"] if len(L) > 1 else L[-1]["obs"]
    print("fights (act, floor, room, hp/max at the first decision, enemies):")
    for r in rooms:
        if r[2] != "Monster":
            print(f"   act {r[0]} floor {r[1]:2d} {r[2]:5s} {r[3]}/{r[4]}  {', '.join(r[5])}")
    print(f"   ({sum(1 for r in rooms if r[2] == 'Monster')} hallway fights)")
    print("card rewards (floor, choice):", "; ".join(f"{f}:{p}" for f, p in picks))
    print("rest sites (floor, choice, hp%):", "; ".join(f"{f}:{k}@{h}%" for f, k, h in rests))
    print("shop actions:", "; ".join(f"{f}:{k} {i}" for f, k, i in shop_buys) or "none")
    deck = collections.Counter(short(c.get("id")) + ("+" if c.get("upgraded") or c.get("up") else "") for c in end.get("deck", []))
    print(f"final deck ({sum(deck.values())}):", ", ".join(f"{k} x{v}" if v > 1 else k for k, v in sorted(deck.items())))
    print("relics:", ", ".join(short(r.get("id")) for r in end.get("relics", [])))


for d in sys.argv[1:]:
    main(d.rstrip("/"))
    print()
