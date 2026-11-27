"""Card picks, upgrades, shops and routes from rl/trace.py output (greedy traces); run from the repo root.
    python3 tools/prefs_extra.py <a>.jsonl.gz [<b>.jsonl.gz ...]"""
import collections, gzip, json, statistics as st, sys
sys.path.insert(0, ".")
from rl.analyze_prefs import name  # game localization: "中文（English）"

C = collections.Counter
def short(cid):
    n = name(cid)
    return n.split("（")[0] if "（" in n else n

def hpbin(info):
    f = info["hp"] / max(info["max_hp"], 1)
    return "<40%" if f < .4 else "40-70%" if f < .7 else "70-99%" if f < 1 else "满血"

def analyze(path):
    steps, eps = collections.defaultdict(list), {}
    for l in gzip.open(path, "rt"):
        d = json.loads(l)
        if d.get("type") == "step":
            steps[d["seed"]].append(d)
        elif d.get("type") == "episode":
            eps[d["seed"]] = d
    out = {}
    wins = [s for s, e in eps.items() if e.get("victory")]
    out["局数"] = len(eps); out["胜率"] = len(wins) / max(len(eps), 1)
    out["平均楼层"] = st.mean(e["info"]["total_floor"] for e in eps.values())
    # card rewards
    pick, skip, offered, taken = C(), C(), C(), C()
    types, rar = C(), C()
    for s, ss in steps.items():
        for d in ss:
            if d["decision"] != "card_reward":
                continue
            c = d["cands"][d["choice"]]
            if c["k"] not in ("reward_card", "skip"):
                continue
            act = d["info"]["act"]
            for x in d["cands"]:
                if x["k"] == "reward_card":
                    offered[x["card"]["id"]] += 1
            if c["k"] == "skip":
                skip[act] += 1
            else:
                pick[act] += 1; taken[c["card"]["id"]] += 1
                types[c["card"]["type"]] += 1; rar[c["card"]["rar"]] += 1
    out["拿牌"] = {"跳过率按幕": {a: round(skip[a] / max(skip[a] + pick[a], 1), 3) for a in (1, 2, 3)},
                 "所拿类型": dict(types), "所拿稀有度": dict(rar),
                 "最常拿": [(short(k), v, round(v / offered[k], 2)) for k, v in taken.most_common(12)],
                 "出现≥15次时拿的比例最高": sorted([(short(k), offered[k], round(taken[k] / offered[k], 2))
                                          for k in offered if offered[k] >= 15], key=lambda x: -x[2])[:10],
                 "出现≥15次时拿的比例最低": sorted([(short(k), offered[k], round(taken[k] / offered[k], 2))
                                          for k in offered if offered[k] >= 15], key=lambda x: x[2])[:10]}
    # rest sites and upgrades
    rest = collections.defaultdict(C); up_cards, up_basic, up_act = C(), C(), C()
    for s, ss in steps.items():
        for d in ss:
            if d["decision"] == "rest_site":
                c = d["cands"][d["choice"]]
                if c["k"] == "rest":
                    rest[hpbin(d["info"])][c.get("key")] += 1
            if d["decision"] == "card_select" and (d["state"].get("ctx") or {}).get("sel") == "deck_upgrade":
                c = d["cands"][d["choice"]]
                if c["k"] == "pick":
                    cid = c["card"]["id"]; up_cards[cid] += 1; up_act[d["info"]["act"]] += 1
                    up_basic["基础牌" if c["card"]["rar"] == "Basic" else "其他"] += 1
    out["篝火"] = {b: {k: v for k, v in rest[b].items()} for b in ("<40%", "40-70%", "70-99%", "满血")}
    out["升级"] = {"次数按幕": dict(up_act), "基础牌占比": round(up_basic["基础牌"] / max(sum(up_basic.values()), 1), 3),
                 "最常升级": [(short(k), v) for k, v in up_cards.most_common(12)]}
    # shops
    visits = []
    for s, ss in steps.items():
        cur = None
        for d in ss:
            if d["decision"] == "shop":
                key = (d["info"]["act"], d["info"]["floor"])
                if cur is None or cur["key"] != key:
                    cur = {"key": key, "gold_in": d["info"]["gold"], "buys": C(), "items": [], "remove_ok": False}
                    visits.append(cur)
                ctx = d["state"].get("ctx") or {}
                if any(x["k"] == "remove" for x in d["cands"]):
                    cur["remove_ok"] = True
                c = d["cands"][d["choice"]]
                if c["k"] in ("buy_card", "buy_relic", "buy_potion", "remove"):
                    cur["buys"][c["k"]] += 1
                    cur["items"].append((c["k"], (c.get("card") or {}).get("id") or c.get("id") or c.get("relic") or c.get("potion")))
                cur["gold_out"] = d["info"]["gold"]
            elif cur is not None and "gold_after" not in cur:
                cur["gold_after"] = d["info"]["gold"]
    removed = C()
    for s, ss in steps.items():
        for d in ss:
            if d["decision"] == "card_select" and (d["state"].get("ctx") or {}).get("sel") == "deck_remove":
                c = d["cands"][d["choice"]]
                if c["k"] == "pick":
                    removed[c["card"]["id"]] += 1
    nv = max(len(visits), 1)
    bought = C()
    for v in visits:
        for k, i in v["items"]:
            bought[(k, i)] += 1
    out["商店"] = {"进店次数/局": round(len(visits) / max(len(eps), 1), 2),
                 "进店时平均金币": round(st.mean(v["gold_in"] for v in visits), 1) if visits else None,
                 "平均花掉": round(st.mean(v["gold_in"] - v.get("gold_after", v.get("gold_out", v["gold_in"])) for v in visits), 1) if visits else None,
                 "什么都不买的比例": round(sum(not v["buys"] for v in visits) / nv, 3),
                 "每次进店购买": {k: round(sum(v["buys"][k] for v in visits) / nv, 2) for k in ("buy_card", "buy_relic", "buy_potion", "remove")},
                 "能删牌时删牌的比例": round(sum(v["buys"]["remove"] > 0 for v in visits if v["remove_ok"]) / max(sum(v["remove_ok"] for v in visits), 1), 3),
                 "删掉的牌": [(short(k), v) for k, v in removed.most_common(6)],
                 "最常买": [(k.replace("buy_", ""), short(i) if i else i, v) for (k, i), v in bought.most_common(14)]}
    # routes
    room_act, elite_av, elite_ch = collections.defaultdict(C), C(), C()
    for s, ss in steps.items():
        for d in ss:
            if d["decision"] != "map_select":
                continue
            c = d["cands"][d["choice"]]
            if c["k"] != "map":
                continue
            nodes = d["state"].get("map_nodes") or []
            t = nodes[c["node"]]["t"] if c["node"] < len(nodes) else "?"
            room_act[d["info"]["act"]][t] += 1
            opts = {nodes[x["node"]]["t"] for x in d["cands"] if x["k"] == "map" and x["node"] < len(nodes)}
            if "Elite" in opts and len(opts) > 1:
                b = hpbin(d["info"]); elite_av[b] += 1; elite_ch[b] += t == "Elite"
    n = max(len(eps), 1)
    out["路线（每局每幕平均进入）"] = {a: {t: round(v / n, 2) for t, v in room_act[a].most_common()} for a in (1, 2, 3)}
    out["有别的路可走时选精英的比例"] = {b: (elite_av[b], round(elite_ch[b] / max(elite_av[b], 1), 2)) for b in ("<40%", "40-70%", "70-99%", "满血")}
    # final decks
    fin = {}
    for s, ss in steps.items():
        nc = [d for d in ss if "deck" in d["state"]]
        if nc:
            fin[s] = nc[-1]["state"]["deck"]
    def deckstats(seeds):
        ds = [fin[s] for s in seeds if s in fin]
        if not ds:
            return None
        return {"局数": len(ds), "张数": round(st.mean(len(x) for x in ds), 1),
                "攻击": round(st.mean(sum(c["type"] == "Attack" for c in x) for x in ds), 1),
                "技能": round(st.mean(sum(c["type"] == "Skill" for c in x) for x in ds), 1),
                "能力": round(st.mean(sum(c["type"] == "Power" for c in x) for x in ds), 1),
                "已升级": round(st.mean(sum(c["up"] > 0 for c in x) for x in ds), 1),
                "打击剩": round(st.mean(sum(c["id"] == "CARD.STRIKE_IRONCLAD" for c in x) for x in ds), 1),
                "防御剩": round(st.mean(sum(c["id"] == "CARD.DEFEND_IRONCLAD" for c in x) for x in ds), 1)}
    out["局末牌组（赢）"] = deckstats(wins)
    out["局末牌组（输）"] = deckstats([s for s in eps if s not in wins])
    return out

for p in sys.argv[1:]:
    print("\n########", p)
    for k, v in analyze(p).items():
        print(f"{k}: {json.dumps(v, ensure_ascii=False)}")
