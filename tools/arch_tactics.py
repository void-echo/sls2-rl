"""Archetype dependence and end-of-turn leftovers from rl/trace.py output (greedy traces); run from the repo root.
    python3 tools/arch_tactics.py <trace.jsonl.gz> [more ...]"""
import collections, gzip, json, sys
sys.path.insert(0, ".")
from rl.analyze_prefs import name  # game localization: "中文（English）"

PS = "CARD.PERFECTED_STRIKE"


def short(cid):
    n = name(cid)
    return n.split("（")[0] if "（" in n else n


def rate(k, n):
    return f"{k}/{n} ({k / n:.0%})" if n else "0/0"


def analyze(path):
    games = collections.defaultdict(list)
    wins = {}
    for line in gzip.open(path, "rt"):
        d = json.loads(line)
        if d.get("type") == "step":
            games[d["seed"]].append(d)
        elif d.get("type") == "episode" and not d.get("aborted"):
            wins[d["seed"]] = bool(d.get("victory"))
    seeds = [s for s in games if s in wins]
    print(f"\n#### {path}: {len(seeds)} games, win {sum(wins[s] for s in seeds) / len(seeds):.1%}")

    # archetype dependence
    final, offered = {}, {}
    for s in seeds:
        ss = sorted(games[s], key=lambda d: d["step"])
        decks = [d["state"]["deck"] for d in ss if "deck" in d["state"]]
        final[s] = decks[-1] if decks else []
        offered[s] = any(c.get("card", {}).get("id") == PS for d in ss
                         if d["decision"] in ("card_reward", "shop") for c in d["cands"])
    has = [s for s in seeds if any(c["id"] == PS for c in final[s])]
    no = [s for s in seeds if s not in has]
    never = [s for s in seeds if not offered[s]]
    print("完美打击：局末有", rate(sum(wins[s] for s in has), len(has)), "胜；局末没有",
          rate(sum(wins[s] for s in no), len(no)), "胜；整局从没遇到", rate(sum(wins[s] for s in never), len(never)), "胜")
    present = collections.Counter(cid for s in seeds for cid in {c["id"] for c in final[s]})
    rows = []
    for cid, n in present.items():
        if 0.08 * len(seeds) <= n <= 0.92 * len(seeds) and not cid.endswith(("STRIKE_IRONCLAD", "DEFEND_IRONCLAD")):
            w_in = sum(wins[s] for s in seeds if any(c["id"] == cid for c in final[s])) / n
            w_out = sum(wins[s] for s in seeds if not any(c["id"] == cid for c in final[s])) / (len(seeds) - n)
            rows.append((w_in - w_out, short(cid), n, w_in, w_out))
    rows.sort()
    print("局末牌组出现率 8–92% 的牌，有这张牌的胜率 − 没有的胜率（最低 8 / 最高 8）：")
    for r in rows[:8] + [None] + rows[-8:]:
        print("   ...") if r is None else print(f"   {r[1]:10s} 出现 {r[2]:3d} 局  有 {r[3]:.0%}  无 {r[4]:.0%}  差 {r[0]:+.0%}")
    top = present.most_common(12)
    print("局末牌组最常见的牌（局数）：", "，".join(f"{short(c)} {n}" for c, n in top))

    # end-of-turn leftovers
    ends = playable_left = zero_left = 0
    zero_seen = zero_played = 0
    left_ids = collections.Counter()
    for s in seeds:
        for d in games[s]:
            cb = d["state"].get("combat")
            if d["decision"] != "combat_play" or not cb:
                continue
            hand = cb.get("hand") or []
            choice = d["cands"][d["choice"]]
            zero = [c for c in hand if c.get("playable") and c.get("cost") == 0]
            if zero:
                zero_seen += 1
                zero_played += choice.get("k") == "play" and hand[choice["hand"]].get("cost") == 0 \
                    if choice.get("k") == "play" and choice.get("hand", -1) < len(hand) else 0
            if choice.get("k") == "end_turn":
                ends += 1
                if any(c.get("playable") for c in hand):
                    playable_left += 1
                if zero:
                    zero_left += 1
                    left_ids.update(c["id"] for c in zero)
    print(f"结束回合 {ends} 次：手里还有可打的牌 {rate(playable_left, ends)}，还有可打的 0 费牌 {rate(zero_left, ends)}")
    print(f"手里有可打的 0 费牌的决策 {zero_seen} 个，其中这一步打出 0 费牌 {rate(zero_played, zero_seen)}")
    print("结束回合时留在手里的 0 费可打牌（次数）：", "，".join(f"{short(c)} {n}" for c, n in left_ids.most_common(10)))


for p in sys.argv[1:]:
    analyze(p)
