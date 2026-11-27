"""Card-reward picks and their link to winning, from rl/trace.py output (one character per trace).

    python3 tools/pick_stats.py a.jsonl.gz [b.jsonl.gz ...] --out picks.json      # on the machine that holds the traces
    python3 tools/pick_stats.py --show picks.json [--top 20]                       # table with localized card names

Per card: how often it was offered in a card reward, how often it was taken, shop purchases, and the win rate of
the games that took it at least once against the games that were offered it but never took it. Both groups reached
an offer of the card, which removes most of the "late cards are only seen by games that survive" bias; what is left is
an association, not the effect of the card (the policy takes a card when the rest of the deck suits it).
"""
import argparse
import collections
import gzip
import json
import math
import sys


def analyze(path):
    steps, eps, meta = collections.defaultdict(list), {}, {}
    for line in gzip.open(path, "rt"):
        d = json.loads(line)
        if d.get("type") == "step":
            if d["decision"] in ("card_reward", "shop"):
                c = d["cands"][d["choice"]]
                offered = [x["card"]["id"] for x in d["cands"] if x.get("k") == "reward_card"] if d["decision"] == "card_reward" else []
                card = (c.get("card") or {}).get("id") if isinstance(c.get("card"), dict) else None
                steps[d["seed"]].append((d["decision"], c.get("k"), card, offered, d["info"].get("act")))
        elif d.get("type") == "episode":
            eps[d["seed"]] = d
        elif d.get("type") == "meta":
            meta = d
    done = {s: bool(e.get("victory")) for s, e in eps.items() if not e.get("aborted")}
    offers, picks, buys = collections.Counter(), collections.Counter(), collections.Counter()
    saw, took = collections.defaultdict(set), collections.defaultdict(set)
    n_reward = n_skip = 0
    for seed, rows in steps.items():
        if seed not in done:
            continue
        for decision, kind, card, offered, act in rows:
            if decision == "card_reward" and kind in ("reward_card", "skip"):
                n_reward += 1
                n_skip += kind == "skip"
                for x in offered:
                    offers[x] += 1
                    saw[x].add(seed)
                if kind == "reward_card":
                    picks[card] += 1
                    took[card].add(seed)
            elif decision == "shop" and kind == "buy_card" and card:
                buys[card] += 1
    cards = {}
    for c in offers:
        t, s = took[c], saw[c] - took[c]
        cards[c] = dict(offers=offers[c], picks=picks[c], buys=buys[c], games_took=len(t), wins_took=sum(done[x] for x in t),
                        games_passed=len(s), wins_passed=sum(done[x] for x in s))
    return dict(trace=path, character=meta.get("character"), ascension=meta.get("ascension"), ckpt=meta.get("ckpt"),
                games=len(done), wins=sum(done.values()), card_rewards=n_reward, skips=n_skip, cards=cards)


def show(path, top):
    sys.path.insert(0, ".")
    from rl.analyze_prefs import name
    for r in json.load(open(path)):
        base = r["wins"] / max(r["games"], 1)
        print(f"\n== {r['character']} A{r['ascension']}: {r['games']} 局，胜率 {100 * base:.1f}%，卡牌奖励 {r['card_rewards']} 次，跳过 {100 * r['skips'] / max(r['card_rewards'], 1):.0f}%")
        print("| # | 牌 | 拿的次数 | 出现次数 | 拿取率 | 拿了的局胜率 | 见过没拿的局胜率 | 差（±标准误） |")
        print("|---:|---|---:|---:|---:|---:|---:|---:|")
        rows = sorted(r["cards"].items(), key=lambda kv: -kv[1]["picks"])[:top]
        for i, (cid, c) in enumerate(rows, 1):
            a = c["wins_took"] / max(c["games_took"], 1)
            b = c["wins_passed"] / max(c["games_passed"], 1) if c["games_passed"] else float("nan")
            se = math.sqrt(a * (1 - a) / max(c["games_took"], 1) + (b * (1 - b) / c["games_passed"] if c["games_passed"] else 0))
            diff = f"{100 * (a - b):+.0f}（±{100 * se:.0f}）" if c["games_passed"] >= 10 else "样本少"
            passed = f"{100 * b:.0f}%（{c['games_passed']} 局）" if c["games_passed"] else "-"
            print(f"| {i} | {name(cid)} | {c['picks']} | {c['offers']} | {100 * c['picks'] / c['offers']:.0f}% | {100 * a:.0f}%（{c['games_took']} 局） | {passed} | {diff} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--show")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    if a.show:
        return show(a.show, a.top)
    merged = {}
    for r in (analyze(p) for p in a.traces):  # traces of one character on disjoint seeds add up
        m = merged.setdefault((r["character"], r["ascension"]), dict(r, trace=[], cards={}, games=0, wins=0, card_rewards=0, skips=0))
        m["trace"].append(r["trace"])
        for k in ("games", "wins", "card_rewards", "skips"):
            m[k] += r[k]
        for cid, c in r["cards"].items():
            t = m["cards"].setdefault(cid, dict.fromkeys(c, 0))
            for k, v in c.items():
                t[k] += v
    json.dump(list(merged.values()), open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
