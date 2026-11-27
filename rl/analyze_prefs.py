"""Summarize a policy's play preferences from rl/trace.py output into a Chinese Markdown report.

    python -m rl.analyze_prefs traces.jsonl.gz [more.jsonl.gz ...] --out docs/results/<name>/preferences.md

Everything here is descriptive statistics of what the policy chose when it had a choice; nothing feeds back
into training. Card and event names come from the game's own localization tables (English + Chinese).
"""
import argparse
import collections
import gzip
import json
import math
import os
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOC_DIRS = {"en": [os.path.join(REPO, "engine", "localization_eng"),
                   os.path.join(REPO, "_external", "wuhao21_sts2-cli", "localization_eng")],
            "zh": [os.path.join(REPO, "engine", "localization_zhs"),
                   os.path.join(REPO, "_external", "wuhao21_sts2-cli", "localization_zhs")]}


def load_loc():
    loc = {"en": {}, "zh": {}}
    for lang, dirs in LOC_DIRS.items():
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".json"):
                    try:
                        loc[lang].setdefault(f[:-5], {}).update(json.load(open(os.path.join(d, f))))
                    except (OSError, ValueError):
                        pass
            break
    return loc


LOC = load_loc()
TABLE = {"CARD": "cards", "RELIC": "relics", "POTION": "potions", "EVENT": "events", "MONSTER": "monsters",
         "ENCOUNTER": "encounters", "POWER": "powers"}


def name(mid):
    """'CARD.BASH' -> '重击 Bash' using the game's localization; falls back to the raw entry."""
    if not mid or "." not in str(mid):
        return str(mid)
    kind, entry = str(mid).split(".", 1)
    t = TABLE.get(kind)

    def look(lang):
        # Ancient events (Neow, Orobas, ...) live in their own table.
        for tn in ([t, "ancients"] if kind == "EVENT" else [t]):
            tab = LOC[lang].get(tn, {}) if tn else {}
            v = tab.get(f"{entry}.title") or tab.get(f"{entry}.name")
            if v:
                return v
        return None

    zh, en = look("zh"), look("en")
    if zh and en:
        return f"{zh}（{en}）"
    return zh or en or entry


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def pct(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load(paths):
    meta, steps, eps = [], collections.defaultdict(list), []
    for p in paths:
        for line in gzip.open(p, "rt"):
            r = json.loads(line)
            if r["type"] == "meta":
                meta.append(r)
            elif r["type"] == "step":
                steps[r["seed"]].append(r)
            else:
                eps.append(r)
    for s in steps.values():
        s.sort(key=lambda r: r["step"])
    return meta, steps, eps


def hp_frac(info):
    return (info or {}).get("hp", 0) / max(1, (info or {}).get("max_hp", 1))


def analyze(meta, steps, eps, min_offers=5):
    L = []
    done = [e for e in eps if not e.get("aborted")]
    wins = [e for e in done if e.get("victory")]
    p, lo, hi = wilson(len(wins), len(done))
    m = meta[0] if meta else {}
    floors = sorted((e.get("info") or {}).get("total_floor", 0) for e in done)
    L.append("## 概况\n")
    L.append(f"检查点 `{m.get('ckpt')}`（第 {m.get('update')} 次更新），角色 {m.get('character')}，"
             f"难度 A{m.get('ascension')}，{'按策略概率采样' if m.get('sample') else '每步取概率最高的动作'}。"
             f"共 {len(eps)} 个种子，其中 {len(eps) - len(done)} 局因引擎错误中断、不计入统计。"
             f"完成的 {len(done)} 局里赢了 {len(wins)} 局，胜率 {pct(p)}（95% 置信区间 {pct(lo)} 到 {pct(hi)}）。")
    if floors:
        q = statistics.quantiles(floors, n=4) if len(floors) >= 4 else [floors[0], statistics.median(floors), floors[-1]]
        L.append(f"结束时的楼层中位数是 {statistics.median(floors):g}，四分位数 {q[0]:g} 到 {q[-1]:g}，最高 {floors[-1]}。\n")

    # Deaths: enemies present at the last combat step of each lost run.
    deaths = collections.Counter()
    for e in done:
        if e.get("victory"):
            continue
        last = next((r for r in reversed(steps.get(e["seed"], [])) if (r["state"] or {}).get("combat")), None)
        if last:
            ids = sorted({x["id"] for x in last["state"]["combat"]["enemies"]})
            deaths[" + ".join(name(i) for i in ids) or "?"] += 1
    if deaths:
        L.append("### 死在哪里\n")
        L.append(table([(k, v, pct(v / max(1, sum(deaths.values())))) for k, v in deaths.most_common(12)],
                       ["最后一场战斗的敌人", "局数", "占失败局"]) + "\n")

    all_steps = [r for s in steps.values() for r in s]

    # Card rewards.
    offered, picked, prob = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    n_reward, n_skip = 0, 0
    for r in all_steps:
        if r["decision"] != "card_reward":
            continue
        n_reward += 1
        ch = r["cands"][r["choice"]]
        if ch.get("k") in ("skip",):
            n_skip += 1
        probs = r["probs"] if isinstance(r["probs"], list) else None
        for i, c in enumerate(r["cands"]):
            card = c.get("card")
            if not card:
                continue
            offered[card["id"]] += 1
            if i == r["choice"]:
                picked[card["id"]] += 1
            if probs:
                prob[card["id"]].append(probs[i])
    if n_reward:
        L.append("## 选卡\n")
        L.append(f"一共 {n_reward} 次卡牌奖励，跳过了 {n_skip} 次（{pct(n_skip / n_reward)}）。下表只列被提供过至少 "
                 f"{min_offers} 次的卡，按选取率排序；“平均概率”是策略给这张卡的平均概率，比选取率更平滑。\n")
        rows = []
        for cid, n in offered.items():
            if n >= min_offers:
                rows.append((name(cid), n, picked[cid], picked[cid] / n, statistics.mean(prob[cid]) if prob[cid] else float("nan")))
        rows.sort(key=lambda x: -x[3])
        fmt = lambda rr: [(a, b, c, pct(d), pct(e)) for a, b, c, d, e in rr]
        hdr = ["卡牌", "被提供", "被选", "选取率", "平均概率"]
        L.append("最常拿的卡：\n\n" + table(fmt(rows[:20]), hdr) + "\n")
        if len(rows) > 20:
            L.append("最不愿意拿的卡：\n\n" + table(fmt(rows[-15:][::-1]), hdr) + "\n")

    # Rest sites.
    rest = [r for r in all_steps if r["decision"] == "rest_site"]
    if rest:
        L.append("## 篝火\n")
        kinds = collections.Counter(r["cands"][r["choice"]].get("key") for r in rest)
        L.append("选择分布：" + "，".join(f"{k} {v} 次（{pct(v / len(rest))}）" for k, v in kinds.most_common()) + "。"
                 "作为参照，SpireBench 统计的顶级主播在 StS1 A20 的休息率约 9%，前沿 LLM 是 57% 到 75%。\n")
        bins = collections.defaultdict(lambda: [0, 0])
        for r in rest:
            b = min(4, int(hp_frac(r["info"]) * 5))
            bins[b][0] += r["cands"][r["choice"]].get("key") == "HEAL"
            bins[b][1] += 1
        L.append(table([(f"{20 * b}%–{20 * b + 20}%", n, pct(h / n)) for b, (h, n) in sorted(bins.items())],
                       ["进篝火时的血量比例", "次数", "选择休息的比例"]) + "\n")

    # Map choices: preference for room types relative to availability.
    avail, chosen = collections.Counter(), collections.Counter()
    n_map_choices = 0
    for r in all_steps:
        if r["decision"] != "map_select" or len(r["cands"]) < 2:
            continue
        nodes = (r["state"] or {}).get("map_nodes") or []
        types = [nodes[c["node"]]["t"] if isinstance(c.get("node"), int) and 0 <= c["node"] < len(nodes) else "?"
                 for c in r["cands"]]
        if len(set(types)) < 2:
            continue
        n_map_choices += 1
        for t in set(types):
            avail[t] += 1
        chosen[types[r["choice"]]] += 1
    if n_map_choices:
        L.append("## 路线\n")
        L.append(f"只统计下一步有不同房间类型可选的 {n_map_choices} 个岔路口。“可选时被选中的比例”是：这类房间出现在选项里时，"
                 "策略选它的频率。\n")
        L.append(table([(t, avail[t], chosen[t], pct(chosen[t] / avail[t])) for t in sorted(avail, key=lambda t: -chosen[t] / avail[t])],
                       ["房间类型", "出现在选项里", "被选中", "可选时被选中的比例"]) + "\n")

    # HP entering boss fights.
    boss_hp = collections.defaultdict(list)
    for s in steps.values():
        seen = set()
        for r in s:
            g = (r["state"] or {}).get("g") or {}
            if g.get("room") == "Boss" and (r["state"] or {}).get("combat") and g.get("act") not in seen:
                seen.add(g.get("act"))
                boss_hp[g.get("act")].append(hp_frac(r["info"]))
    if boss_hp:
        L.append("## 打 boss 时的血量\n")
        L.append(table([(f"第 {a} 幕", len(v), pct(statistics.median(v))) for a, v in sorted(boss_hp.items())],
                       ["幕", "进入 boss 战的局数", "进场血量比例中位数"]) +
                 "\n\n参照：SpireBench 里 StS1 A20 顶级主播打第一幕 boss 时血量中位数约 53%，前沿 LLM 约 90%。\n")

    # Events.
    ev = collections.defaultdict(collections.Counter)
    for r in all_steps:
        if r["decision"] == "event_choice":
            ctx = (r["state"] or {}).get("ctx") or {}
            ev[ctx.get("event") or "?"][r["cands"][r["choice"]].get("key") or r["cands"][r["choice"]].get("k")] += 1
    if ev:
        L.append("## 事件\n")
        rows = []
        for e, cnt in sorted(ev.items(), key=lambda x: -sum(x[1].values())):
            tot = sum(cnt.values())
            rows.append((name(e), tot, "；".join(f"{str(k).split('.options.')[-1]} {pct(v / tot)}" for k, v in cnt.most_common(4))))
        L.append(table(rows[:25], ["事件", "决策次数", "选项分布（前四）"]) + "\n")

    # Shops.
    shop = [r for r in all_steps if r["decision"] == "shop"]
    if shop:
        kinds = collections.Counter(r["cands"][r["choice"]].get("k") for r in shop)
        L.append("## 商店\n")
        L.append("商店里每一步的选择：" + "，".join(f"{k} {v} 次" for k, v in kinds.most_common()) + "。\n")
        bought = collections.Counter()
        for r in shop:
            c = r["cands"][r["choice"]]
            if c.get("k") == "buy_card":
                bought[c["card"]["id"]] += 1
            elif c.get("k") in ("buy_relic", "buy_potion"):
                bought[c.get("id")] += 1
        if bought:
            L.append(table([(name(k), v) for k, v in bought.most_common(15)], ["买过的东西", "次数"]) + "\n")

    # Potions and combat style.
    pot_use, pot_disc = collections.Counter(), collections.Counter()
    turns, hp_lost, waste, n_end = [], [], 0, 0
    for s in steps.values():
        cur, start_hp, max_round = None, None, 0
        for r in s:
            cb = (r["state"] or {}).get("combat")
            ch = r["cands"][r["choice"]]
            if ch.get("k") == "potion":
                pot_use[(r["state"] or {}).get("g", {}).get("room")] += 1
            if ch.get("k") == "discard_potion":
                pot_disc[(r["state"] or {}).get("g", {}).get("room")] += 1
            if cb:
                if cur is None:
                    cur, start_hp, max_round = True, r["info"].get("hp", 0), 0
                max_round = max(max_round, cb.get("round") or 0)
                if ch.get("k") == "end_turn":
                    n_end += 1
                    if (cb.get("energy") or 0) > 0 and any(c.get("playable") for c in cb.get("hand") or []):
                        waste += 1
            elif cur is not None:
                turns.append(max_round)
                hp_lost.append(start_hp - r["info"].get("hp", 0))
                cur = None
    L.append("## 战斗风格与药水\n")
    if turns:
        L.append(f"完整打完的战斗共 {len(turns)} 场，平均 {statistics.mean(turns):.1f} 回合，每场平均掉血 "
                 f"{statistics.mean(hp_lost):.1f}（中位数 {statistics.median(hp_lost):g}）。")
    if n_end:
        L.append(f"结束回合时还有能量且手里有可打的牌的比例是 {pct(waste / n_end)}。")
    L.append(f"药水使用 {sum(pot_use.values())} 次（按房间类型：{dict(pot_use)}），丢弃 {sum(pot_disc.values())} 次。\n")

    # Final decks.
    decks = [e.get("final", {}).get("deck") or [] for e in done if e.get("final")]
    if not any(decks):
        decks = []
        for s in steps.values():
            last = next((r for r in reversed(s) if (r["state"] or {}).get("deck")), None)
            if last:
                decks.append(last["state"]["deck"])
    if decks:
        L.append("## 最终牌组\n")
        sizes = [len(d) for d in decks]
        types = collections.Counter(c.get("type") for d in decks for c in d)
        ups = sum(1 for d in decks for c in d if c.get("up"))
        L.append(f"牌组张数中位数 {statistics.median(sizes):g}（{min(sizes)} 到 {max(sizes)}）；平均每局升级过的牌 "
                 f"{ups / len(decks):.1f} 张；类型构成：" + "，".join(f"{k} {v / len(decks):.1f} 张" for k, v in types.most_common()) + "。\n")
        cc = collections.Counter(c["id"] for d in decks for c in d)
        L.append(table([(name(k), v, f"{v / len(decks):.2f}") for k, v in cc.most_common(20)], ["卡牌", "总张数", "平均每局"]) + "\n")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="策略的游戏偏好")
    ap.add_argument("--min_offers", type=int, default=5)
    a = ap.parse_args(argv)
    meta, steps, eps = load(a.traces)
    body = analyze(meta, steps, eps, a.min_offers)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(f"# {a.title}\n\n本报告由 `rl/analyze_prefs.py` 从 `rl/trace.py` 的轨迹自动生成，只描述策略在有选择时选了什么。\n\n")
        f.write(body + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
