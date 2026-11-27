"""Compare policy choices with engine-executed counterfactual critic values.

    python tools/critic_cf.py --seeds 6 --workers 2 --out_dir tmp/diag/critic_cf/local
    python tools/critic_cf.py --summarize tmp/diag/critic_cf/local
"""
import argparse
import atexit
from collections import Counter, defaultdict
import concurrent.futures as cf
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.death_fork import dump_line, read_jsonl  # noqa: E402
from tools.engine_runner import Engine  # noqa: E402
from rl.encode import Encoder  # noqa: E402
from rl.ppo import to_torch  # noqa: E402
from rl.train import act_batch, amp_context, load_policy  # noqa: E402

KINDS = {"card_reward": {"reward_card", "skip"},
         "shop": {"buy_card", "buy_relic", "buy_potion", "remove", "leave"},
         "rest_site": {"rest"}}
LABELS = {"card_reward": "拿牌", "shop": "商店", "rest_site": "篝火"}
SHOP = {"buy_card": "买牌", "buy_relic": "买遗物", "buy_potion": "买药", "remove": "删牌", "leave": "离开"}
WORKER = None


def comparable(r):
    obs = dict(r["obs"])
    obs["relics"] = [{k: v for k, v in relic.items() if k != "status"} for relic in obs.get("relics", [])]
    return {"decision": r["decision"], "done": r["done"], "victory": r.get("victory"),
            "obs": obs, "cands": r["cands"]}


def differences(expected, actual, path=""):
    if type(expected) is not type(actual):
        return [{"path": path, "expected": expected, "actual": actual}]
    if isinstance(expected, dict):
        out = []
        for k in sorted(expected.keys() | actual.keys()):
            p = f"{path}.{k}" if path else k
            if k not in expected or k not in actual:
                out.append({"path": p, "expected": expected.get(k), "actual": actual.get(k),
                            "missing": "expected" if k not in expected else "actual"})
            else:
                out.extend(differences(expected[k], actual[k], p))
        return out
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [{"path": path, "expected": expected, "actual": actual}]
        return [d for i, (a, b) in enumerate(zip(expected, actual))
                for d in differences(a, b, f"{path}[{i}]")]
    return [] if expected == actual else [{"path": path, "expected": expected, "actual": actual}]


def card_key(card):
    return card["id"], card.get("up", 0)


class Worker:
    def __init__(self, args):
        self.args = args
        torch.set_num_threads(args.threads)
        self.model, self.vnorm, self.cfg, ck = load_policy(args.ckpt, args.device)
        self.model.eval()
        self.enc = Encoder(self.cfg.caps)
        self.buffer = self.enc.alloc()
        self.amp = amp_context(self.cfg, torch.device(args.device))
        self.meta = {"checkpoint": args.ckpt, "engine": args.engine, "device": args.device,
                     "update": ck["update"], "env_steps": ck["env_steps"], "caps": vars(self.cfg.caps)}
        self.eng = Engine(args.engine, timeout=120,
                          stderr_path=str(Path(args.out_dir) / f"engine_{mp.current_process().name}.stderr"))
        atexit.register(self.eng.close)

    def send(self, cmd):
        r = self.eng.send(cmd)
        if r.get("type") != "rl":
            raise RuntimeError(r.get("message", str(r)))
        if not r["done"] and not r["cands"]:
            raise RuntimeError("nonterminal response without candidates")
        return r

    def act(self, index):
        return self.send({"cmd": "rl_act", "i": index})

    def policy(self, r):
        b = self.enc.encode(r, self.buffer)
        T = min(self.cfg.caps.max_tok, max(8, (int(b["n_tok"]) + 7) // 8 * 8))
        C = min(self.cfg.caps.max_cand, max(8, (int(b["n_cand"]) + 7) // 8 * 8))
        batch = {k: (a[:T] if k.startswith("tok_") else a[:C] if k.startswith("cand_") else a)[None]
                 for k, a in b.items()}
        a, _, v, logits = act_batch(self.model, to_torch(batch, self.args.device), self.amp,
                                    greedy=True, return_logits=True)
        p = torch.softmax(logits[0], -1).cpu().tolist()[:len(r["cands"])]
        p += [0.0] * (len(r["cands"]) - len(p))
        return int(a.item()), float(self.vnorm.denormalize(v).item()), p

    def value(self, r):
        return self.policy(r)[1]

    def check(self, row, expected, actual, phase, index=None):
        diff = differences(comparable(expected), comparable(actual))
        check = {"phase": phase, "index": index, "ok": not diff}
        if diff:
            check["differences"] = diff
        row["replay_checks"].append(check)

    def replay(self, row, expected, phase, suffix=()):
        r = self.send({"cmd": "rl_load", "id": row["save_id"]})
        for i in row["room_actions"] + list(suffix):
            r = self.act(i)
        self.check(row, expected, r, phase, list(suffix))
        return r

    def complete(self, r, return_shop=False):
        actions = []
        while not r["done"] and (r["decision"] == "card_select" or
                                  (return_shop and r["decision"] != "shop")):
            a, _, p = self.policy(r)
            confirms = [i for i, c in enumerate(r["cands"]) if c["k"] == "confirm"]
            picks = [i for i, c in enumerate(r["cands"]) if c["k"] == "pick"]
            if r["decision"] == "card_select":
                a = confirms[0] if confirms else max(picks, key=lambda i: p[i])
            actions.append({"index": a, "candidate": r["cands"][a]})
            r = self.act(a)
            if len(actions) >= self.args.max_steps:
                raise RuntimeError("selection completion step limit")
        return r, actions

    def candidate(self, row, original, item):
        i, cand = item["index"], item["candidate"]
        self.replay(row, original, "candidate")
        r = self.act(i)
        if cand["k"] == "remove" and (r["decision"] != "card_select" or
                                       r["obs"]["ctx"].get("sel") != "deck_remove"):
            raise RuntimeError("shop removal did not enter deck_remove")
        if cand["k"] not in ("remove", "rest") or r["decision"] != "card_select":
            item.update(q=self.value(r), next_decision=r["decision"])
            return
        selection = r
        sel = r["obs"]["ctx"].get("sel")
        a, value, p = self.policy(r)
        item.update(selection=sel, selection_value=value, selection_policy_index=a, children=[])
        seen = set()
        for j, pick in enumerate(selection["cands"]):
            if pick["k"] != "pick" or card_key(pick["card"]) in seen:
                continue
            seen.add(card_key(pick["card"]))
            if item["children"]:
                self.replay(row, selection, "subselection", (i,))
            r = self.act(j)
            r, completion = self.complete(r, return_shop=cand["k"] == "remove")
            item["children"].append({"index": j, "candidate": pick, "probability": p[j],
                                     "completion": completion, "q": self.value(r),
                                     "next_decision": r["decision"]})
        if not item["children"]:
            raise RuntimeError("selection without pick candidates")
        best = max(item["children"], key=lambda c: c["q"])
        item.update(q=best["q"], best_child_index=best["index"])

    def evaluate(self, row, r):
        started = time.perf_counter()
        expected_next = self.act(row["policy_index"])
        row["first_next"] = comparable(expected_next)
        for item in row["candidates"]:
            if item["evaluated"]:
                self.candidate(row, r, item)
        self.replay(row, r, "continue_current")
        resumed = self.act(row["policy_index"])
        self.check(row, expected_next, resumed, "continue_next")
        row["replay_ok"] = all(c["ok"] for c in row["replay_checks"])
        evaluated = [c for c in row["candidates"] if c["q"] is not None]
        row["evaluated_probability"] = sum(c["probability"] for c in evaluated)
        row["policy_weighted_q"] = sum(c["probability"] * c["q"] for c in evaluated)
        row["critic_index"] = max(evaluated, key=lambda c: c["q"])["index"]
        row["policy_q"] = row["candidates"][row["policy_index"]]["q"]
        row["seconds"] = time.perf_counter() - started
        return resumed

    def main_run(self, index):
        seed = f"{self.args.prefix}_{index}"
        started = time.perf_counter()
        rows, actions, shops, pending = [], [], set(), None
        save_id, entry_hp, steps, r = None, None, 0, None
        run = {"seed": seed, "seed_index": index, "victory": None, "aborted": None, **self.meta}
        try:
            r = self.send({"cmd": "rl_start", "character": "Ironclad", "ascension": 0, "seed": seed})
            while not r["done"]:
                saved = r.get("info", {}).get("save")
                if saved and saved["kind"] == "room" and saved["id"] != save_id:
                    save_id, actions = saved["id"], []
                    entry_hp = {k: r["obs"]["g"][k] for k in ("hp", "max_hp")}
                a, value, p = self.policy(r)
                if pending is not None:
                    if r["decision"] == "card_select":
                        pending["policy_subchoices"].append({"index": a, "candidate": r["cands"][a],
                                                             "selection": r["obs"]["ctx"].get("sel")})
                    else:
                        pending = None
                allowed = KINDS.get(r["decision"], set())
                if any(c["k"] in allowed for c in r["cands"]):
                    if save_id is None:
                        raise RuntimeError("decision without a room-entry save")
                    g = r["obs"]["g"]
                    room = (g["act"], g["total_floor"])
                    row = {"seed": seed, "seed_index": index, "step": steps, "act": g["act"],
                           "floor": g["floor"], "total_floor": g["total_floor"], "decision": r["decision"],
                           "hp": g["hp"], "max_hp": g["max_hp"], "gold": g["gold"], "entry_hp": entry_hp,
                           "value": value, "policy_index": a, "policy_subchoices": [],
                           "save_id": save_id, "room_actions": list(actions), "obs": r["obs"],
                           "first_shop": r["decision"] == "shop" and room not in shops,
                           "candidates": [{"index": i, "candidate": c, "probability": p[i],
                                           "evaluated": c["k"] in allowed, "q": None}
                                          for i, c in enumerate(r["cands"])],
                           "replay_checks": [], "replay_ok": None, "aborted": None}
                    rows.append(row)
                    if r["decision"] == "shop":
                        shops.add(room)
                    evaluating = time.perf_counter()
                    try:
                        next_r = self.evaluate(row, r)
                    except Exception as ex:
                        row.update(aborted=str(ex), seconds=time.perf_counter() - evaluating)
                        row["replay_ok"] = False
                        raise
                    if r["cands"][a]["k"] in ("rest", "remove") and next_r["decision"] == "card_select":
                        pending = row
                else:
                    next_r = self.act(a)
                actions.append(a)
                r = next_r
                steps += 1
                if steps >= self.args.max_steps:
                    raise RuntimeError("main run step limit")
            run.update(victory=r["victory"], abandoned=r.get("info", {}).get("abandoned", False))
        except Exception as ex:
            run["aborted"] = str(ex)
        run.update(steps=steps, seconds=time.perf_counter() - started, decisions=len(rows),
                   replay_anomalies=sum(row["replay_ok"] is False for row in rows),
                   replay_checks=sum(len(row["replay_checks"]) for row in rows),
                   replay_mismatches=sum(not c["ok"] for row in rows for c in row["replay_checks"]),
                   act=r["obs"]["g"]["act"] if r else None,
                   floor=r["obs"]["g"]["total_floor"] if r else None)
        return rows, run


def init_worker(args):
    global WORKER
    WORKER = Worker(args)


def run_seed(index):
    return WORKER.main_run(index)


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def number(value):
    return "—" if value is None or not np.isfinite(value) else f"{value:.4f}"


def percent(k, n):
    return f"{k}/{n} ({k / n:.1%})" if n else "0/0 (—)"


def distribution(values, thresholds=(.01,)):
    values = list(values)
    return [len(values), number(mean(values)), number(float(np.median(values)) if values else None),
            *(percent(sum(v > t for v in values), len(values)) for t in thresholds)]


def item(row, which):
    return row["candidates"][row[f"{which}_index"]]


def best_child(candidate):
    return max(candidate.get("children", []), key=lambda c: c["q"], default=None)


def advantages(rows, kind):
    groups = defaultdict(list)
    for row in rows:
        evaluated = [c for c in row["candidates"] if c["q"] is not None]
        for c in evaluated:
            if c["candidate"]["k"] != kind:
                continue
            others = [x["q"] for x in evaluated if x["index"] != c["index"]]
            if others:
                ident = c["candidate"].get("card", {}).get("id") or c["candidate"].get("id")
                groups[ident].append(c["q"] - mean(others))
    return sorted(((key, len(vals), mean(vals)) for key, vals in groups.items()),
                  key=lambda t: (-t[2], t[0]))


def summarize(out_dir):
    out = Path(out_dir)
    decisions = sorted(read_jsonl(out / "decisions.jsonl"), key=lambda r: (r["seed_index"], r["step"]))
    runs = sorted(read_jsonl(out / "runs.jsonl"), key=lambda r: r["seed_index"])
    rows = [r for r in decisions if not r["aborted"]]
    checks = [c for r in decisions for c in r["replay_checks"]]
    lines = ["# 构筑决策价值诊断", "",
             f"对局 {len(runs)}；通关 {sum(r['victory'] is True for r in runs)}；"
             f"中止 {sum(bool(r['aborted']) for r in runs)}；评估决策 {len(decisions)}；完整评估 {len(rows)}。", "",
             "价值差使用原始价值单位，0.01 约为 1 个百分点；一致性使用已评估候选的原始策略概率（不重新归一化），删牌和篝火选牌取子选择最大 Q。",
             "重放比较 decision、done、victory、obs、cands，仅排除 obs.relics[].status；需选多张牌时枚举首张，其余按策略贪心选牌并在可确认时提交。", ""]

    def table(headers, data):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in data)
        lines.append("")

    lines += ["## 一致性与重放", "",
              f"决策重放异常率：{percent(sum(r['replay_ok'] is False for r in decisions), len(decisions))}；"
              f"逐次校验异常率：{percent(sum(not c['ok'] for c in checks), len(checks))}。", ""]
    consistency = []
    for kind, label in LABELS.items():
        rs = [r for r in rows if r["decision"] == kind]
        x, y = [r["policy_weighted_q"] for r in rs], [r["value"] for r in rs]
        corr = float(np.corrcoef(x, y)[0, 1]) if len(rs) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0 else None
        consistency.append((label, len(rs), number(mean(a - b for a, b in zip(x, y))),
                            number(mean(abs(a - b) for a, b in zip(x, y))), number(corr),
                            number(mean(r["evaluated_probability"] for r in rs)),
                            sum(r["policy_q"] is None for r in rs),
                            percent(sum(r["replay_ok"] is False for r in rs), len(rs))))
    table(["类型", "决策数", "ΣπQ − V 均值", "平均绝对差", "Pearson 相关", "平均概率覆盖", "策略选择未评估", "重放异常率"], consistency)
    paths = Counter()
    for check in checks:
        for diff in check.get("differences", []):
            path = diff["path"]
            if path.startswith("obs.relics["):
                path = "obs.relics[]" + path.split("]", 1)[1]
            paths[path] += 1
    if paths:
        table(["重放差异字段", "校验差异次数"], sorted(paths.items()))
    errors = Counter(r["aborted"] for r in runs if r["aborted"])
    if errors:
        table(["异常", "次数"], sorted(errors.items()))

    cards = [r for r in rows if r["decision"] == "card_reward"]
    comparable_cards = [r for r in cards if r["policy_q"] is not None]
    gaps = [item(r, "critic")["q"] - r["policy_q"] for r in comparable_cards]
    lines += ["## 拿牌", "",
              f"首选一致：{percent(sum(r['critic_index'] == r['policy_index'] for r in comparable_cards), len(comparable_cards))}；"
              f"策略跳过：{percent(sum(item(r, 'policy')['candidate']['k'] == 'skip' for r in cards), len(cards))}；"
              f"价值首选跳过：{percent(sum(item(r, 'critic')['candidate']['k'] == 'skip' for r in cards), len(cards))}。", ""]
    table(["样本数", "Q最大 − Q策略 均值", "中位数", ">0.005", ">0.01", ">0.02"],
          [distribution(gaps, (.005, .01, .02))])
    stats = defaultdict(lambda: {"n": 0, "policy": 0, "critic": 0, "adv": []})
    for r in cards:
        by_id = defaultdict(list)
        evaluated = [c for c in r["candidates"] if c["q"] is not None]
        for c in evaluated:
            if c["candidate"]["k"] == "reward_card":
                by_id[c["candidate"]["card"]["id"]].append(c)
        for ident, cs in by_id.items():
            s = stats[ident]
            s["n"] += 1
            s["policy"] += any(c["index"] == r["policy_index"] for c in cs)
            s["critic"] += any(c["index"] == r["critic_index"] for c in cs)
            for c in cs:
                other = [x["q"] for x in evaluated if x["index"] != c["index"]]
                if other:
                    s["adv"].append(c["q"] - mean(other))
    eligible = [(ident, s) for ident, s in stats.items() if s["n"] >= 10]
    headers = ["卡牌 id", "出现决策数", "策略选取率", "价值首选率", "首选率差（价值−策略）", "平均 Q − 其余均值"]
    for title, sign in (("价值函数比策略更偏爱（前 15）", -1), ("策略比价值函数更偏爱（前 15）", 1)):
        lines += [f"### {title}", ""]
        ranked = sorted(eligible, key=lambda t: (sign * (t[1]["critic"] - t[1]["policy"]) / t[1]["n"], t[0]))[:15]
        table(headers, [(ident, s["n"], percent(s["policy"], s["n"]), percent(s["critic"], s["n"]),
                         number((s["critic"] - s["policy"]) / s["n"]), number(mean(s["adv"]))) for ident, s in ranked])

    shops = [r for r in rows if r["decision"] == "shop"]
    first = [r for r in shops if r["first_shop"]]
    lines += ["## 商店", "", f"进店首个决策 {len(first)}；全部商店决策 {len(shops)}。", ""]
    table(["首选类型", "策略", "价值函数"],
          [(label, percent(sum(item(r, "policy")["candidate"]["k"] == k for r in first), len(first)),
            percent(sum(item(r, "critic")["candidate"]["k"] == k for r in first), len(first))) for k, label in SHOP.items()] +
          [("其他", percent(sum(item(r, "policy")["candidate"]["k"] not in SHOP for r in first), len(first)), "0")])
    removals, remove_policy, remove_leave = Counter(), [], []
    for r in shops:
        remove = next((c for c in r["candidates"] if c["candidate"]["k"] == "remove" and c["q"] is not None), None)
        leave = next((c for c in r["candidates"] if c["candidate"]["k"] == "leave"), None)
        if remove is not None:
            child = best_child(remove)
            if child:
                removals[child["candidate"]["card"]["id"]] += 1
            if r["policy_q"] is not None:
                remove_policy.append(remove["q"] - r["policy_q"])
            if leave is not None and leave["q"] is not None:
                remove_leave.append(remove["q"] - leave["q"])
    table(["全部商店决策的价值差", "样本数", "均值", "中位数", ">0.01"],
          [("删牌最佳 − 策略所选", *distribution(remove_policy)),
           ("删牌最佳 − 离开", *distribution(remove_leave))])
    lines += ["### 最想删的牌（全部可删牌决策，前 10）", ""]
    table(["卡牌 id", "次数"], sorted(removals.items(), key=lambda t: (-t[1], t[0]))[:10])
    for k in ("buy_relic", "buy_card", "buy_potion"):
        lines += [f"### {SHOP[k]}相对价值（全部商店决策，前 10）", ""]
        table(["id", "候选次数", "平均 Q − 其余均值"],
              [(ident, n, number(v)) for ident, n, v in advantages(shops, k)[:10]])

    rests = [r for r in rows if r["decision"] == "rest_site"]
    lines += ["## 篝火", ""]
    options = sorted({c["candidate"].get("key", c["candidate"]["k"]) for r in rests for c in r["candidates"]
                      if c["candidate"]["k"] == "rest"} |
                     {item(r, "policy")["candidate"].get("key", item(r, "policy")["candidate"]["k"]) for r in rests})
    data = []
    for label, lo, hi in (("<40%", 0, .4), ("40–70%", .4, .7), ("70–99%", .7, 1), ("满血", 1, float("inf"))):
        rs = [r for r in rests if lo <= r["entry_hp"]["hp"] / r["entry_hp"]["max_hp"] < hi]
        for option in options:
            counts = [sum(item(r, who)["candidate"].get("key", item(r, who)["candidate"]["k"]) == option for r in rs)
                      for who in ("policy", "critic")]
            data.append((label, len(rs), option, percent(counts[0], len(rs)), percent(counts[1], len(rs))))
    table(["进入时血量", "决策数", "选项", "策略", "价值函数"], data)
    upgrades, actual_upgrades = Counter(), Counter()
    for r in rests:
        for c in r["candidates"]:
            if c.get("selection") == "deck_upgrade" and best_child(c):
                upgrades[best_child(c)["candidate"]["card"]["id"]] += 1
        for c in r["policy_subchoices"]:
            if c["selection"] == "deck_upgrade" and c["candidate"]["k"] == "pick":
                actual_upgrades[c["candidate"]["card"]["id"]] += 1
    for title, counts in (("价值函数最想升级（每次可升级决策，前 10）", upgrades), ("策略实际升级（前 10）", actual_upgrades)):
        lines += [f"### {title}", ""]
        table(["卡牌 id", "次数"], sorted(counts.items(), key=lambda t: (-t[1], t[0]))[:10])
    lines += ["## 耗时", ""]
    table(["项目", "次数", "平均秒数"], [("完整对局（含评估）", len(runs), number(mean(r["seconds"] for r in runs)))] +
          [(f"{label}评估", len(rs), number(mean(r["seconds"] for r in rs))) for kind, label in LABELS.items()
           for rs in [[r for r in rows if r["decision"] == kind]]])
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", "--engine_dir", default=str(REPO / "tmp/server_setup/engine_v2s_64fa3a6"))
    ap.add_argument("--ckpt", default=str(REPO / "tmp/ckpts/v2r_s1_bigb_u8100.pt"))
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--prefix", default="critcf")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, choices=(1, 2), default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", "--out-dir", default=str(REPO / "tmp/diag/critic_cf/local"))
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--summarize", metavar="OUT_DIR")
    args = ap.parse_args(argv)
    if args.summarize:
        summarize(args.summarize)
        return
    if min(args.seeds, args.workers, args.max_steps) < 1:
        ap.error("seeds, workers and max_steps must be positive")
    args.out_dir = str(Path(args.out_dir).expanduser().resolve())
    args.engine = str(Path(args.engine).expanduser().resolve())
    args.ckpt = str(Path(args.ckpt).expanduser().resolve())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (out / "decisions.jsonl").open("w") as df, (out / "runs.jsonl").open("w") as rf, \
            cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                                   initializer=init_worker, initargs=(args,)) as pool:
        futures = [pool.submit(run_seed, i) for i in range(args.seeds)]
        for done, future in enumerate(cf.as_completed(futures), 1):
            rows, run = future.result()
            for row in rows:
                dump_line(df, row)
            dump_line(rf, run)
            print(f"main {done}/{args.seeds}: {run['seed']} victory={run['victory']} decisions={len(rows)} "
                  f"replay_anomalies={run['replay_anomalies']} aborted={run['aborted']} "
                  f"({run['seconds']:.1f}s; total {time.perf_counter() - started:.1f}s)", flush=True)
    summarize(out)


if __name__ == "__main__":
    main()
