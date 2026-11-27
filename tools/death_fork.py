"""Diagnose combat losses by replaying room-entry saves with the trained policy.

    python tools/death_fork.py --seeds 40 --workers 4 --out_dir tmp/diag/death_fork2/local
    python tools/death_fork.py --summarize tmp/diag/death_fork2/local
"""
import argparse
import atexit
from collections import Counter
import concurrent.futures as cf
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.engine_runner import Engine  # noqa: E402
from rl.encode import Encoder  # noqa: E402
from rl.ppo import to_torch  # noqa: E402
from rl.train import act_batch, amp_context, load_policy  # noqa: E402

ROOMS = ("Monster", "Elite", "Boss")
LABELS = {"Monster": "普通战斗", "Elite": "精英", "Boss": "boss"}
CLASSES = ("战斗内不稳", "入场血量不够", "入场状态打不过")
CONTROL_SEED = 20261030
WORKER = None


def dump_line(f, row):
    f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    f.flush()


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def combat(r):
    return r["obs"].get("combat") is not None


def outcome(r):
    return {"done": r["done"], "victory": r.get("victory"), "decision": r["decision"],
            "hp": r["obs"]["g"]["hp"], "max_hp": r["obs"]["g"]["max_hp"],
            "potions": [p["id"] for p in r["obs"].get("potions", [])]}


def control_group(node):
    if node["room"] == "Elite":
        return "Elite"
    if node["room"] == "Boss" and node["act"] in (2, 3):
        return f"Boss{node['act']}"
    return None


class Worker:
    def __init__(self, args):
        self.args = args
        torch.set_num_threads(args.threads)
        self.model, self.vnorm, self.cfg, ck = load_policy(args.ckpt, args.device)
        self.model.eval()
        self.enc = Encoder(self.cfg.caps)
        self.buffer = self.enc.alloc()
        self.amp = amp_context(self.cfg, torch.device(args.device))
        self.meta = {"checkpoint": args.ckpt, "update": ck["update"], "env_steps": ck["env_steps"],
                     "caps": vars(self.cfg.caps)}
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

    def policy(self, r, greedy=True):
        b = self.enc.encode(r, self.buffer)
        T = min(self.cfg.caps.max_tok, max(8, (int(b["n_tok"]) + 7) // 8 * 8))
        C = min(self.cfg.caps.max_cand, max(8, (int(b["n_cand"]) + 7) // 8 * 8))
        batch = {k: (a[:T] if k.startswith("tok_") else a[:C] if k.startswith("cand_") else a)[None]
                 for k, a in b.items()}
        a, _, v = act_batch(self.model, to_torch(batch, self.args.device), self.amp, greedy=greedy)
        return int(a.item()), float(self.vnorm.denormalize(v).item())

    def state(self, r, value=None):
        if value is None and not r["done"]:  # a terminal state has no value estimate
            value = self.policy(r)[1]
        obs, g = r["obs"], r["obs"]["g"]
        return {"hp": g["hp"], "max_hp": g["max_hp"], "gold": g["gold"],
                "potions": [p["id"] for p in obs.get("potions", [])],
                "relics": [p["id"] for p in obs.get("relics", [])],
                "deck_size": len(obs.get("deck", [])), "value": value,
                "decision": r["decision"], "done": r["done"], "victory": r.get("victory")}

    def save(self, node):
        saved = self.eng.send({"cmd": "rl_save_get", "id": node["save_id"]})
        if saved.get("type") != "save" or saved["kind"] != "room":
            raise RuntimeError(f"room save unavailable: {saved}")
        path = Path("saves") / f"{node['seed_index']:06d}_{node['save_id']}.json"
        (Path(self.args.out_dir) / path).write_text(json.dumps(saved), encoding="utf-8")
        node["save_file"] = str(path)

    def main_run(self, index):
        seed = f"{self.args.prefix}_{index}"
        started = time.perf_counter()
        nodes, active, latest, steps = [], None, None, 0
        run = {"seed": seed, "seed_index": index, "victory": None, "aborted": None, **self.meta}
        try:
            r = self.send({"cmd": "rl_start", "character": "Ironclad", "ascension": self.args.ascension, "seed": seed})
            while True:
                save = r.get("info", {}).get("save")
                if save and save["kind"] == "room":
                    latest = save
                if active is not None and active["combat_end"] is None and (r["done"] or not combat(r)):
                    active["won"] = not r.get("info", {}).get("abandoned", False) and (
                        not r["done"] or r.get("victory") is True)
                    active["combat_end"] = outcome(r)
                if active is not None and (r["done"] or r["decision"] == "map_select"):
                    active["exit"] = self.state(r)
                    active = None
                if r["done"]:
                    run.update(victory=r["victory"], abandoned=r.get("info", {}).get("abandoned", False),
                               act=r["info"]["act"], floor=r["info"]["total_floor"])
                    break
                a, value = self.policy(r)
                g = r["obs"]["g"]
                if active is None and combat(r) and g["room"] in ROOMS:
                    if latest is None:
                        raise RuntimeError("combat without a room-entry save")
                    active = {"seed": seed, "seed_index": index, "act": g["act"], "floor": g["floor"],
                              "total_floor": g["total_floor"], "room": g["room"],
                              "encounter": r["obs"]["combat"]["enc"], "save_id": latest["id"],
                              "entry": self.state(r, value), "exit": None, "won": None,
                              "combat_steps": 0, "potions_used": 0, "actions": [], "combat_end": None,
                              "aborted": None}
                    nodes.append(active)
                    if control_group(active):
                        self.save(active)
                if active is not None and active["combat_end"] is None:
                    active["actions"].append({"decision": r["decision"], "index": a, "candidate": r["cands"][a]})
                    active["combat_steps"] += 1
                    active["potions_used"] += r["cands"][a]["k"] == "potion"
                r = self.send({"cmd": "rl_act", "i": a})
                steps += 1
                if steps >= self.args.max_steps:
                    raise RuntimeError("main run step limit")
        except Exception as ex:
            run["aborted"] = str(ex)
            if active is not None:
                active["aborted"] = str(ex)
        run.update(steps=steps, main_seconds=time.perf_counter() - started)
        failure = None
        if (not run["aborted"] and run["victory"] is False and not run.get("abandoned")
                and nodes and nodes[-1]["won"] is False and nodes[-1]["combat_end"] is not None
                and nodes[-1]["exit"]["hp"] == 0):
            node = nodes[-1]
            if "save_file" not in node:
                self.save(node)
            failure = self.diagnose(node)
        return nodes, failure, run

    def replay(self, node, greedy, heal, sample_seed, capture=False):
        started = time.perf_counter()
        result = {"won": None, "hp": None, "potions_used": 0, "combat_steps": 0,
                  "sample_seed": sample_seed, "aborted": None}
        actions = []
        torch.manual_seed(sample_seed)
        try:
            r = self.send({"cmd": "rl_load", "id": node.get("load_id", node["save_id"])})
            if not combat(r):
                raise RuntimeError("room load did not restore combat entry")
            if heal:
                target = r["obs"]["g"]["max_hp"]
                before = r["obs"]["combat"]["player"]["hp"]
                r = self.send({"cmd": "set_player", "hp": target})
                r = self.send({"cmd": "dbg_obs"})
                observed = r["obs"]["combat"]["player"]["hp"]
                result["heal_check"] = {"before": before, "target": target, "combat_hp": observed,
                                        "global_hp": r["obs"]["g"]["hp"],
                                        "ok": observed == target and r["obs"]["g"]["hp"] == target}
                if not result["heal_check"]["ok"]:
                    raise RuntimeError("set_player did not heal the combat player")
            while not r["done"] and combat(r):
                a, _ = self.policy(r, greedy)
                if capture:
                    actions.append({"decision": r["decision"], "index": a, "candidate": r["cands"][a]})
                result["potions_used"] += r["cands"][a]["k"] == "potion"
                result["combat_steps"] += 1
                r = self.send({"cmd": "rl_act", "i": a})
                if result["combat_steps"] >= self.args.max_steps:
                    raise RuntimeError("replay step limit")
            if r.get("info", {}).get("abandoned"):
                raise RuntimeError("replay abandoned")
            result.update(won=not r["done"] or r.get("victory") is True,
                          hp=r["obs"]["g"]["hp"], ending=outcome(r))
        except Exception as ex:
            result["aborted"] = str(ex)
        result["seconds"] = time.perf_counter() - started
        if capture:
            same_actions = actions == node["actions"]
            same_end = result.get("ending") == node["combat_end"]
            result["reproduced"] = (not result["aborted"] and same_actions and same_end
                                    and result["potions_used"] == node["potions_used"]
                                    and result["combat_steps"] == node["combat_steps"])
            if not result["reproduced"]:
                first = next((i for i, (a, b) in enumerate(zip(actions, node["actions"])) if a != b),
                             min(len(actions), len(node["actions"])))
                result["mismatch"] = {"first_action": first, "actual_actions": actions,
                                      "expected_actions": node["actions"], "expected_ending": node["combat_end"]}
        return result

    def sample(self, node, heal=False):
        base = 1000000 + node["seed_index"] * 1000000 + node["save_id"] * 1000
        return [self.replay(node, False, heal, base + i) for i in range(self.args.k)]

    def diagnose(self, node):
        result = dict(node)
        result["k"] = self.args.k
        result["a"] = [self.replay(node, True, False, 0, capture=True)]
        result["b"] = self.sample(node)
        first = self.replay(node, False, True,
                            1000000 + node["seed_index"] * 1000000 + node["save_id"] * 1000)
        if first.get("heal_check", {}).get("ok"):
            result["c"] = [first] + [self.replay(node, False, True, first["sample_seed"] + i)
                                     for i in range(1, self.args.k)]
            result["d"] = [self.replay(node, True, True, 0)]
        else:
            result.update(c=[], d=[], heal_unavailable=first)
        return result

    def control(self, node):
        saved = json.loads((Path(self.args.out_dir) / node["save_file"]).read_text())
        loaded = self.send({"cmd": "rl_load", "data": saved["data"]})
        local = dict(node, load_id=loaded["info"]["save"]["id"])
        return dict(node, k=self.args.k, b=self.sample(local))


def init_worker(args):
    global WORKER
    WORKER = Worker(args)


def run_seed(index):
    return WORKER.main_run(index)


def run_control(node):
    return WORKER.control(node)


def rate(row, arm):
    trials = row.get(arm, [])
    if len(trials) != (1 if arm in ("a", "d") else row["k"]) or any(t["aborted"] for t in trials):
        return None
    return sum(t["won"] for t in trials) / len(trials)


def classify(row):
    b, c = rate(row, "b"), rate(row, "c")
    if b is not None and b >= .5:
        return CLASSES[0]
    if b is None or c is None:
        return "未分类"
    return CLASSES[1] if c >= .5 else CLASSES[2]


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def number(v):
    return "—" if v is None or not np.isfinite(v) else f"{v:.4f}"


def percent(k, n):
    return f"{k}/{n} ({k / n:.1%})" if n else "0/0 (—)"


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        result[order[i:j]] = (i + j + 1) / 2
        i = j
    return result


def auc(rows):
    y = np.array([n["won"] for n in rows], dtype=bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if not pos or not neg:
        return None
    r = ranks([n["entry"]["value"] for n in rows])
    return float((r[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def spearman(rows):
    if len(rows) < 2:
        return None
    x = ranks([n["entry"]["value"] for n in rows])
    y = ranks([rate(n, "b") for n in rows])
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def groups(rows, encounters=False):
    out = [("全部", list(rows))]
    for act in (1, 2, 3):
        out.append((f"第{act}幕 boss", [n for n in rows if n["act"] == act and n["room"] == "Boss"]))
    for room in ("Elite", "Monster"):
        out.append((LABELS[room], [n for n in rows if n["room"] == room]))
    if encounters:
        for enc in sorted({n["encounter"] for n in rows if n["act"] == 3 and n["room"] == "Boss"}):
            out.append((f"第3幕 {enc}", [n for n in rows if n["act"] == 3 and n["encounter"] == enc]))
    return out


def summarize(out_dir):
    out = Path(out_dir)
    nodes = read_jsonl(out / "nodes.jsonl")
    failures = read_jsonl(out / "results.jsonl")
    controls = read_jsonl(out / "controls.jsonl")
    runs = read_jsonl(out / "runs.jsonl")
    completed = [n for n in nodes if n["won"] is not None and not n["aborted"] and n["exit"] is not None]
    lines = []

    def table(headers, rows):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
        lines.append("")

    replay_trials = [t for n in failures for arm in ("a", "b", "c", "d") for t in n[arm]]
    replay_trials += [t for n in controls for t in n["b"]]
    checks = [n["a"][0] for n in failures]
    mismatches = [n for n in failures if not n["a"][0]["reproduced"]]
    unavailable = [n for n in failures if n.get("heal_unavailable")]
    aborts = [t for t in replay_trials if t["aborted"]]
    lines += ["# 战斗失败诊断", "", f"主局 {len(runs)}；通关 {sum(r['victory'] is True for r in runs)}；"
              f"中止 {sum(bool(r['aborted']) for r in runs)}；放弃 {sum(bool(r.get('abandoned')) for r in runs)}；"
              f"战斗节点 {len(nodes)}；战斗失败 {len(failures)}；对照节点 {len(controls)}。", "",
              f"(a) 动作与结局完全复现：{percent(sum(t['reproduced'] for t in checks), len(checks))}。", ""]
    if mismatches:
        lines += [f"**存档复现异常：{len(mismatches)} 场。**", ""]
        table(["种子", "幕", "楼层", "遭遇", "首个不同动作"],
              [(n["seed"], n["act"], n["floor"], n["encounter"], n["a"][0]["mismatch"]["first_action"])
               for n in mismatches])
    heal_checks = [t["heal_check"] for t in replay_trials if "heal_check" in t]
    lines += [f"补血观测校验：{percent(sum(c['ok'] for c in heal_checks), len(heal_checks))}；"
              f"其中入场未满血 {sum(c['before'] < c['target'] for c in heal_checks)} 次；"
              f"重打中止 {len(aborts)} 次；补血不可用 {len(unavailable)} 场。", ""]
    if unavailable:
        lines += ["缺少能在首个战斗决策前修改玩家生物 HP 的可用命令，相关 (c)(d) 已跳过。", ""]
    if aborts or any(r["aborted"] for r in runs):
        table(["异常", "次数"], sorted(Counter([t["aborted"] for t in aborts] +
              [r["aborted"] for r in runs if r["aborted"]]).items()))
    lines += ["## 失败分布与归类", ""]
    table(["分组", "失败占比", *CLASSES, "未分类", "(a) 复现", "(d) 胜率"],
          [(label, percent(len(rows), len(failures)),
            *(percent(sum(classify(n) == cls for n in rows), len(rows)) for cls in CLASSES),
            sum(classify(n) == "未分类" for n in rows),
            percent(sum(n["a"][0]["reproduced"] for n in rows), len(rows)),
            number(mean(rate(n, "d") for n in rows))) for label, rows in groups(failures, True)])
    lines += ["## 原血量采样胜率分布", "", "每场等权；混合结果概率为 1 − p⁴ − (1−p)⁴。", ""]
    table(["样本", "分组", "完整节点数", "平均胜率", "K=4 混合结果概率"],
          [(name, label, len(ps), number(mean(ps)), number(mean(1 - p ** 4 - (1 - p) ** 4 for p in ps)))
           for name, data in (("失败组", failures), ("对照组", controls))
           for label, rows in groups(data)
           for ps in [[rate(n, "b") for n in rows if rate(n, "b") is not None]]])
    lines += ["## 进入时价值估计的区分度", ""]
    boss_rows, corr_rows = [], []
    for act in (2, 3):
        ns = [n for n in completed if n["act"] == act and n["room"] == "Boss"]
        rs = [n for n in failures + controls if n["act"] == act and n["room"] == "Boss" and rate(n, "b") is not None]
        boss_rows.append((f"第{act}幕 boss", len(ns), sum(n["won"] for n in ns), number(auc(ns))))
        corr_rows.append((f"第{act}幕 boss", len(rs), number(spearman(rs))))
    table(["分组", "节点数", "获胜数", "AUC"], boss_rows)
    table(["分组", "重打节点数", "价值与 (b) 胜率 Spearman"], corr_rows)
    table(["分组", "重打节点数", "价值与 (b) 胜率 Spearman"],
          [(label, len(rs), number(spearman(rs))) for label, rows in groups(failures + controls)
           for rs in [[n for n in rows if rate(n, "b") is not None]]])
    bin_rows = []
    for act in (2, 3):
        ns = [n for n in completed if n["act"] == act and n["room"] == "Boss"]
        if not ns:
            continue
        cuts = np.quantile([n["entry"]["value"] for n in ns], np.arange(1, 10) / 10)
        buckets = [[] for _ in range(10)]
        for n in ns:
            buckets[int(np.searchsorted(cuts, n["entry"]["value"], side="right"))].append(n)
        for i, bucket in enumerate(buckets):
            values = [n["entry"]["value"] for n in bucket]
            bin_rows.append((f"第{act}幕 boss", i + 1, number(min(values) if values else None),
                             number(max(values) if values else None), len(bucket),
                             number(mean(n["won"] for n in bucket))))
    table(["分组", "价值十分位", "最小值", "最大值", "节点数", "实际胜率"], bin_rows)
    lines += ["## 节点前后变化", "", "离开值取奖励处理后的首个地图决策或局终；平均掉血为进入 HP − 离开 HP。", ""]
    delta_rows = []
    for act in (1, 2, 3):
        for room in ROOMS:
            ns = [n for n in completed if n["act"] == act and n["room"] == room]
            delta_rows.append((act, LABELS[room], len(ns),
                               number(mean(n["entry"]["hp"] - n["exit"]["hp"] for n in ns)),
                               number(mean(n["potions_used"] for n in ns)),
                               number(mean(n["exit"]["gold"] - n["entry"]["gold"] for n in ns)),
                               number(mean(bool(Counter(n["exit"]["relics"]) - Counter(n["entry"]["relics"])) for n in ns)),
                               number(mean(n["exit"]["value"] - n["entry"]["value"] for n in ns
                                           if n["exit"]["value"] is not None))))
    table(["幕", "房型", "节点数", "平均掉血", "平均用药", "平均金币变化", "获得遗物比例", "平均价值变化（不含局终）"],
          delta_rows)
    lines += ["回归仅用获胜的普通战斗和精英节点（boss 节点跨幕、带 boss 奖励，局终没有价值估计），包含截距；"
              "HP、药水数及其余自变量均为离开值 − 进入值。", ""]
    regression = []
    for act in (1, 2, 3):
        ns = [n for n in completed if n["act"] == act and n["won"] and n["room"] != "Boss"
              and n["exit"]["value"] is not None]
        if not ns:
            regression.append((act, 0, 0, *("—" for _ in range(7))))
            continue
        X, y = [], []
        for n in ns:
            a, b = n["entry"], n["exit"]
            X.append([1, b["hp"] - a["hp"], len(b["potions"]) - len(a["potions"]),
                      len(b["relics"]) - len(a["relics"]), b["gold"] - a["gold"], b["deck_size"] - a["deck_size"]])
            y.append(b["value"] - a["value"])
        X, y = np.array(X, dtype=float), np.array(y)
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        total = ((y - y.mean()) ** 2).sum()
        r2 = float(1 - ((y - X @ beta) ** 2).sum() / total) if total > 0 else None
        regression.append((act, len(ns), rank, *(number(v) for v in beta), number(r2)))
    table(["幕", "节点数", "矩阵秩/6", "截距", "ΔHP", "Δ药水数", "Δ遗物数", "Δ金币", "Δ牌组大小", "R²"], regression)
    lines += ["## 墙钟时间", ""]
    table(["阶段", "次数", "平均秒数"],
          [("主局（含存档导出）", len(runs), number(mean(r["main_seconds"] for r in runs))),
           ("单遍重打（含读档及补血校验）", len(replay_trials), number(mean(t["seconds"] for t in replay_trials)))])
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", "--engine_dir", default=str(REPO / "tmp/server_setup/engine_v2s_64fa3a6"))
    ap.add_argument("--ckpt", default=str(REPO / "tmp/ckpts/v2r_boss_s1_u7800.pt"))
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--prefix", default="deathfork")
    ap.add_argument("--k", "--K", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, choices=(1, 2), default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", "--out-dir", default=str(REPO / "tmp/diag/death_fork2/local"))
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--summarize", metavar="OUT_DIR")
    args = ap.parse_args(argv)
    if args.summarize:
        summarize(args.summarize)
        return
    if min(args.seeds, args.workers, args.k, args.max_steps) < 1:
        ap.error("seeds, workers, k and max_steps must be positive")
    args.out_dir = str(Path(args.out_dir).expanduser().resolve())
    args.engine = str(Path(args.engine).expanduser().resolve())
    args.ckpt = str(Path(args.ckpt).expanduser().resolve())
    out = Path(args.out_dir)
    (out / "saves").mkdir(parents=True, exist_ok=True)
    candidates = []
    started = time.perf_counter()
    with (out / "nodes.jsonl").open("w") as nf, (out / "results.jsonl").open("w") as rf, \
            (out / "controls.jsonl").open("w") as cf_out, (out / "runs.jsonl").open("w") as ef, \
            cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                                   initializer=init_worker, initargs=(args,)) as pool:
        futures = [pool.submit(run_seed, i) for i in range(args.seeds)]
        for done, future in enumerate(cf.as_completed(futures), 1):
            nodes, failure, run = future.result()
            for node in nodes:
                dump_line(nf, node)
                if node["won"] is True and not node["aborted"] and control_group(node):
                    candidates.append(node)
            dump_line(ef, run)
            if failure:
                dump_line(rf, failure)
            print(f"main {done}/{args.seeds}: {run['seed']} victory={run['victory']} "
                  f"failure={failure is not None} aborted={run['aborted']} ({time.perf_counter() - started:.1f}s)", flush=True)
        rng = random.Random(CONTROL_SEED)
        selected = []
        for group, cap in (("Boss3", 100), ("Boss2", 60), ("Elite", 60)):
            rows = sorted((n for n in candidates if control_group(n) == group),
                          key=lambda n: (n["seed_index"], n["save_id"]))
            selected.extend(rng.sample(rows, min(cap, len(rows))))
        futures = [pool.submit(run_control, node) for node in selected]
        for done, future in enumerate(cf.as_completed(futures), 1):
            row = future.result()
            dump_line(cf_out, row)
            if done % 10 == 0 or done == len(selected):
                print(f"controls {done}/{len(selected)} ({time.perf_counter() - started:.1f}s)", flush=True)
    summarize(out)
    print(f"summary: {out / 'summary.md'} ({time.perf_counter() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
