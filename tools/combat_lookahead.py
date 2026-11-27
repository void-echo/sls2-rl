"""Evaluate Forgotten Ritual and one-step combat value lookahead on CPU."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from multiprocessing.util import Finalize
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.force import force_choice, force_matches
from tools.death_fork import Engine, Worker, act_batch, combat, dump_line, outcome, read_jsonl, to_torch

RITUAL = "CARD.FORGOTTEN_RITUAL"
WIN_VALUE = float("inf")
LOSS_VALUE = -WIN_VALUE
WORKER = None
RANDOM = re.compile(r"random|shuffl", re.I)
RANDOM_COMBAT_POWERS = {
    "POWER.DECREE_OF_UNMAKING_POWER", "POWER.HAUNT_POWER", "POWER.JUGGERNAUT_POWER",
    "POWER.SERPENT_FORM_POWER", "POWER.SOW_POWER", "POWER.TRASH_TO_TREASURE_POWER",
}


class ReplayMismatch(RuntimeError):
    pass


def observation(response):
    return response["decision"], response["obs"], response["cands"]


def exhausted(response):
    return (response["obs"].get("combat") or {}).get("turn", {}).get("exhausted", 0)


def card_plays(response):
    if combat(response):
        hand = response["obs"]["combat"]["hand"]
        for i, cand in enumerate(response["cands"]):
            if cand["k"] == "play":
                yield i, hand[cand["hand"]]


def descriptions(engine, name, category):
    path = Path(engine) / "localization_eng" / f"{name}.json"
    return {
        f"{category}.{key.removesuffix('.description')}": re.sub(r"\[[^\]]+\]", "", value)
        for key, value in json.loads(path.read_text()).items()
        if key.endswith(".description")
    }


def card_identity(card):
    return tuple(card.get(k) for k in ("id", "up", "ench", "ench_n", "aff", "aff_n"))


def intents(combat_obs):
    return [(e["id"], e.get("slot"), e.get("alive"), e.get("move"), e.get("then"), e["intents"])
            for e in combat_obs["enemies"]]


class PrefixEngine(Engine):
    def _reader(self):
        for line in self.p.stdout:
            if line.startswith("{"):
                self.q.put(line)
        self.q.put(None)

    def _read(self, timeout):
        return json.loads(super()._read(timeout))

    def replay_prefix(self, prefix):
        self.p.stdin.write("".join(f'{{"cmd":"rl_act","i":{i}}}\n' for i in prefix))
        self.p.stdin.flush()
        first_failure = None
        for _ in prefix:
            line = super()._read(self.timeout)
            if not line.startswith('{"type":"rl","done":false') and first_failure is None:
                first_failure = json.loads(line).get("message", "combat ended inside the prefix")
        if first_failure is not None:
            raise ReplayMismatch(first_failure)
        return json.loads(line)


class Room:
    def __init__(self, save):
        self.save = save
        self.prefix = []
        self.branch_id = None
        self.cache = {}
        self.observations = {}


class Diagnostic(Worker):
    def __init__(self, args):
        super().__init__(args)
        self.branch_engine = PrefixEngine(args.engine, timeout=120)
        Finalize(None, self.close, exitpriority=10)
        self.cards = descriptions(args.engine, "cards", "CARD")
        self.potions = descriptions(args.engine, "potions", "POTION")
        self.steps = None

    def close(self):
        self.branch_engine.close()
        self.eng.close()

    def send(self, cmd):
        if self.steps is not None:
            self.steps[f"main_{cmd['cmd']}"] += 1
        return super().send(cmd)

    def distribution(self, response):
        b = self.enc.encode(response, self.buffer)
        t = min(self.cfg.caps.max_tok, max(8, (int(b["n_tok"]) + 7) // 8 * 8))
        c = min(self.cfg.caps.max_cand, max(8, (int(b["n_cand"]) + 7) // 8 * 8))
        batch = {k: (a[:t] if k.startswith("tok_") else a[:c] if k.startswith("cand_") else a)[None]
                 for k, a in b.items()}
        a, _, v, logits = act_batch(self.model, to_torch(batch, self.args.device), self.amp,
                                    greedy=True, return_logits=True)
        return int(a.item()), float(self.vnorm.denormalize(v).item()), logits[0, :int(b["n_cand"])]

    def capture_room(self, save_id):
        self.steps["main_rl_save_get"] += 1
        saved = self.eng.send({"cmd": "rl_save_get", "id": save_id})
        if saved.get("kind") != "room":
            raise RuntimeError(f"room save {save_id} unavailable")
        return Room(saved)

    def branch_send(self, cmd):
        self.steps[f"branch_{cmd['cmd']}"] += 1
        response = self.branch_engine.send(cmd)
        if response.get("type") != "rl":
            raise RuntimeError(response.get("message", str(response)))
        return response

    def restore(self, room, expected):
        load = {"id": room.branch_id} if room.branch_id is not None else {"data": room.save["data"]}
        response = self.branch_send({"cmd": "rl_load", **load})
        room.branch_id = response["info"]["save"]["id"]
        if room.prefix:
            self.steps["branch_prefix_rl_act"] += len(room.prefix)
            response = self.branch_engine.replay_prefix(room.prefix)
        if observation(response) != observation(expected):
            raise ReplayMismatch("decision, observation or candidates diverged after prefix replay")

    def branch_act(self, i):
        return self.branch_send({"cmd": "rl_act", "i": i})

    def ritual_branch(self, i):
        response = self.branch_act(i)
        while combat(response) and not response["done"] and response["decision"] == "card_select":
            a, _ = self.policy(response)
            response = self.branch_act(a)
        return response

    def ritual_point(self, response, room, action, value, logits, stats):
        if response["decision"] != "combat_play" or exhausted(response):
            return None
        plays = list(card_plays(response))
        rituals = [i for i, card in plays if card["id"] == RITUAL]
        triggers = [
            i for i, card in plays if card["id"] != RITUAL
            and ("Exhaust" in card.get("kw", ()) or re.search(r"\bexhaust\b", self.cards.get(card["id"], ""), re.I)
                 or any(p["id"] == "POWER.CORRUPTION_POWER" for p in response["obs"]["combat"]["player"]["pw"])
                 and card["type"] == "Skill")
        ]
        if not rituals or not triggers:
            return None
        stats["screened"] += 1
        if room is None:
            stats["missing_room"] += 1
            return None
        probabilities = torch.softmax(logits, -1).tolist()
        ritual = max(rituals, key=probabilities.__getitem__)
        try:
            valid = []
            for i in sorted(triggers, key=probabilities.__getitem__, reverse=True):
                self.restore(room, response)
                after = self.ritual_branch(i)
                if exhausted(after) <= exhausted(response):
                    continue
                options = [j for j, card in card_plays(after) if card["id"] == RITUAL]
                valid.append((i, after, options))
            if not valid:
                return None
            stats["eligible"] += 1
            stats["greedy_exhaust_eligible"] += any(i == action for i, _, _ in valid)
            viable = next(((i, after, options) for i, after, options in valid if options), None)
            if viable is None:
                stats["ritual_unavailable_after_exhaust"] += 1
                return None
            i, after, options = viable
            _, after_value, after_logits = self.distribution(after)
            after_p = torch.softmax(after_logits, -1).tolist()
            j = max(options, key=after_p.__getitem__)
            self.restore(room, response)
            after_repeat = self.ritual_branch(i)
            if observation(after_repeat) != observation(after):
                raise ReplayMismatch("exhaust action diverged during replay")
            following = self.ritual_branch(j)
            self.restore(room, response)
            direct = self.ritual_branch(ritual)
            if following["done"] or direct["done"]:
                stats["terminal_value_unavailable"] += 1
                return None
            following_value = self.policy(following)[1]
            direct_value = self.policy(direct)[1]
        except ReplayMismatch:
            stats["discarded_replay_points"] += 1
            return None
        except RuntimeError as exc:
            stats[f"branch_error: {exc}"] += 1
            return None
        hand = response["obs"]["combat"]["hand"]
        trigger_ids = {r: hand[response["cands"][r]["hand"]]["id"] for r, _, _ in valid}
        ritual_p = sum(probabilities[r] for r in rituals)
        return {
            "exhaust_card": trigger_ids[i], "exhaust_index": i, "all_exhaust_actions": trigger_ids,
            "ritual_index": ritual, "ritual_index_after": j,
            "greedy_index": action, "greedy_exhaust": action in trigger_ids,
            "greedy_selected_exhaust": action == i,
            "p_ritual": ritual_p, "p_exhaust": probabilities[i],
            "p_any_exhaust": sum(probabilities[r] for r in trigger_ids),
            "p_other": sum(p for r, p in enumerate(probabilities) if r not in rituals and r != i),
            "p_ritual_after": sum(after_p[r] for r in options),
            "v": value, "v_after_exhaust": after_value, "v_after_ritual": following_value,
            "v_direct_ritual": direct_value,
            "exhausted_after": exhausted(after),
            "ritual_glow_after": after["obs"]["combat"]["hand"][after["cands"][j]["hand"]].get("glow"),
            "candidate_probabilities": probabilities, "candidates": response["cands"],
            "room": response["obs"]["g"]["room"], "encounter": response["obs"]["combat"]["enc"],
            "floor": response["obs"]["g"]["total_floor"], "turn": response["obs"]["combat"]["pturn"],
        }

    def full_run(self, index):
        seed = f"{self.args.prefix}_{index}"
        self.steps = Counter()
        stats, points, room, latest = Counter(), [], None, None
        started = time.perf_counter()
        row = {"seed": seed, "seed_index": index, "victory": None, "error": None}
        try:
            response = self.send({"cmd": "rl_start", "character": "Ironclad", "ascension": 10, "seed": seed})
            for step in range(self.args.max_steps):
                if response["done"]:
                    row.update(victory=response["victory"], ending=outcome(response))
                    break
                save = response["info"].get("save")
                if save and save["kind"] == "room" and save["id"] != latest:
                    latest = save["id"]
                    room = None
                    has_ritual = any(c["id"] == RITUAL for c in response["obs"]["deck"])
                    if self.args.mode == "c" or has_ritual:
                        room = self.capture_room(latest)
                a, v, logits = self.distribution(response)
                if self.args.mode == "a":
                    matches = force_matches(response, {RITUAL}, set())
                    if any(matches):
                        a = force_choice(matches, logits)
                    if combat(response):
                        point = self.ritual_point(response, room, a, v, logits, stats)
                        if point is not None:
                            points.append(dict(point, step=step, seed=seed))
                elif combat(response):
                    if room is None:
                        stats["missing_room"] += 1
                    else:
                        try:
                            a = self.lookahead(response, room, a, "safe", stats)
                        except ReplayMismatch:
                            stats["policy_fallback"] += 1
                        room.cache.clear()
                        room.observations.clear()
                if room is not None:
                    room.prefix.append(a)
                response = self.send({"cmd": "rl_act", "i": a})
            else:
                raise RuntimeError("run step limit")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        return dict(row, points=points, stats=dict(stats), engine_steps=dict(self.steps),
                    seconds=time.perf_counter() - started)


    def safe_precheck(self, response, i):
        a = response["cands"][i]
        kind = a["k"]
        before = response["obs"]
        cb = before["combat"]
        card = cb["hand"][a["hand"]] if kind == "play" else None
        if card:
            desc = self.cards.get(card["id"])
            if card.get("tgt") == "RandomEnemy" or desc is None or RANDOM.search(desc):
                return False
        elif kind == "potion":
            potion = next(p for p in before["potions"] if p["slot"] == a["slot"])
            desc = self.potions.get(potion["id"])
            if desc is None or RANDOM.search(desc):
                return False
        elif kind not in ("discard_potion", "pick", "confirm", "skip"):
            return False
        powers = {p["id"] for e in [cb["player"], *cb["allies"], *cb["enemies"]] for p in e["pw"]}
        if powers & RANDOM_COMBAT_POWERS:
            return False
        relics = {r["id"] for r in before["relics"]}
        if card and (card["type"] == "Power" and "RELIC.MUMMIFIED_HAND" in relics
                     or card["type"] == "Attack" and "RELIC.KUSARIGAMA" in relics):
            return False
        return True

    def safe_after(self, response, i, after):
        if after["done"] or not combat(after):
            return False
        a = response["cands"][i]
        before_obs, after_obs = response["obs"], after["obs"]
        before, new = before_obs["combat"], after_obs["combat"]
        if (before["pturn"] != new["pturn"] or before["round"] != new["round"]
                or before.get("draw_top", []) != new.get("draw_top", [])
                or before["turn"].get("drawn_extra") != new["turn"].get("drawn_extra")):
            return False
        if Counter(map(card_identity, before["draw"])) != Counter(map(card_identity, new["draw"])):
            return False
        hand = list(map(card_identity, before["hand"]))
        played = hand.pop(a["hand"]) if a["k"] == "play" else None
        if hand != list(map(card_identity, new["hand"])):
            return False
        destinations = 0
        for pile in ("discard", "exhaust"):
            old_pile = list(map(card_identity, before[pile]))
            new_pile = list(map(card_identity, new[pile]))
            if new_pile != old_pile:
                if played is None or new_pile != [*old_pile, played]:
                    return False
                destinations += 1
        if destinations > 1 or intents(before) != intents(new):
            return False
        if exhausted(response) != exhausted(after) and any(
                r["id"] == "RELIC.FORGOTTEN_SOUL" for r in before_obs["relics"]):
            return False
        slot = a.get("slot") if a["k"] in ("potion", "discard_potion") else None
        if [p for p in before_obs["potions"] if p["slot"] != slot] != after_obs["potions"]:
            return False
        if after["decision"] == "card_select":
            if after_obs["ctx"].get("sel") not in ("hand", "hand_discard", "hand_upgrade", "combat_pile"):
                return False
            public = set(hand)
            for pile in ("discard", "exhaust"):
                public.update(map(card_identity, new[pile]))
            if any(card_identity(cand["card"]) not in public for cand in after["cands"] if cand["k"] == "pick"):
                return False
            if after_obs["ctx"].get("pile") in ("Draw", "DrawPile"):
                return False
        return True

    def successor_value(self, response):
        if response.get("info", {}).get("abandoned"):
            raise RuntimeError("combat branch abandoned")
        if not response["done"] and combat(response):
            return self.policy(response)[1]
        return WIN_VALUE if not response["done"] or response["victory"] else LOSS_VALUE

    def lookahead(self, response, room, greedy, mode, stats):
        stats["decisions"] += 1
        prefix = tuple(room.prefix)
        expected = observation(response)
        if room.observations.setdefault(prefix, expected) != expected:
            stats["discarded_replay_points"] += 1
            raise ReplayMismatch("cached prefix observation diverged")
        trials = room.cache.setdefault(prefix, {})
        candidates = range(len(response["cands"]))
        short_circuit = self.args.mode == "c"
        if short_circuit:
            if len(candidates) == 1 or not self.safe_precheck(response, greedy):
                return greedy
            candidates = [greedy, *(i for i in candidates if i != greedy)]
        for i in candidates:
            is_possible = self.safe_precheck(response, i) if mode == "safe" else False
            if mode == "safe" and not is_possible:
                continue
            if i not in trials:
                try:
                    self.restore(room, response)
                except ReplayMismatch:
                    stats["discarded_replay_points"] += 1
                    raise
                after = self.branch_act(i)
                safe = is_possible and self.safe_after(response, i, after)
                trials[i] = [after, safe, None]
                if short_circuit and i == greedy and not safe:
                    return greedy
            else:
                stats["cached_successors"] += 1
        if mode == "safe":
            eligible = [i for i, (_, safe, _) in trials.items() if safe]
            if not short_circuit:
                stats["two_safe"] += len(eligible) >= 2
            if greedy not in eligible:
                return greedy
        else:
            eligible = list(trials)
        for i in eligible:
            trial = trials[i]
            if trial[2] is None:
                trial[2] = self.successor_value(trial[0])
        chosen = max(eligible, key=lambda i: (trials[i][2], i == greedy, -i))
        stats["changed"] += chosen != greedy
        return chosen

    def boss_mode(self, load, room, mode):
        self.steps = Counter()
        stats, actions = Counter(), []
        started = time.perf_counter()
        result = {"won": None, "hp": None, "error": None}
        room.prefix.clear()
        try:
            r = self.send({"cmd": "rl_load", **load})
            while not r["done"] and combat(r):
                a, _ = self.policy(r)
                i = self.lookahead(r, room, a, mode, stats)
                trades = room.cache[tuple(room.prefix)]
                predicted = trades[i][0] if i in trades else None
                actions.append({"decision": r["decision"], "index": i, "candidate": r["cands"][i]})
                if mode == "oracle":
                    room.cache.pop(tuple(room.prefix))
                    room.observations.pop(tuple(room.prefix))
                room.prefix.append(i)
                r = self.send({"cmd": "rl_act", "i": i})
                if predicted is not None and observation(r) != observation(predicted):
                    stats["discarded_replay_points"] += 1
                    raise ReplayMismatch("selected candidate diverged between branch and main engine")
                if len(room.prefix) >= self.args.max_steps:
                    raise RuntimeError("replay step limit")
            if r.get("info", {}).get("abandoned"):
                raise RuntimeError("replay abandoned")
            result.update(won=not r["done"] or r["victory"] is True,
                          hp=r["obs"]["g"]["hp"], ending=outcome(r))
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        return dict(result, actions=actions, stats=dict(stats), engine_steps=dict(self.steps),
                    seconds=time.perf_counter() - started)

    def boss(self, node):
        out = {k: node[k] for k in ("seed", "seed_index", "encounter", "total_floor", "won", "combat_end")}
        out["modes"] = {}
        self.steps = Counter()
        try:
            saved = json.loads((Path(self.args.source) / node["save_file"]).read_text())
            loaded = self.send({"cmd": "rl_load", "data": saved["data"]})
            load_id = loaded["info"]["save"]["id"]
            local = dict(node, load_id=load_id)
            baseline = self.replay(local, True, False, 0, capture=True)
            out["modes"]["policy"] = dict(baseline, error=baseline["aborted"], engine_steps=dict(self.steps))
            if not baseline["reproduced"]:
                out["error"] = baseline["aborted"] or "original greedy action trace or result diverged"
                return out
            room = Room(saved)
            for mode in ("safe", "oracle"):
                out["modes"][mode] = self.boss_mode({"id": load_id}, room, mode)
            out["error"] = next((m["error"] for m in out["modes"].values() if m["error"]), None)
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def boss_nodes(args):
    source = Path(args.source)
    rows = read_jsonl(source / "results.jsonl") + read_jsonl(source / "controls.jsonl")
    return [n for n in rows if n["act"] == 3 and n["room"] == "Boss"]


def run_boss(node):
    return WORKER.boss(node)

def init_worker(args):
    global WORKER
    WORKER = Diagnostic(args)


def run_seed(index):
    return WORKER.full_run(index)


def job_id(job):
    if isinstance(job, dict):
        return job["seed_index"], job.get("total_floor")
    return job, None


def execute(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (f"a_{Path(args.ckpt).stem}.jsonl" if args.mode == "a" else f"{args.mode}.jsonl")
    seen = read_jsonl(path)
    seen_ids = {job_id(r) for r in seen}
    points = sum(len(r.get("points", [])) for r in seen)
    if args.mode == "a" and points >= args.points:
        return
    with path.open("a") as f, ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                                                  initializer=init_worker, initargs=(args,)) as pool:
        done = len(seen)
        jobs = boss_nodes(args) if args.mode == "b" else range(args.seeds)
        job = run_boss if args.mode == "b" else run_seed
        pending = [j for j in jobs if job_id(j) not in seen_ids]
        wave_size = args.workers * 2 if args.mode == "a" else len(pending) or 1
        for offset in range(0, len(pending), wave_size):
            wave = pending[offset:offset + wave_size]
            for future in as_completed([pool.submit(job, j) for j in wave]):
                row = future.result()
                dump_line(f, row)
                points += len(row.get("points", []))
                done += 1
                if done % 10 == 0 or row.get("error"):
                    progress = f", points={points}" if args.mode == "a" else ""
                    print(f"{args.mode} {done}/{len(jobs)}{progress}, {row['seed']}: "
                          f"{row.get('error') or 'ok'}", flush=True)
            if args.mode == "a" and points >= args.points:
                break
    if args.mode == "a" and points < args.points:
        raise RuntimeError(f"only {points} complete points from {done} seeds")
    print(f"{args.mode} {done}/{len(jobs)}, points={points}, {path}", flush=True)


def table(lines, headers, rows):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    lines.append("")


def rate(count, total):
    return f"{count}/{total}（{count / total:.1%}）" if total else "0/0"


def mean(values, digits=2):
    values = list(values)
    return f"{np.mean(values):.{digits}f}" if values else "—"


def distribution(values):
    values = list(values)
    return tuple(f"{v:.4f}" for v in (np.mean(values), *np.quantile(values, [.25, .5, .75]))) if values else ("—",) * 4


def report_a(lines, datasets):
    lines += ["## A. Forgotten Ritual 知行差距", ""]
    metrics = (
        ("在 s 打 Ritual 的 P", lambda p: p["p_ritual"]),
        ("在 s 打所选消耗动作的 P", lambda p: p["p_exhaust"]),
        ("在 s 打任一已确认消耗动作的 P", lambda p: p["p_any_exhaust"]),
        ("在 s 打其他动作的 P", lambda p: p["p_other"]),
        ("在 s′ 打 Ritual 的 P", lambda p: p["p_ritual_after"]),
        ("V(s)", lambda p: p["v"]),
        ("V(s′)，先消耗", lambda p: p["v_after_exhaust"]),
        ("V(s″)，先消耗再打 Ritual", lambda p: p["v_after_ritual"]),
        ("V(s_direct)，直接打 Ritual", lambda p: p["v_direct_ritual"]),
        ("V(s″) − V(s_direct)", lambda p: p["v_after_ritual"] - p["v_direct_ritual"]),
    )
    summary, distributions, conclusions = [], [], []
    for name, rows in datasets.items():
        stats = Counter()
        for row in rows:
            stats.update(row["stats"])
        points = [p for row in rows for p in row["points"]]
        complete = len(points)
        summary.append((name, len(rows), sum(bool(r["error"]) for r in rows), stats["screened"],
                        stats["eligible"], complete, stats["discarded_replay_points"],
                        stats["ritual_unavailable_after_exhaust"], stats["terminal_value_unavailable"],
                        stats["missing_room"], sum(v for k, v in stats.items() if k.startswith("branch_error:")),
                        rate(stats["greedy_exhaust_eligible"], stats["eligible"]),
                        rate(sum(p["greedy_exhaust"] for p in points), complete),
                        rate(sum(p["greedy_selected_exhaust"] for p in points), complete)))
        for label, metric in metrics:
            distributions.append((name, label, *distribution(metric(p) for p in points)))
        if points:
            before = np.mean([p["p_ritual"] for p in points])
            after = np.mean([p["p_ritual_after"] for p in points])
            delta = np.mean([p["v_after_ritual"] - p["v_direct_ritual"] for p in points])
            conclusions.append(f"{name} 的 Ritual 概率 {before:.1%}→{after:.1%}，配对 V 差 {delta:+.4f}")
    table(lines, ("检查点", "开局数", "中止局", "粗筛点", "确认消耗点", "完整点", "重放丢弃点",
                  "消耗后 Ritual 不可打", "局终无 V", "缺存档", "其他分支错误",
                  "任一消耗动作贪心首选（确认点）", "任一消耗动作贪心首选（完整点）",
                  "所选消耗动作贪心首选（完整点）"), summary)
    table(lines, ("检查点", "量", "均值", "Q1", "中位数", "Q3"), distributions)
    lines += [
        "。".join(conclusions) + "。" if conclusions else "尚无完整节点。",
        "同一决策点有多个候选时，选策略概率最高且经实打确认消耗、仍能接 Ritual 的动作作两步比较。"
        "Ritual 概率合计同名候选，其他动作概率排除 Ritual 和所选消耗动作。"
        "实际首选比例也列出任一经验证的消耗动作。",
        "A10 主局按贪心走，奖励与商店强拿 Ritual，消耗选牌跟随贪心。V 按 death_fork 的口径经 ValueNorm 反归一化，以上分布按完整点等权。",
        "",
    ]


def action_steps(mode):
    return sum(value for key, value in mode["engine_steps"].items() if key.endswith("rl_act"))


def report_b(lines, rows):
    lines += ["## B. 第三幕 Boss 战的一步价值前瞻", ""]
    summary, missing = [], []
    for mode in ("policy", "safe", "oracle"):
        attempts = [r["modes"][mode] for r in rows if mode in r["modes"]]
        good = [(r, r["modes"][mode]) for r in rows if mode in r["modes"]
                and r["modes"]["policy"]["reproduced"] and r["modes"][mode]["error"] is None]
        losses = [m for r, m in good if not r["won"]]
        wins = [m for r, m in good if r["won"]]
        name = "oracle（信息泄漏上界）" if mode == "oracle" else mode
        summary.append((name, rate(sum(m["won"] for m in losses), len(losses)),
                        rate(sum(not m["won"] for m in wins), len(wins)),
                        mean(m["hp"] for m in losses + wins),
                        mean(m["hp"] for m in losses), mean(m["hp"] for m in wins),
                        mean(m["hp"] for m in losses + wins if m["won"]),
                        mean(action_steps(m) for m in losses + wins),
                        mean(sum(m["engine_steps"].values()) for m in losses + wins),
                        mean(m["seconds"] for m in losses + wins)))
        missing.append((name, len(attempts), len(good),
                        sum(m.get("stats", {}).get("discarded_replay_points", 0) for m in attempts),
                        sum(bool(m["error"]) for m in attempts)))
    table(lines, ("模式", "原 78 败中败转胜", "原 100 胜中胜转败", "平均余血（败为 0）",
                  "原败组平均余血", "原胜组平均余血", "胜场平均余血",
                  "每场 rl_act 均数", "每场引擎命令均数", "每场平均秒数"), summary)
    table(lines, ("模式", "尝试战斗", "可配对战斗", "分支复现失败决策点", "模式中止"), missing)
    safe_stats = Counter()
    for r in rows:
        m = r["modes"].get("safe")
        if m and m["error"] is None:
            safe_stats.update(m["stats"])
    table(lines, ("safe 决策点", "至少两个 safe 候选", "前瞻改变策略首选"),
          [(safe_stats["decisions"], rate(safe_stats["two_safe"], safe_stats["decisions"]),
            rate(safe_stats["changed"], safe_stats["decisions"]))])
    baseline_mismatches = sum(not r["modes"].get("policy", {}).get("reproduced") for r in rows)
    lines += [
        f"safe 把 {summary[1][1]} 的原败场打成胜场，{summary[1][2]} 的原胜场打成败场。"
        f"oracle 对应为 {summary[2][1]} 和 {summary[2][2]}，它能看到后继抽牌与掷随机数的结果，仅作信息泄漏上界。",
        f"基线有 {baseline_mismatches}/{len(rows)} 场未复现原贪心动作与结局，整场剔除。"
        "分支重放观测或合法候选不一致时，剔除该决策点并将其所在模式整场排除，表中分母为各自可配对场次。",
        "rl_act 计入前缀重放与候选探索。命令总数还计入读档，oracle 复用了同一观测和动作前缀上 safe 已执行的候选。",
        "",
        "**safe 判定。** z 的 engine_v3 在观测与 info 中没有 RNG 流计数，采用可见状态近似。"
        "结束回合、随机目标或带随机/洗牌描述的牌和药水、已知可能触发随机效果的遗物和能力，以及无法检查战斗牌堆的终战动作均排除。"
        "执行后要求回合、抽牌堆张数及组成、已知牌顶、额外抽牌计数与敌方意图一致，手牌只允许移除打出牌，"
        "弃牌堆和消耗堆至多接收这张牌，药水只允许移除指定槽位，并只接受由原有可见牌构成的选牌提示。"
        "原贪心首选通过筛选才比较 safe 后继 V，同分保留原首选。oracle 胜终态取 +∞，负终态取 −∞。",
        "",
    ]


def report_c(lines, rows, source):
    baseline = {r["seed_index"]: r for r in read_jsonl(source / "runs.jsonl")}
    for row in rows:
        original = baseline.get(row["seed_index"])
        row["baseline_victory"] = original["victory"] if original and not original["aborted"] else None
    paired = [r for r in rows if r["error"] is None and r["baseline_victory"] is not None]
    lines += ["## C. 整局 safe 前瞻", ""]
    table(lines, ("开局数", "可配对局", "贪心胜率", "safe 胜率", "败转胜", "胜转败", "分支重放丢弃点"),
          [(len(rows), len(paired), rate(sum(r["baseline_victory"] for r in paired), len(paired)),
            rate(sum(r["victory"] for r in paired), len(paired)),
            sum(not r["baseline_victory"] and r["victory"] for r in paired),
            sum(r["baseline_victory"] and not r["victory"] for r in paired),
            sum(r["stats"].get("discarded_replay_points", 0) for r in rows))])
    first, last = min(rows, key=job_id)["seed"], max(rows, key=job_id)["seed"]
    lines += [
        f"种子从 `{first}` 到 `{last}`，贪心基线取 death_fork 的同种子主局记录。"
        "无法复现前缀的决策点回退到贪心，整局配对统计两边都跑完的种子。",
        "",
    ]


def report(args):
    out = Path(args.out_dir)
    datasets = {}
    lines = ["# 战斗价值前瞻诊断", ""]
    for path in sorted(out.glob("a_*.jsonl")):
        datasets[path.stem[2:]] = read_jsonl(path)
    boss = read_jsonl(out / "b.jsonl")
    full = read_jsonl(out / "c.jsonl")
    report_a(lines, datasets)
    report_b(lines, boss)
    if full:
        report_c(lines, full, Path(args.source))
    for name, rows in [*((f"a_{name}", rows) for name, rows in datasets.items()), ("b", boss), ("c", full)]:
        if rows:
            ordered = sorted(rows, key=job_id)
            (out / f"{name}.json").write_text(json.dumps(ordered, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("a", "b", "c", "report"))
    ap.add_argument("--engine", default="engine_v3")
    ap.add_argument("--ckpt", default="runs/iron_avg39800.pt")
    ap.add_argument("--out-dir", dest="out_dir", required=True)
    ap.add_argument("--source", default="runs/diag/death_fork_iron39800")
    ap.add_argument("--prefix", default="a10fork2")
    ap.add_argument("--seeds", type=int)
    ap.add_argument("--points", type=int, default=300)
    ap.add_argument("--workers", type=int, choices=range(1, 20), default=12, help="at most 12 workers, each with two engine processes")
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=3000)
    ap.add_argument("--device", choices=("cpu",), default="cpu")
    args = ap.parse_args()
    if args.seeds is None:
        args.seeds = 300 if args.mode == "c" else 1000
    args.workers = min(args.workers, 12)
    args.threads = 1
    if args.mode == "report":
        report(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
