"""Machine-readable behavior statistics from rl.trace, with optional checkpoint sensitivity probes.

    python -m rl.behavior_metrics traces.jsonl.gz [more.jsonl.gz ...] --out metrics.json [--ckpt policy.pt]

Rates have k, n, rate and a 95% Wilson ci95; means have n, total and mean. Empty samples are null.
Decision statistics use all recorded steps; per-run statistics exclude aborted/incomplete episodes.
Use trace --behavior_obs for exact terminal decks, potion IDs and checkpoint probes.
"""
import argparse
import collections
import copy
import gzip
import json
import os

from .analyze_prefs import wilson

HEALING_POTIONS = {"BLOOD_POTION", "REGEN_POTION", "FRUIT_JUICE"}
PROBE_DECISIONS = ("card_reward", "rest_site", "map_select", "shop")
HP_BINS = ("<40%", "40-70%", "70-99%", "full")


def rate(k, n):
    if not n:
        return {"k": k, "n": n, "rate": None, "ci95": [None, None]}
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "ci95": [0. if k == 0 else lo, 1. if k == n else hi]}


def mean(values):
    return {"n": len(values), "total": sum(values), "mean": sum(values) / len(values) if values else None}


def load_traces(paths):
    meta, runs = [], []
    for path in paths:
        by_seed = {}
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r["type"] == "meta":
                    meta.append(r)
                    continue
                run = by_seed.setdefault(r["seed"], {"seed": r["seed"], "steps": [], "episode": None})
                if r["type"] == "step":
                    run["steps"].append(r)
                elif r["type"] == "episode":
                    run["episode"] = r
        for run in by_seed.values():
            run["steps"].sort(key=lambda r: r["step"])
        runs.extend(by_seed.values())
    return meta, runs


def state(r):
    return r.get("obs") or r.get("state") or r.get("final") or {}


def hp_fraction(r):
    g = state(r).get("g") or {}
    info = r.get("info") or {}
    hp, max_hp = info.get("hp", g.get("hp")), info.get("max_hp", g.get("max_hp"))
    return hp / max_hp if hp is not None and max_hp and max_hp > 0 else None


def hp_bin(hp):
    return HP_BINS[0 if hp < .4 else 1 if hp < .7 else 2 if hp < 1 else 3]


def room_key(r):
    g, info = state(r).get("g") or {}, r.get("info") or {}
    return (g.get("act", info.get("act")), info.get("total_floor", g.get("total_floor", g.get("floor"))))


def potion_id(r, chosen):
    if r.get("potion"):
        return r["potion"].get("id")
    if chosen.get("id"):
        return chosen["id"]
    return next((p.get("id") for p in state(r).get("potions") or [] if isinstance(p, dict)
                 and p.get("slot") == chosen.get("slot")), None)


def removal_count(rows):
    n = 0
    for r in rows:
        ctx = state(r).get("ctx") or {}
        if r["decision"] != "card_select" or ctx.get("sel") != "deck_remove":
            continue
        c = r["cands"][r["choice"]]
        picked = len(ctx.get("picked") or [])
        if c.get("k") == "confirm":
            n += picked
        elif c.get("k") == "pick" and picked + 1 >= ctx.get("max", 1):
            n += picked + 1
    return n


def final_deck(run):
    ep = run["episode"]
    for s, source in ((ep.get("obs") or {}, "terminal"), (ep.get("final") or {}, "terminal")):
        if "deck" in s:
            return s["deck"], source
    for r in reversed(run["steps"]):
        s = state(r)
        if "deck" in s:
            return s["deck"], "last_observed"
    return None, "missing"


def compute_metrics(runs, meta=()):
    rows = [r for run in runs for r in run["steps"]]
    done = [run for run in runs if run["episode"] and not run["episode"].get("aborted")]
    rest, bins = collections.Counter(), {b: collections.Counter() for b in HP_BINS}
    elite_chosen = elite_offered = sphere_big = sphere_n = 0
    potion_n = potion_known = healing_n = healing_full = outside = 0
    reward_n = reward_skip = 0
    picked_types = collections.Counter()
    for run in runs:
        entry_hp = {}
        for r in run["steps"]:
            c = r["cands"][r["choice"]]
            d, s = r["decision"], state(r)
            if d == "rest_site":
                key = room_key(r)
                entry_hp.setdefault(key, hp_fraction(r))
                if c.get("k") == "rest":
                    action = c.get("key", "unknown")
                    rest[action] += 1
                    hp = entry_hp[key]
                    if hp is not None:
                        bins[hp_bin(hp)][action] += 1
            if d == "map_select" and c.get("k") == "map":
                nodes = (s.get("map") or {}).get("nodes") or s.get("map_nodes") or []

                def is_elite(cd):
                    idx = cd.get("node")
                    return cd.get("k") == "map" and isinstance(idx, int) and 0 <= idx < len(nodes) and nodes[idx].get("t") == "Elite"

                if any(is_elite(cd) for cd in r["cands"]):
                    elite_offered += 1
                    elite_chosen += is_elite(c)
            if d == "crystal_sphere" and c.get("k") == "sphere":
                sphere_n += 1
                sphere_big += c.get("tool") == "big"
            if c.get("k") == "potion":
                potion_n += 1
                outside += not bool(s.get("combat"))
                pid = potion_id(r, c)
                potion_known += pid is not None
                hp = hp_fraction(r)
                if pid and pid.split(".")[-1] in HEALING_POTIONS and hp is not None:
                    healing_n += 1
                    healing_full += hp >= 1
            if d == "card_reward" and c.get("k") in ("reward_card", "skip"):
                reward_n += 1
                reward_skip += c.get("k") == "skip"
                if c.get("k") == "reward_card":
                    picked_types[(c.get("card") or {}).get("type", "unknown")] += 1

    elites, floors, acts, sizes, defense, removals = [], [], [], [], [], []
    deck_sources = collections.Counter()
    for run in done:
        ep = run["episode"]
        visited = {room_key(r) for r in run["steps"] + [ep] if (state(r).get("g") or {}).get("room") == "Elite"}
        elites.append(len(visited))
        floor = (ep.get("info") or {}).get("total_floor", (state(ep).get("g") or {}).get("total_floor"))
        if floor is not None:
            floors.append(floor)
        acts.append(max(((state(r).get("g") or {}).get("act", (r.get("info") or {}).get("act", 0))
                         for r in run["steps"] + [ep]), default=0))
        deck, source = final_deck(run)
        deck_sources[source] += 1
        if deck is not None:
            sizes.append(len(deck))
            if all("v" in c for c in deck if c.get("type") == "Skill"):
                defense.append(sum(c.get("type") == "Skill" and "Block" in (c.get("v") or {}) for c in deck))
        removals.append(removal_count(run["steps"]))

    nr, npicked = sum(rest.values()), sum(picked_types.values())
    wins = sum(bool(run["episode"].get("victory")) for run in done)
    return {
        "schema_version": 1,
        "meta": list(meta),
        "samples": {"runs": len(runs), "completed": len(done), "aborted": sum(bool(run["episode"] and run["episode"].get("aborted")) for run in runs),
                    "incomplete": sum(run["episode"] is None for run in runs), "decisions": len(rows)},
        "rest_site": {
            "choices": {k: rate(rest[k], nr) for k in sorted(set(rest) | {"SMITH", "HEAL"})},
            "hp_bins": {b: {k.lower(): rate(cnt[k], sum(cnt.values())) for k in ("SMITH", "HEAL")}
                        for b, cnt in bins.items()},
            "full_hp_heal": rate(bins["full"]["HEAL"], sum(bins["full"].values())),
        },
        "elite": {"entries_per_run": mean(elites), "choice_when_available": rate(elite_chosen, elite_offered)},
        "crystal_sphere": {"big_tool": rate(sphere_big, sphere_n)},
        "potions": {"healing_at_full_hp": rate(healing_full, healing_n), "outside_combat": rate(outside, potion_n),
                    "identified": rate(potion_known, potion_n)},
        "card_reward": {"skip": rate(reward_skip, reward_n),
                        "picked_types": {k: rate(picked_types[k], npicked) for k in sorted(set(picked_types) | {"Attack", "Skill", "Power"})}},
        "deck": {"size": mean(sizes), "defensive_skills": mean(defense), "sources": dict(deck_sources),
                 "removals_per_run": mean(removals)},
        "overall": {"win": rate(wins, len(done)), "floor": mean(floors),
                    "passed_act1_boss": rate(sum(a > 1 or bool(run["episode"].get("victory")) for a, run in zip(acts, done)), len(done)),
                    "passed_act2_boss": rate(sum(a > 2 or bool(run["episode"].get("victory")) for a, run in zip(acts, done)), len(done))},
    }


def checkpoint_policy(ckpt, device, batch_size):
    import torch

    from .encode import Encoder
    from .ppo import to_torch
    from .train import amp_context, load_policy

    device = torch.device(device)
    model, _, cfg, _ = load_policy(os.path.expanduser(ckpt), device)
    model.eval()
    enc, amp_ctx = Encoder(cfg.caps), amp_context(cfg, device)

    def predict(records):
        out = []
        for start in range(0, len(records), batch_size):
            part = records[start:start + batch_size]
            batch = enc.alloc(len(part))
            for j, r in enumerate(part):
                for k, v in enc.encode(r).items():
                    batch[k][j] = v
            with torch.no_grad(), amp_ctx():
                logits, _ = model(to_torch(batch, device))
            for j in range(len(part)):
                n = int(batch["n_cand"][j])
                out.append(torch.softmax(logits[j, :n].float(), -1).cpu().numpy())
        return out

    return predict


def sensitivity(runs, predict, batch_size=16):
    import numpy as np

    bosses = collections.defaultdict(set)
    rows = [r for run in runs for r in run["steps"]]
    for r in rows + [run["episode"] for run in runs if run["episode"]]:
        g = state(r).get("g") or {}
        if g.get("boss"):
            bosses[(g.get("act_id"), g.get("act"))].add(g["boss"])
    raw = {d: {"n": 0, "missing_obs": 0, "no_alternative_boss": 0, "boss_states": 0,
               "boss_tv": [], "boss_flip": [], "hp_tv": [], "hp_flip": []} for d in PROBE_DECISIONS}
    for d in PROBE_DECISIONS:
        eligible = []
        o = raw[d]
        for r in rows:
            if r["decision"] != d or len(r["cands"]) < 2:
                continue
            o["n"] += 1
            if not r.get("obs"):
                o["missing_obs"] += 1
            else:
                eligible.append(r)
        for start in range(0, len(eligible), batch_size):
            part = eligible[start:start + batch_size]
            baseline = predict(part)
            probes, refs = [], []
            for r, p0 in zip(part, baseline):
                g = r["obs"].get("g") or {}
                alts = sorted(bosses[(g.get("act_id"), g.get("act"))] - {g.get("boss")}) if g.get("boss") else []
                o["boss_states"] += bool(alts)
                o["no_alternative_boss"] += not bool(alts)
                for b in alts:
                    r2 = copy.deepcopy(r)
                    r2["obs"]["g"]["boss"] = b
                    probes.append(r2)
                    refs.append(("boss", p0))
                if g.get("hp") is not None:
                    r2 = copy.deepcopy(r)
                    r2["obs"]["g"]["hp"] = max(1, int(g["hp"] * .7))
                    if r2["obs"].get("combat"):
                        r2["obs"]["combat"]["player"]["hp"] = r2["obs"]["g"]["hp"]
                    probes.append(r2)
                    refs.append(("hp", p0))
            for (kind, p0), p in zip(refs, predict(probes)):
                o[f"{kind}_tv"].append(.5 * float(np.abs(p0 - p).sum()))
                o[f"{kind}_flip"].append(int(p0.argmax() != p.argmax()))

    def summarize(o):
        return {"n": o["n"], "missing_obs": o["missing_obs"], "no_alternative_boss": o["no_alternative_boss"],
                "boss_states": o["boss_states"],
                **{label: {"tv": mean(o[f"{kind}_tv"]), "argmax_flip": rate(sum(o[f"{kind}_flip"]), len(o[f"{kind}_flip"]))}
                   for kind, label in (("boss", "boss"), ("hp", "hp_minus_30pct"))}}

    combined = {k: sum(o[k] for o in raw.values()) if isinstance(raw[PROBE_DECISIONS[0]][k], int)
                else [v for o in raw.values() for v in o[k]] for k in raw[PROBE_DECISIONS[0]]}
    return {"bosses": [{"act_id": k[0], "act": k[1], "bosses": sorted(v)} for k, v in sorted(bosses.items(), key=lambda kv: str(kv[0]))],
            "by_decision": {d: summarize(o) for d, o in raw.items()}, "all": summarize(combined)}


def metric_rows(result, prefix=""):
    for key, v in result.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(v, dict):
            if "rate" in v or "mean" in v:
                yield path, v
            else:
                yield from metric_rows(v, path)


def print_summary(result):
    print(f"{'metric':64s} {'n':>7s} {'value':>10s} {'95% Wilson':>19s}")
    for path, v in metric_rows(result):
        if "rate" in v:
            value = "-" if v["rate"] is None else f"{v['rate']:.1%}"
            lo, hi = v["ci95"]
            ci = "-" if lo is None else f"[{lo:.1%}, {hi:.1%}]"
        else:
            value = "-" if v["mean"] is None else f"{v['mean']:.4f}"
            ci = ""
        print(f"{path:64s} {v['n']:7d} {value:>10s} {ci:>19s}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args(argv)
    if a.batch_size < 1:
        ap.error("--batch_size must be positive")
    paths = [os.path.expanduser(p) for p in a.traces]
    meta, runs = load_traces(paths)
    result = compute_metrics(runs, meta)
    result["traces"] = [os.path.abspath(p) for p in paths]
    if a.ckpt:
        result["boss_sensitivity"] = sensitivity(runs, checkpoint_policy(a.ckpt, a.device, a.batch_size), a.batch_size)
        result["boss_sensitivity"]["ckpt"] = os.path.abspath(os.path.expanduser(a.ckpt))
    out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print_summary(result)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
