#!/usr/bin/env python3
"""Diff headless vs real-client digests and report the FIRST divergence per run.

Each run directory holds headless.jsonl (record.py) and client.jsonl (the ParityOracle mod). Both carry, per
decision step, {decision, obs, cands} with the RL observation schema of engine/RL_PROTOCOL.md. The comparator walks
the steps in order and, inside a step, compares decision → obs (fixed key order) → cands, reporting the first path
whose values differ, with both values. Differences matching a KNOWN_ARTIFACTS rule are listed separately and do not
count as the divergence (values that are not game state, documented in the report).

Usage: compare.py <run_dir> [<run_dir> ...] [--json out.json] [--all-diffs]
       compare.py --selftest      (synthetic input, no game data needed)
"""
import argparse
import json
import os
import re
import sys

FLOAT_TOL = 1e-6
OBS_ORDER = ["g", "deck", "relics", "potions", "combat", "map", "ctx"]


def load(path):
    steps, meta, end = {}, None, None
    if not os.path.exists(path):
        return steps, meta, end
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "meta" in d:
                meta = d["meta"]
            elif "end" in d and "step" not in d:
                end = d
            elif "step" in d and "decision" in d:
                steps[d["step"]] = d
            elif "error" in d or "exec_error" in d:
                end = d
    return steps, meta, end


def diffs(a, b, path, out, limit=50):
    """Collect differing leaf paths between two JSON values (lists compared index-wise, then by length)."""
    if len(out) >= limit:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        keys = list(a.keys()) + [k for k in b.keys() if k not in a]
        for k in keys:
            if k not in a:
                out.append((f"{path}.{k}", "<missing>", b[k]))
            elif k not in b:
                out.append((f"{path}.{k}", a[k], "<missing>"))
            else:
                diffs(a[k], b[k], f"{path}.{k}", out, limit)
        return
    if isinstance(a, list) and isinstance(b, list):
        for i in range(min(len(a), len(b))):
            diffs(a[i], b[i], f"{path}[{i}]", out, limit)
        if len(a) != len(b):
            out.append((f"{path}.len", len(a), len(b)))
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if abs(a - b) > FLOAT_TOL:
            out.append((path, a, b))
        return
    if a != b:
        out.append((path, a, b))


def step_diffs(h, c):
    out = []
    if h.get("decision") != c.get("decision"):
        out.append(("decision", h.get("decision"), c.get("decision")))
    ho, co = h.get("obs") or {}, c.get("obs") or {}
    for k in OBS_ORDER + [k for k in ho if k not in OBS_ORDER]:
        diffs(ho.get(k), co.get(k), f"obs.{k}", out, limit=200)
    diffs(h.get("cands"), c.get("cands"), "cands", out, limit=400)
    return out


# Known definitional differences between the headless harness and the client, and client-side values that are not
# game state. Each rule gets (path, h, c, hstep, cstep) and returns a reason string if the difference is expected.
# (v0 had a rule for the synthetic MapRoom the old harness entered after every room; v1 no longer does that.)
def _trial_entrant(path, hv, cv, hs, cs):
    # Trial.CalculateVars draws EntrantNumber, a number printed in the event text, from Rng.Chaotic, which the
    # client's UI also consumes (animations, merchant lines, relic holder tweens); it never feeds back into play.
    # The observation no longer exports it (ObsShared.OptionVars); the rule stays for recordings made before.
    if path.endswith(".EntrantNumber"):
        return "trial_entrant_number"
    return None


LOC_KEY = re.compile(r"^[A-Z0-9_]+(\.[A-Za-z0-9_]+)+$")


def _localized_sv(path, hv, cv, hs, cs):
    # ctx.sv holds an event's StringVar values, i.e. display text. The headless LocManager returns the loc key
    # (e.g. "SOWN.title"), the client returns text in its UI language; both name the same model.
    if path.startswith("obs.ctx.sv.") and isinstance(hv, str) and isinstance(cv, str):
        return "localized_string_var"
    # Some event options use display text as their TextKey (Doll Room: the doll relic's title): same split.
    if path.endswith(".key") and isinstance(hv, str) and isinstance(cv, str) and LOC_KEY.match(hv) and not LOC_KEY.match(cv):
        return "localized_option_key"
    return None


def _hook_selection_ctx(path, hv, cv, hs, cs):
    # Engine obs bug of the recordings before cdd2ce5 (see the v1 parity report): for a selection started inside a
    # turn-start hook (HookPlayerChoiceContext) the engine's CardSelectCmd capture was gone by the time the selector
    # ran, so the headless ctx said sel "unknown" without source/prompt; the client keeps them. The current engine
    # captures the context in an AsyncLocal and refuses a selection without one, so this no longer occurs in new
    # recordings; the rule stays for re-comparing old ones.
    hctx = ((hs.get("obs") or {}).get("ctx") or {})
    if hs.get("decision") == "card_select" and hctx.get("sel") == "unknown" and path in ("obs.ctx.sel", "obs.ctx.source", "obs.ctx.prompt"):
        return "engine_bug:hook_selection_ctx"
    return None


KNOWN_ARTIFACTS = [_trial_entrant, _localized_sv, _hook_selection_ctx]


def classify(path, hv, cv, hs, cs):
    for rule in KNOWN_ARTIFACTS:
        r = rule(path, hv, cv, hs, cs)
        if r:
            return r
    return None


def compare_run(run_dir, all_diffs=False):
    hsteps, _, hend = load(os.path.join(run_dir, "headless.jsonl"))
    csteps, cmeta, cend = load(os.path.join(run_dir, "client.jsonl"))
    spec = {}
    if os.path.exists(os.path.join(run_dir, "run.json")):
        with open(os.path.join(run_dir, "run.json")) as f:
            spec = json.load(f)
    res = {"run": os.path.basename(os.path.normpath(run_dir)), "character": spec.get("character"), "seed": spec.get("seed"),
           "headless_steps": len(hsteps), "client_steps": len(csteps), "client_end": (cend or {}).get("end"),
           "first_divergence": None, "artifacts": {}, "match_kinds": {}}
    if not csteps:
        res["status"] = "no_client_data"
        return res
    compared = 0
    for k in sorted(hsteps):
        if k not in csteps:
            break
        h, c = hsteps[k], csteps[k]
        if c.get("decision") == "stuck":
            res["stuck"] = {"step": k, "headless_decision": h.get("decision"), "note": c.get("note")}
            break
        act = c.get("action") or {}
        if act:
            res["match_kinds"][act.get("match")] = res["match_kinds"].get(act.get("match"), 0) + 1
        compared += 1
        real = []
        for path, hv, cv in step_diffs(h, c):
            why = classify(path, hv, cv, h, c)
            if why:
                res["artifacts"][why] = res["artifacts"].get(why, 0) + 1
            else:
                real.append((path, hv, cv))
        if real and res["first_divergence"] is None:
            path, hv, cv = real[0]
            res["first_divergence"] = {
                "step": k, "decision_headless": h.get("decision"), "decision_client": c.get("decision"),
                "field": path, "headless": hv, "client": cv, "n_diffs_at_step": len(real),
                "other_fields": [p for p, _, _ in real[1:8]],
                "floor": (h.get("obs") or {}).get("g", {}).get("total_floor"),
                "room": (h.get("obs") or {}).get("g", {}).get("room"),
                "prev_action": (csteps.get(k - 1) or {}).get("action", {}).get("want"),
                "client_note": c.get("note"),
            }
            if not all_diffs:
                break
    res["compared_steps"] = compared
    if res["first_divergence"]:
        res["status"] = "diverged"
    elif res.get("stuck"):
        res["status"] = "client_stuck"  # harness could not reach the next decision; no state difference before it
    elif compared == len(hsteps):
        res["status"] = "identical"
    else:
        res["status"] = "identical_prefix"  # client stopped early (stuck/limit) without a state difference
    return res


def selftest(verbose=True, scratch=None):
    """Synthetic input: a headless log with the current schema (g.odds / act_rooms, combat.played / turn / draw_top /
    pturn, card rewards with inline cards / alts, an event with string vars), and client logs derived from it: identical, one diverging field, one known artifact
    plus a later real divergence, a stuck client, no client data. Returns True when every case is classified right."""
    import shutil
    scratch = scratch or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp", "parity", "selftest")
    card = {"id": "CARD.STRIKE_IRONCLAD", "up": 0, "cost": 1, "x": False, "star": -1, "type": "Attack", "rar": "Basic",
            "tgt": "AnyEnemy", "v": {"Damage": 6}, "kw": [], "ench": None, "ench_n": 0, "aff": None, "aff_n": 0}
    g = {"act": 1, "floor": 1, "total_floor": 1, "asc": 0, "gold": 99, "hp": 80, "max_hp": 80, "char": "CHARACTER.IRONCLAD",
         "potion_slots": 3, "room": "Monster", "boss": "ENCOUNTER.X", "boss2": None, "act_id": "ACT.OVERGROWTH",
         "removals_used": 0, "odds": {"q_monster": 0.1, "q_shop": 0.03, "q_treasure": 0.02, "rare": -0.05, "potion": 0.4},
         "act_rooms": [{"t": "Event", "id": "EVENT.NEOW"}, {"t": "Monster", "id": "ENCOUNTER.X_WEAK"}]}
    combat = {"round": 1, "played": 1, "turn": {"first": 1, "attacks": 1}, "enc": "ENCOUNTER.X_WEAK", "energy": 2, "max_energy": 3, "stars": 0, "hand": [dict(card, playable=True, pv={"Damage": 6})],
              "draw": [], "draw_top": [], "pturn": 1, "discard": [card], "exhaust": [], "player": {"id": "CHARACTER.IRONCLAD", "hp": 80, "max_hp": 80,
              "block": 0, "alive": True, "pw": []}, "allies": [], "enemies": [], "orbs": [], "orb_slots": 0}
    def obs(**kw):
        o = {"g": g, "deck": [card], "relics": [], "potions": [], "combat": None, "map": {"nodes": [], "edges": [], "cur": -1}, "ctx": {}}
        o.update(kw)
        return o
    steps = [
        {"decision": "event_choice", "obs": obs(ctx={"event": "EVENT.X", "sv": {"Card": "STRIKE_IRONCLAD.title"}}),
         "cands": [{"k": "option", "key": "EVENT.X.pages.0.options.A", "v": {}}]},
        {"decision": "combat_play", "obs": obs(combat=combat), "cands": [{"k": "play", "hand": 0, "tgt": 0}, {"k": "end_turn"}]},
        {"decision": "rewards", "obs": obs(ctx={"terminal": True}),
         "cands": [{"k": "reward", "idx": 0, "rt": "Card", "n": 1, "cards": [card], "alts": ["Skip"]}, {"k": "proceed"}]},
        {"decision": "card_reward", "obs": obs(ctx={"alts": ["Skip"]}), "cands": [{"k": "reward_card", "card": card}, {"k": "skip"}]},
        {"decision": "game_over", "obs": obs(), "cands": []},
    ]
    def write(name, client_steps, end="game_over"):
        d = os.path.join(scratch, name)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        with open(os.path.join(d, "run.json"), "w") as f:
            json.dump({"id": name, "character": "Ironclad", "ascension": 0, "seed": name}, f)
        with open(os.path.join(d, "headless.jsonl"), "w") as f:
            for i, s in enumerate(steps):
                f.write(json.dumps(dict(s, step=i, info={})) + "\n")
        if client_steps is not None:
            with open(os.path.join(d, "client.jsonl"), "w") as f:
                f.write(json.dumps({"meta": {"id": name}}) + "\n")
                for i, s in enumerate(client_steps):
                    line = dict(s, step=i)
                    if i < len(client_steps) - 1 and s["decision"] not in ("game_over", "stuck"):
                        line["action"] = {"want": s["cands"][0], "idx": 0, "match": "exact"}
                    f.write(json.dumps(line) + "\n")
                f.write(json.dumps({"end": end, "steps": len(client_steps) - 1, "seconds": 1.0}) + "\n")
        return d
    def changed(i, path, value):
        cs = json.loads(json.dumps(steps))
        o = cs[i]
        for k in path[:-1]:
            o = o[k]
        o[path[-1]] = value
        return cs
    cases = []
    r = compare_run(write("identical", steps))
    cases.append(("identical", r["status"] == "identical" and r["compared_steps"] == len(steps)))
    r = compare_run(write("played", changed(1, ["obs", "combat", "played"], 0)))
    fd = r["first_divergence"] or {}
    cases.append(("diverged on combat.played", r["status"] == "diverged" and fd.get("step") == 1 and fd.get("field") == "obs.combat.played"))
    both = changed(0, ["obs", "ctx", "sv", "Card"], "Strike")
    both[2]["cands"][0]["cards"] = []
    r = compare_run(write("artifact_then_cards", both))
    fd = r["first_divergence"] or {}
    cases.append(("localized sv counted, then diverged on inline cards",
                  r["artifacts"].get("localized_string_var") == 1 and fd.get("step") == 2 and fd.get("field", "").startswith("cands[0].cards")))
    stuck = steps[:3] + [{"decision": "stuck", "obs": {}, "cands": [], "note": "synthetic"}]
    r = compare_run(write("stuck", stuck, end="stuck"))
    cases.append(("client stuck", r["status"] == "client_stuck" and r["stuck"]["step"] == 3))
    r = compare_run(write("no_client", None))
    cases.append(("no client data", r["status"] == "no_client_data"))
    ok = all(c[1] for c in cases)
    if verbose:
        for name, good in cases:
            print(f"  {'ok  ' if good else 'FAIL'} {name}")
        print("selftest:", "ok" if ok else "FAILED", f"(synthetic runs in {scratch})")
    return ok


def short(v, n=140):
    s = json.dumps(v, sort_keys=True) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*")
    ap.add_argument("--json")
    ap.add_argument("--all-diffs", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="classify synthetic runs (tmp/parity/selftest/) and exit")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    if not a.run_dirs:
        ap.error("no run directories")
    results = [compare_run(d, a.all_diffs) for d in a.run_dirs if os.path.isdir(d)]
    for r in results:
        fd = r["first_divergence"]
        print(f"{r['run']:<28} {r['status']:<17} compared {r['compared_steps'] if 'compared_steps' in r else 0:>4}/"
              f"{r['headless_steps']:<4} client_end={r['client_end']} matches={r['match_kinds']} artifacts={r['artifacts']}")
        if fd:
            print(f"    first divergence @step {fd['step']} ({fd['decision_headless']} vs {fd['decision_client']}, "
                  f"floor {fd['floor']}, room {fd['room']}): {fd['field']}")
            print(f"      headless: {short(fd['headless'])}")
            print(f"      client:   {short(fd['client'])}")
            if fd["other_fields"]:
                print(f"      also: {', '.join(fd['other_fields'])}")
            if fd["prev_action"] is not None:
                print(f"      after action: {short(fd['prev_action'])}")
        if r.get("stuck"):
            print(f"    client stuck @step {r['stuck']['step']} (headless {r['stuck']['headless_decision']}): {r['stuck']['note']}")
    statuses = {}
    for r in results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print("summary:", statuses)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
