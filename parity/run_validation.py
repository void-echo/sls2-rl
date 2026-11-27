#!/usr/bin/env python3
"""Real-client validation of the headless engine in one command (parity oracle, see parity/README.md).

  run_validation.py record   record the validation batch on the private engine copy (no game): regular runs
                             (random / greedy / per-turn search, mostly Ironclad, A0 and A10), scenario runs for every
                             navigation rule of engine/RL_PROTOCOL.md ("No navigation cycles") and for the custom
                             screens, the two orphaned-hook-action cases; writes manifest.json and coverage.json
  run_validation.py check    everything short of launching the game: builds the mod, validates every run directory
                             and the per-session job files (oracle.py check), runs compare.py's self-test on synthetic
                             input, prints the plan and the expected duration
  run_validation.py run      the validation: pre-flight check, installs the mod, plays all sessions unattended through
                             oracle.py (backup, other mods off, windowed + muted, watchdog, log-flood guard, quit,
                             restore + byte-identical check per session), scans each session's godot.log for engine
                             faults (the orphaned-hook-action error), removes the mod at the end whatever happens,
                             compares every run and prints / writes the summary (summary.txt, summary.json)

`run` options: --chunk N (runs per game session, default 8), --only g1,g2 (groups), --resume (skip runs that already
have a finished client.jsonl), --keep-mod (leave the mod installed, inert), --max-sessions N.
Batch directory: tmp/parity/runs/<batch>/ (default v2, the orphaned-hook-action cases in <batch>/orphan_hook/),
engine copy: tmp/parity_engine_v2/ (see headless.py). The earlier batch v1b (engine 3405f5c) is kept as recorded and
played; `--batch v1b` still reads it.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import compare  # noqa: E402
import oracle  # noqa: E402
from headless import DEFAULT_ENGINE  # noqa: E402

RUNS_ROOT = os.path.join(ROOT, "tmp", "parity", "runs")
ORPHAN_MAKER = os.path.join(ROOT, "tmp", "parity", "orphan_hook", "make_case.py")


def _potions(*ids):
    return [{"cmd": "dbg_procure_potion", "id": i} for i in ids]


ANYTIME = _potions("FRUIT_JUICE", "BLOOD_POTION", "ENTROPIC_BREW")   # usable on reward / selection screens
ANYTIME2 = _potions("FRUIT_JUICE", "BLOOD_POTION")


def room(t, **kw):
    return dict({"cmd": "rl_enter_room", "type": t}, **kw)


def _cards(*ids):
    return [{"cmd": "dbg_add_card", "id": i} for i in ids]


COUNTER_RELICS = ["KUNAI", "SHURIKEN", "PEN_NIB", "LETTER_OPENER", "ORNAMENTAL_FAN", "NUNCHAKU"]


# (group, record.py options). Seeds are "vb_<group>_<n>"; every group records into <batch>/<group>/.
GROUPS = [
    # regular runs (58): mostly Ironclad, A0 and A10, three test drivers
    ("rand", dict(runs=20, characters="Ironclad", ascension="0,0,10", policy="random", max_steps=400)),
    ("rand5", dict(runs=4, characters="Silent,Defect,Regent,Necrobinder", ascension="0,10", policy="random", max_steps=300)),
    # v0.113.1 (batch v3): the new character Fool (arrows, melody tags, tarot cards, pets) and more of the other four
    ("fool", dict(runs=10, characters="Fool", ascension="0,0,10", policy="random", max_steps=400)),
    ("foolg", dict(runs=6, characters="Fool", ascension="0,10", policy="greedy", explore=0.1, max_steps=700)),
    ("rand5b", dict(runs=10, characters="Silent,Defect,Regent,Necrobinder,Ironclad", ascension="0,10", policy="random", max_steps=300)),
    ("greedy", dict(runs=12, characters="Ironclad", ascension="0,10", policy="greedy", explore=0.1, max_steps=700)),
    ("search", dict(runs=6, characters="Ironclad", ascension="0,10", policy="search", rollouts=6, max_steps=700)),
    ("nav", dict(runs=10, characters="Ironclad", ascension="0,10", policy="nav", max_steps=400, setup=ANYTIME)),
    ("rooms", dict(runs=6, characters="Ironclad", ascension="0,10", policy="greedy", explore=0.2,
                   prefer_rooms="Shop,RestSite,Treasure", max_steps=600)),
    # scenario runs: every removed navigation cycle
    ("drift", dict(runs=3, characters="Ironclad", policy="nav", max_steps=300,
                   setup=[{"cmd": "obtain_relics", "ids": ["DRIFTWOOD"]}] + ANYTIME)),
    ("grimoire", dict(runs=4, characters="Ironclad", policy="nav", max_steps=250, play_first="CARD.FORBIDDEN_GRIMOIRE",
                      setup=[{"cmd": "dbg_add_card", "id": "FORBIDDEN_GRIMOIRE"}] + ANYTIME2 + [room("Monster")])),
    ("smith", dict(runs=2, characters="Ironclad", policy="nav", max_steps=200, setup=ANYTIME2 + [room("Rest")])),
    ("cook", dict(runs=2, characters="Ironclad", policy="nav", max_steps=200,
                  setup=[{"cmd": "obtain_relics", "ids": ["MEAT_CLEAVER"]}] + ANYTIME2 + [room("Rest")])),
    ("shoprem", dict(runs=2, characters="Ironclad", policy="nav", max_steps=200, setup=ANYTIME2 + [room("Shop")])),
    ("sozu", dict(runs=2, characters="Ironclad", policy="nav", max_steps=250,
                  setup=[{"cmd": "obtain_relics", "ids": ["SOZU", "WHITE_BEAST_STATUE"]}, room("Shop")])),
    # custom screens and later acts
    ("sphere", dict(runs=2, characters="Ironclad", policy="random", max_steps=200, setup=[room("Event", event="CRYSTAL_SPHERE")])),
    ("trial", dict(runs=2, characters="Ironclad", policy="random", max_steps=200, setup=[room("Event", event="TRIAL")])),
    ("fakem", dict(runs=1, characters="Ironclad", policy="random", max_steps=200, setup=[room("Event", event="FAKE_MERCHANT")])),
    ("fakefoul", dict(runs=1, characters="Ironclad", policy="random", max_steps=200,
                      setup=_potions("FOUL_POTION") + [room("Event", event="FAKE_MERCHANT")])),
    ("act2", dict(runs=2, characters="Ironclad", ascension="0,10", policy="search", rollouts=6, max_steps=400,
                  setup=[{"cmd": "dbg_enter_act", "act": 2}])),
    ("act3", dict(runs=2, characters="Ironclad", ascension="0,10", policy="search", rollouts=6, max_steps=400,
                  setup=[{"cmd": "dbg_enter_act", "act": 3}])),
    # scenario runs for the observation fields added after 3405f5c: known draw-pile top, dead enemies that stay to
    # revive, stunned moves, the shop shelf and blocked potion rewards, an extra turn, orb order, counter relics (and the
    # mod's wait for their activation flash), an option relic with its own state (Orobas' Sea Glass)
    ("headbutt", dict(runs=2, characters="Ironclad", ascension="0,10", policy="random", max_steps=200, play_first="CARD.HEADBUTT",
                      setup=_cards("HEADBUTT", "HEADBUTT", "HEADBUTT") + [room("Monster")])),
    ("decim", dict(runs=2, characters="Ironclad", ascension="0,10", policy="greedy", max_steps=200, play_first="CARD.BLUDGEON",
                   setup=_cards("BLUDGEON", "BLUDGEON", "BLUDGEON") + [room("Elite", encounter="DECIMILLIPEDE_ELITE")])),
    ("whistle", dict(runs=2, characters="Ironclad", ascension="0,10", policy="random", max_steps=200, play_first="CARD.WHISTLE",
                     setup=_cards("WHISTLE", "WHISTLE") + [room("Monster")])),
    ("shopfull", dict(runs=2, characters="Silent", ascension="0,10", policy="random", max_steps=150,
                      setup=_potions("FIRE_POTION", "BLOCK_POTION", "SWIFT_POTION") + [room("Shop")])),
    ("rewfull", dict(runs=2, characters="Ironclad", ascension="0,10", policy="random", max_steps=200, hoard_potions=True,
                     setup=_potions("FIRE_POTION", "BLOCK_POTION", "SWIFT_POTION")
                     + [{"cmd": "obtain_relics", "ids": ["WHITE_BEAST_STATUE"]}, room("Monster")])),
    ("amber", dict(runs=2, characters="Ironclad", ascension="0,10", policy="random", max_steps=200, play_first="POTION.AMBERGRIS",
                   setup=_potions("AMBERGRIS") + [room("Monster")])),
    ("orbs", dict(runs=2, characters="Defect", ascension="0,10", policy="random", max_steps=200,
                  setup=_cards("BALL_LIGHTNING", "COOLHEADED", "GLACIER", "ZAP") + [room("Monster")])),
    ("counters", dict(runs=2, characters="Ironclad", ascension="0,10", policy="random", max_steps=200,
                      setup=[{"cmd": "obtain_relics", "ids": COUNTER_RELICS}, room("Monster")])),
    ("orobas", dict(runs=2, characters="Ironclad", ascension="0,10", policy="random", max_steps=150,
                    setup=[room("Event", event="OROBAS")])),
]
# Played first in the game: short cases, then scenarios, then the long regular runs.
ORDER = ["fool", "foolg", "rand5b", "orphan_hook", "drift", "grimoire", "smith", "cook", "shoprem", "sozu", "sphere", "trial", "fakem", "fakefoul",
         "headbutt", "decim", "whistle", "shopfull", "rewfull", "amber", "orbs", "counters", "orobas",
         "act2", "act3", "nav", "rand5", "rand", "rooms", "greedy", "search"]


def batch_dir(batch):
    return os.path.join(RUNS_ROOT, batch)


def orphan_dir(batch):
    """The orphaned-hook-action cases of a batch (batch v1b kept them in tmp/parity/runs/orphan_hook/)."""
    return os.path.join(RUNS_ROOT, "orphan_hook") if batch == "v1b" else os.path.join(batch_dir(batch), "orphan_hook")


def rel(p):
    return os.path.relpath(os.path.abspath(p), ROOT)


# ------------------------------------------------------------------ record

def record_args(group, opt, out, engine):
    a = [sys.executable, os.path.join(HERE, "record.py"), "--out", out, "--seed-prefix", f"vb_{group}_", "--fresh",
         "--engine", engine, "--runs", str(opt["runs"]), "--characters", opt["characters"],
         "--ascension", opt.get("ascension", "0"), "--policy", opt["policy"], "--max-steps", str(opt["max_steps"])]
    for k, flag in (("explore", "--explore"), ("rollouts", "--rollouts"), ("prefer_rooms", "--prefer-rooms"),
                    ("play_first", "--play-first")):
        if k in opt:
            a += [flag, str(opt[k])]
    if opt.get("hoard_potions"):
        a.append("--hoard-potions")
    if opt.get("setup"):
        a += ["--setup", json.dumps(opt["setup"])]
    return a


def run_dirs_in(d):
    return sorted(os.path.join(d, x) for x in os.listdir(d) if os.path.exists(os.path.join(d, x, "run.json"))) if os.path.isdir(d) else []


def cmd_record(a):
    out = batch_dir(a.batch)
    os.makedirs(out, exist_ok=True)
    groups = [g for g in GROUPS if not a.only or g[0] in a.only.split(",")]
    t0 = time.time()

    def one(g):
        name, opt = g
        d = os.path.join(out, name)
        if run_dirs_in(d) and not a.force:
            return name, f"kept ({len(run_dirs_in(d))} runs; --force to re-record)"
        if a.force and os.path.isdir(d):
            import shutil
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "record.log"), "w") as log:
            r = subprocess.run(record_args(name, opt, d, a.engine), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        return name, f"exit {r.returncode}, {len(run_dirs_in(d))} runs, {time.time() - t0:.0f}s"

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for name, msg in pool.map(one, groups):
            print(f"  {name:<10} {msg}", flush=True)
    if not a.only or "orphan_hook" in a.only.split(","):
        record_orphans(a)
    write_manifest(a.batch, a.engine)
    cov = coverage(load_manifest(a.batch))
    with open(os.path.join(out, "coverage.json"), "w") as f:
        json.dump(cov, f, indent=1)
    print_coverage(cov)
    print(f"recorded in {time.time() - t0:.0f}s; manifest {rel(os.path.join(out, 'manifest.json'))}")


def record_orphans(a):
    """The two orphaned-hook-action cases (docs/survey/harness_audit.md), recorded into the batch with its engine
    (re-recorded when stale or with --force)."""
    od = orphan_dir(a.batch)
    stale = [d for d in run_dirs_in(od) if oracle.check_run_dir(d)[0]]
    if run_dirs_in(od) and not stale and not a.force:
        print(f"  orphan_hook kept ({len(run_dirs_in(od))} runs)")
        return
    if not os.path.exists(ORPHAN_MAKER):
        print(f"  orphan_hook: {rel(ORPHAN_MAKER)} missing, skipped")
        return
    if a.batch == "v1b":
        raise SystemExit("batch v1b is the played record of engine 3405f5c; record a new batch instead")
    if os.path.isdir(od):
        import shutil
        shutil.rmtree(od)
    r = subprocess.run([sys.executable, ORPHAN_MAKER, "--engine", a.engine, "--out", od], cwd=ROOT,
                       capture_output=True, text=True)
    print("  orphan_hook", (r.stdout.strip().splitlines() or ["?"])[-1], f"(exit {r.returncode})")


def engine_commit(engine):
    p = os.path.join(engine, "ENGINE_COMMIT")
    if os.path.exists(p):
        return open(p).read().strip()
    return None


def write_manifest(batch, engine):
    out = batch_dir(batch)
    groups = {name: [rel(d) for d in run_dirs_in(os.path.join(out, name))] for name, _ in GROUPS}
    groups["orphan_hook"] = [rel(d) for d in run_dirs_in(orphan_dir(batch))]
    man = {"batch": batch, "engine": rel(engine), "engine_commit": engine_commit(engine), "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
           "groups": {g: groups[g] for g in ORDER if groups.get(g)}}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)


def load_manifest(batch):
    p = os.path.join(batch_dir(batch), "manifest.json")
    if not os.path.exists(p):
        raise SystemExit(f"{rel(p)} missing: run `python3 parity/run_validation.py record` first")
    with open(p) as f:
        return json.load(f)


def manifest_dirs(man, only=None):
    out = []
    for g in ORDER:
        if only and g not in only:
            continue
        out += [(g, os.path.join(ROOT, d)) for d in man["groups"].get(g, [])]
    return out


# ------------------------------------------------------------------ coverage of the recorded batch

def _jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def new_fields(h, inc):
    """The observation fields added after engine 3405f5c that a recorded step exercises."""
    obs, cands = h["obs"], h["cands"]
    g, cb, ctx = obs.get("g") or {}, obs.get("combat"), obs.get("ctx") or {}
    if g.get("removals_used"):
        inc("new:g.removals_used>0")
    if len({x.get("id") for x in g.get("act_rooms") or [] if x.get("t") in ("Monster", "Elite")}) >= 2:
        inc("new:g.act_rooms>=2 fights")
    if any(r.get("cnt") is not None for r in obs.get("relics") or []):
        inc("new:relic.cnt")
    if any(r.get("status") for r in obs.get("relics") or []):
        inc("new:relic.status")
    if any(p.get("v") for p in obs.get("potions") or []):
        inc("new:potion.v")
    if ctx.get("unavailable"):
        inc("new:ctx.unavailable")
    if ctx.get("blocked"):
        inc("new:ctx.blocked")
    if cb:
        if cb.get("draw_top"):
            inc("new:combat.draw_top")
        if cb.get("pturn") != cb.get("round"):
            inc("new:combat.pturn!=round")
        if len(cb.get("orbs") or []) >= 2:
            inc("new:combat.orbs>=2")
        for e in cb.get("enemies") or []:
            if not e.get("alive"):
                inc("new:enemy dead, listed")
            if e.get("move") == "STUNNED":
                inc("new:enemy.move STUNNED")
            for k in ("move", "then", "prev", "slot", "d"):
                if k in e:
                    inc("new:enemy." + k)
        for c in cb.get("hand") or []:
            for k in ("glow", "played_turn", "cost_mods", "replay"):
                if k in c:
                    inc("new:hand card." + k)
        for cr in [cb.get("player")] + (cb.get("enemies") or []):
            for p in (cr or {}).get("pw") or []:
                for k in ("v", "d", "cards", "dn"):
                    if k in p:
                        inc("new:power." + k)
    for c in cands:
        if c["k"] == "play" and "pv" in c:
            inc("new:play.pv " + ("targeted" if c["tgt"] >= 0 else "all/random"))
        if c["k"] == "rest" and "heal" in c:
            inc("new:rest.heal")
        if c["k"] == "option" and "relic_d" in c:
            inc("new:option.relic_d")


def run_coverage(d, cov):
    head = {h["step"]: h for h in _jsonl(os.path.join(d, "headless.jsonl")) if "decision" in h}
    acts = _jsonl(os.path.join(d, "actions.jsonl"))
    c = cov.setdefault("counts", {})

    def inc(k, n=1):
        c[k] = c.get(k, 0) + n
    screen = None   # (kind, n_inputs, reward idx) of the reward sub-screen being shown
    held = set()
    service = None
    for a in acts:
        h = head[a["step"]]
        dec, cand, obs = h["decision"], a["cand"], h["obs"]
        inc("decision:" + dec)
        cb = obs.get("combat")
        if cb and cb.get("played", 0) > 0:
            inc("obs:combat.played>0")
        new_fields(h, inc)
        if dec == "rewards":
            if (obs.get("ctx") or {}).get("must_take_all"):
                inc("rewards:must_take_all")
            offered = {x.get("idx") for x in h["cands"] if x["k"] == "reward"}
            for i in list(held):
                if i in offered:
                    inc("rewards:held_reward_offered_again")
                    held.discard(i)
            if any(x["k"] == "reward" and x.get("rt") == "Card" and "alts" in x for x in h["cands"]):
                inc("rewards:card_reward_inline")
            screen = (cand.get("rt"), 0, cand.get("idx")) if cand["k"] == "reward" and cand.get("rt") in ("Card", "RemoveCard") else None
        elif dec == "card_reward" and screen and screen[0] == "Card":
            if cand["k"] == "skip":
                inc("card_reward:skip_declines" if screen[1] == 0 else "card_reward:skip_backout_held")
                if screen[1] > 0:
                    held.add(screen[2])
                screen = None
            else:
                if cand["k"] in ("reroll", "potion", "discard_potion"):
                    inc(f"card_reward:{cand['k']}")
                screen = (screen[0], screen[1] + 1, screen[2])
        elif dec == "card_select" and screen and screen[0] == "RemoveCard":
            if cand["k"] == "skip":
                inc("removal_reward:skip_declines" if screen[1] == 0 else "removal_reward:skip_backout_held")
                if screen[1] > 0:
                    held.add(screen[2])
                screen = None
            elif cand["k"] in ("pick", "confirm"):
                inc("removal_reward:removed")
                screen = None
            else:
                screen = (screen[0], screen[1] + 1, screen[2])
        if dec == "card_select" and service:
            ks = {x["k"] for x in h["cands"]}
            inc(f"{service}:selection" + ("_WITH_SKIP" if "skip" in ks else "_no_skip"))
            if cand["k"] in ("potion", "discard_potion"):
                inc(f"{service}:potion_inside")
        if dec != "card_select":
            service = None
        if dec == "rest_site" and cand["k"] == "rest" and cand["key"] in ("SMITH", "COOK"):
            service = "rest:" + cand["key"]
        if dec == "shop":
            if cand["k"] == "remove":
                service = "shop:REMOVE"
            relics = {r["id"] for r in obs.get("relics", [])}
            if "RELIC.SOZU" in relics:
                inc("shop:with_sozu")
                if any(x["k"] == "buy_potion" for x in h["cands"]):
                    inc("shop:with_sozu_BUY_POTION_OFFERED")
            inc("shop:buy_potion_offered", sum(1 for x in h["cands"] if x["k"] == "buy_potion"))
        if dec == "event_choice" and (obs.get("ctx") or {}).get("event") == "EVENT.TRIAL":
            inc("event:TRIAL")
        if cand["k"] in ("reroll", "sacrifice", "alt"):
            inc("cand:" + cand["k"])
        inc(f"act:{obs['g']['act']}")
    spec = json.load(open(os.path.join(d, "run.json")))
    cov.setdefault("ends", {})
    cov["ends"][spec.get("end")] = cov["ends"].get(spec.get("end"), 0) + 1
    for k in ("character", "ascension", "policy"):
        cov.setdefault(k, {})
        v = str(spec.get(k))
        cov[k][v] = cov[k].get(v, 0) + 1
    return len(acts)


def coverage(man):
    cov = {"runs": 0, "steps": 0}
    for g, d in manifest_dirs(man):
        cov["runs"] += 1
        cov["steps"] += run_coverage(d, cov)
    cov["counts"] = dict(sorted(cov.get("counts", {}).items()))
    return cov


def print_coverage(cov):
    print(f"coverage: {cov['runs']} runs, {cov['steps']} recorded steps; ends {cov.get('ends')}")
    print(f"  characters {cov.get('character')}, ascension {cov.get('ascension')}, policies {cov.get('policy')}")
    for k, v in cov["counts"].items():
        print(f"  {k:<44} {v}")


# ------------------------------------------------------------------ check (no launch)

def preflight(man, chunk, only=None, build=True):
    dirs = [d for _, d in manifest_dirs(man, only)]
    ok = True
    if build:
        try:
            oracle.build()
        except SystemExit as ex:
            print("PROBLEM: mod build failed:", ex)
            ok = False
    st = compare.selftest(verbose=False)
    print(f"compare.py self-test on synthetic input: {'ok' if st else 'FAILED'}")
    ok &= st
    good, plan = oracle.check(dirs, chunk=chunk)
    ok &= good
    if man.get("engine_commit"):
        print(f"recorded with {man['engine']} (engine {man['engine_commit']}) on {man['recorded']}")
    return ok, dirs, plan


def cmd_check(a):
    man = load_manifest(a.batch)
    only = a.only.split(",") if a.only else None
    ok, dirs, plan = preflight(man, a.chunk, only)
    print(f"ready: {ok}. `python3 parity/run_validation.py run` would play {plan['runs']} runs ({plan['steps']} steps) in "
          f"{plan['sessions']} game sessions of up to {a.chunk} runs: about {plan['estimated_s'] / 60:.0f} min "
          f"(hard cap {plan['hard_cap_s'] / 60:.0f} min)")
    raise SystemExit(0 if ok else 1)


# ------------------------------------------------------------------ run (launches the game)

class Tee:
    def __init__(self, path):
        self.f = open(path, "a")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def finished(d):
    p = os.path.join(d, "client.jsonl")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    try:
        last = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        return False
    return "end" in last and "step" not in last   # Driver.RunOne's closing line


# Game-log lines that mean an engine-level fault in the client even when the digests match (the orphaned hook
# action logs the first, the interactive branch of a failed action the second).
LOG_ALARMS = ("didn't find it in any queue", "completed with exception")


def scan_game_log():
    """Counts of LOG_ALARMS in the last session's godot.log (recreated at every launch)."""
    p = oracle.GAME_LOG
    try:
        if os.path.getsize(p) > oracle.LOG_FLOOD:
            return {"godot.log too large to scan": 1}
        with open(p, errors="replace") as f:
            text = f.read()
    except OSError:
        return {}
    return {a: text.count(a) for a in LOG_ALARMS if a in text}


def summarize(man, dirs_by_group, out, log_alarms=None):
    results = {}
    for g, d in dirs_by_group:
        r = compare.compare_run(d)
        r["group"] = g
        results[d] = r
    lines = []
    statuses, by_group, artifacts = {}, {}, {}
    steps = 0
    for d, r in results.items():
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        bg = by_group.setdefault(r["group"], {})
        bg[r["status"]] = bg.get(r["status"], 0) + 1
        steps += r.get("compared_steps", 0)
        for k, v in r["artifacts"].items():
            artifacts[k] = artifacts.get(k, 0) + v
    lines.append(f"parity validation, batch {man['batch']} (engine {man.get('engine_commit')}), {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"runs {len(results)}, compared steps {steps}; status {statuses}; known artifacts {artifacts}")
    if log_alarms is not None:
        lines.append(f"godot.log alarms per session: {log_alarms or 'none'}")
    for g in ORDER:
        if g in by_group:
            lines.append(f"  {g:<12} {by_group[g]}")
    for d, r in results.items():
        if r["status"] not in ("identical",):
            fd = r.get("first_divergence")
            line = f"  {r['group']}/{r['run']}: {r['status']} (compared {r.get('compared_steps', 0)}/{r['headless_steps']}, client_end={r['client_end']})"
            if fd:
                line += (f" first divergence @step {fd['step']} {fd['decision_headless']}/{fd['decision_client']} {fd['field']}: "
                         f"headless {compare.short(fd['headless'], 100)} | client {compare.short(fd['client'], 100)}")
            if r.get("stuck"):
                line += f" stuck: {r['stuck']}"
            lines.append(line)
    text = "\n".join(lines)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(list(results.values()), f, indent=1)
    return text


def cmd_run(a):
    man = load_manifest(a.batch)
    out = batch_dir(a.batch)
    sys.stdout = Tee(os.path.join(out, "validation.log"))
    print(f"\n=== run_validation {time.strftime('%Y-%m-%d %H:%M:%S')}")
    only = a.only.split(",") if a.only else None
    if oracle.game_running():
        raise SystemExit("the game is running; quit it first (a session never attaches to a running game)")
    ok, _, plan = preflight(man, a.chunk, only)
    if not ok:
        raise SystemExit("pre-flight check failed; nothing was launched")
    todo = [(g, d) for g, d in manifest_dirs(man, only) if not (a.resume and finished(d))]
    chunks = [todo[i:i + a.chunk] for i in range(0, len(todo), a.chunk)]
    if a.max_sessions:
        chunks = chunks[:a.max_sessions]
    steps = {d: sum(1 for _ in open(os.path.join(d, "actions.jsonl"))) for _, d in todo}
    est = sum(oracle.SESSION_OVERHEAD_S + oracle.CLIENT_S_PER_STEP * sum(steps[d] for _, d in c) for c in chunks)
    print(f"{len(todo)} runs ({sum(steps.values())} steps) in {len(chunks)} sessions; expected about {est / 60:.0f} min", flush=True)
    t0 = time.time()
    alarms = {}
    try:
        oracle.install()
        for j, c in enumerate(chunks):
            print(f"=== session {j + 1}/{len(chunks)}: {', '.join(os.path.basename(d) for _, d in c)}", flush=True)
            try:
                oracle.run_session([d for _, d in c], reinstall=False)
            except SystemExit as ex:   # e.g. the game got started by hand in between: stop, do not fight it
                print("session refused:", ex)
                break
            except Exception as ex:
                print(f"session {j + 1} failed: {type(ex).__name__}: {ex}")
            found = scan_game_log()
            if found:
                alarms[f"session {j + 1}"] = found
            print(f"   godot.log: {found or 'no alarms'}")
            for g, d in c:
                r = compare.compare_run(d)
                fd = r.get("first_divergence")
                print(f"   {g}/{r['run']}: {r['status']} {r.get('compared_steps', 0)}/{r['headless_steps']}"
                      + (f" @{fd['step']} {fd['field']}" if fd else ""), flush=True)
            print(f"   elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        if oracle.game_running():
            oracle.quit_game()
        job = os.path.join(oracle.CONTROL, "job.json")
        if os.path.exists(job):
            os.remove(job)
        if a.keep_mod:
            print("mod left installed (inert without tmp/parity/control/job.json)")
        else:
            oracle.uninstall()
    print(summarize(man, manifest_dirs(man, only), out, alarms))
    print(f"total {(time.time() - t0) / 60:.1f} min; summary in {rel(os.path.join(out, 'summary.txt'))}, log in "
          f"{rel(os.path.join(out, 'validation.log'))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default="v2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="record the batch on the private engine copy (no game)")
    r.add_argument("--engine", default=DEFAULT_ENGINE)
    r.add_argument("--jobs", type=int, default=6, help="groups recorded in parallel")
    r.add_argument("--only", default="", help="comma list of groups")
    r.add_argument("--force", action="store_true", help="re-record groups that already have runs")
    c = sub.add_parser("check", help="build the mod and validate everything without launching the game")
    c.add_argument("--chunk", type=int, default=8)
    c.add_argument("--only", default="")
    v = sub.add_parser("run", help="play the batch in the real client (launches the game)")
    v.add_argument("--chunk", type=int, default=8, help="runs per game session")
    v.add_argument("--only", default="", help="comma list of groups")
    v.add_argument("--resume", action="store_true", help="skip runs that already have a finished client.jsonl")
    v.add_argument("--keep-mod", action="store_true", help="leave the (inert) mod installed afterwards")
    v.add_argument("--max-sessions", type=int, default=0)
    a = ap.parse_args()
    {"record": cmd_record, "check": cmd_check, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
