#!/usr/bin/env python3
"""Navigation-cycle detector: flags every exact repeat of (decision, obs, cands) within one room / combat turn.

A repeat means a sequence of actions led back to an identical earlier decision state, i.e. a pure navigation loop
(for example opening a card reward from the rewards screen and skipping it). Under a deterministic policy (same
state -> same action) such a loop never ends, which is how a greedy evaluation policy gets stuck. The engine must
not offer one (engine/RL_PROTOCOL.md, "No navigation cycles").

Policies (--policy, comma list; the runs are spread over them):
  random   seeded weighted-random choice (tools/engine_runner.py `choose`)
  det      the same weights, but the draw is a hash of the state: a deterministic function of (decision, obs, cands)
  backout  deterministic and loop-seeking: backs out of every sub-screen that offers it (skip / cancel), opens every
           sub-screen it can (rewards, shop removal, Smith, Cook), otherwise acts like det
Under det and backout a repeat would loop forever, so the step after a detected repeat is a seeded random choice
(the run goes on and can find further loops); under random play simply continues.

  python3 tools/cycle_check.py --runs 300 --asc 0,10 --policy random,det,backout --workers 9
  python3 tools/cycle_check.py --runs 100 --god --policy det,backout --workers 9
Output: every cycle with its action path, a count per cycle signature, and the usual run outcomes (fullrun.py's
problem classes). Exit code 1 if any cycle or problem was found.
"""
import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_runner import Engine, EngineDied, Timeout, check_response, terminal_problem  # noqa: E402

CHARS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder", "Fool"]
GOOD = {"win", "loss", "abandon"}


def weights(r):
    w = []
    for c in r["cands"]:
        k = c["k"]
        w.append(0.05 if k == "discard_potion" else 0.5 if k == "end_turn" else 0.3 if k == "proceed" and r["decision"] == "rewards" else 1.0)
    return w


def pick_weighted(w, u):
    tot, acc = sum(w), 0.0
    for i, x in enumerate(w):
        acc += x
        if u * tot < acc:
            return i
    return len(w) - 1


def state_u(blob, salt):
    return int(hashlib.md5(salt.encode() + blob).hexdigest()[:13], 16) / float(16 ** 13)


def backout_choice(r, u):
    pri = []
    for c in r["cands"]:
        k, d = c["k"], r["decision"]
        if k == "skip" and d in ("card_reward", "card_select"):
            pri.append(3)
        elif k == "reward" or (k == "remove" and d == "shop") or (k == "rest" and c.get("key") in ("SMITH", "COOK")):
            pri.append(2)
        else:
            pri.append(0)
    best = max(pri)
    if best == 0:
        return None
    idx = [i for i, p in enumerate(pri) if p == best]
    return idx[int(u * len(idx)) % len(idx)]


def describe(r, c):
    k = c["k"]
    extra = c.get("rt") or c.get("key") or (c.get("card") or {}).get("id") or c.get("id")
    if isinstance(extra, str) and "." in extra:
        extra = extra.split(".")[-1]
    return f"{r['decision']}:{k}" + (f"({extra})" if extra else "")


def segment(r):
    g = r["obs"]["g"]
    cb = r["obs"].get("combat")
    return (g["act"], g["total_floor"], g.get("room"), cb["round"] if cb else None)


def play(eng, spec, max_steps):
    policy = spec["policy"]
    rng = random.Random(spec["policy_seed"])
    salt = f"{policy}:{spec['policy_seed']}"
    res = {"spec": spec, "steps": 0, "cycles": [], "decisions": collections.Counter()}
    r = eng.send({"cmd": "rl_start", "character": spec["character"], "ascension": spec["ascension"], "seed": spec["seed"]}, 120)
    if spec.get("god") and r.get("type") == "rl":
        r = eng.send({"cmd": "set_player", "hp": 9999, "max_hp": 9999})
    seg, seen, path = None, {}, []
    while True:
        if r.get("type") != "rl":
            res.update(result="error", msg=(r.get("message") or str(r))[:1500])
            break
        probs = check_response(r)
        if probs:
            res.update(result="schema", msg="; ".join(probs)[:600])
            break
        if r["done"]:
            bad = terminal_problem(r)
            res.update(result=bad or ("win" if r["victory"] else "abandon" if r["info"].get("abandoned") else "loss"),
                       floor=r["info"]["total_floor"], act=r["obs"]["g"]["act"])
            break
        if not r["cands"]:
            res.update(result="no_cands", msg=r["decision"])
            break
        blob = json.dumps([r["decision"], r["obs"], r["cands"]], sort_keys=True).encode()
        key = hashlib.sha1(blob).digest()
        s = segment(r)
        if s != seg:
            seg, seen, path = s, {}, []
        if key in seen:
            loop = path[seen[key]:]
            res["cycles"].append({"step": res["steps"], "len": len(loop), "sig": " > ".join(loop)[:400],
                                  "floor": r["obs"]["g"]["total_floor"], "act": r["obs"]["g"]["act"]})
            seen, path = {}, []   # count each further repeat once
            escape = True
        else:
            escape = False
        seen[key] = len(path)
        if res["steps"] >= max_steps:
            res.update(result="cap", msg=f"{max_steps} steps at {r['decision']}")
            break
        res["decisions"][r["decision"]] += 1
        if policy == "random" or escape:
            i = rng.choices(range(len(r["cands"])), weights=weights(r))[0]
        else:
            u = state_u(blob, salt)
            i = backout_choice(r, u) if policy == "backout" else None
            if i is None:
                i = pick_weighted(weights(r), u)
        path.append(describe(r, r["cands"][i]))
        r = eng.send({"cmd": "rl_act", "i": i})
        res["steps"] += 1
    res["decisions"] = dict(res["decisions"])
    return res


def worker(args, specs, out_q):
    eng = None
    for spec in specs:
        t0 = time.time()
        try:
            if eng is None:
                eng = Engine(args.engine, timeout=args.timeout, stderr_path=args.stderr)
            res = play(eng, spec, args.max_steps)
        except Timeout:
            res = {"spec": spec, "result": "timeout", "cycles": []}
            eng.p.kill()
            eng = None
        except EngineDied:
            res = {"spec": spec, "result": "died", "cycles": []}
            eng = None
        res["secs"] = round(time.time() - t0, 2)
        out_q.put(res)
        if res["result"] not in GOOD and eng is not None:
            eng.close()
            eng = None
    if eng:
        eng.close()
    out_q.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine"))
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--asc", default="0,10")
    ap.add_argument("--chars", default=",".join(CHARS))
    ap.add_argument("--policy", default="random,det,backout")
    ap.add_argument("--god", action="store_true", help="9999 HP after rl_start (reaches acts 2-3)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-prefix", default="cyc")
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stderr", default=None)
    args = ap.parse_args()
    chars, ascs, pols = args.chars.split(","), [int(x) for x in args.asc.split(",")], args.policy.split(",")
    specs = []
    for i in range(args.runs):
        specs.append({"character": chars[i % len(chars)], "ascension": ascs[(i // len(chars)) % len(ascs)],
                      "policy": pols[(i // (len(chars) * len(ascs))) % len(pols)],
                      "seed": f"{args.seed_prefix}{'g' if args.god else ''}_{i}", "policy_seed": i, "god": args.god})
    out = args.out or os.path.join("tmp", "nav", f"cycles{'_god' if args.god else ''}.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    q = mp.Queue()
    shards = [specs[i::args.workers] for i in range(args.workers)]
    procs = [mp.Process(target=worker, args=(args, s, q)) for s in shards if s]
    for p in procs:
        p.start()
    done, alive, t0 = [], len(procs), time.time()
    with open(out, "w") as f:
        while alive:
            r = q.get()
            if r is None:
                alive -= 1
                continue
            done.append(r)
            f.write(json.dumps(r, default=list) + "\n")
            f.flush()
            for c in r.get("cycles", []):
                print(f"  [cycle] {r['spec']['policy']} {r['spec']['character']} A{r['spec']['ascension']} {r['spec']['seed']} "
                      f"step {c['step']} floor {c['floor']}: {c['sig'][:200]}", flush=True)
            if r["result"] not in GOOD:
                print(f"  [{r['result']}] {r['spec']} {str(r.get('msg', ''))[:300]}", flush=True)
            if len(done) % 50 == 0:
                print(f"  ... {len(done)}/{len(specs)} in {time.time() - t0:.0f}s", flush=True)
    for p in procs:
        p.join()
    n_cyc = sum(len(r.get("cycles", [])) for r in done)
    sigs = collections.Counter(c["sig"] for r in done for c in r.get("cycles", []))
    for pol in pols:
        sub = [r for r in done if r["spec"]["policy"] == pol]
        print(f"policy {pol}: {len(sub)} runs, runs with a cycle {sum(1 for r in sub if r.get('cycles'))}, "
              f"cycles {sum(len(r.get('cycles', [])) for r in sub)}, outcomes {dict(collections.Counter(r['result'] for r in sub))}")
    print("cycle signatures:", len(sigs))
    for s, n in sigs.most_common(30):
        print(f"  {n:5d}  {s[:240]}")
    problems = sum(1 for r in done if r["result"] not in GOOD)
    steps = sum(r.get("steps", 0) for r in done)
    print(f"cycles: {n_cyc}  problems: {problems}  total steps {steps}, wall {time.time() - t0:.0f}s")
    sys.exit(1 if n_cyc or problems else 0)


if __name__ == "__main__":
    main()
