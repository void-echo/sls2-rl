#!/usr/bin/env python3
"""Same-process vs fresh-process identity of runs.

For each spec (character, ascension, seed, policy seed; half of them in 9999-HP "god" mode so they go deep into
acts 2-3) the hash chain of every (decision, obs, cands) of the run is computed
  fresh  - each run in its own new engine process,
  seq    - all runs one after another in ONE long-lived process,
  par    - P long-lived processes in parallel, each running a shuffled subset (every spec at least twice overall,
           at different positions after different predecessors).
Every run must have the identical hash in all three. Usage:
  python3 tools/determinism.py --runs 40 --parallel 32
"""
import argparse
import collections
import concurrent.futures as cf
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_runner import Engine, play_run  # noqa: E402

CHARS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder", "Fool"]


def specs_for(n):
    out = []
    for i in range(n):
        out.append({"character": CHARS[i % len(CHARS)], "ascension": (0, 10)[(i // len(CHARS)) % 2], "seed": f"det_{i}",
                    "policy_seed": 1000 + i, "god": i % 2 == 1})
    return out


def run_fresh(args, spec):
    eng = Engine(args.engine, timeout=args.timeout)
    try:
        return play_run(eng, spec, max_steps=args.max_steps)
    finally:
        eng.close()


def run_many(args, specs):
    eng = Engine(args.engine, timeout=args.timeout)
    try:
        return [play_run(eng, s, max_steps=args.max_steps) for s in specs]
    finally:
        eng.close()


def key(s):
    return s["seed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine"))
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--parallel", type=int, default=32)
    ap.add_argument("--per-engine", type=int, default=3)
    ap.add_argument("--workers", type=int, default=9, help="concurrency of the fresh phase")
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()
    specs = specs_for(args.runs)

    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        fresh = {key(r["spec"]): r for r in ex.map(lambda s: run_fresh(args, s), specs)}
    print(f"fresh: {len(fresh)} runs in {time.time() - t0:.0f}s; outcomes {dict(collections.Counter(r['result'] for r in fresh.values()))}")

    t0 = time.time()
    seq = {key(r["spec"]): r for r in run_many(args, specs)}
    print(f"seq:   {len(seq)} runs in one process in {time.time() - t0:.0f}s")

    rng = random.Random(7)
    plan = []
    pool = []
    for i in range(args.parallel):
        if len(pool) < args.per_engine:
            pool += rng.sample(specs, len(specs))
        plan.append([pool.pop() for _ in range(args.per_engine)])
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.parallel) as ex:
        par_results = list(ex.map(lambda ss: run_many(args, ss), plan))
    par = collections.defaultdict(list)
    for rs in par_results:
        for r in rs:
            par[key(r["spec"])].append(r)
    print(f"par:   {sum(len(v) for v in par.values())} runs in {args.parallel} parallel processes in {time.time() - t0:.0f}s "
          f"(covering {len(par)} specs)")

    bad = 0
    for s in specs:
        k = key(s)
        hs = {fresh[k]["hash"], seq[k]["hash"]} | {r["hash"] for r in par.get(k, [])}
        if len(hs) != 1:
            bad += 1
            print(f"MISMATCH {s}: fresh {fresh[k]['result']}/{fresh[k]['steps']} seq {seq[k]['result']}/{seq[k]['steps']} "
                  f"par {[(r['result'], r['steps']) for r in par.get(k, [])]}")
    deep = sum(1 for r in fresh.values() if max(r["acts"] or [0]) >= 2)
    print(f"identical: {len(specs) - bad}/{len(specs)} specs (fresh == same-process == parallel); "
          f"{deep} runs reached act 2+, {sum(1 for r in fresh.values() if r['result'] == 'win')} victories; "
          f"steps per run median {sorted(r['steps'] for r in fresh.values())[len(fresh) // 2]}")
    problems = [r for r in list(fresh.values()) + list(seq.values()) if r["result"] not in ("win", "loss", "abandon")]
    for r in problems[:10]:
        print("  problem:", r["result"], r["spec"], str(r.get("msg", ""))[:300])
    sys.exit(1 if bad or problems else 0)


if __name__ == "__main__":
    main()
