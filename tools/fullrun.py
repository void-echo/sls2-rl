#!/usr/bin/env python3
"""Many random full runs in parallel long-lived engines: errors, fake terminals, stuck states, acts/bosses reached.

  python3 tools/fullrun.py --runs 500 --asc 0,10 --workers 9 --out tmp/v1dev/fullrun.jsonl
  python3 tools/fullrun.py --runs 100 --god --workers 9      # 9999 HP: random play reaches the act 3 boss / The Architect

Each worker keeps one engine process for all its runs (the same-process reset is part of what is tested).
A run's result is one of: win, loss, abandon (genuine terminals), or error / schema / fake_loss / fake_win / stuck /
no_cands / cap / timeout / died (problems).
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_runner import Engine, EngineDied, Timeout, play_run  # noqa: E402

CHARS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder", "Fool"]
GOOD = {"win", "loss", "abandon"}


def worker(args, specs, out_q):
    eng = None
    for spec in specs:
        t0 = time.time()
        try:
            if eng is None:
                eng = Engine(args.engine, timeout=args.timeout, stderr_path=args.stderr)
            res = play_run(eng, spec, max_steps=args.max_steps)
        except Timeout:
            res = {"spec": spec, "result": "timeout"}
            eng.p.kill()
            eng = None
        except EngineDied:
            res = {"spec": spec, "result": "died"}
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
    ap.add_argument("--god", action="store_true", help="9999 HP after rl_start (scenario construction to reach acts 2-3)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-prefix", default="fr")
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stderr", default=None)
    args = ap.parse_args()
    chars = args.chars.split(",")
    ascs = [int(x) for x in args.asc.split(",")]
    specs = []
    for i in range(args.runs):
        specs.append({"character": chars[i % len(chars)], "ascension": ascs[(i // len(chars)) % len(ascs)],
                      "seed": f"{args.seed_prefix}{'g' if args.god else ''}_{i}", "policy_seed": i, "god": args.god})
    out = args.out or os.path.join("tmp", "v1dev", f"fullrun{'_god' if args.god else ''}.jsonl")
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
            if r["result"] not in GOOD:
                print(f"  [{r['result']}] {r['spec']} {str(r.get('msg', ''))[:300]}", flush=True)
            if len(done) % 50 == 0:
                print(f"  ... {len(done)}/{len(specs)} in {time.time() - t0:.0f}s", flush=True)
    for p in procs:
        p.join()
    outcomes = collections.Counter(r["result"] for r in done)
    print("outcomes:", dict(outcomes))
    print("problems:", sum(v for k, v in outcomes.items() if k not in GOOD))
    for asc in ascs:
        sub = [r for r in done if r["spec"]["ascension"] == asc]
        print(f"A{asc}: {len(sub)} runs, outcomes {dict(collections.Counter(r['result'] for r in sub))}, "
              f"max act reached {collections.Counter(max(r.get('acts') or [0]) for r in sub)}")
    bosses = collections.Counter(b for r in done for b in r.get("bosses", []))
    print("boss fights seen (act:enemies):", len(bosses))
    for b, n in sorted(bosses.items()):
        print(f"  {n:4d}  {b}")
    steps = sum(r.get("steps", 0) for r in done)
    print(f"total steps {steps}, wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
