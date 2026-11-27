#!/usr/bin/env python3
"""Smoke test for the engine's RL protocol: random legal policy, many runs in ONE process.

Checks that rl_start resets cleanly between runs, that every decision offers candidates, that every response passes
rl/protocol.py's schema check, that terminals are genuine (a loss has hp 0 unless the run was abandoned, a victory
happens in the last act), and that replays are deterministic. Reports throughput, rl_start time and the per-decision
latency (time of the rl_act that answers a decision of that kind).
Usage: rl_smoke.py <engine dir> [--runs N] [--character C] [--ascension A] [--seed-prefix P] [--replays K]
"""
import argparse
import collections
import hashlib
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.protocol import check_response  # noqa: E402


def launch(engine_dir):
    engine_dir = os.path.abspath(engine_dir)
    dotnet = os.path.expanduser("~/.dotnet/dotnet")
    dll = os.path.join(engine_dir, "src/Sts2Headless/bin/Debug/net9.0/Sts2Headless.dll")
    env = dict(os.environ, DOTNET_ROOT=os.path.dirname(dotnet), DOTNET_CLI_TELEMETRY_OPTOUT="1")
    p = subprocess.Popen([dotnet, dll], cwd=engine_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)

    def read():
        while True:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            if line.startswith("{"):
                return json.loads(line)

    def send(cmd):
        p.stdin.write(json.dumps(cmd) + "\n")
        p.stdin.flush()
        return read()

    assert read()["type"] == "ready"
    return p, send


def terminal_problem(r):
    info = r.get("info") or {}
    if not r["victory"] and info.get("hp", 0) > 0 and not info.get("abandoned"):
        return "fake_loss"
    if r["victory"] and r["obs"]["g"]["act"] < 3:
        return "fake_win"
    return None


def play(send, seed, policy_seed, character, ascension, stats, max_steps=6000, check=True):
    rng = random.Random(policy_seed)
    h = hashlib.sha256()
    t0 = time.perf_counter()
    r = send({"cmd": "rl_start", "character": character, "ascension": ascension, "seed": seed})
    stats["start_time"].append(time.perf_counter() - t0)
    steps = 0
    while steps < max_steps:
        if r.get("type") != "rl":
            return {"result": "error", "steps": steps, "msg": r.get("message", "")[:400], "hash": h.hexdigest(), "seed": seed}
        if check:
            probs = check_response(r)
            if probs:
                return {"result": "schema", "steps": steps, "msg": "; ".join(probs)[:400], "hash": h.hexdigest(), "seed": seed}
        h.update(json.dumps(r["obs"], sort_keys=True).encode())
        if r["done"]:
            bad = terminal_problem(r)
            if bad:
                return {"result": bad, "steps": steps, "hash": h.hexdigest(), "seed": seed, "info": r["info"]}
            res = "win" if r["victory"] else ("abandon" if r["info"].get("abandoned") else "loss")
            return {"result": res, "steps": steps, "floor": r["info"]["total_floor"], "hash": h.hexdigest()}
        cands = r["cands"]
        if not cands:
            return {"result": "no_cands", "steps": steps, "decision": r["decision"], "hash": h.hexdigest(), "seed": seed}
        stats["decision"][r["decision"]] += 1
        stats["ncands"].append(len(cands))
        k = rng.randrange(len(cands))
        stats["kind"][cands[k]["k"]] += 1
        dec = r["decision"]
        t0 = time.perf_counter()
        r = send({"cmd": "rl_act", "i": k})
        dt = time.perf_counter() - t0
        stats["step_time"].append(dt)
        stats["lat_" + dec].append(dt)
        steps += 1
    return {"result": "timeout", "steps": steps, "hash": h.hexdigest(), "seed": seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine_dir")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--seed-prefix", default="smoke")
    ap.add_argument("--replays", type=int, default=3)
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()

    def new_stats():
        st = collections.defaultdict(list)
        st["decision"], st["kind"] = collections.Counter(), collections.Counter()
        return st

    stats = new_stats()
    proc, send = launch(a.engine_dir)
    results = [play(send, f"{a.seed_prefix}_{i}", i, a.character, a.ascension, stats, check=not a.no_check) for i in range(a.runs)]
    # Replay the first runs in the same process to check reset + determinism.
    replays = [play(send, f"{a.seed_prefix}_{i}", i, a.character, a.ascension, new_stats(), check=False)
               for i in range(min(a.replays, a.runs))]
    proc.kill()

    st = stats["step_time"]
    outcomes = collections.Counter(r["result"] for r in results)
    print("outcomes:", dict(outcomes))
    print("errors:", sum(v for k, v in outcomes.items() if k not in ("win", "loss", "abandon")))
    print("final floors:", sorted(r.get("floor", -1) for r in results))
    print(f"steps: {len(st)}  pure step rate: {len(st) / max(sum(st), 1e-9):.0f}/s  "
          f"start_run avg: {1000 * sum(stats['start_time']) / len(stats['start_time']):.0f} ms "
          f"(first {1000 * stats['start_time'][0]:.0f} ms)")
    print("decisions:", dict(stats["decision"]))
    print("chosen kinds:", dict(stats["kind"]))
    lat = {k[4:]: v for k, v in stats.items() if k.startswith("lat_")}
    print("latency per decision kind (ms, mean / p95, n):")
    for k in sorted(lat, key=lambda k: -len(lat[k])):
        v = sorted(lat[k])
        print(f"  {k:14s} {1000 * sum(v) / len(v):6.2f} / {1000 * v[int(0.95 * (len(v) - 1))]:6.2f}  n={len(v)}")
    nc = sorted(stats["ncands"])
    print(f"candidates per decision: median {nc[len(nc) // 2]}, max {nc[-1]}")
    print("replay deterministic in same process:", [r["hash"] for r in results[:len(replays)]] == [r["hash"] for r in replays])
    for r in results:
        if r["result"] not in ("win", "loss", "abandon"):
            print("  problem:", r)


if __name__ == "__main__":
    main()
