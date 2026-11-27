#!/usr/bin/env python3
"""Throughput + determinism probe for the real StS2 engine running headless via sts2-cli.

One long-lived engine process plays many full runs with a uniformly random legal policy,
so process start-up is excluded from the per-step numbers. Running the same (game seed,
policy seed) twice and hashing every observation tells us whether the engine is
reproducible from seed + action list, which search and parity checking both rely on.

Usage: bench_realengine.py <sts2-cli dir> [--runs N] [--character C] [--ascension A]
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


class Engine:
    def __init__(self, cli_dir):
        cli_dir = os.path.abspath(cli_dir)
        dotnet = os.path.expanduser("~/.dotnet/dotnet")
        dll = os.path.join(cli_dir, "src/Sts2Headless/bin/Debug/net9.0/Sts2Headless.dll")
        cmd = [dotnet, dll] if os.path.exists(dll) else [
            dotnet, "run", "--no-build", "--project",
            os.path.join(cli_dir, "src/Sts2Headless/Sts2Headless.csproj")]
        env = dict(os.environ, DOTNET_ROOT=os.path.dirname(dotnet), DOTNET_CLI_TELEMETRY_OPTOUT="1")
        self.proc = subprocess.Popen(cmd, cwd=cli_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
        while True:
            msg = self._read()
            if msg.get("type") == "ready":
                break

    def _read(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)

    def send(self, cmd):
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()
        return self._read()

    def close(self):
        try:
            self.send({"cmd": "quit"})
        except Exception:
            pass
        self.proc.kill()


def legal_actions(state):
    """Enumerate the actions the protocol exposes at this decision point."""
    d = state.get("decision", "")
    acts = []
    if d == "map_select":
        acts = [("select_map_node", {"col": c["col"], "row": c["row"]}) for c in state.get("choices", [])]
    elif d == "combat_play":
        energy = state.get("energy", 0)
        n_enemies = len([e for e in state.get("enemies", []) if e.get("hp", 1) > 0])
        for c in state.get("hand", []):
            if not c.get("can_play") or c.get("cost", 0) > energy:
                continue
            if c.get("target_type") == "AnyEnemy" and n_enemies > 1:
                acts += [("play_card", {"card_index": c["index"], "target_index": t}) for t in range(n_enemies)]
            else:
                acts.append(("play_card", {"card_index": c["index"]}))
        for p in state.get("potions", []) or []:
            if p.get("can_use"):
                acts.append(("use_potion", {"potion_index": p["index"]}))
        acts.append(("end_turn", {}))
    elif d == "card_reward":
        acts = [("select_card_reward", {"card_index": c["index"]}) for c in state.get("cards", [])]
        acts.append(("skip_card_reward", {}))
    elif d == "event_choice":
        opts = [o for o in state.get("options", []) if not o.get("is_locked")]
        acts = [("choose_option", {"option_index": o["index"]}) for o in opts] or [("leave_room", {})]
    elif d == "rest_site":
        opts = [o for o in state.get("options", []) if o.get("is_enabled", True)]
        acts = [("choose_option", {"option_index": o["index"]}) for o in opts] or [("leave_room", {})]
    elif d == "bundle_select":
        acts = [("select_bundle", {"bundle_index": i}) for i in range(len(state.get("bundles", [])) or 1)]
    elif d == "card_select":
        cards = state.get("cards", [])
        acts = [("select_cards", {"indices": str(c.get("index", i))}) for i, c in enumerate(cards)]
        if not acts:
            acts = [("skip_select", {})]
    elif d == "shop":
        acts = [("leave_room", {})]
    else:
        acts = [("proceed", {})]
    return acts


def play(engine, seed, policy_seed, character, ascension, timing, max_steps=3000):
    rng = random.Random(policy_seed)
    h = hashlib.sha256()
    t0 = time.perf_counter()
    state = engine.send({"cmd": "start_run", "character": character, "seed": seed, "ascension": ascension})
    timing["start_run"].append(time.perf_counter() - t0)
    steps, errors = 0, 0
    while steps < max_steps:
        h.update(json.dumps(state, sort_keys=True).encode())
        if state.get("type") == "error":
            errors += 1
            if errors > 50:
                return {"result": "error", "steps": steps, "hash": h.hexdigest(), "msg": state.get("message")}
            # Fall back to a safe action after a rejected one.
            state = engine.send({"cmd": "action", "action": "end_turn"})
            continue
        d = state.get("decision", "")
        if d == "game_over":
            return {"result": "win" if state.get("victory") else "loss", "steps": steps,
                    "floor": state.get("floor"), "act": state.get("act"), "hash": h.hexdigest()}
        name, args = rng.choice(legal_actions(state))
        t0 = time.perf_counter()
        state = engine.send({"cmd": "action", "action": name, "args": args})
        timing[d].append(time.perf_counter() - t0)
        steps += 1
    return {"result": "timeout", "steps": steps, "hash": h.hexdigest()}


def one_run(job):
    """Worker entry for the parallel scaling test: one fresh engine process per run."""
    cli_dir, i, character, ascension = job
    timing = collections.defaultdict(list)
    eng = Engine(cli_dir)
    try:
        r = play(eng, f"bench_{i}", i, character, ascension, timing)
    finally:
        eng.close()
    r["step_time"] = sum(sum(v) for k, v in timing.items() if k != "start_run")
    r["start_time"] = sum(timing["start_run"])
    return r


def scaling(a):
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(a.cli_dir, i, a.character, a.ascension) for i in range(a.runs)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(one_run, jobs))
    wall = time.perf_counter() - t0
    steps = sum(r["steps"] for r in results)
    busy = sum(r["step_time"] for r in results)
    start = sum(r["start_time"] for r in results)
    print(f"workers={a.workers} runs={a.runs} steps={steps} wall={wall:.1f}s -> "
          f"{steps / wall:.0f} steps/s aggregate, {a.runs / wall:.2f} runs/s")
    print(f"  per-process step rate {steps / busy:.0f} steps/s; start_run avg {start / a.runs * 1000:.0f} ms; "
          f"share of time in start_run {start / (start + busy):.0%}")
    print("  outcomes:", collections.Counter(r["result"] for r in results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0, help="if >0, run the parallel scaling test")
    ap.add_argument("cli_dir")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--ascension", type=int, default=0)
    a = ap.parse_args()
    if a.workers > 0:
        return scaling(a)

    # The engine keeps global singletons alive between runs, so every run gets a fresh process.
    timing = collections.defaultdict(list)
    results, boot = [], []
    t0 = time.perf_counter()
    for i in range(a.runs):
        tb = time.perf_counter()
        eng = Engine(a.cli_dir)
        boot.append(time.perf_counter() - tb)
        results.append(play(eng, f"bench_{i}", i, a.character, a.ascension, timing))
        eng.close()
    wall = time.perf_counter() - t0
    t_start, t_ready = 0.0, sum(boot) / len(boot)

    # Determinism: replay the first few (seed, policy) pairs in fresh processes.
    fresh_proc = []
    for i in range(min(3, a.runs)):
        eng = Engine(a.cli_dir)
        fresh_proc.append(play(eng, f"bench_{i}", i, a.character, a.ascension, collections.defaultdict(list))["hash"])
        eng.close()
    same_proc = fresh_proc

    total_steps = sum(r["steps"] for r in results)
    print(f"process boot (until ready): {t_ready - t_start:.2f}s avg")
    print(f"runs: {a.runs}  steps: {total_steps}  wall: {wall:.2f}s  "
          f"-> {total_steps / wall:.1f} steps/s, {a.runs / wall:.2f} runs/s (incl. process boot + IPC)")
    print("outcomes:", collections.Counter(r["result"] for r in results))
    print("floors reached:", sorted(r.get("floor") or 0 for r in results))
    print("per-decision latency (ms): mean / p50 / p99 / n")
    for d, ts in sorted(timing.items()):
        ts = sorted(ts)
        print(f"  {d:14s} {1000 * sum(ts) / len(ts):7.2f} {1000 * ts[len(ts) // 2]:7.2f} "
              f"{1000 * ts[int(len(ts) * 0.99) - 1]:8.2f} {len(ts):6d}")
    orig = [r["hash"] for r in results[:len(same_proc)]]
    print("deterministic across processes (same seed + same actions):", orig == fresh_proc)
    step_time = sum(sum(v) for k, v in timing.items() if k != "start_run")
    n_steps = sum(len(v) for k, v in timing.items() if k != "start_run")
    print(f"pure step throughput (excl. boot/start_run): {n_steps / step_time:.1f} steps/s")
    errs = [r for r in results if r["result"] in ("error", "timeout")]
    for r in errs[:5]:
        print("  problem run:", r)


if __name__ == "__main__":
    main()
