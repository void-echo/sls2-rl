"""Shared helpers for the engine test tools (fullrun.py, determinism.py, scenario_tests.py).

Engine: one engine process speaking the RL JSON-lines protocol, with a per-command timeout.
play_run: one run under a seeded random policy, returning outcome, an obs/cands hash chain and run statistics.
"""
import hashlib
import json
import os
import queue
import random
import subprocess
import sys
import threading

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from rl.protocol import check_response  # noqa: E402


class EngineDied(Exception):
    pass


class Timeout(Exception):
    pass


class Engine:
    def __init__(self, engine_dir=None, timeout=30.0, stderr_path=None):
        self.engine_dir = os.path.abspath(engine_dir or os.path.join(REPO, "engine"))
        self.timeout = timeout
        dotnet = os.path.expanduser("~/.dotnet/dotnet")
        dll = os.path.join(self.engine_dir, "src/Sts2Headless/bin/Debug/net9.0/Sts2Headless.dll")
        env = dict(os.environ, DOTNET_ROOT=os.path.dirname(dotnet), DOTNET_CLI_TELEMETRY_OPTOUT="1")
        err = open(stderr_path, "a") if stderr_path else subprocess.DEVNULL
        self.p = subprocess.Popen([dotnet, dll], cwd=self.engine_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=err, text=True, bufsize=1, env=env)
        self.q = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        r = self._read(120)
        if r.get("type") != "ready":
            raise EngineDied(f"engine did not start: {r}")

    def _reader(self):
        for line in self.p.stdout:
            if line.startswith("{"):
                try:
                    self.q.put(json.loads(line))
                except ValueError as e:
                    self.q.put({"type": "error", "message": f"bad json: {e}"})
        self.q.put(None)

    def _read(self, timeout):
        try:
            r = self.q.get(timeout=timeout)
        except queue.Empty:
            raise Timeout()
        if r is None:
            raise EngineDied()
        return r

    def send(self, cmd, timeout=None):
        try:
            self.p.stdin.write(json.dumps(cmd) + "\n")
            self.p.stdin.flush()
        except (BrokenPipeError, OSError):
            raise EngineDied()
        return self._read(timeout or self.timeout)

    def close(self):
        try:
            self.p.stdin.write('{"cmd":"quit"}\n')
            self.p.stdin.flush()
            self.p.wait(5)
        except Exception:
            self.p.kill()


def terminal_problem(r):
    """A run may only end as a death (hp 0), an abandon (info.abandoned) or a victory in the last act."""
    info = r.get("info") or {}
    if r["victory"] is False and info.get("hp", 0) > 0 and not info.get("abandoned"):
        return "fake_loss"
    if r["victory"] and r["obs"]["g"]["act"] < 3:
        return "fake_win"
    return None


def choose(rng, r):
    """Uniform random policy with light shaping so runs move on (end turn and potion discards down-weighted)."""
    w = []
    for c in r["cands"]:
        k = c["k"]
        w.append(0.05 if k == "discard_potion" else 0.5 if k == "end_turn" else 0.3 if k == "proceed" and r["decision"] == "rewards" else 1.0)
    return rng.choices(range(len(r["cands"])), weights=w)[0]


def play_run(eng, spec, max_steps=20000, stuck_limit=200, check=True):
    """spec: dict(character, ascension, seed, policy_seed, god=False). Returns a result dict."""
    rng = random.Random(spec["policy_seed"])
    h = hashlib.sha256()
    res = {"spec": spec, "steps": 0, "acts": set(), "bosses": set(), "decisions": {}}
    r = eng.send({"cmd": "rl_start", "character": spec["character"], "ascension": spec["ascension"], "seed": spec["seed"]}, 120)
    if spec.get("god") and r.get("type") == "rl":
        r = eng.send({"cmd": "set_player", "hp": 9999, "max_hp": 9999})
    last, repeat = None, 0
    while True:
        if r.get("type") != "rl":
            res.update(result="error", msg=(r.get("message") or str(r))[:1500])
            break
        if check:
            probs = check_response(r)
            if probs:
                res.update(result="schema", msg="; ".join(probs)[:600])
                break
        blob = json.dumps([r["decision"], r["obs"], r["cands"]], sort_keys=True).encode()
        h.update(blob)
        g = r["obs"]["g"]
        res["acts"].add(g["act"])
        if r["obs"].get("combat") is not None and g.get("room") == "Boss":
            res["bosses"].add(f"{g['act']}:" + ",".join(sorted(e["id"] for e in r["obs"]["combat"]["enemies"])))
        if r["done"]:
            bad = terminal_problem(r)
            res.update(result=bad or ("win" if r["victory"] else "abandon" if r["info"].get("abandoned") else "loss"),
                       floor=r["info"]["total_floor"], act=g["act"], info=r["info"])
            break
        if not r["cands"]:
            res.update(result="no_cands", msg=r["decision"])
            break
        d = hashlib.md5(blob).hexdigest()
        repeat = repeat + 1 if d == last else 0
        last = d
        if repeat >= stuck_limit:
            res.update(result="stuck", msg=f"{r['decision']} x{repeat}")
            break
        if res["steps"] >= max_steps:
            res.update(result="cap", msg=f"{max_steps} steps at {r['decision']} act {g['act']} floor {g['total_floor']}")
            break
        res["decisions"][r["decision"]] = res["decisions"].get(r["decision"], 0) + 1
        r = eng.send({"cmd": "rl_act", "i": choose(rng, r)})
        res["steps"] += 1
    res["hash"] = h.hexdigest()
    res["acts"] = sorted(res["acts"])
    res["bosses"] = sorted(res["bosses"])
    return res
