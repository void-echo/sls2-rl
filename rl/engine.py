"""Engine processes speaking the RL JSON-lines protocol.

Both classes share one non-blocking interface so a worker can drive many engines at once:
    start()        spawn (non-blocking); the first message poll() returns is {"type": "ready"}
    send(obj)      write one command
    poll()         return the next JSON message if one is available, else None
    fileno()       fd to select() on, or None when responses are available immediately
    pending_ok()   True if poll() can return something without waiting on the fd
    restart(), close()
plus blocking helpers `request()`, `reset()` and `step()` for scripts and evaluation tools.
Dead or misbehaving engines raise `EngineDead`.

Smoke-test a real engine (random policy, schema check, timings, encoder stats):
    python -m rl.engine --engine real --engine_dir ~/sls2-rl/engine --episodes 3
"""
import collections
import fcntl
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class EngineDead(RuntimeError):
    pass


def default_engine_cmd(engine_dir):
    dotnet = os.path.expanduser("~/.dotnet/dotnet")
    dll = os.path.join(engine_dir, "src", "Sts2Headless", "bin", "Debug", "net9.0", "Sts2Headless.dll")
    return [dotnet, dll]


class EngineProcess:
    """The real engine (or any JSON-lines program) as a subprocess."""

    def __init__(self, cmd=None, engine_dir=None, env=None, cwd=None, log_path=None, name="engine"):
        self.engine_dir = os.path.expanduser(engine_dir or os.path.join(REPO, "engine"))
        self.cmd = cmd or default_engine_cmd(self.engine_dir)
        self.env = dict(os.environ)
        self.env.setdefault("DOTNET_ROOT", os.path.expanduser("~/.dotnet"))
        # no diagnostic IPC: an engine killed on timeout would leave dotnet-diagnostic-* / clr-debug-pipe-* in /tmp
        self.env.setdefault("DOTNET_EnableDiagnostics", "0")
        self.env.update(env or {})
        self.cwd = cwd or self.engine_dir
        self.log_path = log_path
        self.name = name
        self.proc = None
        self.buf = b""
        self.lines = collections.deque()
        self.log_tail = collections.deque(maxlen=40)  # last non-JSON stdout lines, for error reports
        self.n_starts = 0

    # ------------------------------------------------------------ lifecycle
    def start(self):
        self.close()
        err = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                     cwd=self.cwd, env=self.env, bufsize=0, start_new_session=True)
        if self.log_path:
            err.close()
        fd = self.proc.stdout.fileno()
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)
        self.buf = b""
        self.lines.clear()
        self.n_starts += 1

    def restart(self):
        self.start()

    def close(self):
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            if p.poll() is None:
                try:
                    p.stdin.write(b'{"cmd":"quit"}\n')
                    p.stdin.flush()
                except OSError:
                    pass
                try:
                    p.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=5)
        except Exception:
            pass
        for f in (p.stdin, p.stdout):
            try:
                f.close()
            except Exception:
                pass

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    # ------------------------------------------------------------ io
    def fileno(self):
        return self.proc.stdout.fileno()

    def pending_ok(self):
        return bool(self.lines)

    def send(self, obj):
        if self.proc is None:
            raise EngineDead(f"{self.name}: not started")
        try:
            self.proc.stdin.write((json.dumps(obj, separators=(",", ":")) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise EngineDead(f"{self.name}: write failed ({e}); {self.diagnose()}")

    def _fill(self):
        while True:
            try:
                chunk = os.read(self.proc.stdout.fileno(), 1 << 20)
            except BlockingIOError:
                return
            if not chunk:
                if not self.lines:
                    raise EngineDead(f"{self.name}: stdout closed; {self.diagnose()}")
                return
            self.buf += chunk
            if b"\n" in chunk:
                *full, self.buf = self.buf.split(b"\n")
                for ln in full:
                    s = ln.strip()
                    if s.startswith(b"{"):
                        self.lines.append(s)
                    elif s:
                        self.log_tail.append(s.decode(errors="replace")[:300])

    def poll(self):
        if not self.lines:
            if self.proc is None:
                raise EngineDead(f"{self.name}: not started")
            self._fill()
        if self.lines:
            ln = self.lines.popleft()
            try:
                return json.loads(ln)
            except json.JSONDecodeError as e:
                raise EngineDead(f"{self.name}: bad JSON ({e}): {ln[:200]!r}")
        return None

    def recv(self, timeout=60.0):
        import select
        deadline = time.monotonic() + timeout
        while True:
            msg = self.poll()
            if msg is not None:
                return msg
            left = deadline - time.monotonic()
            if left <= 0:
                raise EngineDead(f"{self.name}: timeout after {timeout:.0f}s; {self.diagnose()}")
            select.select([self.fileno()], [], [], min(left, 1.0))

    def diagnose(self):
        rc = None if self.proc is None else self.proc.poll()
        tail = " | ".join(list(self.log_tail)[-5:])
        return f"returncode={rc} last log: {tail}"

    # ------------------------------------------------------------ blocking helpers
    def request(self, obj, timeout=60.0):
        self.send(obj)
        return self.recv(timeout)

    def wait_ready(self, timeout=180.0):
        msg = self.recv(timeout)
        if msg.get("type") != "ready":
            raise EngineDead(f"{self.name}: expected ready, got {str(msg)[:200]}")
        return msg

    def reset(self, character="Ironclad", ascension=0, seed="0", timeout=180.0):
        if not self.alive:
            self.start()
            self.wait_ready(timeout)
        return self.request({"cmd": "rl_start", "character": character, "ascension": ascension, "seed": str(seed)}, timeout)

    def step(self, i, timeout=60.0):
        return self.request({"cmd": "rl_act", "i": int(i)}, timeout)


class FakeEngine:
    """In-process FakeGame with the EngineProcess interface (responses are available immediately)."""

    def __init__(self, p_error=0.0, seed=None, name="fake", **game_kw):
        from .fake_engine import FakeServer
        self.server = FakeServer(p_error=p_error, seed=seed, **game_kw)
        self.name = name
        self.out = collections.deque()
        self.started = False
        self.n_starts = 0

    def start(self):
        self.out.clear()
        self.server.game, self.server.broken = None, False
        self.out.append({"type": "ready", "version": "fake-inproc"})
        self.started = True
        self.n_starts += 1

    restart = start

    def close(self):
        self.started = False

    @property
    def alive(self):
        return self.started

    def fileno(self):
        return None

    def pending_ok(self):
        return bool(self.out)

    def send(self, obj):
        if not self.started:
            raise EngineDead(f"{self.name}: not started")
        if obj.get("cmd") == "quit":
            self.started = False
            return
        self.out.append(self.server.handle(obj))

    def poll(self):
        if not self.started:
            raise EngineDead(f"{self.name}: not started")
        return self.out.popleft() if self.out else None

    def recv(self, timeout=60.0):
        msg = self.poll()
        if msg is None:
            raise EngineDead(f"{self.name}: no response pending")
        return msg

    def request(self, obj, timeout=60.0):
        self.send(obj)
        return self.recv(timeout)

    def wait_ready(self, timeout=60.0):
        return self.recv(timeout)

    def reset(self, character="Ironclad", ascension=0, seed="0", timeout=60.0):
        if not self.started:
            self.start()
            self.wait_ready()
        return self.request({"cmd": "rl_start", "character": character, "ascension": ascension, "seed": str(seed)})

    def step(self, i, timeout=60.0):
        return self.request({"cmd": "rl_act", "i": int(i)})


def make_engine(kind, engine_dir=None, engine_cmd=None, engine_env=None, log_path=None, name="engine",
                fake_p_error=0.0, fake_args=()):
    """kind: 'real' (sts2 engine), 'fake' (in-process), 'fakeproc' (fake engine as a subprocess)."""
    if kind == "fake":
        return FakeEngine(p_error=fake_p_error, name=name)
    if kind == "fakeproc":
        cmd = [sys.executable, "-m", "rl.fake_engine", "--p_error", str(fake_p_error), *fake_args]
        env = {"PYTHONPATH": REPO + os.pathsep + os.environ.get("PYTHONPATH", "")}
        return EngineProcess(cmd=cmd, cwd=REPO, env=env, log_path=log_path, name=name)
    if kind == "real":
        return EngineProcess(cmd=engine_cmd, engine_dir=engine_dir, env=engine_env, log_path=log_path, name=name)
    raise ValueError(f"unknown engine kind {kind!r}")


def _smoke(argv=None):
    """Random-policy smoke test: schema check, timings per decision kind, encoder caps usage."""
    import argparse
    import random
    import statistics
    from .encode import Encoder
    from .protocol import check_response

    ap = argparse.ArgumentParser(description=_smoke.__doc__)
    ap.add_argument("--engine", default="real", choices=("real", "fake", "fakeproc"))
    ap.add_argument("--engine_dir", default=None)
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--seed", default="smoke")
    ap.add_argument("--dump", default=None, help="write the first response of each decision kind to this dir")
    a = ap.parse_args(argv)
    eng = make_engine(a.engine, engine_dir=a.engine_dir)
    enc = Encoder()
    rng = random.Random(0)
    t0 = time.monotonic()
    eng.start()
    print("ready:", eng.wait_ready(300), f"({time.monotonic() - t0:.2f}s)")
    times = collections.defaultdict(list)
    enc_times, problems, seen = [], collections.Counter(), set()
    stats = collections.defaultdict(list)
    for ep in range(a.episodes):
        t = time.monotonic()
        R = eng.request({"cmd": "rl_start", "character": a.character, "ascension": a.ascension,
                         "seed": f"{a.seed}{ep}"}, timeout=300)
        times["rl_start"].append(time.monotonic() - t)
        n = 0
        while R.get("type") == "rl" and not R["done"] and n < a.max_steps:
            for p in check_response(R):
                problems[p] += 1
            if a.dump and R["decision"] not in seen:
                os.makedirs(a.dump, exist_ok=True)
                with open(os.path.join(a.dump, R["decision"] + ".json"), "w") as f:
                    json.dump(R, f, indent=1)
                seen.add(R["decision"])
            t = time.monotonic()
            arrs = enc.encode(R)
            enc_times.append(time.monotonic() - t)
            stats["n_tok"].append(int(arrs["n_tok"]))
            stats["n_cand"].append(len(R["cands"]))
            t = time.monotonic()
            dec = R["decision"]
            R = eng.step(rng.randrange(len(R["cands"])))
            times[dec].append(time.monotonic() - t)
            n += 1
        print(f"episode {ep}: steps={n} type={R.get('type')} done={R.get('done')} victory={R.get('victory')} "
              f"info={R.get('info')} {R.get('message', '')}")
    eng.close()
    print("\nper-command latency (ms): kind count median p95")
    for k, v in sorted(times.items(), key=lambda kv: -len(kv[1])):
        v = sorted(v)
        print(f"  {k:16s} {len(v):6d} {1000 * statistics.median(v):8.2f} {1000 * v[int(0.95 * (len(v) - 1))]:8.2f}")
    if enc_times:
        print(f"encode median {1000 * statistics.median(enc_times):.2f} ms; tokens median {statistics.median(stats['n_tok'])} "
              f"max {max(stats['n_tok'])}; cands max {max(stats['n_cand'])}")
    print("encoder truncations:", dict(enc.trunc) or "none")
    print("schema problems:", dict(problems) or "none")


if __name__ == "__main__":
    _smoke()
