"""Thin client for the headless engine's RL protocol (engine/RL_PROTOCOL.md).

The parity tools never talk to `engine/` directly: they run a private copy made with
`rsync -a engine/ tmp/parity_engine_v2/` (then rebuilt there, with the source commit in its ENGINE_COMMIT) so edits to
the live engine (or a training run using it) cannot change a replay mid-way. Older copies: `tmp/parity_engine_v1b/`
(engine 3405f5c, batch v1b) and `tmp/parity_engine_v1/` (before the navigation-cycle rules, the first v1
re-validation).
"""
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENGINE = os.path.join(ROOT, "tmp", "parity_engine_v2")


class Engine:
    def __init__(self, engine_dir=DEFAULT_ENGINE, log_path=None):
        engine_dir = os.path.abspath(engine_dir)
        dotnet = os.path.expanduser("~/.dotnet/dotnet")
        dll = os.path.join(engine_dir, "src/Sts2Headless/bin/Debug/net9.0/Sts2Headless.dll")
        if not os.path.exists(dll):
            raise FileNotFoundError(f"{dll} missing; run: rsync -a engine/ {os.path.relpath(engine_dir, ROOT)}/")
        env = dict(os.environ, DOTNET_ROOT=os.path.dirname(dotnet), DOTNET_CLI_TELEMETRY_OPTOUT="1")
        self._log = open(log_path, "w") if log_path else subprocess.DEVNULL
        self.p = subprocess.Popen([dotnet, dll], cwd=engine_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self._log, text=True, bufsize=1, env=env)
        first = self._read()
        if first.get("type") != "ready":
            raise RuntimeError(f"engine did not start: {first}")

    def _read(self):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            if line.startswith("{"):
                return json.loads(line)

    def send(self, cmd):
        self.p.stdin.write(json.dumps(cmd) + "\n")
        self.p.stdin.flush()
        return self._read()

    def start(self, character, ascension, seed):
        return self.send({"cmd": "rl_start", "character": character, "ascension": ascension, "seed": seed})

    def act(self, i):
        return self.send({"cmd": "rl_act", "i": i})

    def close(self):
        try:
            self.p.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.p.stdin.flush()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()
        if self._log is not subprocess.DEVNULL:
            self._log.close()
