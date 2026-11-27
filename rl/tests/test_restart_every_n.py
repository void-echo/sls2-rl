"""restart_every_n: with restart_every_episode, a fresh engine process only every n-th full game."""
from types import SimpleNamespace

from rl.actors import _Worker
from rl.config import Config


def _calls(n, games):
    w = _Worker.__new__(_Worker)
    w.cfg = Config(restart_every_episode=True, restart_every_n=n).to_dict()
    w.midrun_on = False
    w.runs_in_proc = {0: 0}
    calls = []
    w.eng = {0: SimpleNamespace(close=lambda: calls.append("close"))}
    w.boot = lambda e: (calls.append("boot"), w.runs_in_proc.__setitem__(e, 0))
    w.new_episode = lambda e: calls.append("new")
    for _ in range(games):
        w.next_run(0)
    return calls


def test_default_restarts_every_game():
    assert _calls(1, 3) == ["close", "boot"] * 3


def test_restart_every_third_game():
    assert _calls(3, 7) == ["new", "new", "close", "boot", "new", "new", "close", "boot", "new"]


def test_old_config_dicts_default_to_every_game():
    d = Config(restart_every_episode=True).to_dict()
    d.pop("restart_every_n")
    assert Config.from_dict(d).restart_every_n == 1
