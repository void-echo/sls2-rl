"""log_actions: evaluation keeps each game's action indices, training writes them to actions/*.jsonl.gz, and replaying
the indices on the same seed reproduces the game (FakeEngine)."""
import csv
import glob
import gzip
import json

import rl.train as tr
from rl.config import parse_args
from rl.evaluate import run_eval
from rl.engine import make_engine


def _replay(char, seed, asc, actions):
    eng = make_engine("fake")
    eng.start()
    assert eng.recv(timeout=10)["type"] == "ready"
    r = eng.request({"cmd": "rl_start", "character": char, "ascension": asc, "seed": seed})
    for a in actions:
        assert not r.get("done") and a < len(r["cands"])
        r = eng.request({"cmd": "rl_act", "i": a})
    return r


def test_training_writes_action_log_and_it_replays(tmp_path):
    cfg = parse_args(["--preset", "tiny", "--run_dir", str(tmp_path), "--log_actions", "true", "--total_updates", "3"])
    tr.train(cfg)
    files = sorted(glob.glob(str(tmp_path / "actions" / "*.jsonl.gz")))
    rows = [json.loads(l) for f in files for l in gzip.open(f, "rt")]
    assert rows and all(len(r["a"]) > 0 for r in rows)
    assert "actions" not in next(csv.DictReader(open(tmp_path / "episodes.csv")))
    r = rows[0]
    end = _replay(r["char"], r["seed"], r["asc"], r["a"])
    assert end.get("done") and bool(end["info"].get("victory")) == r["victory"]


def test_default_training_has_no_action_log(tmp_path):
    tr.train(parse_args(["--preset", "tiny", "--run_dir", str(tmp_path), "--total_updates", "1"]))
    assert not (tmp_path / "actions").exists()


def test_evaluation_always_keeps_actions(tmp_path):
    cfg = parse_args(["--preset", "tiny", "--run_dir", str(tmp_path)])
    eps = run_eval(cfg, None, [("Ironclad", f"t_{i}") for i in range(4)], random_policy=True, progress=False)
    assert len(eps) == 4 and all(e["actions"] and len(e["actions"]) == e["steps"] for e in eps)
