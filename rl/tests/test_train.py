"""End-to-end smoke tests of the training loop (CPU, tiny model, FakeEngine). Run from the repo root."""
import csv
import os

from rl.config import parse_args
from rl.train import train


def _rows(run_dir):
    with open(os.path.join(run_dir, "progress.csv")) as f:
        return list(csv.DictReader(f))


def test_train_inproc_with_errors_and_resume(tmp_path):
    rd = str(tmp_path / "run")
    args = ["--preset", "tiny", "--run_dir", rd, "--fake_p_error", "0.003", "--ckpt_every", "2"]
    st = train(parse_args(args + ["--total_updates", "4"]))
    assert st["update"] == 4
    rows = _rows(rd)
    vl = [float(r["v_loss"]) for r in rows]
    assert len(rows) == 4 and min(vl[1:]) < vl[0]  # value loss falls after the first update
    assert os.path.exists(os.path.join(rd, "ckpt", "latest.pt"))
    st = train(parse_args(args + ["--total_updates", "6", "--resume", "auto"]))
    assert st["update"] == 6 and len(_rows(rd)) == 6


def test_train_subprocess_engines_with_crashes(tmp_path):
    rd = str(tmp_path / "run")
    cfg = parse_args(["--preset", "tiny", "--run_dir", rd, "--engine", "fakeproc", "--total_updates", "2",
                      "--fake_args=--p_crash 0.002 --step_ms 0.5", "--restart_every_episode", "true"])
    st = train(cfg)
    assert st["update"] == 2
