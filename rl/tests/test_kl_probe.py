"""kl_probe / lr_kl_signal step: exact KL between the policy before and after an update, as the controller's input."""
import csv

import pytest

import rl.train as tr
from rl.config import Config, parse_args


def _rows(path):
    return list(csv.DictReader(open(path / "progress.csv")))


@pytest.mark.parametrize("extra", [[], ["--async_update", "true"]])
def test_step_kl_logged_and_controls_lr(tmp_path, extra):
    args = ["--preset", "tiny", "--run_dir", str(tmp_path), "--kl_probe", "128", "--lr_kl_signal", "step",
            "--lr_kl_target", "1e-7", "--lr", "1e-3", "--total_updates", "4"] + extra
    tr.train(parse_args(args))
    rows = _rows(tmp_path)
    kl = [float(r["step_kl"]) for r in rows]
    lr = [float(r["lr"]) for r in rows]
    assert all(k > 0 for k in kl) and lr[1] < lr[0] and lr[2] < lr[1]  # step KL far above the tiny target: lr falls
    if extra:
        assert all(float(r["kl_ctrl"]) == pytest.approx(float(r["step_kl"]), rel=1e-3) for r in rows)


def test_zero_lr_gives_zero_step_kl(tmp_path):
    tr.train(parse_args(["--preset", "tiny", "--run_dir", str(tmp_path), "--kl_probe", "128", "--lr", "0", "--total_updates", "2"]))
    assert all(abs(float(r["step_kl"])) < 1e-9 for r in _rows(tmp_path))


def test_step_signal_needs_probe():
    with pytest.raises(ValueError):
        Config(lr_kl_signal="step")
    assert Config().kl_probe == 0 and Config().lr_kl_signal == "approx"
