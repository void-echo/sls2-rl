"""A full disk must not kill a run, damage what is on disk, or lose rows; writing resumes when there is room."""
import csv
import errno
import os
import sys

import numpy as np
import pytest
import torch

import rl.diskfull as df
import rl.eval_watch as watch
import rl.train as tr
from rl.config import Config
from rl.ppo import ValueNorm


def full(*args, **kwargs):
    raise OSError(errno.ENOSPC, "No space left on device")


def test_append_all_cuts_off_a_partial_write(tmp_path, monkeypatch):
    p = tmp_path / "a.csv"
    p.write_bytes(b"h\r\n1\r\n")
    real = os.write
    calls = []
    def short_then_full(fd, data):
        calls.append(len(data))
        if len(calls) == 1:
            return real(fd, bytes(data[:3]))        # the disk takes three bytes of the row ...
        full()                                      # ... and then nothing
    monkeypatch.setattr(df.os, "write", short_then_full)
    with pytest.raises(OSError):
        df.append_all(str(p), b"22222\r\n")
    monkeypatch.undo()
    assert p.read_bytes() == b"h\r\n1\r\n"
    df.append_all(str(p), b"22222\r\n")
    assert p.read_bytes() == b"h\r\n1\r\n22222\r\n"


def test_write_atomic_leaves_the_old_file(tmp_path, monkeypatch):
    p = tmp_path / "x.pt"
    p.write_bytes(b"old")
    monkeypatch.setattr(df.os, "fsync", full)
    with pytest.raises(OSError):
        df.write_atomic(str(p), b"new")
    assert p.read_bytes() == b"old" and not (tmp_path / "x.pt.tmp").exists()


def test_csvlog_keeps_rows_while_the_disk_is_full(tmp_path, monkeypatch, capsys):
    p = tmp_path / "progress.csv"
    log = tr.CSVLog(p, expand=True)
    log.write(dict(update=1, win=0.5))
    with monkeypatch.context() as m:
        m.setattr(tr, "append_all", full)
        m.setattr(tr, "write_atomic", full)
        log.write(dict(update=2, win=0.6))
        log.write_many([dict(update=3, win=0.7, force_win=0.2), dict(update=4, win=0.8)])   # a new column while full
        assert len(log.pending) == 3
    assert [r["update"] for r in csv.DictReader(p.open())] == ["1"]
    log.write(dict(update=5, win=0.9))
    rows = list(csv.DictReader(p.open()))
    assert [r["update"] for r in rows] == ["1", "2", "3", "4", "5"]
    assert rows[2]["force_win"] == "0.2" and rows[0]["force_win"] == "" and not log.pending
    out = capsys.readouterr().out
    assert out.count("not written") == 1 and out.count("writing again") == 1


def test_csvlog_bounds_what_it_keeps(tmp_path, monkeypatch):
    log = tr.CSVLog(tmp_path / "episodes.csv")
    monkeypatch.setattr(tr.CSVLog, "MAX_PENDING", 3)
    monkeypatch.setattr(tr, "append_all", full)
    for u in range(5):
        log.write(dict(update=u))
    assert [r["update"] for r in log.pending] == [2, 3, 4] and log.dropped == 2


def ckpt_parts():
    model = torch.nn.Linear(2, 1)
    return model, torch.optim.Adam(model.parameters()), ValueNorm(), Config()


def test_save_ckpt_returns_false_and_keeps_the_last_good_one(tmp_path, monkeypatch, capsys):
    model, opt, vnorm, cfg = ckpt_parts()
    d = str(tmp_path / "ckpt")
    rng = np.random.default_rng(0)
    first = tr.save_ckpt(d, model, opt, vnorm, cfg, dict(update=25, env_steps=100), 2, rng)
    assert first and os.path.realpath(os.path.join(d, "latest.pt")) == os.path.realpath(first)
    with monkeypatch.context() as m:
        m.setattr(tr, "write_atomic", full)
        assert tr.save_ckpt(d, model, opt, vnorm, cfg, dict(update=50, env_steps=200), 2, rng) is False
    assert sorted(os.listdir(d)) == ["ckpt_000025.pt", "latest.pt"]
    assert torch.load(os.path.join(d, "latest.pt"), weights_only=False)["update"] == 25
    assert "not written" in capsys.readouterr().out
    second = tr.save_ckpt(d, model, opt, vnorm, cfg, dict(update=51, env_steps=204), 2, rng)
    third = tr.save_ckpt(d, model, opt, vnorm, cfg, dict(update=75, env_steps=300), 2, rng)
    assert second and third and sorted(os.listdir(d)) == ["ckpt_000051.pt", "ckpt_000075.pt", "latest.pt"]
    assert torch.load(os.path.join(d, "latest.pt"), weights_only=False)["update"] == 75


def test_save_ckpt_keeps_latest_when_the_link_cannot_be_made(tmp_path, monkeypatch):
    model, opt, vnorm, cfg = ckpt_parts()
    d = str(tmp_path / "ckpt")
    tr.save_ckpt(d, model, opt, vnorm, cfg, dict(update=25, env_steps=100), 5)
    with monkeypatch.context() as m:
        m.setattr(tr.os, "symlink", full)
        assert tr.save_ckpt(d, model, opt, vnorm, cfg, dict(update=50, env_steps=200), 5) is False
    assert os.readlink(os.path.join(d, "latest.pt")) == "ckpt_000025.pt"


def test_quiet_stdio_drops_a_failed_print(monkeypatch):
    class Full:
        def write(self, text):
            full()
        def flush(self):
            full()
    monkeypatch.setattr(sys, "stdout", Full())
    monkeypatch.setattr(sys, "stderr", Full())
    df.quiet_stdio()
    print("still running", flush=True)
    print("still running", file=sys.stderr, flush=True)


def test_archive_failure_leaves_nothing_behind(tmp_path, monkeypatch):
    src, dst = tmp_path / "ckpt_000100.pt", tmp_path / "arch.pt"
    torch.save(dict(model={}, cfg={}, vnorm={}, update=100, env_steps=1, opt={}), src)
    monkeypatch.setattr(df.os, "fsync", full)
    with pytest.raises(OSError):
        watch.archive(str(src), str(dst))
    assert sorted(f.name for f in tmp_path.iterdir()) == ["ckpt_000100.pt"]
    monkeypatch.undo()
    watch.archive(str(src), str(dst))
    assert "opt" not in torch.load(dst, weights_only=False)


def test_watcher_reads_a_full_disk_from_the_evaluation_output():
    assert watch.disk_full("OSError: [Errno 28] No space left on device: 'eval_000100.json'")
    assert not watch.disk_full("RuntimeError: CUDA out of memory")
