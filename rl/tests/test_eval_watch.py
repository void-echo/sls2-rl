import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import rl.eval_watch as watch
from rl.evaluate import summarize


@pytest.mark.parametrize('ascension,sample', [(None, False), (None, True), (0, False), (10, False), (10, True)])
def test_watch_ascension_outputs_and_archive_ownership(tmp_path, monkeypatch, ascension, sample):
    ckpt, arch = tmp_path / 'ckpt', tmp_path / 'eval_ckpts'
    ckpt.mkdir()
    arch.mkdir()
    (ckpt / 'ckpt_000200.pt').write_text('unarchived')
    (arch / 'ckpt_000100.pt').write_text('archived')
    (arch / 'ckpt_000300.pt.tmp').write_text('being archived')
    archived, evaluated = [], []
    def archive(src, dst):
        archived.append(Path(src).name)
        Path(dst).write_text('archived')
    def run(cmd, **kwargs):
        evaluated.append(cmd)
        out = Path(cmd[cmd.index('--out') + 1])
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(dict(summary=dict(all=summarize([
            dict(victory=True, floor=50, steps=100)])))))
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(watch, 'archive', archive)
    monkeypatch.setattr(watch, 'training_alive', lambda name: False)
    monkeypatch.setattr(watch.subprocess, 'run', run)
    args = ['--run_dir', str(tmp_path), '--engine_dir', 'unused']
    if ascension is not None:
        args += ['--ascension', str(ascension)]
    if sample:
        args += ['--sample']
    watch.main(args)
    stem = f'eval_a{ascension}' if ascension is not None else 'eval_sample' if sample else 'eval'
    with (tmp_path / f'{stem}.csv').open() as f:
        rows = list(csv.DictReader(f))
    expected = [100] if ascension is not None else [100, 200]
    assert [int(r['update']) for r in rows] == expected
    assert archived == ([] if ascension is not None else ['ckpt_000200.pt'])
    for cmd, update in zip(evaluated, expected):
        if ascension is None:
            assert '--ascension' not in cmd
        else:
            assert cmd[cmd.index('--ascension') + 1] == str(ascension)
        assert ('--greedy' in cmd) != sample
        folder = f'eval_a{ascension}' if ascension is not None else 'eval'
        suffix = '_sample' if sample and ascension is None else ''
        assert Path(cmd[cmd.index('--out') + 1]) == tmp_path / folder / f'eval_{update:06d}{suffix}.json'
    if ascension is not None:
        assert not (tmp_path / 'eval.csv').exists() and not (tmp_path / 'eval').exists()
        assert (ckpt / 'ckpt_000200.pt').read_text() == 'unarchived'
        assert not (arch / 'ckpt_000200.pt').exists()
    watch.main(args)
    assert len(evaluated) == len(expected)
