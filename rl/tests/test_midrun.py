import csv

from rl.config import Config
from rl.train import AscCurriculum, train


def test_curriculum_ignores_midrun_episodes():
    c = AscCurriculum(Config(asc_curriculum=True, asc_window=2, asc_threshold=.5, asc_max=3), 1)
    assert not c.record(dict(asc=1, victory=True, midrun=True))
    assert c.windows == {}


def test_midrun_starts_load_saves_and_stay_out_of_win_statistics(tmp_path):
    from rl.model import ModelCfg
    cfg = Config(name="midrun", run_dir=str(tmp_path), device="cpu", engine="fakeproc", n_workers=1, envs_per_worker=4,
                 rollout_steps=2048, epochs=1, minibatch=512, micro_batch=512, ckpt_every=100, total_updates=4,
                 restart_every_episode=True, midrun_frac=.7, midrun_rooms="Monster,Elite,Boss", midrun_min_floor=3,
                 midrun_reuse=2, model=ModelCfg(d=32, layers=1, heads=2, ff_mult=2, cand_layers=1))
    train(cfg)
    rows = list(csv.DictReader(open(tmp_path / "episodes.csv")))
    mid = [r for r in rows if r.get("midrun") == "True"]
    full = [r for r in rows if r.get("midrun") != "True"]
    assert mid and full
    assert all(int(r["start_floor"]) >= 3 for r in mid)
    assert not any(r["aborted"] for r in mid), [r["aborted"] for r in mid][:3]
    prog = list(csv.DictReader(open(tmp_path / "progress.csv")))
    assert sum(int(r.get("midrun_eps") or 0) for r in prog) == len(mid)
    assert sum(int(r["u_episodes"] or 0) for r in prog) == len(full)


def test_evaluation_never_uses_midrun():
    from rl.actors import _Worker
    w = _Worker.__new__(_Worker)
    w.midrun_on, w.midrun_saves, w.midrun_next = False, [{"data": "x"}], {}
    assert w.midrun_choice(0) is False
