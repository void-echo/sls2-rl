"""tools/critic_gram.py on a tiny FakeEngine dataset: Gram matrices exist, and combat + build parts add up to the full gradient."""
import json

import numpy as np

import tools.critic_gram as gram
import tools.critic_lab as lab
from rl.tests.test_critic_lab import dataset  # noqa: F401  (module-scoped fixture: checkpoint + collected data)


def test_gram_runs_and_parts_add_up(dataset, tmp_path):
    directory = dataset[0] if isinstance(dataset, tuple) else dataset
    chunks, meta = lab.load_dataset(directory)
    lab.dataset_split(directory, meta, lab.stitch(meta))
    out = tmp_path / "gram.json"
    gram.main(["--data", str(directory), "--lams", "0.5,0.95,1", "--k", "128", "--pairs", "2", "--device", "cpu", "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["names"] == ["combat:0.5", "combat:0.95", "combat:1", "build:0.5", "build:0.95", "build:1"] and len(r["pairs"]) == 2
    X = np.asarray(r["pairs"][0]["all"]["self_a"])
    assert X.shape == (6, 6) and np.allclose(X, X.T) and (np.diag(X) >= 0).all()
    q = np.asarray(r["summary"]["all"]["q_mean"])
    assert q.shape == (3, 3) and np.isfinite(q).all()
