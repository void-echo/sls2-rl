"""lam_pi: the policy's advantages use their own lambda, value targets keep lam / lam_build."""
import numpy as np
import pytest

import rl.train as tr
from rl.config import Config, parse_args
from rl.encode import Caps
from rl.ppo import Rollout


def _buf():
    rng = np.random.default_rng(0)
    buf = Rollout(Caps(), 64, 2, "cpu", track_combat=True)
    for e in range(2):
        buf.env_idx[e] = list(range(e * 32, e * 32 + 32))
    buf.n = 64
    buf.val[:] = rng.normal(size=64)
    buf.rew[:] = rng.normal(size=64) * 0.1
    buf.done[:] = False
    buf.done[20] = True
    buf.trunc[:] = False
    buf.combat[:] = rng.random(64) < 0.7
    return buf, rng.normal(size=2).astype(np.float32)


@pytest.mark.parametrize("lam_build", [None, 0.95])
def test_policy_advantages_use_lam_pi_and_default_is_untouched(lam_build):
    buf, boot = _buf()
    cfg = Config(gamma=1.0, lam=0.95, lam_build=lam_build)
    base = buf.gae(boot, 1.0, 0.95)[0] if lam_build is None else buf.gae_build(boot, 1.0, 0.95, lam_build)[0]
    same, stats = tr.policy_advantages(buf, boot, base, cfg)
    assert same is base and stats == {}
    cfg = Config(gamma=1.0, lam=0.95, lam_build=lam_build, lam_pi=0.5)
    low, stats = tr.policy_advantages(buf, boot, base, cfg)
    want = buf.gae(boot, 1.0, 0.5)[0] if lam_build is None else buf.gae_build(boot, 1.0, 0.5, 0.5)[0]
    assert np.allclose(low, want) and not np.allclose(low, base) and "adv_std_pi" in stats
    cfg = Config(gamma=1.0, lam=0.95, lam_build=lam_build, lam_pi=0.5, lam_pi_build=0.8)
    assert np.allclose(tr.policy_advantages(buf, boot, base, cfg)[0], buf.gae_build(boot, 1.0, 0.5, 0.8)[0])


@pytest.mark.parametrize("extra", [[], ["--async_update", "true"], ["--lam_build", "0.95"]])
def test_training_runs_with_lam_pi(tmp_path, extra):
    st = tr.train(parse_args(["--preset", "tiny", "--run_dir", str(tmp_path), "--lam_pi", "0.5", "--total_updates", "3"] + extra))
    assert st["update"] == 3
