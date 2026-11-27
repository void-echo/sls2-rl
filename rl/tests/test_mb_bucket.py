"""mb_bucket: 8 sorts the whole rollout into token-count strata when 8 minibatches cover it; 1 gives i.i.d. minibatches."""
import numpy as np

from rl.ppo import bucketed_minibatches


def _spread(bucket):
    rng = np.random.default_rng(0)
    n_tok = rng.integers(20, 190, 131072)
    mbs = bucketed_minibatches(n_tok, np.arange(131072), 16384, np.random.default_rng(1), bucket)
    assert sorted(np.concatenate(mbs).tolist()) == list(range(131072)) and len(mbs) == 8
    return [float(n_tok[mb].mean()) for mb in mbs], mbs, n_tok


def test_default_bucket_stratifies_the_whole_rollout():
    means, _, _ = _spread(8)
    assert max(means) - min(means) > 100


def test_bucket_one_is_iid_and_sorted_inside():
    means, mbs, n_tok = _spread(1)
    assert max(means) - min(means) < 3
    assert all(np.all(np.diff(n_tok[mb]) >= 0) for mb in mbs)  # micro-batches still see little padding
