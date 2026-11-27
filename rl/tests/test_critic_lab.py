"""Offline critic fitting and frozen-policy gradient measurements on FakeEngine."""
import json
import math

import numpy as np
import pytest
import torch

from rl.config import parse_args
from rl.model import Policy
from rl.ppo import ValueNorm
from tools import critic_lab as lab


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    directory = tmp_path_factory.mktemp("critic_lab")
    cfg = parse_args(["--preset", "tiny", "--n_workers", "1", "--envs_per_worker", "2",
                      "--gamma", "1", "--lam_build", "0.95", "--micro_batch", "32",
                      "--model.d", "16", "--model.heads", "2", "--model.ff_mult", "2",
                      "--caps.max_tok", "64", "--caps.max_ent", "12", "--caps.max_cand", "32"])
    torch.manual_seed(8)
    model, vnorm = Policy(cfg.model), ValueNorm()
    vnorm.update(np.array([0.25, 1.25]))
    checkpoint = directory / "behaviour.pt"
    torch.save(dict(model=model.state_dict(), vnorm=vnorm.state_dict(), cfg=cfg.to_dict(),
                    update=17, env_steps=9000), checkpoint)
    lab.main(["collect", "--ckpt", str(checkpoint), "--steps", "2048", "--engine", "fake",
              "--ascension", "10", "--out", str(directory), "--device", "cpu", "--restart_every_n", "24"])
    return directory, cfg


def test_collect_archive_and_stitch(dataset):
    directory, cfg = dataset
    chunks, meta = lab.load_dataset(directory)
    info = json.loads((directory / "meta.json").read_text())
    assert (directory / "meta.npz").exists()
    assert len(meta["val"]) == 2048 and len(chunks) == 4
    required = {"act", "logp", "val", "rew", "done", "trunc", "forced", "combat", "decision",
                "env", "episode", "chunk", "adv_prod", "ret_prod", "mask_prod"}
    assert required <= meta.keys()
    assert all(len(meta[key]) == 2048 for key in required)
    assert meta["boot"].shape == (4, cfg.n_envs)
    assert {"checkpoint", "update", "env_steps", "gamma", "lam", "lam_build", "rollout_steps",
            "micro_batch", "vnorm", "args", "engine_errors"} <= info.keys()
    assert info["update"] == 17 and info["env_steps"] == 9000
    assert info["vnorm"] == pytest.approx(dict(mean=0.75, std=0.5))
    for i, (start, stop, obs) in enumerate(chunks):
        assert stop - start == cfg.rollout_steps
        np.testing.assert_array_equal(obs["decision"].numpy(), meta["decision"][start:stop])
        np.testing.assert_array_equal(meta["chunk"][start:stop], i)
        assert obs["tok_type"].shape[1] == max(int(obs["n_tok"].max()), 1)
        assert obs["cand_key"].shape[1] == max(int(obs["n_cand"].max()), 1)
        for key in obs:
            assert (directory / f"chunk_{i}_{key}.npy").exists()
    stitched = lab.stitch(meta)
    assert stitched["complete"].any()
    nxt = stitched["next"]
    ix = np.flatnonzero(nxt >= 0)
    np.testing.assert_array_equal(meta["env"][ix], meta["env"][nxt[ix]])
    np.testing.assert_array_equal(meta["episode"][ix], meta["episode"][nxt[ix]])
    assert (nxt[ix] > ix).all()
    cross = ix[meta["chunk"][ix] != meta["chunk"][nxt[ix]]]
    assert len(cross) > 0
    assert (nxt[meta["done"] | meta["trunc"]] == -1).all()
    complete = stitched["complete"]
    np.testing.assert_allclose(stitched["G"][complete],
                               meta["rew"][complete] + cfg.gamma * np.where(
                                   meta["done"][complete], 0, stitched["G"][np.maximum(nxt[complete], 0)]))


@pytest.mark.parametrize("lam_build", [None, 0.95])
def test_gae_stitched_matches_production_within_rollout(dataset, lam_build):
    directory, cfg = dataset
    _, meta = lab.load_dataset(directory)
    stitched = lab.stitch(meta)
    adv, ret, mask = lab.gae_stitched(meta, stitched, meta["val"], cfg.lam, lam_build)
    internal = np.zeros(len(mask), bool)
    for episode in np.unique(meta["episode"][stitched["complete"]]):
        ix = np.flatnonzero(meta["episode"] == episode)
        if len(np.unique(meta["chunk"][ix])) == 1:
            internal[ix] = True
    assert internal.any()
    if lam_build is None:
        # Obtain the standard production GAE from the same archived rollouts.
        expected_adv, expected_ret = np.zeros_like(adv), np.zeros_like(ret)
        for i, (start, stop) in enumerate(meta["chunk_bounds"]):
            local = {key: value[start:stop] for key, value in meta.items() if key in
                     ("val", "rew", "done", "trunc", "forced", "combat", "env", "episode")}
            local["gamma"] = meta["gamma"]
            buf = lab._ValueRollout(local, lab.stitch(local), local["val"])
            a, r, _ = buf.gae(meta["boot"][i], cfg.gamma, cfg.lam)
            expected_adv[start:stop], expected_ret[start:stop] = a, r
    else:
        expected_adv, expected_ret = meta["adv_prod"], meta["ret_prod"]
    np.testing.assert_allclose(adv[internal], expected_adv[internal], rtol=0, atol=1e-6)
    np.testing.assert_allclose(ret[internal], expected_ret[internal], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(mask[internal], meta["mask_prod"][internal])
    assert not mask[~stitched["complete"]].any()
    assert not adv[~stitched["complete"]].any() and not ret[~stitched["complete"]].any()


@pytest.mark.parametrize("gamma", [1.0, 0.8])
def test_stitch_macro_steps_and_incomplete_episodes(gamma):
    # Env 0's complete run crosses the chunk boundary; env 1 aborts, then completes.
    meta = dict(env=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0]),
                episode=np.array([0, 1, 0, 1, 0, 3, 0, 3, 2]),
                chunk=np.array([0, 0, 0, 0, 1, 1, 1, 1, 1]), gamma=np.asarray(gamma),
                val=np.array([.2, .3, .4, .5, .6, .7, .8, .9, 1.], np.float32),
                rew=np.arange(1, 10, dtype=np.float32),
                done=np.array([0, 0, 0, 0, 0, 0, 1, 1, 0], bool),
                trunc=np.array([0, 0, 0, 1, 0, 0, 0, 0, 0], bool),
                forced=np.array([0, 0, 1, 0, 0, 0, 0, 0, 0], bool),
                combat=np.array([0, 1, 1, 0, 1, 0, 0, 1, 0], bool))
    stitched = lab.stitch(meta)
    np.testing.assert_array_equal(stitched["next"], [2, 3, 4, -1, 6, 7, -1, -1, -1])
    np.testing.assert_array_equal(stitched["complete"], [1, 0, 1, 0, 1, 1, 1, 1, 0])
    for episode, ix in ((0, [0, 2, 4, 6]), (3, [5, 7])):
        for t, i in enumerate(ix):
            expected = sum(gamma ** k * float(meta["rew"][j]) for k, j in enumerate(ix[t:]))
            assert stitched["G"][i] == pytest.approx(expected)
    a, r, mask = lab.gae_stitched(meta, stitched, meta["val"], 1, 1)
    np.testing.assert_allclose(r[stitched["complete"]], stitched["G"][stitched["complete"]], atol=2e-6)
    assert not mask[2] and a[2] != 0
    assert not mask[[1, 3, 8]].any() and not r[[1, 3, 8]].any()
    # Two build decisions enclose two combat steps, giving one discounted macro reward.
    a, _, _ = lab.gae_stitched(meta, stitched, meta["val"], .3, .6)
    macro_reward = meta["rew"][0] + gamma * meta["rew"][2] + gamma ** 2 * meta["rew"][4]
    expected = macro_reward + gamma ** 3 * meta["val"][6] - meta["val"][0]
    expected += gamma ** 3 * .6 * (meta["rew"][6] - meta["val"][6])
    assert a[0] == pytest.approx(expected, abs=2e-6)


@pytest.mark.parametrize("target", ["mc", "lam"])
def test_fit_one_epoch_and_persistent_episode_split(dataset, target):
    directory, cfg = dataset
    lab.main(["fit", "--data", str(directory), "--name", target, "--target", target,
              "--epochs", "1", "--minibatch", "128", "--device", "cpu"])
    output = directory / f"fit_{target}"
    values = np.load(output / "values.npy")
    metrics = json.loads((output / "metrics.json").read_text())
    assert values.shape == (2048,) and np.isfinite(values).all()
    assert (output / "model.pt").exists() and metrics["best_epoch"] == 1
    assert metrics["target"] == target
    assert metrics["vnorm"] == pytest.approx(dict(mean=.75, std=.5))
    assert len(metrics["epochs"]) == 1
    assert {"train_loss", "val_mse", "lr"} <= metrics["epochs"][0].keys()
    for name in ("behaviour", "fitted"):
        assert {"resid", "calibration", "td_mse"} <= metrics[name].keys()
        assert set(metrics[name]["resid"]) == {"all", "combat", "build", "early", "middle", "late"}
        assert len(metrics[name]["calibration"]) == 10
        assert set(metrics[name]["td_mse"]) == {"build→build", "build→combat", "combat→combat", "combat→build", "terminal"}
    with np.load(directory / "split.npz") as data:
        split = {key: data[key] for key in data.files}
    _, meta = lab.load_dataset(directory)
    stitched = lab.stitch(meta)
    sets = [set(split[key]) for key in ("train", "validation", "measure")]
    assert sets[0] and sets[1] and sets[2]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert set.union(*sets) == set(meta["episode"][stitched["complete"]])
    reused = lab.dataset_split(directory, meta, stitched, train_frac=.8, seed=999)
    for key in split:
        np.testing.assert_array_equal(reused[key], split[key])
    # The saved values really come from the selected model and fixed ValueNorm.
    model, vn, loaded_cfg, _ = lab.load_policy(directory / "behaviour.pt")
    model.load_state_dict(torch.load(output / "model.pt", weights_only=True))
    chunks, _ = lab.load_dataset(directory)
    expected = lab.predict_values(model, chunks, np.arange(2048), vn, loaded_cfg, torch.device("cpu"))
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-7)


def test_critic_metrics_known_values():
    meta = dict(env=np.zeros(4, int), episode=np.zeros(4, int), val=np.arange(4, dtype=np.float32),
                rew=np.array([1, 2, 3, 4], np.float32), done=np.array([0, 0, 0, 1], bool),
                trunc=np.zeros(4, bool), forced=np.zeros(4, bool), combat=np.array([0, 0, 1, 1], bool),
                gamma=np.asarray(1.0))
    stitched = lab.stitch(meta)
    metrics = lab.critic_metrics(meta, stitched, meta["val"], np.ones(4, bool))
    assert metrics["resid"]["all"] == pytest.approx(np.mean((np.array([10, 9, 7, 4]) - meta["val"]) ** 2) /
                                                           np.var([10, 9, 7, 4]))
    assert metrics["td_mse"] == {"build→build": 4, "build→combat": 9, "combat→combat": 16,
                                   "combat→build": None, "terminal": 1}
    assert sum(row["n"] for row in metrics["calibration"]) == 4


def test_alignment_formula_and_undefined_reference():
    a, b = {"table": torch.tensor([1., 2.]), "trunk": torch.tensor([3.])}, {
        "table": torch.tensor([2., 1.]), "trunk": torch.tensor([4.])}
    ra, rb = {"table": torch.tensor([.5, 1.]), "trunk": torch.tensor([2.])}, {
        "table": torch.tensor([1., 2.]), "trunk": torch.tensor([1.])}
    reference = lab.gradient_products(ra, rb)
    stats = lab.alignment_stats(a, b, ra, rb, reference)
    for group, names in (("all", ["table", "trunk"]), ("no_table", ["trunk"])):
        va, vb, vra, vrb = [torch.cat([gradient[name] for name in names]).double() for gradient in (a, b, ra, rb)]
        expected = .5 * (va.dot(vrb) / va.norm() + vb.dot(vra) / vb.norm()) / vra.dot(vrb).sqrt()
        assert stats[group]["align"] == pytest.approx(float(expected))
        assert stats[group]["self_cos"] == pytest.approx(float(va.dot(vb) / (va.norm() * vb.norm())))
    negative = {name: -value for name, value in rb.items()}
    stats = lab.alignment_stats(a, b, ra, negative, lab.gradient_products(ra, negative))
    assert all(row["align"] is None for row in stats.values())


def test_snr_behaviour_gain_and_shared_measure_pairs(dataset):
    directory, cfg = dataset
    output = directory / "snr.json"
    lab.main(["snr", "--data", str(directory), "--values", "behaviour", "--lams", str(cfg.lam),
              "--k", "256", "--pairs", "2", "--out", str(output), "--device", "cpu"])
    result = json.loads(output.read_text())
    production = result["estimators"][result["production"]]
    assert set(result["estimators"]) == {result["production"], "ref"}
    with np.load(directory / "split.npz") as split:
        measure = set(split["measure"])
    for group in ("all", "no_table"):
        assert production["summary"][group]["gain"] == 1
        assert {"mean", "se", "n_pairs"} <= production["summary"][group]["align"].keys()
    for pair, reference in zip(production["pairs"], result["estimators"]["ref"]["pairs"]):
        a, b = set(pair["episode_ids_a"]), set(pair["episode_ids_b"])
        assert a and b and not a & b and a | b <= measure
        assert pair["episode_ids_a"] == reference["episode_ids_a"]
        assert pair["episode_ids_b"] == reference["episode_ids_b"]
        for row in pair["groups"].values():
            assert {"dot", "norm_a", "norm_b", "dot_a_ref_b", "dot_b_ref_a", "ref_dot"} <= row.keys()
            if row["ref_dot"] <= 0:
                assert row["align"] is None
            else:
                expected = .5 * (row["dot_a_ref_b"] / row["norm_a"] + row["dot_b_ref_a"] / row["norm_b"])
                assert row["align"] == pytest.approx(expected / math.sqrt(row["ref_dot"]))


def test_snr_fitted_values_auto_production_and_reference(dataset):
    directory, cfg = dataset
    # Independent fit artifacts suffice here; no optimizer work is needed by snr.
    _, meta = lab.load_dataset(directory)
    output = directory / "fit_probe"
    output.mkdir()
    np.save(output / "values.npy", meta["val"] + .1)
    result = lab.main(["snr", "--data", str(directory), "--values", "fit_probe", "--lams", "0.5,1",
                       "--k", "256", "--pairs", "2", "--out", str(directory / "snr_fit.json"), "--device", "cpu"])
    assert result["production"] in result["estimators"]
    assert result["estimators"]["ref"]["value"] == "fit_probe"
    assert "fit_probe:1" in result["estimators"]
    for pair in range(2):
        rows = [e["pairs"][pair] for e in result["estimators"].values()]
        assert all(row["episode_ids_a"] == rows[0]["episode_ids_a"] for row in rows)
        assert all(row["episode_ids_b"] == rows[0]["episode_ids_b"] for row in rows)


@pytest.mark.parametrize("overrides", [dict(group_k=2), dict(branch_decisions="rest_site")])
def test_collect_rejects_replay_configs(tmp_path, monkeypatch, overrides):
    cfg = parse_args(["--preset", "tiny"])
    for key, value in overrides.items():
        setattr(cfg, key, value)
    monkeypatch.setattr(lab, "load_policy", lambda *args: (None, None, cfg, dict(update=0, env_steps=0)))
    with pytest.raises(ValueError, match="group_k=1"):
        lab.main(["collect", "--ckpt", "unused.pt", "--out", str(tmp_path), "--device", "cpu"])
