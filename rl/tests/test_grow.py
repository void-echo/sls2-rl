import copy
import csv
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

import rl.train as tr
from rl.config import Config, parse_args
from rl.encode import Encoder
from rl.engine import FakeEngine
from rl.model import ModelCfg, Policy
from rl.ppo import ValueNorm, sliced, to_torch


def _equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _equal(a, b)
    else:
        assert actual == expected


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("grow")
    torch.manual_seed(7)
    cfg = Config(device="cpu", engine="fake", model=ModelCfg(d=64, layers=2, heads=4, cand_layers=1))
    engine, encoder = FakeEngine(), Encoder(cfg.caps)
    try:
        obs = [encoder.encode(engine.reset(seed=f"grow-{i}")) for i in range(3)]
    finally:
        engine.close()
    batch = sliced(to_torch({k: np.stack([o[k] for o in obs]) for k in obs[0]}, "cpu"), torch.arange(3))
    model = Policy(cfg.model)
    opt = tr.make_optimizer(model, cfg)
    for _ in range(3):
        opt.zero_grad()
        logits, value = model(batch)
        loss = -logits.log_softmax(-1)[:, 0].mean() + (value - 0.7).square().mean()
        loss.backward()
        opt.step()
    vnorm = ValueNorm()
    vnorm.update([0.2, 0.8])
    state = dict(update=3, env_steps=1536, episodes=5, wall=1.5, lr_now=2e-4, kl_ema=0.001,
                 asc_frontier=4, cuda_rng_state=[torch.get_rng_state()])
    src = tr.save_ckpt(str(root / "source"), model, opt, vnorm, cfg, state, 1, np.random.default_rng(7))
    paths = {}
    for init in ("random", "copy"):
        dst = root / f"{init}.pt"
        result = subprocess.run([sys.executable, "-m", "tools.grow_ckpt", "--src", src, "--dst", str(dst),
                                 "--add_layers", "2", "--add_cand_layers", "1", "--init", init, "--seed", "17"],
                                check=True, capture_output=True, text=True)
        assert "parameters:" in result.stdout and "grow_params:" in result.stdout
        paths[init] = dst
    return src, paths, batch


@pytest.mark.parametrize("init", ["random", "copy"])
def test_grow_preserves_function_and_adam_by_name(checkpoints, init):
    src, paths, batch = checkpoints
    old = torch.load(src, map_location="cpu", weights_only=False)
    grown = torch.load(paths[init], map_location="cpu", weights_only=False)
    cfg = Config.from_dict(grown["cfg"])
    assert cfg.model.layers == 4 and cfg.model.cand_layers == 2
    assert cfg.grow_params == "blocks.1.,blocks.3.,cand_blocks.1."
    old_model, model = Policy(Config.from_dict(old["cfg"]).model), Policy(cfg.model)
    old_model.load_state_dict(old["model"])
    model.load_state_dict(grown["model"])
    with torch.no_grad():
        logits0, value0 = old_model.eval()(batch)
        logits1, value1 = model.eval()(batch)
    mask = torch.arange(logits0.shape[1])[None] < batch["n_cand"].long()[:, None]
    assert (logits0[mask] - logits1[mask]).abs().max().item() < 1e-5
    assert (value0 - value1).abs().max().item() < 1e-5

    names = [n for n, _ in model.named_parameters()]
    indices = dict(zip(names, grown["opt"]["param_groups"][0]["params"]))
    old_group = old["opt"]["param_groups"][0]
    for (name, p), i in zip(old_model.named_parameters(), old_group["params"]):
        if name.startswith("blocks."):
            tower, block, suffix = name.split(".", 2)
            new_name = f"{tower}.{int(block) * 2}.{suffix}"
        else:
            new_name = name
        assert torch.equal(p, grown["model"][new_name])
        if i in old["opt"]["state"]:
            _equal(grown["opt"]["state"][indices[new_name]], old["opt"]["state"][i])
    prefixes = tuple(cfg.grow_params.split(","))
    for name, i in indices.items():
        if name.startswith(prefixes):
            assert i not in grown["opt"]["state"]
    assert grown["opt"]["param_groups"][0]["params"] == list(range(len(names)))
    _equal({k: v for k, v in grown["opt"]["param_groups"][0].items() if k != "params"},
           {k: v for k, v in old_group.items() if k != "params"})
    for tower, pairs in (("blocks", [(0, 1), (1, 3)]), ("cand_blocks", [(0, 1)])):
        for before, after in pairs:
            block = getattr(model, tower)[after]
            outputs = [block.attn.o] if tower == "blocks" else [block.self_attn.o, block.cross.o]
            for layer in [*outputs, block.mlp[-1]]:
                assert torch.count_nonzero(layer.weight) == 0 and torch.count_nonzero(layer.bias) == 0
            if init == "copy":
                previous = getattr(old_model, tower)[before]
                for name, p in block.named_parameters():
                    if not name.startswith(("attn.o.", "self_attn.o.", "cross.o.", "mlp.2.")):
                        assert torch.equal(p, dict(previous.named_parameters())[name])
    assert "cuda_rng_state" not in grown
    for key in old.keys() - {"model", "opt", "cfg", "cuda_rng_state"}:
        _equal(grown[key], old[key])
    expected_cfg = copy.deepcopy(old["cfg"])
    expected_cfg["model"].update(layers=4, cand_layers=2)
    expected_cfg["grow_params"] = cfg.grow_params
    assert grown["cfg"] == expected_cfg


def _saved_states(model, sd, prefixes):
    named = list(model.named_parameters())
    groups = ([n for n, _ in named if not n.startswith(prefixes)],
              [n for n, _ in named if n.startswith(prefixes)])
    return {name: sd["state"][i] for group, names in zip(sd["param_groups"], groups)
            for name, i in zip(names, group["params"]) if i in sd["state"]}


@pytest.mark.parametrize("asynchronous", [False, True])
def test_grow_training_lr_and_resume_by_name(checkpoints, tmp_path, monkeypatch, asynchronous):
    _, paths, _ = checkpoints
    grown = torch.load(paths["random"], map_location="cpu", weights_only=False)
    args = ["--preset", "tiny", "--engine", "fake", "--run_dir", str(tmp_path),
            "--model.layers", "4", "--model.cand_layers", "2", "--grow_params", grown["cfg"]["grow_params"],
            "--grow_lr_mult", "3", "--async_update", str(asynchronous), "--ckpt_every", "1",
            "--rollout_steps", "64", "--minibatch", "32", "--micro_batch", "32", "--epochs", "1",
            "--lr_kl_target", "0.002" if asynchronous else "0"]
    restored, rates = [], []
    real_load, real_update = tr.load_opt_state, tr.ppo_update

    def load(opt, model, sd, cfg, saved_cfg):
        real_load(opt, model, sd, cfg, saved_cfg)
        if len(sd["param_groups"]) == 1:
            expected = {name: sd["state"][i] for (name, _), i in
                        zip(model.named_parameters(), sd["param_groups"][0]["params"]) if i in sd["state"]}
        else:
            expected = _saved_states(model, sd, tuple(saved_cfg["grow_params"].split(",")))
        for name, p in model.named_parameters():
            if name in expected:
                _equal(opt.state[p], expected[name])
            else:
                assert p not in opt.state
        restored.append(True)

    def update(model, opt, *a, **kw):
        assert len(opt.param_groups) == 2
        assert opt.param_groups[1]["lr"] / opt.param_groups[0]["lr"] == pytest.approx(3)
        rates.append(opt.param_groups[0]["lr"])
        return real_update(model, opt, *a, **kw)

    monkeypatch.setattr(tr, "load_opt_state", load)
    monkeypatch.setattr(tr, "ppo_update", update)
    assert tr.train(parse_args(args + ["--resume", str(paths["random"]), "--total_updates", "5"]))["update"] == 5
    ck = torch.load(tmp_path / "ckpt/latest.pt", map_location="cpu", weights_only=False)
    assert ck["opt"]["param_groups"][1]["lr"] / ck["opt"]["param_groups"][0]["lr"] == pytest.approx(3)
    assert tr.train(parse_args(args + ["--resume", "auto", "--total_updates", "6"]))["update"] == 6
    assert len(restored) == 2 and len(rates) == 3
    with (tmp_path / "progress.csv").open() as f:
        assert [float(row["lr"]) for row in csv.DictReader(f)] == rates

    model = Policy(Config.from_dict(ck["cfg"]).model)
    expected = _saved_states(model, ck["opt"], tuple(ck["cfg"]["grow_params"].split(",")))
    for cfg in (replace(Config.from_dict(ck["cfg"]), grow_lr_mult=1),
                replace(Config.from_dict(ck["cfg"]), grow_lr_mult=1, weight_decay=0.1),
                replace(Config.from_dict(ck["cfg"]), grow_params="blocks.0.,cand_blocks.0.", grow_lr_mult=5)):
        opt = tr.make_optimizer(model, cfg)
        real_load(opt, model, ck["opt"], cfg, ck["cfg"])
        for name, p in model.named_parameters():
            if name in expected:
                _equal(opt.state[p], expected[name])
        if cfg.grow_lr_mult != 1:
            assert [g["lr_mult"] for g in opt.param_groups] == [1, 5]
        else:
            assert all("lr_mult" not in g for g in opt.param_groups)
        if cfg.weight_decay:
            back_cfg = Config.from_dict(ck["cfg"])
            back = tr.make_optimizer(model, back_cfg)
            real_load(back, model, opt.state_dict(), back_cfg, cfg.to_dict())
            for name, p in model.named_parameters():
                if name in expected:
                    _equal(back.state[p], expected[name])


def test_grow_multiplier_one_keeps_single_group():
    model = Policy(ModelCfg(d=64, layers=2, heads=4, cand_layers=1))
    baseline = tr.make_optimizer(model, Config())
    for cfg in (Config(grow_params="blocks.1.,cand_blocks.0.", grow_lr_mult=1), Config(grow_lr_mult=3)):
        opt = tr.make_optimizer(model, cfg)
        assert len(opt.param_groups) == 1
        assert all(p is q for p, q in zip(opt.param_groups[0]["params"], model.parameters()))
        _equal(opt.state_dict(), baseline.state_dict())
