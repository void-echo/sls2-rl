import torch

from rl.config import Config
from rl.model import ModelCfg, Policy
from rl.train import load_opt_state, make_optimizer


def _tiny():
    torch.manual_seed(0)
    return Policy(ModelCfg(d=32, layers=1, heads=4, cand_layers=1))


def _step(model, opt):
    opt.zero_grad()
    sum(p.float().pow(2).sum() for p in model.parameters()).backward()
    opt.step()


def test_no_decay_keeps_single_group():
    opt = make_optimizer(_tiny(), Config(weight_decay=0.0))
    assert len(opt.param_groups) == 1 and opt.param_groups[0]["weight_decay"] == 0.0


def test_decay_groups_exclude_embeddings_norms_biases():
    model = _tiny()
    opt = make_optimizer(model, Config(weight_decay=0.1))
    names = {id(p): n for n, p in model.named_parameters()}
    decayed = {names[id(p)] for p in opt.param_groups[0]["params"]}
    kept = {names[id(p)] for p in opt.param_groups[1]["params"]}
    assert opt.param_groups[0]["weight_decay"] == 0.1 and opt.param_groups[1]["weight_decay"] == 0.0
    assert "table" in kept and "type_emb.weight" in kept
    assert "blocks.0.mlp.0.weight" in decayed and "blocks.0.attn.q.weight" in decayed and "logit.1.weight" in decayed
    assert "blocks.0.ln1.weight" in kept and "blocks.0.mlp.0.bias" in kept
    assert len(decayed) + len(kept) == len(list(model.parameters()))


def test_resume_across_groupings_keeps_adam_moments():
    model = _tiny()
    old = make_optimizer(model, Config(weight_decay=0.0))
    _step(model, old)
    sd = old.state_dict()
    new = make_optimizer(model, Config(weight_decay=0.1))
    load_opt_state(new, model, sd, Config(weight_decay=0.1))
    for p in model.parameters():
        assert torch.equal(new.state[p]["exp_avg"], old.state[p]["exp_avg"])
        assert torch.equal(new.state[p]["exp_avg_sq"], old.state[p]["exp_avg_sq"])
    _step(model, new)
    back = make_optimizer(model, Config(weight_decay=0.0))
    load_opt_state(back, model, new.state_dict(), Config(weight_decay=0.0))
    for p in model.parameters():
        assert torch.equal(back.state[p]["exp_avg"], new.state[p]["exp_avg"])
    assert back.param_groups[0]["weight_decay"] == 0.0


def test_same_grouping_takes_decay_from_config():
    model = _tiny()
    opt = make_optimizer(model, Config(weight_decay=0.1))
    _step(model, opt)
    again = make_optimizer(model, Config(weight_decay=0.3))
    load_opt_state(again, model, opt.state_dict(), Config(weight_decay=0.3))
    assert again.param_groups[0]["weight_decay"] == 0.3 and again.param_groups[1]["weight_decay"] == 0.0
