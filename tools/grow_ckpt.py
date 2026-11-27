"""Insert identity residual blocks into a training checkpoint."""
import argparse
from dataclasses import replace

import torch

from rl.model import ModelCfg, Policy, n_params


def grow_checkpoint(src, dst, add_layers, add_cand_layers, init="random", seed=0):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    old_cfg = ModelCfg(**ck["cfg"]["model"])
    for name, count, depth in (("add_layers", add_layers, old_cfg.layers),
                               ("add_cand_layers", add_cand_layers, old_cfg.cand_layers)):
        if not 0 <= count <= depth:
            raise ValueError(f"{name} must be between 0 and {depth}")
    if old_cfg.value_tower and not old_cfg.value_layers and add_layers:
        raise ValueError("growing layers with an implicit value_tower depth is unsupported")
    if init not in ("random", "copy"):
        raise ValueError(f"unknown initialization: {init}")
    if len(ck["opt"]["param_groups"]) != 1:
        raise ValueError("source optimizer must have a single parameter group")

    old_model = Policy(old_cfg)
    old_model.load_state_dict(ck["model"])
    new_cfg = replace(old_cfg, layers=old_cfg.layers + add_layers,
                      cand_layers=old_cfg.cand_layers + add_cand_layers)
    torch.manual_seed(seed)
    model = Policy(new_cfg)
    positions, grow_params = {}, []
    with torch.no_grad():
        for tower, count in (("blocks", add_layers), ("cand_blocks", add_cand_layers)):
            old_blocks, new_blocks = getattr(old_model, tower), getattr(model, tower)
            offset = 0
            for i, old_block in enumerate(old_blocks):
                j = i + offset
                positions[tower, str(i)] = str(j)
                if i < len(old_blocks) - count:
                    continue
                block = new_blocks[j + 1]
                if init == "copy":
                    block.load_state_dict(old_block.state_dict())
                outputs = ([block.attn.o] if tower == "blocks" else [block.self_attn.o, block.cross.o])
                for layer in [*outputs, block.mlp[-1]]:
                    layer.weight.zero_()
                    layer.bias.zero_()
                grow_params.append(f"{tower}.{j + 1}.")
                offset += 1

    name_map = {}
    weights = model.state_dict()
    for name, value in old_model.state_dict().items():
        parts = name.split(".", 2)
        new_name = (f"{parts[0]}.{positions[parts[0], parts[1]]}.{parts[2]}"
                    if parts[0] in ("blocks", "cand_blocks") else name)
        name_map[name] = new_name
        weights[new_name] = value
    model.load_state_dict(weights)

    old_group = ck["opt"]["param_groups"][0]
    old_names = [n for n, _ in old_model.named_parameters()]
    if len(old_group["params"]) != len(old_names):
        raise ValueError("source optimizer parameter count does not match the model")
    new_indices = {n: i for i, (n, _) in enumerate(model.named_parameters())}
    state = {new_indices[name_map[name]]: ck["opt"]["state"][i]
             for name, i in zip(old_names, old_group["params"]) if i in ck["opt"]["state"]}
    ck["model"] = model.state_dict()
    ck["opt"] = {**ck["opt"], "state": state,
                 "param_groups": [{**old_group, "params": list(range(len(new_indices)))}]}
    ck["cfg"]["model"] = {**ck["cfg"]["model"], "layers": new_cfg.layers, "cand_layers": new_cfg.cand_layers}
    ck["cfg"]["grow_params"] = ",".join(grow_params)
    ck.pop("cuda_rng_state", None)
    torch.save(ck, dst)
    print(f"parameters: {n_params(old_model)} -> {n_params(model)}")
    print(f"grow_params: {ck['cfg']['grow_params']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--add_layers", type=int, default=0)
    parser.add_argument("--add_cand_layers", type=int, default=0)
    parser.add_argument("--init", choices=("random", "copy"), default="random")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    grow_checkpoint(**vars(args))


if __name__ == "__main__":
    main()
