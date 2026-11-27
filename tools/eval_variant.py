"""Evaluate variants of a policy that need no retraining: the average of several checkpoints' weights (Polyak-Ruppert / SWA)
and sampling at a temperature instead of greedy. Same seeds and episode format as rl.evaluate, so results can be paired.

    python tools/eval_variant.py --ckpts a.pt b.pt c.pt --ascension 10 --n_seeds 2000 --seed_prefix fresh1105 \
        --engine_dir ~/sls2-rl/engine_v2 --device cuda:0 --out out.json            # greedy on the averaged weights
    python tools/eval_variant.py --ckpts c.pt --temperature 0.25 ...               # single checkpoint, T = 0.25
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.evaluate import run_eval, summarize  # noqa: E402
from rl.train import load_policy  # noqa: E402


class Tempered(torch.nn.Module):
    """Divides the logits by T; masked candidates stay at a large negative value, so they are still never sampled."""

    def __init__(self, model, temperature):
        super().__init__()
        self.model, self.temperature = model, temperature

    def forward(self, obs):
        logits, v = self.model(obs)
        return logits / self.temperature, v


def averaged_state(paths):
    """Uniform average of the floating-point entries of several checkpoints; other entries come from the last one."""
    out = None
    for i, p in enumerate(paths):
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        if out is None:
            out = {k: v.clone().double() if v.is_floating_point() else v.clone() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if v.is_floating_point():
                    out[k] += (v.double() - out[k]) / (i + 1)
                else:
                    out[k] = v.clone()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (argmax)")
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--n_seeds", type=int, default=500)
    ap.add_argument("--seed_prefix", default="fresh1105")
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--seed_file", default="", help="JSON (a list, or an object with a \"seeds\" list) or one seed per line; "
                    "replaces seed_prefix: the n_seeds seeds from seed_start on, used as game seeds verbatim")
    ap.add_argument("--engine_dir", required=True)
    ap.add_argument("--n_workers", type=int, default=8)
    ap.add_argument("--envs_per_worker", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--restart_every_n", type=int, default=None, help="default: the checkpoint's training setting")
    ap.add_argument("--characters", default="", help="comma list; default: the checkpoint's training characters (e.g. Fool to see how another character's model plays the new one)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model, _, cfg, ck = load_policy(a.ckpts[-1], a.device)
    if len(a.ckpts) > 1:
        sd = averaged_state(a.ckpts)
        model.load_state_dict({k: v.to(next(iter(model.state_dict().values())).device).to(model.state_dict()[k].dtype) for k, v in sd.items()})
    model.eval()
    policy = Tempered(model, a.temperature).eval() if a.temperature > 0 else model
    cfg.n_workers, cfg.envs_per_worker = a.n_workers, a.envs_per_worker
    cfg.engine, cfg.engine_dir, cfg.ascension, cfg.restart_every_episode = "real", a.engine_dir, a.ascension, True
    if a.restart_every_n:
        cfg.restart_every_n = a.restart_every_n
    chars = [c for c in (a.characters or cfg.characters).split(",") if c]
    if a.seed_file:
        txt = open(a.seed_file).read()
        try:
            obj = json.loads(txt)
            pool = obj["seeds"] if isinstance(obj, dict) else obj
        except ValueError:
            pool = [x.strip() for x in txt.splitlines() if x.strip()]
        names = [str(x) for x in pool[a.seed_start:a.seed_start + a.n_seeds]]
    else:
        names = [f"{a.seed_prefix}_{i}" for i in range(a.seed_start, a.seed_start + a.n_seeds)]
    seeds = [(chars[i % len(chars)], name) for i, name in enumerate(names)]
    tag = f"{len(a.ckpts)} ckpts ending {os.path.basename(a.ckpts[-1])} u{ck['update']} T={a.temperature} A{a.ascension}"
    print(f"evaluating {tag} on {len(seeds)} seeds", flush=True)
    eps = run_eval(cfg, policy, seeds, greedy=a.temperature <= 0, device=a.device, progress=True)
    s = summarize(eps)
    print(f"{tag}: n={s['n']} win={s['win_rate']:.4f} 95%CI=[{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}] floor={s['floor_mean']:.2f} aborted={s['aborted']}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"tag": tag, "ckpts": a.ckpts, "temperature": a.temperature, "summary": {"all": s}, "episodes": eps}, f)


if __name__ == "__main__":
    main()
