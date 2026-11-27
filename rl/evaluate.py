"""Evaluate a checkpoint on a fixed seed list (no save/load, one attempt per seed).

    python -m rl.evaluate --ckpt runs/x/ckpt/latest.pt --n_seeds 200 --greedy
    python -m rl.evaluate --policy random --engine fake --n_seeds 500     # baseline

Reports win rate with a 95% Wilson interval, the final-floor distribution, and per-character results.
Seeds are `eval_<i>`, disjoint from training seeds (`s<seed>_e<env>_<n>`).
"""
import argparse
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np
import torch

from .actors import ActorPool
from .config import Config
from .ppo import to_torch
from .train import act_batch, amp_context, load_policy


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def summarize(eps):
    done = [e for e in eps if not e.get("aborted")]
    k = sum(bool(e["victory"]) for e in done)
    p, lo, hi = wilson(k, len(done))
    fl = sorted(e["floor"] for e in done)
    out = {"n": len(done), "aborted": len(eps) - len(done), "wins": k, "win_rate": p, "ci95": [lo, hi]}
    if fl:
        out.update(floor_mean=statistics.mean(fl), floor_median=statistics.median(fl),
                   floor_q25=fl[len(fl) // 4], floor_q75=fl[(3 * len(fl)) // 4],
                   floor_hist=dict(sorted(Counter(fl).items())),
                   steps_mean=statistics.mean(e["steps"] for e in done))
    else:
        out.update(floor_mean=float("nan"), floor_median=float("nan"), floor_q25=float("nan"),
                   floor_q75=float("nan"), floor_hist={}, steps_mean=float("nan"))
    return out


def run_eval(cfg, model, seeds, greedy=True, device="cpu", random_policy=False, progress=True, stall_seconds=1800.0):
    """seeds: list of (character, seed). Returns the list of episode dicts."""
    cfg = replace(cfg, group_k=1, branch_decisions="", explore_eps=0.0, force_frac=0.0, taste_frac=0.0,
                  asc_curriculum=False, log_actions=True)
    pool = ActorPool(cfg, seed_list=seeds)
    amp_ctx = amp_context(cfg, torch.device(device))
    ready, eps = set(), []
    rng = np.random.default_rng(0)
    t0, last = time.time(), 0
    last_progress = time.time()
    try:
        while len(eps) < len(seeds):
            if time.time() - last_progress > stall_seconds:
                # Every seed should end as an episode; if some never report, do not spin forever.
                seen = {e.get("seed") for e in eps}
                missing = [s for _, s in seeds if s not in seen]
                print(f"  no progress for {stall_seconds:.0f}s; marking {len(missing)} seeds aborted", flush=True)
                eps += [{"seed": s, "char": None, "victory": None, "floor": 0, "aborted": "eval_stall"} for s in missing]
                break
            for ev in pool.poll(0.05):
                if ev[0] == "res" and ev[5]:
                    ready.add(ev[1])
                elif ev[0] == "obs":
                    ready.add(ev[1])
                elif ev[0] == "ep":
                    eps.append(ev[1])
                    last_progress = time.time()
            if ready:
                envs = sorted(ready)
                ready.clear()
                obs_np = pool.shm.gather(envs)
                if random_policy:
                    a = rng.integers(0, obs_np["n_cand"].astype(np.int64))
                else:
                    a, _, _ = act_batch(model, to_torch(obs_np, device), amp_ctx, greedy=greedy)
                    a = a.cpu().numpy()
                pool.act(envs, a)
            if progress and len(eps) >= last + max(1, len(seeds) // 10):
                last = len(eps)
                print(f"  {len(eps)}/{len(seeds)} episodes, {time.time() - t0:.0f}s", flush=True)
    finally:
        pool.close()
    return eps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--policy", default="model", choices=("model", "random"))
    ap.add_argument("--greedy", action="store_true", help="argmax instead of sampling")
    ap.add_argument("--n_seeds", type=int, default=200)
    ap.add_argument("--seed_prefix", default="eval")
    ap.add_argument("--characters", default="", help="default: the checkpoint's training characters")
    ap.add_argument("--ascension", type=int, default=None)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--engine_dir", default=None)
    ap.add_argument("--restart_every_episode", action="store_true")
    ap.add_argument("--restart_every_n", type=int, default=None, help="default: the checkpoint's training setting")
    ap.add_argument("--fake_p_error", type=float, default=0.0)
    ap.add_argument("--n_workers", type=int, default=8)
    ap.add_argument("--envs_per_worker", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="", help="write the summary + episodes as JSON here")
    a = ap.parse_args(argv)

    if a.policy == "model":
        model, _, cfg, ck = load_policy(a.ckpt, a.device)
        model.eval()
        tag = f"{a.ckpt} (update {ck['update']}, {ck['env_steps']} steps)"
    else:
        model, cfg, tag = None, Config(), "uniform random policy"
    cfg.n_workers, cfg.envs_per_worker = a.n_workers, a.envs_per_worker
    cfg.fake_p_error = a.fake_p_error
    for k in ("engine", "engine_dir", "ascension"):
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))
    if a.restart_every_episode:
        cfg.restart_every_episode = True
    if a.restart_every_n:
        cfg.restart_every_n = a.restart_every_n
    chars = [c for c in (a.characters or cfg.characters).split(",") if c]
    seeds = [(chars[i % len(chars)], f"{a.seed_prefix}_{i}") for i in range(a.n_seeds)]
    print(f"evaluating {tag} on {len(seeds)} seeds ({','.join(chars)} A{cfg.ascension}), "
          f"{'greedy' if a.greedy else 'sampled'}, engine={cfg.engine}", flush=True)
    eps = run_eval(cfg, model, seeds, greedy=a.greedy, device=a.device, random_policy=a.policy == "random")
    summ = {"all": summarize(eps)}
    by = defaultdict(list)
    for e in eps:
        by[e["char"]].append(e)
    if len(by) > 1:
        summ.update({c: summarize(v) for c, v in by.items()})
    for k, s in summ.items():
        if s["n"]:
            print(f"{k:12s} n={s['n']} win={s['win_rate']:.3f} 95%CI=[{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}] "
                  f"floor mean={s['floor_mean']:.2f} median={s['floor_median']} IQR=[{s['floor_q25']},{s['floor_q75']}] "
                  f"aborted={s['aborted']}")
            print(f"{'':12s} floor histogram: {s['floor_hist']}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"tag": tag, "greedy": a.greedy, "summary": summ, "episodes": eps}, f, indent=1)
    return summ


if __name__ == "__main__":
    main()
