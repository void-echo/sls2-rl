"""Learner throughput benchmark on synthetic-but-realistic observations.

    python -m rl.bench --device cuda --envs 256 --steps 128

Observations come from FakeGame random play, inflated toward real-game sizes (deck and piles x2,
~12 relics, 16-row maps) so token counts resemble mid-game StS2. Reports inference samples/s per
batch size and PPO update samples/s for an envs x steps rollout.
"""
import argparse
import random
import time

import numpy as np
import torch

from .config import Config
from .encode import Caps, Encoder
from .fake_engine import FakeServer
from .model import ModelCfg, Policy, n_params
from .ppo import Rollout, ValueNorm, ppo_update, sliced, to_torch
from .train import act_batch, amp_context


def make_obs(n, caps, inflate=2, seed=0):
    srv = FakeServer(rows=16)
    rng = random.Random(seed)
    enc = Encoder(caps)
    arrs = enc.alloc(n)
    i = 0
    R = srv.handle({"cmd": "rl_start", "seed": "b0"})
    ep = 0
    while i < n:
        if R["done"]:
            ep += 1
            R = srv.handle({"cmd": "rl_start", "seed": f"b{ep}"})
            continue
        if inflate:
            R2 = dict(R)
            obs = dict(R["obs"])
            obs["deck"] = obs["deck"] * inflate
            obs["relics"] = obs["relics"] + [{"id": f"RELIC.FAKE_{k:02d}", "v": {"Counter": k}} for k in range(11)]
            if obs["combat"]:
                cb = dict(obs["combat"])
                for p in ("draw", "discard"):
                    cb[p] = cb[p] * inflate
                obs["combat"] = cb
            R2["obs"] = obs
        else:
            R2 = R
        enc.encode(R2, {k: v[i, ...] for k, v in arrs.items()})
        i += 1
        R = srv.handle({"cmd": "rl_act", "i": rng.randrange(len(R["cands"]))})
    return arrs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--micro_batch", type=int, default=1024)
    ap.add_argument("--n_obs", type=int, default=8192, help="distinct observations (tiled to the rollout size)")
    ap.add_argument("--inflate", type=int, default=2, help="deck/pile multiplier (0 = raw fake states)")
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--lam_build", type=float, default=None)
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    caps = Caps()
    t = time.time()
    arrs = make_obs(a.n_obs, caps, inflate=a.inflate)
    nt, nc = arrs["n_tok"].astype(int), arrs["n_cand"].astype(int)
    print(f"made {a.n_obs} obs in {time.time() - t:.1f}s; tokens mean={nt.mean():.1f} p50={np.median(nt):.0f} "
          f"p95={np.percentile(nt, 95):.0f} max={nt.max()}; cands mean={nc.mean():.1f} max={nc.max()}")

    cfg = Config(device=a.device, amp=bool(a.amp), epochs=a.epochs, minibatch=a.minibatch, micro_batch=a.micro_batch,
                 lam_build=a.lam_build, model=ModelCfg(d=a.d, layers=a.layers, heads=a.heads))
    model = Policy(cfg.model).to(dev)
    print(f"model d={a.d} layers={a.layers} heads={a.heads} params={n_params(model) / 1e6:.2f}M amp={cfg.amp}")
    amp_ctx = amp_context(cfg, dev)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)

    model.eval()
    rng = np.random.default_rng(0)
    outer = amp_ctx()  # as in the collector: one autocast context for the whole rollout
    outer.__enter__()
    for bs in (64, 128, 256, 512):
        times, times_io = [], []
        for it in range(12):
            idx = rng.choice(a.n_obs, bs, replace=False)
            t0 = time.time()
            T, C = -(-max(nt[idx].max(), 1) // 8) * 8, -(-max(nc[idx].max(), 1) // 8) * 8
            npb = {k: (v[idx, :T] if k.startswith("tok_") else v[idx, :C] if k.startswith("cand_") else v[idx])
                   for k, v in arrs.items()}
            b = to_torch(npb, dev)
            sync()
            t1 = time.time()
            act_batch(model, b, amp_ctx)
            sync()
            t2 = time.time()
            if it >= 2:
                times.append(t2 - t1)
                times_io.append(t2 - t0)
        print(f"inference bs={bs:4d}: model {bs / np.median(times):9.0f} samples/s ({1000 * np.median(times):.1f} ms), "
              f"incl. gather+H2D {bs / np.median(times_io):9.0f} samples/s")
    outer.__exit__(None, None, None)

    n = a.envs * a.steps
    buf = Rollout(caps, n, a.envs, dev, track_combat=cfg.lam_build is not None)
    for s in range(0, n, 1024):
        idx = np.arange(s, min(n, s + 1024)) % a.n_obs
        b = to_torch({k: v[idx] for k, v in arrs.items()}, dev)
        envs = np.arange(s, s + len(idx)) % a.envs
        buf.add(envs, b, rng.integers(0, nc[idx]), np.full(len(idx), -1.0, np.float32),
                rng.normal(size=len(idx)).astype(np.float32))
    for i in range(n):
        buf.set_result(i, float(rng.normal() * 0.01), rng.random() < 0.002, False)
    boot = np.zeros(a.envs, np.float32)
    if cfg.lam_build is None:
        adv, ret, mask = buf.gae(boot, cfg.gamma, cfg.lam)
    else:
        adv, ret, mask, _ = buf.gae_build(boot, cfg.gamma, cfg.lam, cfg.lam_build)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, eps=1e-5)
    model.train()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for rep in range(2):
        st = ppo_update(model, opt, buf, adv, ret, mask, ValueNorm(), cfg, rng, amp_ctx)
        print(f"PPO update {rep}: {n} samples x {st['epochs_done']} epochs in {st['update_time']:.2f}s -> "
              f"{st['learner_sps']:.0f} samples/s (minibatch {a.minibatch}, micro {a.micro_batch})")
    if dev.type == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
              f"(rollout obs buffer {sum(v.numel() * v.element_size() for v in buf.obs.values()) / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
