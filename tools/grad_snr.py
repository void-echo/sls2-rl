"""Compare independent, episode-disjoint policy gradients at frozen checkpoints.

    PYTHONPATH=. python3 tools/grad_snr.py --ckpt runs/x/ckpt/latest.pt --out snr.json

Rollout/GAE horizons and micro_batch come from the checkpoint. Observations are
staged on CPU and archived in temporary, padding-trimmed, memory-mapped chunks
beside --out, so --steps does not become a GPU allocation. Combat/build are
subsets of the SAME episode-disjoint pair, with no further advantage centering.
The c fit uses their actual harmonic-mean pair sizes in K/(K+c).
"""
import argparse
import contextlib
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.actors import ActorPool
from rl.ppo import Rollout, sliced
from rl.train import Collector, amp_context, load_policy


class EpisodeCollector(Collector):
    """Track real episode ends, including aborts, across collection horizons.

    A replay's terminal `res` is not an episode end: only ActorPool's `ep` event
    advances the ID, keeping combat/decision side trials with their source run.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_ids = np.arange(self.pool.n_envs, dtype=np.int64)

    def handle(self, events, buf):
        for ev in events:
            super().handle([ev], buf)
            if ev[0] == "ep":
                self.episode_ids[ev[1]["env"]] += self.pool.n_envs


class EpisodeRollout(Rollout):
    def add(self, envs, obs, act, logp, val):
        idx = super().add(envs, obs, act, logp, val)
        self.episode_chunks.append(self.collector.episode_ids[np.asarray(envs)].copy())
        return idx

    def reset(self):
        super().reset()
        self.episode_chunks = []


def collect_rollouts(col, cfg, steps):
    """Yield the training collector's CPU rollouts and their bootstrap values."""
    count = 0
    while count < steps:
        horizon = min(cfg.rollout_steps, steps - count)
        old_horizon, cfg.rollout_steps = cfg.rollout_steps, horizon
        buf = EpisodeRollout(cfg.caps, horizon, cfg.n_envs, "cpu", track_combat=True)
        buf.collector = col
        try:
            boot, _ = col.collect(buf)
        finally:
            cfg.rollout_steps = old_horizon
        yield count, buf, boot
        count += buf.n
        del buf


def store_observations(buf, directory, prefix):
    """Archive one padding-trimmed chunk and reopen it as copy-on-write mmap."""
    stored = {}
    for key, value in sliced(buf.obs, slice(0, buf.n)).items():
        path = Path(directory) / f"{prefix}_{key}.npy"
        np.save(path, value.numpy())
        stored[key] = torch.from_numpy(np.load(path, mmap_mode="c"))
    return stored


def collect_samples(model, vnorm, cfg, steps, device, directory, env_steps):
    """Run the training Collector and GAE, then normalize once over all valid data."""
    chunks, advantages, actions, masks, combats, episodes = [], [], [], [], [], []
    pool = ActorPool(cfg, resume_steps=env_steps)
    col = EpisodeCollector(cfg, pool, model, vnorm, device, amp_context(cfg, device), err_log=False)
    count = 0
    try:
        for count, buf, boot in collect_rollouts(col, cfg, steps):
            if cfg.lam_build is None:
                adv, _, mask = buf.gae(boot, cfg.gamma, cfg.lam)
            else:
                adv, _, mask, _ = buf.gae_build(boot, cfg.gamma, cfg.lam, cfg.lam_build)
            if cfg.group_k > 1:
                adv, _ = buf.group_advantages(adv, cfg.gamma, cfg.group_alpha)
            stored = store_observations(buf, directory, len(chunks))
            chunks.append((count, count + buf.n, stored))
            advantages.append(adv)
            actions.append(buf.act[:buf.n].copy())
            masks.append(mask)
            combats.append(buf.combat[:buf.n].copy())
            episodes.append(np.concatenate(buf.episode_chunks))
            count += buf.n
            print(f"  collected {count}/{steps} transitions", flush=True)
            del buf
    finally:
        pool.close()
    adv, mask = np.concatenate(advantages), np.concatenate(masks)
    normalized = np.zeros_like(adv)
    a = adv[mask]
    normalized[mask] = (a - a.mean()) / (a.std() + 1e-8)
    return chunks, normalized, np.concatenate(actions), mask, np.concatenate(combats), np.concatenate(episodes), dict(col.errors)


def episode_pair(episode_ids, valid, k, rng):
    """Assign whole runs to two balanced groups until both reach approximately K.

    Runs can recur in different pairs, but never on both sides of a pair. Short
    collection tails remain with their run; a shortage uses all available runs.
    """
    idx = np.flatnonzero(valid)
    order = idx[np.argsort(episode_ids[idx], kind="stable")]
    groups = np.split(order, np.flatnonzero(np.diff(episode_ids[order])) + 1) if len(order) else []
    sides, sizes = [[], []], [0, 0]
    for j in rng.permutation(len(groups)):
        side = int(sizes[1] < sizes[0])
        sides[side].append(groups[j])
        sizes[side] += len(groups[j])
        if min(sizes) >= k:
            break
    return tuple(np.concatenate(side) if side else np.empty(0, np.int64) for side in sides)


def policy_gradient(model, chunks, indices, advantages, actions, micro_batch, amp_ctx=contextlib.nullcontext):
    """Gradient of mean(A log pi); no ratio, clipping, entropy or value term."""
    model.eval()
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    for start, stop, obs in chunks:
        local = indices[(indices >= start) & (indices < stop)] - start
        # Token buckets keep the same padding reduction used by PPO minibatches.
        local = local[np.argsort(obs["n_tok"][local].numpy(), kind="stable")]
        for offset in range(0, len(local), micro_batch):
            ix = local[offset:offset + micro_batch]
            batch = {key: value.to(device) for key, value in sliced(obs, ix).items()}
            with amp_ctx():
                logits, _ = model(batch)
            act = torch.as_tensor(actions[start + ix], device=device)
            a = torch.as_tensor(advantages[start + ix], device=device)
            logp = torch.log_softmax(logits.float(), -1).gather(1, act[:, None]).squeeze(1)
            (torch.sum(a * logp) / len(indices)).backward()
    # Only two gradient copies live on CPU; value-only parameters have zero gradient.
    return {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None}


def gradient_products(a, b):
    """CPU inner products and squared norms for the two parameter groups."""
    results = {}
    for group in ("all", "no_table"):
        dot = aa = bb = 0.0
        for name in a.keys() | b.keys():
            if group == "no_table" and name == "table":
                continue
            x, y = a.get(name), b.get(name)
            if x is not None:
                aa += float(torch.sum(x.double().square()))
            if y is not None:
                bb += float(torch.sum(y.double().square()))
            if x is not None and y is not None:
                dot += float(torch.sum(x.double() * y.double()))
        results[group] = dict(dot=dot, norm_sq_a=aa, norm_sq_b=bb)
    return results


def gradient_cosines(a, b):
    return {group: min(1.0, max(-1.0, p["dot"] / math.sqrt(p["norm_sq_a"] * p["norm_sq_b"])))
            if p["norm_sq_a"] * p["norm_sq_b"] else None
            for group, p in gradient_products(a, b).items()}


def fit_noise_ratio(sizes, cosines):
    """Unweighted nonlinear least squares over per-K means, with c >= 0.

    Fit in log(1+c) coordinates. A negative/no-signal curve can have its minimum
    at c=infinity; encode that boundary as null plus an explicit status.
    """
    x, y = np.asarray(sizes, np.float64), np.asarray(cosines, np.float64)
    if not len(x):
        return dict(c=None, rmse=None, status="no_defined_cosines")
    def loss(t):
        return float(np.mean((x / (x + np.expm1(t)) - y) ** 2))
    grid = np.linspace(0, np.log1p(x.max() * 1e10), 257)
    best = int(np.argmin([loss(t) for t in grid]))
    if best == len(grid) - 1:
        return dict(c=None, rmse=float(np.sqrt(np.mean(y ** 2))), status="infinite_c")
    lo, hi = grid[max(0, best - 1)], grid[best + 1]
    phi = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        left, right = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if loss(left) < loss(right):
            hi = right
        else:
            lo = left
    t = (lo + hi) / 2
    if loss(0) <= loss(t):
        t = 0.0
    return dict(c=float(np.expm1(t)), rmse=math.sqrt(loss(t)), status="finite")


def summarize(pairs, parameter_group):
    valid = [p for p in pairs if p["cosine"][parameter_group] is not None]
    cos = [p["cosine"][parameter_group] for p in valid]
    sizes = [2 * p["n_a"] * p["n_b"] / (p["n_a"] + p["n_b"]) for p in valid]
    return dict(mean=float(np.mean(cos)) if cos else None,
                se=float(np.std(cos, ddof=1) / math.sqrt(len(cos))) if len(cos) > 1 else None,
                n_pairs=len(cos), effective_k=float(np.mean(sizes)) if sizes else None,
                mean_n_a=float(np.mean([p["n_a"] for p in pairs])),
                mean_n_b=float(np.mean([p["n_b"] for p in pairs])))


def measure_checkpoint(path, args, directory):
    device = torch.device(args.device)
    model, vnorm, cfg, ck = load_policy(path, "cpu")
    metadata = {key: ck[key] for key in ("update", "env_steps")}
    del ck  # discard optimizer state before moving only the policy to the learner device
    model.to(device)
    cfg.engine, cfg.engine_cmd = args.engine, ""
    cfg.asc_curriculum, cfg.ascension = False, args.ascension
    cfg.midrun_frac = 0.0  # plain full games at one level, whatever the run trained on
    cfg.engine_log = False
    for key in ("engine_dir", "n_workers", "envs_per_worker"):
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    print(f"\n{path}: update={metadata['update']} env_steps={metadata['env_steps']}", flush=True)
    chunks, adv, actions, mask, combat, episodes, errors = collect_samples(
        model, vnorm, cfg, args.steps, device, directory, metadata["env_steps"])
    measurements = {kind: {group: [] for group in ("all", "no_table")} for kind in ("all", "combat", "build")}
    self_check = None
    for k in args.batch_sizes:
        rows = {kind: [] for kind in measurements}
        for pair in range(args.pairs):
            left, right = episode_pair(episodes, mask, k, rng)
            run_ids = [np.unique(episodes[ix]).tolist() for ix in (left, right)]
            assert not set(run_ids[0]) & set(run_ids[1])
            for kind in measurements:
                a, b = left, right
                if kind != "all":
                    keep = combat if kind == "combat" else ~combat
                    a, b = a[keep[a]], b[keep[b]]
                if len(a) and len(b):
                    g_a = policy_gradient(model, chunks, a, adv, actions, cfg.micro_batch, amp_context(cfg, device))
                    if self_check is None:
                        again = policy_gradient(model, chunks, a, adv, actions, cfg.micro_batch, amp_context(cfg, device))
                        self_check = gradient_cosines(g_a, again)
                        # GPU sparse ops and bf16 are not bit-reproducible: about 1e-5 below 1 is normal there.
                        assert all(v is not None and abs(v - 1) < 1e-3 for v in self_check.values()), self_check
                        del again
                    g_b = policy_gradient(model, chunks, b, adv, actions, cfg.micro_batch, amp_context(cfg, device))
                    cosine = gradient_cosines(g_a, g_b)
                    del g_a, g_b
                else:
                    cosine = dict(all=None, no_table=None)
                rows[kind].append(dict(pair=pair, n_a=len(a), n_b=len(b), episode_ids_a=run_ids[0],
                                       episode_ids_b=run_ids[1], cosine=cosine))
            print(f"  K={k} pair={pair + 1}/{args.pairs} n={len(left)}/{len(right)}", flush=True)
        for kind, pairs in rows.items():
            for group in measurements[kind]:
                measurements[kind][group].append(dict(k=k, **summarize(pairs, group), pairs=pairs))
    fits = {}
    for kind, groups in measurements.items():
        fits[kind] = {}
        for group, rows in groups.items():
            defined = [r for r in rows if r["mean"] is not None]
            fits[kind][group] = fit_noise_ratio([r["effective_k"] for r in defined], [r["mean"] for r in defined])
    return dict(checkpoint=str(path), **metadata, ascension=cfg.ascension, characters=cfg.characters,
                engine=cfg.engine, steps_collected=len(adv), valid_steps=int(mask.sum()),
                episodes=int(len(np.unique(episodes))), rollout_steps=cfg.rollout_steps,
                micro_batch=cfg.micro_batch, gamma=cfg.gamma, lam=cfg.lam, lam_build=cfg.lam_build,
                engine_errors=errors, self_check_cosine=self_check, measurements=measurements, fits=fits)


def number(value):
    return "n/a" if value is None else f"{value:.6g}"


def print_table(result):
    print(f"\n{result['checkpoint']}  update={result['update']}  env_steps={result['env_steps']}")
    print(f"{'subset':<8} {'params':<9} {'K':>8} {'actual A/B':>17} {'cos mean':>10} {'SE':>10} {'fit c':>12}")
    for kind, groups in result["measurements"].items():
        for group, rows in groups.items():
            fit = result["fits"][kind][group]
            c = "inf" if fit["status"] == "infinite_c" else number(fit["c"])
            for r in rows:
                actual = f"{r['mean_n_a']:.1f}/{r['mean_n_b']:.1f}"
                print(f"{kind:<8} {group:<9} {r['k']:>8} {actual:>17} {number(r['mean']):>10} {number(r['se']):>10} {c:>12}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", nargs="+", action="extend", required=True)
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--engine", default="real", choices=("real", "fake", "fakeproc"))
    ap.add_argument("--engine_dir")
    ap.add_argument("--n_workers", type=int)
    ap.add_argument("--envs_per_worker", type=int)
    ap.add_argument("--steps", type=int, default=524288)
    ap.add_argument("--batch_sizes", default="8192,32768,131072")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.batch_sizes = [int(k) for k in args.batch_sizes.split(",")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for path in args.ckpt:
        with tempfile.TemporaryDirectory(prefix="grad_snr_", dir=args.out.parent) as directory:
            result = measure_checkpoint(path, args, directory)
            results.append(result)
        print_table(result)
    output = dict(fit_model="cos(K) = K / (K + c), c >= 0",
                  fit_k="mean harmonic size 2*n_a*n_b/(n_a+n_b) of defined pairs; per-K means fitted equally",
                  uncertainty="SE over repeated episode-disjoint splits of the same collected data; pairs may reuse episodes",
                  subset_sampling="combat/build filter the same parent pair; advantages normalized globally over valid samples",
                  checkpoints=results)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
