"""Offline critic targets, calibration and frozen-policy gradient alignment.

    python tools/critic_lab.py collect --ckpt X.pt --steps 4194304 --out data/critic
    python tools/critic_lab.py fit --data data/critic --name mc --target mc
    python tools/critic_lab.py snr --data data/critic --values behaviour fit_mc --out snr.json
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.actors import ActorPool
from rl.ppo import Rollout, bucketed_minibatches, sliced
from rl.train import amp_context, load_policy
from tools.grad_snr import (EpisodeCollector, collect_rollouts, episode_pair, gradient_cosines,
                            gradient_products, number, policy_gradient, store_observations)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n")


def collect(args):
    device = torch.device(args.device)
    model, vnorm, cfg, ck = load_policy(args.ckpt, "cpu")
    update, env_steps = ck["update"], ck["env_steps"]
    del ck
    if cfg.group_k != 1 or cfg.branch_decisions:
        raise ValueError("critic_lab requires group_k=1 and empty branch_decisions")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "meta.npz").exists():
        raise ValueError(f"dataset already exists: {directory}")
    model.to(device)
    cfg.engine, cfg.engine_cmd = args.engine, ""
    cfg.asc_curriculum, cfg.ascension = False, args.ascension
    cfg.midrun_frac, cfg.engine_log = 0.0, False
    for key in ("engine_dir", "n_workers", "envs_per_worker", "restart_every_n"):
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    torch.manual_seed(cfg.seed)
    fields = "act logp val rew done trunc forced combat".split()
    parts = {key: [] for key in fields + ["decision", "env", "episode", "chunk", "adv_prod", "ret_prod", "mask_prod"]}
    boots, bounds = [], []
    pool = ActorPool(cfg, resume_steps=env_steps)
    col = EpisodeCollector(cfg, pool, model, vnorm, device, amp_context(cfg, device), err_log=False)
    steps = math.ceil(args.steps / cfg.rollout_steps) * cfg.rollout_steps
    try:
        for start, buf, boot in collect_rollouts(col, cfg, steps):
            i = len(bounds)
            if cfg.lam_build is None:
                adv, ret, mask = buf.gae(boot, cfg.gamma, cfg.lam)
            else:
                adv, ret, mask, _ = buf.gae_build(boot, cfg.gamma, cfg.lam, cfg.lam_build)
            obs = store_observations(buf, directory, f"chunk_{i}")
            observation_keys = list(obs)
            for key in fields:
                parts[key].append(getattr(buf, key)[:buf.n].copy())
            env = np.empty(buf.n, np.int64)
            for e, ix in enumerate(buf.env_idx):
                env[ix] = e
            for key, value in dict(decision=obs["decision"].numpy().copy(), env=env,
                                   episode=np.concatenate(buf.episode_chunks), chunk=np.full(buf.n, i, np.int64),
                                   adv_prod=adv, ret_prod=ret, mask_prod=mask).items():
                parts[key].append(value)
            boots.append(boot)
            bounds.append((start, start + buf.n))
            print(f"  collected {start + buf.n}/{steps} transitions", flush=True)
            del obs, buf
    finally:
        pool.close()
    np.savez(directory / "meta.npz", **{key: np.concatenate(value) for key, value in parts.items()},
             boot=np.stack(boots), chunk_bounds=np.asarray(bounds), gamma=np.asarray(cfg.gamma))
    mean, std = vnorm.mean_std
    info = dict(checkpoint=str(Path(args.ckpt).resolve()), update=update, env_steps=env_steps,
                gamma=cfg.gamma, lam=cfg.lam, lam_build=cfg.lam_build, rollout_steps=cfg.rollout_steps,
                micro_batch=cfg.micro_batch, vnorm=dict(mean=mean, std=std), args=vars(args),
                engine_errors=dict(col.errors), observation_keys=observation_keys, steps_collected=bounds[-1][1])
    write_json(directory / "meta.json", info)
    return info


def load_dataset(directory):
    """Return mmap observation chunks and the flat transition metadata."""
    directory = Path(directory)
    info = json.loads((directory / "meta.json").read_text())
    with np.load(directory / "meta.npz") as data:
        meta = {key: data[key] for key in data.files}
    chunks = []
    for i, (start, stop) in enumerate(meta["chunk_bounds"]):
        obs = {key: torch.from_numpy(np.load(directory / f"chunk_{i}_{key}.npy", mmap_mode="c"))
               for key in info["observation_keys"]}
        chunks.append((int(start), int(stop), obs))
    return chunks, meta


def stitch(meta):
    """Link each env's timeline and compute complete-run Monte Carlo returns."""
    n = len(meta["rew"])
    nxt = np.full(n, -1, np.int64)
    complete = np.zeros(n, bool)
    returns, position = np.zeros(n, np.float64), np.zeros(n, np.float64)
    order = np.argsort(meta["env"], kind="stable")
    env_idx = np.split(order, np.flatnonzero(np.diff(meta["env"][order])) + 1) if n else []
    for ix in env_idx:
        a, b = ix[:-1], ix[1:]
        keep = (meta["episode"][a] == meta["episode"][b]) & ~meta["done"][a] & ~meta["trunc"][a]
        nxt[a[keep]] = b[keep]
    order = np.argsort(meta["episode"], kind="stable")
    episodes = np.split(order, np.flatnonzero(np.diff(meta["episode"][order])) + 1) if n else []
    gamma = float(meta["gamma"])
    for ix in episodes:
        position[ix] = np.arange(len(ix)) / max(len(ix) - 1, 1)
        if meta["done"][ix[-1]] and not meta["trunc"][ix].any():
            complete[ix] = True
            g = 0.0
            for i in ix[::-1]:
                g = float(meta["rew"][i]) + gamma * g
                returns[i] = g
    return dict(next=nxt, complete=complete, G=returns, position=position, env_idx=env_idx)


class _ValueRollout(Rollout):
    """Scalar-only view using the production GAE recursion across full env streams."""

    def __init__(self, meta, stitched, values):
        self.n, self.env_idx = len(values), stitched["env_idx"]
        self.n_envs = len(self.env_idx)
        self.val = np.asarray(values, np.float32)
        for key in ("rew", "done", "trunc", "forced", "combat"):
            setattr(self, key, meta[key])
        self.cut = np.zeros(self.n, bool)
        self.bootstrap = np.zeros(self.n, np.float32)


def gae_stitched(meta, stitched, values, lam, lam_build=None):
    """Production GAE on complete episodes, without rollout boundary bootstraps."""
    if np.shape(values) != np.shape(meta["val"]):
        raise ValueError("values must have one entry per transition")
    if not len(values):
        return np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, bool)
    buf = _ValueRollout(meta, stitched, values)
    boot = np.zeros(buf.n_envs, np.float32)
    if lam_build is None:
        adv, ret, mask = buf.gae(boot, float(meta["gamma"]), lam)
    else:
        adv, ret, mask, _ = buf.gae_build(boot, float(meta["gamma"]), lam, lam_build)
    adv[~stitched["complete"]] = ret[~stitched["complete"]] = 0
    mask &= stitched["complete"]
    return adv, ret, mask


def dataset_split(directory, meta, stitched, train_frac=0.5, seed=0):
    path = Path(directory) / "split.npz"
    if path.exists():
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    ids = np.unique(meta["episode"][stitched["complete"]])
    if len(ids) < 3:
        raise ValueError("need at least three complete episodes for train, validation and measure")
    ids = np.random.default_rng(seed).permutation(ids)
    n_train = min(max(int(len(ids) * train_frac), 2), len(ids) - 1)
    n_val = max(1, round(n_train * 0.05))
    split = dict(train=ids[n_val:n_train], validation=ids[:n_val], measure=ids[n_train:],
                 seed=np.asarray(seed), train_frac=np.asarray(train_frac))
    np.savez(path, **split)
    return split


def valid_mask(meta, stitched):
    return stitched["complete"] & ~meta["trunc"] & ~meta["forced"]


@torch.no_grad()
def predict_values(model, chunks, indices, vnorm, cfg, device):
    model.eval()
    values = np.zeros(chunks[-1][1], np.float32)
    amp_ctx = amp_context(cfg, device)
    for start, stop, obs in chunks:
        local = indices[(indices >= start) & (indices < stop)] - start
        local = local[np.argsort(obs["n_tok"][local].numpy(), kind="stable")]
        for offset in range(0, len(local), cfg.micro_batch):
            ix = local[offset:offset + cfg.micro_batch]
            batch = {key: value.to(device) for key, value in sliced(obs, ix).items()}
            with amp_ctx():
                _, v = model(batch)
            v = v.float().cpu().numpy()
            values[start + ix] = vnorm.denormalize(v) if cfg.value_norm else v
    return values


def critic_metrics(meta, stitched, values, mask):
    g, v = stitched["G"], np.asarray(values, np.float64)
    def resid(keep):
        variance = float(np.var(g[keep])) if keep.any() else 0.0
        return float(np.mean((g[keep] - v[keep]) ** 2) / variance) if variance else None
    thirds = np.minimum((3 * stitched["position"]).astype(int), 2)
    subsets = dict(all=mask, combat=mask & meta["combat"], build=mask & ~meta["combat"])
    subsets.update({name: mask & (thirds == i) for i, name in enumerate(("early", "middle", "late"))})
    order = np.flatnonzero(mask)
    order = order[np.argsort(v[order], kind="stable")]
    calibration = [dict(n=len(ix), v_mean=float(v[ix].mean()) if len(ix) else None,
                        g_mean=float(g[ix].mean()) if len(ix) else None) for ix in np.array_split(order, 10)]
    nxt = stitched["next"]
    nv = np.zeros(len(v), np.float64)
    has_next = nxt >= 0
    nv[has_next] = v[nxt[has_next]]
    delta = meta["rew"] + float(meta["gamma"]) * nv * (1 - meta["done"]) - v
    terminal = meta["done"]
    td = {}
    for combat, label in ((False, "build"), (True, "combat")):
        for next_combat, next_label in ((False, "build"), (True, "combat")):
            keep = mask & has_next & ~terminal & (meta["combat"] == combat)
            keep &= meta["combat"][np.maximum(nxt, 0)] == next_combat
            td[f"{label}→{next_label}"] = float(np.mean(delta[keep] ** 2)) if keep.any() else None
    keep = mask & terminal
    td["terminal"] = float(np.mean(delta[keep] ** 2)) if keep.any() else None
    return dict(resid={key: resid(keep) for key, keep in subsets.items()}, calibration=calibration, td_mse=td)


def fit(args):
    if args.epochs < 1 or args.lr <= 0 or args.minibatch < 1:
        raise ValueError("epochs, lr and minibatch must be positive")
    directory, device = Path(args.data), torch.device(args.device)
    chunks, meta = load_dataset(directory)
    info = json.loads((directory / "meta.json").read_text())
    stitched = stitch(meta)
    split = dataset_split(directory, meta, stitched, args.train_frac, args.seed)
    mask = valid_mask(meta, stitched)
    train = mask & np.isin(meta["episode"], split["train"])
    validation = mask & np.isin(meta["episode"], split["validation"])
    measure = mask & np.isin(meta["episode"], split["measure"])
    train_all = np.flatnonzero(np.isin(meta["episode"], split["train"]))
    model, vnorm, cfg, ck = load_policy(info["checkpoint"], "cpu")
    del ck
    model.to(device)
    model.requires_grad_(True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
    steps_per_epoch = sum(math.ceil(int(train[start:stop].sum()) / args.minibatch) for start, stop, _ in chunks)
    total_steps = args.epochs * steps_per_epoch
    amp_ctx = amp_context(cfg, device)
    output = directory / f"fit_{args.name}"
    output.mkdir(parents=True, exist_ok=True)
    history, best, best_epoch, step = [], math.inf, None, 0
    for epoch in range(args.epochs):
        target = stitched["G"]
        if args.target == "lam":
            values = predict_values(model, chunks, train_all, vnorm, cfg, device)
            _, target, _ = gae_stitched(meta, stitched, values, info["lam"], info["lam_build"])
        target = vnorm.normalize(target) if cfg.value_norm else target
        model.train()
        loss_sum, n_train = 0.0, 0
        for j in rng.permutation(len(chunks)):
            start, stop, obs = chunks[j]
            local = np.flatnonzero(train[start:stop])
            for mb in bucketed_minibatches(obs["n_tok"].numpy(), local, args.minibatch, rng, 1):  # i.i.d. minibatches
                lr = args.lr * (0.55 + 0.45 * math.cos(math.pi * step / max(total_steps - 1, 1)))
                for group in opt.param_groups:
                    group["lr"] = lr
                opt.zero_grad(set_to_none=True)
                for offset in range(0, len(mb), cfg.micro_batch):
                    ix = mb[offset:offset + cfg.micro_batch]
                    batch = {key: value.to(device) for key, value in sliced(obs, ix).items()}
                    with amp_ctx():
                        _, v = model(batch)
                    y = torch.as_tensor(target[start + ix], device=device, dtype=torch.float32)
                    loss = 0.5 * (v.float() - y).square().mean()
                    (loss * (len(ix) / len(mb))).backward()
                    loss_sum += float(loss.detach()) * len(ix)
                    n_train += len(ix)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
        values = predict_values(model, chunks, np.flatnonzero(validation), vnorm, cfg, device)
        val_mse = float(np.mean((values[validation] - stitched["G"][validation]) ** 2))
        history.append(dict(epoch=epoch + 1, train_loss=loss_sum / n_train, val_mse=val_mse, lr=lr))
        if val_mse < best:
            best, best_epoch = val_mse, epoch + 1
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, output / "model.pt")
        print(f"  epoch={epoch + 1} loss={loss_sum / n_train:.6g} val_mse={val_mse:.6g}", flush=True)
    model.load_state_dict(torch.load(output / "model.pt", map_location="cpu", weights_only=True))
    values = predict_values(model, chunks, np.arange(len(mask)), vnorm, cfg, device)
    np.save(output / "values.npy", values)
    mean, std = vnorm.mean_std
    metrics = dict(name=args.name, target=args.target, best_epoch=best_epoch, epochs=history,
                   vnorm=dict(mean=mean, std=std), n_train=int(train.sum()), n_validation=int(validation.sum()),
                   n_measure=int(measure.sum()), behaviour=critic_metrics(meta, stitched, meta["val"], measure),
                   fitted=critic_metrics(meta, stitched, values, measure))
    write_json(output / "metrics.json", metrics)
    return metrics


def mean_se(values):
    values = [v for v in values if v is not None]
    return dict(mean=float(np.mean(values)) if values else None,
                se=float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None,
                n_pairs=len(values))


def alignment_stats(a, b, ref_a, ref_b, reference):
    own = gradient_products(a, b)
    cross_a, cross_b = gradient_products(a, ref_b), gradient_products(b, ref_a)
    cosine = gradient_cosines(a, b)
    groups = {}
    for group, p in own.items():
        na, nb = math.sqrt(p["norm_sq_a"]), math.sqrt(p["norm_sq_b"])
        ref_dot = reference[group]["dot"]
        da, db = cross_a[group]["dot"], cross_b[group]["dot"]
        align = 0.5 * (da / na + db / nb) / math.sqrt(ref_dot) if ref_dot > 0 and na and nb else None
        groups[group] = dict(self_cos=cosine[group], align=align, dot=p["dot"], norm_a=na, norm_b=nb,
                             dot_a_ref_b=da, dot_b_ref_a=db, ref_dot=ref_dot,
                             ref_norm_a=math.sqrt(reference[group]["norm_sq_a"]),
                             ref_norm_b=math.sqrt(reference[group]["norm_sq_b"]))
    return groups


def snr(args):
    if args.k < 1 or args.pairs < 1:
        raise ValueError("k and pairs must be positive")
    lams = [float(lam) for lam in args.lams.split(",")]
    if not lams or any(not 0 <= lam <= 1 for lam in lams):
        raise ValueError("lams must be between 0 and 1")
    directory, device = Path(args.data), torch.device(args.device)
    chunks, meta = load_dataset(directory)
    info = json.loads((directory / "meta.json").read_text())
    stitched = stitch(meta)
    split = dataset_split(directory, meta, stitched)
    valid = valid_mask(meta, stitched) & np.isin(meta["episode"], split["measure"])
    if len(np.unique(meta["episode"][valid])) < 2:
        raise ValueError("SNR needs at least two measure episodes with valid transitions")
    sources = list(dict.fromkeys(args.values))
    ref_source = next((source for source in sources if source != "behaviour"), "behaviour")
    specifications = dict.fromkeys((source, lam) for source in sources for lam in lams)
    specifications[("behaviour", info["lam"])] = None
    production = f"behaviour:{info['lam']:g}"
    estimators, advantages = {}, {}
    cache = {}
    for source, lam in [*specifications, (ref_source, 1.0)]:
        if source not in cache:
            path = directory / (source if source.startswith("fit_") else f"fit_{source}") / "values.npy"
            cache[source] = meta["val"] if source == "behaviour" else np.load(path, mmap_mode="r")
            if cache[source].shape != meta["val"].shape or not np.isfinite(cache[source]).all():
                raise ValueError(f"invalid values for {source}")
        key = f"{source}:{lam:g}"
        if key not in advantages:
            adv, _, _ = gae_stitched(meta, stitched, cache[source], lam, lam)
            normalized = np.zeros_like(adv)
            a = adv[valid]
            normalized[valid] = (a - a.mean()) / (a.std() + 1e-8)
            advantages[key] = normalized
        estimators[key] = dict(value=source, lam=lam, lam_build=lam, pairs=[])
    ref_key = f"{ref_source}:1"
    estimators["ref"] = estimators.pop(ref_key) if (ref_source, 1.0) not in specifications else dict(
        value=ref_source, lam=1.0, lam_build=1.0, pairs=[])
    advantages["ref"] = advantages[ref_key]
    model, vnorm, cfg, ck = load_policy(info["checkpoint"], "cpu")
    del ck, vnorm, cache
    model.to(device)
    rng, amp_ctx = np.random.default_rng(int(split["seed"])), amp_context(cfg, device)
    def gradient(indices, key):
        result = policy_gradient(model, chunks, indices, advantages[key], meta["act"], cfg.micro_batch, amp_ctx)
        model.zero_grad(set_to_none=True)
        return result
    for pair in range(args.pairs):
        a, b = episode_pair(meta["episode"], valid, args.k, rng)
        ref_a, ref_b = gradient(a, "ref"), gradient(b, "ref")
        reference = gradient_products(ref_a, ref_b)
        sampling = dict(pair=pair, n_a=len(a), n_b=len(b),
                        episode_ids_a=np.unique(meta["episode"][a]).tolist(),
                        episode_ids_b=np.unique(meta["episode"][b]).tolist())
        for key, estimator in estimators.items():
            if key == "ref" or advantages[key] is advantages["ref"]:
                groups = alignment_stats(ref_a, ref_b, ref_a, ref_b, reference)
            else:
                g_a, g_b = gradient(a, key), gradient(b, key)
                groups = alignment_stats(g_a, g_b, ref_a, ref_b, reference)
                del g_a, g_b
            estimator["pairs"].append(dict(**sampling, groups=groups))
        del ref_a, ref_b
        print(f"  pair={pair + 1}/{args.pairs} n={len(a)}/{len(b)}", flush=True)
    for estimator in estimators.values():
        estimator["summary"] = {group: {metric: mean_se([p["groups"][group][metric] for p in estimator["pairs"]])
                                          for metric in ("self_cos", "align")} for group in ("all", "no_table")}
    for key, estimator in estimators.items():
        for group, summary in estimator["summary"].items():
            baseline = estimators[production]["summary"][group]["align"]["mean"]
            align = summary["align"]["mean"]
            summary["gain"] = 1.0 if key == production else (align / baseline) ** 2 if baseline and align is not None else None
    output = dict(checkpoint=info["checkpoint"], production=production, reference="ref", k=args.k,
                  n_measure=int(valid.sum()), estimators=estimators)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, output)
    print(f"{'estimator':<28} {'params':<9} {'self_cos':>10} {'align':>10} {'SE':>10} {'gain':>10}")
    for group in ("all", "no_table"):
        ordered = sorted(estimators, key=lambda key: estimators[key]["summary"][group]["align"]["mean"]
                         if estimators[key]["summary"][group]["align"]["mean"] is not None else -math.inf, reverse=True)
        for key in ordered:
            row = estimators[key]["summary"][group]
            print(f"{key:<28} {group:<9} {number(row['self_cos']['mean']):>10} {number(row['align']['mean']):>10} "
                  f"{number(row['align']['se']):>10} {number(row['gain']):>10}")
    return output


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    cp = commands.add_parser("collect")
    cp.add_argument("--ckpt", required=True)
    cp.add_argument("--steps", type=int, default=4194304)
    cp.add_argument("--out", type=Path, required=True)
    cp.add_argument("--ascension", type=int, default=10)
    cp.add_argument("--engine", default="real", choices=("real", "fake", "fakeproc"))
    cp.add_argument("--engine_dir")
    cp.add_argument("--n_workers", type=int)
    cp.add_argument("--envs_per_worker", type=int)
    cp.add_argument("--restart_every_n", type=int)
    fp = commands.add_parser("fit")
    fp.add_argument("--data", type=Path, required=True)
    fp.add_argument("--name", required=True)
    fp.add_argument("--target", choices=("mc", "lam"), required=True)
    fp.add_argument("--lr", type=float, default=1e-4)
    fp.add_argument("--epochs", type=int, default=6)
    fp.add_argument("--train_frac", type=float, default=0.5)
    fp.add_argument("--minibatch", type=int, default=4096)
    fp.add_argument("--seed", type=int, default=0)
    sp = commands.add_parser("snr")
    sp.add_argument("--data", type=Path, required=True)
    sp.add_argument("--values", nargs="+", required=True)
    sp.add_argument("--lams", default="0.5,0.8,0.9,0.95,0.98")
    sp.add_argument("--k", type=int, default=524288)
    sp.add_argument("--pairs", type=int, default=6)
    sp.add_argument("--out", type=Path, required=True)
    for parser in (cp, fp, sp):
        parser.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    return dict(collect=collect, fit=fit, snr=snr)[args.command](args)


if __name__ == "__main__":
    main()
