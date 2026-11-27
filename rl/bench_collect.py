"""GPU collection profiling and training measurements; accepts rl.train flags after its own flags."""
import argparse
import json
import os
import statistics
import time
from collections import defaultdict

import torch

from . import train as tr
from .actors import ActorPool
from .config import parse_args
from .model import Policy
from .ppo import Rollout, ValueNorm


def profile(cfg):
    dev = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    model = Policy(cfg.model).to(dev).eval()
    pool = ActorPool(cfg)
    col = tr.Collector(cfg, pool, model, ValueNorm(), dev, tr.amp_context(cfg, dev))
    buf = Rollout(cfg.caps, cfg.rollout_steps, cfg.n_envs, dev)
    totals, counts = defaultdict(float), defaultdict(int)
    originals = []

    def wrap(obj, name, label):
        original = getattr(obj, name)
        originals.append((obj, name, original))

        def timed(*args, **kwargs):
            t = time.perf_counter()
            result = original(*args, **kwargs)
            if label in ("to_torch", "forward", "buf.add"):
                torch.cuda.current_stream(dev).synchronize()
            totals[label] += time.perf_counter() - t
            counts[label] += 1
            return result

        setattr(obj, name, timed)

    col.collect(buf)
    torch.cuda.synchronize(dev)
    captures = col.fast.captures if col.fast is not None else 0
    wrap(pool.shm, "gather", "gather")
    wrap(tr, "to_torch", "to_torch")
    if col.fast is not None:
        wrap(col.fast, "prepare", "to_torch")
        wrap(col.fast, "_capture", "graph_setup")
        wrap(type(col.fast), "__call__", "forward")
    else:
        wrap(model, "forward", "forward")
    wrap(torch.Tensor, "cpu", "cpu")
    wrap(buf, "add", "buf.add")
    wrap(pool, "act", "pool.act")
    try:
        _, stats = col.collect(buf)
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)
        pool.close()
        col.err_file.close()
    calls = stats["infer_calls"]
    stats.pop("dec_counts")
    return dict(stats=stats, graph_captures=col.fast.captures - captures if col.fast is not None else 0,
                warmup_captures=captures, segment_ms={k: 1000 * v / calls for k, v in totals.items()},
                segment_calls=dict(counts))


def measure_train(cfg):
    devices = [torch.device(cfg.device)]
    if cfg.infer_device and torch.device(cfg.infer_device) not in devices:
        devices.append(torch.device(cfg.infer_device))
    for dev in devices:
        torch.empty(0, device=dev)
        torch.cuda.reset_peak_memory_stats(dev)
    update_times = []
    original = tr.ppo_update

    def timed_update(*args, **kwargs):
        t = time.perf_counter()
        result = original(*args, **kwargs)
        update_times.append(time.perf_counter() - t)
        return result

    tr.ppo_update = timed_update
    t = time.perf_counter()
    try:
        state = tr.train(cfg)
    finally:
        tr.ppo_update = original
    return dict(state=state, elapsed=time.perf_counter() - t, ppo_wall_times=update_times,
                memory={str(dev): dict(allocated_GiB=torch.cuda.max_memory_allocated(dev) / 2**30,
                                       reserved_GiB=torch.cuda.max_memory_reserved(dev) / 2**30)
                        for dev in devices})


def compare(cfg):
    from .bench import make_obs
    from .collect_fast import FastPolicy

    dev = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    model = Policy(cfg.model).to(dev).eval().requires_grad_(False)
    arrays = make_obs(224, cfg.caps, inflate=2)
    rows = []
    for amp in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = amp
        cfg.amp = amp
        fast = FastPolicy(model, cfg.caps, 224, dev, amp)
        for revision in range(2):
            if revision:
                with torch.no_grad():
                    model.value[-1].bias.add_(0.125)
                    model.logit[-1].weight.add_(0.001)
            for b in (72, 99, 128, 224):
                t = min((max(int(arrays["n_tok"][:b].max()), 1) + 7) // 8 * 8, cfg.caps.max_tok)
                c = min((max(int(arrays["n_cand"][:b].max()), 1) + 7) // 8 * 8, cfg.caps.max_cand)
                batch = {k: v[:b, :t] if k.startswith("tok_") else
                         v[:b, :c] if k.startswith("cand_") else v[:b] for k, v in arrays.items()}
                eager = tr.to_torch(batch, dev)
                staged = fast.prepare(batch)
                rng = torch.cuda.get_rng_state(dev)
                ctx = tr.amp_context(cfg, dev)
                ref = tr.act_batch(model, eager, ctx, return_logits=True)
                torch.cuda.set_rng_state(rng, dev)
                out = tr.act_batch(fast, staged, ctx, return_logits=True)
                rows.append(dict(amp=amp, revision=revision, batch=b,
                                 obs_equal=all(torch.equal(eager[k], staged[k]) for k in eager),
                                 actions_equal=torch.equal(ref[0], out[0]),
                                 logp_max_abs=float((ref[1] - out[1]).abs().max()),
                                 value_max_abs=float((ref[2] - out[2]).abs().max()),
                                 logits_max_abs=float((ref[3] - out[3]).abs().max())))
                if revision == 0:
                    timing_state = torch.cuda.get_rng_state(dev)
                    for name, policy, prepare in (("eager", model, lambda x: tr.to_torch(x, dev)),
                                                   ("fast", fast, fast.prepare)):
                        times = []
                        for _ in range(12):
                            start = time.perf_counter()
                            result = tr.act_batch(policy, prepare(batch), ctx)
                            for value in result:
                                value.cpu()
                            times.append(1000 * (time.perf_counter() - start))
                        rows[-1][name + "_ms"] = statistics.median(times)
                    torch.cuda.set_rng_state(timing_state, dev)
        del fast
    for row in rows:
        assert row["obs_equal"] and row["actions_equal"], row
        tolerance = 1e-2 if row["amp"] else 1e-6
        assert max(row[k] for k in ("logp_max_abs", "value_max_abs", "logits_max_abs")) <= tolerance, row
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("profile", "train", "compare"), required=True)
    ap.add_argument("--output", required=True)
    args, rest = ap.parse_known_args()
    cfg = parse_args(rest)
    os.makedirs(cfg.resolved_run_dir(), exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    result = {"profile": profile, "train": measure_train, "compare": compare}[args.mode](cfg)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
