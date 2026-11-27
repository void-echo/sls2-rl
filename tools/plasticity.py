"""Measure checkpoint plasticity on shared, uniformly sampled engine decisions.

    python tools/plasticity.py --collect runs/x/ckpt/latest.pt --engine_dir /abs/engine \
        --n_states 4096 --ascension 0,9 --out tmp/plasticity/states.pt
    python tools/plasticity.py --ckpts random a.pt b.pt --states tmp/plasticity/states.pt \
        --out tmp/plasticity/result.json --md tmp/plasticity/result.md --probe_steps 1000
"""
import argparse
from collections import Counter, defaultdict
import contextlib
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from rl.encode import Caps, Encoder, TT, field_spec  # noqa: E402
from rl.engine import EngineDead, make_engine  # noqa: E402
from rl.model import Policy  # noqa: E402
from rl.ppo import sliced, to_torch  # noqa: E402
from rl.train import act_batch, load_policy  # noqa: E402

THRESHOLDS = (0.025, 0.1)
GROUPS = ("all", "combat", "build")


def collect_states(args):
    model, _, cfg, ck = load_policy(args.collect, args.device)
    del ck
    model.float().eval()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    enc = Encoder(cfg.caps)
    reservoir, slot = enc.alloc(args.n_states), enc.alloc()
    combat = np.zeros(args.n_states, dtype=np.bool_)
    ascension = np.zeros(args.n_states, dtype=np.int16)
    levels = [int(x) for x in args.ascension.split(",")]
    engine = make_engine("real", engine_dir=args.engine_dir)
    seen, episodes = 0, []
    try:
        while seen < max(args.n_states, args.min_decisions) or len(episodes) % len(levels) or not episodes:
            level = levels[len(episodes) % len(levels)]
            seed = f"plasticity_{len(episodes)}"
            steps, aborted = 0, None
            try:
                if cfg.restart_every_episode:
                    engine.close()
                r = engine.reset(character="Ironclad", ascension=level, seed=seed, timeout=cfg.timeout_start)
                while True:
                    if r.get("type") != "rl":
                        raise EngineDead(str(r))
                    if r["done"]:
                        break
                    obs = enc.encode(r, slot)
                    seen += 1
                    j = seen - 1 if seen <= args.n_states else int(rng.integers(seen))
                    if j < args.n_states:
                        for k in field_spec(cfg.caps):
                            reservoir[k][j] = obs[k]
                        combat[j] = bool((obs["tok_type"] == TT["COMBAT"]).any())
                        ascension[j] = level
                    batch = sliced(to_torch({k: v[None] for k, v in obs.items()}, args.device), [0])
                    a, _, _ = act_batch(model, batch, contextlib.nullcontext, greedy=False)
                    r = engine.step(a.item(), timeout=cfg.timeout_step)
                    steps += 1
                    if cfg.max_episode_steps and steps >= cfg.max_episode_steps:
                        raise EngineDead("max_episode_steps")
            except EngineDead as ex:
                aborted = str(ex)
                engine.close()
            episodes.append({"seed": seed, "ascension": level, "steps": steps, "aborted": aborted})
            print(f"collected A{level}: {steps} decisions, {seen} total, {len(episodes)} episodes"
                  + (f", aborted: {aborted}" if aborted else ""), flush=True)
    finally:
        engine.close()
    result = {"obs": to_torch(reservoir, "cpu"), "combat": torch.from_numpy(combat),
              "ascension": torch.from_numpy(ascension), "caps": asdict(cfg.caps), "source": args.collect,
              "sampling": "uniform reservoir over all decisions, complete ascension cycles",
              "seed": 0, "n_seen": seen, "episodes": episodes, "truncations": dict(enc.trunc)}
    torch.save(result, args.out)
    print(f"saved {args.n_states} states: combat={int(combat.sum())}, build={int((~combat).sum())}, "
          f"ascension={dict(Counter(map(int, ascension)))} -> {args.out}", flush=True)


def weight_norms(model):
    squares = defaultdict(float)
    ln_names = {f"{name}.{pname}" for name, module in model.blocks.named_modules()
                if isinstance(module, torch.nn.LayerNorm) for pname, _ in module.named_parameters()}
    for name, p in model.named_parameters():
        v = float(p.detach().norm()) ** 2
        top = name.split(".")[0]
        squares[top] += v
        squares["total"] += v
        if top != "table":
            squares["non_embedding"] += v
        if top == "blocks":
            kind = "layernorm" if name[len("blocks."):] in ln_names else "non_layernorm"
            squares[f"blocks_{kind}"] += v
    return {k: math.sqrt(v) for k, v in squares.items()}


def dormant_stats(mean_abs, n_samples):
    mean_abs = mean_abs.detach().float().cpu()
    scores = mean_abs / mean_abs.mean().clamp(min=torch.finfo(torch.float32).tiny)
    return {"n_samples": n_samples, "n_units": len(scores), "mean_abs": mean_abs.tolist(),
            "scores": scores.tolist(),
            "fraction": {str(t): float((scores <= t).float().mean()) for t in THRESHOLDS}}


def feature_rank(features, delta=0.01):
    x = features.detach().float().cpu()
    n, d = x.shape
    if not n:
        return {"n_samples": 0, "width": d, "max_rank": 0, "srank": None, "entropy_rank": None,
                "delta": delta, "centered": False}
    singular = torch.linalg.svdvals(x)
    total = singular.sum()
    if total == 0:
        srank, erank = 0, 0.0
    else:
        p = singular / total
        srank = min(int(torch.searchsorted(p.cumsum(0), 1 - delta)) + 1, len(p))
        erank = float(torch.exp(-(p * p.clamp(min=torch.finfo(p.dtype).tiny).log()).sum()))
    return {"n_samples": n, "width": d, "max_rank": min(n, d), "srank": srank,
            "entropy_rank": erank, "delta": delta, "centered": False}


class StateBatches:
    def __init__(self, obs, device):
        self.n_tok, self.n_cand = obs["n_tok"].long(), obs["n_cand"].long()
        self.n = len(self.n_tok)
        self.device = device
        self.obs = {k: v.to(device) for k, v in obs.items()}
        self.tokens = {k: v for k, v in self.obs.items() if k.startswith("tok_") or k == "n_tok"}

    def get(self, idx, value_only=False):
        t = int(self.n_tok[idx].max())
        c = 1 if value_only else int(self.n_cand[idx].max())
        return sliced(self.tokens if value_only else self.obs, idx.to(self.device), T=t, C=c)

    def indices(self, batch_size):
        return torch.arange(self.n).split(batch_size)


class Diagnostics:
    def __init__(self, model, states, combat):
        self.model, self.states, self.combat = model, states, combat.bool()
        self.abs_sums = {g: {} for g in GROUPS}
        self.counts = {g: {} for g in GROUPS}
        self.features = {g: defaultdict(list) for g in GROUPS}
        self.attention, self.residual = {}, {}
        self.handles = []
        self.candidate_offset = 0
        sizes = states.n_cand
        owner = torch.repeat_interleave(torch.arange(states.n), sizes)
        rng = torch.Generator().manual_seed(0)
        self.sample_ids = {}
        for g in GROUPS:
            eligible = torch.arange(len(owner))
            if g != "all":
                eligible = eligible[self.combat[owner] == (g == "combat")]
            ids = eligible[torch.randperm(len(eligible), generator=rng)[:8192]]
            self.sample_ids[g] = ids.sort().values
        for i, block in enumerate(model.blocks):
            self.add(block.mlp[1], self.activation(f"blocks.{i}.mlp", "token"))
            self.add(block.attn, self.attention_hook(str(i)))
            self.add(block, self.residual_hook(str(i)))
        for i, block in enumerate(model.cand_blocks):
            self.add(block.mlp[1], self.activation(f"cand_blocks.{i}.mlp", "candidate"))
        self.add(model.logit[2], self.activation("logit", "candidate"))
        self.add(model.value[2], self.activation("value", "state"))
        self.add(model.ln_f, self.global_hook)
        self.handles.append(model.logit.register_forward_pre_hook(self.candidate_hook))

    def add(self, module, hook):
        self.handles.append(module.register_forward_hook(hook))

    def activation(self, name, kind):
        def hook(module, inputs, output):
            x = output.detach().float()
            if kind == "state":
                sums, counts = x.abs(), torch.ones(len(x), device=x.device, dtype=torch.long)
            else:
                mask = self.tmask if kind == "token" else self.cmask
                sums, counts = (x.abs() * mask[..., None]).sum(1), mask.sum(1)
            for g, rows in self.rows.items():
                if not rows.any():
                    continue
                total, count = sums[rows].sum(0), counts[rows].sum()
                self.abs_sums[g][name] = self.abs_sums[g].get(name, 0) + total
                self.counts[g][name] = self.counts[g].get(name, 0) + count
                if name == "value":
                    self.features[g]["value_hidden"].append(x[rows].cpu())
        return hook

    def global_hook(self, module, inputs, output):
        for g, rows in self.rows.items():
            self.features[g]["g"].append(output[rows, 0].detach().cpu())

    def candidate_hook(self, module, inputs):
        x = inputs[0].detach()[self.cmask]
        lo, hi = self.candidate_offset, self.candidate_offset + len(x)
        for g, ids in self.sample_ids.items():
            chosen = ids[(ids >= lo) & (ids < hi)] - lo
            self.features[g]["candidates"].append(x[chosen.to(x.device)].cpu())

    def attention_hook(self, name):
        def hook(module, inputs, output):
            x, mem, mask = inputs
            b, q, d = x.shape
            k = mem.shape[1]
            query = module.q(x).view(b, q, module.h, -1).transpose(1, 2)
            key = module.kv(mem).view(b, k, 2, module.h, -1)[:, :, 0].transpose(1, 2)
            logits = (query @ key.transpose(-2, -1)) / math.sqrt(d // module.h)
            logp = logits.masked_fill(~mask[:, None, None, :], -torch.inf).log_softmax(-1)
            entropy = -(logp.exp() * logp.masked_fill(~mask[:, None, None, :], 0)).sum(-1)
            norm = mask.sum(-1).float().log()
            entropy = torch.where(norm[:, None, None] > 0,
                                  entropy / norm[:, None, None].clamp(min=1e-12), 0)
            total = (entropy * self.tmask[:, None]).sum()
            count = self.tmask.sum() * module.h
            prev = self.attention.get(name, (0, 0))
            self.attention[name] = (prev[0] + total, prev[1] + count)
        return hook

    def residual_hook(self, name):
        def hook(module, inputs, output):
            total = (output.detach().norm(dim=-1) * self.tmask).sum()
            prev = self.residual.get(name, (0, 0))
            self.residual[name] = (prev[0] + total, prev[1] + self.tmask.sum())
        return hook

    @torch.no_grad()
    def run(self, batch_size):
        entropies, values = [], []
        try:
            for idx in self.states.indices(batch_size):
                b = self.states.get(idx)
                dev = b["n_tok"].device
                self.tmask = torch.arange(b["tok_type"].shape[1], device=dev)[None] < b["n_tok"][:, None]
                self.cmask = torch.arange(b["cand_key"].shape[1], device=dev)[None] < b["n_cand"][:, None]
                combat = self.combat[idx].to(dev)
                self.rows = {"all": torch.ones_like(combat), "combat": combat, "build": ~combat}
                logits, value = self.model(b)
                lp = logits.log_softmax(-1)
                entropies.append((-(lp.exp() * lp).sum(-1)).cpu())
                values.append(value.cpu())
                self.candidate_offset += int(b["n_cand"].sum())
        finally:
            for handle in self.handles:
                handle.remove()
        entropies, values = torch.cat(entropies), torch.cat(values)
        by_group = {}
        for g in GROUPS:
            rows = torch.ones_like(self.combat) if g == "all" else self.combat == (g == "combat")
            layers = {name: dormant_stats(total / self.counts[g][name], int(self.counts[g][name]))
                      for name, total in self.abs_sums[g].items()}
            means, by_kind = {}, {}
            for t in map(str, THRESHOLDS):
                means[t] = float(np.mean([v["fraction"][t] for v in layers.values()])) if layers else None
            for kind in ("blocks", "cand_blocks", "logit", "value"):
                items = [v for k, v in layers.items() if k.split(".")[0] == kind]
                by_kind[kind] = {str(t): float(np.mean([v["fraction"][str(t)] for v in items]))
                                 if items else None for t in THRESHOLDS}
            ranks = {name: feature_rank(torch.cat(self.features[g][name]) if self.features[g][name]
                                        else torch.empty(0, self.model.cfg.d))
                     for name in ("g", "value_hidden", "candidates")}
            by_group[g] = {"n_states": int(rows.sum()),
                           "dormant": {"layers": layers, "mean": means, "mean_by_kind": by_kind},
                           "features": ranks, "policy_entropy": float(entropies[rows].mean()) if rows.any() else None,
                           "value_std": float(values[rows].std(correction=0)) if rows.any() else None}
        out = by_group.pop("all")
        out["by_decision_type"] = by_group
        out["attention_entropy"] = {k: float(v / n) for k, (v, n) in self.attention.items()}
        out["residual_norm"] = {k: float(v / n) for k, (v, n) in self.residual.items()}
        out["attention_entropy_mean"] = float(np.mean(list(out["attention_entropy"].values()))) if self.attention else None
        return out


def value_forward(model, b):
    tower = model.cfg.value_tower
    type_emb = model.value_type_emb if tower else model.type_emb
    tok_in = model.value_tok_in if tower else model.tok_in
    blocks = model.value_blocks if tower else model.blocks
    ln_f = model.value_ln_f if tower else model.ln_f
    x = type_emb(b["tok_type"].long()) + tok_in(model.bag(b["tok_key"], b["tok_val"]))
    mask = torch.arange(x.shape[1], device=x.device)[None] < b["n_tok"][:, None]
    for block in blocks:
        x = block(x, mask)
    h = ln_f(x)
    pooled = (h * mask[..., None]).sum(1) / mask.sum(1)[:, None]
    return model.value(torch.cat((h[:, 0], pooled), -1)).squeeze(-1).float()


@torch.no_grad()
def predict_values(model, states, batch_size):
    return torch.cat([value_forward(model, states.get(idx, value_only=True)).cpu()
                      for idx in states.indices(batch_size)])


def random_policy(cfg, seed, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Policy(cfg).to(device)
    return model.float().eval()


def probe_targets(cfg, states, batch_size):
    teacher = random_policy(cfg, 12345, states.device)
    smooth = predict_values(teacher, states, batch_size)
    smooth = (smooth - smooth.mean()) / smooth.std(correction=0).clamp(min=1e-12)
    memory = torch.randn(states.n, generator=torch.Generator().manual_seed(67890))
    return {"smooth": smooth, "memory": memory}


def probe_order(n, steps):
    rng = torch.Generator().manual_seed(0)
    order = []
    while len(order) < steps:
        order.extend(torch.randperm(n, generator=rng).split(256))
    return order[:steps]


def run_probes(model, states, targets, steps, batch_size):
    out = {"steps": steps, "batch_size": 256, "lr": 1e-4, "eps": 1e-5, "weight_decay": 0,
           "order_seed": 0, "smooth_seed": 12345, "memory_seed": 67890}
    order = probe_order(states.n, steps)
    checkpoints = {0, steps} | {s for s in (100, 300) if s <= steps}
    for name, target in targets.items():
        probe = copy.deepcopy(model).eval().requires_grad_(True)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-4, eps=1e-5, weight_decay=0)
        curve = {}
        for step in range(steps + 1):
            if step in checkpoints:
                prediction = predict_values(probe, states, batch_size)
                curve[str(step)] = float(F.mse_loss(prediction, target))
                print(f"  {name} step {step}: full MSE={curve[str(step)]:.6f}", flush=True)
            if step == steps:
                break
            idx = order[step]
            opt.zero_grad(set_to_none=True)
            prediction = value_forward(probe, states.get(idx, value_only=True))
            loss = F.mse_loss(prediction, target[idx].to(states.device))
            loss.backward()
            opt.step()
        out[name] = {"mse": curve, "final_mse": curve[str(steps)],
                     "target_mean": float(target.mean()), "target_std": float(target.std(correction=0))}
        del probe, opt
    return out


def markdown(results):
    lines = ["| Checkpoint | Non-table norm | Blocks (all / non-LN) | Dormant ≤.025 / ≤.1 | "
             "Block / cand / logit / value ≤.1 | g / candidate srank | Attn H | Smooth MSE | Memory MSE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        name = Path(r["checkpoint"]).name
        w = r["weight_norms"]
        prefix = f"| {name} | {w['non_embedding']:.2f} | {w.get('blocks', 0):.2f} / {w.get('blocks_non_layernorm', 0):.2f} |"
        if r["status"] == "skipped":
            lines.append(prefix + f" skipped ({r['reason']}) | — | — | — | — | — |")
            continue
        d, f = r["dormant"], r["features"]
        kinds = " / ".join(f"{d['mean_by_kind'][k]['0.1']:.1%}"
                           if d['mean_by_kind'][k]['0.1'] is not None else "—"
                           for k in ("blocks", "cand_blocks", "logit", "value"))
        attn = r["attention_entropy_mean"]
        lines.append(prefix + f" {d['mean']['0.025']:.1%} / {d['mean']['0.1']:.1%} | {kinds} | "
                     f"{f['g']['srank']} / {f['candidates']['srank']} | "
                     + (f"{attn:.3f}" if attn is not None else "—")
                     + f" | {r['probe']['smooth']['final_mse']:.5f} | {r['probe']['memory']['final_mse']:.5f} |")
    return "\n".join(lines) + "\n"


def diagnose(args):
    data = torch.load(args.states, map_location="cpu", weights_only=False)
    states = StateBatches(data["obs"], args.device)
    state_caps = Caps.from_dict(data["caps"])
    first = next((p for p in args.ckpts if p != "random"), None)
    if first is None:
        raise ValueError("random needs a real checkpoint for its model configuration")
    base_model, _, base_cfg, ck = load_policy(first, "cpu")
    del base_model, ck
    targets, results = {}, []
    for path in args.ckpts:
        started = time.monotonic()
        if path == "random":
            cfg = replace(base_cfg, caps=state_caps)
            model = random_policy(cfg.model, 0, args.device)
            update = None
        else:
            model, _, cfg, ck = load_policy(path, args.device)
            update = ck.get("update")
            del ck
        model.float().eval()
        row = {"checkpoint": path, "update": update, "model_cfg": asdict(cfg.model), "caps": asdict(cfg.caps),
               "weight_norms": weight_norms(model), "status": "ok"}
        if path == "random":
            row["config_source"] = first
            row["encoding_source"] = "states (random initialization has no training encoding)"
        print(f"{path}: blocks={row['weight_norms'].get('blocks', 0):.3f}", flush=True)
        if cfg.caps != state_caps:
            different = {k: {"checkpoint": v, "states": asdict(state_caps)[k]}
                         for k, v in asdict(cfg.caps).items() if v != asdict(state_caps)[k]}
            row.update(status="skipped", reason="caps mismatch: " + ", ".join(different), caps_diff=different)
            print(f"  skipped: {row['reason']}", flush=True)
        else:
            row.update(Diagnostics(model, states, data["combat"]).run(args.batch_size))
            key = json.dumps(asdict(cfg.model), sort_keys=True)
            if key not in targets:
                targets[key] = probe_targets(cfg.model, states, args.batch_size)
            row["probe"] = run_probes(model, states, targets[key], args.probe_steps, args.batch_size)
        row["seconds"] = time.monotonic() - started
        results.append(row)
        del model
        output = {"states": args.states, "n_states": states.n, "caps": asdict(state_caps),
                  "decision_counts": {"combat": int(data["combat"].sum()), "build": int((~data["combat"]).sum())},
                  "ascension_counts": dict(Counter(map(int, data["ascension"]))),
                  "definitions": {"features": "uncentered activation matrices; singular values, not squared values",
                                  "dormant_mean": "unweighted mean of per-layer fractions",
                                  "attention": "valid queries, keys and heads; entropy / log(n_keys), zero for one key",
                                  "value": "raw Policy output, without ValueNorm denormalization",
                                  "probe": "value path only; all parameters trainable; fp32, fresh optimizer per target"},
                  "checkpoints": results}
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2, allow_nan=False)
        if args.md:
            Path(args.md).write_text(markdown(results))
        print(f"  finished in {row['seconds']:.1f}s", flush=True)
    print(markdown(results), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collect")
    mode.add_argument("--ckpts", nargs="+")
    ap.add_argument("--engine_dir")
    ap.add_argument("--n_states", type=int, default=4096)
    ap.add_argument("--min_decisions", type=int, default=0,
                    help="keep playing until this many decisions were seen (the reservoir thins them to n_states)")
    ap.add_argument("--ascension", default="0")
    ap.add_argument("--states")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    ap.add_argument("--probe_steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    if args.n_states <= 0 or args.batch_size <= 0 or args.probe_steps < 0 or args.threads <= 0:
        ap.error("n_states, batch_size and threads must be positive; probe_steps must be nonnegative")
    if args.collect and (not args.engine_dir or not Path(args.engine_dir).is_absolute()):
        ap.error("--collect requires an absolute --engine_dir")
    if args.ckpts and not args.states:
        ap.error("--ckpts requires --states")
    torch.set_num_threads(args.threads)
    if torch.device(args.device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    for path in (args.out, args.md):
        if path:
            Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
    if args.collect:
        collect_states(args)
    else:
        diagnose(args)


if __name__ == "__main__":
    main()
