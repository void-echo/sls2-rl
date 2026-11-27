"""Play a checkpoint on fixed seeds and record compact per-decision traces for behaviour analysis.

    python -m rl.trace --ckpt runs/x/ckpt/latest.pt --engine_dir ~/sls2-rl/engine --n_seeds 200 \
        --n_engines 32 --out runs/x/traces/pref.jsonl.gz [--sample] [--device cuda]

Engines are stepped in lockstep and the policy runs batched. Each line of the output is either a step record
(decision, run info, a compact view of the state, every candidate, the policy's probabilities and value, the
chosen index) or an episode record. Combat steps keep only what explains the choice (hand, energy, enemies),
non-combat steps keep the deck/relics/potions and the full candidate list. Seeds `pref_<i>` are disjoint from
training (`s<seed>_e<env>_<n>`) and evaluation (`eval_<i>`) seeds. See rl/analyze_prefs.py for the report.
With --behavior_obs, also save full obs for card_reward/rest_site/map_select/shop, the used potion's
descriptor, and terminal obs (including the full deck) for rl.behavior_metrics.
"""
import argparse
import gzip
import json
import os
import time

import numpy as np
import torch

from .encode import Encoder
from .engine import EngineProcess
from .force import force_choice, force_matches, parse_force_list
from .ppo import to_torch
from .train import amp_context, load_policy

TOPK = 8


def card_view(c):
    return {"id": c.get("id"), "up": c.get("up", 0), "type": c.get("type"), "cost": c.get("cost"), "rar": c.get("rar")}


def creature_view(e):
    v = {"id": e.get("id"), "hp": e.get("hp"), "max_hp": e.get("max_hp"), "block": e.get("block"),
         "pw": [(p.get("id"), p.get("n")) for p in e.get("pw") or []]}
    if "intents" in e:
        v["intents"] = [(i.get("t"), i.get("dmg"), i.get("hits")) for i in e.get("intents") or []]
    return v


def compact_state(R):
    obs = R.get("obs") or {}
    g = obs.get("g") or {}
    s = {"g": g, "relics": [r.get("id") for r in obs.get("relics") or []],
         "potions": [p.get("id") for p in obs.get("potions") or []]}
    cb = obs.get("combat")
    if cb:
        s["combat"] = {
            "round": cb.get("round"), "energy": cb.get("energy"), "max_energy": cb.get("max_energy"),
            "hand": [dict(card_view(c), playable=c.get("playable")) for c in cb.get("hand") or []],
            "n_draw": len(cb.get("draw") or []), "n_discard": len(cb.get("discard") or []),
            "n_exhaust": len(cb.get("exhaust") or []),
            "player": creature_view(cb.get("player") or {}),
            "enemies": [creature_view(e) for e in cb.get("enemies") or []],
        }
    else:
        s["deck"] = [card_view(c) for c in obs.get("deck") or []]
        s["ctx"] = obs.get("ctx")
        m = obs.get("map") or {}
        nodes = m.get("nodes") or []
        s["map_cur"] = nodes[m["cur"]] if isinstance(m.get("cur"), int) and 0 <= m["cur"] < len(nodes) else None
        s["map_nodes"] = nodes if R.get("decision") == "map_select" else None
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--engine_dir", default="~/sls2-rl/engine")
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--n_seeds", type=int, default=200)
    ap.add_argument("--seed_prefix", default="pref")
    ap.add_argument("--n_engines", type=int, default=32)
    ap.add_argument("--sample", action="store_true", help="sample from the policy instead of argmax")
    ap.add_argument("--behavior_obs", action="store_true", help="save observations needed by rl.behavior_metrics")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--restart_every_episode", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max_steps", type=int, default=0, help="abort an episode after this many decisions (0 = no limit)")
    ap.add_argument("--force_ids", default="", help="diagnostic: comma list of card ids taken whenever offered (card reward, shop)")
    ap.add_argument("--force_vars", default="", help="diagnostic: also force cards having any of these dynamic-var names")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    device = torch.device(a.device)
    model, _, cfg, ck = load_policy(a.ckpt, device)
    model.eval()
    amp_ctx = amp_context(cfg, device)
    enc = Encoder(cfg.caps)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = gzip.open(a.out, "wt")
    out.write(json.dumps({"type": "meta", "ckpt": os.path.abspath(a.ckpt), "update": ck.get("update"),
                          "character": a.character, "ascension": a.ascension, "sample": a.sample,
                          "n_seeds": a.n_seeds, "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")

    force_ids = parse_force_list(a.force_ids)
    force_vars = parse_force_list(a.force_vars)
    seeds = [f"{a.seed_prefix}_{i}" for i in range(a.n_seeds)]
    nxt = 0
    engines = [EngineProcess(engine_dir=a.engine_dir, name=f"trace{i}") for i in range(min(a.n_engines, len(seeds)))]
    slot = [None] * len(engines)  # (seed, step, R)
    started = [False] * len(engines)

    def start(i):
        nonlocal nxt
        if nxt >= len(seeds):
            slot[i] = None
            return
        seed = seeds[nxt]
        nxt += 1
        eng = engines[i]
        try:
            if a.restart_every_episode or not started[i]:
                eng.restart()
                ready = eng.recv(timeout=300)
                if ready.get("type") != "ready":
                    raise RuntimeError(f"engine did not boot: {ready}")
                started[i] = True
            R = eng.request({"cmd": "rl_start", "character": a.character, "ascension": a.ascension, "seed": seed},
                            timeout=300)
        except Exception as ex:  # noqa: BLE001
            R = {"type": "error", "message": f"start: {ex}"}
        if R.get("type") != "rl" or R.get("done"):
            out.write(json.dumps({"type": "episode", "seed": seed, "aborted": str(R.get("message", R))[:300]}) + "\n")
            return start(i)
        slot[i] = (seed, 0, R)

    for i in range(len(engines)):
        start(i)

    t0, n_steps, n_eps = time.time(), 0, 0
    while any(s is not None for s in slot):
        live = [i for i, s in enumerate(slot) if s is not None]
        batch = enc.alloc(len(live))
        for j, i in enumerate(live):
            o = enc.encode(slot[i][2])
            for k, v in batch.items():
                v[j] = o[k]
        with torch.no_grad(), amp_ctx():
            logits, value = model(to_torch(batch, device))
        logits = logits.float()
        if a.sample:
            u = torch.rand_like(logits).clamp_(1e-10, 1 - 1e-7)
            choice = (logits - torch.log(-torch.log(u))).argmax(-1)
        else:
            choice = logits.argmax(-1)
        probs = torch.softmax(logits, -1).cpu().numpy()
        choice = choice.cpu().numpy()
        value = value.float().cpu().numpy()
        sent, failed = [], {}
        for j, i in enumerate(live):
            seed, step, R = slot[i]
            nc = len(R["cands"])
            p = probs[j][:nc]
            k = int(choice[j]) if int(choice[j]) < nc else int(np.argmax(p))
            forced = False
            if force_ids or force_vars:
                best = force_choice(force_matches(R, force_ids, force_vars), probs[j])
                if best is not None:
                    forced, k = best != k, best
            top = np.argsort(-p)[:TOPK]
            rec = {"type": "step", "seed": seed, "step": step, "decision": R.get("decision"), "info": R.get("info"),
                   "state": compact_state(R), "cands": R["cands"], "probs": [round(float(x), 4) for x in p]
                   if nc <= 64 else {int(t): round(float(p[t]), 4) for t in top},
                   "choice": k, "value": round(float(value[j]), 4)}
            if forced:
                rec["forced"] = True
            if a.behavior_obs:
                if R.get("decision") in ("card_reward", "rest_site", "map_select", "shop"):
                    rec["obs"] = R.get("obs")
                if R["cands"][k].get("k") == "potion":
                    rec["potion"] = next((p for p in (R.get("obs") or {}).get("potions") or []
                                          if p.get("slot") == R["cands"][k].get("slot")), None)
            out.write(json.dumps(rec) + "\n")
            sent.append((i, seed, step))
            try:
                engines[i].send({"cmd": "rl_act", "i": k})
            except Exception as ex:  # noqa: BLE001
                failed[i] = f"act send: {ex}"
        # Engines compute concurrently between the send pass above and this receive pass.
        for i, seed, step in sent:
            if i in failed:
                R2 = {"type": "error", "message": failed.pop(i)}
            else:
                try:
                    R2 = engines[i].recv(timeout=a.timeout)
                except Exception as ex:  # noqa: BLE001
                    R2 = {"type": "error", "message": f"act: {ex}"}
            if R2.get("type") != "rl":
                started[i] = False
            n_steps += 1
            if R2.get("type") != "rl":
                out.write(json.dumps({"type": "episode", "seed": seed, "steps": step + 1,
                                      "aborted": str(R2.get("message", R2))[:300]}) + "\n")
                start(i)
            elif R2.get("done"):
                ep = {"type": "episode", "seed": seed, "steps": step + 1,
                      "victory": bool(R2.get("victory")), "info": R2.get("info"), "final": compact_state(R2)}
                if a.behavior_obs:
                    ep["obs"] = R2.get("obs")
                out.write(json.dumps(ep) + "\n")
                n_eps += 1
                start(i)
            elif a.max_steps and step + 1 >= a.max_steps:
                out.write(json.dumps({"type": "episode", "seed": seed, "steps": step + 1,
                                      "aborted": "max_steps", "info": R2.get("info")}) + "\n")
                started[i] = False
                start(i)
            else:
                slot[i] = (seed, step + 1, R2)
        if n_steps and n_steps % 5000 < len(live):
            print(f"{n_steps} steps, {n_eps} episodes, {n_steps / (time.time() - t0):.0f} steps/s", flush=True)
    out.close()
    for e in engines:
        e.close()
    print(f"done: {n_eps} episodes, {n_steps} steps -> {a.out}")


if __name__ == "__main__":
    main()
