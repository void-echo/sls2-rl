"""Find the questionable moments of one recorded run (parity/record.py --policy model) by asking the model itself.

Re-encodes every decision, computes the policy's probabilities and the (denormalized) value, then prints
  - the largest value changes across one decision (what the model itself counts as a big loss or gain),
  - ends of turn with energy left and playable cards, with the incoming attack damage,
  - the least confident choices and every non-combat choice below a confidence threshold.
It is a measurement only: nothing here feeds training.

    python tools/run_moments.py <run_dir> [--ckpt X] [--top 15] [--out moments.json]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from rl.encode import Encoder  # noqa: E402
from rl.ppo import to_torch  # noqa: E402
from rl.train import load_policy  # noqa: E402


def short(x):
    return (x or "?").split(".")[-1].replace("_", " ").title()


def describe(step, cand):
    """A readable name of a candidate in the context of its observation."""
    obs = step["obs"]
    k = cand.get("k")
    combat = obs.get("combat") or {}
    if k == "play":
        hand = combat.get("hand", [])
        i = cand.get("hand")
        name = short(hand[i]["id"]) + ("+" if hand[i].get("up") else "") if i is not None and i < len(hand) else "?"
        t = cand.get("tgt", -1)
        en = combat.get("enemies", [])
        return f"play {name}" + (f" -> {short(en[t]['id'])}" if 0 <= t < len(en) else "")
    if k == "potion":
        pots = obs.get("potions", [])
        slot = cand.get("slot")
        name = next((short(p["id"]) for p in pots if p.get("slot") == slot), "?")
        return f"drink {name}"
    if k == "discard_potion":
        pots = obs.get("potions", [])
        slot = cand.get("slot")
        name = next((short(p["id"]) for p in pots if p.get("slot") == slot), "?")
        return f"discard {name}"
    card = cand.get("card")
    if isinstance(card, dict):
        return f"{k} {short(card.get('id'))}" + ("+" if card.get("up") else "")
    for key in ("id", "key", "relic", "t", "type", "kind"):
        if cand.get(key) is not None:
            return f"{k} {short(str(cand[key]))}"
    return str(k)


def incoming(obs):
    total = 0
    for e in (obs.get("combat") or {}).get("enemies", []):
        if not e.get("alive", True):
            continue
        for it in e.get("intents") or []:
            if it.get("dmg"):
                total += it["dmg"] * (it.get("hits") or 1)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = a.run_dir.rstrip("/")
    run = json.load(open(f"{d}/run.json"))
    L = [json.loads(line) for line in open(f"{d}/headless.jsonl")]
    A = {x["step"]: x for x in map(json.loads, open(f"{d}/actions.jsonl"))}
    model, vn, cfg, _ = load_policy(a.ckpt or run["ckpt"], torch.device("cpu"))
    model.float().eval()
    torch.set_num_threads(max(os.cpu_count() // 2, 1))
    enc = Encoder(cfg.caps)
    rows = []
    for x in L:
        if x["decision"] == "game_over" or not x["cands"]:
            continue
        batch = enc.alloc(1)
        o = enc.encode({**x, "type": "rl", "done": False})
        for k in batch:
            batch[k][0] = o[k]
        with torch.no_grad():
            logits, value = model(to_torch(batch, torch.device("cpu")))
        n = len(x["cands"])
        p = torch.softmax(logits[0, :n].float(), -1).numpy()
        act = A.get(x["step"], {}).get("i", int(p.argmax()))
        v = float(vn.denormalize(value[0].float()))
        g = x["obs"]["g"]
        c = x["obs"].get("combat") or {}
        rows.append({"step": x["step"], "decision": x["decision"], "floor": g["total_floor"], "act": g["act"], "room": g["room"],
                     "hp": (c.get("player") or {}).get("hp", g["hp"]), "max_hp": g["max_hp"], "block": (c.get("player") or {}).get("block", 0),
                     "energy": c.get("energy"), "incoming": incoming(x["obs"]), "value": v, "p": float(p[act]), "pmax": float(p.max()),
                     "choice": describe(x, x["cands"][act]), "best": describe(x, x["cands"][int(p.argmax())]),
                     "alts": [(describe(x, x["cands"][int(i)]), round(float(p[i]), 3)) for i in np.argsort(-p)[:3]],
                     "playable": sum(1 for q in x["cands"] if q.get("k") == "play"),
                     "hand": [short(h["id"]) for h in c.get("hand", [])]})
    for r, nxt in zip(rows, rows[1:]):
        r["dv"] = nxt["value"] - r["value"]
    rows[-1]["dv"] = 0.0
    print(f"== {run['id']}: {run['end']}, {len(rows)} decisions; value start {rows[0]['value']:.3f}, last {rows[-1]['value']:.3f}")
    fmt = lambda r: (f"step {r['step']:3d} F{r['floor']:2d} {r['room'][:5]:5s} hp {r['hp']}/{r['max_hp']} block {r['block']} "
                     f"incoming {r['incoming']} V {r['value']:.3f} dV {r['dv']:+.3f} p {r['p']:.2f}: {r['choice']}")
    print(f"\n-- largest value drops across one decision (top {a.top})")
    for r in sorted(rows, key=lambda r: r["dv"])[:a.top]:
        print("  " + fmt(r))
    print(f"\n-- largest value gains (top {min(a.top, 8)})")
    for r in sorted(rows, key=lambda r: -r["dv"])[:min(a.top, 8)]:
        print("  " + fmt(r))
    ends = [r for r in rows if r["decision"] == "combat_play" and r["choice"] == "end_turn" and (r["energy"] or 0) > 0 and r["playable"] > 0]
    print(f"\n-- ended the turn with energy left and playable cards: {len(ends)} of "
          f"{sum(r['choice'] == 'end_turn' for r in rows)} ends of turn")
    for r in ends:
        print(f"  step {r['step']:3d} F{r['floor']:2d} {r['room'][:5]:5s} hp {r['hp']}/{r['max_hp']} block {r['block']} incoming {r['incoming']} energy {r['energy']} "
              f"p {r['p']:.2f}, {r['playable']} playable, hand {r['hand']}, next best {r['alts'][:2]}")
    low = [r for r in rows if r["p"] < 0.5]
    build = [r for r in low if r["decision"] != "combat_play"]
    print(f"\n-- choices with p < 0.5 (argmax recording, so the model was merely unsure): {len(low)}; outside combat: {len(build)}")
    for r in sorted(build, key=lambda r: r["p"])[:a.top]:
        print(f"  step {r['step']:3d} F{r['floor']:2d} {r['decision']:12s} p {r['p']:.2f} {r['alts']}")
    print("\n-- potions and discards")
    for r in rows:
        if r["choice"].startswith(("drink", "discard")):
            print("  " + fmt(r))
    if a.out:
        json.dump(rows, open(a.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
