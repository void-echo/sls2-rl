"""Is the value function consistent across room boundaries and calibrated by stage? Input: a sampled trace (rl.trace --sample).
For a well-fitted V, the mean TD error r + V(s') - V(s) is zero on every kind of transition; a systematic jump when entering a room
or leaving combat means the out-of-combat and in-combat observations disagree. Also bins V against the realised return to go.

    python tools/td_boundaries.py runs/diag/td_a10_kl1.jsonl.gz --ckpt plast/ckpts/x.pt   # the checkpoint supplies the value scale
"""
import argparse
import collections
import gzip
import json
import statistics as st

import torch

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--floor_reward", type=float, default=0.01)
ap.add_argument("--boss_reward", type=float, default=0.2)
a = ap.parse_args()
vn = torch.load(a.ckpt, map_location="cpu", weights_only=False)["vnorm"]
m = vn["m"] / vn["w"]
s = (max(vn["m2"] / vn["w"] - m * m, 1e-4)) ** 0.5
print(f"value scale: mean {m:.4f} std {s:.4f}")
games, final = collections.defaultdict(list), {}
for line in gzip.open(a.trace, "rt"):
    d = json.loads(line)
    if d.get("type") == "step":
        g = d["state"].get("g") or {}
        games[d["seed"]].append(dict(step=d["step"], v=d["value"] * s + m, dec=d["decision"], combat="combat" in d["state"],
                                     room=g.get("room"), floor=(d["info"] or {}).get("total_floor") or 0, act=(d["info"] or {}).get("act") or 0,
                                     hp=g.get("hp"), max_hp=g.get("max_hp"), pots=len(d["state"].get("potions") or [])))
    elif d.get("type") == "episode":
        final[d["seed"]] = d
print("games", len(games), "with final", len(final), "sample final", {k: v for k, v in next(iter(final.values())).items() if k not in ("deck", "relics", "obs")})
td = collections.defaultdict(list)
cal = collections.defaultdict(list)
for seed, S in games.items():
    S.sort(key=lambda x: x["step"])
    e = final.get(seed)
    if not e or e.get("aborted"):
        continue
    win = bool(e.get("victory"))
    rets = [0.0] * len(S)
    G = 1.0 if win else 0.0
    for i in range(len(S) - 1, -1, -1):
        if i + 1 < len(S):
            r = a.floor_reward * max(S[i + 1]["floor"] - S[i]["floor"], 0) + a.boss_reward * max(S[i + 1]["act"] - S[i]["act"], 0)
            G += r
        rets[i] = G
    for i, x in enumerate(S):
        if i + 1 < len(S):
            y = S[i + 1]
            r = a.floor_reward * max(y["floor"] - x["floor"], 0) + a.boss_reward * max(y["act"] - x["act"], 0)
            d = r + y["v"] - x["v"]
            if x["combat"] and y["combat"]:
                kind = f"in combat {y['room']}"
            elif not x["combat"] and y["combat"]:
                kind = f"enter {y['room']}"
            elif x["combat"] and not y["combat"]:
                kind = f"leave {x['room']}"
            else:
                kind = f"build {x['dec']}"
        else:
            d = (1.0 if win else 0.0) - x["v"]
            kind = f"terminal {'win' if win else 'death'} {x['room']}"
        td[kind].append(d)
        if x["combat"] and (i == 0 or not S[i - 1]["combat"]):  # first decision of a fight
            stage = f"a{x['act']} {x['room']}" if x["room"] != "Boss" or x["floor"] < 48 else f"boss{x['floor'] - 47} (floor {x['floor']})"
            cal[stage].append((x["v"], rets[i], win, (x["hp"] or 0) / max(x["max_hp"] or 1, 1), x["pots"]))
print("\n== mean TD error by transition (n, mean, stderr)")
for k, v in sorted(td.items(), key=lambda kv: -len(kv[1])):
    if len(v) >= 20:
        print(f"  {k:32s} n={len(v):7d} mean {st.mean(v):+.4f} se {st.stdev(v) / len(v) ** .5:.4f} sd {st.stdev(v):.3f}")
print("\n== value at the first decision of a fight vs realised return to go")
for k, v in sorted(cal.items()):
    if len(v) >= 20:
        print(f"  {k:28s} n={len(v):5d} V {st.mean(x[0] for x in v):.3f} G {st.mean(x[1] for x in v):.3f} win {st.mean(x[2] for x in v):.3f} hp {st.mean(x[3] for x in v):.2f} pots {st.mean(x[4] for x in v):.2f}")
print("\n== calibration of V in bins, bosses of act 3")
for k in sorted(cal):
    if k.startswith("boss"):
        v = sorted(cal[k])
        n = max(len(v) // 5, 1)
        print(" ", k, " ".join(f"V{st.mean(x[0] for x in v[i:i + n]):.2f}->G{st.mean(x[1] for x in v[i:i + n]):.2f}" for i in range(0, len(v), n)))
