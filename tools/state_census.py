"""State census: the union of every field the game's run and combat state holds at decision points, over many
random runs (9999 HP so they reach acts 2-3), written as a TSV to classify against the observation
(docs/02_观测与编码方法论.md, section 4). Uses the engine's dbg_census command.

  python3 tools/state_census.py --engine engine --runs 30 --out tmp/obs_audit/census.tsv
"""
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_runner import Engine  # noqa: E402

CHARS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder", "Fool"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="engine")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--every", type=int, default=20, help="census every N steps, plus at every new decision kind")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    eng = Engine(a.engine, timeout=120)
    fields, statics, kinds_seen = {}, {}, set()
    n_census = 0
    for i in range(a.runs):
        rng = random.Random(i)
        R = eng.send({"cmd": "rl_start", "character": CHARS[i % len(CHARS)], "ascension": 10 * (i % 2), "seed": f"census_{i}"}, 120)
        R = eng.send({"cmd": "set_player", "hp": 9999, "max_hp": 9999}, 120)
        for step in range(6000):
            if R.get("type") != "rl" or R.get("done"):
                break
            kind = R["decision"]
            if step % a.every == 0 or kind not in kinds_seen:
                kinds_seen.add(kind)
                C = eng.send({"cmd": "dbg_census"}, 300)
                n_census += 1
                for f in C.get("fields", []):
                    key = (f["type"], f["field"])
                    g = fields.setdefault(key, dict(f, n=0, states=0, decisions=set()))
                    g["states"] += 1
                    g["n"] += f["n"]
                    g["decisions"].add(kind)
                for f in C.get("statics", []):
                    statics.setdefault((f["type"], f["field"]), f)
                R = eng.send({"cmd": "dbg_obs"}, 120)  # re-observe (the census command returns no decision)
            cands = R["cands"]
            w = [0.2 if c["k"] == "discard_potion" else (3.0 if c["k"] in ("play", "reward", "pick", "confirm") else 1.0)
                 for c in cands]
            R = eng.send({"cmd": "rl_act", "i": rng.choices(range(len(cands)), weights=w)[0]}, 120)
        print(f"run {i}: {step} steps, {len(fields)} fields, {len(statics)} statics, {n_census} censuses", flush=True)
    with open(a.out, "w", newline="") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["kind", "type", "field", "ftype", "readonly", "states", "n", "example", "first_path", "decisions"])
        for (t, f), g in sorted(fields.items()):
            w.writerow(["instance", t, f, g["ftype"], g["readonly"], g["states"], g["n"], g["ex"], g["path"],
                        ",".join(sorted(g["decisions"]))])
        for (t, f), g in sorted(statics.items()):
            w.writerow(["static", t, f, g["ftype"], "", "", "", g["ex"], "", ""])
    print(f"{len(fields)} instance fields, {len(statics)} statics -> {a.out}")


if __name__ == "__main__":
    main()
