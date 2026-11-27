#!/usr/bin/env python3
"""Play random runs across all characters and save a few raw protocol states per decision type.

The saved samples are what the observation encoder is designed against.
Usage: sample_protocol.py <sts2-cli dir> <out dir> [runs_per_character]
"""
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_realengine import Engine, legal_actions  # noqa: E402

CHARACTERS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder", "Fool"]


def main():
    cli_dir, out_dir = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    os.makedirs(out_dir, exist_ok=True)
    seen = collections.defaultdict(list)
    counts = collections.Counter()
    for ch in CHARACTERS:
        for i in range(n):
            rng = random.Random(i)
            eng = Engine(cli_dir)
            try:
                state = eng.send({"cmd": "start_run", "character": ch, "seed": f"sample_{ch}_{i}", "ascension": 0})
                for _ in range(3000):
                    kind = state.get("decision") or state.get("type")
                    counts[kind] += 1
                    if len(seen[kind]) < 3 and rng.random() < 0.3 or not seen[kind]:
                        seen[kind].append(state)
                    if state.get("decision") == "game_over" or state.get("type") == "error":
                        break
                    name, args = rng.choice(legal_actions(state))
                    state = eng.send({"cmd": "action", "action": name, "args": args})
            finally:
                eng.close()
    for kind, states in seen.items():
        with open(os.path.join(out_dir, f"{kind}.json"), "w") as f:
            json.dump(states, f, ensure_ascii=False, indent=1)
    print(counts)


if __name__ == "__main__":
    main()
