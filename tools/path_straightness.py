"""CPU-only straightness of a checkpoint path, overall and by Policy module.

    python3 tools/path_straightness.py --ckpts runs/x/ckpt --out path.json

Directory input uses ckpt_*.pt (excluding latest.pt); explicit paths retain the
supplied update order. mmap loads only model tensor pages, not optimizer data.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


MODULE_PREFIXES = {
    "trunk": ("blocks.",),
    "candidate": ("cand_blocks.",),
    "logit": ("logit.",),
    "value": ("value.",),
}
GROUPS = ("all", "no_table", *MODULE_PREFIXES)


def parameter_groups(name):
    groups = ["all"]
    if name != "table":
        groups.append("no_table")
    groups.extend(group for group, prefixes in MODULE_PREFIXES.items() if name.startswith(prefixes))
    return groups


def distances(a, b):
    squares = dict.fromkeys(GROUPS, 0.0)
    for name, value in a.items():
        delta = b[name].double() - value.double()
        squared = float(torch.sum(delta.square()))
        for group in parameter_groups(name):
            squares[group] += squared
    return {group: math.sqrt(value) for group, value in squares.items()}


def path_straightness(states, windows):
    steps = [distances(a, b) for a, b in zip(states, states[1:])]
    result = {group: [] for group in GROUPS}
    for w in windows:
        ratios = {group: [] for group in GROUPS}
        for i in range(len(states) - w):
            chord = steps[i] if w == 1 else distances(states[i], states[i + w])
            for group in GROUPS:
                length = sum(step[group] for step in steps[i:i + w])
                if length:
                    ratios[group].append(chord[group] / length)
        for group, values in ratios.items():
            result[group].append(dict(window=w, mean=float(np.mean(values)) if values else None,
                                      n_valid=len(values), n_windows=max(0, len(states) - w),
                                      random_walk_baseline=1 / math.sqrt(w)))
    return result


def read_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return ck["model"], dict(path=str(path), update=ck["update"], env_steps=ck["env_steps"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--windows", default="1,2,4,8")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    paths = [Path(p) for p in args.ckpts]
    if len(paths) == 1 and paths[0].is_dir():
        directory = paths[0]
        if (directory / "ckpt").is_dir():
            directory = directory / "ckpt"
        paths = sorted(directory.glob("ckpt_*.pt"))
    states, metadata = [], []
    for path in paths:
        state, meta = read_checkpoint(path)
        states.append(state)
        metadata.append(meta)
    # Checkpoint metadata defines the order for directory input as well.
    if len(args.ckpts) == 1 and Path(args.ckpts[0]).is_dir():
        ordered = sorted(zip(states, metadata), key=lambda item: item[1]["update"])
        states, metadata = [item[0] for item in ordered], [item[1] for item in ordered]
    windows = [int(w) for w in args.windows.split(",")]
    result = path_straightness(states, windows)
    output = dict(checkpoints=metadata, groups={"all": "all model parameters", "no_table": "all except table",
                                               **{k: list(v) for k, v in MODULE_PREFIXES.items()}},
                  window_unit="checkpoint intervals, not necessarily single PPO updates",
                  zero_length="undefined ratios excluded; n_valid reports remaining windows",
                  results=result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print("Checkpoints: " + ", ".join(f"u{m['update']} ({m['env_steps']} steps)" for m in metadata))
    print(f"{'module':<12} {'w':>4} {'straightness':>14} {'random walk':>13} {'valid/windows':>15}")
    for group, rows in result.items():
        for row in rows:
            value = "n/a" if row["mean"] is None else f"{row['mean']:.6f}"
            print(f"{group:<12} {row['window']:>4} {value:>14} {row['random_walk_baseline']:>13.6f} "
                  f"{row['n_valid']:>7}/{row['n_windows']:<7}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
