"""Trace checkpoints sequentially and collect behavior JSONs plus one comparison CSV.

    .venv/bin/python tools/behavior_sweep.py --checkpoint base runs/base/eval_ckpts/ckpt_000300.pt \
        --checkpoint grouped runs/grouped/eval_ckpts/ckpt_000300.pt --device cuda --n_engines 32
"""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from rl.behavior_metrics import metric_rows  # noqa: E402


def csv_row(name, ckpt, result, json_path):
    row = {"experiment": name, "update": result["meta"][0]["update"], "ckpt": str(ckpt), "json": str(json_path)}
    row.update({f"samples.{k}": v for k, v in result["samples"].items()})
    for path, metric in metric_rows(result):
        row[f"{path}.n"] = metric["n"]
        if "rate" in metric:
            row[f"{path}.k"] = metric["k"]
            row[f"{path}.rate"] = metric["rate"]
            row[f"{path}.ci95_lo"], row[f"{path}.ci95_hi"] = metric["ci95"]
        else:
            row[f"{path}.mean"] = metric["mean"]
            row[f"{path}.total"] = metric["total"]
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", nargs=2, action="append", metavar=("NAME", "CKPT"), required=True)
    ap.add_argument("--n_seeds", type=int, default=100)
    ap.add_argument("--n_engines", type=int, default=2)
    ap.add_argument("--engine_dir", default=str(REPO / "engine"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--seed_prefix", default="pref")
    ap.add_argument("--runs_dir", default=str(REPO / "runs"))
    ap.add_argument("--csv", help="comparison CSV (default: <runs_dir>/behavior.csv)")
    a = ap.parse_args(argv)
    if a.n_seeds < 1 or a.n_engines < 1:
        ap.error("--n_seeds and --n_engines must be positive")
    runs_dir = Path(a.runs_dir).expanduser().resolve()
    csv_path = Path(a.csv).expanduser().resolve() if a.csv else runs_dir / "behavior.csv"
    entries = [(name, Path(path).expanduser().resolve()) for name, path in a.checkpoint]
    rows = []
    for name, ckpt in entries:
        behavior_dir = runs_dir / name / "behavior"
        behavior_dir.mkdir(parents=True, exist_ok=True)
        trace_path = behavior_dir / f"{ckpt.stem}.jsonl.gz"
        print(f"{name}: {ckpt}", flush=True)
        subprocess.run([sys.executable, "-m", "rl.trace", "--ckpt", str(ckpt), "--engine_dir", os.path.expanduser(a.engine_dir),
                        "--n_seeds", str(a.n_seeds), "--n_engines", str(a.n_engines), "--device", a.device,
                        "--character", a.character, "--ascension", str(a.ascension), "--seed_prefix", a.seed_prefix,
                        "--behavior_obs", "--out", str(trace_path)], cwd=REPO, check=True)
        with gzip.open(trace_path, "rt") as f:
            update = json.loads(f.readline())["update"]
        if update is None:
            raise ValueError(f"checkpoint has no update: {ckpt}")
        saved_trace = behavior_dir / f"u{update}.jsonl.gz"
        trace_path.replace(saved_trace)
        json_path = behavior_dir / f"u{update}.json"
        subprocess.run([sys.executable, "-m", "rl.behavior_metrics", str(saved_trace), "--ckpt", str(ckpt),
                        "--device", a.device, "--out", str(json_path)], cwd=REPO, check=True)
        with open(json_path) as f:
            rows.append(csv_row(name, ckpt, json.load(f), json_path))
        fields = list(dict.fromkeys(k for row in rows for k in row))
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
