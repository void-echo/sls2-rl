"""Follow a training run: archive every N-th checkpoint and evaluate it on held-out seeds.

    python -m rl.eval_watch --run_dir runs/<name> --every 100 --engine_dir ~/sls2-rl/engine_v1 [--device cuda:0]

Training keeps only the latest few checkpoints, so this copies each `ckpt_<update>.pt` with update % every == 0
into <run_dir>/eval_ckpts/ (kept forever: later behaviour analysis compares them), runs rl.evaluate greedily on
the fixed seeds eval_0..n-1, and appends one row per checkpoint to <run_dir>/eval.csv. It exits when the
training process is gone and every archived checkpoint has been evaluated.
"""
import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import time

FIELDS = ["update", "mode", "env_steps", "n", "aborted", "win_rate", "ci_lo", "ci_hi", "floor_mean", "floor_median",
          "floor_q25", "floor_q75", "eval_seconds", "time"]


def archive(src, dst):
    """Copy a checkpoint without optimizer state (about a third of the size); evaluation needs only the policy.
    Raises OSError when the disk does not take it; nothing is left at `dst` then."""
    import torch
    from .diskfull import write_atomic
    ck = torch.load(src, map_location="cpu", weights_only=False)
    keep = {k: ck[k] for k in ("model", "cfg", "vnorm", "update", "env_steps") if k in ck}
    data = io.BytesIO()
    torch.save(keep, data)
    write_atomic(dst, data.getbuffer())


def disk_full(text):
    return any(s in text for s in ("No space left on device", "Errno 28", "Disk quota exceeded", "Errno 122"))


def training_alive(name):
    r = subprocess.run(["pgrep", "-f", f"[r]l[.]train --name {re.escape(name)}([[:space:]]|$)"], capture_output=True)
    return r.returncode == 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--n_seeds", type=int, default=200)
    ap.add_argument("--engine_dir", required=True)
    ap.add_argument("--n_workers", type=int, default=8)
    ap.add_argument("--envs_per_worker", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--sample", action="store_true", help="sample from the policy instead of greedy argmax")
    ap.add_argument("--ascension", type=int, default=None)
    a = ap.parse_args(argv)
    from .diskfull import append_all, quiet_stdio
    quiet_stdio()

    run_dir = os.path.abspath(os.path.expanduser(a.run_dir))
    name = os.path.basename(run_dir.rstrip("/"))
    ck_dir, arch = os.path.join(run_dir, "ckpt"), os.path.join(run_dir, "eval_ckpts")
    os.makedirs(arch, exist_ok=True)
    csv_path = os.path.join(run_dir, "eval_sample.csv" if a.sample else "eval.csv")
    if a.ascension is not None:
        csv_path = os.path.join(run_dir, f"eval_a{a.ascension}.csv")
    done = set()
    if os.path.exists(csv_path):
        done = {int(r["update"]) for r in csv.DictReader(open(csv_path))}
    attempts, retry_at = {}, {}
    while True:
        if a.ascension is None and os.path.isdir(ck_dir):
            for f in sorted(os.listdir(ck_dir)):
                m = re.fullmatch(r"ckpt_(\d+)\.pt", f)
                if m and int(m.group(1)) % a.every == 0 and not os.path.exists(os.path.join(arch, f)):
                    try:
                        archive(os.path.join(ck_dir, f), os.path.join(arch, f))
                    except OSError as e:    # disk full, or training rotated the checkpoint away: the next poll looks again
                        print(f"archiving {f} failed ({e})", flush=True)
        todo = sorted(int(re.fullmatch(r"ckpt_(\d+)\.pt", f).group(1)) for f in os.listdir(arch)
                      if re.fullmatch(r"ckpt_(\d+)\.pt", f))
        todo = [u for u in todo if u not in done]
        if not todo:
            if not training_alive(name):
                return
            time.sleep(a.poll)
            continue
        pending = [u for u in todo if time.monotonic() >= retry_at.get(u, 0)]
        if not pending:
            time.sleep(a.poll)
            continue
        u = pending[0]
        ckpt = os.path.join(arch, f"ckpt_{u:06d}.pt")
        out = os.path.join(run_dir, "eval", f"eval_{u:06d}{'_sample' if a.sample else ''}.json")
        if a.ascension is not None:
            out = os.path.join(run_dir, f"eval_a{a.ascension}", f"eval_{u:06d}.json")
        t0 = time.time()
        cmd = [sys.executable, "-m", "rl.evaluate", "--ckpt", ckpt] + ([] if a.sample else ["--greedy"]) + ["--n_seeds", str(a.n_seeds),
               "--engine", "real", "--engine_dir", a.engine_dir, "--n_workers", str(a.n_workers),
               "--envs_per_worker", str(a.envs_per_worker), "--device", a.device, "--out", out]
        if a.ascension is not None:
            cmd += ["--ascension", str(a.ascension)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            print(f"eval of update {u} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}", flush=True)
            if disk_full(r.stdout + r.stderr):  # not the checkpoint's fault: wait for room, do not count the attempt
                retry_at[u] = time.monotonic() + a.poll
                continue
            attempts[u] = attempts.get(u, 0) + 1
            if attempts[u] >= 3:
                print(f"giving up evaluation of update {u} after 3 failures", flush=True)
                done.add(u)
            else:
                retry_at[u] = time.monotonic() + a.poll
            continue
        res = json.load(open(out))
        s = res["summary"]["all"]
        steps = re.search(r"(\d+) steps\)", res.get("tag", ""))
        row = {"update": u, "mode": "sample" if a.sample else "greedy", "env_steps": steps.group(1) if steps else "", "n": s["n"], "aborted": s["aborted"],
               "win_rate": s["win_rate"], "ci_lo": s["ci95"][0], "ci_hi": s["ci95"][1], "floor_mean": s["floor_mean"],
               "floor_median": s["floor_median"], "floor_q25": s["floor_q25"], "floor_q75": s["floor_q75"],
               "eval_seconds": round(time.time() - t0), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        text = io.StringIO(newline="")
        w = csv.DictWriter(text, FIELDS)
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            w.writeheader()
        w.writerow(row)
        while True:     # the evaluation is done and its json is on disk; only the row is waiting for room
            try:
                append_all(csv_path, text.getvalue().encode())
                break
            except OSError as e:
                print(f"{csv_path}: row of update {u} not written ({e}); trying again in {a.poll:.0f}s", flush=True)
                time.sleep(a.poll)
        print(f"update {u}: win {s['win_rate']:.3f} [{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}] floor {s['floor_mean']:.1f}",
              flush=True)
        done.add(u)


if __name__ == "__main__":
    main()
