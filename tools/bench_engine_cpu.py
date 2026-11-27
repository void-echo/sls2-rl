#!/usr/bin/env python3
"""Record greedy episodes, then measure gamma engine CPU with fixed action replays."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
from pathlib import Path
import select
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
HZ = os.sysconf("SC_CLK_TCK")
TIMING_FIELDS = ("t_exec_ms", "t_obs_ms")


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def cpu_s(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / HZ


def rss_mb(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError(f"VmRSS missing for process {pid}")


def runtime_env(overrides):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("DOTNET_", "COMPlus_", "CORECLR_")) and k != "STS2_LIB"}
    env.update(DOTNET_ROOT=str(Path.home() / ".dotnet"), DOTNET_EnableDiagnostics="0")
    env.update(overrides)
    return env


def config(name, engine, build="Release", env=None, dll=None):
    engine = Path(engine).expanduser().resolve()
    return {"config": name, "engine_dir": str(engine),
            "dll": str(dll or engine / "src/Sts2Headless/bin" / build / "net9.0/Sts2Headless.dll"),
            "env": {"DOTNET_ROOT": str(Path.home() / ".dotnet"),
                    "DOTNET_EnableDiagnostics": "0", **(env or {})}}


class Engine:
    def __init__(self, cfg, log):
        self.log = open(log, "ab")
        self.born = time.monotonic()
        self.proc = subprocess.Popen([str(Path.home() / ".dotnet/dotnet"), cfg["dll"]],
                                     cwd=cfg["engine_dir"], env=runtime_env(cfg["env"]),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, bufsize=0)
        self.buf = b""
        msg, self.ready_raw = self.recv(300)
        if msg.get("type") != "ready":
            self.close()
            raise RuntimeError(f"engine init: {msg}")
        self.ready_cpu_s = cpu_s(self.proc.pid)
        self.ready_wall_s = time.monotonic() - self.born

    def recv(self, timeout=120):
        deadline = time.monotonic() + timeout
        while True:
            if b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                line += b"\n"
                if line.lstrip().startswith(b"{"):
                    return json.loads(line), line
                self.log.write(line)
                continue
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"engine {self.proc.pid}: response timeout")
            if select.select([self.proc.stdout], [], [], left)[0]:
                data = os.read(self.proc.stdout.fileno(), 1 << 20)
                if not data:
                    raise RuntimeError(f"engine {self.proc.pid}: stdout closed")
                self.buf += data

    def send(self, obj):
        self.proc.stdin.write((json.dumps(obj, separators=(",", ":")) + "\n").encode())
        self.proc.stdin.flush()

    def request(self, obj, timeout=120):
        self.send(obj)
        return self.recv(timeout)

    def close(self):
        if self.proc.poll() is None:
            try:
                self.send({"cmd": "quit"})
                self.proc.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait()
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.log.close()


class Trace:
    def __init__(self, path):
        self.file = gzip.open(path, "wb", compresslevel=1)
        self.raw = hashlib.sha1()
        self.behavior = hashlib.sha1()
        self.line_sha1 = []
        self.behavior_line_sha1 = []

    def add(self, msg, line):
        self.file.write(line)
        h = hashlib.sha1(line).digest()
        self.raw.update(h)
        self.line_sha1.append(h.hex())
        state = dict(msg)
        if isinstance(state.get("info"), dict):
            state["info"] = {k: v for k, v in state["info"].items() if k not in TIMING_FIELDS}
        h = hashlib.sha1(json.dumps(state, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False).encode()).digest()
        self.behavior.update(h)
        self.behavior_line_sha1.append(h.hex())

    def finish(self):
        self.file.close()
        return {"sha1": self.raw.hexdigest(), "behavior_sha1": self.behavior.hexdigest(),
                "line_sha1": self.line_sha1, "behavior_line_sha1": self.behavior_line_sha1}


def start_command(record, episode):
    return {"cmd": "rl_start", "character": record["character"],
            "ascension": record["ascension"], "seed": episode["seed"]}


def require_rl(msg):
    if msg.get("type") != "rl":
        raise RuntimeError(str(msg)[:2000])


def record(a):
    import torch
    from rl.encode import Encoder
    from rl.ppo import to_torch
    from rl.train import act_batch, amp_context, load_policy

    torch.set_num_threads(1)
    device = torch.device(a.device)
    model, _, cfg, ck = load_policy(a.ckpt, device)
    model.eval()
    encoder = Encoder(cfg.caps)
    amp = amp_context(cfg, device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c = config("record", a.engine_dir, "Debug")
    data = {"character": "Ironclad", "ascension": 10, "seed_prefix": a.seed_prefix,
            "checkpoint": str(Path(a.ckpt).resolve()), "checkpoint_update": ck.get("update"),
            "device": str(device), "greedy": True, "engine": c, "episodes": []}
    slots = {}
    next_ep = 0

    def new_slot(slot):
        nonlocal next_ep
        if next_ep >= a.episodes:
            return
        index = next_ep
        next_ep += 1
        ep = {"index": index, "seed": f"{a.seed_prefix}_{index}", "actions": []}
        eng = Engine(c, out / f"record_{index:02d}.stderr.log")
        trace = Trace(out / f"record_{index:02d}.jsonl.gz")
        msg, line = eng.request(start_command(data, ep), 300)
        trace.add(msg, line)
        require_rl(msg)
        slots[slot] = [eng, ep, trace, msg]

    try:
        for slot in range(min(a.workers, a.episodes)):
            new_slot(slot)
        while slots:
            live = sorted(slots)
            batch = encoder.alloc(len(live))
            for j, slot in enumerate(live):
                encoded = encoder.encode(slots[slot][3])
                for key in batch:
                    batch[key][j] = encoded[key]
            actions, _, _ = act_batch(model, to_torch(batch, device), amp, greedy=True)
            actions = actions.cpu().tolist()
            for slot, action in zip(live, actions):
                eng, ep, trace, msg = slots[slot]
                if not 0 <= action < len(msg["cands"]):
                    raise RuntimeError(f"invalid policy action {action}")
                ep["actions"].append(action)
                eng.send({"cmd": "rl_act", "i": action})
            for slot in live:
                eng, ep, trace, _ = slots[slot]
                msg, line = eng.recv()
                trace.add(msg, line)
                require_rl(msg)
                slots[slot][3] = msg
                if msg["done"]:
                    ep.update(steps=len(ep["actions"]), victory=msg.get("victory"),
                              final_info=msg.get("info"), **trace.finish())
                    eng.close()
                    del slots[slot]
                    data["episodes"].append(ep)
                    data["episodes"].sort(key=lambda x: x["index"])
                    save_json(out / "actions.json", data)
                    print(f"record {ep['seed']}: steps={ep['steps']} victory={ep['victory']} "
                          f"floor={ep['final_info'].get('total_floor')}", flush=True)
                    new_slot(slot)
    finally:
        for eng, _, trace, _ in slots.values():
            eng.close()
            trace.file.close()
    return data


def replay_chain(cfg, record_data, episodes, directory, rss=False):
    results = []
    eng = None
    previous_cpu = 0.0
    previous_wall = None
    try:
        for position, ep in enumerate(episodes):
            trace = Trace(directory / f"episode_{ep['index']:02d}.jsonl.gz")
            result = {"index": ep["index"], "seed": ep["seed"], "chain_position": position,
                      "steps": 0, "completed": False, "error": None}
            startup_cpu = 0.0
            start_wall = time.monotonic()
            try:
                if eng is None:
                    eng = Engine(cfg, directory / f"chain_{episodes[0]['index']:02d}.stderr.log")
                    previous_cpu = 0.0
                    previous_wall = eng.born
                start_wall = previous_wall
                result.update(pid=eng.proc.pid, process_ready_cpu_s=eng.ready_cpu_s,
                              process_ready_wall_s=eng.ready_wall_s, cpu_before_s=previous_cpu)
                msg, line = eng.request(start_command(record_data, ep), 300)
                startup_cpu = cpu_s(eng.proc.pid)
                startup_wall = time.monotonic()
                result.update(startup_cpu_s=startup_cpu - previous_cpu,
                              startup_wall_s=startup_wall - start_wall)
                trace.add(msg, line)
                require_rl(msg)
                for action in ep["actions"]:
                    if msg.get("done") or not 0 <= action < len(msg["cands"]):
                        raise RuntimeError(f"replay diverged before action {result['steps']}")
                    msg, line = eng.request({"cmd": "rl_act", "i": action})
                    result["steps"] += 1
                    if result["steps"] == len(ep["actions"]):
                        end_cpu = cpu_s(eng.proc.pid)
                        end_wall = time.monotonic()
                    trace.add(msg, line)
                    require_rl(msg)
                result.update(completed=bool(msg.get("done")), victory=msg.get("victory"),
                              final_info=msg.get("info"))
                if not result["completed"]:
                    raise RuntimeError("recorded actions ended before terminal response")
                if rss:
                    result["vmrss_mb"] = rss_mb(eng.proc.pid)
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                end_cpu = cpu_s(eng.proc.pid) if eng is not None else previous_cpu
                end_wall = time.monotonic()
            result.update(cpu_at_end_s=end_cpu, total_cpu_s=end_cpu - previous_cpu,
                          action_cpu_s=end_cpu - startup_cpu, wall_s=end_wall - start_wall,
                          **trace.finish())
            results.append(result)
            save_json(directory / f"episode_{ep['index']:02d}.json", result)
            if len(episodes) > 1:
                print(f"{cfg['config']} pid={result.get('pid')} "
                      f"episode={position + 1}/{len(episodes)} seed={ep['seed']} "
                      f"rss_mb={result.get('vmrss_mb')}", flush=True)
            previous_cpu, previous_wall = end_cpu, end_wall
            if result["error"]:
                if eng is not None:
                    eng.close()
                eng = None
    finally:
        if eng is not None:
            eng.close()
    return results


def summarize(cfg, episodes, reference, elapsed, workers, reuse):
    baseline = {e["seed"]: e for e in (reference or episodes)}
    for ep in episodes:
        ref = baseline[ep["seed"]]
        ep["identical"] = ep["completed"] and ep["sha1"] == ref["sha1"]
        ep["behavior_identical"] = ep["completed"] and ep["behavior_sha1"] == ref["behavior_sha1"]
        ep["first_raw_difference"] = next((i for i, pair in enumerate(zip(ep["line_sha1"], ref["line_sha1"]))
                                            if pair[0] != pair[1]), None)
        ep["first_behavior_difference"] = next((i for i, pair in enumerate(zip(ep["behavior_line_sha1"],
                                                                              ref["behavior_line_sha1"]))
                                                 if pair[0] != pair[1]), None)
    steps = sum(e["steps"] for e in episodes)
    total = sum(e["total_cpu_s"] for e in episodes)
    starts = [e.get("startup_cpu_s", 0.0) for e in episodes]
    cold = [e.get("startup_cpu_s", 0.0) for e in episodes if e["chain_position"] == 0]
    warm = [e.get("startup_cpu_s", 0.0) for e in episodes if e["chain_position"] > 0]
    action = sum(e["action_cpu_s"] for e in episodes)
    processes = {}
    for ep in episodes:
        if "pid" not in ep:
            continue
        process = processes.setdefault(ep["pid"], {"pid": ep["pid"], "episode_indices": [], "vmrss_mb": []})
        process["episode_indices"].append(ep["index"])
        if "vmrss_mb" in ep:
            process["vmrss_mb"].append(ep["vmrss_mb"])
    row = {**cfg, "startup_cpu_s_mean": statistics.mean(starts),
           "startup_cpu_s_median": statistics.median(starts),
           "cold_startup_cpu_s_mean": statistics.mean(cold),
           "warm_startup_cpu_s_mean": statistics.mean(warm) if warm else None,
           "startup_cpu_s_total": sum(starts), "total_cpu_s": total, "steps": steps,
           "cpu_ms_per_step": total * 1000 / steps if steps else None,
           "action_cpu_ms_per_step": action * 1000 / steps if steps else None,
           "wall_s": sum(e["wall_s"] for e in episodes), "elapsed_wall_s": elapsed,
           "identical_episodes": sum(e["identical"] for e in episodes),
           "behavior_identical_episodes": sum(e["behavior_identical"] for e in episodes),
           "total_episodes": len(episodes), "completed_episodes": sum(e["completed"] for e in episodes),
           "errors": [{"seed": e["seed"], "error": e["error"]} for e in episodes if e["error"]],
           "parallelism": workers, "episodes_per_process": reuse,
           "process_count": len(processes),
           "vmrss_mb_by_process": list(processes.values()), "vmrss_unit": "MiB (VmRSS kB / 1024)",
           "raw_hash": "SHA1(concatenated binary SHA1 of each exact JSON line including newline; rl_start then rl_act)",
           "behavior_hash_excluded_info_fields": list(TIMING_FIELDS), "cpu_clock_ticks_per_second": HZ}
    return row


def replay_config(cfg, record_data, out, reference=None, workers=8, reuse=1, rss=False):
    directory = Path(out) / cfg["config"]
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / "config.json", cfg)
    episodes = record_data["episodes"]
    chains = [episodes[i:i + reuse] for i in range(0, len(episodes), reuse)]
    started = time.monotonic()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(chains))) as pool:
        tasks = [pool.submit(replay_chain, cfg, record_data, chain, directory, rss) for chain in chains]
        for task in concurrent.futures.as_completed(tasks):
            done = task.result()
            results.extend(done)
            print(f"{cfg['config']}: {len(results)}/{len(episodes)} episodes", flush=True)
    results.sort(key=lambda x: x["index"])
    row = summarize(cfg, results, reference, time.monotonic() - started, min(workers, len(chains)), reuse)
    save_json(directory / "results.json", {"summary": row, "episodes": results})
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row, results


def suite(a):
    data = json.loads(Path(a.record).read_text())
    baseline = config("baseline", a.engine_dir, "Debug")
    release = config("release", a.release_dir)
    variants = [baseline, release]
    settings = [("tiered_pgo0", {"DOTNET_TieredPGO": "0"}),
                ("tiered_compilation0", {"DOTNET_TieredCompilation": "0"}),
                ("quickjit_loops_osr_default", {"DOTNET_TC_QuickJitForLoops": "1"}),
                ("readytorun1", {"DOTNET_ReadyToRun": "1"}),
                ("readytorun0", {"DOTNET_ReadyToRun": "0"}),
                ("gc_concurrent0", {"DOTNET_gcConcurrent": "0"}),
                ("gen0_64mb", {"DOTNET_GCgen0size": "0x4000000"}),
                ("gc_server0", {"DOTNET_gcServer": "0"})]
    variants += [config("release_" + name, a.release_dir, env=env) for name, env in settings]
    variants.append(config("release_crossgen2", a.r2r_dir))
    out = Path(a.out)
    if a.configs:
        catalog = {cfg["config"]: cfg for cfg in variants}
        best_env = {"DOTNET_TieredPGO": "0", "DOTNET_GCgen0size": "0x4000000"}
        catalog["release_best_combined"] = config("release_best_combined", a.release_dir, env=best_env)
        catalog["baseline_best_combined"] = config("baseline_best_combined", a.engine_dir, "Debug", env=best_env)
        names = [name.strip() for name in a.configs.split(",") if name.strip()]
        unknown = set(names) - catalog.keys()
        if unknown:
            raise ValueError(f"unknown configs: {', '.join(sorted(unknown))}")
        report_path = out / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else []
        reference_path = out / "baseline/results.json"
        reference = json.loads(reference_path.read_text())["episodes"] if reference_path.exists() else None
        for name in names:
            cfg = dict(catalog[name])
            if a.reuse != 1:
                cfg["config"] += f"_reuse{a.reuse}"
            if reference is None and (name != "baseline" or a.reuse != 1):
                raise ValueError("run baseline with --reuse 1 in this output directory first")
            row, results = replay_config(cfg, data, out, reference, a.workers, a.reuse, a.rss)
            if name == "baseline" and a.reuse == 1:
                reference = results
            report = [r for r in report if r["config"] != row["config"]] + [row]
            save_json(report_path, report)
        return
    report = []
    reference = None
    for cfg in variants:
        row, results = replay_config(cfg, data, out, reference, a.workers, a.reuse, a.rss)
        report.append(row)
        if reference is None:
            reference = results
        save_json(out / "report.json", report)
    release_ms = report[1]["cpu_ms_per_step"]
    candidates = sorted((r for r in report[2:-1] if r["behavior_identical_episodes"] == len(data["episodes"]) and
                         r["cpu_ms_per_step"] < release_ms), key=lambda r: r["cpu_ms_per_step"])
    overrides = {}
    sources = []
    for row in candidates:
        changes = {k: v for k, v in row["env"].items() if release["env"].get(k) != v}
        if not changes or "DOTNET_ReadyToRun" in changes:
            continue
        if "DOTNET_TieredCompilation" in overrides and any(k in changes for k in
                                                          ("DOTNET_TieredPGO", "DOTNET_TC_QuickJitForLoops")):
            continue
        if "DOTNET_TieredCompilation" in changes and any(k in overrides for k in
                                                       ("DOTNET_TieredPGO", "DOTNET_TC_QuickJitForLoops")):
            continue
        if "DOTNET_TieredCompilation" in changes:
            overrides.pop("DOTNET_TieredPGO", None)
            overrides.pop("DOTNET_TC_QuickJitForLoops", None)
        overrides.update(changes)
        sources.append(row["config"])
        if len(overrides) >= 3:
            break
    r2r = report[-1]
    best_engine = a.r2r_dir if (r2r["behavior_identical_episodes"] == len(data["episodes"]) and
                               r2r["cpu_ms_per_step"] < release_ms) else a.release_dir
    combo = config("release_best_combined", best_engine, env=overrides)
    combo["selected_from"] = sources + (["release_crossgen2"] if best_engine == a.r2r_dir else ["release"])
    row, _ = replay_config(combo, data, out, reference, a.workers, a.reuse, a.rss)
    report.append(row)
    save_json(out / "report.json", report)
    reuse_cfg = {**baseline, "config": "baseline_reuse8"}
    row, _ = replay_config(reuse_cfg, data, out, reference, a.workers, reuse=8, rss=a.rss)
    report.append(row)
    save_json(out / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--ckpt", required=True)
    rec.add_argument("--engine-dir", default=str(REPO / "engine_v2"))
    rec.add_argument("--device", default="cuda:0")
    rec.add_argument("--episodes", type=int, default=16)
    rec.add_argument("--seed-prefix", default="benchcpu")
    rec.add_argument("--workers", type=int, default=8)
    rec.add_argument("--out", required=True)
    rep = sub.add_parser("replay")
    rep.add_argument("--record", required=True)
    rep.add_argument("--engine-dir", default=str(REPO / "engine_v2"))
    rep.add_argument("--release-dir", default=str(REPO / "engine_bench_release"))
    rep.add_argument("--r2r-dir", default=str(REPO / "engine_bench_release_r2r"))
    rep.add_argument("--workers", type=int, default=8)
    rep.add_argument("--configs", default="", help="comma-separated preset names; reuse variants compare with the saved baseline")
    rep.add_argument("--reuse", type=int, default=1, help="episodes per engine process")
    rep.add_argument("--rss", action="store_true", help="record VmRSS after each episode")
    rep.add_argument("--out", required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("workers must be between 1 and 12")
    if args.mode == "replay" and args.reuse < 1:
        parser.error("reuse must be at least 1")
    if args.mode == "record":
        record(args)
    else:
        suite(args)


if __name__ == "__main__":
    main()
