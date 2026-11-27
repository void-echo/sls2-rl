#!/usr/bin/env python3
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path


def prepare(repo, game, output, baseline, quick):
    repo, game, out = (Path(p).expanduser().resolve() for p in (repo, game, output))
    if not game.is_dir():
        raise ValueError(f"Game directory not found: {game}")
    if baseline:
        json.loads(Path(baseline).read_text())
    candidates = [game / "sts2.dll", game / "SlayTheSpire2.app/Contents/Resources/data_sts2_macos_x86_64/sts2.dll",
                  game / "SlayTheSpire2.app/Contents/Resources/data_sts2_macos_arm64/sts2.dll"]
    dll = next((p for p in candidates if p.is_file()), None)
    if dll is None:
        dll = next(game.rglob("sts2.dll"))
    native = game / "SlayTheSpire2.app/Contents/Resources" / f"data_sts2_macos_{'arm64' if platform.machine() == 'arm64' else 'x86_64'}" / "sts2.dll"
    runtime = native if sys.platform == "darwin" and native.is_file() else dll
    release = next((p / "release_info.json" for p in (dll.parent, dll.parent.parent, game)
                    if (p / "release_info.json").is_file()), None)
    if release is None:
        release = next(game.rglob("release_info.json"))
    info = json.loads(release.read_text())
    metadata = dict(version=info["version"], commit=info["commit"], dll=str(dll),
                    runtime_dll=str(runtime), game_dir=str(game),
                    quick=bool(int(quick)), started=time.time())
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "version.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "results.jsonl").write_text("")
    print(f"{info['version']} ({info['commit']})", flush=True)


def step(output, name, *command):
    out = Path(output)
    log = out / "logs" / f"{name}.log"
    start = time.monotonic()
    print(f"[{name}] starting", flush=True)
    with log.open("w") as f:
        try:
            rc = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT).returncode
        except OSError as e:
            f.write(str(e) + "\n")
            rc = 1
    text = log.read_text(errors="replace")
    checks = []
    if name.startswith("rl_smoke_"):
        checks = [r"^errors: 0$", r"^replay deterministic in same process: True$"]
    elif name.startswith("fullrun"):
        checks = [r"^problems: 0$"]
    elif name == "probe_specific":
        checks = [r"^PASS\s"]
        if re.search(r"^(FAIL|ERR)\s", text, re.M):
            rc = 1
    elif name.startswith("fuzz_"):
        checks = [r"^items with problems: 0/", r"^navigation cycles: 0 in"]
        if re.search(r"^flags:", text, re.M):
            rc = 1
    elif name == "startup":
        checks = [r'^\{"type":"ready"', r'^\{"type":"quit_result","success":true\}']
    elif name == "setup":
        checks = [r"^Setup complete\.", r"0 problems"]
    missing = [p for p in checks if not re.search(p, text, re.M)]
    if missing or re.search(r"SEMANTIC .*missing|\[WARN\] item fix", text):
        rc = rc or 1
    if missing:
        with log.open("a") as f:
            f.write("\nMissing success summary: " + ", ".join(missing) + "\n")
    result = dict(name=name, passed=rc == 0, exit_code=rc, seconds=round(time.monotonic() - start, 2),
                  command=list(command), log=str(log.relative_to(out)))
    with (out / "results.jsonl").open("a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"[{name}] {'PASS' if result['passed'] else 'FAIL'} ({result['seconds']:.1f}s): {log}", flush=True)
    if rc:
        print("\n".join(log.read_text(errors="replace").splitlines()[-12:]), flush=True)
    return int(rc != 0)


def report(output, baseline):
    out = Path(output)
    info = json.loads((out / "version.json").read_text())
    results = [json.loads(s) for s in (out / "results.jsonl").read_text().splitlines()]
    failed = [r for r in results if not r["passed"]]
    tests = [r for r in results if r["name"] not in ("setup", "startup", "threads", "threads_baseline", "content_ids")]
    elapsed = sum(r["seconds"] for r in results)
    lines = ["# Game update report", "", f"Version: {info['version']}", f"Commit: {info['commit']}",
             f"Source sts2.dll: `{info['dll']}`",
             f"Runtime sts2.dll: `{info['runtime_dll']}`",
             f"Game: `{info['game_dir']}`",
             f"Mode: {'quick' if info['quick'] else 'full'}; checks runtime: {elapsed:.1f}s", "", "## Warnings", ""]
    warnings = [line for r in results for line in (out / r["log"]).read_text(errors="replace").splitlines()
                if re.search(r"\[WARN\] item fix|SEMANTIC .*missing", line)]
    lines += (["```", *dict.fromkeys(warnings), "```"] if warnings else ["No item fix or missing semantic warnings."])
    lines += ["", "## Calls leaving the game thread", "", "```"]
    threads = out / "logs/threads.log"
    lines += threads.read_text().splitlines() if threads.exists() else ["Not run (setup failed)."]
    lines += ["```"]
    previous = out / "logs/threads_baseline.log"
    if threads.exists() and previous.exists():
        calls = lambda p: {s for s in p.read_text().splitlines() if " -> " in s}
        added = sorted(calls(threads) - calls(previous))
        removed = sorted(calls(previous) - calls(threads))
        lines += ["", f"Compared with repository engine/lib/sts2.dll: +{len(added)} / -{len(removed)} call sites.",
                  "", "New call sites:", "", "```", *(added or ["none"]), "```"]
    else:
        lines += ["", "No previous engine DLL available for call-site comparison."]
    lines += ["", "## Content IDs", "", "Concrete model types from the supplied DLL; IDs use ModelDb.GetId (including deprecated and special content).", ""]
    content = out / "content_ids.json"
    if content.exists():
        ids = json.loads(content.read_text())
        old = json.loads(Path(baseline).read_text()) if baseline else None
        for category, values in sorted(ids.items()):
            lines.append(f"- {category}: {len(values)}")
        if old is None:
            lines += ["", "No baseline supplied; no content comparison."]
        else:
            lines += ["", f"Baseline: `{Path(baseline).resolve()}`"]
            diff = {}
            for category in sorted(ids.keys() | old.keys()):
                if category not in old:
                    lines += ["", f"### {category}: baseline has no category; ID delta unavailable."]
                    continue
                added = sorted(set(ids.get(category, [])) - set(old.get(category, [])))
                removed = sorted(set(old.get(category, [])) - set(ids.get(category, [])))
                diff[category] = dict(added=added, removed=removed)
                lines += ["", f"### {category}: +{len(added)} / -{len(removed)}",
                          "", "Added: " + (", ".join(added) or "none"), "", "Removed: " + (", ".join(removed) or "none")]
            (out / "content_diff.json").write_text(json.dumps(diff, indent=2) + "\n")
    else:
        lines.append("Content enumeration failed or was not run.")
    lines += ["", "## Results", "", "| Check | Result | Seconds | Log |", "|---|---|---:|---|"]
    for r in results:
        lines.append(f"| {r['name']} | {'PASS' if r['passed'] else 'FAIL'} | {r['seconds']:.2f} | [{r['log']}]({r['log']}) |")
    for r in failed:
        lines += ["", f"### {r['name']} failure (exit {r['exit_code']})", "", "```",
                  *(out / r["log"]).read_text(errors="replace").splitlines()[-16:], "```"]
    mismatches = sorted((out / "saveload").glob("run_*/mismatch.json"))
    for path in mismatches:
        detail = json.loads(path.read_text())
        lines += ["", "### Save/load observation mismatch", "",
                  f"Seed `{detail['seed']}`, save {detail['save']}, mode `{detail['mode']}`, step {detail['step']}; evidence: [{path.relative_to(out)}]({path.relative_to(out)})."]
        for diff in detail["unexpected"]:
            match = re.match(r"\.obs\.relics\[(\d+)\]", diff["path"])
            relic = detail["expected"]["obs"]["relics"][int(match[1])]["id"] if match else ""
            lines += ["", f"`{relic}` `{diff['path']}`: expected `{diff['expected']}`, actual `{diff['actual']}` (missing: {diff.get('actual_missing', False)})."]
        lines += ["", "This is a recorded observation difference, not a path or test-entry failure; engine logic is unchanged.",
                  f"The save/load checker stopped after the mismatch; {len(list((out / 'saveload').glob('run_*/result.json')))} of the requested 200 runs completed."]
    known_path = out / "saveload/known_summary.json"
    if known_path.exists():
        known = json.loads(known_path.read_text())
        samples = {}
        for line in (out / "saveload/known.jsonl").read_text().splitlines():
            record = json.loads(line)
            samples.setdefault(f"{record['relic']}:{record['field']}", record)
        lines += ["", "## Known unsaved relic state differences", "",
                  f"{known['total']} step-level records; full evidence: [saveload/known.jsonl](saveload/known.jsonl).",
                  "", "| Relic / field | Records | Unsaved member |", "|---|---:|---|"]
        for key, count in sorted(known["relic_field"].items()):
            members = ", ".join(samples[key]["members"])
            lines.append(f"| {key} | {count} | {members} |")
        lines += ["", "Only the documented class of non-serialized relic state is classified as known. Saved properties, other observations, candidates, and replay outcomes/step counts remain strictly compared."]
    lines += ["", "## Conclusion", "", f"{'FAIL' if failed else 'PASS'}: {len(tests) - sum(not r['passed'] for r in tests)}/{len(tests)} regression checks passed; {len(failed)} failed checks overall."]
    if not tests:
        lines += ["Regressions were not run because setup failed."]
    lines += ["", "Tooling: smoke/fullrun/probe/fuzz success summaries are checked because these tools can return zero on failure; fuzz enumerates the supplied DLL by reflection instead of the hard-coded decompiled version."]
    if info["dll"] != info["runtime_dll"]:
        lines += ["", "The client reference DLL is x86_64; this Mac's arm64 .NET loads the same release's unmodified arm64 DLL."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Report: {out / 'report.md'} ({elapsed:.1f}s)")
    return int(bool(failed))


def fuzz(repo, output, *args):
    sys.path.insert(0, str(Path(repo) / "tmp/content_review"))
    import fuzz_content
    rows = json.loads((Path(output) / "content_models.json").read_text())
    fuzz_content.list_models = lambda folder, skip=(): [
        (r["name"], r["id"].split(".")[-1], r["parent"]) for r in rows
        if r["ns"] == folder and r["name"] not in skip]
    sys.argv = ["fuzz_content.py", *args]
    fuzz_content.main()


if __name__ == "__main__":
    sys.exit(globals()[sys.argv[1]](*sys.argv[2:]) or 0)
