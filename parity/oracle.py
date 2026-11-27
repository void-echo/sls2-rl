#!/usr/bin/env python3
"""Drive the ParityOracle mod in the real StS2 client on this Mac.

  oracle.py build                 build the mod DLL
  oracle.py install               build + copy into <app>/Contents/MacOS/mods/ParityOracle (+ config pointing at the control dir)
  oracle.py uninstall             remove the mod folder from the game
  oracle.py run <run_dir> ...     game session(s) replaying the given recorded runs (see record.py); --chunk N splits
                                  them into sessions of N runs, each one launched, quit and restored separately
  oracle.py check <run_dir> ...   no launch: validate the run directories (action log against the headless digest,
                                  setup commands the mod supports, current schema), write the job files a run would
                                  use to tmp/parity/control_check/, check the game/settings paths, print the plan

A session backs up the user's StS2 data (settings, profiles, Steam-cloud copy) first, temporarily edits settings.save
(other mods disabled, windowed 1280x720, muted, intro skipped, full FPS in background), launches the game through
Steam, waits for the mod's done.json, quits the game and restores settings.save. The mod itself suppresses every
save/progress/history/replay write while a job runs, and the script verifies afterwards that profile files are
byte-identical to the backup.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TMP = os.path.join(ROOT, "tmp", "parity")
CONTROL = os.path.join(TMP, "control")
BUILD = os.environ.get("PARITY_BUILD_DIR", os.path.join(HERE, "mod", "bin", "Release"))
APP = os.path.expanduser("~/Library/Application Support/Steam/steamapps/common/Slay the Spire 2/SlayTheSpire2.app")
MOD_DIR = os.path.join(APP, "Contents", "MacOS", "mods", "ParityOracle")
USERDATA = os.path.expanduser("~/Library/Application Support/SlayTheSpire2")
STEAM_ID64 = os.environ.get("STS2_STEAM_ID64", "YOUR_STEAMID64")  # your SteamID64: the folder name under USERDATA/steam/
ACCOUNT = os.path.join(USERDATA, "steam", STEAM_ID64)
SETTINGS = os.path.join(ACCOUNT, "settings.save")
STEAM_ACCOUNT_ID = str(int(STEAM_ID64) - 76561197960265728) if STEAM_ID64.isdigit() else "YOUR_STEAM_ACCOUNT_ID"
STEAM_REMOTE = os.path.expanduser(f"~/Library/Application Support/Steam/userdata/{STEAM_ACCOUNT_ID}/2868840/remote")
GAME_LOG = os.path.join(USERDATA, "logs", "godot.log")
DOTNET = os.path.expanduser("~/.dotnet/dotnet")


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def build():
    env = dict(os.environ, DOTNET_ROOT=os.path.dirname(DOTNET), DOTNET_CLI_TELEMETRY_OPTOUT="1")
    r = subprocess.run([DOTNET, "build", "-c", "Release", "-o", BUILD, os.path.join(HERE, "mod", "ParityOracle.csproj")],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        raise SystemExit("mod build failed")
    print("built", os.path.join(BUILD, "ParityOracle.dll"))


def install():
    build()
    os.makedirs(MOD_DIR, exist_ok=True)
    shutil.copy2(os.path.join(BUILD, "ParityOracle.dll"), MOD_DIR)
    shutil.copy2(os.path.join(HERE, "mod", "ParityOracle.json"), MOD_DIR)
    os.makedirs(CONTROL, exist_ok=True)
    with open(os.path.join(MOD_DIR, "ParityOracle.config.json"), "w") as f:
        json.dump({"control_dir": CONTROL}, f, indent=1)
    print("installed to", MOD_DIR)


def uninstall():
    if os.path.isdir(MOD_DIR):
        shutil.rmtree(MOD_DIR)
    mods = os.path.dirname(MOD_DIR)
    if os.path.isdir(mods) and not os.listdir(mods):
        os.rmdir(mods)
    print("removed", MOD_DIR)


def game_running():
    # `ps -o command` truncates without -ww and the process name is exactly 16 chars, so check both ways.
    if subprocess.run(["pgrep", "-x", "Slay the Spire 2"], capture_output=True).returncode == 0:
        return True
    out = subprocess.run(["ps", "-axww", "-o", "command="], capture_output=True, text=True).stdout
    return any("SlayTheSpire2.app/Contents/MacOS/" in l for l in out.splitlines())


def quit_game(timeout=20, hard=False):
    if not game_running():
        return
    if not hard:
        subprocess.run(["osascript", "-e", 'quit app "Slay the Spire 2"'], capture_output=True)
        t0 = time.time()
        while game_running() and time.time() - t0 < timeout:
            time.sleep(1)
    if game_running():
        subprocess.run(["pkill", "-f", "SlayTheSpire2.app/Contents/MacOS/Slay the Spire 2"])
        time.sleep(3)
    if game_running():
        subprocess.run(["pkill", "-9", "-f", "SlayTheSpire2.app/Contents/MacOS/Slay the Spire 2"])
        time.sleep(2)


LOG_DIR = os.path.dirname(GAME_LOG)
LOG_FLOOD = 200 * 1024 * 1024   # godot.log growth that means a runaway loop (a normal session writes < 1 MB)


def godot_log_size():
    try:
        return os.path.getsize(GAME_LOG)
    except OSError:
        return 0


def drop_flooded_logs(since):
    """Delete game log files written during the session that a runaway loop blew up (they are pure spam)."""
    for fn in os.listdir(LOG_DIR):
        p = os.path.join(LOG_DIR, fn)
        try:
            st = os.stat(p)
        except OSError:
            continue
        if fn.startswith("godot") and st.st_mtime >= since and st.st_size > LOG_FLOOD:
            print(f"deleting flooded game log {fn} ({st.st_size / 1e9:.1f} GB)")
            os.remove(p)


def tree_hashes(root):
    out = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            with open(p, "rb") as f:
                out[os.path.relpath(p, root)] = hashlib.sha1(f.read()).hexdigest()
    return out


def backup():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst = os.path.join(TMP, "backups", stamp)
    os.makedirs(dst)
    shutil.copytree(ACCOUNT, os.path.join(dst, "account"))
    if os.path.isdir(STEAM_REMOTE):
        shutil.copytree(STEAM_REMOTE, os.path.join(dst, "steam_remote"))
    shutil.copytree(os.path.join(USERDATA, "mod_configs"), os.path.join(dst, "mod_configs"), dirs_exist_ok=True)
    return dst


def patch_settings(display=None):
    """display (watch sessions only): {"fullscreen": bool, "volume": float} for a person watching or recording; default windowed 1280x720, muted."""
    with open(SETTINGS) as f:
        s = json.load(f)
    ms = s.setdefault("mod_settings", {})
    for m in ms.get("mod_list", []):
        if m.get("id") != "ParityOracle":
            m["is_enabled"] = False
    ms["mods_enabled"] = True
    user_window = s.get("window_size")
    s.update({"fullscreen": False, "window_size": {"X": 1280, "Y": 720}, "volume_master": 0.0,
              "skip_intro_logo": True, "limit_fps_in_background": False})
    if display:
        s["fullscreen"] = bool(display.get("fullscreen"))
        if display.get("fullscreen") and user_window is not None:
            s["window_size"] = user_window
        s["volume_master"] = float(display.get("volume", 0.0))
    with open(SETTINGS, "w") as f:
        json.dump(s, f, indent=2)


def restore_settings(bak):
    for fn in ("settings.save", "settings.save.backup"):
        src = os.path.join(bak, "account", fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ACCOUNT, fn))


def restore_changed(bak, changed):
    """Put every file the session touched back to its backed-up bytes (or delete it if it was new)."""
    roots = {"account": ACCOUNT, "steam_remote": STEAM_REMOTE}
    for rel in changed:
        top, sub = rel.split("/", 1)
        live, saved = os.path.join(roots[top], sub), os.path.join(bak, top, sub)
        if os.path.exists(saved):
            os.makedirs(os.path.dirname(live), exist_ok=True)
            shutil.copy2(saved, live)
        elif os.path.exists(live):
            os.remove(live)
    for top, live_root in roots.items():  # drop directories the session created
        for dp, dns, fns in sorted(os.walk(live_root), key=lambda t: -len(t[0])):
            rel = os.path.relpath(dp, live_root)
            if rel != "." and not os.listdir(dp) and not os.path.exists(os.path.join(bak, top, rel)):
                os.rmdir(dp)


def verify_untouched(bak):
    changed = []
    before = tree_hashes(os.path.join(bak, "account"))
    after = tree_hashes(ACCOUNT)
    for k in sorted(set(before) | set(after)):
        if before.get(k) != after.get(k):
            changed.append("account/" + k)
    if os.path.isdir(os.path.join(bak, "steam_remote")):
        b2, a2 = tree_hashes(os.path.join(bak, "steam_remote")), tree_hashes(STEAM_REMOTE)
        for k in sorted(set(b2) | set(a2)):
            if b2.get(k) != a2.get(k):
                changed.append("steam_remote/" + k)
    return changed


def session_timeout(run_dirs):
    """Hard cap for one session: launch + per-run overhead + ~0.6 s per recorded step (the client needs ~0.2 s)."""
    steps = 0
    for d in run_dirs:
        try:
            with open(os.path.join(d, "actions.jsonl")) as f:
                steps += sum(1 for _ in f)
        except OSError:
            pass
    return 240 + 30 * len(run_dirs) + 0.6 * steps


def make_job(run_dirs, control, out_name="client.jsonl", step_timeout=30, max_steps=None, watch=None):
    """The job.json the mod reads (parity/mod/src/Driver.cs RunJob / RunOne)."""
    runs = []
    for d in run_dirs:
        d = os.path.abspath(d)
        with open(os.path.join(d, "run.json")) as f:
            spec = json.load(f)
        r = {"id": spec["id"], "character": spec["character"], "ascension": spec["ascension"], "seed": spec["seed"],
             "actions": os.path.join(d, "actions.jsonl"), "out": os.path.join(d, out_name), "setup": spec.get("setup") or []}
        if max_steps:
            r["max_steps"] = max_steps
        runs.append(r)
    job = {"runs": runs, "done": os.path.join(control, "done.json"), "quit_when_done": True, "step_timeout": step_timeout}
    if watch:
        job["watch"] = {k: v for k, v in watch.items() if k != "display"}  # display is applied to settings.save, not read by the mod
    return job


def watch_paused_seconds(control):
    try:
        with open(os.path.join(control, "paused.json")) as f:
            status = json.load(f)
        seconds = status["paused_seconds"]
        if status["paused"]:
            seconds += max(0, min(time.time(), status["updated_at"] + 5) - status["paused_since"])
        return seconds
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def run_session(run_dirs, out_name="client.jsonl", timeout=None, step_timeout=30, max_steps=None, watchdog=60, reinstall=True,
                watch=None):
    if game_running():
        raise SystemExit("the game is already running; quit it first")
    if timeout is None:
        timeout = session_timeout(run_dirs)
        if watch:  # animations at the chosen speed plus the pause before each input
            timeout += (watch.get("step_delay", 0) + 3) * sum(
                sum(1 for _ in open(os.path.join(d, "actions.jsonl"))) for d in run_dirs)
    if reinstall or not os.path.exists(os.path.join(MOD_DIR, "ParityOracle.dll")):
        install()
    os.makedirs(CONTROL, exist_ok=True)
    watching = bool(watch and watch.get("step_delay", 0) > 0)
    for fn in ("done.json", "job.running.json") + (("paused.json",) if watching else ()):
        p = os.path.join(CONTROL, fn)
        if os.path.exists(p):
            os.remove(p)
    job = make_job(run_dirs, CONTROL, out_name, step_timeout, max_steps, watch)
    done = job["done"]
    session_start = time.time()
    bak = backup()
    print(f"backup at {bak}; session timeout {timeout:.0f}s, watchdog {watchdog:.0f}s", flush=True)
    log_off = os.path.getsize(os.path.join(CONTROL, "oracle.log")) if os.path.exists(os.path.join(CONTROL, "oracle.log")) else 0
    try:
        patch_settings((watch or {}).get("display"))
        with open(os.path.join(CONTROL, "job.json"), "w") as f:
            json.dump(job, f, indent=1)
        subprocess.run(["open", "steam://run/2868840"])
        for _ in range(40):  # sanity-check process detection once the game is up
            time.sleep(1)
            if game_running():
                print("game process detected")
                break
        else:
            print("WARNING: game process not detected after 40 s")
        t0 = time.time()
        log0 = 0   # godot.log is recreated at launch
        last = ""
        last_size, last_progress = -1, time.time()
        pause0 = watch_paused_seconds(CONTROL) if watching else 0
        last_pause = 0
        while time.time() - t0 - (max(last_pause, watch_paused_seconds(CONTROL) - pause0) if watching else 0) < timeout:
            time.sleep(3)
            if os.path.exists(done):
                break
            if watching:
                paused_for = max(last_pause, watch_paused_seconds(CONTROL) - pause0)
                last_progress += paused_for - last_pause
                last_pause = paused_for
            lp = os.path.join(CONTROL, "oracle.log")
            size = os.path.getsize(lp) if os.path.exists(lp) else 0
            if size != last_size:
                last_size, last_progress = size, time.time()
            elif time.time() - last_progress > watchdog:
                print(f"watchdog: no mod progress for {watchdog:.0f}s, killing the game")
                quit_game(timeout=5)
                break
            if os.path.exists(lp):
                with open(lp) as f:
                    f.seek(log_off)
                    tail = f.read().strip().splitlines()
                if tail and tail[-1] != last:
                    last = tail[-1]
                    print("  ", last[:160], flush=True)
            if godot_log_size() - log0 > LOG_FLOOD:
                print("godot.log is flooding (runaway loop in the client), killing the game")
                quit_game(hard=True)
                break
            if time.time() - t0 > 120 and not game_running() and not os.path.exists(done):
                misses = locals().get("misses", 0) + 1
                if misses >= 3:
                    print("game exited without done.json")
                    break
            else:
                misses = 0
        else:
            print("session timeout")
    finally:
        time.sleep(3)
        quit_game()
        drop_flooded_logs(session_start)
        restore_settings(bak)
        for fn in ("job.json",):
            p = os.path.join(CONTROL, fn)
            if os.path.exists(p):
                os.remove(p)
        changed = verify_untouched(bak)
        print("user data changed by the session:", changed or "nothing")
        if changed:
            restore_changed(bak, changed)
            print("restored from backup; still different:", verify_untouched(bak) or "nothing")
    if game_running():
        print("WARNING: game still running after the session; killing it")
        quit_game()
    if os.path.exists(done):
        with open(done) as f:
            print(f.read()[:3000])


# ------------------------------------------------------------------ no-launch check

CHARACTERS = {"Ironclad", "Silent", "Defect", "Regent", "Necrobinder", "Fool"}
# Scenario commands parity/mod/src/Scenario.cs performs in the client (real game paths only).
SCENARIO_CMDS = {"obtain_relics", "dbg_obtain_relic", "dbg_procure_potion", "dbg_add_card", "rl_enter_room", "enter_room",
                 "dbg_enter_act"}
CHECK_CONTROL = os.path.join(TMP, "control_check")
CLIENT_S_PER_STEP = 0.25     # measured ~0.21 s per decision in the v1 re-validation
SESSION_OVERHEAD_S = 100     # launch through Steam, main menu, per-run menu round trips, quit, backup/restore


def _jsonl(path):
    out = []
    with open(path) as f:
        for n, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as ex:
                    raise ValueError(f"{os.path.basename(path)} line {n + 1}: {ex}")
    return out


def check_run_dir(d):
    """Problems (the session would be wasted) and warnings for one recorded run directory; also its step count."""
    problems, warnings = [], []
    name = os.path.basename(os.path.normpath(d))
    for fn in ("run.json", "actions.jsonl", "headless.jsonl"):
        if not os.path.exists(os.path.join(d, fn)):
            return [f"{name}: {fn} missing"], warnings, 0
    try:
        with open(os.path.join(d, "run.json")) as f:
            spec = json.load(f)
        actions = _jsonl(os.path.join(d, "actions.jsonl"))
        head = _jsonl(os.path.join(d, "headless.jsonl"))
    except (ValueError, json.JSONDecodeError) as ex:
        return [f"{name}: {ex}"], warnings, 0
    for k in ("id", "character", "ascension", "seed"):
        if k not in spec:
            problems.append(f"{name}: run.json lacks {k}")
    if spec.get("character") not in CHARACTERS:
        problems.append(f"{name}: unknown character {spec.get('character')}")
    if not isinstance(spec.get("ascension"), int) or not 0 <= spec["ascension"] <= 10:
        problems.append(f"{name}: bad ascension {spec.get('ascension')}")
    for cmd in spec.get("setup") or []:
        if not isinstance(cmd, dict) or cmd.get("cmd") not in SCENARIO_CMDS:
            problems.append(f"{name}: setup command the mod cannot perform: {cmd}")
    steps = {h["step"]: h for h in head if "decision" in h}
    errors = [h for h in head if "error" in h]
    if errors:
        warnings.append(f"{name}: headless run ended with an engine error at step {errors[0]['step']} (compared up to it)")
    for n, a in enumerate(actions):
        if a.get("step") != n:
            problems.append(f"{name}: action {n} has step {a.get('step')}")
            break
        h = steps.get(n)
        if h is None:
            problems.append(f"{name}: action {n} has no headless step")
            break
        cand = a.get("cand")
        if not isinstance(cand, dict) or "k" not in cand:
            problems.append(f"{name}: action {n} has no candidate descriptor")
            break
        i = a.get("i")
        if h.get("decision") != a.get("decision") or not isinstance(i, int) or not 0 <= i < len(h["cands"]) or h["cands"][i] != cand:
            problems.append(f"{name}: action {n} does not match headless step {n} (decision/index/descriptor)")
            break
        if a.get("kind_pos") != sum(1 for c in h["cands"][:i] if c["k"] == cand["k"]):
            problems.append(f"{name}: action {n} kind_pos inconsistent")
            break
    if len(steps) < len(actions):
        problems.append(f"{name}: {len(actions)} actions but {len(steps)} headless steps")
    # Schema of the engine the oracle mirrors (the shared observation, engine/src/Shared/ObsShared.cs): stale
    # recordings would diverge on schema, not on the game.
    stale = set()
    for h in steps.values():
        obs = h.get("obs") or {}
        g = obs.get("g") or {}
        for k in ("removals_used", "odds", "act_rooms"):
            if k not in g:
                stale.add("g." + k)
        cb = obs.get("combat")
        if cb is not None:
            for k in ("played", "turn", "enc", "draw_top", "pturn"):
                if k not in cb:
                    stale.add("combat." + k)
        for p in obs.get("potions") or []:
            if "v" not in p:
                stale.add("potion v")
        for c in h.get("cands") or []:
            if c.get("k") == "reward" and c.get("rt") == "Card" and "cards" not in c:
                stale.add("reward cards")
    if stale:
        problems.append(f"{name}: recorded with an older engine (missing {', '.join(sorted(stale))}); re-record it")
    return problems, warnings, len(actions)


def check(run_dirs, out_name="client.jsonl", step_timeout=30, max_steps=None, chunk=0, verbose=True):
    """Everything a run would do short of launching the game. Returns (ok, plan)."""
    problems, warnings = [], []
    if not os.path.exists(DOTNET):
        problems.append(f"dotnet missing at {DOTNET} (needed to build the mod)")
    data_dir = os.path.join(APP, "Contents", "Resources", "data_sts2_macos_arm64")
    for p in (APP, os.path.join(data_dir, "sts2.dll"), os.path.join(data_dir, "0Harmony.dll")):
        if not os.path.exists(p):
            problems.append(f"game file missing: {p}")
    try:
        with open(SETTINGS) as f:
            st = json.load(f)
        if not isinstance(st.get("mod_settings", {}), dict):
            problems.append("settings.save: mod_settings is not an object")
        others = [m.get("id") for m in st.get("mod_settings", {}).get("mod_list", []) if m.get("is_enabled") and m.get("id") != "ParityOracle"]
        if others:
            warnings.append(f"settings.save: {len(others)} other mod(s) enabled; each session disables them and restores them afterwards")
    except (OSError, json.JSONDecodeError) as ex:
        problems.append(f"settings.save unreadable: {ex}")
    for p in (ACCOUNT, USERDATA):
        if not os.path.isdir(p):
            problems.append(f"user data directory missing: {p}")
    if game_running():
        warnings.append("the game is running now; a session refuses to start until it is quit")
    if os.path.exists(os.path.join(CONTROL, "job.json")):
        warnings.append(f"{CONTROL}/job.json exists: the mod would run it at the next game start; a session overwrites it")
    total_steps, per_dir = 0, {}
    for d in run_dirs:
        p, w, n = check_run_dir(d)
        problems += p
        warnings += w
        per_dir[d] = n
        total_steps += n
    k = chunk or len(run_dirs) or 1
    chunks = [run_dirs[i:i + k] for i in range(0, len(run_dirs), k)]
    os.makedirs(CHECK_CONTROL, exist_ok=True)
    for j, c in enumerate(chunks):
        job = make_job(c, CONTROL, out_name, step_timeout, max_steps)
        text = json.dumps(job, indent=1)
        if json.loads(text) != job:
            problems.append(f"session {j + 1}: job does not round-trip through JSON")
        for r in job["runs"]:
            for key in ("actions", "out"):
                if not os.path.isabs(r[key]):
                    problems.append(f"{r['id']}: {key} path is not absolute")
            if not os.path.isdir(os.path.dirname(r["out"])):
                problems.append(f"{r['id']}: output directory missing")
        with open(os.path.join(CHECK_CONTROL, f"job_session{j + 1:02d}.json"), "w") as f:
            f.write(text)
    est = sum(SESSION_OVERHEAD_S + CLIENT_S_PER_STEP * sum(per_dir[d] for d in c) for c in chunks)
    cap = sum(session_timeout(c) for c in chunks)
    plan = {"runs": len(run_dirs), "steps": total_steps, "sessions": len(chunks), "estimated_s": est, "hard_cap_s": cap,
            "job_files": CHECK_CONTROL}
    if verbose:
        for w in warnings:
            print("warning:", w)
        for p in problems:
            print("PROBLEM:", p)
        print(f"check: {len(run_dirs)} runs, {total_steps} steps, {len(chunks)} session(s); estimated {est / 60:.0f} min "
              f"(hard cap {cap / 60:.0f} min); job files in {CHECK_CONTROL}; {len(problems)} problem(s), {len(warnings)} warning(s)")
    return not problems, plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("install")
    sub.add_parser("uninstall")
    r = sub.add_parser("run")
    r.add_argument("run_dirs", nargs="+")
    r.add_argument("--timeout", type=float, default=None, help="hard cap per session (s); default scales with the steps")
    r.add_argument("--chunk", type=int, default=0, help="runs per game session (0: all in one)")
    r.add_argument("--step-timeout", type=float, default=30, help="mod gives up on a decision after this long (s)")
    r.add_argument("--watchdog", type=float, default=60, help="quit the game if the mod log is silent this long (s)")
    r.add_argument("--max-steps", type=int, default=None)
    r.add_argument("--out-name", default="client.jsonl")
    r.add_argument("--watch", default="", help="for a person watching: animation speed Normal or Fast (default Instant)")
    r.add_argument("--fullscreen", action="store_true", help="with --watch: run the game fullscreen (for recording); default windowed 1280x720")
    r.add_argument("--volume", type=float, default=0.0, help="with --watch: master volume 0-1 (default muted)")
    r.add_argument("--step-delay", type=float, default=0.0,
                   help="with --watch and >0: delay before each input (s), show shops; Space pause/resume, R replay this node, B previous node; paused time suspends watchdog/timeout")
    c = sub.add_parser("check", help="validate run dirs and job files without launching the game")
    c.add_argument("run_dirs", nargs="+")
    c.add_argument("--chunk", type=int, default=0)
    c.add_argument("--step-timeout", type=float, default=30)
    c.add_argument("--max-steps", type=int, default=None)
    c.add_argument("--out-name", default="client.jsonl")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "install":
        install()
    elif a.cmd == "uninstall":
        uninstall()
    elif a.cmd == "check":
        dirs = [d for d in a.run_dirs if os.path.isdir(d)]
        ok, _ = check(dirs, a.out_name, a.step_timeout, a.max_steps, a.chunk)
        raise SystemExit(0 if ok else 1)
    elif a.cmd == "run":
        dirs = [d for d in a.run_dirs if os.path.isdir(d)]
        k = a.chunk or len(dirs)
        for i in range(0, len(dirs), k):
            print(f"=== session {i // k + 1}/{(len(dirs) + k - 1) // k}: {len(dirs[i:i + k])} runs", flush=True)
            watch = {"fast_mode": a.watch, "step_delay": a.step_delay,
                     "display": {"fullscreen": a.fullscreen, "volume": a.volume}} if a.watch else None
            run_session(dirs[i:i + k], a.out_name, a.timeout, a.step_timeout, a.max_steps, a.watchdog, watch=watch)
            time.sleep(5)


if __name__ == "__main__":
    main()
