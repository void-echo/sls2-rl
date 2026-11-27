#!/usr/bin/env python3
"""Replay room-entry saves through the client load path, then check common random numbers.

python3 tools/saveload_check.py --runs 200 --workers 4
python3 tools/saveload_check.py --runs 200 --replay-only --resume-from OLD_OUT --out NEW_OUT
All traces, save data, mismatches and summaries go under --out.
Runs with fewer than three room saves replay every available save, then an extra
seed in the same character/ascension stratum supplies the three-point check.
"""
import argparse
import collections
import concurrent.futures as cf
import gzip
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_runner import Engine, choose, terminal_problem  # noqa: E402

CHARS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder", "Fool"]
FIELDS = ("decision", "obs", "cands", "done", "victory")
STOP = None
RELIC_METADATA = {}
UNSAVED_RELIC_VARS = {("RELIC.WONGOS_MYSTERY_TICKET", "RemainingCombats")}

REFLECTION_SOURCE = r'''
using System.Collections;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Loader;
using System.Text.Json;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Runs;

internal static class Program
{
    const BindingFlags All = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
    static readonly Dictionary<short, OpCode> Codes = typeof(OpCodes).GetFields().Where(f => f.FieldType == typeof(OpCode))
        .Select(f => (OpCode)f.GetValue(null)!).ToDictionary(o => o.Value);

    static void Main(string[] args)
    {
        var engine = args[0];
        var lib = args[1];
        AssemblyLoadContext.Default.Resolving += (ctx, name) =>
        {
            var paths = new[] { Path.Combine(engine, "src/Sts2Headless/bin/Debug/net9.0", name.Name + ".dll"),
                                Path.Combine(lib, name.Name + ".dll") };
            var path = paths.FirstOrDefault(File.Exists);
            return path == null ? null : ctx.LoadFromAssemblyPath(path);
        };
        Run(args[2]);
    }

    static bool Saved(MemberInfo m) => m.CustomAttributes.Any(a => a.AttributeType == typeof(SavedPropertyAttribute));
    static string Name(FieldInfo f) => f.Name.StartsWith('<') ? f.Name[1..f.Name.IndexOf('>')] : f.Name.TrimStart('_');
    static bool Card(Type t)
    {
        static bool One(Type x) => typeof(CardModel).IsAssignableFrom(x) || x == typeof(SerializableCard);
        return One(t) || (t.IsArray && One(t.GetElementType()!)) || (t.IsGenericType && t.GetGenericArguments().Length == 1
            && One(t.GetGenericArguments()[0]) && typeof(IEnumerable).IsAssignableFrom(t));
    }
    static bool Number(Type t) => t == typeof(int) || t == typeof(long) || t == typeof(float) || t == typeof(double) || t == typeof(decimal);

    static HashSet<FieldInfo> Reads(MethodBase method, HashSet<Type> hierarchy, HashSet<MethodBase> seen)
    {
        var fields = new HashSet<FieldInfo>();
        if (!seen.Add(method)) return fields;
        var il = method.GetMethodBody()?.GetILAsByteArray();
        if (il == null) return fields;
        for (int i = 0; i < il.Length;)
        {
            short value = il[i++];
            if (value == 0xfe) value = (short)(0xfe00 | il[i++]);
            var op = Codes[value];
            var operand = op.OperandType;
            var types = method.DeclaringType?.GetGenericArguments();
            var methods = method.IsGenericMethod ? method.GetGenericArguments() : null;
            if (operand == OperandType.InlineField)
            {
                var field = method.Module.ResolveField(BitConverter.ToInt32(il, i), types, methods)!;
                if (!field.IsStatic && hierarchy.Contains(field.DeclaringType!)) fields.Add(field);
            }
            if (operand == OperandType.InlineMethod)
            {
                var call = method.Module.ResolveMethod(BitConverter.ToInt32(il, i), types, methods)!;
                if (hierarchy.Contains(call.DeclaringType!)) fields.UnionWith(Reads(call, hierarchy, seen));
            }
            i += operand switch
            {
                OperandType.InlineNone => 0,
                OperandType.ShortInlineBrTarget or OperandType.ShortInlineI or OperandType.ShortInlineVar => 1,
                OperandType.InlineVar => 2,
                OperandType.InlineI8 or OperandType.InlineR => 8,
                OperandType.InlineSwitch => 4 + 4 * BitConverter.ToInt32(il, i),
                _ => 4,
            };
        }
        return fields;
    }

    static void Run(string output)
    {
        var relics = new SortedDictionary<string, object>();
        foreach (var type in typeof(RelicModel).Assembly.GetTypes().Where(t => !t.IsAbstract && !t.ContainsGenericParameters
                     && t != typeof(RelicModel) && typeof(RelicModel).IsAssignableFrom(t)))
        {
            var hierarchy = new HashSet<Type>();
            for (var t = type; t != null && t != typeof(object); t = t.BaseType) hierarchy.Add(t);
            var own = new List<Type>();
            for (var t = type; t != typeof(RelicModel); t = t.BaseType!) own.Add(t);
            var props = hierarchy.SelectMany(t => t.GetProperties(All)).ToArray();
            var saved = props.Where(Saved).ToArray();
            var savedReads = saved.ToDictionary(p => p, p => p.GetMethod == null ? new HashSet<FieldInfo>()
                : Reads(p.GetMethod, hierarchy, new HashSet<MethodBase>()));
            var fields = own.SelectMany(t => t.GetFields(All)).ToArray();
            string[] SavedBy(FieldInfo f) => saved.Where(p => string.Equals(p.Name, Name(f), StringComparison.OrdinalIgnoreCase)
                || savedReads[p].Contains(f)).Select(p => p.DeclaringType!.FullName + "::" + p.Name).Distinct().ToArray();
            object Describe(FieldInfo f) => new { member = f.DeclaringType!.FullName + "::" + f.Name,
                property = props.Where(p => string.Equals(p.Name, Name(f), StringComparison.OrdinalIgnoreCase)).Select(p => p.Name).ToArray(),
                saved_by = SavedBy(f) };
            var data = new SortedDictionary<string, object>();
            foreach (var f in fields)
            {
                if (f.IsInitOnly || f.IsLiteral || f.Name.Contains('<') || SavedBy(f).Length != 0) continue;
                var t = Nullable.GetUnderlyingType(f.FieldType) ?? f.FieldType;
                bool sum = typeof(IDictionary).IsAssignableFrom(t) && t.IsGenericType && Number(t.GetGenericArguments()[^1]);
                if (Number(t) || t == typeof(bool) || t == typeof(string) || t.IsEnum || sum
                    || (typeof(AbstractModel).IsAssignableFrom(t) && !typeof(CardModel).IsAssignableFrom(t)))
                    data[Name(f) + (sum ? ".sum" : "")] = Describe(f);
            }
            var cards = fields.Where(f => Card(f.FieldType)).ToArray();
            var unresolved = saved.Where(p => Card(p.PropertyType) && !cards.Any(f => savedReads[p].Contains(f)
                || string.Equals(p.Name, Name(f), StringComparison.OrdinalIgnoreCase))).Select(p => p.Name).ToArray();
            var status = typeof(RelicModel).GetField("_status", All)!;
            var dynamicVars = typeof(RelicModel).GetField("_dynamicVars", All)!;
            relics[ModelDb.GetId(type).ToString()] = new { type = type.FullName, d = data,
                status = new { allowed = SavedBy(status).Length == 0, member = Describe(status) },
                dynamic_vars = new { allowed = SavedBy(dynamicVars).Length == 0, member = Describe(dynamicVars) },
                cards = new { allowed = cards.Length > 0 && cards.All(f => SavedBy(f).Length == 0) && unresolved.Length == 0,
                    members = cards.Select(Describe).ToArray(), unresolved_saved = unresolved },
                saved_properties = saved.Select(p => p.DeclaringType!.FullName + "::" + p.Name).ToArray() };
        }
        File.WriteAllText(output, JsonSerializer.Serialize(new { game_mvid = typeof(RelicModel).Module.ModuleVersionId, relics },
            new JsonSerializerOptions { WriteIndented = true }));
    }
}
'''


def reflect_relics(engine, out):
    engine = Path(engine).resolve()
    lib = Path(os.environ.get("STS2_LIB") or engine / "lib").expanduser().resolve()
    directory = out / "reflection"
    directory.mkdir(parents=True, exist_ok=True)
    project = ET.Element("Project", Sdk="Microsoft.NET.Sdk")
    group = ET.SubElement(project, "PropertyGroup")
    for key, value in {"OutputType": "Exe", "TargetFramework": "net9.0", "ImplicitUsings": "enable", "Nullable": "enable"}.items():
        ET.SubElement(group, key).text = value
    refs = ET.SubElement(project, "ItemGroup")
    for name, path in {"sts2": lib / "sts2.dll", "GodotSharp": engine / "src/Sts2Headless/bin/Debug/net9.0/GodotSharp.dll"}.items():
        ET.SubElement(ET.SubElement(refs, "Reference", Include=name), "HintPath").text = str(path)
    ET.ElementTree(project).write(directory / "Reflect.csproj", encoding="unicode")
    (directory / "Program.cs").write_text(REFLECTION_SOURCE)
    dotnet = str(Path.home() / ".dotnet/dotnet")
    env = dict(os.environ, DOTNET_ROOT=str(Path.home() / ".dotnet"))
    output = (out / "relic_metadata.json").resolve()
    with (directory / "build.log").open("w") as log:
        subprocess.run([dotnet, "build", str(directory / "Reflect.csproj"), "-o", str(directory / "bin")],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (directory / "run.log").open("w") as log:
        subprocess.run([dotnet, str(directory / "bin/Reflect.dll"), str(engine), str(lib), str(output)],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return str(output)


def init_worker(stop, metadata_path):
    global STOP, RELIC_METADATA
    STOP = stop
    RELIC_METADATA = json.loads(Path(metadata_path).read_text())["relics"]


def state(r):
    return json.dumps([r.get(k) for k in FIELDS], sort_keys=True, separators=(",", ":"))


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(a.keys() | b.keys()):
            if k not in a or k not in b:
                out.append({"path": f"{path}.{k}", "expected": a.get(k), "actual": b.get(k),
                            "expected_missing": k not in a, "actual_missing": k not in b})
            else:
                out.extend(differences(a[k], b[k], f"{path}.{k}"))
        return out
    if isinstance(a, list) and isinstance(b, list):
        out = [] if len(a) == len(b) else [{"path": path + ".length", "expected": len(a), "actual": len(b)}]
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(differences(x, y, f"{path}[{i}]"))
        return out
    equal = type(a) is type(b) and a == b
    if equal and isinstance(a, float):
        equal = json.dumps(a) == json.dumps(b)
    return [] if equal else [{"path": path, "expected": a, "actual": b}]


def known_difference(diff, expected, actual):
    match = re.fullmatch(r"\.obs\.relics\[(\d+)\]\.(d\.(.+)|cards(?:\[\d+\]|\.length)?|status|v\.([^.]+))", diff["path"])
    if match is None:
        return None
    pair = []
    for response in (expected, actual):
        relics = response.get("obs", {}).get("relics", [])
        index = int(match[1])
        if index >= len(relics):
            return None
        pair.append(relics[index])
    relic = pair[0].get("id")
    if relic != pair[1].get("id") or relic not in RELIC_METADATA:
        return None
    metadata = RELIC_METADATA[relic]
    if match[4] is not None:
        field = match[4]
        dynamic_vars = metadata["dynamic_vars"]
        if (relic, field) not in UNSAVED_RELIC_VARS or not dynamic_vars["allowed"]:
            return None
        return {"relic": relic, "field": "v." + field, "members": [dynamic_vars["member"]["member"]],
                "expected": pair[0]["v"].get(field), "actual": pair[1]["v"].get(field)}
    if match[2] == "status":
        status = metadata["status"]
        if not status["allowed"]:
            return None
        return {"relic": relic, "field": "status", "members": [status["member"]["member"]],
                "expected": pair[0].get("status"), "actual": pair[1].get("status"),
                "expected_missing": "status" not in pair[0], "actual_missing": "status" not in pair[1]}
    field = match[3]
    if field is not None:
        member = metadata["d"].get(field)
        if member is None or member["saved_by"]:
            return None
        values = [r.get("d", {}) for r in pair]
        return {"relic": relic, "field": "d." + field, "members": [member["member"]],
                "expected": values[0].get(field), "actual": values[1].get(field),
                "expected_missing": field not in values[0], "actual_missing": field not in values[1]}
    cards = metadata["cards"]
    if not cards["allowed"] or not cards["members"] or any(m["saved_by"] for m in cards["members"]):
        return None
    return {"relic": relic, "field": "cards", "members": [m["member"] for m in cards["members"]],
            "expected": pair[0].get("cards"), "actual": pair[1].get("cards"),
            "expected_missing": "cards" not in pair[0], "actual_missing": "cards" not in pair[1]}


def expand_relic_fields(diffs):
    expanded = []
    for diff in diffs:
        if re.fullmatch(r"\.obs\.relics\[\d+\]\.d", diff["path"]):
            a, b = diff["expected"], diff["actual"]
            if (a is None or isinstance(a, dict)) and (b is None or isinstance(b, dict)) and (a or b):
                expanded.extend(differences(a or {}, b or {}, diff["path"]))
                continue
        expanded.append(diff)
    return expanded


def require(r):
    if r.get("type") != "rl":
        raise RuntimeError(json.dumps(r))
    if r.get("done"):
        problem = terminal_problem(r)
        if problem:
            raise RuntimeError(problem)
    elif not r.get("cands"):
        raise RuntimeError("no legal actions")
    return r


def checked(eng, command):
    if STOP is not None and STOP.is_set():
        raise RuntimeError("stopped after a save/load failure in another worker")
    return require(eng.send(command, 120))


def fail_comparison(expected, actual, out, context, diffs, unexpected):
    detail = dict(context, differences=diffs, unexpected=unexpected, expected=expected, actual=actual)
    (out / "mismatch.json").write_text(json.dumps(detail, indent=2))
    if STOP is not None:
        STOP.set()
    raise RuntimeError(f"save/load mismatch: {context}; {unexpected[:4]}")


def compare(expected, actual, out, context):
    if state(expected) != state(actual):
        diffs = expand_relic_fields(differences({k: expected.get(k) for k in FIELDS}, {k: actual.get(k) for k in FIELDS}))
        known, unexpected = {}, []
        for diff in diffs:
            member = known_difference(diff, expected, actual)
            if member is None:
                unexpected.append(diff)
            else:
                key = (re.match(r"\.obs\.relics\[(\d+)\]", diff["path"])[1], member["field"])
                known.setdefault(key, dict(diff, **member))
        if known:
            g = expected["obs"]["g"]
            with (out / "known.jsonl").open("a") as f:
                for d in known.values():
                    record = dict(context, type="known", pattern="unsaved_relic_member", run=out.name,
                                  act=g["act"], floor=g["floor"], room=g["room"], decision=expected["decision"], **d)
                    f.write(json.dumps(record) + "\n")
        if diffs and not unexpected:
            return
        fail_comparison(expected, actual, out, context, diffs, unexpected)


def ending(response, prefix=0):
    outcome = "running" if not response["done"] else "win" if response["victory"] else (
        "abandon" if response["info"].get("abandoned") else "loss")
    return {"outcome": outcome, "steps": prefix + response["info"]["steps"]}


def replay(eng, point, records, commands, out, fresh, stats, end=None):
    command = {"cmd": "rl_load", "data": point["save"]["data"]} if fresh else {"cmd": "rl_load", "id": point["save"]["id"]}
    t0 = time.perf_counter()
    r = checked(eng, command)
    if not fresh:
        stats["load_ms"].append((time.perf_counter() - t0) * 1000)
    start = point["index"]
    context = {"seed": point["seed"], "save": point["save"]["id"], "mode": "fresh" if fresh else "same", "step": start}
    compare(records[start], r, out, context)
    stats["compared_states"] += 1
    stop = len(commands) if end is None else end
    for i in range(start, stop):
        r = checked(eng, commands[i])
        context["step"] = i + 1
        compare(records[i + 1], r, out, context)
        stats["compared_steps"] += 1
        stats["compared_states"] += 1
    if end is None:
        original = ending(records[-1])
        restored = ending(r, sum(c["cmd"] == "rl_act" for c in commands[:start]))
        record = dict(context, original=original, restored=restored, equal=original == restored,
                      replayed_commands=stop - start, total_commands=stop)
        with (out / "endings.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if original != restored:
            diffs = differences(original, restored, ".ending")
            fail_comparison(records[-1], r, out, context, diffs, diffs)
        stats["ending_checks"] += 1
    return r


def card_screen(r):
    return (r["decision"] == "rewards" and r["obs"]["ctx"].get("terminal")
            and any(c["k"] == "reward" and c.get("rt") == "Card" for c in r["cands"]))


def reward_branch(eng, point, target, records, commands, out, pick, stats):
    r = replay(eng, point, records, commands, out, False, stats, end=target)
    reward = next(i for i, c in enumerate(r["cands"]) if c["k"] == "reward" and c.get("rt") == "Card")
    r = checked(eng, {"cmd": "rl_act", "i": reward})
    choices = [i for i, c in enumerate(r["cands"]) if c["k"] == "reward_card"]
    if len(choices) < 2:
        return None
    chosen = r["cands"][choices[pick]]
    r = checked(eng, {"cmd": "rl_act", "i": choices[pick]})
    for _ in range(100):
        if r["done"]:
            return {"card": chosen, "enc": None}
        if r["decision"] == "map_select":
            maps = [i for i, c in enumerate(r["cands"]) if c["k"] == "map"]
            if not maps:
                raise RuntimeError("no next map coordinate")
            r = checked(eng, {"cmd": "rl_act", "i": maps[0]})
            combat = r["obs"].get("combat")
            g = r["obs"]["g"]
            encounter = combat["enc"] if combat else g["act_rooms"][-1]["id"] if g["room"] in ("Monster", "Elite", "Boss") else None
            return {"card": chosen, "enc": encounter, "room": g["room"]}
        proceed = [i for i, c in enumerate(r["cands"]) if c["k"] in ("proceed", "skip", "leave", "confirm")]
        r = checked(eng, {"cmd": "rl_act", "i": proceed[0] if proceed else 0})
    raise RuntimeError("reward branch did not reach the next room")


def run_one(eng, args, n, stats, quotas, attempt=0):
    suffix = f"_retry_{attempt}" if attempt else ""
    out = Path(args.out) / f"run_{n:04d}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "known.jsonl").unlink(missing_ok=True)
    (out / "mismatch.json").unlink(missing_ok=True)
    (out / "endings.jsonl").unlink(missing_ok=True)
    rng = random.Random(10000 + n + attempt * 100000)
    spec = {"cmd": "rl_start", "character": CHARS[n % len(CHARS)], "ascension": (0, 10)[(n // len(CHARS)) % 2], "seed": f"{args.seed_prefix}_{n}{suffix}"}
    commands, records, points = [], [], []
    seen = 0
    r = checked(eng, spec)
    with gzip.open(out / "trace.jsonl.gz", "wt") as trace:
        previous = spec
        for step in range(args.max_steps):
            records.append(r)
            trace.write(json.dumps({"command": previous, "response": r}, separators=(",", ":")) + "\n")
            latest = (r.get("info") or {}).get("save")
            if latest and latest["id"] > seen:
                for save_id in range(seen + 1, latest["id"] + 1):
                    saved = eng.send({"cmd": "rl_save_get", "id": save_id})
                    if saved.get("type") != "save":
                        raise RuntimeError(str(saved))
                    stats["data_bytes"].append(len(saved["data"].encode()))
                    stats["save_kinds"][saved["kind"]] += 1
                    stats["response_bytes"].append(len(json.dumps(saved, separators=(",", ":")).encode()))
                    point = {"save": saved, "index": len(commands), "seed": spec["seed"]}
                    points.append(point)
                seen = latest["id"]
            if r["done"]:
                break
            if step == 0 and (n // 10) % 4 == 0:
                previous = {"cmd": "set_player", "hp": 9999, "max_hp": 9999}
            else:
                previous = {"cmd": "rl_act", "i": choose(rng, r)}
            commands.append(previous)
            r = checked(eng, previous)
        else:
            raise RuntimeError("step cap reached")
    room_points = [p for p in points if p["save"]["kind"] == "room"]
    eligible = len(room_points) >= 3
    stats["attempts"] += 1
    stats["short_runs"] += int(not eligible)
    stats["runs"] += int(eligible)
    stats["baseline_steps"] += len(commands)
    if eligible:
        stats["characters"][spec["character"]] += 1
        stats["ascensions"][str(spec["ascension"])] += 1
        stats["god_runs"] += int((n // 10) % 4 == 0)
    selected = rng.sample(room_points, 3) if eligible else list(room_points)
    # Add the points that expand room/act coverage, while retaining three random points in every run.
    covered = set(stats["coverage"])
    for p in selected:
        obs = records[p["index"]]["obs"]
        covered.add(f"{obs['g']['act']}:{obs['g']['room']}:{p['save']['room']}")
    for p in room_points:
        obs = records[p["index"]]["obs"]
        key = f"{obs['g']['act']}:{obs['g']['room']}:{p['save']['room']}"
        if key not in covered:
            selected.append(p)
            covered.add(key)
    for p in selected:
        (out / f"save_{p['save']['id']}.json").write_text(json.dumps(p["save"]))
        replay(eng, p, records, commands, out, False, stats)
        fresh = Engine(args.engine, timeout=120, stderr_path=str(out / "fresh.stderr"))
        try:
            replay(fresh, p, records, commands, out, True, stats)
        finally:
            fresh.close()
        obs = records[p["index"]]["obs"]
        stats["coverage"][f"{obs['g']['act']}:{obs['g']['room']}:{p['save']['room']}"] += 1
        stats["replayed_points"] += 1
    if args.replay_only:
        return finish_run(out, spec, commands, records, points, selected, eligible, stats)
    finished = next((p for p in points if p["save"]["kind"] == "finished"), None)
    if finished:
        before = checked(eng, {"cmd": "dbg_obs"})
        rejected = eng.send({"cmd": "rl_load", "id": finished["save"]["id"]})
        if rejected.get("type") != "error" or "pre-finished" not in rejected.get("message", ""):
            raise RuntimeError(f"finished save not rejected: {rejected}")
        compare(before, checked(eng, {"cmd": "dbg_obs"}), out, {"test": "rejected load preserves decision"})
        stats["finished_rejected"] += 1
    for p in room_points:
        initial = records[p["index"]]
        if (initial["obs"]["g"]["room"] in ("Monster", "Elite", "Boss") and initial["obs"].get("combat")
                and len(initial["cands"]) >= 2 and stats["crn_combats"] < quotas):
            branches = []
            first = choose(rng, initial)
            actions, shared = None, None
            for branch in range(2):
                entry = replay(eng, p, records, commands, out, False, stats, end=p["index"])
                combat = entry["obs"]["combat"]
                branches.append({"enc": combat["enc"], "enemies": combat["enemies"]})
                entry = checked(eng, {"cmd": "rl_act", "i": first})
                if shared is not None:
                    compare(shared, entry, out, {"test": "common first combat action", "save": p["save"]["id"]})
                if entry["done"] or len(entry["cands"]) < 2:
                    break
                shared = entry
                if actions is None:
                    actions = rng.sample(range(len(entry["cands"])), 2)
                checked(eng, {"cmd": "rl_act", "i": actions[branch]})
            else:
                if branches[0] != branches[1]:
                    raise RuntimeError("common-random-number combat mismatch")
                with (out / "crn.jsonl").open("a") as f:
                    f.write(json.dumps({"test": "combat", "save": p["save"]["id"], "first": first, "actions": actions, "branches": branches}) + "\n")
                stats["crn_combats"] += 1
        if stats["crn_rewards"] < quotas:
            stop = next((q["index"] for q in room_points if q["index"] > p["index"]), len(records))
            target = next((i for i in range(p["index"], stop) if card_screen(records[i])), None)
            if target is not None:
                branches = [reward_branch(eng, p, target, records, commands, out, b, stats) for b in range(2)]
                if all(branches):
                    if branches[0]["card"] == branches[1]["card"]:
                        continue
                    if branches[0]["enc"] != branches[1]["enc"]:
                        raise RuntimeError(f"common-random-number reward mismatch: {branches}")
                    with (out / "crn.jsonl").open("a") as f:
                        f.write(json.dumps({"test": "rewards", "save": p["save"]["id"], "branches": branches}) + "\n")
                    stats["crn_rewards"] += 1
                    stats["crn_reward_next_combats"] += int(branches[0]["enc"] is not None)
    return finish_run(out, spec, commands, records, points, selected, eligible, stats)


def finish_run(out, spec, commands, records, points, selected, eligible, stats):
    checked_points = []
    for p in selected:
        g = records[p["index"]]["obs"]["g"]
        checked_points.append({"id": p["save"]["id"], "index": p["index"],
                               "coverage": f"{g['act']}:{g['room']}:{p['save']['room']}"})
    endings = [json.loads(line) for line in (out / "endings.jsonl").read_text().splitlines()]
    assert len(endings) == len(selected) * 2 and all(e["equal"] for e in endings)
    (out / "result.json").write_text(json.dumps({"spec": spec, "steps": len(commands), "saves": len(points),
        "room_saves": sum(p["save"]["kind"] == "room" for p in points), "replayed": len(selected),
        "three_point_run": eligible, "validated_modes": ["same", "fresh"], "points": checked_points,
        "ending": ending(records[-1]), "endings": endings}))
    stats["completed"].append(str(out.resolve()))
    return eligible


def reuse_run(args, n, attempt, stats):
    if not args.resume_from:
        return None
    suffix = f"_retry_{attempt}" if attempt else ""
    out = Path(args.resume_from) / f"run_{n:04d}{suffix}"
    if not (out / "result.json").exists() or (out / "mismatch.json").exists():
        return None
    result = json.loads((out / "result.json").read_text())
    if "endings" not in result:
        return None
    spec = result["spec"]
    if spec != {"cmd": "rl_start", "character": CHARS[n % len(CHARS)], "ascension": (0, 10)[(n // len(CHARS)) % 2],
                "seed": f"{args.seed_prefix}_{n}{suffix}"}:
        return None
    points = result.get("points")
    if points is None:
        saved = {}
        for path in out.glob("save_*.json"):
            save = json.loads(path.read_text())
            saved[save["id"]] = save
        points = []
        seen = 0
        with gzip.open(out / "trace.jsonl.gz", "rt") as trace:
            for i, line in enumerate(trace):
                r = json.loads(line)["response"]
                latest = r["info"]["save"]["id"]
                g = r["obs"]["g"]
                for k in range(seen + 1, latest + 1):
                    if k in saved:
                        points.append({"id": k, "index": i, "coverage": f"{g['act']}:{g['room']}:{saved[k]['room']}"})
                seen = latest
        assert r["done"] and i == result["steps"]
    assert len(points) == result["replayed"] == len({p["id"] for p in points})
    eligible = result["three_point_run"]
    assert not eligible or len(points) >= 3
    assert len(result["endings"]) == 2 * len(points) and all(e["equal"] for e in result["endings"])
    stats["attempts"] += 1
    stats["runs"] += int(eligible)
    stats["short_runs"] += int(not eligible)
    stats["reused_runs"] += int(eligible)
    stats["baseline_steps"] += result["steps"]
    stats["replayed_points"] += len(points)
    stats["compared_steps"] += 2 * sum(result["steps"] - p["index"] for p in points)
    stats["compared_states"] += 2 * sum(result["steps"] - p["index"] + 1 for p in points)
    stats["ending_checks"] += len(result["endings"])
    for p in points:
        stats["coverage"][p["coverage"]] += 1
    if eligible:
        stats["characters"][spec["character"]] += 1
        stats["ascensions"][str(spec["ascension"])] += 1
        stats["god_runs"] += int((n // 10) % 4 == 0)
    stats["completed"].append(str(out.resolve()))
    return eligible


def worker(args, shard, quota):
    stats = {k: 0 for k in ("runs", "attempts", "short_runs", "reused_runs", "baseline_steps", "compared_steps", "compared_states", "ending_checks", "god_runs", "replayed_points", "finished_rejected",
                            "crn_combats", "crn_rewards", "crn_reward_next_combats")}
    stats.update({k: [] for k in ("load_ms", "data_bytes", "response_bytes")})
    stats["completed"] = []
    stats.update({k: collections.Counter() for k in ("characters", "ascensions", "coverage", "save_kinds")})
    eng = Engine(args.engine, timeout=120, stderr_path=str(Path(args.out) / f"worker_{shard[0]}.stderr"))
    try:
        for n in shard:
            attempt = 0
            while True:
                eligible = reuse_run(args, n, attempt, stats)
                if eligible is None:
                    eligible = run_one(eng, args, n, stats, quota, attempt)
                if eligible:
                    break
                attempt += 1
            print(f"run {n}: replayed {stats['replayed_points']} points, {stats['compared_steps']} compared steps", flush=True)
    except Exception:
        STOP.set()
        raise
    finally:
        eng.close()
        (Path(args.out) / f"worker_{shard[0]}_progress.json").write_text(json.dumps({
            "completed": stats["completed"], "reused_runs": stats["reused_runs"],
            "compared_states": stats["compared_states"], "replayed_points": stats["replayed_points"]}))
    return stats


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": values[0], "median": values[len(values) // 2],
            "p95": values[math.ceil(len(values) * .95) - 1], "max": values[-1]}


def summarize_known(directories, out):
    records = []
    for directory in sorted({Path(d).resolve() for d in directories}):
        path = directory / "known.jsonl"
        if path.exists():
            records.extend(json.loads(line) for line in path.read_text().splitlines())
    with (out / "known.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    summary = {"total": len(records)}
    for key in ("run", "seed", "save", "mode", "step", "decision", "act", "room", "relic", "field"):
        summary[key] = dict(collections.Counter(str(r[key]) for r in records))
    summary["values"] = dict(collections.Counter(f"{r['expected']}->{r['actual']}" for r in records))
    summary["run_save"] = dict(collections.Counter(f"{r['run']}:{r['save']}" for r in records))
    summary["runs"] = len({r["run"] for r in records})
    summary["points"] = len({(r["run"], r["save"]) for r in records})
    summary["relic_field"] = dict(collections.Counter(f"{r['relic']}:{r['field']}" for r in records))
    (out / "known_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default=str(Path(__file__).resolve().parents[1] / "engine"))
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    ap.add_argument("--seed-prefix", default="saveload")
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--crn", type=int, default=100)
    ap.add_argument("--replay-only", action="store_true", help="only check room-save replay; skip other acceptance checks")
    ap.add_argument("--resume-from", help="reuse completed replay results from this directory (with --replay-only)")
    ap.add_argument("--relic-metadata", help="reflection table generated by this tool for the current game DLL")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "tmp/engine_saveload/check"))
    args = ap.parse_args()
    if args.resume_from and not args.replay_only:
        ap.error("--resume-from requires --replay-only")
    if args.replay_only:
        args.crn = 0
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "failure.json").unlink(missing_ok=True)
    metadata_path = args.relic_metadata or reflect_relics(args.engine, Path(args.out))
    shards = [list(range(i, args.runs, args.workers)) for i in range(args.workers)]
    t0 = time.perf_counter()
    results = []
    try:
        with cf.ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(mp.Event(), metadata_path)) as pool:
            futures = [pool.submit(worker, args, shard, args.crn // args.workers + int(i < args.crn % args.workers))
                       for i, shard in enumerate(shards) if shard]
            for f in cf.as_completed(futures):
                results.append(f.result())
    except Exception as e:
        out = Path(args.out)
        directories = list(out.glob("run_*"))
        for path in out.glob("worker_*_progress.json"):
            directories.extend(json.loads(path.read_text())["completed"])
        mismatches = sorted(out.glob("run_*/mismatch.json"))
        failure = {"error": str(e), "mismatches": [str(p) for p in mismatches],
                   "known": summarize_known(directories, out)}
        if mismatches:
            detail = json.loads(mismatches[0].read_text())
            failure["error"] = f"save/load mismatch: {mismatches[0].parent.name}: {detail['unexpected']}"
        (out / "failure.json").write_text(json.dumps(failure, indent=2))
        raise
    combined = {}
    for r in results:
        for k, v in r.items():
            if k not in combined:
                combined[k] = [] if isinstance(v, list) else collections.Counter() if isinstance(v, dict) else 0
            combined[k] += v
    for k in ("load_ms", "data_bytes", "response_bytes"):
        combined[k] = distribution(combined[k])
    combined["seconds"] = time.perf_counter() - t0
    combined["passed"] = combined["runs"] == args.runs
    combined["endings_equal"] = True
    combined["outcomes"] = {}
    combined["run_endings"] = []
    for directory in combined["completed"]:
        result = json.loads((Path(directory) / "result.json").read_text())
        row = {"seed": result["spec"]["seed"], "character": result["spec"]["character"],
               "ascension": result["spec"]["ascension"], "three_point_run": result["three_point_run"],
               "original": result["ending"], "checks": len(result["endings"]),
               "equal": all(e["equal"] for e in result["endings"])}
        for mode in ("same", "fresh"):
            checked_endings = [e for e in result["endings"] if e["mode"] == mode]
            row[mode] = {"checks": len(checked_endings), "outcomes": sorted({e["restored"]["outcome"] for e in checked_endings}),
                         "steps": sorted({e["restored"]["steps"] for e in checked_endings}),
                         "equal": all(e["equal"] for e in checked_endings)}
        combined["run_endings"].append(row)
        combined["endings_equal"] &= combined["run_endings"][-1]["equal"]
    combined["outcomes"] = dict(collections.Counter(e["original"]["outcome"] for e in combined["run_endings"] if e["three_point_run"]))
    with (Path(args.out) / "run_endings.jsonl").open("w") as f:
        for row in combined["run_endings"]:
            f.write(json.dumps(row) + "\n")
    combined["passed"] &= combined["endings_equal"]
    if not args.replay_only:
        combined["passed"] &= (combined["crn_combats"] >= args.crn and combined["crn_rewards"] >= args.crn
                               and combined["finished_rejected"] > 0)
    combined["known"] = summarize_known(combined["completed"], Path(args.out))
    combined["replay_only"] = args.replay_only
    if args.replay_only:
        for key in ("load_ms", "data_bytes", "response_bytes", "save_kinds", "finished_rejected",
                    "crn_combats", "crn_rewards", "crn_reward_next_combats"):
            combined.pop(key)
    if args.runs >= 200:
        coverage = [k.split(":") for k in combined["coverage"]]
        combined["passed"] &= ({"1", "2", "3"} <= {k[0] for k in coverage}
                               and {"Monster", "Elite", "Boss", "Shop", "RestSite", "Event", "Treasure"} <= {k[1] for k in coverage}
                               and "Unknown" in {k[2] for k in coverage})
    (Path(args.out) / "summary.json").write_text(json.dumps(combined, indent=2))
    print(json.dumps(combined, indent=2))
    return 0 if combined["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
