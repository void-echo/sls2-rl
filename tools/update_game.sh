#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DOTNET_ROOT="${DOTNET_ROOT:-$HOME/.dotnet}"
export PATH="$DOTNET_ROOT:$PATH"
PYTHON="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_PROCESSOR_COUNT=9
unset STS2_LIB STUB_LIB

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <game_dir> <out_dir> [--quick] [--baseline <content_ids.json>]" >&2
    exit 2
fi
GAME="$1"; OUT="$2"; shift 2
QUICK=0; BASELINE=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --quick) QUICK=1; shift ;;
        --baseline) BASELINE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
HELPER="$REPO/tools/update_game_support.py"
if [ -n "$BASELINE" ]; then
    BASELINE="$("$PYTHON" -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).expanduser().resolve())' "$BASELINE")"
fi
"$PYTHON" "$HELPER" prepare "$REPO" "$GAME" "$OUT" "$BASELINE" "$QUICK"
OUT="$(cd "$OUT" && pwd)"
ENGINE="$REPO/engine"
GAME_DATA="$("$PYTHON" -c 'import json,sys; from pathlib import Path; print(Path(json.load(open(sys.argv[1]))["runtime_dll"]).parent)' "$OUT/version.json")"
export STS2_LIB="$ENGINE/lib"
export PATH="$(dirname "$PYTHON"):$PATH"
cd "$REPO"
FAILED=0
step() {
    "$PYTHON" "$HELPER" step "$OUT" "$@" || FAILED=1
}
finish() {
    "$PYTHON" "$HELPER" report "$OUT" "$BASELINE"
}

step setup bash -c 'cd "$1" && bash setup.sh "$2"' _ "$ENGINE" "$GAME_DATA"
if [ "$FAILED" -ne 0 ]; then finish || true; exit 1; fi
step startup bash -c 'cd "$1" && printf '\''{"cmd":"quit"}\n'\'' | dotnet src/Sts2Headless/bin/Debug/net9.0/Sts2Headless.dll' _ "$ENGINE"
step threads dotnet "$ENGINE/tools/GodotStubGen/bin/Release/net9.0/GodotStubGen.dll" threads "$ENGINE/lib/sts2.dll"

mkdir -p "$OUT/content_reflection"
cat > "$OUT/content_reflection/Content.csproj" <<'EOF'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType><TargetFramework>net9.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings>
  </PropertyGroup>
</Project>
EOF
cat > "$OUT/content_reflection/Program.cs" <<'EOF'
using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;

var engine = Path.GetFullPath(args[0]);
AssemblyLoadContext.Default.Resolving += (ctx, name) => {
    var path = new[] { Path.Combine(engine, "src/Sts2Headless/bin/Debug/net9.0", name.Name + ".dll"),
                      Path.Combine(engine, "lib", name.Name + ".dll") }.FirstOrDefault(File.Exists);
    return path == null ? null : ctx.LoadFromAssemblyPath(path);
};
var game = AssemblyLoadContext.Default.LoadFromAssemblyPath(Path.Combine(engine, "lib/sts2.dll"));
var models = "MegaCrit.Sts2.Core.Models.";
var getId = game.GetType(models + "ModelDb", true).GetMethod("GetId", new[] { typeof(Type) });
var groups = new Dictionary<string, string> { ["cards"] = "CardModel", ["relics"] = "RelicModel",
    ["potions"] = "PotionModel", ["powers"] = "PowerModel", ["monsters"] = "MonsterModel",
    ["encounters"] = "EncounterModel", ["events"] = "EventModel", ["enchantments"] = "EnchantmentModel" };
var ids = new SortedDictionary<string, string[]>();
var rows = new List<object>();
foreach (var (group, model) in groups) {
    var parent = game.GetType(models + model, true);
    var types = game.GetTypes().Where(t => !t.IsAbstract && !t.ContainsGenericParameters && t != parent
        && parent.IsAssignableFrom(t)).OrderBy(t => t.FullName).ToArray();
    string Id(Type t) => getId.Invoke(null, new object[] { t }).ToString();
    ids[group] = types.Select(Id).Distinct().Order().ToArray();
    rows.AddRange(types.Select(t => new { group, id = Id(t), name = t.Name, ns = t.Namespace, parent = t.BaseType.Name }));
}
var options = new JsonSerializerOptions { WriteIndented = true };
File.WriteAllText(args[1], JsonSerializer.Serialize(ids, options) + "\n");
File.WriteAllText(args[2], JsonSerializer.Serialize(rows, options) + "\n");
foreach (var (group, values) in ids) Console.WriteLine($"{group}: {values.Length}");
EOF
step content_ids bash -c 'dotnet build "$1/Content.csproj" -v q -o "$1/bin" && dotnet "$1/bin/Content.dll" "$2" "$3/content_ids.json" "$3/content_models.json"' _ "$OUT/content_reflection" "$ENGINE" "$OUT"

step oracle_build env STS2GameDataDir="$GAME_DATA" PARITY_BUILD_DIR="$OUT/oracle_build" "$PYTHON" "$REPO/parity/oracle.py" build

RUNS=30
[ "$QUICK" -eq 0 ] || RUNS=5
for c in Ironclad Silent Defect Regent Necrobinder Fool; do
    step "rl_smoke_$c" "$PYTHON" "$REPO/tools/rl_smoke.py" "$ENGINE" --runs "$RUNS" --character "$c"
done
step scenario_tests "$PYTHON" "$REPO/tools/scenario_tests.py" --engine "$ENGINE"
RUNS=40
[ "$QUICK" -eq 0 ] || RUNS=8
PER_ENGINE=$(( (RUNS * 2 + 7) / 8 ))
step determinism "$PYTHON" "$REPO/tools/determinism.py" --engine "$ENGINE" --runs "$RUNS" --parallel 8 --per-engine "$PER_ENGINE" --workers 9
if [ "$QUICK" -eq 0 ]; then
    step fullrun "$PYTHON" "$REPO/tools/fullrun.py" --engine "$ENGINE" --runs 300 --asc 0,10 --workers 9 --out "$OUT/fullrun.jsonl"
    step fullrun_god "$PYTHON" "$REPO/tools/fullrun.py" --engine "$ENGINE" --runs 100 --god --workers 9 --out "$OUT/fullrun_god.jsonl"
    step cycle_check "$PYTHON" "$REPO/tools/cycle_check.py" --engine "$ENGINE" --runs 600 --asc 0,10 --workers 9 --out "$OUT/cycle_check.jsonl"
    step cycle_check_god "$PYTHON" "$REPO/tools/cycle_check.py" --engine "$ENGINE" --runs 300 --god --workers 9 --out "$OUT/cycle_check_god.jsonl"
    step probe_specific "$PYTHON" "$REPO/tmp/content_review/probe_specific.py" --engine "$ENGINE"
    for suite in events relics potions cards encounters neow; do
        step "fuzz_$suite" "$PYTHON" "$HELPER" fuzz "$REPO" "$OUT" --engine "$ENGINE" --suite "$suite" --workers 9 --strict --out "$OUT/fuzz_$suite.jsonl"
    done
    step saveload_check "$PYTHON" "$REPO/tools/saveload_check.py" --engine "$ENGINE" --runs 200 --workers 4 --out "$OUT/saveload"
fi
finish || FAILED=1
exit "$FAILED"
