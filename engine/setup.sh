#!/bin/bash
# setup.sh — copy the game's DLLs into lib/ (or lib_x64/), regenerate the Godot stubs, build.
#
# The game DLLs are used unmodified (no IL patching): every harness change to game behaviour is a Harmony patch in
# src/Sts2Headless/Patches.cs. GodotSharp.dll is copied as lib/GodotSharp.api.dll and only read as metadata by
# tools/GodotStubGen (the process loads the headless stub built from src/GodotStubs instead).
#
# Prerequisites: Slay the Spire 2 installed via Steam; .NET 9 SDK (dotnet on PATH, or ~/.dotnet).
#
# Usage:
#   ./setup.sh                      # auto-detect the macOS arm64 data dir (or $STS2_GAME_DIR / first argument), lib/
#   ./setup.sh /path/to/data_dir    # explicit game data dir (the folder that contains sts2.dll)
#   ./setup.sh --x64 [data_dir]     # x86_64 DLLs into lib_x64/ (for Linux x86_64 deployment; default source:
#                                   #   SlayTheSpire2.app/Contents/Resources/data_sts2_macos_x86_64). Does not build.
#                                   #   On the Linux machine: copy lib_x64/ to lib/ (or set STS2_LIB=<dir>).
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=arm
if [ "${1:-}" = "--x64" ]; then MODE=x64; shift; fi
GAME_DIR="${1:-${STS2_GAME_DIR:-}}"
if [ -n "$GAME_DIR" ]; then GAME_DIR="$(cd "$GAME_DIR" && pwd)"; fi
cd "$REPO_DIR"

MAC_RES="$HOME/Library/Application Support/Steam/steamapps/common/Slay the Spire 2/SlayTheSpire2.app/Contents/Resources"
if [ -z "$GAME_DIR" ]; then
    case "$(uname -s)-$MODE" in
        Darwin-arm) GAME_DIR="$MAC_RES/data_sts2_macos_arm64"; [ -d "$GAME_DIR" ] || GAME_DIR="$MAC_RES/data_sts2_macos_x86_64" ;;
        Darwin-x64) GAME_DIR="$MAC_RES/data_sts2_macos_x86_64" ;;
        Linux-*)
            GAME_DIR="$HOME/.steam/steam/steamapps/common/Slay the Spire 2"
            [ -d "$GAME_DIR" ] || GAME_DIR="$HOME/.local/share/Steam/steamapps/common/Slay the Spire 2" ;;
    esac
fi
if [ ! -f "$GAME_DIR/sts2.dll" ]; then
    found=$(find "$GAME_DIR" -name sts2.dll -print -quit 2>/dev/null || true)
    [ -n "$found" ] && GAME_DIR="$(dirname "$found")"
fi
if [ ! -f "$GAME_DIR/sts2.dll" ]; then
    echo "Game data directory with sts2.dll not found: $GAME_DIR"
    echo "Usage: ./setup.sh [--x64] /path/to/data_dir"
    exit 1
fi
echo "Game data directory: $GAME_DIR"

OUT=lib
[ "$MODE" = x64 ] && OUT=lib_x64
mkdir -p "$OUT"
DLLS=(sts2.dll SmartFormat.dll SmartFormat.ZString.dll Sentry.dll Sentry.Godot.dll Steamworks.NET.dll
      MonoMod.Backports.dll MonoMod.ILHelpers.dll 0Harmony.dll System.IO.Hashing.dll)
for dll in "${DLLS[@]}"; do
    if [ -f "$GAME_DIR/$dll" ]; then cp "$GAME_DIR/$dll" "$OUT/$dll"; echo "  copied $dll"
    else echo "  MISSING $dll (the build or the run may fail)"; fi
done
cp "$GAME_DIR/GodotSharp.dll" "$OUT/GodotSharp.api.dll" && echo "  copied GodotSharp.dll -> $OUT/GodotSharp.api.dll (metadata only)"
rm -f "$OUT/sts2.dll.original" "$OUT/sts2.dll.ilpatched_v0"

if [ "$MODE" = x64 ]; then
    echo "x64 libraries in $OUT/ (the managed code is identical to lib/: same IL, different PE machine type)."
    exit 0
fi

# Refresh text alongside the engine so descriptions match the installed balance.
PCK_PATH="$(find "$GAME_DIR" "$(dirname "$GAME_DIR")" -maxdepth 2 -name '*.pck' -print -quit 2>/dev/null || true)"
if [ -n "$PCK_PATH" ]; then python3 scripts/extract_localization.py "$PCK_PATH"
else echo "No game PCK found; keeping existing localization tables."; fi

DOTNET="${DOTNET:-}"
if [ -z "$DOTNET" ]; then
    if command -v dotnet >/dev/null 2>&1; then DOTNET=dotnet
    elif [ -x "$HOME/.dotnet/dotnet" ]; then DOTNET="$HOME/.dotnet/dotnet"; export DOTNET_ROOT="$HOME/.dotnet"
    else echo ".NET SDK not found (install .NET 9 or set DOTNET)"; exit 1; fi
fi
echo ".NET SDK: $DOTNET ($($DOTNET --version))"

# Godot stubs: regenerate the inert part for this sts2.dll and verify every reference resolves exactly.
DOTNET="$DOTNET" ./tools/gen_stubs.sh

"$DOTNET" build src/Sts2Headless/Sts2Headless.csproj 2>&1 | tail -3
echo "Setup complete. Smoke test: python3 ../tools/rl_smoke.py . --runs 5"
