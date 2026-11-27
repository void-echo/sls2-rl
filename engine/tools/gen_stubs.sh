#!/bin/bash
# Regenerate src/GodotStubs/Generated/Generated.cs for the sts2.dll in lib/ and verify the stub (see engine/README.md,
# "Godot stubs"). Run after every game update (setup.sh calls it). Exit code != 0 means a reference cannot be
# satisfied automatically: a member of a hand-written semantic type is missing (implement it in src/GodotStubs/Semantic
# with Godot's semantics) or the generated code does not compile.
set -eo pipefail
ENGINE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ENGINE"
DOTNET="${DOTNET:-dotnet}"
command -v "$DOTNET" >/dev/null 2>&1 || { DOTNET="$HOME/.dotnet/dotnet"; export DOTNET_ROOT="$HOME/.dotnet"; }
LIB="${STUB_LIB:-lib}"
HANDONLY="$ENGINE/tools/.handonly_stub"

"$DOTNET" build tools/GodotStubGen -c Release -v q -nologo 2>&1 | grep -E "error|Warn|warn" || true
GEN="$ENGINE/tools/GodotStubGen/bin/Release/net9.0/GodotStubGen.dll"

# 1. the hand-written (semantic) part alone
rm -rf "$HANDONLY"
"$DOTNET" build src/GodotStubs -p:StubsHandOnly=true -o "$HANDONLY" -v q -nologo 2>&1 | grep -E " error " || true
[ -f "$HANDONLY/GodotSharp.dll" ] || { echo "hand-written stub build failed"; exit 1; }

# 2. generate the inert remainder from the real GodotSharp API
"$DOTNET" "$GEN" gen "$LIB/sts2.dll" "$LIB/GodotSharp.api.dll" "$HANDONLY/GodotSharp.dll" src/GodotStubs/Generated/Generated.cs

# 3. build the full stub and check every TypeRef/MemberRef of sts2.dll (exact signatures, kinds, override access)
"$DOTNET" build src/GodotStubs -v q -nologo 2>&1 | grep -E " error " && { echo "generated stub does not compile"; exit 1; }
"$DOTNET" "$GEN" check "$LIB/sts2.dll" src/GodotStubs/bin/Debug/net9.0/GodotSharp.dll "$LIB/GodotSharp.api.dll"
rm -rf "$HANDONLY"
# The hand-only build shares obj/ with the full build; rebuild so the next project reference picks up the full stub.
"$DOTNET" build src/GodotStubs -v q -nologo >/dev/null
