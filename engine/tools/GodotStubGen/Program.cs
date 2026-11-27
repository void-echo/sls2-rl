// GodotStubGen: keeps the headless GodotSharp stub (engine/src/GodotStubs) complete for a given sts2.dll.
//
// The CLR binds every type and member reference of sts2.dll by exact name and signature when a method that uses it
// is compiled; a reference the stub lacks makes the whole method (an async method: its whole state machine) throw
// MissingMethodException / MissingFieldException / TypeLoadException, even if the line is never reached. This tool
// reads the metadata of sts2.dll, of the real GodotSharp.dll shipped with the game, and of the stub, and
//   gen   writes engine/src/GodotStubs/Generated/Generated.cs with every referenced type/member the hand-written
//         stub sources lack, declared exactly like the real GodotSharp (kind, base type, accessibility, virtuality,
//         static-ness, signature) with inert bodies. Types listed in Semantic (pure-managed value types and helpers
//         whose behaviour game logic depends on) are never generated: a missing member there is an error that a
//         human fixes in the hand-written semantic sources (copied from Godot's MIT-licensed C# sources).
//   check verifies the built stub: every TypeRef / MemberRef / attribute constructor / MethodImpl of sts2.dll into
//         GodotSharp resolves with an exact signature (incl. custom modifiers and valuetype/class markers), and every
//         virtual stub member has the real accessibility (an override may not reduce access). Exit code 1 on failure.
//
// Usage:
//   GodotStubGen gen   <sts2.dll> <real GodotSharp.dll> <hand-written stub GodotSharp.dll> <out.cs>
//   GodotStubGen check <sts2.dll> <stub GodotSharp.dll> [<real GodotSharp.dll>]
//   GodotStubGen threads <sts2.dll>     gameplay call sites that can move work off the game thread or onto a timer
//                                        (Task.ContinueWith without scheduler, Task.Run, StartNew, ThreadPool, Thread,
//                                        ConfigureAwait, Task.Delay, WaitAsync(TimeSpan), timers); review after updates.
using System.Collections.Immutable;
using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;
using System.Text;

static class Program
{
    /// <summary>Types implemented by hand with Godot's semantics (never generated).</summary>
    public static readonly HashSet<string> Semantic = new()
    {
        "Godot.Mathf", "Godot.Vector2", "Godot.Vector2I", "Godot.Vector3", "Godot.Vector3I", "Godot.Vector4", "Godot.Vector4I",
        "Godot.Rect2", "Godot.Rect2I", "Godot.Color", "Godot.Colors", "Godot.Transform2D", "Godot.Transform3D", "Godot.Basis",
        "Godot.Quaternion", "Godot.Projection", "Godot.Plane", "Godot.Aabb",
        "Godot.StringName", "Godot.NodePath", "Godot.Callable", "Godot.Variant", "Godot.Signal", "Godot.SignalAwaiter",
        "Godot.StringExtensions", "Godot.GD",
    };

    static int Main(string[] args)
    {
        if (args.Length >= 5 && args[0] == "gen") return Gen(args[1], args[2], args[3], args[4]);
        if (args.Length >= 2 && args[0] == "refs")
        {
            var refs = new Sts2Refs(args[1]);
            foreach (var t in refs.Types.OrderBy(x => x)) Console.WriteLine("T " + t);
            foreach (var m in refs.Members.OrderBy(x => x.Owner).ThenBy(x => x.Key)) Console.WriteLine($"M {m.Owner} :: {(m.IsField ? "field " + m.Name + " : " + m.Sig : m.Key)}");
            return 0;
        }
        if (args.Length >= 3 && args[0] == "check") return Check(args[1], args[2], args.Length > 3 ? args[3] : null);
        if (args.Length >= 2 && args[0] == "threads") return ThreadScan.Run(args[1]);
        Console.Error.WriteLine("usage: GodotStubGen gen <sts2.dll> <real GodotSharp.dll> <stub GodotSharp.dll> <out.cs> | check <sts2.dll> <stub GodotSharp.dll> [<real GodotSharp.dll>]");
        return 2;
    }

    // ───────────────────────────────────────── check

    static int Check(string sts2Path, string stubPath, string? realPath)
    {
        var refs = new Sts2Refs(sts2Path);
        var stub = new Asm(stubPath);
        var real = realPath != null ? new Asm(realPath) : null;
        var missing = new SortedSet<string>();
        foreach (var t in refs.Types)
            if (!stub.Types.ContainsKey(t)) missing.Add($"type {t}");
        foreach (var m in refs.Members)
        {
            if (!stub.Types.TryGetValue(m.Owner, out var st)) { missing.Add($"type {m.Owner} (owner of {m.Name})"); continue; }
            if (m.IsField ? !st.Fields.ContainsKey(m.Name) || st.Fields[m.Name] != m.Sig : !st.Methods.Contains(m.Key))
                missing.Add($"{m.Owner}::{(m.IsField ? m.Name + " : " + m.Sig : m.Key)}");
        }
        // Valuetype/class kind of every referenced type must agree with the stub definition.
        foreach (var (name, isValue) in refs.Kinds)
            if (stub.Types.TryGetValue(name, out var st) && st.IsValueType != isValue)
                missing.Add($"kind mismatch {name}: sts2 references it as {(isValue ? "valuetype" : "class")}");
        int accessProblems = 0;
        if (real != null)
            foreach (var (name, st) in stub.Types)
            {
                if (!real.Types.TryGetValue(name, out var rt)) continue;
                foreach (var (key, acc) in st.VirtualAccess)
                    if (rt.VirtualAccess.TryGetValue(key, out var racc) && racc != acc)
                    {
                        missing.Add($"access {name}::{key}: stub {acc}, real {racc}");
                        accessProblems++;
                    }
                if (st.BaseType != rt.BaseType && !(st.BaseType == null && rt.BaseType == "System.Object"))
                    missing.Add($"base {name}: stub {st.BaseType ?? "-"}, real {rt.BaseType ?? "-"}");
            }
        foreach (var s in missing) Console.WriteLine("MISSING " + s);
        Console.WriteLine($"stub check: {refs.Types.Count} referenced types, {refs.Members.Count} referenced members, {missing.Count} problems");
        return missing.Count == 0 ? 0 : 1;
    }

    // ───────────────────────────────────────── gen

    static int Gen(string sts2Path, string realPath, string stubPath, string outPath)
    {
        var refs = new Sts2Refs(sts2Path);
        var real = new Asm(realPath);
        var stub = new Asm(stubPath);
        var emit = new Emitter(real, stub);
        int errors = 0;

        foreach (var t in refs.Types) emit.NeedType(t);
        foreach (var m in refs.Members)
        {
            if (!real.Types.TryGetValue(m.Owner, out var rt)) { Console.Error.WriteLine($"ERROR {m.Owner}::{m.Name}: owner not in the real GodotSharp"); errors++; continue; }
            bool present = stub.Types.TryGetValue(m.Owner, out var st)
                           && (m.IsField ? st.Fields.ContainsKey(m.Name) : st.Methods.Contains(m.Key));
            if (present) continue;
            if (Semantic.Contains(TopLevel(m.Owner)))
            {
                Console.Error.WriteLine($"SEMANTIC {m.Owner}::{(m.IsField ? m.Name : m.Key)} is missing from the hand-written stub (implement it with Godot's semantics)");
                errors++;
                continue;
            }
            if (!emit.NeedMember(m)) { Console.Error.WriteLine($"ERROR {m.Owner}::{m.Key}: no matching member in the real GodotSharp"); errors++; }
        }
        emit.Close();
        foreach (var w in emit.Warnings.Distinct()) Console.Error.WriteLine("WARN " + w);
        File.WriteAllText(outPath, emit.Render());
        Console.WriteLine($"generated {emit.TypeCount} types / {emit.MemberCount} members -> {outPath} ({errors} errors)");
        return errors == 0 ? 0 : 1;
    }

    public static string TopLevel(string name) => name.Split('/')[0];
}

/// <summary>Canonical (binding-relevant) text of a signature type: valuetype markers and custom modifiers included.</summary>
sealed class CanonProvider : ISignatureTypeProvider<string, object?>
{
    public static readonly CanonProvider I = new();
    public string GetArrayType(string e, ArrayShape s) => $"{e}[{s.Rank}]";
    public string GetByReferenceType(string e) => e + "&";
    public string GetFunctionPointerType(MethodSignature<string> s) => $"fnptr({s.ReturnType}({string.Join(",", s.ParameterTypes)}))";
    public string GetGenericInstantiation(string g, ImmutableArray<string> a) => $"{g}<{string.Join(",", a)}>";
    public string GetGenericMethodParameter(object? c, int i) => "!!" + i;
    public string GetGenericTypeParameter(object? c, int i) => "!" + i;
    public string GetModifiedType(string m, string u, bool req) => $"{u} {(req ? "modreq" : "modopt")}({m})";
    public string GetPinnedType(string e) => e + " pinned";
    public string GetPointerType(string e) => e + "*";
    public string GetPrimitiveType(PrimitiveTypeCode t) => t.ToString();
    public string GetSZArrayType(string e) => e + "[]";
    public string GetTypeFromDefinition(MetadataReader r, TypeDefinitionHandle h, byte k) => (k == 0x11 ? "valuetype " : "") + Names.Of(r, h);
    public string GetTypeFromReference(MetadataReader r, TypeReferenceHandle h, byte k) => (k == 0x11 ? "valuetype " : "") + Names.Of(r, h);
    public string GetTypeFromSpecification(MetadataReader r, object? c, TypeSpecificationHandle h, byte k) =>
        r.GetTypeSpecification(h).DecodeSignature(this, c);
}

static class Names
{
    public static string Of(MetadataReader r, TypeDefinitionHandle h)
    {
        var td = r.GetTypeDefinition(h);
        var name = r.GetString(td.Name);
        var dt = td.GetDeclaringType();
        if (!dt.IsNil) return Of(r, dt) + "/" + name;
        var ns = r.GetString(td.Namespace);
        return ns.Length > 0 ? ns + "." + name : name;
    }

    public static string Of(MetadataReader r, TypeReferenceHandle h)
    {
        var tr = r.GetTypeReference(h);
        var name = r.GetString(tr.Name);
        if (tr.ResolutionScope.Kind == HandleKind.TypeReference) return Of(r, (TypeReferenceHandle)tr.ResolutionScope) + "/" + name;
        var ns = r.GetString(tr.Namespace);
        return ns.Length > 0 ? ns + "." + name : name;
    }

    public static string MethodKey(string name, MethodSignature<string> sig) =>
        $"{(sig.Header.IsInstance ? "instance " : "static ")}{sig.ReturnType} {name}`{sig.GenericParameterCount}({string.Join(",", sig.ParameterTypes)})";
}

/// <summary>Everything sts2.dll references in GodotSharp.</summary>
sealed class Sts2Refs
{
    public sealed record Member(string Owner, string Name, bool IsField, string Sig, string Key, MemberReferenceHandle Handle);

    public readonly HashSet<string> Types = new();
    public readonly List<Member> Members = new();
    public readonly Dictionary<string, bool> Kinds = new();
    private readonly MetadataReader _r;
    private readonly HashSet<AssemblyReferenceHandle> _godot = new();

    public Sts2Refs(string path)
    {
        var pe = new PEReader(File.OpenRead(path));
        _r = pe.GetMetadataReader();
        foreach (var h in _r.AssemblyReferences)
            if (_r.GetString(_r.GetAssemblyReference(h).Name) == "GodotSharp") _godot.Add(h);
        foreach (var h in _r.TypeReferences)
            if (IsGodot(h)) Types.Add(Names.Of(_r, h));
        foreach (var h in _r.MemberReferences)
        {
            var mr = _r.GetMemberReference(h);
            var owner = OwnerName(mr.Parent);
            if (owner == null) continue;
            var name = _r.GetString(mr.Name);
            if (mr.GetKind() == MemberReferenceKind.Field)
            {
                var ft = mr.DecodeFieldSignature(CanonProvider.I, null);
                Members.Add(new(owner, name, true, ft, name, h));
            }
            else
            {
                var sig = mr.DecodeMethodSignature(CanonProvider.I, null);
                Members.Add(new(owner, name, false, "", Names.MethodKey(name, sig), h));
            }
        }
        // valuetype/class markers used for Godot types anywhere in signatures (member refs, typespecs, locals).
        var kindProvider = new KindProvider(this);
        foreach (var h in _r.MemberReferences)
        {
            var mr = _r.GetMemberReference(h);
            try
            {
                if (mr.GetKind() == MemberReferenceKind.Field) mr.DecodeFieldSignature(kindProvider, null);
                else mr.DecodeMethodSignature(kindProvider, null);
            }
            catch { }
        }
        for (int i = 1; i <= _r.GetTableRowCount(TableIndex.TypeSpec); i++)
            try { _r.GetTypeSpecification(MetadataTokens.TypeSpecificationHandle(i)).DecodeSignature(kindProvider, null); } catch { }
        for (int i = 1; i <= _r.GetTableRowCount(TableIndex.StandAloneSig); i++)
            try
            {
                var ss = _r.GetStandaloneSignature(MetadataTokens.StandaloneSignatureHandle(i));
                if (ss.GetKind() == StandaloneSignatureKind.LocalVariables) ss.DecodeLocalSignature(kindProvider, null);
            }
            catch { }
        foreach (var h in _r.MethodDefinitions)
            try { _r.GetMethodDefinition(h).DecodeSignature(kindProvider, null); } catch { }
        foreach (var h in _r.FieldDefinitions)
            try { _r.GetFieldDefinition(h).DecodeSignature(kindProvider, null); } catch { }
    }

    private bool IsGodot(TypeReferenceHandle h)
    {
        var tr = _r.GetTypeReference(h);
        if (tr.ResolutionScope.Kind == HandleKind.TypeReference) return IsGodot((TypeReferenceHandle)tr.ResolutionScope);
        return tr.ResolutionScope.Kind == HandleKind.AssemblyReference && _godot.Contains((AssemblyReferenceHandle)tr.ResolutionScope);
    }

    private string? OwnerName(EntityHandle parent)
    {
        if (parent.Kind == HandleKind.TypeReference)
            return IsGodot((TypeReferenceHandle)parent) ? Names.Of(_r, (TypeReferenceHandle)parent) : null;
        if (parent.Kind == HandleKind.TypeSpecification)
        {
            var br = _r.GetBlobReader(_r.GetTypeSpecification((TypeSpecificationHandle)parent).Signature);
            if (br.ReadSignatureTypeCode() != SignatureTypeCode.GenericTypeInstance) return null;
            br.ReadSignatureTypeCode();
            var g = br.ReadTypeHandle();
            return g.Kind == HandleKind.TypeReference && IsGodot((TypeReferenceHandle)g) ? Names.Of(_r, (TypeReferenceHandle)g) : null;
        }
        return null;
    }

    private sealed class KindProvider : ISignatureTypeProvider<int, object?>
    {
        private readonly Sts2Refs _o;
        public KindProvider(Sts2Refs o) { _o = o; }
        public int GetArrayType(int e, ArrayShape s) => 0;
        public int GetByReferenceType(int e) => 0;
        public int GetFunctionPointerType(MethodSignature<int> s) => 0;
        public int GetGenericInstantiation(int g, ImmutableArray<int> a) => 0;
        public int GetGenericMethodParameter(object? c, int i) => 0;
        public int GetGenericTypeParameter(object? c, int i) => 0;
        public int GetModifiedType(int m, int u, bool r) => 0;
        public int GetPinnedType(int e) => 0;
        public int GetPointerType(int e) => 0;
        public int GetPrimitiveType(PrimitiveTypeCode t) => 0;
        public int GetSZArrayType(int e) => 0;
        public int GetTypeFromDefinition(MetadataReader r, TypeDefinitionHandle h, byte k) => 0;
        public int GetTypeFromReference(MetadataReader r, TypeReferenceHandle h, byte k)
        {
            if (_o.IsGodot(h) && (k == 0x11 || k == 0x12)) _o.Kinds[Names.Of(r, h)] = k == 0x11;
            return 0;
        }
        public int GetTypeFromSpecification(MetadataReader r, object? c, TypeSpecificationHandle h, byte k) =>
            r.GetTypeSpecification(h).DecodeSignature(this, c);
    }
}

/// <summary>Index of an assembly's type definitions with canonical member keys.</summary>
sealed class Asm
{
    public sealed class TypeInfo
    {
        public TypeDefinitionHandle Handle;
        public string Name = "";
        public string? BaseType;
        public bool IsValueType, IsEnum, IsInterface, IsDelegate;
        public readonly HashSet<string> Methods = new();
        public readonly Dictionary<string, MethodDefinitionHandle> MethodHandles = new();
        public readonly Dictionary<string, string> Fields = new();
        public readonly Dictionary<string, FieldDefinitionHandle> FieldHandles = new();
        public readonly Dictionary<string, string> VirtualAccess = new();
        public readonly HashSet<string> MemberNames = new();
    }

    public readonly MetadataReader R;
    public readonly Dictionary<string, TypeInfo> Types = new();

    public Asm(string path)
    {
        var pe = new PEReader(File.OpenRead(path));
        R = pe.GetMetadataReader();
        foreach (var h in R.TypeDefinitions)
        {
            var td = R.GetTypeDefinition(h);
            var ti = new TypeInfo { Handle = h, Name = Names.Of(R, h) };
            if (!td.BaseType.IsNil) ti.BaseType = TypeName(td.BaseType);
            ti.IsValueType = ti.BaseType is "System.ValueType" or "System.Enum";
            ti.IsEnum = ti.BaseType == "System.Enum";
            ti.IsDelegate = ti.BaseType == "System.MulticastDelegate";
            ti.IsInterface = (td.Attributes & TypeAttributes.Interface) != 0;
            foreach (var mh in td.GetMethods())
            {
                var md = R.GetMethodDefinition(mh);
                var name = R.GetString(md.Name);
                var key = Names.MethodKey(name, md.DecodeSignature(CanonProvider.I, null));
                ti.Methods.Add(key);
                ti.MethodHandles[key] = mh;
                ti.MemberNames.Add(name);
                if ((md.Attributes & MethodAttributes.Virtual) != 0)
                    ti.VirtualAccess[key] = (md.Attributes & MethodAttributes.MemberAccessMask).ToString();
            }
            foreach (var fh in td.GetFields())
            {
                var fd = R.GetFieldDefinition(fh);
                var name = R.GetString(fd.Name);
                ti.Fields[name] = fd.DecodeSignature(CanonProvider.I, null);
                ti.FieldHandles[name] = fh;
                ti.MemberNames.Add(name);
            }
            foreach (var ph in td.GetProperties()) ti.MemberNames.Add(R.GetString(R.GetPropertyDefinition(ph).Name));
            foreach (var eh in td.GetEvents()) ti.MemberNames.Add(R.GetString(R.GetEventDefinition(eh).Name));
            Types[ti.Name] = ti;
        }
    }

    public string TypeName(EntityHandle h) => h.Kind switch
    {
        HandleKind.TypeDefinition => Names.Of(R, (TypeDefinitionHandle)h),
        HandleKind.TypeReference => Names.Of(R, (TypeReferenceHandle)h),
        HandleKind.TypeSpecification => R.GetTypeSpecification((TypeSpecificationHandle)h).DecodeSignature(CanonProvider.I, null),
        _ => "?",
    };
}

/// <summary>C# source text of a signature type (custom modifiers dropped, byref as a trailing '&amp;').</summary>
sealed class CsProvider : ISignatureTypeProvider<string, CsProvider.Ctx>
{
    public sealed record Ctx(IReadOnlyList<string> TypeParams, IReadOnlyList<string> MethodParams);
    private readonly Asm _a;
    public readonly HashSet<string> SeenGodotTypes = new();
    public CsProvider(Asm a) { _a = a; }

    public string GetArrayType(string e, ArrayShape s) => $"{e}[{new string(',', s.Rank - 1)}]";
    public string GetByReferenceType(string e) => e + "&";
    public string GetFunctionPointerType(MethodSignature<string> s) => $"delegate* unmanaged<{string.Join(", ", s.ParameterTypes.Append(s.ReturnType))}>";
    public string GetGenericInstantiation(string g, ImmutableArray<string> a) => $"{g}<{string.Join(", ", a)}>";
    public string GetGenericMethodParameter(Ctx c, int i) => c.MethodParams[i];
    public string GetGenericTypeParameter(Ctx c, int i) => c.TypeParams[i];
    public string GetModifiedType(string m, string u, bool req) => u;
    public string GetPinnedType(string e) => e;
    public string GetPointerType(string e) => e + "*";
    public string GetPrimitiveType(PrimitiveTypeCode t) => t switch
    {
        PrimitiveTypeCode.Boolean => "bool", PrimitiveTypeCode.Char => "char", PrimitiveTypeCode.SByte => "sbyte",
        PrimitiveTypeCode.Byte => "byte", PrimitiveTypeCode.Int16 => "short", PrimitiveTypeCode.UInt16 => "ushort",
        PrimitiveTypeCode.Int32 => "int", PrimitiveTypeCode.UInt32 => "uint", PrimitiveTypeCode.Int64 => "long",
        PrimitiveTypeCode.UInt64 => "ulong", PrimitiveTypeCode.Single => "float", PrimitiveTypeCode.Double => "double",
        PrimitiveTypeCode.String => "string", PrimitiveTypeCode.Object => "object", PrimitiveTypeCode.IntPtr => "nint",
        PrimitiveTypeCode.UIntPtr => "nuint", PrimitiveTypeCode.Void => "void", PrimitiveTypeCode.TypedReference => "global::System.TypedReference",
        _ => throw new NotSupportedException(t.ToString()),
    };
    public string GetSZArrayType(string e) => e + "[]";
    public string GetTypeFromDefinition(MetadataReader r, TypeDefinitionHandle h, byte k)
    {
        var n = Names.Of(r, h);
        SeenGodotTypes.Add(n);
        return Cs(n);
    }
    public string GetTypeFromReference(MetadataReader r, TypeReferenceHandle h, byte k) => Cs(Names.Of(r, h));
    public string GetTypeFromSpecification(MetadataReader r, Ctx c, TypeSpecificationHandle h, byte k) =>
        r.GetTypeSpecification(h).DecodeSignature(this, c);

    /// <summary>"Godot.Collections.Array`1" -> "global::Godot.Collections.Array", "Godot.Node/SignalName" -> "global::Godot.Node.SignalName".</summary>
    public static string Cs(string metaName)
    {
        var parts = metaName.Split('/').Select(StripArity);
        return "global::" + string.Join(".", parts);
    }

    public static string StripArity(string s)
    {
        int i = s.IndexOf('`');
        return i < 0 ? s : s[..i];
    }
}

sealed class Emitter
{
    private readonly Asm _real, _stub;
    private readonly CsProvider _cs;
    public readonly List<string> Warnings = new();

    private sealed class TypeEmit
    {
        public string Name = "";
        public bool HandWritten;
        public bool FullDecl;          // emit kind, base, interfaces (type absent from the stub)
        public readonly List<MemberEmit> Members = new();
        public readonly HashSet<string> Keys = new();
    }

    private sealed class MemberEmit
    {
        public string Key = "";
        public Func<Func<string, string, bool>, string> Render = _ => "";
    }

    private readonly SortedDictionary<string, TypeEmit> _types = new(StringComparer.Ordinal);
    private readonly Queue<string> _pendingTypes = new();
    public int TypeCount => _types.Values.Count(t => t.FullDecl);
    public int MemberCount => _types.Values.Sum(t => t.Members.Count);

    public Emitter(Asm real, Asm stub)
    {
        _real = real;
        _stub = stub;
        _cs = new CsProvider(real);
    }

    // ───────────── types

    public void NeedType(string name)
    {
        if (_types.ContainsKey(name)) return;
        if (_stub.Types.ContainsKey(name))
        {
            _types[name] = new TypeEmit { Name = name, HandWritten = true };
            return;
        }
        if (!_real.Types.TryGetValue(name, out var rt))
        {
            Warnings.Add($"type {name} referenced but not in the real GodotSharp");
            return;
        }
        if (Program.Semantic.Contains(Program.TopLevel(name)))
        {
            Warnings.Add($"SEMANTIC type {name} is missing from the hand-written stub");
            return;
        }
        var te = new TypeEmit { Name = name, FullDecl = true };
        _types[name] = te;
        var td = _real.R.GetTypeDefinition(rt.Handle);
        if (!td.GetDeclaringType().IsNil) NeedType(Names.Of(_real.R, td.GetDeclaringType()));
        if (!td.BaseType.IsNil) NeedTypeIfGodot(td.BaseType);
        foreach (var ih in td.GetInterfaceImplementations())
            NeedTypeIfGodot(_real.R.GetInterfaceImplementation(ih).Interface);
        if (rt.IsEnum || rt.IsDelegate)
        {
            // Delegates need their Invoke signature types; enums their underlying type only.
            if (rt.IsDelegate)
                foreach (var mh in td.GetMethods())
                    if (_real.R.GetString(_real.R.GetMethodDefinition(mh).Name) == "Invoke") SigTypes(mh);
        }
    }

    private void NeedTypeIfGodot(EntityHandle h)
    {
        if (h.Kind == HandleKind.TypeDefinition) NeedType(Names.Of(_real.R, (TypeDefinitionHandle)h));
        else if (h.Kind == HandleKind.TypeSpecification)
        {
            var before = _cs.SeenGodotTypes.ToList();
            _real.R.GetTypeSpecification((TypeSpecificationHandle)h).DecodeSignature(_cs, new CsProvider.Ctx(Enumerable.Range(0, 8).Select(i => "T" + i).ToList(), Array.Empty<string>()));
            foreach (var t in _cs.SeenGodotTypes.Except(before).ToList()) NeedType(t);
        }
    }

    private void SigTypes(MethodDefinitionHandle mh)
    {
        _cs.SeenGodotTypes.Clear();
        var md = _real.R.GetMethodDefinition(mh);
        md.DecodeSignature(_cs, CtxFor(mh));
        foreach (var t in _cs.SeenGodotTypes.ToList()) _pendingTypes.Enqueue(t);
    }

    public void Close()
    {
        while (_pendingTypes.Count > 0) NeedType(_pendingTypes.Dequeue());
        // Every generated class needs an accessible parameterless constructor: sts2 classes derive from them.
        foreach (var te in _types.Values.Where(t => t.FullDecl).ToList())
        {
            var rt = _real.Types[te.Name];
            var td = _real.R.GetTypeDefinition(rt.Handle);
            if (rt.IsValueType || rt.IsInterface || rt.IsDelegate || IsStatic(td)) continue;
            const string key = "instance Void .ctor`0()";
            if (te.Keys.Contains(key)) continue;
            te.Keys.Add(key);
            var simple = CsProvider.StripArity(te.Name.Split('/').Last().Split('.').Last());
            te.Members.Add(new MemberEmit { Key = key, Render = _ => $"public {simple}() {{ }}" });
        }
        while (_pendingTypes.Count > 0) NeedType(_pendingTypes.Dequeue());
    }

    private static bool IsStatic(TypeDefinition td) =>
        (td.Attributes & TypeAttributes.Abstract) != 0 && (td.Attributes & TypeAttributes.Sealed) != 0;

    // ───────────── members

    public bool NeedMember(Sts2Refs.Member m)
    {
        NeedType(m.Owner);
        if (!_types.TryGetValue(m.Owner, out var te)) return false;
        var rt = _real.Types[m.Owner];
        if (m.IsField)
        {
            if (!rt.FieldHandles.TryGetValue(m.Name, out var fh)) return false;
            AddField(te, rt, fh);
            return true;
        }
        if (!rt.MethodHandles.TryGetValue(m.Key, out var mh)) return false;
        AddMethodOrAccessor(te, rt, mh);
        return true;
    }

    private CsProvider.Ctx CtxFor(MethodDefinitionHandle mh)
    {
        var md = _real.R.GetMethodDefinition(mh);
        var td = _real.R.GetTypeDefinition(md.GetDeclaringType());
        return new CsProvider.Ctx(GenericNames(td.GetGenericParameters()), GenericNames(md.GetGenericParameters()));
    }

    private List<string> GenericNames(GenericParameterHandleCollection c) =>
        c.Select(h => _real.R.GetString(_real.R.GetGenericParameter(h).Name)).ToList();

    private void AddField(TypeEmit te, Asm.TypeInfo rt, FieldDefinitionHandle fh)
    {
        var fd = _real.R.GetFieldDefinition(fh);
        var name = _real.R.GetString(fd.Name);
        if (!te.Keys.Add("field " + name)) return;
        var td = _real.R.GetTypeDefinition(rt.Handle);
        _cs.SeenGodotTypes.Clear();
        var type = fd.DecodeSignature(_cs, new CsProvider.Ctx(GenericNames(td.GetGenericParameters()), Array.Empty<string>()));
        foreach (var t in _cs.SeenGodotTypes.ToList()) _pendingTypes.Enqueue(t);
        var attrs = fd.Attributes;
        var acc = Access((MethodAttributes)(int)(attrs & FieldAttributes.FieldAccessMask));
        bool isStatic = (attrs & FieldAttributes.Static) != 0;
        bool ro = (attrs & FieldAttributes.InitOnly) != 0;
        string init = isStatic && type == "global::Godot.StringName" ? $" = new global::Godot.StringName(\"{name}\")" : "";
        te.Members.Add(new MemberEmit
        {
            Key = "field " + name,
            Render = _ => $"{acc} {(isStatic ? "static " : "")}{(ro ? "readonly " : "")}{type} {Ident(name)}{init};",
        });
    }

    private void AddMethodOrAccessor(TypeEmit te, Asm.TypeInfo rt, MethodDefinitionHandle mh)
    {
        var td = _real.R.GetTypeDefinition(rt.Handle);
        foreach (var ph in td.GetProperties())
        {
            var pd = _real.R.GetPropertyDefinition(ph);
            var acc = pd.GetAccessors();
            if (acc.Getter == mh || acc.Setter == mh) { AddProperty(te, rt, ph); return; }
        }
        foreach (var eh in td.GetEvents())
        {
            var ed = _real.R.GetEventDefinition(eh);
            var acc = ed.GetAccessors();
            if (acc.Adder == mh || acc.Remover == mh) { AddEvent(te, rt, eh); return; }
        }
        AddMethod(te, rt, mh);
    }

    private sealed record ParamInfo(string Type, string Name, string Prefix);

    private (string ret, List<ParamInfo> ps, MethodSignature<string> canon) Signature(MethodDefinitionHandle mh)
    {
        var md = _real.R.GetMethodDefinition(mh);
        _cs.SeenGodotTypes.Clear();
        var sig = md.DecodeSignature(_cs, CtxFor(mh));
        foreach (var t in _cs.SeenGodotTypes.ToList()) _pendingTypes.Enqueue(t);
        var canon = md.DecodeSignature(CanonProvider.I, null);
        var names = new string[sig.ParameterTypes.Length];
        var attrs = new ParameterAttributes[sig.ParameterTypes.Length];
        var readOnly = new bool[sig.ParameterTypes.Length];
        foreach (var ph in md.GetParameters())
        {
            var p = _real.R.GetParameter(ph);
            if (p.SequenceNumber == 0 || p.SequenceNumber > names.Length) continue;
            names[p.SequenceNumber - 1] = _real.R.GetString(p.Name);
            attrs[p.SequenceNumber - 1] = p.Attributes;
            foreach (var ch in p.GetCustomAttributes())
                if (AttrName(ch) == "System.Runtime.CompilerServices.IsReadOnlyAttribute") readOnly[p.SequenceNumber - 1] = true;
        }
        var used = new HashSet<string>();
        var ps = new List<ParamInfo>();
        for (int i = 0; i < sig.ParameterTypes.Length; i++)
        {
            var t = sig.ParameterTypes[i];
            string prefix = "";
            if (t.EndsWith("&"))
            {
                t = t[..^1];
                bool modreqIn = canon.ParameterTypes[i].Contains("modreq(System.Runtime.InteropServices.InAttribute)");
                prefix = readOnly[i] || modreqIn ? "in " : (attrs[i] & ParameterAttributes.Out) != 0 ? "out " : "ref ";
            }
            var n = string.IsNullOrEmpty(names[i]) ? "p" + i : names[i];
            while (!used.Add(n)) n += "_";
            ps.Add(new ParamInfo(t, Ident(n), prefix));
        }
        return (sig.ReturnType, ps, canon);
    }

    private string AttrName(CustomAttributeHandle ch)
    {
        var ca = _real.R.GetCustomAttribute(ch);
        EntityHandle parent = ca.Constructor.Kind == HandleKind.MemberReference
            ? _real.R.GetMemberReference((MemberReferenceHandle)ca.Constructor).Parent
            : _real.R.GetMethodDefinition((MethodDefinitionHandle)ca.Constructor).GetDeclaringType();
        return _real.TypeName(parent);
    }

    private static string Access(MethodAttributes a) => (a & MethodAttributes.MemberAccessMask) switch
    {
        MethodAttributes.Public => "public",
        MethodAttributes.Family => "protected",
        MethodAttributes.FamORAssem => "protected internal",
        MethodAttributes.Assembly => "internal",
        MethodAttributes.FamANDAssem => "private protected",
        _ => "private",
    };

    /// <summary>"virtual" / "override" / "" for a real method, resolved against what the final stub declares in the base chain.</summary>
    private string Virtuality(Asm.TypeInfo rt, MethodDefinition md, string key, Func<string, string, bool> declaredVirtualInBase, bool isValueOrSealed)
    {
        var a = md.Attributes;
        if ((a & MethodAttributes.Virtual) == 0 || (a & MethodAttributes.Static) != 0) return "";
        var name = _real.R.GetString(md.Name);
        bool objectOverride = (name is "ToString" or "GetHashCode" or "Equals" or "Finalize") && (a & MethodAttributes.NewSlot) == 0;
        if (objectOverride) return "override ";
        if ((a & MethodAttributes.Final) != 0 && (a & MethodAttributes.NewSlot) != 0) return "";   // interface implementation
        if ((a & MethodAttributes.NewSlot) == 0)
            return declaredVirtualInBase(rt.Name, key) ? "override " : (isValueOrSealed ? "" : "virtual ");
        return isValueOrSealed ? "" : "virtual ";
    }

    private void AddMethod(TypeEmit te, Asm.TypeInfo rt, MethodDefinitionHandle mh)
    {
        var md = _real.R.GetMethodDefinition(mh);
        var name = _real.R.GetString(md.Name);
        var (ret, ps, canon) = Signature(mh);
        var key = Names.MethodKey(name, canon);
        if (!te.Keys.Add(key)) return;
        var td = _real.R.GetTypeDefinition(rt.Handle);
        bool isStatic = (md.Attributes & MethodAttributes.Static) != 0;
        var acc = Access(md.Attributes);
        var gen = GenericNames(md.GetGenericParameters());
        var genText = gen.Count > 0 ? "<" + string.Join(", ", gen) + ">" : "";
        var plist = string.Join(", ", ps.Select(p => $"{p.Prefix}{p.Type} {p.Name}"));
        bool unsafeSig = ret.Contains('*') || ps.Any(p => p.Type.Contains('*'));
        var simple = CsProvider.StripArity(rt.Name.Split('/').Last().Split('.').Last());
        bool sealedOrValue = rt.IsValueType || (td.Attributes & TypeAttributes.Sealed) != 0;
        string outs = string.Concat(ps.Where(p => p.Prefix == "out ").Select(p => $"{p.Name} = default!; "));

        if (name == ".ctor")
        {
            te.Members.Add(new MemberEmit { Key = key, Render = _ => $"{acc} {(unsafeSig ? "unsafe " : "")}{simple}({plist}) {{ {outs}}}" });
            return;
        }
        if (name == ".cctor" || name == "Finalize") return;
        if (rt.IsInterface)
        {
            te.Members.Add(new MemberEmit { Key = key, Render = _ => isStatic
                ? $"public static {ret} {Ident(name)}{genText}({plist}) {{ {outs}{Ret(ret)}}}"
                : $"{ret} {Ident(name)}{genText}({plist});" });
            return;
        }
        if (name.StartsWith("op_") && Operator(name, ps.Count) is { } op)
        {
            // C# requires paired operators; add the partner from the real type when needed.
            if (Partner(name) is { } partner)
                foreach (var (k2, h2) in rt.MethodHandles)
                    if (_real.R.GetString(_real.R.GetMethodDefinition(h2).Name) == partner && !te.Keys.Contains(k2)) AddMethod(te, rt, h2);
            string decl = name is "op_Implicit" or "op_Explicit"
                ? $"{acc} static {(unsafeSig ? "unsafe " : "")}{op} operator {ret}({plist})"
                : $"{acc} static {(unsafeSig ? "unsafe " : "")}{ret} operator {op}({plist})";
            te.Members.Add(new MemberEmit { Key = key, Render = _ => $"{decl} {{ {outs}{Ret(ret)}}}" });
            return;
        }
        te.Members.Add(new MemberEmit
        {
            Key = key,
            Render = baseVirtual =>
            {
                var virt = Virtuality(rt, md, key, baseVirtual, sealedOrValue);
                return $"{acc} {(isStatic ? "static " : "")}{virt}{(unsafeSig ? "unsafe " : "")}{ret} {Ident(name)}{genText}({plist}) {{ {outs}{Ret(ret)}}}";
            },
        });
    }

    private void AddProperty(TypeEmit te, Asm.TypeInfo rt, PropertyDefinitionHandle ph)
    {
        var pd = _real.R.GetPropertyDefinition(ph);
        var name = _real.R.GetString(pd.Name);
        var acc = pd.GetAccessors();
        var any = !acc.Getter.IsNil ? acc.Getter : acc.Setter;
        var md = _real.R.GetMethodDefinition(any);
        var (ret, ps, canon) = Signature(any);
        string type = !acc.Getter.IsNil ? ret : ps[^1].Type;
        bool refReturn = type.EndsWith("&");
        if (refReturn)
            type = (canon.ReturnType.Contains("modreq(System.Runtime.InteropServices.InAttribute)") ? "ref readonly " : "ref ") + type[..^1];
        var idx = !acc.Getter.IsNil ? ps : ps.Take(ps.Count - 1).ToList();
        var pkey = $"prop {name}({string.Join(",", idx.Select(p => p.Type))})";
        if (!te.Keys.Add(pkey)) return;
        foreach (var h in new[] { acc.Getter, acc.Setter })
            if (!h.IsNil) te.Keys.Add(Names.MethodKey(_real.R.GetString(_real.R.GetMethodDefinition(h).Name), _real.R.GetMethodDefinition(h).DecodeSignature(CanonProvider.I, null)));
        if (_stub.Types.TryGetValue(rt.Name, out var st) && st.MemberNames.Contains(name))
        {
            Warnings.Add($"property {rt.Name}.{name} exists in the hand-written stub with different accessors; add them there");
            return;
        }
        bool isStatic = (md.Attributes & MethodAttributes.Static) != 0;
        var td = _real.R.GetTypeDefinition(rt.Handle);
        bool sealedOrValue = rt.IsValueType || (td.Attributes & TypeAttributes.Sealed) != 0;
        string Acc(MethodDefinitionHandle h) => Access(_real.R.GetMethodDefinition(h).Attributes);
        var outer = Access(md.Attributes);
        if (!acc.Getter.IsNil && !acc.Setter.IsNil && Acc(acc.Getter) != Acc(acc.Setter))
            outer = AccessRank(Acc(acc.Getter)) >= AccessRank(Acc(acc.Setter)) ? Acc(acc.Getter) : Acc(acc.Setter);
        string Part(MethodDefinitionHandle h, string kw) => h.IsNil ? "" : (Acc(h) != outer ? Acc(h) + " " : "") + kw + "; ";
        bool unsafeSig = type.Contains('*');
        var mkey = Names.MethodKey(_real.R.GetString(md.Name), canon);
        if (rt.IsInterface)
        {
            te.Members.Add(new MemberEmit { Key = pkey, Render = _ => $"{type} {(idx.Count > 0 ? "this[" + string.Join(", ", idx.Select(p => $"{p.Type} {p.Name}")) + "]" : Ident(name))} {{ {Part(acc.Getter, "get")}{Part(acc.Setter, "set")}}}" });
            return;
        }
        te.Members.Add(new MemberEmit
        {
            Key = pkey,
            Render = baseVirtual =>
            {
                var virt = Virtuality(rt, md, mkey, baseVirtual, sealedOrValue);
                if (idx.Count > 0)
                {
                    var plist = string.Join(", ", idx.Select(p => $"{p.Type} {p.Name}"));
                    var get = acc.Getter.IsNil ? "" : refReturn ? "get => throw new global::System.NotSupportedException(); " : $"get => {RetExpr(type)}; ";
                    var set = acc.Setter.IsNil ? "" : "set { } ";
                    return $"{outer} {virt}{(unsafeSig ? "unsafe " : "")}{type} this[{plist}] {{ {get}{set}}}";
                }
                if (refReturn)
                    return $"{outer} {(isStatic ? "static " : "")}{virt}{type} {Ident(name)} {{ get => throw new global::System.NotSupportedException(); }}";
                var init = isStatic || !IsValueTypeName(type) && RetExpr(type) != "default!" ? $" = {RetExpr(type)};" : "";
                if (rt.IsValueType && !isStatic) init = "";
                return $"{outer} {(isStatic ? "static " : "")}{virt}{(unsafeSig ? "unsafe " : "")}{type} {Ident(name)} {{ {Part(acc.Getter, "get")}{Part(acc.Setter, "set")}}}{init}";
            },
        });
    }

    private static bool IsValueTypeName(string t) => t is "bool" or "int" or "long" or "float" or "double" or "uint" or "ulong" or "short" or "ushort" or "byte" or "sbyte" or "char";

    private static int AccessRank(string a) => a switch { "public" => 5, "protected internal" => 4, "protected" => 3, "internal" => 2, "private protected" => 1, _ => 0 };

    private void AddEvent(TypeEmit te, Asm.TypeInfo rt, EventDefinitionHandle eh)
    {
        var ed = _real.R.GetEventDefinition(eh);
        var name = _real.R.GetString(ed.Name);
        var acc = ed.GetAccessors();
        var md = _real.R.GetMethodDefinition(acc.Adder);
        var (_, ps, canon) = Signature(acc.Adder);
        var ekey = "event " + name;
        if (!te.Keys.Add(ekey)) return;
        foreach (var h in new[] { acc.Adder, acc.Remover })
            if (!h.IsNil) te.Keys.Add(Names.MethodKey(_real.R.GetString(_real.R.GetMethodDefinition(h).Name), _real.R.GetMethodDefinition(h).DecodeSignature(CanonProvider.I, null)));
        bool isStatic = (md.Attributes & MethodAttributes.Static) != 0;
        var a = Access(md.Attributes);
        var td = _real.R.GetTypeDefinition(rt.Handle);
        bool sealedOrValue = rt.IsValueType || (td.Attributes & TypeAttributes.Sealed) != 0;
        var mkey = Names.MethodKey(_real.R.GetString(md.Name), canon);
        te.Members.Add(new MemberEmit
        {
            Key = ekey,
            Render = baseVirtual => rt.IsInterface
                ? $"event {ps[0].Type} {Ident(name)};"
                : $"{a} {(isStatic ? "static " : "")}{Virtuality(rt, md, mkey, baseVirtual, sealedOrValue)}event {ps[0].Type}? {Ident(name)};",
        });
    }

    private static string? Operator(string name, int arity) => name switch
    {
        "op_Implicit" => "implicit", "op_Explicit" => "explicit",
        "op_Addition" => "+", "op_Subtraction" => "-", "op_Multiply" => "*", "op_Division" => "/", "op_Modulus" => "%",
        "op_UnaryNegation" => "-", "op_UnaryPlus" => "+", "op_Equality" => "==", "op_Inequality" => "!=",
        "op_LessThan" => "<", "op_GreaterThan" => ">", "op_LessThanOrEqual" => "<=", "op_GreaterThanOrEqual" => ">=",
        "op_BitwiseAnd" => "&", "op_BitwiseOr" => "|", "op_ExclusiveOr" => "^", "op_LogicalNot" => "!", "op_OnesComplement" => "~",
        "op_True" => "true", "op_False" => "false", "op_Increment" => "++", "op_Decrement" => "--",
        _ => null,
    };

    private static string? Partner(string name) => name switch
    {
        "op_Equality" => "op_Inequality", "op_Inequality" => "op_Equality",
        "op_LessThan" => "op_GreaterThan", "op_GreaterThan" => "op_LessThan",
        "op_LessThanOrEqual" => "op_GreaterThanOrEqual", "op_GreaterThanOrEqual" => "op_LessThanOrEqual",
        "op_True" => "op_False", "op_False" => "op_True",
        _ => null,
    };

    private static string Ret(string t) => t == "void" ? "" : t.EndsWith("&") ? "throw new global::System.NotSupportedException(); " : $"return {RetExpr(t)}; ";

    /// <summary>Inert return values: empty strings / collections / completed tasks rather than null where callers iterate or await.</summary>
    private static string RetExpr(string t)
    {
        if (t == "string") return "\"\"";
        if (t is "global::Godot.StringName" or "global::Godot.NodePath" or "global::Godot.SignalAwaiter") return $"new {t}()";
        if (t.StartsWith("global::Godot.Collections.Array") || t.StartsWith("global::Godot.Collections.Dictionary")) return $"new {t}()";
        if (t.EndsWith("[]") && !t.Contains(',')) return $"global::System.Array.Empty<{t[..^2]}>()";
        if (t == "global::System.Threading.Tasks.Task") return "global::System.Threading.Tasks.Task.CompletedTask";
        if (t.StartsWith("global::System.Threading.Tasks.Task<")) return $"global::System.Threading.Tasks.Task.FromResult<{t["global::System.Threading.Tasks.Task<".Length..^1]}>(default!)";
        return "default!";
    }

    private static readonly HashSet<string> Keywords = new()
    {
        "abstract","as","base","bool","break","byte","case","catch","char","checked","class","const","continue","decimal","default",
        "delegate","do","double","else","enum","event","explicit","extern","false","finally","fixed","float","for","foreach","goto",
        "if","implicit","in","int","interface","internal","is","lock","long","namespace","new","null","object","operator","out",
        "override","params","private","protected","public","readonly","ref","return","sbyte","sealed","short","sizeof","stackalloc",
        "static","string","struct","switch","this","throw","true","try","typeof","uint","ulong","unchecked","unsafe","ushort","using",
        "virtual","void","volatile","while",
    };

    private static string Ident(string n) => Keywords.Contains(n) ? "@" + n : n;

    // ───────────── rendering

    /// <summary>Does the final stub declare (or will it generate) a virtual member with this key in a base type of <paramref name="type"/>?</summary>
    private bool DeclaredVirtualInBase(string type, string key)
    {
        var name = key.Split(' ', 3)[2];
        for (var t = BaseOf(type); t != null; t = BaseOf(t))
        {
            if (_stub.Types.TryGetValue(t, out var st) && st.VirtualAccess.ContainsKey(key)) return true;
            if (_types.TryGetValue(t, out var te) && te.Keys.Contains(key) && _real.Types.TryGetValue(t, out var rt)
                && rt.MethodHandles.TryGetValue(key, out var mh) && (_real.R.GetMethodDefinition(mh).Attributes & MethodAttributes.Virtual) != 0)
                return true;
        }
        return false;
    }

    private string? BaseOf(string type)
    {
        if (_real.Types.TryGetValue(type, out var rt)) return rt.BaseType != null && _real.Types.ContainsKey(rt.BaseType) ? rt.BaseType : null;
        if (_stub.Types.TryGetValue(type, out var st)) return st.BaseType != null && _stub.Types.ContainsKey(st.BaseType) ? st.BaseType : null;
        return null;
    }

    public string Render()
    {
        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated> by engine/tools/GodotStubGen from the real GodotSharp API and the references of sts2.dll.");
        sb.AppendLine("// Do not edit: rerun engine/tools/gen_stubs.sh after a game update. Hand-written semantic types live next to this folder.");
        sb.AppendLine("// </auto-generated>");
        sb.AppendLine("#pragma warning disable CS0067, CS0108, CS0114, CS0109, CS0414, CS0649, CS0660, CS0661, CS0169, CS8618, CS8625, CS8603, CS8600, CS8601, CS8602, CS8765, CS8767, CS0628, CS3021");
        sb.AppendLine("#nullable disable");
        var roots = _types.Values.Where(t => !t.Name.Contains('/') && (t.FullDecl || t.Members.Count > 0 || HasEmittedNested(t.Name)))
            .GroupBy(t => Namespace(t.Name)).OrderBy(g => g.Key, StringComparer.Ordinal);
        foreach (var g in roots)
        {
            sb.AppendLine();
            sb.AppendLine($"namespace {g.Key}");
            sb.AppendLine("{");
            foreach (var t in g.OrderBy(t => t.Name, StringComparer.Ordinal)) RenderType(sb, t, 1);
            sb.AppendLine("}");
        }
        return sb.ToString();
    }

    private bool HasEmittedNested(string name) =>
        _types.Values.Any(t => t.Name.StartsWith(name + "/") && (t.FullDecl || t.Members.Count > 0));

    private static string Namespace(string name)
    {
        int i = name.LastIndexOf('.');
        return i < 0 ? "Godot" : name[..i];
    }

    private void RenderType(StringBuilder sb, TypeEmit t, int depth)
    {
        var ind = new string('\t', depth);
        var rt = _real.Types[t.Name];
        var td = _real.R.GetTypeDefinition(rt.Handle);
        var simple = CsProvider.StripArity(t.Name.Split('/').Last().Split('.').Last());
        var gp = GenericNames(td.GetGenericParameters());
        // Nested types of generic types repeat the outer parameters in metadata; C# declares only their own.
        int outerArity = 0;
        if (!td.GetDeclaringType().IsNil) outerArity = _real.R.GetTypeDefinition(td.GetDeclaringType()).GetGenericParameters().Count;
        var ownGp = gp.Skip(outerArity).ToList();
        var gtext = ownGp.Count > 0 ? "<" + string.Join(", ", ownGp) + ">" : "";
        var ctx = new CsProvider.Ctx(gp, Array.Empty<string>());
        var access = (td.Attributes & TypeAttributes.VisibilityMask) is TypeAttributes.Public or TypeAttributes.NestedPublic ? "public" :
                     (td.Attributes & TypeAttributes.VisibilityMask) is TypeAttributes.NestedFamily ? "protected" : "public";
        if (rt.IsEnum)
        {
            if (!t.FullDecl) return;
            string underlying = "int";
            var vals = new List<string>();
            foreach (var fh in td.GetFields())
            {
                var fd = _real.R.GetFieldDefinition(fh);
                var fname = _real.R.GetString(fd.Name);
                if ((fd.Attributes & FieldAttributes.Static) == 0) { underlying = fd.DecodeSignature(_cs, ctx); continue; }
                var c = _real.R.GetConstant(fd.GetDefaultValue());
                var br = _real.R.GetBlobReader(c.Value);
                object v = c.TypeCode switch
                {
                    ConstantTypeCode.Int32 => br.ReadInt32(), ConstantTypeCode.Int64 => br.ReadInt64(), ConstantTypeCode.UInt32 => br.ReadUInt32(),
                    ConstantTypeCode.UInt64 => br.ReadUInt64(), ConstantTypeCode.Int16 => br.ReadInt16(), ConstantTypeCode.UInt16 => br.ReadUInt16(),
                    ConstantTypeCode.Byte => br.ReadByte(), ConstantTypeCode.SByte => br.ReadSByte(), _ => 0,
                };
                vals.Add($"{Ident(fname)} = {(v is ulong u ? u + "UL" : v.ToString())}");
            }
            sb.AppendLine($"{ind}{access} enum {simple} : {underlying}");
            sb.AppendLine($"{ind}{{");
            foreach (var v in vals) sb.AppendLine($"{ind}\t{v},");
            sb.AppendLine($"{ind}}}");
            return;
        }
        if (rt.IsDelegate)
        {
            if (!t.FullDecl) return;
            var invoke = td.GetMethods().First(h => _real.R.GetString(_real.R.GetMethodDefinition(h).Name) == "Invoke");
            var (ret, ps, _) = Signature(invoke);
            sb.AppendLine($"{ind}{access} delegate {ret} {simple}{gtext}({string.Join(", ", ps.Select(p => $"{p.Prefix}{p.Type} {p.Name}"))});");
            return;
        }
        string kind = rt.IsInterface ? "interface" : rt.IsValueType ? "struct" : "class";
        string mods = "";
        if (kind == "class")
        {
            if (IsStatic(td)) mods = "static ";
            else if (t.FullDecl && (td.Attributes & TypeAttributes.Sealed) != 0) mods = "sealed ";
            else if (t.FullDecl && (td.Attributes & TypeAttributes.Abstract) != 0) mods = "abstract ";
        }
        string bases = "";
        if (t.FullDecl)
        {
            var list = new List<string>();
            if (kind == "class" && !td.BaseType.IsNil)
            {
                var b = _real.TypeName(td.BaseType);
                if (b != "System.Object") list.Add(td.BaseType.Kind == HandleKind.TypeSpecification
                    ? _real.R.GetTypeSpecification((TypeSpecificationHandle)td.BaseType).DecodeSignature(_cs, ctx)
                    : CsProvider.Cs(b));
            }
            foreach (var ih in td.GetInterfaceImplementations())
            {
                var ifh = _real.R.GetInterfaceImplementation(ih).Interface;
                var iname = _real.TypeName(ifh);
                // System interfaces are left out (their members would all need implementations); Godot ones are kept.
                if (!iname.StartsWith("Godot.")) continue;
                list.Add(ifh.Kind == HandleKind.TypeSpecification
                    ? _real.R.GetTypeSpecification((TypeSpecificationHandle)ifh).DecodeSignature(_cs, ctx)
                    : CsProvider.Cs(iname));
            }
            if (list.Count > 0) bases = " : " + string.Join(", ", list);
        }
        else if (_stub.Types.TryGetValue(t.Name, out var st))
        {
            kind = st.IsInterface ? "interface" : st.IsValueType ? "struct" : "class";
        }
        bool unsafeType = t.Members.Any(m => m.Render(DeclaredVirtualInBase).Contains('*'));
        sb.AppendLine($"{ind}{access} {(unsafeType ? "unsafe " : "")}{mods}partial {kind} {simple}{gtext}{bases}");
        sb.AppendLine($"{ind}{{");
        foreach (var m in t.Members.OrderBy(m => m.Key, StringComparer.Ordinal))
            sb.AppendLine($"{ind}\t{m.Render((ty, k) => DeclaredVirtualInBase(t.Name, k))}");
        foreach (var n in _types.Values.Where(n => n.Name.StartsWith(t.Name + "/") && !n.Name[(t.Name.Length + 1)..].Contains('/')
                                                    && (n.FullDecl || n.Members.Count > 0 || HasEmittedNested(n.Name)))
                     .OrderBy(n => n.Name, StringComparer.Ordinal))
            RenderType(sb, n, depth + 1);
        sb.AppendLine($"{ind}}}");
    }
}

/// <summary>IL scan of sts2.dll (outside the UI namespaces) for calls that leave the game thread or start timers.</summary>
static class ThreadScan
{
    static readonly (string type, string name)[] Watch =
    {
        ("System.Threading.Tasks.Task", "ContinueWith"), ("System.Threading.Tasks.Task`1", "ContinueWith"),
        ("System.Threading.Tasks.Task", "Run"), ("System.Threading.Tasks.TaskFactory", "StartNew"),
        ("System.Threading.ThreadPool", "QueueUserWorkItem"), ("System.Threading.ThreadPool", "UnsafeQueueUserWorkItem"),
        ("System.Threading.Thread", ".ctor"), ("System.Threading.Tasks.Task", "ConfigureAwait"),
        ("System.Threading.Tasks.Task`1", "ConfigureAwait"), ("System.Threading.Tasks.Task", "Delay"),
        ("System.Threading.Tasks.Task", "WaitAsync"), ("System.Threading.Tasks.Task`1", "WaitAsync"),
        ("System.Threading.Timer", ".ctor"), ("System.Timers.Timer", ".ctor"),
    };

    public static int Run(string path)
    {
        using var pe = new PEReader(File.OpenRead(path));
        var r = pe.GetMetadataReader();
        var hits = new SortedSet<string>();
        foreach (var th in r.TypeDefinitions)
        {
            var td = r.GetTypeDefinition(th);
            var top = td;
            while (!top.GetDeclaringType().IsNil) top = r.GetTypeDefinition(top.GetDeclaringType());
            var ns = r.GetString(top.Namespace);
            if (ns.Contains(".Nodes", StringComparison.OrdinalIgnoreCase) || ns.Contains("DevConsole") || ns.Contains("AutoSlay")) continue;
            var owner = Names.Of(r, th);
            foreach (var mh in td.GetMethods())
            {
                var md = r.GetMethodDefinition(mh);
                if (md.RelativeVirtualAddress == 0) continue;
                var il = pe.GetMethodBody(md.RelativeVirtualAddress).GetILReader();
                while (il.RemainingBytes > 0)
                {
                    int op = il.ReadByte();
                    if (op == 0xFE) op = 0xFE00 | il.ReadByte();
                    if (op == 0x45) { int n = il.ReadInt32(); il.Offset += 4 * n; continue; }
                    int size = OperandSize(op);
                    if (op is 0x28 or 0x6F or 0x73)
                    {
                        var tok = MetadataTokens.EntityHandle(il.ReadInt32());
                        var (t, n) = Target(r, tok);
                        if (t != null && Watch.Contains((t, n!))) hits.Add($"{ns}: {owner}.{r.GetString(md.Name)} -> {t}::{n}");
                        continue;
                    }
                    il.Offset += size;
                }
            }
        }
        foreach (var h in hits) Console.WriteLine(h);
        Console.WriteLine($"{hits.Count} call sites");
        return 0;
    }

    static (string?, string?) Target(MetadataReader r, EntityHandle tok)
    {
        if (tok.Kind == HandleKind.MethodSpecification) tok = r.GetMethodSpecification((MethodSpecificationHandle)tok).Method;
        if (tok.Kind != HandleKind.MemberReference) return (null, null);
        var mr = r.GetMemberReference((MemberReferenceHandle)tok);
        string? parent = mr.Parent.Kind switch
        {
            HandleKind.TypeReference => Names.Of(r, (TypeReferenceHandle)mr.Parent),
            HandleKind.TypeSpecification => GenericDef(r, (TypeSpecificationHandle)mr.Parent),
            _ => null,
        };
        return (parent, r.GetString(mr.Name));
    }

    static string? GenericDef(MetadataReader r, TypeSpecificationHandle h)
    {
        var br = r.GetBlobReader(r.GetTypeSpecification(h).Signature);
        if (br.ReadSignatureTypeCode() != SignatureTypeCode.GenericTypeInstance) return null;
        br.ReadSignatureTypeCode();
        var g = br.ReadTypeHandle();
        return g.Kind == HandleKind.TypeReference ? Names.Of(r, (TypeReferenceHandle)g) : null;
    }

    static int OperandSize(int op) => op switch
    {
        0x20 or 0x22 or 0x28 or 0x29 or 0x6F or 0x70 or 0x71 or 0x72 or 0x73 or 0x74 or 0x75 or 0x79 or 0x7B or 0x7C or 0x7D
            or 0x7E or 0x7F or 0x80 or 0x81 or 0x8C or 0x8D or 0x8F or 0xA3 or 0xA4 or 0xA5 or 0xC2 or 0xC6 or 0xD0 or 0x38
            or 0x39 or 0x3A or 0x3B or 0x3C or 0x3D or 0x3E or 0x3F or 0x40 or 0x41 or 0x42 or 0x43 or 0x44 or 0xDD
            or 0xFE06 or 0xFE07 or 0xFE15 or 0xFE16 or 0xFE1C => 4,
        0x21 or 0x23 => 8,
        0x0E or 0x0F or 0x10 or 0x11 or 0x12 or 0x13 or 0x1F or 0x2B or 0x2C or 0x2D or 0x2E or 0x2F or 0x30 or 0x31 or 0x32
            or 0x33 or 0x34 or 0x35 or 0x36 or 0x37 or 0xDE or 0xFE12 => 1,
        0xFE09 or 0xFE0A or 0xFE0B or 0xFE0C or 0xFE0D or 0xFE0E => 2,
        _ => 0,
    };
}
