using System.Collections;
using System.Reflection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

/// <summary>
/// State census for the observation audit (docs/02_观测与编码方法论.md, tools/state_census.py): every instance field
/// reachable from the run state and the combat state, i.e. the game's complete state at this decision, plus every
/// mutable static field of the game assembly. Each field is classified offline as exported, hidden, cosmetic,
/// derivable or constant. Debug only, never used in training; reading has no side effects (fields only, no getters).
/// </summary>
public partial class RunSimulator
{
    private const int CensusMaxDepth = 16;
    private const int CensusMaxItems = 64;  // elements followed per collection: the field set, not the data, is wanted

    public Dictionary<string, object?> DebugCensus()
    {
        var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
        var rec = new Dictionary<string, Dictionary<string, object?>>();

        void Note(FieldInfo f, object? val, string path)
        {
            var key = f.DeclaringType!.FullName + "::" + f.Name;
            if (!rec.TryGetValue(key, out var r))
            {
                rec[key] = r = new Dictionary<string, object?>
                {
                    ["type"] = f.DeclaringType.FullName, ["field"] = f.Name, ["ftype"] = TypeName(f.FieldType),
                    ["readonly"] = f.IsInitOnly, ["n"] = 0, ["ex"] = Sample(val), ["path"] = path,
                };
            }
            r["n"] = (int)r["n"]! + 1;
        }

        void Visit(object? o, string path, int depth)
        {
            if (o == null || depth > CensusMaxDepth) return;
            var t = o.GetType();
            if (IsLeaf(t) || o is Delegate || o is Task || IsUi(t)) return;
            if (!t.IsValueType && !seen.Add(o)) return;
            if (o is AbstractModel { IsMutable: false }) return;  // canonical content (ModelDb), constant
            if (o is IDictionary dict)
            {
                int i = 0;
                foreach (DictionaryEntry e in dict)
                {
                    if (i++ >= CensusMaxItems) break;
                    Visit(e.Key, path + "{k}", depth + 1);
                    Visit(e.Value, path + "{v}", depth + 1);
                }
                return;
            }
            if (o is IEnumerable seq && !IsGame(t))
            {
                int i = 0;
                foreach (var x in seq)
                {
                    if (i >= CensusMaxItems) break;
                    Visit(x, $"{path}[{i++}]", depth + 1);
                }
                return;
            }
            if (!IsGame(t)) return;  // other framework objects: not game state
            for (var ty = t; ty != null && ty != typeof(object); ty = ty.BaseType)
                foreach (var f in ty.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
                {
                    if (typeof(Delegate).IsAssignableFrom(f.FieldType)) continue;  // event handlers, not state
                    object? val;
                    try { val = f.GetValue(o); } catch { continue; }
                    Note(f, val, path);
                    Visit(val, path + "." + f.Name, depth + 1);
                }
        }

        Visit(RunManager.Instance.DebugOnlyGetState(), "RunState", 0);
        if (CombatManager.Instance.IsInProgress)
        {
            Visit(CombatManager.Instance.DebugOnlyGetState(), "CombatState", 0);
            Visit(CombatManager.Instance, "CombatManager", 0);
        }

        var statics = new List<Dictionary<string, object?>>();
        foreach (var t in typeof(RunManager).Assembly.GetTypes())
        {
            if (!IsGame(t) || IsUi(t) || t.ContainsGenericParameters) continue;
            foreach (var f in t.GetFields(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                if (f.IsLiteral || f.IsInitOnly || f.Name.Contains('<') || typeof(Delegate).IsAssignableFrom(f.FieldType)) continue;
                object? val;
                try { val = f.GetValue(null); } catch { continue; }
                statics.Add(new Dictionary<string, object?>
                {
                    ["type"] = t.FullName, ["field"] = f.Name, ["ftype"] = TypeName(f.FieldType), ["ex"] = Sample(val),
                });
            }
        }
        return new Dictionary<string, object?> { ["type"] = "census", ["fields"] = rec.Values.ToList(), ["statics"] = statics };
    }

    private static bool IsGame(Type t) => t.Namespace?.StartsWith("MegaCrit.Sts2") == true;

    private static bool IsUi(Type t) =>
        t.Namespace?.StartsWith("Godot") == true || t.Namespace?.StartsWith("MegaCrit.Sts2.Core.Nodes") == true
        || t.Namespace?.StartsWith("MegaCrit.Sts2.Core.Animation") == true;

    private static bool IsLeaf(Type t) =>
        t.IsPrimitive || t.IsEnum || t == typeof(string) || t == typeof(decimal) || t == typeof(DateTime)
        || t == typeof(TimeSpan) || t == typeof(Guid) || (Nullable.GetUnderlyingType(t) is { } u && IsLeaf(u));

    private static string TypeName(Type t) =>
        t.IsGenericType ? t.Name.Split('`')[0] + "<" + string.Join(",", t.GetGenericArguments().Select(TypeName)) + ">" : t.Name;

    private static string Sample(object? v)
    {
        var s = v switch
        {
            null => "null",
            AbstractModel m => m.Id.ToString(),
            ICollection c => $"{TypeName(v.GetType())}[{c.Count}]",
            _ when IsLeaf(v.GetType()) => v.ToString() ?? "",
            _ => TypeName(v.GetType()),
        };
        return s.Length > 60 ? s[..60] : s;
    }
}
