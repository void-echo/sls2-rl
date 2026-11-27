using System.Reflection;
using System.Reflection.Emit;
using HarmonyLib;
using MegaCrit.Sts2.Core.Models;

namespace Sts2Headless.Revert;

[AttributeUsage(AttributeTargets.Method)]
internal sealed class LegacyAttribute(string id, string group, string member, Type? staticType = null) : Attribute
{
    public string Id { get; } = id;
    public string Group { get; } = group;
    public string Member { get; } = member;
    public Type? StaticType { get; } = staticType;
}

internal static class PatchRevert
{
    private static readonly Dictionary<MethodBase, List<MethodInfo>> Replacements = new();
    public static IReadOnlyCollection<string> Active { get; private set; } = Array.Empty<string>();

    public static void Apply()
    {
        var value = Environment.GetEnvironmentVariable("SLS2_REVERT");
        if (string.IsNullOrWhiteSpace(value)) return;
        var methods = typeof(PatchRevert).Assembly.GetTypes()
            .Where(t => t.Namespace == typeof(PatchRevert).Namespace)
            .SelectMany(t => t.GetMethods(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic))
            .Select(m => (Method: m, Spec: m.GetCustomAttribute<LegacyAttribute>()))
            .Where(x => x.Spec != null).ToList();
        var selected = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var token in value.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
        {
            var matches = methods.Where(x => token.Equals("all", StringComparison.OrdinalIgnoreCase)
                || token.Equals(x.Spec!.Id, StringComparison.OrdinalIgnoreCase)
                || token.Equals(x.Spec.Group, StringComparison.OrdinalIgnoreCase)).ToList();
            if (matches.Count == 0 && !new[] { "ironclad_cards", "silent_cards", "defect_cards", "regent_cards", "necrobinder_cards", "shared_cards", "relics", "potions", "powers", "monsters", "events", "all" }.Contains(token, StringComparer.OrdinalIgnoreCase)) throw new ArgumentException($"SLS2_REVERT: unsupported model or group '{token}'");
            foreach (var match in matches) selected.Add(match.Spec!.Id);
        }
        foreach (var (method, spec) in methods.Where(x => selected.Contains(x.Spec!.Id)))
        {
            var parameters = method.GetParameters();
            var modelType = spec!.StaticType ?? parameters[0].ParameterType;
            var skip = spec.StaticType == null ? 1 : 0;
            var target = AccessTools.Method(modelType, spec!.Member, parameters.Skip(skip).Select(p => p.ParameterType).ToArray())
                ?? throw new MissingMethodException(modelType.FullName, spec.Member);
            target = AccessTools.DeclaredMethod(target.DeclaringType, target.Name, target.GetParameters().Select(p => p.ParameterType).ToArray());
            if (target.ReturnType != method.ReturnType) throw new InvalidOperationException($"Return type: {method.Name}");
            if (!Replacements.TryGetValue(target, out var entries)) Replacements[target] = entries = new();
            entries.Add(method);
        }
        var harmony = new Harmony("sts2headless.patch-revert.v0111");
        foreach (var target in Replacements.Keys)
            harmony.Patch(target, transpiler: new HarmonyMethod(typeof(PatchRevert), nameof(Replace)));
        Active = selected.Order(StringComparer.Ordinal).ToArray();
        Console.Error.WriteLine("[SLS2_REVERT] " + string.Join(',', Active));
    }

    private static IEnumerable<CodeInstruction> Replace(IEnumerable<CodeInstruction> instructions, MethodBase __originalMethod, ILGenerator generator)
    {
        foreach (var method in Replacements[__originalMethod])
        {
            if (__originalMethod.IsStatic)
            {
                for (var i = 0; i < method.GetParameters().Length; i++)
                    yield return new CodeInstruction(OpCodes.Ldarg, i);
                yield return new CodeInstruction(OpCodes.Call, method);
                yield return new CodeInstruction(OpCodes.Ret);
                yield break;
            }
            var next = generator.DefineLabel();
            var modelType = method.GetParameters()[0].ParameterType;
            yield return new CodeInstruction(OpCodes.Ldarg_0);
            yield return new CodeInstruction(OpCodes.Isinst, modelType);
            yield return new CodeInstruction(OpCodes.Brfalse, next);
            yield return new CodeInstruction(OpCodes.Ldarg_0);
            yield return new CodeInstruction(OpCodes.Castclass, modelType);
            for (var i = 1; i < method.GetParameters().Length; i++)
                yield return new CodeInstruction(OpCodes.Ldarg, i);
            yield return new CodeInstruction(OpCodes.Call, method);
            yield return new CodeInstruction(OpCodes.Ret);
            var nop = new CodeInstruction(OpCodes.Nop);
            nop.labels.Add(next);
            yield return nop;
        }
        foreach (var instruction in instructions) yield return instruction;
    }

    internal static void Flash(AbstractModel model) => AccessTools.Method(model.GetType(), "Flash").Invoke(model, null);
    internal static void Finish(EventModel model, string key) => AccessTools.Method(typeof(EventModel), "SetEventFinished",
        new[] { typeof(MegaCrit.Sts2.Core.Localization.LocString) }).Invoke(model,
        new object[] { new MegaCrit.Sts2.Core.Localization.LocString("events", key) });
    internal static void Shuffle<T>(MegaCrit.Sts2.Core.Random.Rng rng, IList<T> items)
    {
        for (var i = items.Count - 1; i > 0; i--)
        {
            var j = rng.NextInt(i + 1);
            (items[i], items[j]) = (items[j], items[i]);
        }
    }

    internal static T Get<T>(object model, string name) => (T)AccessTools.Property(model.GetType(), name).GetValue(model)!;
    internal static void Set(object model, string name, object value) => AccessTools.Property(model.GetType(), name).SetValue(model, value);
    internal static object Data(PowerModel model) => AccessTools.Field(typeof(PowerModel), "_internalData").GetValue(model)!;
}
