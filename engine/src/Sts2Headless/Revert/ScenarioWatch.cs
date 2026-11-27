using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.Commands;

namespace Sts2Headless.Revert;

internal static class ScenarioWatch
{
    private static object? _subject;
    private static PropertyInfo? _property;
    private static bool _installed;
    public static object? Value { get; private set; }

    public static void Start(object subject, string member)
    {
        _subject = subject;
        _property = AccessTools.Property(subject.GetType(), member);
        Value = null;
        if (_installed) return;
        var harmony = new Harmony("sts2headless.revert-scenario-watch");
        foreach (var (type, name) in new[]
        {
            (typeof(CardPileCmd), "AddGeneratedCardToCombat"),
            (typeof(CardPileCmd), "Draw"),
            (typeof(CreatureCmd), "GainBlock"),
            (typeof(PowerCmd), "Apply"),
        })
            foreach (var method in AccessTools.GetDeclaredMethods(type).Where(m => m.Name == name && !m.ContainsGenericParameters))
                harmony.Patch(method, prefix: new HarmonyMethod(typeof(ScenarioWatch), nameof(Record)));
        _installed = true;
    }

    private static void Record()
    {
        if (_subject == null || Value != null) return;
        Value = _property!.GetValue(_subject);
    }
}
