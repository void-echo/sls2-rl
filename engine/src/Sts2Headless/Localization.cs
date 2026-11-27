using HarmonyLib;
using MegaCrit.Sts2.Core.Localization;

namespace Sts2Headless;

/// <summary>
/// LocManager without Godot. LocManager.Initialize needs the platform layer, so an uninitialized instance
/// gets the English tables from localization_eng/ and the game's own SmartFormat setup. Text only feeds
/// descriptions and logs; the RL observation uses model ids. The patches make lookups total (missing keys
/// return the key) so no gameplay path can throw on text.
/// </summary>
internal static class Localization
{
    public static void Init()
    {
        var instanceProp = typeof(LocManager).GetProperty("Instance",
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public)!;
        var instance = System.Runtime.CompilerServices.RuntimeHelpers.GetUninitializedObject(typeof(LocManager));
        instanceProp.SetValue(null, instance);

        var tables = new Dictionary<string, LocTable>();
        var locDir = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "localization_eng");
        if (Directory.Exists(locDir))
        {
            foreach (var file in Directory.GetFiles(locDir, "*.json"))
            {
                try
                {
                    var data = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(file));
                    if (data != null) tables[Path.GetFileNameWithoutExtension(file)] = new LocTable(Path.GetFileNameWithoutExtension(file), data);
                }
                catch (Exception e) { Console.Error.WriteLine($"[WARN] loc table {file}: {e.Message}"); }
            }
        }
        else Console.Error.WriteLine($"[WARN] Localization dir not found: {locDir}");

        var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        typeof(LocManager).GetField("_tables", flags)!.SetValue(instance, tables);
        typeof(LocManager).GetField("_engTables", flags)?.SetValue(instance, tables);
        try { typeof(LocManager).GetProperty("Language")?.SetValue(instance, "eng"); } catch { }
        try { typeof(LocManager).GetProperty("CultureInfo")?.SetValue(instance, System.Globalization.CultureInfo.InvariantCulture); } catch { }
        try
        {
            typeof(LocManager).GetMethod("LoadLocFormatters", flags)?.Invoke(instance, null);
        }
        catch (Exception e) { Console.Error.WriteLine($"[WARN] LoadLocFormatters: {e.InnerException?.Message ?? e.Message}"); }

        var h = new Harmony("sts2headless.loc");
        var pre = typeof(Localization);
        h.Patch(typeof(LocTable).GetMethod("GetRawText", new[] { typeof(string) }), new HarmonyMethod(pre.GetMethod(nameof(GetRawTextPrefix))));
        h.Patch(typeof(LocTable).GetMethod("GetLocString"), new HarmonyMethod(pre.GetMethod(nameof(GetLocStringPrefix))));
        h.Patch(typeof(LocTable).GetMethod("HasEntry"), new HarmonyMethod(pre.GetMethod(nameof(TruePrefix))));
        h.Patch(typeof(LocTable).GetMethod("IsLocalKey"), new HarmonyMethod(pre.GetMethod(nameof(TruePrefix))));
        h.Patch(typeof(LocString).GetMethod("Exists", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public),
            new HarmonyMethod(pre.GetMethod(nameof(TruePrefix))));
        h.Patch(typeof(LocTable).GetMethod("GetLocStringsWithPrefix"), new HarmonyMethod(pre.GetMethod(nameof(EmptyListPrefix))));
    }

    public static bool GetRawTextPrefix(string key, ref string __result)
    {
        __result = key;
        return false;
    }

    private static readonly System.Reflection.FieldInfo? _tableName =
        typeof(LocTable).GetField("_name", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);

    public static bool GetLocStringPrefix(LocTable __instance, string key, ref LocString __result)
    {
        __result = new LocString(_tableName?.GetValue(__instance) as string ?? "_unknown", key);
        return false;
    }

    public static bool TruePrefix(ref bool __result)
    {
        __result = true;
        return false;
    }

    public static bool EmptyListPrefix(ref IReadOnlyList<LocString> __result)
    {
        __result = new List<LocString>();
        return false;
    }
}
