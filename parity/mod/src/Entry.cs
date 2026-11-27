using System.Reflection;
using System.Text;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace ParityOracle;

/// <summary>
/// Mod entry point. The mod is inert unless the control directory named in ParityOracle.config.json (next to the
/// DLL) contains a job.json. Only then does it patch saving off, install the card-selection capture, switch to
/// Instant mode and start the replay driver.
/// </summary>
[ModInitializer(nameof(Init))]
public static class Entry
{
    public static string ControlDir = "";
    private static StreamWriter? _log;
    private static readonly object _logLock = new();

    public static void Init()
    {
        try
        {
            var modDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? "";
            var cfgPath = Path.Combine(modDir, "ParityOracle.config.json");
            if (!File.Exists(cfgPath))
            {
                GD.Print("[ParityOracle] no config, staying inert");
                return;
            }
            using var cfg = System.Text.Json.JsonDocument.Parse(File.ReadAllText(cfgPath));
            ControlDir = cfg.RootElement.GetProperty("control_dir").GetString() ?? "";
            var jobPath = Path.Combine(ControlDir, "job.json");
            if (!File.Exists(jobPath))
            {
                GD.Print($"[ParityOracle] no job at {jobPath}, staying inert");
                return;
            }
            _log = new StreamWriter(Path.Combine(ControlDir, "oracle.log"), append: true, new UTF8Encoding(false)) { AutoFlush = true };
            Log($"=== ParityOracle init, job {jobPath}");
            var job = File.ReadAllText(jobPath);
            // Consume the job so a crash or a normal relaunch does not replay it again.
            File.Move(jobPath, Path.Combine(ControlDir, "job.running.json"), overwrite: true);
            ApplyPatches();
            var tree = (SceneTree)Engine.GetMainLoop();
            _ = Driver.RunJob(tree, job);
        }
        catch (Exception ex)
        {
            GD.PrintErr($"[ParityOracle] init failed: {ex}");
            Log($"init failed: {ex}");
        }
    }

    public static void Log(string msg)
    {
        var line = $"{DateTime.Now:HH:mm:ss.fff} {msg}";
        GD.Print("[ParityOracle] " + msg);
        lock (_logLock)
        {
            try { _log?.WriteLine(line); } catch { }
        }
    }

    // ---- Save suppression: oracle runs must never touch the player's profile, run saves, history or replays. ----
    // (The run save keeps its timing: SaveRunLater.)

    private static void ApplyPatches()
    {
        var h = new Harmony("sls2rl.parityoracle");
        var skipVoid = new HarmonyMethod(typeof(Entry).GetMethod(nameof(SkipVoid), BindingFlags.Static | BindingFlags.NonPublic));
        var skipTask = new HarmonyMethod(typeof(Entry).GetMethod(nameof(SkipTask), BindingFlags.Static | BindingFlags.NonPublic));
        var saveRunLater = new HarmonyMethod(typeof(Entry).GetMethod(nameof(SaveRunLater), BindingFlags.Static | BindingFlags.NonPublic));
        var sm = typeof(SaveManager);
        int n = 0;
        foreach (var m in sm.GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.DeclaredOnly))
        {
            bool target = m.Name is "SaveRun" or "SaveProgressFile" or "SavePrefsFile" or "SaveSettings" or "SaveProfile"
                or "SaveRunHistory" or "DeleteCurrentRun" or "DeleteCurrentMultiplayerRun" or "IncrementNumReloads";
            if (!target) continue;
            if (m.Name == "SaveRun" && m.ReturnType == typeof(Task)) h.Patch(m, prefix: saveRunLater);
            else if (m.ReturnType == typeof(void)) h.Patch(m, prefix: skipVoid);
            else if (m.ReturnType == typeof(Task)) h.Patch(m, prefix: skipTask);
            else continue;
            n++;
        }
        // Lowest level too: every file write/delete/rename of the save stores (local Godot files and Steam cloud).
        // Progress writes such as SetFtuesEnabled bypass SaveManager.Save*; this catches them.
        foreach (var t in new[] { typeof(GodotFileIo), typeof(CloudSaveStore), typeof(MegaCrit.Sts2.Core.Platform.Steam.SteamRemoteSaveStore) })
            foreach (var m in t.GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.DeclaredOnly))
            {
                if (!(m.Name.StartsWith("Write") || m.Name.StartsWith("Delete") || m.Name.StartsWith("Rename") || m.Name is "CreateDirectory" or "ForgetFile")) continue;
                if (m.ReturnType == typeof(void)) h.Patch(m, prefix: skipVoid);
                else if (m.ReturnType == typeof(Task)) h.Patch(m, prefix: skipTask);
                else continue;
                n++;
            }
        var wr = typeof(RunManager).GetMethod("WriteReplay", BindingFlags.Instance | BindingFlags.Public);
        if (wr != null) { h.Patch(wr, prefix: skipVoid); n++; }
        Log($"save suppression patches applied: {n}");
        SelectionCapture.Apply(h);
        // The known draw-pile top (engine/src/Shared/KnownTop.cs): record cards put on top / shuffles, as the engine does.
        int kt = 0;
        foreach (var (target, prefix) in Sts2Obs.KnownTop.PatchTargets()) { h.Patch(target, prefix: new HarmonyMethod(prefix)); kt++; }
        Log($"known draw-pile top patches applied: {kt}");
        // PunchOff.PunchEachOther is an ambient animation loop whose only yields are Cmd.Wait calls; with
        // FastMode=Instant those complete synchronously, so the loop never yields and freezes the main thread while
        // flooding godot.log (seen 2026-10-28: 8 GB in ~2 minutes). It only animates the two layout creatures and
        // spawns hit-spark VFX (no game state), so it is skipped while a job runs.
        var punch = typeof(MegaCrit.Sts2.Core.Models.Events.PunchOff).GetMethod("PunchEachOther", BindingFlags.Instance | BindingFlags.NonPublic);
        if (punch != null) { h.Patch(punch, prefix: skipTask); Log("PunchOff.PunchEachOther skipped (Instant-mode busy loop)"); }
        else Log("WARNING: PunchOff.PunchEachOther not found");
    }

    private static bool SkipVoid() => false;

    private static bool SkipTask(ref Task __result)
    {
        __result = Task.CompletedTask;
        return false;
    }

    /// <summary>
    /// SaveManager.SaveRun writes nothing, but completes like a real save: a real write goes through
    /// GodotFileIo.WriteFileAsync (FileAccessStream.WriteAsync runs on the thread pool), so every awaited run save yields
    /// and its caller resumes on the main thread in a later frame. Completing synchronously would change game behaviour:
    /// CombatManager.EndCombatInternal awaits the save before SetCombatState(NotInCombat), and a combat that ends inside
    /// the inline continuation of a hook action's completion would cancel that action before ActionQueueSet.PopAction
    /// ("Tried to pop action GenericHookGameAction ... didn't find it in any queue"; the engine's Patches.PatchSaving
    /// keeps the same timing headless). The oracle starts runs with shouldSave: false, so RunSaveManager.SaveRun would
    /// return at once anyway; this stub is what restores the yield.
    /// </summary>
    private static bool SaveRunLater(AbstractRoom? preFinishedRoom, ref Task __result)
    {
        if (Watch.Enabled) Watch.Capture(preFinishedRoom);
        __result = NextFrame();
        return false;
    }

    private static async Task NextFrame()
    {
        var tree = (SceneTree)Engine.GetMainLoop();
        await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
    }
}
