using System.Reflection;
using System.Text;
using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Entities.Ascension;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Odds;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;
using MegaCrit.Sts2.Core.Unlocks;

namespace ParityOracle;

/// <summary>
/// Runs a job: for each run, start it in the client with the v1 headless engine's exact construction, replay the
/// headless action log step by step, and append one digest line per decision point to the run's output JSONL.
/// </summary>
public static class Driver
{
    public static SceneTree? Tree;
    private static ProgressState? _userProgress;

    public static async Task Frames(SceneTree tree, int n)
    {
        for (int i = 0; i < n; i++) await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
    }

    private static async Task<bool> Until(SceneTree tree, Func<bool> cond, double timeoutSec)
    {
        var t0 = DateTime.UtcNow;
        while ((DateTime.UtcNow - t0).TotalSeconds < timeoutSec)
        {
            try { if (cond()) return true; } catch { }
            await Frames(tree, 2);
        }
        return false;
    }

    public static async Task RunJob(SceneTree tree, string jobText)
    {
        Tree = tree;
        using var job = JsonDocument.Parse(jobText);
        var root = job.RootElement;
        string? donePath = root.TryGetProperty("done", out var dp) ? dp.GetString() : null;
        bool quit = !root.TryGetProperty("quit_when_done", out var q) || q.GetBoolean();
        double stepTimeout = root.TryGetProperty("step_timeout", out var st) ? st.GetDouble() : 45;
        // Watch mode (for a person watching a replay): an animation speed other than Instant and a pause before each input.
        var fastMode = FastModeType.Instant;
        _stepDelay = 0;
        if (root.TryGetProperty("watch", out var watch))
        {
            if (watch.TryGetProperty("fast_mode", out var fm) && Enum.TryParse<FastModeType>(fm.GetString(), out var parsed)) fastMode = parsed;
            if (watch.TryGetProperty("step_delay", out var sd)) _stepDelay = sd.GetDouble();
        }
        var summary = new List<Dictionary<string, object?>>();
        FastModeType? oldFast = null;
        try
        {
            if (_stepDelay > 0) Watch.Start(tree);
            Entry.Log("waiting for main menu");
            if (!await Until(tree, () => NGame.Instance?.MainMenu != null, 180)) throw new TimeoutException("main menu never appeared");
            await Frames(tree, 60);
            oldFast = SaveManager.Instance.PrefsSave.FastMode;
            SaveManager.Instance.PrefsSave.FastMode = fastMode;
            _userProgress = SaveManager.Instance.Progress;
            Entry.Log($"main menu up; FastMode {oldFast} -> {fastMode}, step delay {_stepDelay}s");

            foreach (var run in root.GetProperty("runs").EnumerateArray())
            {
                var res = new Dictionary<string, object?> { ["id"] = run.GetProperty("id").GetString() };
                try { await RunOne(tree, run, stepTimeout, res); }
                catch (Exception ex)
                {
                    Entry.Log($"run {res["id"]} failed: {ex}");
                    res["error"] = ex.GetType().Name + ": " + ex.Message;
                }
                summary.Add(res);
                if (Watch.Enabled) Watch.EndRun();
                await BackToMenu(tree);
            }
        }
        catch (Exception ex)
        {
            Entry.Log("job failed: " + ex);
            summary.Add(new() { ["job_error"] = ex.ToString() });
        }
        finally
        {
            if (Watch.Enabled) Watch.Stop();
            RestoreUserProgress();
            if (oldFast.HasValue) SaveManager.Instance.PrefsSave.FastMode = oldFast.Value;
            if (donePath != null) File.WriteAllText(donePath, Json.Write(new Dictionary<string, object?> { ["runs"] = summary }));
            Entry.Log("job done");
            try { File.Delete(Path.Combine(Entry.ControlDir, "job.running.json")); } catch { }
        }
        if (quit)
        {
            await Frames(tree, 10);
            tree.Quit();
        }
    }

    private static void RestoreUserProgress()
    {
        if (_userProgress != null && !ReferenceEquals(SaveManager.Instance.Progress, _userProgress))
            SaveManager.Instance.Progress = _userProgress;
    }

    private static async Task BackToMenu(SceneTree tree)
    {
        try
        {
            Client.UninstallSelector();
            RestoreUserProgress();
            if (RunManager.Instance.DebugOnlyGetState() != null || NGame.Instance.MainMenu == null)
            {
                Entry.Log("returning to main menu");
                await NGame.Instance.ReturnToMainMenu();
                await Until(tree, () => NGame.Instance.MainMenu != null, 60);
                await Frames(tree, 30);
            }
        }
        catch (Exception ex) { Entry.Log("BackToMenu: " + ex); }
    }

    private static readonly PropertyInfo? _uniqueId = typeof(ProgressState).GetProperty("UniqueId");

    /// <summary>
    /// Build the run exactly like the v1 engine's Harness.StartRun: Rng.Chaotic and the profile made a function of the
    /// seed (fresh ProgressState, UniqueId "SLS2RL1"), Player.CreateForNewRun with UnlockState.all, acts from
    /// ActModel.GetRandomList on the "act_selection" RNG (TestMode semantics: plain random choice, see below),
    /// RunState.CreateForNewRun, SetUpNewSingleplayer, CardRarityOdds.regularCommonOdds reset to this run's ascension;
    /// then the client's own NGame.StartRun (asset preload, FinalizeStartingRelics, Launch, NRun scene, EnterAct(0)).
    /// </summary>
    private static async Task StartRun(SceneTree tree, string character, int ascension, string seed, Dictionary<string, object?> res)
    {
        var ch = ModelDb.AllCharacters.FirstOrDefault(c => string.Equals(c.Id.Entry, character, StringComparison.OrdinalIgnoreCase))
                 ?? throw new ArgumentException("unknown character " + character);
        ulong h = StringHelper.GetDeterministicHashCode(seed);
        Rng.Chaotic.LoadFromSerializable(new Rng(h, "headless_chaotic").ToSerializable());

        var progress = ProgressState.CreateDefault();
        _uniqueId?.SetValue(progress, "SLS2RL1");
        progress.EnableFtues = false;   // UI only; the engine never shows FTUEs
        SaveManager.Instance.Progress = progress;

        var unlock = UnlockState.all;
        var player = Player.CreateForNewRun(ch, unlock, 1uL);
        var acts = RandomActsTestModeSemantics(new Rng(h, "act_selection"), unlock);
        try
        {
            // What the shipping GetRandomList (TestMode off) would pick with this fresh profile: it forces every
            // not-yet-discovered act variant. Recorded for the report only.
            res["acts_shipping_fresh_profile"] = ActModel.GetRandomList(new Rng(h, "act_selection"), unlock, isMultiplayer: false)
                .Select(a => a.Id.ToString()).ToList();
        }
        catch (Exception ex) { res["acts_shipping_fresh_profile"] = "error: " + ex.Message; }

        var state = RunState.CreateForNewRun(new[] { player }, acts.Select(a => a.ToMutable()).ToList(),
            Array.Empty<ModifierModel>(), GameMode.Standard, false, ascension, seed);
        RunManager.Instance.SetUpNewSingleplayer(state, shouldSave: false);
        var odds = typeof(CardRarityOdds).GetField("regularCommonOdds", BindingFlags.Public | BindingFlags.Static)!;
        res["common_odds_before"] = (double)(float)odds.GetValue(null)!;
        odds.SetValue(null, AscensionHelper.GetValueIfAscension(AscensionLevel.Scarcity, 0.615f, 0.6f));
        res["acts"] = acts.Select(a => a.Id.ToString()).ToList();
        res["started_with_neow"] = state.ExtraFields.StartedWithNeow;
        Client.InstallSelector();
        Obs.ResetRun();   // map cache, cards played this turn, known draw-pile top (engine: CleanUpRun / ResetRlState)
        var startRun = typeof(NGame).GetMethod("StartRun", BindingFlags.Instance | BindingFlags.NonPublic)
                       ?? throw new MissingMethodException("NGame.StartRun");
        Entry.Log($"starting run {character} A{ascension} seed={seed} acts={string.Join(",", (List<string>)res["acts"]!)}");
        // The main menu stops its music before a run is entered (NMainMenu.OnContinueButtonPressedAsync, character select);
        // starting the run directly skipped that, so the menu track kept playing over the run. The run's own music is
        // started by NRun._Ready -> NRunMusicController.UpdateMusic. Audio only, no effect on game state.
        MegaCrit.Sts2.Core.Nodes.Audio.NAudioManager.Instance?.StopMusic();
        await (Task)startRun.Invoke(NGame.Instance, new object[] { state })!;
        Entry.Log("run started");
    }

    /// <summary>
    /// ActModel.GetRandomList with TestMode on, which is what the engine runs (TestMode stays on there): for each act
    /// index one uniformly random unlocked variant. Shipping (TestMode off) additionally forces any non-default variant
    /// missing from Progress.DiscoveredActs; with the engine's fresh profile that would always be the second variant.
    /// </summary>
    private static List<ActModel> RandomActsTestModeSemantics(Rng rng, UnlockState unlock)
    {
        var list = new List<ActModel>();
        var byIndex = ModelDb.ActsByIndex;
        for (int i = 0; i < byIndex.Count; i++)
        {
            var options = byIndex[i].Where(a => a.IsUnlocked(unlock)).ToList();
            list.Add(rng.NextItem(options) ?? throw new InvalidOperationException($"No unlocked acts for index {i}!"));
        }
        return list;
    }

    private static double _stepDelay;

    private static async Task RunOne(SceneTree tree, JsonElement run, double stepTimeout, Dictionary<string, object?> res)
    {
        var character = run.GetProperty("character").GetString()!;
        int asc = run.TryGetProperty("ascension", out var a) ? a.GetInt32() : 0;
        var seed = run.GetProperty("seed").GetString()!;
        var outPath = run.GetProperty("out").GetString()!;
        int maxSteps = run.TryGetProperty("max_steps", out var ms) ? ms.GetInt32() : int.MaxValue;
        var actions = new List<JsonElement>();
        var docs = new List<JsonDocument>();
        foreach (var line in File.ReadAllLines(run.GetProperty("actions").GetString()!))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var d = JsonDocument.Parse(line);
            docs.Add(d);
            actions.Add(d.RootElement);
        }
        Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);
        using var w = new StreamWriter(outPath, append: false, new UTF8Encoding(false)) { AutoFlush = true };
        var sw = System.Diagnostics.Stopwatch.StartNew();
        if (Watch.Enabled) Watch.BeginRun(Math.Min(maxSteps, actions.Count));
        await StartRun(tree, character, asc, seed, res);
        if (run.TryGetProperty("setup", out var setup) && setup.ValueKind == JsonValueKind.Array && setup.GetArrayLength() > 0)
        {
            // The engine applies scenario commands to the state rl_start returned: reach that decision first.
            var first = await Client.WaitForDecision(tree, stepTimeout);
            Entry.Log($"scenario: first decision {first.Decision}, applying {setup.GetArrayLength()} setup command(s)");
            foreach (var cmd in setup.EnumerateArray())
            {
                // Wait for the command's game task like the engine does before answering (a room switch takes frames).
                var t = Scenario.Apply(cmd);
                var t0 = DateTime.UtcNow;
                while (!t.IsCompleted && (DateTime.UtcNow - t0).TotalSeconds < stepTimeout) await Frames(tree, 2);
                if (!t.IsCompleted) throw new TimeoutException("scenario setup command did not complete: " + cmd.GetRawText());
                await Frames(tree, 4);
            }
            res["setup"] = setup.GetArrayLength();
            if (Watch.Enabled) Watch.ClearSnapshots();
        }
        w.WriteLine(Json.Write(new Dictionary<string, object?> { ["meta"] = res }));
        int step = 0;
        string end = "actions_exhausted";
        for (; step < Math.Min(maxSteps, actions.Count + 1); step++)
        {
            try
            {
                if (Watch.Enabled)
                {
                    Watch.AtStep(step);
                    await Watch.WaitForPlayback(tree);
                }
                var t0 = sw.Elapsed.TotalMilliseconds;
                if (_stepDelay > 0)  // watch mode: let animations and screen transitions finish before reading the screen
                {
                    var until = DateTime.UtcNow.AddSeconds(_stepDelay);
                    while (DateTime.UtcNow < until)
                    {
                        if (Watch.Enabled) await Watch.WaitForPlayback(tree);
                        await Frames(tree, 1);
                    }
                }
                var dpt = await Client.WaitForDecision(tree, stepTimeout);
                var line = new Dictionary<string, object?>
                {
                    ["step"] = step,
                    ["decision"] = dpt.Decision,
                    ["obs"] = dpt.Obs,
                    ["cands"] = dpt.Cands.Select(c => c.Desc).ToList(),
                    ["t_wait_ms"] = sw.Elapsed.TotalMilliseconds - t0,
                };
                if (dpt.Note != null) line["note"] = dpt.Note;
                if (dpt.Decision is "game_over" or "stuck")
                {
                    w.WriteLine(Json.Write(line));
                    end = dpt.Decision;
                    break;
                }
                if (step >= actions.Count) { w.WriteLine(Json.Write(line)); break; }
                var act = actions[step];
                var want = act.GetProperty("cand");
                var wantCanon = Json.Canon(want);
                int idx = dpt.Cands.FindIndex(c => Json.Canon(c.Desc) == wantCanon);
                // Watch mode only: with animations a screen can be read mid-transition (the rewards screen still closing,
                // the chest's Skip appearing after its timer); read it again until the recorded input is there.
                for (var until = DateTime.UtcNow.AddSeconds(8); idx < 0 && _stepDelay > 0 && DateTime.UtcNow < until;)
                {
                    await Frames(tree, 15);
                    dpt = await Client.WaitForDecision(tree, stepTimeout);
                    idx = dpt.Cands.FindIndex(c => Json.Canon(c.Desc) == wantCanon);
                }
                string match = "exact";
                if (idx < 0)
                {
                    // Fall back to the same position among candidates of the same kind, so a digest-only mismatch inside
                    // a descriptor (e.g. a card var) does not stop the replay; the comparator still reports it.
                    var k = want.GetProperty("k").GetString();
                    int kindPos = act.TryGetProperty("kind_pos", out var kp) ? kp.GetInt32() : -1;
                    var sameKind = dpt.Cands.Select((c, j) => (c, j)).Where(t => (string?)t.c.Desc["k"] == k).ToList();
                    if (kindPos >= 0 && kindPos < sameKind.Count) { idx = sameKind[kindPos].j; match = "fallback_kind_pos"; }
                    else match = "none";
                }
                line["action"] = new Dictionary<string, object?> { ["want"] = want, ["idx"] = idx, ["match"] = match };
                if (Watch.Enabled) await Watch.WaitForPlayback(tree);
                w.WriteLine(Json.Write(line));
                if (idx < 0)
                {
                    end = "no_matching_candidate";
                    break;
                }
                Entry.Log($"step {step} {dpt.Decision}: {Json.Canon(dpt.Cands[idx].Desc)} ({match})");
                Nav.BeforeInput(dpt.Screen, dpt.Cands[idx].Desc.GetValueOrDefault("k") as string);
                if (Watch.Enabled) Watch.BeforeInput(step);
                try { await dpt.Cands[idx].Exec(); }
                catch (Exception ex)
                {
                    Entry.Log($"exec failed at step {step}: {ex}");
                    w.WriteLine(Json.Write(new Dictionary<string, object?> { ["exec_error"] = ex.Message, ["step"] = step }));
                    end = "exec_error";
                    break;
                }
                await Frames(tree, 2);
            }
            catch (Watch.RewindRequested)
            {
                var restoredStep = await Watch.Rewind(tree);
                if (restoredStep is { } rewindStep)
                {
                    w.WriteLine(Json.Write(new Dictionary<string, object?> { ["rewind"] = rewindStep }));
                    step = rewindStep - 1;
                }
                else step--;
            }
        }
        res["steps"] = step;
        res["end"] = end;
        res["seconds"] = sw.Elapsed.TotalSeconds;
        w.WriteLine(Json.Write(new Dictionary<string, object?> { ["end"] = end, ["steps"] = step, ["seconds"] = sw.Elapsed.TotalSeconds }));
        foreach (var d in docs) d.Dispose();
        Entry.Log($"run {res["id"]} ended: {end} after {step} steps in {sw.Elapsed.TotalSeconds:F1}s");
    }
}
