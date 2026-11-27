using System.Runtime.Loader;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Sts2Headless;

/// <summary>
/// JSON-lines front end (engine/RL_PROTOCOL.md). One command per stdin line, one JSON object per stdout line.
/// Training commands: rl_start, rl_act, rl_save_get, rl_load, quit. Scenario/debug commands (tests only): rl_enter_room (alias
/// enter_room), set_player, obtain_relics, dbg_obtain_relic, dbg_procure_potion, dbg_add_card, dbg_enter_act,
/// dbg_obs, dbg_info.
/// </summary>
internal static class Program
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = false,
    };

    /// <summary>Directory containing sts2.dll: STS2_LIB, else the nearest lib/ walking up from the binary.</summary>
    private static string ResolveLibDirectory()
    {
        var envLib = Environment.GetEnvironmentVariable("STS2_LIB");
        if (!string.IsNullOrWhiteSpace(envLib))
        {
            var p = Path.GetFullPath(envLib.Trim());
            if (File.Exists(Path.Combine(p, "sts2.dll"))) return p;
        }
        var dir = AppContext.BaseDirectory;
        for (var depth = 0; depth < 16 && !string.IsNullOrEmpty(dir); depth++)
        {
            var candidate = Path.Combine(dir, "lib");
            if (File.Exists(Path.Combine(candidate, "sts2.dll"))) return Path.GetFullPath(candidate);
            dir = Directory.GetParent(dir)?.FullName ?? "";
        }
        return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "lib"));
    }

    private static void Main()
    {
        var libDir = ResolveLibDirectory();
        AssemblyLoadContext.Default.Resolving += (ctx, name) =>
        {
            var path = Path.Combine(libDir, name.Name + ".dll");
            return File.Exists(path) ? ctx.LoadFromAssemblyPath(Path.GetFullPath(path)) : null;
        };
        AppDomain.CurrentDomain.UnhandledException += (_, e) => Console.Error.WriteLine($"[FATAL] Unhandled: {e.ExceptionObject}");
        // A task that faulted without anyone observing it is a game exception the run did not survive: report it
        // on the current run instead of swallowing it.
        TaskScheduler.UnobservedTaskException += (_, e) =>
        {
            Console.Error.WriteLine($"[ERROR] Unobserved task exception: {e.Exception}");
            RunSimulator.ReportBackgroundFault(e.Exception);
            e.SetObserved();
        };
        Run();
    }

    // Separate method so that sts2.dll is resolved only after the Resolving handler is installed.
    private static void Run()
    {
        try { RunSimulator.Init(); GameLoop.Instance.Run(() => { }); }
        catch (Exception e)
        {
            WriteLine(new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"init failed: {e}" });
            return;
        }
        var sim = new RunSimulator();
        WriteLine(new Dictionary<string, object?> { ["type"] = "ready", ["version"] = "1.0.0", ["protocol"] = 1 });

        string? line;
        while ((line = Console.ReadLine()) != null)
        {
            line = line.Trim();
            if (line.Length == 0) continue;
            Dictionary<string, object?> result;
            bool quit = false;
            try
            {
                var cmd = JsonSerializer.Deserialize<JsonElement>(line);
                var name = cmd.TryGetProperty("cmd", out var c) ? c.GetString() ?? "" : "";
                quit = name == "quit";
                Dictionary<string, object?>? r = null;
                GameLoop.Instance.Run(() => r = Handle(sim, name, cmd));
                result = r!;
            }
            catch (JsonException ex)
            {
                result = new() { ["type"] = "error", ["message"] = $"Invalid JSON: {ex.Message}" };
            }
            catch (Exception ex)
            {
                result = new() { ["type"] = "error", ["message"] = $"{ex.GetType().Name}: {ex.Message}", ["stack_trace"] = ex.ToString() };
            }
            WriteLine(result);
            if (quit) break;
        }
    }

    private static Dictionary<string, object?> Handle(RunSimulator sim, string name, JsonElement cmd)
    {
        switch (name)
        {
            case "rl_start":
                return sim.RlStart(
                    cmd.TryGetProperty("character", out var ch) ? ch.GetString() ?? "Ironclad" : "Ironclad",
                    cmd.TryGetProperty("ascension", out var asc) && asc.ValueKind == JsonValueKind.Number ? asc.GetInt32() : 0,
                    cmd.TryGetProperty("seed", out var s) ? s.ValueKind == JsonValueKind.String ? s.GetString() : s.ToString() : null);
            case "rl_act":
                return sim.RlAct(cmd.GetProperty("i").GetInt32());
            case "rl_save_get":
                return sim.RlSaveGet(cmd.GetProperty("id").GetInt32());
            case "rl_load":
                return sim.RlLoad(cmd.TryGetProperty("id", out var saveId) ? saveId.GetInt32() : null,
                    cmd.TryGetProperty("data", out var data) ? data.GetString() : null);
            case "rl_enter_room":
            case "enter_room":
                return sim.EnterRoom(cmd);
            case "set_player":
                return sim.SetPlayer(cmd);
            case "obtain_relics":
            case "dbg_obtain_relic":
                return sim.ObtainRelics(cmd);
            case "dbg_procure_potion":
                return sim.ProcurePotion(cmd);
            case "dbg_add_card":
                return sim.AddCard(cmd);
            case "dbg_enter_act":
                return sim.EnterActDebug(cmd);
            case "dbg_revert":
                return sim.DebugRevert(cmd);
            case "dbg_revert_state":
                return sim.DebugRevertState();
            case "dbg_obs":
                return sim.RlObserve();
            case "dbg_info":
                return sim.DebugInfo();
            case "dbg_census":
                return sim.DebugCensus();
            case "quit":
                sim.CleanUpRun();
                return new() { ["type"] = "quit_result", ["success"] = true };
            default:
                return new() { ["type"] = "error", ["message"] = $"Unknown command: {name}" };
        }
    }

    private static void WriteLine(Dictionary<string, object?> data)
    {
        Console.Out.WriteLine(JsonSerializer.Serialize(data, JsonOpts));
        Console.Out.Flush();
    }
}
