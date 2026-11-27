using System.Reflection;
using System.Text.Json;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Odds;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Validation;

namespace ParityOracle;

public static class Watch
{
    public sealed class RewindRequested : Exception { }

    private sealed record Snapshot(int Step, string Location, string Run, string Progress, string Chaotic,
        float CommonOdds, CharacterModel? CookieCharacter);

    public static bool Enabled { get; private set; }
    private static bool _active, _paused, _loading, _ready;
    private static int _step, _total, _nextStep, _rewind;
    private static double _pausedSeconds;
    private static DateTime _pausedSince, _lastStatus;
    private static SceneTree? _tree;
    private static CanvasLayer? _layer;
    private static Label? _label;
    private static string _notice = "";
    private static readonly List<Snapshot> _snapshots = new();
    private static readonly FieldInfo CookieCharacter = typeof(YummyCookie)
        .GetField("_cachedRandomCharacter", BindingFlags.Static | BindingFlags.NonPublic)!;
    private static string StatusPath => Path.Combine(Entry.ControlDir, "paused.json");

    public static void Start(SceneTree tree)
    {
        Enabled = true;
        _tree = tree;
        _pausedSeconds = 0;
        _paused = false;
        _lastStatus = DateTime.MinValue;
        var h = new Harmony("sls2rl.parityoracle.watch");
        h.Patch(typeof(NGame).GetMethod(nameof(NGame._Input)),
            prefix: new HarmonyMethod(typeof(Watch).GetMethod(nameof(Input), BindingFlags.Static | BindingFlags.NonPublic)));
        _layer = new CanvasLayer { Layer = 100 };
        _label = new Label { Position = new Vector2(12, 8), MouseFilter = Control.MouseFilterEnum.Ignore };
        _label.AddThemeFontSizeOverride("font_size", 16);
        _label.AddThemeColorOverride("font_color", Colors.White);
        _label.AddThemeColorOverride("font_shadow_color", Colors.Black);
        _label.AddThemeConstantOverride("shadow_offset_x", 1);
        _label.AddThemeConstantOverride("shadow_offset_y", 1);
        _layer.AddChild(_label);
        tree.Root.AddChild(_layer);
        tree.ProcessFrame += Tick;
        WriteStatus();
    }

    public static void BeginRun(int total)
    {
        _active = true;
        _ready = false;
        _step = _nextStep = 0;
        _total = total;
        ClearSnapshots();
    }

    public static void ClearSnapshots()
    {
        _snapshots.Clear();
        _rewind = 0;
        _notice = "";
    }

    public static void EndRun()
    {
        _active = false;
        _ready = false;
        SetPaused(false);
        ClearSnapshots();
    }

    public static void Stop()
    {
        EndRun();
        Enabled = false;
        if (_tree != null) _tree.ProcessFrame -= Tick;
        _layer?.QueueFree();
        _layer = null;
        _label = null;
        File.Delete(StatusPath);
    }

    private static bool Input(NGame __instance, InputEvent inputEvent)
    {
        if (!Enabled || !_active || inputEvent is not InputEventKey key) return true;
        var code = key.PhysicalKeycode == Key.None ? key.Keycode : key.PhysicalKeycode;
        if (code is not (Key.Space or Key.R or Key.B)) return true;
        __instance.GetViewport().SetInputAsHandled();
        if (key.Pressed && !key.Echo)
        {
            if (code == Key.Space) SetPaused(!_paused);
            else if (!_loading && _ready) _rewind = code == Key.R ? 1 : 2;
            else _notice = "Rewind unavailable while loading";
        }
        return false;
    }

    private static void SetPaused(bool paused)
    {
        if (_paused == paused) return;
        var now = DateTime.UtcNow;
        if (paused) _pausedSince = now;
        else _pausedSeconds += (now - _pausedSince).TotalSeconds;
        _paused = paused;
        WriteStatus();
    }

    private static void WriteStatus()
    {
        _lastStatus = DateTime.UtcNow;
        var status = new Dictionary<string, object?>
        {
            ["paused"] = _paused,
            ["updated_at"] = new DateTimeOffset(_lastStatus).ToUnixTimeMilliseconds() / 1000.0,
            ["paused_since"] = _paused ? new DateTimeOffset(_pausedSince).ToUnixTimeMilliseconds() / 1000.0 : null,
            ["paused_seconds"] = _pausedSeconds,
        };
        File.WriteAllText(StatusPath + ".tmp", Json.Write(status));
        File.Move(StatusPath + ".tmp", StatusPath, overwrite: true);
    }

    private static void Tick()
    {
        if ((DateTime.UtcNow - _lastStatus).TotalSeconds >= 1) WriteStatus();
        if (_label == null) return;
        var rs = Obs.Rs;
        _label.Text = $"Step {_step}/{_total}  Floor {rs?.ActFloor ?? 0} {rs?.CurrentRoom?.RoomType.ToString() ?? "Loading"}"
            + $"  {(_paused ? "PAUSED" : "Playing")}  Space: pause/resume  R: this node  B: previous node"
            + (_notice.Length == 0 ? "" : "  | " + _notice);
    }

    public static void AtStep(int step)
    {
        _ready = true;
        _step = _nextStep = step;
    }
    public static void BeforeInput(int step) => _nextStep = step + 1;

    public static async Task WaitForPlayback(SceneTree tree)
    {
        while (Enabled && _active)
        {
            if (_rewind != 0 && !_loading)
            {
                if (_snapshots.Count == 0 || Client.ReadyForRewind()) throw new RewindRequested();
                await Driver.Frames(tree, 1);
                continue;
            }
            if (!_paused) return;
            await Driver.Frames(tree, 1);
        }
    }

    private static string Location(RunState state) =>
        $"{state.CurrentActIndex}:{state.ActFloor}:{string.Join('/', state.VisitedMapCoords)}";

    public static void Capture(AbstractRoom? preFinishedRoom)
    {
        if (!Enabled || !_active || _loading || preFinishedRoom != null || Obs.Rs is not { } state) return;
        var save = RunManager.Instance.ToSave(null);
        _snapshots.Add(new Snapshot(_nextStep, Location(state),
            JsonSerializer.Serialize(save, JsonSerializationUtility.GetTypeInfo<SerializableRun>()),
            JsonSerializer.Serialize(SaveManager.Instance.Progress.ToSerializable(), JsonSerializationUtility.GetTypeInfo<SerializableProgress>()),
            JsonSerializer.Serialize(Rng.Chaotic.ToSerializable(), JsonSerializationUtility.GetTypeInfo<SerializableRng>()),
            CardRarityOdds.regularCommonOdds, CookieCharacter.GetValue(null) as CharacterModel));
        if (_snapshots.Count > 8) _snapshots.RemoveAt(0);
    }

    public static async Task<int?> Rewind(SceneTree tree)
    {
        int back = _rewind;
        _rewind = 0;
        int current = Obs.Rs is { } state ? _snapshots.FindLastIndex(s => s.Location == Location(state)) : -1;
        int index = back == 2 ? (current >= 0 ? current - 1 : _snapshots.Count - 1) : current;
        if (index < 0)
        {
            _notice = "Rewind unavailable: no room-entry snapshot";
            return null;
        }
        var snapshot = _snapshots[index];
        var save = JsonSerializer.Deserialize(snapshot.Run, JsonSerializationUtility.GetTypeInfo<SerializableRun>())!;
        var progress = JsonSerializer.Deserialize(snapshot.Progress, JsonSerializationUtility.GetTypeInfo<SerializableProgress>())!;
        var chaotic = JsonSerializer.Deserialize(snapshot.Chaotic, JsonSerializationUtility.GetTypeInfo<SerializableRng>())!;
        _loading = true;
        _notice = "Rewinding...";
        try
        {
            Client.UninstallSelector();
            await Driver.Frames(tree, 2);
            await NGame.Instance.ReturnToMainMenu();
            await Driver.Frames(tree, 2);
            SaveManager.Instance.Progress = ProgressState.FromSerializable(progress, new DeserializationContext());
            Rng.Chaotic.LoadFromSerializable(chaotic);
            CardRarityOdds.regularCommonOdds = snapshot.CommonOdds;
            CookieCharacter.SetValue(null, snapshot.CookieCharacter);
            var restored = RunState.FromSerializable(save);
            await RunManager.Instance.SetUpSavedSingleplayer(restored, save);
            Client.InstallSelector();
            Obs.ResetRun();
            await NGame.Instance.LoadRun(restored, null);
            await Driver.Frames(tree, 2);
            _snapshots.RemoveRange(index + 1, _snapshots.Count - index - 1);
            AtStep(snapshot.Step);
            _notice = "";
            return snapshot.Step;
        }
        finally { _loading = false; }
    }
}
