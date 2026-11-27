using System.Reflection;
using System.Text.Json;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Odds;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Validation;

namespace Sts2Headless;

public partial class RunSimulator
{
    private sealed record RunSnapshot(int Version, int Id, string Kind, int Floor, string Room,
        string Run, string Progress, string Chaotic, float CommonOdds, string? CookieCharacter);

    private readonly Dictionary<int, RunSnapshot> _saves = new();
    private int _saveId;
    private RunSnapshot? _latestSave;
    private static readonly FieldInfo _cookieCharacter = typeof(YummyCookie)
        .GetField("_cachedRandomCharacter", BindingFlags.Static | BindingFlags.NonPublic)!;

    internal void CaptureSave(AbstractRoom? preFinishedRoom)
    {
        if (_runState == null) throw new InvalidOperationException("save outside a run");
        var save = new RunSnapshot(1, ++_saveId,
            preFinishedRoom == null ? "room" : preFinishedRoom is CombatRoom ? "finished" : "event",
            _runState.ActFloor,
            preFinishedRoom?.RoomType.ToString() ?? _runState.CurrentMapPoint!.PointType.ToString(),
            JsonSerializer.Serialize(RunManager.Instance.ToSave(preFinishedRoom), JsonSerializationUtility.GetTypeInfo<SerializableRun>()),
            JsonSerializer.Serialize(SaveManager.Instance.Progress.ToSerializable(), JsonSerializationUtility.GetTypeInfo<SerializableProgress>()),
            JsonSerializer.Serialize(Rng.Chaotic.ToSerializable(), JsonSerializationUtility.GetTypeInfo<SerializableRng>()),
            CardRarityOdds.regularCommonOdds, (_cookieCharacter.GetValue(null) as CharacterModel)?.Id.Entry);
        _saves.Add(save.Id, save);
        _latestSave = save;
    }

    private Dictionary<string, object?>? SaveInfo() => _latestSave is { } s
        ? new() { ["id"] = s.Id, ["kind"] = s.Kind, ["floor"] = s.Floor, ["room"] = s.Room }
        : null;

    public Dictionary<string, object?> RlSaveGet(int id)
    {
        if (!_saves.TryGetValue(id, out var s)) return ErrorResponse($"rl_save_get: unknown save id {id}");
        return new()
        {
            ["type"] = "save", ["id"] = s.Id, ["kind"] = s.Kind, ["floor"] = s.Floor,
            ["room"] = s.Room, ["data"] = JsonSerializer.Serialize(s),
        };
    }

    /// <summary>The client's Continue path, with NGame.LoadRun's game-object work in place of its UI.</summary>
    public Dictionary<string, object?> RlLoad(int? id, string? data)
    {
        RunSnapshot s;
        if (data != null)
            s = JsonSerializer.Deserialize<RunSnapshot>(data) ?? throw new JsonException("empty save data");
        else if (id is { } key && _saves.TryGetValue(key, out var stored)) s = stored;
        else return ErrorResponse($"rl_load: unknown save id {id}");
        if (s.Version != 1) return ErrorResponse($"rl_load: unsupported save version {s.Version}");
        if (s.Kind != "room")
            return ErrorResponse($"rl_load: {s.Kind} saves cannot be loaded: the client re-enters pre-finished rooms and repeats entry hooks (e.g. Maw Bank grants gold again); load the room entry save and replay its actions instead");
        var progress = JsonSerializer.Deserialize(s.Progress, JsonSerializationUtility.GetTypeInfo<SerializableProgress>())!;
        var chaotic = JsonSerializer.Deserialize(s.Chaotic, JsonSerializationUtility.GetTypeInfo<SerializableRng>())!;

        CleanUpRun();
        if (RunManager.Instance.ActionExecutor is { IsRunning: true } || RunManager.Instance.ActionQueueSet is { IsEmpty: false })
            throw new InvalidOperationException("dirty process: the previous run left game actions running or queued");
        Current = this;
        if (_saves.TryGetValue(s.Id, out var previous) && previous != s) s = s with { Id = _saveId + 1 };
        _saveId = Math.Max(_saveId, s.Id);
        _saves[s.Id] = s;
        _latestSave = s;
        SaveManager.Instance.Progress = ProgressState.FromSerializable(progress, new DeserializationContext());
        Rng.Chaotic.LoadFromSerializable(chaotic);
        typeof(CardRarityOdds).GetField("regularCommonOdds", BindingFlags.Public | BindingFlags.Static)!.SetValue(null, s.CommonOdds);
        _cookieCharacter.SetValue(null, s.CookieCharacter == null ? null
            : ModelDb.GetById<CharacterModel>(new ModelId("CHARACTER", s.CookieCharacter)));
        var save = JsonSerializer.Deserialize(s.Run, JsonSerializationUtility.GetTypeInfo<SerializableRun>())!;
        if (save.PreFinishedRoom != null) return ErrorResponse("rl_load: room save contains a pre-finished room");
        _runState = RunState.FromSerializable(save);
        RunTask(RunManager.Instance.SetUpSavedSingleplayer(_runState, save), "SetUpSavedSingleplayer");
        Diag.Attach();
        _selectorScope = RunManager.Instance.CardSelectorStack.UseLocalSelector(_selector);
        _relicSync = RunManager.Instance.TreasureRoomRelicSynchronizer;
        _relicSync.RelicsAwarded += OnRelicsAwarded;
        RunManager.Instance.Launch();
        RunTask(RunManager.Instance.GenerateMap(), "GenerateMap");
        RunTask(RunManager.Instance.LoadIntoLatestMapCoord(AbstractRoom.FromSerializable(save.PreFinishedRoom, _runState)), "LoadIntoLatestMapCoord");
        return RlRespond(Decide());
    }
}
