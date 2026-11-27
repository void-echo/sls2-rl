using System.Reflection;
using System.Text.Json;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace ParityOracle;

/// <summary>
/// Scenario setup for targeted runs: the same game calls as the engine's scenario commands (engine/src/Sts2Headless/
/// Debug.cs), applied at the same point (after rl_start, i.e. at the first decision). Only commands that go through
/// real game paths are supported: obtain_relics (RelicCmd.Obtain), dbg_procure_potion (PotionCmd.TryToProcure),
/// dbg_add_card (CardPileCmd.Add), rl_enter_room / enter_room (RunManager.EnterRoomDebug, the dev-console path),
/// dbg_enter_act (RunManager.EnterAct).
/// </summary>
public static class Scenario
{
    private static string? Str(JsonElement cmd, string name) =>
        cmd.TryGetProperty(name, out var e) && e.ValueKind == JsonValueKind.String ? e.GetString() : null;

    private static int? Int(JsonElement cmd, string name) =>
        cmd.TryGetProperty(name, out var e) && e.ValueKind == JsonValueKind.Number ? e.GetInt32() : null;

    private static List<string> Strs(JsonElement cmd, string name)
    {
        var l = new List<string>();
        if (cmd.TryGetProperty(name, out var e) && e.ValueKind == JsonValueKind.Array)
            foreach (var x in e.EnumerateArray())
                if (x.GetString() is { } s) l.Add(s);
        return l;
    }

    private static T Model<T>(string category, string id) where T : AbstractModel =>
        ModelDb.GetByIdOrNull<T>(new ModelId(category, id.Contains('.') ? id[(id.IndexOf('.') + 1)..] : id))
        ?? throw new ArgumentException($"unknown {category.ToLowerInvariant()} {id}");

    private static readonly MethodInfo _afterMapLocationChanged =
        typeof(RunManager).GetMethod("AfterMapLocationChanged", BindingFlags.Instance | BindingFlags.NonPublic)!;

    /// <summary>Starts the command and returns its task; the driver waits for it, as the engine's Decide does.</summary>
    public static Task Apply(JsonElement cmd)
    {
        var rs = Obs.Rs ?? throw new InvalidOperationException("no run");
        var p = rs.Players[0];
        var name = Str(cmd, "cmd");
        Entry.Log("scenario setup: " + cmd.GetRawText());
        switch (name)
        {
            case "obtain_relics":
            case "dbg_obtain_relic":
            {
                var relics = Strs(cmd, "ids").Concat(Str(cmd, "id") is { } one ? new[] { one } : Array.Empty<string>())
                    .Select(id => Model<RelicModel>("RELIC", id)).ToList();
                return Run(ObtainAll(relics, p));
            }
            case "dbg_procure_potion":
            {
                var m = Model<PotionModel>("POTION", Str(cmd, "id") ?? "");
                return Run(PotionCmd.TryToProcure(m.ToMutable(), p));
            }
            case "dbg_add_card":
            {
                var m = Model<CardModel>("CARD", Str(cmd, "id") ?? "");
                return Run(CardPileCmd.Add(rs.CreateCard(m, p), PileType.Deck));
            }
            case "rl_enter_room":
            case "enter_room":
                return EnterRoom(cmd, rs);
            case "dbg_enter_act":
            {
                int act = Int(cmd, "act") ?? 1;
                return Run(RunManager.Instance.EnterAct(act - 1, doTransition: false));
            }
            default:
                throw new ArgumentException("unsupported scenario command " + name);
        }
    }

    private static Task Run(Task t)
    {
        TaskHelper.RunSafely(t);
        return t;
    }

    private static async Task ObtainAll(List<RelicModel> relics, MegaCrit.Sts2.Core.Entities.Players.Player p)
    {
        foreach (var r in relics) await RelicCmd.Obtain(r.ToMutable(), p);
    }

    /// <summary>Engine RunSimulator.EnterRoom: ActionExecutor.Pause, EnterRoomDebug(showTransition: false), AfterMapLocationChanged.</summary>
    private static Task EnterRoom(JsonElement cmd, RunState rs)
    {
        var type = (Str(cmd, "type") ?? "").ToLowerInvariant();
        var enc = Str(cmd, "encounter");
        var evId = Str(cmd, "event");
        RoomType roomType = type switch
        {
            "monster" or "combat" => RoomType.Monster,
            "elite" => RoomType.Elite,
            "boss" => RoomType.Boss,
            "shop" or "merchant" => RoomType.Shop,
            "rest" or "rest_site" or "restsite" => RoomType.RestSite,
            "event" => RoomType.Event,
            "treasure" => RoomType.Treasure,
            _ => throw new ArgumentException($"unknown room type '{type}'"),
        };
        AbstractModel? model = null;
        if (enc != null) model = Model<EncounterModel>("ENCOUNTER", enc).ToMutable();
        else if (roomType is RoomType.Monster or RoomType.Elite) model = Model<EncounterModel>("ENCOUNTER", "SHRINKER_BEETLE_WEAK").ToMutable();
        if (evId != null) { model = Model<EventModel>("EVENT", evId); roomType = RoomType.Event; }
        if (model is EncounterModel em) roomType = em.RoomType;
        if (Int(cmd, "floor") is { } floor) rs.ActFloor = floor;
        switch (Str(cmd, "at"))
        {
            case null: break;
            case "boss": rs.AddVisitedMapCoord(rs.Map.BossMapPoint.coord); break;
            case "boss2":
                rs.AddVisitedMapCoord(rs.Map.BossMapPoint.coord);
                rs.AddVisitedMapCoord(rs.Map.SecondBossMapPoint!.coord);
                break;
        }
        RunManager.Instance.ActionExecutor.Pause();
        return Run(EnterRoomTask(roomType, model));
    }

    private static async Task EnterRoomTask(RoomType roomType, AbstractModel? model)
    {
        await RunManager.Instance.EnterRoomDebug(roomType, MapPointType.Unassigned, model, showTransition: false);
        _afterMapLocationChanged.Invoke(RunManager.Instance, null);
    }
}
