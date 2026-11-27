using System.Text.Json;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

/// <summary>
/// Scenario construction for tests (never used by training): put the player in a given state, give items through
/// the real acquisition commands, enter an arbitrary room or act. Every command answers with a normal RL response
/// (or {"type":"error"}), so a test continues with rl_act from there.
/// </summary>
public partial class RunSimulator
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

    /// <summary>Run a game task the way a UI callback would (fire and forget on the game loop) and report the next decision.</summary>
    private Dictionary<string, object?> RunDebugTask(Func<Task> work)
    {
        if (_runState == null) return ErrorResponse("no run in progress");
        TaskHelper.RunSafely(work());
        return RlRespond(Decide());
    }

    /// <summary>
    /// {"cmd":"set_player", "hp", "max_hp", "gold", "relics":[ids], "deck":[ids], "potions":[ids]}: overwrite the player
    /// state directly (relics are attached with AddRelicInternal, without AfterObtained effects).
    /// </summary>
    public Dictionary<string, object?> SetPlayer(JsonElement cmd)
    {
        if (_runState == null) return ErrorResponse("no run in progress");
        try
        {
            var p = _runState.Players[0];
            if (Int(cmd, "max_hp") is { } mhp) p.Creature.SetMaxHpInternal(mhp);
            if (Int(cmd, "hp") is { } hp) p.Creature.SetCurrentHpInternal(hp);
            if (Int(cmd, "gold") is { } gold) p.Gold = gold;
            if (cmd.TryGetProperty("relics", out _))
            {
                foreach (var r in p.Relics.ToList()) p.RemoveRelicInternal(r, silent: true);
                foreach (var id in Strs(cmd, "relics")) p.AddRelicInternal(Model<RelicModel>("RELIC", id).ToMutable(), -1, silent: true);
            }
            if (cmd.TryGetProperty("deck", out _))
            {
                foreach (var c in p.Deck.Cards.ToList()) _runState.RemoveCard(c);
                p.Deck.Clear(silent: true);
                foreach (var id in Strs(cmd, "deck")) p.Deck.AddInternal(_runState.CreateCard(Model<CardModel>("CARD", id), p), silent: true);
            }
            if (cmd.TryGetProperty("potions", out _))
            {
                foreach (var pot in p.Potions.ToList()) p.DiscardPotionInternal(pot, silent: true);
                foreach (var id in Strs(cmd, "potions"))
                {
                    var r = p.AddPotionInternal(Model<PotionModel>("POTION", id).ToMutable(), -1, silent: true);
                    if (!r.success) return ErrorResponse($"set_player: no slot for potion {id}");
                }
            }
            return RlRespond(Decide());
        }
        catch (Exception e) { return ErrorResponse($"set_player: {e.GetType().Name}: {e.Message}"); }
    }

    /// <summary>{"cmd":"obtain_relics","ids":[...]}: RelicCmd.Obtain for each (AfterObtained runs; prompts surface as decisions).</summary>
    public Dictionary<string, object?> ObtainRelics(JsonElement cmd)
    {
        if (_runState == null) return ErrorResponse("no run in progress");
        var p = _runState.Players[0];
        List<RelicModel> relics;
        try { relics = Strs(cmd, "ids").Concat(Str(cmd, "id") is { } one ? new[] { one } : Array.Empty<string>()).Select(id => Model<RelicModel>("RELIC", id)).ToList(); }
        catch (Exception e) { return ErrorResponse(e.Message); }
        return RunDebugTask(async () =>
        {
            foreach (var r in relics) await RelicCmd.Obtain(r.ToMutable(), p);
        });
    }

    public Dictionary<string, object?> ProcurePotion(JsonElement cmd)
    {
        if (_runState == null) return ErrorResponse("no run in progress");
        var p = _runState.Players[0];
        PotionModel m;
        try { m = Model<PotionModel>("POTION", Str(cmd, "id") ?? ""); } catch (Exception e) { return ErrorResponse(e.Message); }
        return RunDebugTask(async () => await PotionCmd.TryToProcure(m.ToMutable(), p));
    }

    public Dictionary<string, object?> AddCard(JsonElement cmd)
    {
        if (_runState == null) return ErrorResponse("no run in progress");
        var p = _runState.Players[0];
        CardModel m;
        try { m = Model<CardModel>("CARD", Str(cmd, "id") ?? ""); } catch (Exception e) { return ErrorResponse(e.Message); }
        var rs = _runState;
        return RunDebugTask(async () => await CardPileCmd.Add(rs.CreateCard(m, p), PileType.Deck));
    }

    /// <summary>
    /// {"cmd":"rl_enter_room","type":"Monster|Elite|Boss|Shop|Rest|Event|Treasure","encounter":ID,"event":ID,"at":"boss|boss2","floor":n}:
    /// RunManager.EnterRoomDebug (the game's own dev-console path: exit rooms, map point history entry, CreateRoom,
    /// EnterRoom) preceded by ActionExecutor.Pause like EnterMapPointInternal. "at" first marks the act's boss
    /// (or second boss) map point as visited, so the boss's proceed button behaves as on the real map.
    /// </summary>
    public Dictionary<string, object?> EnterRoom(JsonElement cmd)
    {
        if (_runState == null) return ErrorResponse("no run in progress");
        var rs = _runState;
        RoomType roomType;
        AbstractModel? model = null;
        try
        {
            var type = (Str(cmd, "type") ?? "").ToLowerInvariant();
            var enc = Str(cmd, "encounter");
            var evId = Str(cmd, "event");
            roomType = type switch
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
            if (enc != null) model = Model<EncounterModel>("ENCOUNTER", enc).ToMutable();
            else if (roomType is RoomType.Monster or RoomType.Elite) model = Model<EncounterModel>("ENCOUNTER", "SHRINKER_BEETLE_WEAK").ToMutable();
            if (evId != null) { model = Model<EventModel>("EVENT", evId); roomType = RoomType.Event; }
            if (roomType == RoomType.Event && model == null) throw new ArgumentException("event room needs 'event'");
            if (model is EncounterModel em) roomType = em.RoomType;
            if (Int(cmd, "floor") is { } floor) rs.ActFloor = floor;
            switch (Str(cmd, "at"))
            {
                case null: break;
                case "boss": rs.AddVisitedMapCoord(rs.Map.BossMapPoint.coord); break;
                case "boss2":
                    if (rs.Map.SecondBossMapPoint == null) throw new ArgumentException("this act has no second boss");
                    rs.AddVisitedMapCoord(rs.Map.BossMapPoint.coord);
                    rs.AddVisitedMapCoord(rs.Map.SecondBossMapPoint.coord);
                    break;
                default: throw new ArgumentException("'at' must be boss or boss2");
            }
        }
        catch (Exception e) { return ErrorResponse($"rl_enter_room: {e.Message}"); }
        _travelRoom = null;
        RunManager.Instance.ActionExecutor.Pause();
        return RunDebugTask(async () =>
        {
            await RunManager.Instance.EnterRoomDebug(roomType, MapPointType.Unassigned, model, showTransition: false);
            // EnterMapPointInternal ends with AfterMapLocationChanged (map votes are accepted from the new location);
            // EnterRoomDebug does not, so a later map move from a debug room would be rejected.
            _afterMapLocationChanged.Invoke(RunManager.Instance, null);
        });
    }

    private static readonly System.Reflection.MethodInfo _afterMapLocationChanged =
        typeof(RunManager).GetMethod("AfterMapLocationChanged", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;

    /// <summary>{"cmd":"dbg_enter_act","act":2}: RunManager.EnterAct(act-1) (new map, the act's ancient start node).</summary>
    public Dictionary<string, object?> EnterActDebug(JsonElement cmd)
    {
        if (_runState == null) return ErrorResponse("no run in progress");
        int act = Int(cmd, "act") ?? 1;
        if (act < 1 || act > _runState.Acts.Count) return ErrorResponse($"act must be 1..{_runState.Acts.Count}");
        _travelRoom = null;
        return RunDebugTask(() => RunManager.Instance.EnterAct(act - 1, doTransition: false));
    }

    public Dictionary<string, object?> DebugInfo()
    {
        var rs = _runState;
        if (rs == null) return ErrorResponse("no run in progress");
        var p = rs.Players[0];
        List<string>? sphere = null;
        if (_prompts.OfType<SpherePrompt>().FirstOrDefault() is { } sp)
            sphere = (typeof(MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame)
                .GetField("_revealed", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)?.GetValue(sp.Game) as System.Collections.IEnumerable)
                ?.Cast<object>().Select(o => o.GetType().Name).ToList();
        return new Dictionary<string, object?>
        {
            ["type"] = "info",
            ["room"] = rs.CurrentRoom?.GetType().Name,
            ["room_count"] = rs.CurrentRoomCount,
            ["event"] = LocalEvent()?.Id.ToString(),
            ["event_finished"] = LocalEvent()?.IsFinished,
            ["prompts"] = _prompts.Where(x => x.IsWaiting).Select(x => x.GetType().Name).ToList(),
            ["decision"] = _rlDecision?.Kind,
            ["is_game_over"] = rs.IsGameOver,
            ["abandoned"] = RunManager.Instance.IsAbandoned,
            ["won"] = _won,
            ["dead"] = p.Creature?.IsDead,
            ["hp"] = p.Creature?.CurrentHp,
            ["relics"] = p.Relics.Select(r => r.Id.Entry).ToList(),
            ["deck_n"] = p.Deck.Cards.Count,
            ["act"] = rs.CurrentActIndex,
            ["acts"] = rs.Acts.Select(a => a.Id.ToString()).ToList(),
            ["second_boss"] = rs.Map?.SecondBossMapPoint != null,
            ["sphere_revealed"] = sphere,
            ["potion_slots"] = p.MaxPotionCount,
            ["potions"] = p.Potions.Select(x => x.Id.Entry).ToList(),
            ["can_use_potions"] = p.CanUseOrRemovePotions,
            ["combat"] = CombatManager.Instance.IsInProgress,
            ["foreign_posts"] = GameLoop.Instance.ForeignPosts,
            ["state"] = DescribeState(),
            ["recent"] = Diag.Last(40).ToList(),
        };
    }
}
