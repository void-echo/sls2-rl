using System.Reflection;
using System.Text.Json;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Random;

namespace Sts2Headless;

public partial class RunSimulator
{
    public class LateBlitzState : DispatchProxy
    {
        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args) => targetMethod!.Name switch
        {
            "get_GameMode" => MegaCrit.Sts2.Core.Runs.GameMode.Blitz,
            "get_TotalFloor" => 100,
            _ => targetMethod.Invoke(MegaCrit.Sts2.Core.Runs.NullRunState.Instance, args),
        };
    }

    private object? _revertDebugValue;
    private object? _revertDebugSubject;
    private CardModel? _revertDebugClone;

    public Dictionary<string, object?> DebugRevertState()
    {
        var p = _runState!.Players[0];
        Dictionary<string, object?> Counters(object set) => set.GetType().GetProperties()
            .Where(x => x.PropertyType == typeof(Rng))
            .ToDictionary(x => x.Name, x => (object?)AccessTools.Field(typeof(Rng), "_counter").GetValue(x.GetValue(set)));
        return new()
        {
            ["type"] = "revert_debug", ["value"] = _revertDebugValue,
            ["run_rng"] = Counters(_runState.Rng), ["player_rng"] = Counters(p.PlayerRng),
            ["active"] = Revert.PatchRevert.Active,
            ["watch_value"] = Revert.ScenarioWatch.Value,
            ["draw_order"] = p.PlayerCombatState.DrawPile.Cards.Select(c => c.Id.ToString()).ToArray(),
            ["spoils_quests"] = _runState.Map.GetAllMapPoints().Sum(point => point.Quests.Count(q => q is MegaCrit.Sts2.Core.Models.Cards.SpoilsMap)),
        };
    }

    public Dictionary<string, object?> DebugRevert(JsonElement cmd)
    {
        var p = _runState!.Players[0];
        var combat = p.Creature.CombatState!;
        CardModel Card(JsonElement spec)
        {
            var id = spec.ValueKind == JsonValueKind.String ? spec.GetString()! : spec.GetProperty("id").GetString()!;
            var card = combat.CreateCard(Model<CardModel>("CARD", id), p);
            if (spec.ValueKind == JsonValueKind.Object)
            {
                if (Int(spec, "up") is { } up) for (var i = 0; i < up; i++) CardCmd.Upgrade(card);
                if (Int(spec, "cost") is { } cost) card.EnergyCost.SetThisCombat(cost);
            }
            return card;
        }
        object Subject(string id)
        {
            if (id == "$player") return p.Creature;
            if (id == "$card") return p.PlayerCombatState.AllCards.First();
            if (id == "$enemy") return combat.Enemies[0];
            if (id == "$enemy2") return combat.Enemies[1];
            if (id == "$combat") return combat;
            if (id == "$pcs") return p.PlayerCombatState;
            if (id == "$history") return CombatManager.Instance.History;
            if (id == "$clone") return _revertDebugClone ??= p.Deck.Cards.Last();
            if (id == "$last") return _revertDebugSubject!;
            var bits = id.Split('.');
            var modelId = new ModelId(bits[0], bits[1]);
            return bits[0] switch
            {
                "CARD" => p.PlayerCombatState.AllCards.Concat(p.Deck.Cards).FirstOrDefault(c => c.Id == modelId)
                    ?? combat.CreateCard(Model<CardModel>("CARD", id), p),
                "RELIC" => p.Relics.First(r => r.Id == modelId),
                "POWER" => p.Creature.Powers.Concat(combat.Enemies.SelectMany(e => e.Powers)).First(r => r.Id == modelId),
                "MONSTER" => combat.Enemies.First(e => e.Monster.Id == modelId).Monster,
                "EVENT" => LocalEvent()!,
                _ => throw new ArgumentException(id),
            };
        }
        var op = Str(cmd, "op")!;
        if (op == "get")
        {
            var subject = Subject(Str(cmd, "subject")!);
            var name = Str(cmd, "member")!;
            var value = AccessTools.Property(subject.GetType(), name)?.GetValue(subject)
                ?? AccessTools.Field(subject.GetType(), name)?.GetValue(subject);
            _revertDebugValue = value is Enum ? value.ToString() : value;
            return DebugRevertState();
        }
        return RunDebugTask(async () =>
        {
            if (op == "setup")
            {
                _revertDebugClone = null;
                await CardPileCmd.RemoveFromCombat(p.PlayerCombatState.AllCards.ToList());
                foreach (var name in new[] { "hand", "draw", "discard", "exhaust", "play" })
                    if (cmd.TryGetProperty(name, out var specs))
                        foreach (var spec in specs.EnumerateArray())
                            await CardPileCmd.AddGeneratedCardToCombat(Card(spec), Enum.Parse<PileType>(name, true), p);
                p.PlayerCombatState.Energy = Int(cmd, "energy") ?? 99;
                foreach (var enemy in combat.Enemies)
                {
                    enemy.SetMaxHpInternal(10000);
                    enemy.SetCurrentHpInternal(10000);
                }
                _revertDebugValue = null;
            }
            else if (op == "power")
            {
                var target = Str(cmd, "target") switch { "enemy" => combat.Enemies[0], "enemy2" => combat.Enemies[1], _ => p.Creature };
                var applier = Str(cmd, "applier") == "enemy" ? combat.Enemies[0] : p.Creature;
                var power = Model<PowerModel>("POWER", Str(cmd, "id")!).ToMutable();
                await PowerCmd.Apply(new ThrowingPlayerChoiceContext(),
                    power, target, Int(cmd, "n") ?? 1, applier, null);
                _revertDebugSubject = power;
            }
            else if (op == "summon_ally")
            {
                await OstyCmd.Summon(new BlockingPlayerChoiceContext(), p, 10, null);
                foreach (var power in p.Osty.Powers.ToList()) await PowerCmd.Remove(power);
            }
            else if (op == "spoils_coord")
            {
                var card = (MegaCrit.Sts2.Core.Models.Cards.SpoilsMap)Subject("CARD.SPOILS_MAP");
                card.SpoilsActIndex = _runState.CurrentActIndex;
                var point = _runState.Map.GetAllMapPoints().First(p => p.PointType == MegaCrit.Sts2.Core.Map.MapPointType.Treasure);
                AccessTools.Property(card.GetType(), "SpoilsCoord").SetValue(card, point.coord);
            }
            else if (op == "guard_team")
            {
                var guarded = Model<PowerModel>("POWER", "GUARDED_POWER").ToMutable();
                await PowerCmd.Apply(new ThrowingPlayerChoiceContext(), guarded, p.Creature, 1, p.Creature, null);
                await PowerCmd.Apply(new ThrowingPlayerChoiceContext(), Model<PowerModel>("POWER", "TANK_POWER").ToMutable(), p.Creature, 1, p.Creature, null);
            }
            else if (op == "remove_power")
                await PowerCmd.Remove((PowerModel)Subject(Str(cmd, "subject")!));
            else if (op == "field")
            {
                var subject = Subject(Str(cmd, "subject")!);
                var member = Str(cmd, "member")!;
                var property = AccessTools.Property(subject.GetType(), member);
                var field = AccessTools.Field(subject.GetType(), member);
                var type = property?.PropertyType ?? field!.FieldType;
                var json = cmd.GetProperty("value");
                object? val = json.ValueKind == JsonValueKind.String && json.GetString() == "$owner" ? p
                    : JsonSerializer.Deserialize(json, type);
                if (property != null) property.SetValue(subject, val); else field!.SetValue(subject, val);
            }
            else if (op == "watch")
                Revert.ScenarioWatch.Start(Subject(Str(cmd, "subject")!), Str(cmd, "member")!);
            else if (op == "move")
            {
                var card = (CardModel)Subject(Str(cmd, "subject")!);
                var pile = Enum.Parse<PileType>(Str(cmd, "pile")!, true);
                if (pile == PileType.Deck && !_runState.ContainsCard(card))
                    AccessTools.Method(_runState.GetType(), "AddCard", new[] { typeof(CardModel) }).Invoke(_runState, new object[] { card });
                await CardPileCmd.Add(card, pile);
            }
            else if (op == "invoke")
            {
                var subject = Subject(Str(cmd, "subject")!);
                _revertDebugSubject = subject;
                var method = AccessTools.Method(subject.GetType(), Str(cmd, "member")!);
                var args = cmd.GetProperty("args").EnumerateArray().ToArray();
                var parameters = method.GetParameters();
                var values = args.Select((arg, index) =>
                {
                    var type = parameters[index].ParameterType;
                    if (arg.ValueKind == JsonValueKind.Null) return null;
                    if (arg.ValueKind != JsonValueKind.String || !arg.GetString()!.StartsWith('$'))
                        return type.IsEnum ? Enum.Parse(type, arg.GetString()!) : JsonSerializer.Deserialize(arg, type);
                    var key = arg.GetString()!;
                    return key switch
                    {
                        "$context" => new BlockingPlayerChoiceContext(), "$player" => p.Creature, "$owner" => p,
                        "$enemy" => combat.Enemies[0], "$enemies" => combat.Enemies.ToList(),
                        "$dead_enemy" => new List<Creature> { combat.Enemies[0] },
                        "$participants" => new List<Creature> { p.Creature }, "$combat" => combat,
                        "$room" => _runState.CurrentRoom,
                        "$late_blitz" => DispatchProxy.Create<MegaCrit.Sts2.Core.Runs.IRunState, LateBlitzState>(), "$card" => p.PlayerCombatState.AllCards.First(),
                        "$rooms" => new HashSet<MegaCrit.Sts2.Core.Rooms.RoomType>(Enum.GetValues<MegaCrit.Sts2.Core.Rooms.RoomType>()),
                        "$options" => new List<MegaCrit.Sts2.Core.Entities.RestSite.RestSiteOption>(),
                        "$hatch_options" => new List<MegaCrit.Sts2.Core.Entities.RestSite.RestSiteOption> { new MegaCrit.Sts2.Core.Entities.RestSite.HatchRestSiteOption(p) },
                        "$card_rewards" => p.PlayerCombatState.AllCards.Select(c => new CardCreationResult(c)).ToList(),
                        "$allies" => combat.Allies.Where(c => c != p.Creature).ToList(),
                        "$play" => new CardPlay
                        {
                            Card = p.PlayerCombatState.AllCards.First(), Player = p, Target = combat.Enemies[0],
                            ResultPile = PileType.Discard, IsAutoPlay = cmd.TryGetProperty("auto", out var a) && a.GetBoolean(),
                            PlayIndex = 0, PlayCount = 1,
                            Resources = new ResourceInfo { EnergySpent = 0, EnergyValue = Int(cmd, "energy_value") ?? 0, StarsSpent = 0, StarValue = 0 },
                        },
                        _ => Subject(key[1..]),
                    };
                }).ToArray();
                var result = method.Invoke(subject, values);
                if (result is Task task)
                {
                    await task;
                    result = task.GetType().GetProperty("Result")?.GetValue(task);
                    if (result?.GetType().FullName?.Contains("VoidTaskResult") == true) result = null;
                }
                _revertDebugValue = result switch
                {
                    AbstractModel model => model.Id.ToString(),
                    IEnumerable<MegaCrit.Sts2.Core.Rooms.RoomType> rooms => rooms.Select(x => x.ToString()).ToArray(),
                    Enum e => e.ToString(),
                    _ => result,
                };
            }
            else throw new ArgumentException(op);
        });
    }
}
