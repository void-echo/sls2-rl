using System.Collections.Concurrent;
using System.Reflection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Combat.History;
using MegaCrit.Sts2.Core.Combat.History.Entries;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Entities.Rewards;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Models.Powers;
using MegaCrit.Sts2.Core.MonsterMoves;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace Sts2Obs;

/// <summary>
/// Observation state kept between decisions by whoever builds observations (the engine's RunSimulator, the parity
/// mod's Obs): the cards played this turn as of the last combat observation, the map node list of the current act,
/// and whether reading the known draw-pile top may record (see <see cref="KnownTop"/>).
/// </summary>
public sealed class ObsState
{
    /// <summary>Cards played this turn; set while the combat observation is built and kept until the next one.</summary>
    public HashSet<object>? PlayedThisTurn;

    // Node list for the current act; cached per map object so indices are stable within an act.
    public object? MapKey;
    public List<MapPoint> MapNodes = new();

    /// <summary>Read the known draw-pile top without recording the reading (<see cref="KnownTop.Peek"/>): the parity
    /// mod reads the screen many times while it settles and records only at the decision it reports.</summary>
    public bool PeekKnownTop;

    public void ResetMap()
    {
        MapKey = null;
        MapNodes = new();
    }
}

/// <summary>
/// The RL observation (engine/RL_PROTOCOL.md) and the data parts of the candidate descriptors, read from live game
/// objects. One source for both builders of observations: the headless engine (engine/src/Sts2Headless/
/// RunSimulator.Rl.cs) and the real-client parity oracle (parity/mod/src/Obs.cs, Client.cs) compile this file, so
/// the two cannot drift apart. Nothing here depends on how the decision was reached (prompts, screens, the game loop):
/// what does is passed in (the ctx builder, declined rewards, pending discards, whether a selection screen is open).
/// </summary>
public static class ObsShared
{
    #region Observation

    /// <summary>The whole observation. <paramref name="ctx"/> is built after the combat part (it may show cards, and
    /// cards are tagged with the cards played this turn that the combat part reads).</summary>
    public static Dictionary<string, object?> Observation(RunState rs, Player player, ObsState st, Func<Dictionary<string, object?>> ctx)
    {
        return new Dictionary<string, object?>
        {
            ["g"] = new Dictionary<string, object?>
            {
                ["act"] = rs.CurrentActIndex + 1,
                ["floor"] = rs.ActFloor,
                ["total_floor"] = rs.TotalFloor,
                ["asc"] = rs.AscensionLevel,
                ["gold"] = player.Gold,
                ["hp"] = player.Creature?.CurrentHp ?? 0,
                ["max_hp"] = player.Creature?.MaxHp ?? 0,
                ["char"] = player.Character?.Id.ToString(),
                ["potion_slots"] = player.MaxPotionCount,
                ["room"] = rs.CurrentRoom?.RoomType.ToString(),
                ["boss"] = rs.Act?.BossEncounter?.Id?.ToString(),
                ["boss2"] = rs.Act?.SecondBossEncounter?.Id?.ToString(),
                ["act_id"] = rs.Act?.Id.ToString(),
                ["removals_used"] = player.ExtraFields?.CardShopRemovalsUsed ?? 0,  // the next shop removal's price
                ["odds"] = Odds(rs, player),
                ["act_rooms"] = ActRooms(rs),
            },
            ["deck"] = player.Deck?.Cards?.Where(c => c != null).Select(c => Card(c, st.PlayedThisTurn)).ToList() ?? new(),
            ["relics"] = player.Relics?.Select(Relic).ToList() ?? new(),
            ["potions"] = player.PotionSlots.Select((p, i) => (p, i)).Where(x => x.p != null).Select(x => new Dictionary<string, object?>
            {
                ["id"] = x.p!.Id.ToString(),
                ["v"] = Vars(x.p!.DynamicVars, x.p),  // the numbers its tooltip shows (heal, damage, ...)
                ["slot"] = x.i,
            }).ToList(),
            ["combat"] = CombatManager.Instance.IsInProgress ? Combat(player, st) : null,
            ["map"] = Map(rs, st),
            ["ctx"] = ctx(),
        };
    }

    public static Dictionary<string, object?> Card(CardModel c, ISet<object>? playedThisTurn, bool inHand = false)
    {
        var d = new Dictionary<string, object?>
        {
            ["id"] = c.Id.ToString(),
            ["up"] = c.CurrentUpgradeLevel,
            ["cost"] = c.EnergyCost?.GetResolved() ?? 0,
            ["x"] = c.EnergyCost?.CostsX ?? false,
            ["star"] = c.CurrentStarCost,
            ["type"] = c.Type.ToString(),
            ["tags"] = c.Tags.Where(t => t != CardTag.None).Select(t => t.ToString()).ToList(),
            ["rar"] = c.Rarity.ToString(),
            ["tgt"] = c.TargetType.ToString(),
            ["v"] = Vars(c.DynamicVars, c),
            ["kw"] = c.Keywords?.Where(k => k != CardKeyword.None).Select(k => k.ToString()).ToList() ?? new List<string>(),
            ["ench"] = c.Enchantment?.Id.ToString(),
            ["ench_n"] = SafeInt(() => c.Enchantment?.Amount ?? 0),
            ["aff"] = c.Affliction?.Id.ToString(),
            ["aff_n"] = SafeInt(() => c.Affliction?.Amount ?? 0),
        };
        try
        {
            // how long the card's cost modifiers last (the cost shown already includes them)
            var mods = CostMods(c);
            if (mods != null) d["cost_mods"] = mods;
            var smods = StarMods(c);
            if (smods != null) d["star_mods"] = smods;
        }
        catch { }
        try
        {
            int replay = c.GetEnchantedReplayCount();  // "Replay X" on the card (Transfigure, Hidden Gem, Sword Sage...)
            if (replay > 0) d["replay"] = replay;
            var kw = (List<string>)d["kw"]!;  // one-turn Retain / Sly, shown on the card like the keyword
            if (c.ShouldRetainThisTurn && !kw.Contains("Retain")) kw.Add("Retain");
            if (c.IsSlyThisTurn && !kw.Contains("Sly")) kw.Add("Sly");
            if (playedThisTurn?.Contains(c) == true) d["played_turn"] = true;  // Bolas, Thrumming Hatchet, Fetch
        }
        catch { }
        var own = OwnState(c, typeof(CardModel));  // Rampage's extra damage, Up My Sleeve's plays this combat, ...
        if (own.Count > 0) d["d"] = own;
        if (c.Enchantment != null)
        {
            var ench = OwnState(c.Enchantment, typeof(EnchantmentModel));  // Momentum's extra damage, Glam used
            if (ench.Count > 0) d["ench_d"] = ench;
            var es = c.Enchantment.Status.ToString();  // a spent Swift / Sown / Vigorous is greyed out on the card
            if (es != "Normal") d["ench_s"] = es;
        }
        if (inHand)
        {
            d["playable"] = CardPlayable(c);
            try
            {
                // the gold / red glow of a card in hand: conditions such as Evil Eye's "if you exhausted a card"
                if (c.ShouldGlowGold) d["glow"] = "gold";
                else if (c.ShouldGlowRed) d["glow"] = "red";
            }
            catch { }
            // Values as displayed on screen (after Strength, Weak, Frail...). The preview mutates the
            // live card, so it is bracketed with ClearPreview exactly like CombatPlayState does.
            var pv = new Dictionary<string, object?>();
            try
            {
                c.DynamicVars.ClearPreview();
                c.UpdateDynamicVarPreview(CardPreviewMode.Normal, c.CurrentTarget, c.DynamicVars);
                foreach (var dv in c.DynamicVars.Values) pv[dv.Name] = (double)dv.PreviewValue;
            }
            catch { }
            finally { try { c.DynamicVars.ClearPreview(); } catch { } }
            d["pv"] = pv;
        }
        return d;
    }

    private static bool CardPlayable(CardModel c) => c.CanPlay(out _, out _);

    private static readonly FieldInfo? _localCostMods = typeof(CardEnergyCost).GetField("_localModifiers", BindingFlags.Instance | BindingFlags.NonPublic);
    private static readonly FieldInfo? _tempStarCosts = typeof(CardModel).GetField("_temporaryStarCosts", BindingFlags.Instance | BindingFlags.NonPublic);

    /// <summary>When a cost modifier wears off: at the end of the turn and when played (ThisTurn), only when played
    /// (UntilPlayed), only at the end of the turn (EndOfTurn), at the end of combat (ThisCombat).</summary>
    private static string Expiry(bool endOfTurn, bool whenPlayed) =>
        (endOfTurn, whenPlayed) switch { (true, true) => "ThisTurn", (false, true) => "UntilPlayed", (true, false) => "EndOfTurn", _ => "ThisCombat" };

    /// <summary>The card's local energy-cost modifiers in the order they apply (CardEnergyCost._localModifiers), as
    /// their expiry kinds; null without any.</summary>
    private static List<string>? CostMods(CardModel c)
    {
        if (_localCostMods?.GetValue(c.EnergyCost) is not IEnumerable<LocalCostModifier> mods) return null;
        List<string>? l = null;
        foreach (var m in mods)
            (l ??= new()).Add(Expiry(m.Expiration.HasFlag(LocalCostModifierExpiration.EndOfTurn), m.Expiration.HasFlag(LocalCostModifierExpiration.WhenPlayed)));
        return l;
    }

    /// <summary>The card's temporary star costs (CardModel._temporaryStarCosts, the last one applies), as expiry kinds.</summary>
    private static List<string>? StarMods(CardModel c)
    {
        if (_tempStarCosts?.GetValue(c) is not IEnumerable<TemporaryCardCost> costs) return null;
        List<string>? l = null;
        foreach (var t in costs)
            if (t != null) (l ??= new()).Add(Expiry(t.ClearsWhenTurnEnds, t.ClearsWhenCardIsPlayed));
        return l;
    }

    /// <summary>The odds a player who tracks the run's history can compute (each is a function of results already
    /// shown; only the next roll is hidden): the unknown-room odds of a monster, shop and treasure room
    /// (UnknownMapPointOdds), the card rarity offset (CardRarityOdds) and the potion drop chance (PotionRewardOdds).</summary>
    public static Dictionary<string, object?> Odds(RunState rs, Player player)
    {
        var d = new Dictionary<string, object?>();
        static double R(float x) => Math.Round((double)x, 6);
        try
        {
            var u = rs.Odds?.UnknownMapPoint;
            if (u != null)
            {
                d["q_monster"] = R(u.MonsterOdds);
                d["q_shop"] = R(u.ShopOdds);
                d["q_treasure"] = R(u.TreasureOdds);
            }
        }
        catch { }
        try
        {
            var po = player.PlayerOdds;
            if (po != null)
            {
                d["rare"] = R(po.CardRarity.CurrentValue);
                d["potion"] = R(po.PotionReward.CurrentValue);
            }
        }
        catch { }
        return d;
    }

    /// <summary>This act's rooms so far, in order (RunState.MapPointHistory of the current act; a map point can hold
    /// two rooms, e.g. an event that turns into a fight): room type and, when it has one, the model id (the
    /// encounter of a fight, the event of an event room). The player remembers which fights and elites the act has
    /// already shown (elites come without replacement from the act's three, the first fights from the weak pool).</summary>
    public static List<Dictionary<string, object?>> ActRooms(RunState rs)
    {
        var list = new List<Dictionary<string, object?>>();
        try
        {
            var hist = rs.MapPointHistory;
            if (rs.CurrentActIndex < 0 || rs.CurrentActIndex >= hist.Count) return list;
            foreach (var point in hist[rs.CurrentActIndex])
                foreach (var room in point.Rooms)
                {
                    var r = new Dictionary<string, object?> { ["t"] = room.RoomType.ToString() };
                    if (room.ModelId != null) r["id"] = room.ModelId.ToString();
                    list.Add(r);
                }
        }
        catch { }
        return list;
    }

    /// <summary>NCardPlay.ShowMultiCreatureTargetingVisuals: an all/random-enemy card is previewed against the only
    /// hittable enemy if there is one, else in MultiCreatureTargeting mode (a value all enemies share).</summary>
    public static Dictionary<string, object?> MultiTargetPreview(CardModel c)
    {
        var pv = new Dictionary<string, object?>();
        try
        {
            var hittable = c.CombatState?.HittableEnemies;
            var target = hittable is { Count: 1 } ? hittable[0] : null;
            c.DynamicVars.ClearPreview();
            c.UpdateDynamicVarPreview(CardPreviewMode.MultiCreatureTargeting, target, c.DynamicVars);
            foreach (var dv in c.DynamicVars.Values)
                if (dv is not StringVar) pv[dv.Name] = (double)dv.PreviewValue;
        }
        catch { }
        finally { try { c.DynamicVars.ClearPreview(); } catch { } }
        return pv;
    }

    /// <summary>The card's values as shown while it is aimed at <paramref name="target"/> (Vulnerable, Intangible's
    /// cap of 1, ...). Bracketed with ClearPreview like the untargeted preview in <see cref="Card"/>.</summary>
    public static Dictionary<string, object?> TargetPreview(CardModel c, Creature target)
    {
        var pv = new Dictionary<string, object?>();
        try
        {
            c.DynamicVars.ClearPreview();
            c.UpdateDynamicVarPreview(CardPreviewMode.Normal, target, c.DynamicVars);
            foreach (var dv in c.DynamicVars.Values)
                if (dv is not StringVar) pv[dv.Name] = (double)dv.PreviewValue;
        }
        catch { }
        finally { try { c.DynamicVars.ClearPreview(); } catch { } }
        return pv;
    }

    private static readonly ConcurrentDictionary<Type, PropertyInfo[]> _savedProps = new();

    /// <summary>Dynamic vars plus any [SavedProperty] scalar on the model (relic counters such as Tuning Fork's SkillsPlayed).</summary>
    public static Dictionary<string, object?> Vars(DynamicVarSet? vars, object? model)
    {
        var v = new Dictionary<string, object?>();
        try
        {
            if (vars != null)
                foreach (var dv in vars.Values)
                    if (dv is not StringVar) v[dv.Name] = (double)dv.BaseValue;
        }
        catch { }
        if (model == null) return v;
        var props = _savedProps.GetOrAdd(model.GetType(), t => t
            .GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            .Where(p => p.GetIndexParameters().Length == 0
                        && p.CustomAttributes.Any(a => a.AttributeType.Name == "SavedPropertyAttribute")
                        && (p.PropertyType == typeof(int) || p.PropertyType == typeof(bool) || p.PropertyType == typeof(decimal)
                            || p.PropertyType == typeof(long) || p.PropertyType == typeof(float) || p.PropertyType == typeof(double)))
            .ToArray());
        foreach (var p in props)
        {
            try
            {
                var val = p.GetValue(model);
                v["saved." + p.Name] = val is bool b ? (b ? 1.0 : 0.0) : Convert.ToDouble(val);
            }
            catch { }
        }
        return v;
    }

    public static Dictionary<string, object?>? StringVars(DynamicVarSet? vars)
    {
        Dictionary<string, object?>? sv = null;
        try
        {
            if (vars != null)
                foreach (var dv in vars.Values)
                    if (dv is StringVar s && !string.IsNullOrEmpty(s.StringValue)) (sv ??= new())[dv.Name] = s.StringValue;
        }
        catch { }
        return sv;
    }

    public static int SafeInt(Func<int> f)
    {
        try { return f(); } catch { return 0; }
    }

    public static Dictionary<string, object?> Creature(Creature cr, string id) => new()
    {
        ["id"] = id,
        ["hp"] = cr.CurrentHp,
        ["max_hp"] = cr.MaxHp,
        ["block"] = cr.Block,
        ["alive"] = cr.IsAlive,
        ["pw"] = cr.Powers?.Select(Power).ToList() ?? new(),
    };

    // Only what the player sees or has seen goes into the observation: values a tooltip or the relic bar shows,
    // and counts and cards a player who remembers this run knows (a stolen card, cards played this turn).

    private static readonly FieldInfo? _powerVarsField = typeof(PowerModel).GetField("_dynamicVars", BindingFlags.Instance | BindingFlags.NonPublic);
    private static readonly PropertyInfo? _powerCanonicalVars = typeof(PowerModel).GetProperty("CanonicalVars", BindingFlags.Instance | BindingFlags.NonPublic);
    private static readonly FieldInfo? _powerDataField = typeof(PowerModel).GetField("_internalData", BindingFlags.Instance | BindingFlags.NonPublic);

    /// <summary>A power: id and amount; its tooltip vars (<c>v</c>); its own counters and flags, in its fields or its
    /// private data (<c>d</c>: Sloth's cards this turn, Juggling's attacks this turn, Hardened Shell's damage taken this
    /// turn); cards it holds (<c>cards</c>: the card Swipe stole, Nightmare's card).</summary>
    public static Dictionary<string, object?> Power(PowerModel pw)
    {
        var d = new Dictionary<string, object?> { ["id"] = pw.Id.ToString(), ["n"] = pw.Amount };
        try
        {
            if (pw.DisplayAmount != pw.Amount) d["dn"] = pw.DisplayAmount;  // the number on the icon (Orbit, Tender, ...)
        }
        catch { }
        var v = new Dictionary<string, object?>();
        try
        {
            // The var set is created lazily by the DynamicVars getter, so it is read through the field: reading
            // must not change the power, and a set nobody created yet still holds the canonical values.
            var vars = (_powerVarsField?.GetValue(pw) as DynamicVarSet)?.Values
                       ?? _powerCanonicalVars?.GetValue(pw) as IEnumerable<DynamicVar>;
            foreach (var dv in vars ?? Enumerable.Empty<DynamicVar>())
                if (dv is not StringVar) v[dv.Name] = (double)dv.BaseValue;
        }
        catch { }
        if (v.Count > 0) d["v"] = v;
        object? data = null;
        try { data = _powerDataField?.GetValue(pw); } catch { }
        var own = OwnState(pw, typeof(PowerModel));
        if (data != null)
            foreach (var kv in OwnState(data, typeof(object))) own[kv.Key] = kv.Value;
        if (own.Count > 0) d["d"] = own;
        var cards = HeldCards(pw, typeof(PowerModel));
        if (data != null) cards.AddRange(HeldCards(data, typeof(object)));
        if (cards.Count > 0) d["cards"] = cards;
        return d;
    }

    /// <summary>A relic: id, numeric vars and saved counters (<c>v</c>); the counter the relic bar shows (<c>cnt</c>,
    /// Kunai's attacks this turn, Velvet Choker's cards this turn); its bar status when not Normal (<c>Disabled</c> =
    /// greyed out, <c>Active</c> = glowing); its own counters and flags (<c>d</c>); cards it holds (Pael's Tooth).</summary>
    public static Dictionary<string, object?> Relic(RelicModel r)
    {
        var d = new Dictionary<string, object?> { ["id"] = r.Id.ToString(), ["v"] = Vars(r.DynamicVars, r) };
        try
        {
            if (r.ShowCounter) d["cnt"] = r.DisplayAmount;
        }
        catch { }
        if (r.Status != RelicStatus.Normal) d["status"] = r.Status.ToString();
        var own = OwnState(r, typeof(RelicModel));
        if (own.Count > 0) d["d"] = own;
        var cards = HeldCards(r, typeof(RelicModel));
        if (cards.Count > 0) d["cards"] = cards;
        return d;
    }

    private static readonly ConcurrentDictionary<Type, FieldInfo?> _activatingField = new();

    /// <summary>A relic in its activation flash (the 21 counter relics' DoActivateVisuals: <c>_isActivating</c> for
    /// <c>Cmd.Wait(1f)</c>), during which its DisplayAmount shows the threshold and its Status is reset. The wait is
    /// skipped headless and in Instant mode; the parity mod waits until no relic is activating before it reads a
    /// decision (an animation window, not game state).</summary>
    public static bool IsActivating(RelicModel r)
    {
        var f = _activatingField.GetOrAdd(r.GetType(), t =>
        {
            for (var ty = t; ty != null && ty != typeof(RelicModel); ty = ty.BaseType)
            {
                var fi = ty.GetField("_isActivating", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
                if (fi != null && fi.FieldType == typeof(bool)) return fi;
            }
            return null;
        });
        try { return f?.GetValue(r) is true; }
        catch { return false; }
    }

    // Fields that only drive presentation or tests (activation flashes, skins, dialogue, sound), left out.
    private static readonly HashSet<string> _cosmeticFields = new(StringComparer.Ordinal)
    {
        "isActivating", "testEnergyCostOverride", "skin", "hasSpoken", "speechUsed", "attackSfxStrength",
        "shouldPlaySpawnAnimation", "isFinishedTriggering", "showedCapReachedMessage",
        // set only with TestMode off (animation tracks): would differ from the client
        "initialAmount", "initialTargetPosition", "pressureBuildupIdx",
        // the random starting move: the first intent shows it, the history has it after that
        "starterMoveIdx", "starterMoveIndex",
    };

    private static readonly ConcurrentDictionary<(Type, Type), FieldInfo[]> _ownFields = new();
    private static readonly ConcurrentDictionary<(Type, Type), PropertyInfo[]> _savedTextProps = new();

    private static bool IsNumber(Type t) =>
        t == typeof(int) || t == typeof(decimal) || t == typeof(long) || t == typeof(float) || t == typeof(double);

    /// <summary>The scalar state a model keeps in its own mutable instance fields, declared by its class and bases
    /// below <paramref name="stopAt"/>: counters, flags and modes (a monster asleep or burrowed, Rampage's extra
    /// damage). Numbers and booleans (0/1) as numbers, enums and strings as text. Fields backing a [SavedProperty]
    /// (already in <c>v</c>), readonly, compiler-generated and cosmetic fields are left out.</summary>
    public static Dictionary<string, object?> OwnState(object obj, Type stopAt)
    {
        var fields = _ownFields.GetOrAdd((obj.GetType(), stopAt), key =>
        {
            const BindingFlags all = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
            var saved = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (var ty = key.Item1; ty != null && ty != key.Item2 && ty != typeof(object); ty = ty.BaseType)
                foreach (var p in ty.GetProperties(all))
                    if (p.CustomAttributes.Any(a => a.AttributeType.Name == "SavedPropertyAttribute")) saved.Add(p.Name);
            var list = new List<FieldInfo>();
            for (var ty = key.Item1; ty != null && ty != key.Item2 && ty != typeof(object); ty = ty.BaseType)
                foreach (var f in ty.GetFields(all))
                {
                    if (f.IsInitOnly || f.IsLiteral || f.Name.Contains('<')) continue;
                    var name = f.Name.TrimStart('_');
                    if (saved.Contains(name) || _cosmeticFields.Contains(name)) continue;
                    var ft = Nullable.GetUnderlyingType(f.FieldType) ?? f.FieldType;
                    if (ft == typeof(int) || ft == typeof(bool) || ft == typeof(decimal) || ft == typeof(long)
                        || ft == typeof(float) || ft == typeof(double) || ft.IsEnum || ft == typeof(string)
                        || (typeof(AbstractModel).IsAssignableFrom(ft) && !typeof(CardModel).IsAssignableFrom(ft))
                        || (typeof(System.Collections.IDictionary).IsAssignableFrom(ft) && ft.IsGenericType
                            && IsNumber(ft.GetGenericArguments()[^1])))
                        list.Add(f);
                }
            return list.ToArray();
        });
        var s = new Dictionary<string, object?>();
        // [SavedProperty] values that are not numbers (Vars takes the numbers): enums, strings, model ids, e.g. Mad
        // Science's rider or the character a Sea Glass offers. Lists of cards are held cards; Skin is cosmetic.
        var textProps = _savedTextProps.GetOrAdd((obj.GetType(), stopAt), key =>
        {
            var list = new List<PropertyInfo>();
            for (var ty = key.Item1; ty != null && ty != key.Item2 && ty != typeof(object); ty = ty.BaseType)
                foreach (var p in ty.GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
                {
                    if (p.GetIndexParameters().Length > 0 || p.Name == "Skin") continue;
                    if (!p.CustomAttributes.Any(a => a.AttributeType.Name == "SavedPropertyAttribute")) continue;
                    var pt = Nullable.GetUnderlyingType(p.PropertyType) ?? p.PropertyType;
                    if (pt.IsEnum || pt == typeof(string) || pt.Name == "ModelId") list.Add(p);
                }
            return list.ToArray();
        });
        foreach (var p in textProps)
        {
            try
            {
                var val = p.GetValue(obj);
                if (val != null) s[p.Name] = val.ToString();
            }
            catch { }
        }
        foreach (var f in fields)
        {
            try
            {
                var val = f.GetValue(obj);
                var name = f.Name.TrimStart('_');
                switch (val)
                {
                    case null: break;
                    case bool b: s[name] = b ? 1.0 : 0.0; break;
                    case string str: s[name] = str; break;
                    case Enum e: s[name] = e.ToString(); break;
                    case AbstractModel m: s[name] = m.Id.ToString(); break;  // e.g. Fabricator's last summon
                    case System.Collections.IDictionary dict:  // e.g. the Strength Possess took, per creature
                        double sum = 0;
                        foreach (var x in dict.Values) sum += Convert.ToDouble(x);
                        s[name + ".sum"] = sum;
                        break;
                    default: s[name] = Convert.ToDouble(val); break;
                }
            }
            catch { }
        }
        return s;
    }

    private static readonly ConcurrentDictionary<Type, FieldInfo[]> _cardFields = new();

    private static bool IsCardType(Type t)
    {
        static bool Card(Type x) => typeof(CardModel).IsAssignableFrom(x) || x == typeof(SerializableCard);
        if (Card(t)) return true;
        if (t.IsArray) return Card(t.GetElementType()!);
        return t.IsGenericType && t.GetGenericArguments().Length == 1 && Card(t.GetGenericArguments()[0])
               && typeof(System.Collections.IEnumerable).IsAssignableFrom(t);
    }

    /// <summary>Ids of the cards held in instance fields of <paramref name="obj"/>'s own class and its bases up to
    /// <paramref name="stopAt"/> (exclusive).</summary>
    public static List<string> HeldCards(object obj, Type stopAt)
    {
        var fields = _cardFields.GetOrAdd(obj.GetType(), t =>
        {
            var list = new List<FieldInfo>();
            for (var ty = t; ty != null && ty != stopAt && ty != typeof(object); ty = ty.BaseType)
                list.AddRange(ty.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
                    .Where(f => IsCardType(f.FieldType)));
            return list.ToArray();
        });
        var ids = new List<string>();
        foreach (var f in fields)
        {
            try { AddCardIds(f.GetValue(obj), ids); } catch { }
        }
        return ids;
    }

    private static void AddCardIds(object? v, List<string> ids)
    {
        switch (v)
        {
            case CardModel c:  // id, +upgrades, @enchantment: what the tooltip shows of the card
                ids.Add(c.Id + (c.CurrentUpgradeLevel > 0 ? $"+{c.CurrentUpgradeLevel}" : "") + (c.Enchantment != null ? "@" + c.Enchantment.Id : ""));
                break;
            case SerializableCard s:
                if (s.Id != null) ids.Add(s.Id + (s.CurrentUpgradeLevel > 0 ? $"+{s.CurrentUpgradeLevel}" : ""));
                break;
            case System.Collections.IEnumerable e:
                foreach (var x in e) AddCardIds(x, ids);
                break;
        }
    }

    #endregion

    #region Monster moves

    /// <summary>What the screen shows of one intent: type, displayed damage and hits, status card count.</summary>
    public static string IntentSig(AbstractIntent intent, List<Creature>? players, Creature owner)
    {
        var s = intent.IntentType.ToString();
        if (intent is AttackIntent atk && players != null)
        {
            try { s += $":{atk.GetSingleDamage(players, owner)}x{Math.Max(1, atk.Repeats)}"; } catch { }
        }
        else if (intent is StatusIntent st)
            s += $"#{st.CardCount}";
        return s;
    }

    private static readonly PropertyInfo? _condBranches = typeof(ConditionalBranchState).GetProperty("States", BindingFlags.Instance | BindingFlags.NonPublic);

    /// <summary>The id of the move the monster will perform, when a player who knows the monster can tell it: the
    /// client shows only the intents, but the bestiary lists each monster's moves and the player has seen the moves
    /// it performed. Moves outside its state machine (STUNNED, REVIVE_MOVE: set by events the player sees) are
    /// shown as they are. Otherwise the candidates are the moves its pattern allows after its last performed
    /// pattern move, through random and conditional branches, minus those the branch rules exclude (can't repeat,
    /// once only, cooldown). If no random branch was crossed the move follows from visible state (all conditional
    /// branches test visible state) and is exported; else it is exported only if no other candidate shows the same
    /// intents now. Before its first move it is compared with all its moves (some encounters pick the opening move
    /// at random). This way the id never reveals which of two look-alike moves a random branch rolled.</summary>
    public static string? NextMoveId(MonsterModel? m, List<string> performed, List<Creature>? players, Creature owner)
    {
        var next = m?.NextMove;
        var sm = m?.MoveStateMachine;
        if (next == null || sm == null || next.StateId == "UNSET_MOVE") return null;
        if (next.Intents.Count == 0 || next.Intents.Any(i => i.IntentType is IntentType.Hidden or IntentType.Unknown)) return null;
        if (!sm.States.ContainsKey(next.StateId)) return next.StateId;
        try
        {
            var (cands, random, known) = PatternCandidates(m!, performed);
            if (known && cands.Contains(next) && !random) return next.StateId;
            string Sig(MoveState mv) => string.Join("|", mv.Intents.Select(i => IntentSig(i, players, owner)));
            var sig = Sig(next);
            IEnumerable<MoveState> pool = known && cands.Contains(next) ? cands : sm.States.Values.OfType<MoveState>();
            foreach (var mv in pool)
                if (mv.StateId != next.StateId && Sig(mv) == sig) return null;
            return next.StateId;
        }
        catch { return null; }
    }

    /// <summary>The moves the pattern allows next, from the last performed move that belongs to the state machine
    /// (moves set outside it, STUNNED or REVIVE_MOVE, are skipped: the pattern resumes from the move before).
    /// <c>random</c>: a random branch was crossed; <c>known</c>: false before the first pattern move or when the
    /// pattern cannot be followed.</summary>
    public static (HashSet<MoveState> cands, bool random, bool known) PatternCandidates(MonsterModel m, List<string> performed)
    {
        var sm = m.MoveStateMachine!;
        var cands = new HashSet<MoveState>();
        var history = performed.Where(id => sm.States.ContainsKey(id)).ToList();
        if (history.Count == 0) return (cands, true, false);
        if (sm.States[history[0]] is not MoveState lm) return (cands, true, false);
        var from = lm.FollowUpState ?? (lm.FollowUpStateId != null && sm.States.TryGetValue(lm.FollowUpStateId, out var f) ? f : null);
        bool unknown = from == null, random = false;
        if (from != null) Reachable(from, sm, history, cands, ref unknown, ref random, 0);
        return (cands, random, !unknown);
    }

    private static void Reachable(MonsterState s, MonsterMoveStateMachine sm, List<string> performed, HashSet<MoveState> into,
                                  ref bool unknown, ref bool random, int depth)
    {
        if (depth > 8) { unknown = true; return; }
        switch (s)
        {
            case MoveState mv:
                into.Add(mv);
                return;
            case RandomBranchState rb:
                random = true;
                foreach (var w in rb.States)
                    if (BranchAllowed(w, performed))
                    {
                        if (sm.States.TryGetValue(w.stateId, out var t)) Reachable(t, sm, performed, into, ref unknown, ref random, depth + 1);
                        else unknown = true;
                    }
                return;
            case ConditionalBranchState cb:
                if (_condBranches?.GetValue(cb) is System.Collections.IEnumerable list)
                {
                    foreach (var b in list)
                    {
                        if (b?.GetType().GetField("id")?.GetValue(b) is string id && sm.States.TryGetValue(id, out var t))
                            Reachable(t, sm, performed, into, ref unknown, ref random, depth + 1);
                        else unknown = true;
                    }
                }
                else unknown = true;
                return;
            default:
                unknown = true;
                return;
        }
    }

    /// <summary>RandomBranchState's repeat and cooldown rules, applied to the performed moves (newest first).</summary>
    private static bool BranchAllowed(RandomBranchState.StateWeight w, List<string> performed)
    {
        switch (w.repeatType)
        {
            case MoveRepeatType.UseOnlyOnce:
                if (performed.Contains(w.stateId)) return false;
                break;
            case MoveRepeatType.CannotRepeat:
                if (performed.Count > 0 && performed[0] == w.stateId) return false;
                break;
            case MoveRepeatType.CanRepeatXTimes:
                if (w.maxTimes > 0 && performed.Count >= w.maxTimes && performed.Take(w.maxTimes).All(x => x == w.stateId)) return false;
                break;
        }
        return w.cooldown <= 0 || !performed.Take(w.cooldown).Contains(w.stateId);
    }

    /// <summary>The move that will follow the identified next move when the monster's pattern fixes it (the next
    /// move's follow-up is itself a move, not a random or conditional branch), as a player who knows the monster
    /// expects it. The game rolls it only after the next move is performed.</summary>
    public static string? FollowingMoveId(MonsterModel? m, string? nextId, List<string> performed, List<Creature>? players, Creature owner)
    {
        if (nextId == null || m?.MoveStateMachine == null || m.NextMove is not { } next || next.StateId != nextId) return null;
        try
        {
            var sm = m.MoveStateMachine;
            var follow = next.FollowUpState
                         ?? (next.FollowUpStateId != null && sm.States.TryGetValue(next.FollowUpStateId, out var s) ? s : null);
            if (follow is not MoveState mv) return null;
            if (sm.States.ContainsKey(next.StateId)) return mv.StateId;  // a pattern move with a fixed successor
            // A stun resumes with the move rolled before it (or one an effect names): say it only if the player can
            // tell which, i.e. it is outside the pattern's candidates or the only candidate that looks like it.
            var (cands, random, known) = PatternCandidates(m, performed);
            if (!known || !cands.Contains(mv)) return known || !sm.States.ContainsKey(mv.StateId) ? mv.StateId : null;
            if (!random) return mv.StateId;
            string Sig(MoveState x) => string.Join("|", x.Intents.Select(i => IntentSig(i, players, owner)));
            var sig = Sig(mv);
            return cands.Any(x => x != mv && Sig(x) == sig) ? null : mv.StateId;
        }
        catch { return null; }
    }

    /// <summary>The moves a monster performed this combat, newest first (at most <paramref name="n"/>); its move
    /// state machine depends on them.</summary>
    public static List<string> PerformedMoves(MonsterModel? m, IReadOnlyList<CombatHistoryEntry>? hist, int n = 3)
    {
        var moves = new List<string>();
        if (m == null || hist == null) return moves;
        for (int i = hist.Count - 1; i >= 0 && moves.Count < n; i--)
            if (hist[i] is MonsterPerformedMoveEntry me && ReferenceEquals(me.Monster, m))
                moves.Add(me.Move.Id);
        return moves;
    }

    #endregion

    #region Combat

    /// <summary>Enemies still in the fight, in combat order, including the dead that stay to revive (Decimillipede
    /// segments, Test Subject, illusions): the observation's enemy list and the targets' indices.</summary>
    public static List<Creature> CombatEnemies() =>
        CombatManager.Instance.DebugOnlyGetState()?.Enemies?.Where(e => e != null).ToList() ?? new();

    /// <summary>NTargetManager: an enemy can be targeted while alive, hittable and no hook forbids it.</summary>
    public static bool Targetable(Creature c) =>
        c.Side == CombatSide.Enemy && c.IsAlive && c.IsHittable && c.CombatState != null
        && Hook.ShouldAllowTargeting(c.CombatState, c, out _);

    /// <summary>The combat part; refreshes <see cref="ObsState.PlayedThisTurn"/>.</summary>
    public static Dictionary<string, object?>? Combat(Player player, ObsState st)
    {
        var pcs = player.PlayerCombatState;
        var cs = CombatManager.Instance.DebugOnlyGetState();
        if (pcs == null || cs == null) return null;
        var playerCreatures = cs.PlayerCreatures?.ToList();
        var history = CombatManager.Instance.History?.Entries as IReadOnlyList<CombatHistoryEntry>;
        try
        {
            st.PlayedThisTurn = CombatManager.Instance.History!.CardPlaysStarted
                .Where(e => e.HappenedThisTurn(cs) && e.CardPlay.Player == player).Select(e => e.CardPlay.Card)
                .ToHashSet(ReferenceEqualityComparer.Instance);
        }
        catch { st.PlayedThisTurn = null; }
        var played = st.PlayedThisTurn;

        var enemies = CombatEnemies().Select(e =>
        {
            var d = Creature(e, e.Monster?.Id.ToString() ?? "MONSTER.UNKNOWN");
            var intents = new List<Dictionary<string, object?>>();
            try
            {
                foreach (var intent in e.Monster?.NextMove?.Intents ?? Enumerable.Empty<AbstractIntent>())
                {
                    var it = new Dictionary<string, object?> { ["t"] = intent.IntentType.ToString(), ["dmg"] = 0, ["hits"] = 0 };
                    if (intent is AttackIntent atk && playerCreatures != null)
                    {
                        try { it["dmg"] = atk.GetSingleDamage(playerCreatures, e); it["hits"] = Math.Max(1, atk.Repeats); }
                        catch { }
                    }
                    else if (intent is StatusIntent si)
                        it["n"] = si.CardCount;  // the number on the intent's label
                    intents.Add(it);
                }
            }
            catch { }
            d["intents"] = intents;
            var performed = PerformedMoves(e.Monster, history, int.MaxValue);
            var move = NextMoveId(e.Monster, performed, playerCreatures, e);
            if (move != null) d["move"] = move;
            var then = FollowingMoveId(e.Monster, move, performed, playerCreatures, e);
            if (then != null) d["then"] = then;
            if (performed.Count > 0) d["prev"] = performed.Take(3).ToList();
            if (performed.Count > 0)  // every move it performed this combat, counted (once-only moves, call-for-backup)
                d["done"] = performed.GroupBy(x => x).ToDictionary(g => g.Key, g => (object?)g.Count());
            if (e.SlotName != null) d["slot"] = e.SlotName;
            if (e.Monster != null)
            {
                var own = OwnState(e.Monster, typeof(MonsterModel));  // asleep, burrowed, charging, phase, respawns...
                if (own.Count > 0) d["d"] = own;
            }
            if (!Targetable(e)) d["untargetable"] = true;
            return d;
        }).ToList();

        var allies = pcs.Pets.Select(pet => Creature(pet, pet.Monster!.Id.ToString())).ToList();

        var orbs = new List<Dictionary<string, object?>>();
        int orbSlots = 0;
        try
        {
            if (pcs.OrbQueue != null)
            {
                orbSlots = pcs.OrbQueue.Capacity;
                int slot = 0;
                foreach (var orb in pcs.OrbQueue.Orbs)
                    orbs.Add(new Dictionary<string, object?>
                    {
                        ["id"] = orb.Id.ToString(),
                        ["slot"] = slot++,  // evoke and Loop act on the leftmost orb
                        ["passive"] = (double)orb.PassiveVal,
                        ["evoke"] = (double)orb.EvokeVal,
                    });
            }
        }
        catch { }

        return new Dictionary<string, object?>
        {
            ["round"] = cs.RoundNumber,
            // Cards the player started playing this turn, autoplays included (the count Normality, Echo Form, ... use).
            ["played"] = CardsPlayedThisTurn(cs, player),
            ["turn"] = TurnCounts(cs, player),  // this turn: first plays, attacks/skills/powers, draws, exhausts, hp lost...
            ["enc"] = cs.Encounter?.Id.ToString(),
            ["energy"] = pcs.Energy,
            ["max_energy"] = pcs.MaxEnergy,
            ["stars"] = Math.Max(0, pcs.Stars),
            ["arrows_available"] = pcs.AvailableArrowCount,
            ["arrows"] = pcs.Arrows.Select((a, i) => new Dictionary<string, object?>
            {
                ["slot"] = i,
                ["status"] = a.Status.ToString(),
                ["destroy_after_impaled"] = a.DestroyAfterImpaled,
                ["tgt"] = CombatEnemies().FindIndex(e => e.GetPowerInstances<ImpaledPower>().Any(p => p.Arrows.Contains(a))),
            }).ToList(),
            ["hand"] = pcs.Hand?.Cards?.Select(c => Card(c, played, inHand: true)).ToList() ?? new(),
            // Draw pile order is hidden information; sorting by id exposes only the composition.
            ["draw"] = (pcs.DrawPile?.Cards?.Select(c => Card(c, played)) ?? Enumerable.Empty<Dictionary<string, object?>>())
                .OrderBy(CanonicalJson, StringComparer.Ordinal).ToList(),  // sorted by everything shown: no order leaks
            // the cards on top whose order the player knows (put there in plain sight), top first
            ["draw_top"] = pcs.DrawPile != null
                ? (st.PeekKnownTop ? KnownTop.Peek(pcs.DrawPile) : KnownTop.Current(pcs.DrawPile)).Select(c => Card(c, played)).ToList()
                : new(),
            ["pturn"] = pcs.TurnNumber,  // the player's turn number: differs from round after an extra turn
            ["discard"] = pcs.DiscardPile?.Cards?.Select(c => Card(c, played)).ToList() ?? new(),
            ["exhaust"] = pcs.ExhaustPile?.Cards?.Select(c => Card(c, played)).ToList() ?? new(),
            ["player"] = Creature(player.Creature, player.Character?.Id.ToString() ?? "CHARACTER.UNKNOWN"),
            ["allies"] = allies,
            ["enemies"] = enemies,
            ["orbs"] = orbs,
            ["orb_slots"] = orbSlots,
        };
    }

    /// <summary>This turn's counts some powers use besides <c>played</c>: cards played once each (Echo Form counts
    /// first plays of a series), attacks and skills among them (Nostalgia), status cards drawn (Iteration).</summary>
    public static Dictionary<string, object?> TurnCounts(ICombatState cs, Player player)
    {
        var d = new Dictionary<string, object?>();
        try
        {
            var plays = CombatManager.Instance.History.CardPlaysStarted
                .Where(e => e.HappenedThisTurn(cs) && e.CardPlay.Player == player && e.CardPlay.IsFirstInSeries).ToList();
            d["first"] = plays.Count;
            d["attacks"] = plays.Count(e => e.CardPlay.Card.Type == CardType.Attack);
            d["skills"] = plays.Count(e => e.CardPlay.Card.Type == CardType.Skill);
            d["powers"] = plays.Count(e => e.CardPlay.Card.Type == CardType.Power);
            var lastAttack = plays.LastOrDefault(e => e.CardPlay.Card.Type == CardType.Attack);
            if (lastAttack != null) d["last_attack"] = lastAttack.CardPlay.Card.Id.ToString();  // History Course
            var h = CombatManager.Instance.History.Entries.Where(e => e.HappenedThisTurn(cs)).ToList();
            d["status_drawn"] = h.OfType<CardDrawnEntry>().Count(e => e.Card.Owner == player && e.Card.Type == CardType.Status);
            d["drawn_extra"] = h.OfType<CardDrawnEntry>().Count(e => e.Card.Owner == player && !e.FromHandDraw);
            d["exhausted"] = h.OfType<CardExhaustedEntry>().Count(e => e.Card.Owner == player);
            d["discarded"] = h.OfType<CardDiscardedEntry>().Count(e => e.Card.Owner == player);
            d["hp_lost"] = h.OfType<DamageReceivedEntry>().Where(e => e.Result.Receiver == player.Creature).Sum(e => e.Result.UnblockedDamage);
            d["wah"] = MelodyCmd.PlayedWahThisTurn(player);
            d["cho"] = MelodyCmd.PlayedChoThisTurn(player);
            d["ki"] = MelodyCmd.PlayedKiThisTurn(player);
            d["melody_cards"] = MelodyCmd.MelodyCardsPlayedThisTurn(player);
            d["arrows_shot"] = h.OfType<ArrowShotEntry>().Count(e => e.Actor == player.Creature);
            d["energy_spent"] = h.OfType<EnergySpentEntry>().Where(e => e.Actor == player.Creature).Sum(e => e.Amount);
        }
        catch { }
        return d;
    }

    public static string CanonicalJson(Dictionary<string, object?> d) => System.Text.Json.JsonSerializer.Serialize(d);

    public static int CardsPlayedThisTurn(ICombatState cs, Player player)
    {
        try
        {
            return CombatManager.Instance.History.CardPlaysStarted.Count(e => e.HappenedThisTurn(cs) && e.CardPlay.Player == player);
        }
        catch { return 0; }
    }

    #endregion

    #region Map

    /// <summary>The node list of the current act (rows in order, then the start and boss nodes), cached per map.</summary>
    public static List<MapPoint> MapNodes(RunState? rs, ObsState st)
    {
        var map = rs?.Map;
        if (map == null) return st.MapNodes = new();
        if (ReferenceEquals(map, st.MapKey)) return st.MapNodes;
        var nodes = new List<MapPoint>();
        for (int row = 0; row < map.GetRowCount(); row++)
            foreach (var p in map.GetPointsInRow(row))
                if (p != null) nodes.Add(p);
        foreach (var extra in new[] { map.StartingMapPoint, map.BossMapPoint, map.SecondBossMapPoint })
            if (extra != null && !nodes.Contains(extra)) nodes.Add(extra);
        st.MapKey = map;
        return st.MapNodes = nodes;
    }

    /// <summary>The index of a map point in <see cref="MapNodes"/> (by coordinate): a map candidate's <c>node</c>.</summary>
    public static int NodeIndex(RunState? rs, ObsState st, MapPoint p) =>
        MapNodes(rs, st).FindIndex(n => n.coord.col == p.coord.col && n.coord.row == p.coord.row);

    public static Dictionary<string, object?> Map(RunState? rs, ObsState st)
    {
        var nodes = MapNodes(rs, st);
        var index = new Dictionary<MapPoint, int>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < nodes.Count; i++) index[nodes[i]] = i;
        var edges = new List<int[]>();
        for (int i = 0; i < nodes.Count; i++)
            foreach (var ch in nodes[i].Children ?? Enumerable.Empty<MapPoint>())
                if (index.TryGetValue(ch, out var j)) edges.Add(new[] { i, j });
        var map = rs?.Map;
        // The second boss follows the first one directly (NMapScreen draws that link).
        if (map?.SecondBossMapPoint != null && index.TryGetValue(map.BossMapPoint, out var b1) && index.TryGetValue(map.SecondBossMapPoint, out var b2))
            if (!edges.Any(e => e[0] == b1 && e[1] == b2)) edges.Add(new[] { b1, b2 });
        var cur = rs?.CurrentMapCoord;
        int curIdx = -1;
        if (cur.HasValue) curIdx = nodes.FindIndex(p => p.coord.col == cur.Value.col && p.coord.row == cur.Value.row);
        return new Dictionary<string, object?>
        {
            ["nodes"] = nodes.Select(p =>
            {
                var n = new Dictionary<string, object?>
                {
                    ["c"] = (int)p.coord.col,
                    ["r"] = (int)p.coord.row,
                    ["t"] = p.PointType.ToString(),
                };
                var quests = MapQuests(p);
                if (quests != null) n["q"] = quests;
                return n;
            }).ToList(),
            ["edges"] = edges,
            ["cur"] = curIdx,
        };
    }

    private static readonly PropertyInfo? _questsProp = typeof(MapPoint).GetProperty("Quests");

    /// <summary>Quest markers on a node (Fur Coat, Spoils Map): ids of the models that placed them.</summary>
    private static List<string>? MapQuests(MapPoint p)
    {
        try
        {
            if (_questsProp?.GetValue(p) is System.Collections.IEnumerable qs)
            {
                List<string>? l = null;
                foreach (var q in qs)
                    if (q != null) (l ??= new()).Add(q is AbstractModel m ? m.Id.ToString() : q.ToString() ?? "?");
                return l;
            }
        }
        catch { }
        return null;
    }

    /// <summary>NMapScreen.RecalculateTravelability + MapTravel.GetTravelablePointsFrom (free travel: Winged Boots, Flight).</summary>
    public static List<MapPoint> TravelablePoints(RunState rs)
    {
        var map = rs.Map;
        if (map == null) return new();
        if (!Hook.ShouldProceedToNextMapPoint(rs)) return new();
        var visited = rs.VisitedMapCoords;
        if (visited.Count == 0) return new() { map.StartingMapPoint };
        var last = visited[^1];
        if (map.SecondBossMapPoint != null && last == map.BossMapPoint.coord) return new() { map.SecondBossMapPoint };
        if (last.row == map.GetRowCount() - 1) return new() { map.BossMapPoint };
        var pt = map.GetPoint(last);
        if (pt != null) return MapTravel.GetTravelablePointsFrom(rs, pt).ToList();
        return new() { map.StartingMapPoint };
    }

    #endregion

    #region ctx (per decision)

    /// <summary>card_select: the selection's bounds, the picked cards and what the entry point was asked for.</summary>
    public static Dictionary<string, object?> SelectionCtx(int min, int max, IEnumerable<CardModel> picked, string kind, bool cancelable,
                                                           string? source, string? pile, string? prompt, ISet<object>? played)
    {
        var ctx = new Dictionary<string, object?>();
        ctx["min"] = min;
        ctx["max"] = max;
        ctx["picked"] = picked.Select(c => Card(c, played)).ToList();
        ctx["sel"] = kind;
        if (cancelable) ctx["cancelable"] = true;
        if (source != null) ctx["source"] = source;
        if (pile != null) ctx["pile"] = pile;
        if (prompt != null) ctx["prompt"] = prompt;
        return ctx;
    }

    /// <summary>card_reward: the alternatives the screen shows.</summary>
    public static Dictionary<string, object?> CardRewardCtx(IEnumerable<CardRewardAlternative> alts) =>
        new() { ["alts"] = alts.Select(a => a.OptionId).ToList() };

    /// <summary>rewards: terminal screen, must-take-all, and the potion rewards a full belt blocks (shown on screen,
    /// takeable after discarding a potion). Declined rewards are left out like taken ones.</summary>
    public static Dictionary<string, object?> RewardsCtx(Player player, RewardsSet set, bool terminal, Func<Reward, bool> declined)
    {
        var ctx = new Dictionary<string, object?>();
        ctx["terminal"] = terminal;
        if (set.DisallowSkipping) ctx["must_take_all"] = true;
        var blocked = set.Rewards.OfType<PotionReward>()
            .Where(r => !r.SuccessfullySelected && !declined(r) && r.Potion != null && !CanReceivePotion(player, r.Potion))
            .Select(r => r.Potion!.Id.ToString()).ToList();
        if (blocked.Count > 0) ctx["blocked"] = blocked;
        return ctx;
    }

    public static Dictionary<string, object?> SphereCtx(CrystalSphereMinigame g)
    {
        var cells = new List<Dictionary<string, object?>>();
        for (int x = 0; x < g.GridSize.X; x++)
            for (int y = 0; y < g.GridSize.Y; y++)
            {
                var c = g.cells[x, y];
                // Only what the screen shows: fog state, and the item under a cleared cell.
                cells.Add(new Dictionary<string, object?>
                {
                    ["x"] = x, ["y"] = y, ["hidden"] = c.IsHidden,
                    ["item"] = !c.IsHidden && c.Item != null ? c.Item.GetType().Name : null,
                });
            }
        return new Dictionary<string, object?>
        {
            ["divinations"] = g.DivinationCount,
            ["w"] = g.GridSize.X,
            ["h"] = g.GridSize.Y,
            ["cells"] = cells,
        };
    }

    public static Dictionary<string, object?> TreasureCtx(bool opened) => new() { ["opened"] = opened };

    public static Dictionary<string, object?> EventCtx(EventModel? ev)
    {
        var ctx = new Dictionary<string, object?>();
        ctx["event"] = ev?.Id.ToString();
        var sv = StringVars(ev?.DynamicVars);
        if (sv != null) ctx["sv"] = sv;
        return ctx;
    }

    public static Dictionary<string, object?> ShopCtx(AbstractRoom? room, Player player, ISet<object>? played)
    {
        var ctx = new Dictionary<string, object?>();
        if (room is MerchantRoom mr)
        {
            ctx["removal_cost"] = RunManager.Instance.MerchantSynchronizer.GetLocalInventory()?.CardRemovalEntry?.Cost ?? -1;
            if (RunManager.Instance.MerchantSynchronizer.GetLocalInventory() is { } sinv) ctx["unavailable"] = UnavailableStock(sinv, player, played);
        }
        return ctx;
    }

    public static Dictionary<string, object?> FakeMerchantCtx(EventModel? ev, Player player, ISet<object>? played)
    {
        var ctx = new Dictionary<string, object?>();
        ctx["event"] = ev?.Id.ToString();
        if (ev is FakeMerchant fm) ctx["unavailable"] = UnavailableStock(fm.Inventory, player, played);
        return ctx;
    }

    public static Dictionary<string, object?> RestCtx(bool done)
    {
        var ctx = new Dictionary<string, object?>();
        if (done) ctx["done"] = true;
        return ctx;
    }

    /// <summary>Stocked shelf items that are not candidates (too expensive, or a potion that cannot be procured):
    /// the player sees them, and they bear on discarding a potion, throwing a Foul Potion or saving gold.</summary>
    public static List<Dictionary<string, object?>> UnavailableStock(MerchantInventory inv, Player player, ISet<object>? played)
    {
        var list = new List<Dictionary<string, object?>>();
        try
        {
            foreach (var e in (inv.CharacterCardEntries ?? Enumerable.Empty<MerchantCardEntry>()).Concat(inv.ColorlessCardEntries ?? Enumerable.Empty<MerchantCardEntry>()))
                if (e.IsStocked && !e.EnoughGold && e.CreationResult?.Card is { } card)
                    list.Add(new() { ["k"] = "card", ["card"] = Card(card, played), ["price"] = e.Cost });
            foreach (var e in inv.RelicEntries)
                if (e.IsStocked && !e.EnoughGold && e.Model != null)
                    list.Add(new() { ["k"] = "relic", ["id"] = e.Model.Id.ToString(), ["price"] = e.Cost });
            foreach (var e in inv.PotionEntries)
                if (e.IsStocked && e.Model != null && (!e.EnoughGold || !CanBuyPotion(player, e.Model)))
                    list.Add(new() { ["k"] = "potion", ["id"] = e.Model.Id.ToString(), ["price"] = e.Cost });
        }
        catch { }
        return list;
    }

    #endregion

    #region Candidate descriptors and legality shared by both

    /// <summary>A card play: hand index, target index (-1: none) and the values the card shows while aimed (a single
    /// enemy target), or in all/random-enemy targeting.</summary>
    public static Dictionary<string, object?> PlayDesc(CardModel c, int hand, int tgt, Creature? target)
    {
        var d = new Dictionary<string, object?> { ["k"] = "play", ["hand"] = hand, ["tgt"] = tgt };
        if (target != null) d["pv"] = TargetPreview(c, target);
        else if (c.TargetType is TargetType.AllEnemies or TargetType.RandomEnemy) d["pv"] = MultiTargetPreview(c);
        return d;
    }

    public static Dictionary<string, object?> PotionDesc(int slot, int tgt) => new() { ["k"] = "potion", ["slot"] = slot, ["tgt"] = tgt };

    public static Dictionary<string, object?> DiscardPotionDesc(int slot) => new() { ["k"] = "discard_potion", ["slot"] = slot };

    public static Dictionary<string, object?> PickDesc(int src, CardModel card, ISet<object>? played) =>
        new() { ["k"] = "pick", ["src"] = src, ["card"] = Card(card, played) };

    public static Dictionary<string, object?> BundleDesc(IEnumerable<CardModel> cards, ISet<object>? played) =>
        new() { ["k"] = "bundle", ["cards"] = cards.Select(c => Card(c, played)).ToList() };

    public static Dictionary<string, object?> RewardCardDesc(CardModel card, ISet<object>? played) =>
        new() { ["k"] = "reward_card", ["card"] = Card(card, played) };

    /// <summary>A card reward alternative: Skip, REROLL (Driftwood), SACRIFICE (Pael's Wing), any other id.</summary>
    public static Dictionary<string, object?> AltDesc(string optionId)
    {
        var k = optionId switch { "Skip" => "skip", "REROLL" => "reroll", "SACRIFICE" => "sacrifice", _ => "alt" };
        var desc = new Dictionary<string, object?> { ["k"] = k };
        if (k == "alt") desc["key"] = optionId;
        return desc;
    }

    public static Dictionary<string, object?> RewardDesc(Reward r, int i, ISet<object>? played)
    {
        var d = new Dictionary<string, object?> { ["k"] = "reward", ["idx"] = i };
        switch (r)
        {
            case GoldReward g: d["rt"] = "Gold"; d["n"] = g.Amount; break;
            case PotionReward p: d["rt"] = "Potion"; d["id"] = p.Potion?.Id.ToString(); break;
            case RelicReward rr: d["rt"] = "Relic"; d["id"] = rr.Relic?.Id.ToString(); d["rar"] = rr.Rarity.ToString(); break;
            case CardReward cr:
                // Everything the card reward screen shows, so that opening it is never needed just to look.
                d["rt"] = "Card";
                d["n"] = cr.Cards.Count();
                d["cards"] = cr.Cards.Select(c => Card(c, played)).ToList();
                var alts = CardRewardAlts(cr);
                if (alts != null) d["alts"] = alts;
                break;
            case SpecialCardReward sc:
                d["rt"] = "SpecialCard";
                if (_specialCard.GetValue(sc) is CardModel card) d["card"] = Card(card, played);
                break;
            case CardRemovalReward: d["rt"] = "RemoveCard"; break;
            case LinkedRewardSet ls: d["rt"] = "Linked"; d["n"] = ls.Rewards.Count; break;
            default: d["rt"] = r.GetType().Name; break;
        }
        return d;
    }

    /// <summary>The alternatives the screen would show (CardRewardAlternative.Generate: Skip, REROLL, hook additions such as SACRIFICE).</summary>
    public static List<string>? CardRewardAlts(CardReward cr)
    {
        // Generate is pure (it only builds the list); it throws for more than two alternatives, as opening would.
        try { return CardRewardAlternative.Generate(cr).Select(a => a.OptionId).ToList(); }
        catch (InvalidOperationException) { return null; }
    }

    private static readonly FieldInfo _specialCard = typeof(SpecialCardReward).GetField("_card", BindingFlags.Instance | BindingFlags.NonPublic)!;

    /// <summary>An event option: text key, the event's numbers plus the option relic's, the relic and its own state
    /// (which character's pool a Sea Glass offers), hover-tip models, proceed.</summary>
    public static Dictionary<string, object?> OptionDesc(EventModel ev, EventOption o)
    {
        var desc = new Dictionary<string, object?>
        {
            ["k"] = "option",
            ["key"] = o.TextKey,
            ["v"] = OptionVars(ev, o),
        };
        if (o.Relic != null)
        {
            desc["relic"] = o.Relic.Id.ToString();
            var rs = OwnState(o.Relic, typeof(RelicModel));  // e.g. which character's pool a Sea Glass offers
            if (rs.Count > 0) desc["relic_d"] = rs;
        }
        var tips = TipIds(o);
        if (tips != null) desc["tips"] = tips;
        if (o.IsProceed) desc["proceed"] = true;
        return desc;
    }

    private static readonly PropertyInfo? _disableOnChosen = typeof(EventOption).GetProperty("DisableOnChosen", BindingFlags.Instance | BindingFlags.NonPublic);

    /// <summary>EventOption.Chosen does nothing for an option that was chosen and disables on choice.</summary>
    public static bool IsSpentOption(EventOption o)
    {
        try { return o.WasChosen && _disableOnChosen?.GetValue(o) is true; }
        catch { return false; }
    }

    /// <summary>Numeric event vars (the description's numbers) plus the option relic's vars (Neow / ancients).</summary>
    public static Dictionary<string, object?> OptionVars(EventModel ev, EventOption o)
    {
        var v = new Dictionary<string, object?>();
        try
        {
            foreach (var dv in ev.DynamicVars.Values)
                if (dv is not StringVar && dv.Name != "EntrantNumber")  // Trial's flavour number, from Rng.Chaotic
                    v[dv.Name] = (double)dv.BaseValue;
        }
        catch { }
        try
        {
            var relic = o.Relic;
            if (relic == null && o.TextKey != null)
            {
                var parts = o.TextKey.Split('.');
                relic = ModelDb.GetByIdOrNull<RelicModel>(new ModelId("RELIC", parts[^1]));
            }
            if (relic != null)
                foreach (var dv in relic.DynamicVars.Values)
                    if (dv is not StringVar) v[dv.Name] = (double)dv.BaseValue;
        }
        catch { }
        return v;
    }

    /// <summary>Hover tips of an option (Relic Trader, Ranwid, The Future of Potions...): the models they show.</summary>
    public static List<string>? TipIds(EventOption o)
    {
        List<string>? l = null;
        try
        {
            foreach (var t in o.HoverTips ?? Enumerable.Empty<MegaCrit.Sts2.Core.HoverTips.IHoverTip>())
            {
                var m = t?.CanonicalModel;
                if (m != null) (l ??= new()).Add(m.Id.ToString());
            }
        }
        catch { }
        return l;
    }

    /// <summary>A rest site option; the heal option carries the amount on its button (before the max-HP cap).</summary>
    public static Dictionary<string, object?> RestDesc(RestSiteOption o, Player player)
    {
        var rd = new Dictionary<string, object?> { ["k"] = "rest", ["key"] = o.OptionId };
        if (o is HealRestSiteOption)
            try { rd["heal"] = (double)HealRestSiteOption.GetHealAmount(player); } catch { }
        return rd;
    }

    public static Dictionary<string, object?> BuyCardDesc(CardModel card, int price, ISet<object>? played) =>
        new() { ["k"] = "buy_card", ["card"] = Card(card, played), ["price"] = price };

    public static Dictionary<string, object?> BuyRelicDesc(RelicModel relic, int price) =>
        new() { ["k"] = "buy_relic", ["id"] = relic.Id.ToString(), ["price"] = price };

    public static Dictionary<string, object?> BuyPotionDesc(PotionModel potion, int price) =>
        new() { ["k"] = "buy_potion", ["id"] = potion.Id.ToString(), ["price"] = price };

    public static Dictionary<string, object?> RemoveDesc(int price) => new() { ["k"] = "remove", ["price"] = price };

    public static Dictionary<string, object?> TreasureDesc(RelicModel relic) => new() { ["k"] = "treasure", ["id"] = relic.Id.ToString() };

    public static Dictionary<string, object?> SphereDesc(string tool, int x, int y) =>
        new() { ["k"] = "sphere", ["tool"] = tool, ["x"] = x, ["y"] = y };

    public static Dictionary<string, object?> MapDesc(int node) => new() { ["k"] = "map", ["node"] = node };

    /// <summary>A candidate without data: end_turn, confirm, skip, proceed, open, leave.</summary>
    public static Dictionary<string, object?> Simple(string k) => new() { ["k"] = k };

    /// <summary>
    /// A potion reward can be taken unless PotionCmd.TryToProcure fails with TooFull (procuring allowed, no free slot):
    /// PotionReward.OnSelect treats NotAllowed (Sozu) as taken.
    /// </summary>
    public static bool CanReceivePotion(Player player, PotionModel potion) =>
        player.HasOpenPotionSlots || !Hook.ShouldProcurePotion(player.RunState, player.Creature.CombatState, potion, player);

    /// <summary>
    /// MerchantPotionEntry.OnTryPurchase only succeeds when TryToProcure does (no TooFull, no NotAllowed such as Sozu);
    /// otherwise the click only shows the failure message and changes nothing.
    /// </summary>
    public static bool CanBuyPotion(Player player, PotionModel potion) =>
        player.HasOpenPotionSlots && Hook.ShouldProcurePotion(player.RunState, player.Creature.CombatState, potion, player);

    /// <summary>NPotionPopup's use-button rules; <paramref name="pendingDiscard"/>: NPotionHolder.DisableUntilPotionRemoved
    /// after a discard was queued; <paramref name="inSelectionScreen"/>: NPlayerHand.IsInCardSelection / an
    /// ICardSelector overlay is open (read only for combat-only potions).</summary>
    public static bool CanUsePotion(Player me, PotionModel p, bool pendingDiscard, Func<bool> inSelectionScreen)
    {
        if (p.IsQueued || pendingDiscard || me.Creature.IsDead || !me.CanUseOrRemovePotions || !p.PassesCustomUsabilityCheck)
            return false;
        switch (p.Usage)
        {
            case PotionUsage.AnyTime: return true;
            case PotionUsage.CombatOnly:
                return CombatManager.Instance.IsInProgress && me.Creature.CombatState?.CurrentSide == me.Creature.Side
                       && me.Creature.IsAlive && !inSelectionScreen() && !CombatManager.Instance.PlayerActionsDisabled;
            default: return false;
        }
    }

    public static bool CanDiscardPotion(Player me, PotionModel p, bool pendingDiscard) =>
        !p.IsQueued && !pendingDiscard && !me.Creature.IsDead && me.CanUseOrRemovePotions;

    #endregion
}
