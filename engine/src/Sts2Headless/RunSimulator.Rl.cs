using System.Diagnostics;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Rewards;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using Sts2Obs;

namespace Sts2Headless;

/// <summary>
/// RL interface (engine/RL_PROTOCOL.md). Every response carries a structured observation read straight from
/// live game objects plus the full list of legal actions; Python answers with an index. Each candidate is a
/// descriptor plus the exact UI call it stands for (<see cref="RlCand.Exec"/>); the candidate lists are built
/// from the current <see cref="Decision"/> with the UI's own enable rules, so nothing the client would not let
/// the player do is offered and nothing it would is left out. The observation and the descriptors' data are
/// <see cref="ObsShared"/> (engine/src/Shared/), which the real-client parity mod compiles too.
/// </summary>
public partial class RunSimulator
{
    private sealed record RlCand(Dictionary<string, object?> Desc, Action Exec);

    private List<RlCand>? _rlCands;
    private Decision? _rlDecision;
    private int _rlSteps;
    private long _rlExecTicks, _rlObsTicks;

    private void ResetRlState()
    {
        _rlCands = null;
        _rlDecision = null;
        _rlSteps = 0;
        _rlExecTicks = _rlObsTicks = 0;
        _obs.ResetMap();
    }

    public Dictionary<string, object?> RlStart(string character, int ascension, string? seed)
    {
        var d = StartRun(character, ascension, seed);
        return RlRespond(d);
    }

    public Dictionary<string, object?> RlAct(int i)
    {
        if (_rlCands == null || i < 0 || i >= _rlCands.Count)
            return ErrorResponse($"rl_act: invalid candidate index {i} (have {_rlCands?.Count ?? 0})");
        var c = _rlCands[i];
        _rlCands = null;
        _rlSteps++;
        // A pick only extends a card selection (the screen's own state); every other input on a screen changes the game.
        var kind = c.Desc.GetValueOrDefault("k") as string;
        if (_rlDecision?.Prompt is { } screen && kind != "pick") screen.Changes++;
        if (_rlDecision?.Prompt is RewardsPrompt rewards && kind is "potion" or "discard_potion") rewards.Held.Clear();
        Diag.Note("input: " + string.Join(",", c.Desc.Select(kv => kv.Value is Dictionary<string, object?> sub ? $"{kv.Key}={sub.GetValueOrDefault("id")}"
                                                                    : kv.Value is string or int or bool ? $"{kv.Key}={kv.Value}" : null).Where(x => x != null)));
        var t0 = Stopwatch.GetTimestamp();
        Decision d;
        try
        {
            c.Exec();
            d = Decide();
        }
        catch (Exception e)
        {
            d = Err(FaultText(e));
        }
        _rlExecTicks += Stopwatch.GetTimestamp() - t0;
        return RlRespond(d);
    }

    /// <summary>Re-observe the current decision without acting (debug / after a debug command).</summary>
    internal Dictionary<string, object?> RlObserve() => RlRespond(Decide());

    private static Dictionary<string, object?> ErrorResponse(string msg) =>
        new() { ["type"] = "error", ["message"] = msg };

    private Dictionary<string, object?> RlRespond(Decision d)
    {
        _rlCands = null;
        _rlDecision = d;
        if (d.Kind == "error" || _runState == null)
            return ErrorResponse(d.Error ?? "no run in progress");
        var player = _runState.Players[0];
        var t0 = Stopwatch.GetTimestamp();
        Dictionary<string, object?> obs;
        List<RlCand> cands;
        try
        {
            obs = RlObs(player, d);
            cands = d.Done ? new List<RlCand>() : RlCandidates(player, d);
        }
        catch (Exception e)
        {
            return ErrorResponse($"observation failed: {e.GetType().Name}: {e.Message} | {Shorten(e.ToString())}");
        }
        _rlObsTicks += Stopwatch.GetTimestamp() - t0;
        if (!d.Done && cands.Count == 0)
            return ErrorResponse($"integrity: decision {d.Kind} has no legal action: {DescribeState()}");
        _rlCands = cands;
        return new Dictionary<string, object?>
        {
            ["type"] = "rl",
            ["done"] = d.Done,
            ["victory"] = d.Done ? d.Victory : null,
            ["decision"] = d.Kind,
            ["obs"] = obs,
            ["cands"] = cands.Select(c => c.Desc).ToList(),
            ["info"] = new Dictionary<string, object?>
            {
                ["act"] = _runState.CurrentActIndex + 1,
                ["floor"] = _runState.ActFloor,
                ["total_floor"] = _runState.TotalFloor,
                ["hp"] = player.Creature?.CurrentHp ?? 0,
                ["max_hp"] = player.Creature?.MaxHp ?? 0,
                ["gold"] = player.Gold,
                ["steps"] = _rlSteps,
                ["save"] = SaveInfo(),
                ["abandoned"] = d.Abandoned,
                ["t_exec_ms"] = _rlExecTicks * 1000.0 / Stopwatch.Frequency,
                ["t_obs_ms"] = _rlObsTicks * 1000.0 / Stopwatch.Frequency,
                ["warn"] = null,
            },
        };
    }

    #region Observation

    // The observation itself is Sts2Obs.ObsShared (engine/src/Shared/ObsShared.cs), shared with the parity mod. What
    // persists between decisions (cards played this turn, the act's map nodes) is kept here.
    private readonly ObsState _obs = new();

    private Dictionary<string, object?> RlObs(Player player, Decision d) =>
        ObsShared.Observation(_runState!, player, _obs, () => RlCtx(player, d));

    private Dictionary<string, object?> RlCard(CardModel c, bool inHand = false) => ObsShared.Card(c, _obs.PlayedThisTurn, inHand);

    private List<MapPoint> RlMapNodes() => ObsShared.MapNodes(_runState, _obs);

    private Dictionary<string, object?> RlCtx(Player player, Decision d)
    {
        switch (d.Prompt)
        {
            case CardSelectPrompt cs:
                return ObsShared.SelectionCtx(cs.Min, cs.Max, cs.Picked.Select(i => cs.Options[i]), cs.Ctx.Kind, cs.Cancelable,
                                              cs.Ctx.Source, cs.Ctx.Pile, cs.Ctx.PromptKey, _obs.PlayedThisTurn);
            case CardRewardPrompt cr:
                return ObsShared.CardRewardCtx(cr.Alts);
            case RewardsPrompt rp:
                return ObsShared.RewardsCtx(player, rp.Set, rp.Terminal, rp.Declined.Contains);
            case SpherePrompt sp:
                return ObsShared.SphereCtx(sp.Game);
            case ChestPrompt:
                return ObsShared.TreasureCtx(opened: false);
            case TreasurePrompt:
                return ObsShared.TreasureCtx(opened: true);
        }
        switch (d.Kind)
        {
            case "event_choice": return ObsShared.EventCtx(LocalEvent());
            case "shop": return ObsShared.ShopCtx(_runState?.CurrentRoom, player, _obs.PlayedThisTurn);
            case "fake_merchant": return ObsShared.FakeMerchantCtx(LocalEvent(), player, _obs.PlayedThisTurn);
            case "rest_site": return ObsShared.RestCtx(RestSucceeded);
        }
        return new Dictionary<string, object?>();
    }

    #endregion

    #region Candidates

    private List<RlCand> RlCandidates(Player player, Decision d)
    {
        var cands = new List<RlCand>();
        void Add(Dictionary<string, object?> desc, Action exec) => cands.Add(new RlCand(desc, exec));
        var played = _obs.PlayedThisTurn;

        switch (d.Prompt)
        {
            case CardSelectPrompt cs:
                CardSelectCands(cs, Add);
                break;
            case BundlePrompt bp:
                for (int i = 0; i < bp.Bundles.Count; i++)
                {
                    int bi = i;
                    Add(ObsShared.BundleDesc(bp.Bundles[i], played), () => bp.Choose(bi));
                }
                break;
            case CardRewardPrompt cr:
                for (int i = 0; i < cr.Options.Count; i++)
                {
                    int ci = i;
                    Add(ObsShared.RewardCardDesc(cr.Options[i].Card, played), () => cr.Choose(ci));
                }
                for (int a = 0; a < cr.Alts.Count; a++)
                {
                    int ai = cr.Options.Count + a;
                    var desc = ObsShared.AltDesc(cr.Alts[a].OptionId);
                    Action exec = () => cr.Choose(ai);
                    // Skip (EndSelectionAndDoNotCompleteReward: CardReward.OnSelect returns false, the reward stays) on a
                    // screen opened from a rewards screen: see BackOut.
                    if (cr.FromReward is { } o && cr.Alts[a].AfterSelected == PostAlternateCardRewardAction.EndSelectionAndDoNotCompleteReward)
                    {
                        var back = BackOut(cr, o, () => cr.Choose(ai));
                        if (back == null) continue;
                        exec = back;
                    }
                    Add(desc, exec);
                }
                break;
            case RewardsPrompt rp:
                RewardsCands(player, rp, Add);
                break;
            case ChestPrompt chest:
                Add(ObsShared.Simple("open"), () => chest.Tcs.TrySetResult());
                break;
            case TreasurePrompt tp:
                for (int i = 0; i < tp.Relics.Count; i++)
                {
                    int ri = i;
                    Add(ObsShared.TreasureDesc(tp.Relics[i]), () => tp.Tcs.TrySetResult(ri));
                }
                Add(ObsShared.Simple("skip"), () => tp.Tcs.TrySetResult(null));
                break;
            case SpherePrompt sp:
                SphereCands(sp, Add);
                break;
            case null:
                RoomCands(player, d, Add);
                break;
        }
        // The potion belt is usable on top of every screen (NPotionPopup rules); in combat_play the potion
        // candidates are emitted in RoomCands to keep the v0 order (plays, potions, end_turn).
        if (d.Kind != "combat_play") PotionCands(player, Add);
        return cands;
    }

    private void RoomCands(Player player, Decision d, Action<Dictionary<string, object?>, Action> Add)
    {
        var played = _obs.PlayedThisTurn;
        switch (d.Kind)
        {
            case "combat_play":
            {
                var pcs = player.PlayerCombatState!;
                var targets = CombatEnemies();
                var hand = pcs.Hand?.Cards?.ToList() ?? new();
                for (int i = 0; i < hand.Count; i++)
                {
                    var c = hand[i];
                    int hi = i;
                    switch (c.TargetType)
                    {
                        case TargetType.AnyEnemy:
                            for (int t = 0; t < targets.Count; t++)
                            {
                                var target = targets[t];
                                if (Targetable(target) && c.CanPlayTargeting(target))
                                    Add(ObsShared.PlayDesc(c, hi, t, target), () => PlayCard(c, target));
                            }
                            break;
                        case TargetType.AnyAlly:
                            // NTargetManager: another living player (none in single player).
                            break;
                        default:
                            if (c.CanPlayTargeting(null))
                                Add(ObsShared.PlayDesc(c, hi, -1, null), () => PlayCard(c, null));
                            break;
                    }
                }
                PotionCands(player, Add);
                Add(ObsShared.Simple("end_turn"), EndTurn);
                break;
            }
            case "map_select":
            {
                var nodes = RlMapNodes();
                foreach (var pt in TravelablePoints())
                {
                    var p = pt;
                    int idx = nodes.FindIndex(n => n.coord.col == p.coord.col && n.coord.row == p.coord.row);
                    Add(ObsShared.MapDesc(idx), () => TravelTo(p));
                }
                break;
            }
            case "event_choice":
            {
                var ev = LocalEvent()!;
                var opts = ev.CurrentOptions;
                for (int i = 0; i < opts.Count; i++)
                {
                    var o = opts[i];
                    if (o.IsLocked || ObsShared.IsSpentOption(o)) continue;
                    int oi = i;
                    Add(ObsShared.OptionDesc(ev, o), () => ChooseEventOption(oi));
                }
                break;
            }
            case "rest_site":
            {
                var room = (RestSiteRoom)_runState!.CurrentRoom!;
                var opts = room.Options;
                for (int i = 0; i < opts.Count; i++)
                {
                    if (!opts[i].IsEnabled) continue;
                    int oi = i;
                    Add(ObsShared.RestDesc(opts[i], player), () => ChooseRestOption(oi));
                }
                if (RestSucceeded) Add(ObsShared.Simple("leave"), LeaveRoom);
                break;
            }
            case "shop":
            {
                if (_runState!.CurrentRoom is MerchantRoom mr && RunManager.Instance.MerchantSynchronizer.GetLocalInventory() is { } inv)
                {
                    var cardEntries = inv.CharacterCardEntries.Concat(inv.ColorlessCardEntries).ToList();
                    foreach (var e in cardEntries)
                        if (e.IsStocked && e.EnoughGold && e.CreationResult?.Card is { } card)
                        {
                            var entry = e;
                            Add(ObsShared.BuyCardDesc(card, e.Cost, played), () => Purchase(entry, inv));
                        }
                    foreach (var e in inv.RelicEntries)
                        if (e.IsStocked && e.EnoughGold && e.Model != null)
                        {
                            var entry = e;
                            Add(ObsShared.BuyRelicDesc(e.Model, e.Cost), () => Purchase(entry, inv));
                        }
                    foreach (var e in inv.PotionEntries)
                        if (e.IsStocked && e.EnoughGold && e.Model != null && ObsShared.CanBuyPotion(player, e.Model))
                        {
                            var entry = e;
                            Add(ObsShared.BuyPotionDesc(e.Model, e.Cost), () => Purchase(entry, inv));
                        }
                    var removal = inv.CardRemovalEntry;
                    if (removal != null && !removal.Used && removal.EnoughGold && removal.IsStocked)
                        Add(ObsShared.RemoveDesc(removal.Cost), () => Purchase(removal, inv));
                }
                Add(ObsShared.Simple("leave"), LeaveRoom);
                break;
            }
            case "fake_merchant":
            {
                var fm = (FakeMerchant)LocalEvent()!;
                var inv = fm.Inventory;
                foreach (var e in inv.RelicEntries)
                    if (e.IsStocked && e.EnoughGold && e.Model != null)
                    {
                        var entry = e;
                        Add(ObsShared.BuyRelicDesc(e.Model, e.Cost), () => Purchase(entry, inv));
                    }
                Add(ObsShared.Simple("leave"), LeaveRoom);
                break;
            }
        }
    }

    /// <summary>
    /// A selection is exposed as sequential picks (any order, no repeats) plus confirm once MinSelect is reached;
    /// reaching MaxSelect submits. Cancel ("skip") only where the screen has a cancel/skip button: Cancelable prefs, or
    /// FromChooseACardScreen with canSkip (its MinSelect is then 0, so skip == confirm with nothing picked).
    /// </summary>
    private void CardSelectCands(CardSelectPrompt cs, Action<Dictionary<string, object?>, Action> Add)
    {
        bool chooseACard = cs.Ctx.Kind == "choose_a_card";
        if (cs.Picked.Count < cs.Max)
            for (int i = 0; i < cs.Options.Count; i++)
            {
                if (cs.Picked.Contains(i)) continue;
                int si = i;
                Add(ObsShared.PickDesc(i, cs.Options[i], _obs.PlayedThisTurn), () =>
                {
                    cs.Picked.Add(si);
                    if (cs.Picked.Count >= cs.Max) cs.Submit();
                });
            }
        if (chooseACard)
        {
            if (cs.Min == 0) Add(ObsShared.Simple("skip"), cs.Cancel);
            return;
        }
        if (cs.Picked.Count >= cs.Min && (cs.Picked.Count < cs.Max || cs.Max == 0) && !(cs.Min == 0 && cs.Picked.Count == 0 && cs.Cancelable))
            Add(ObsShared.Simple("confirm"), cs.Submit);
        if (!cs.Cancelable) return;
        // Smith, Cook and the merchant's removal (RunSimulator.TagOrigin): cancelling returns to the rest site / shop with
        // the option still there, and choosing it again reproduces this screen, so the cancel is never offered. Nothing
        // is lost: the only other inputs here are potions, which do the same when used at the rest site or the shop.
        if (cs.RoomService != null) return;
        if (cs.FromReward is { } o)
        {
            var back = BackOut(cs, o, cs.Cancel);
            if (back != null) Add(ObsShared.Simple("skip"), back);
            return;
        }
        Add(ObsShared.Simple("skip"), cs.Cancel);
    }

    /// <summary>
    /// Backing out of a reward's screen (card reward Skip, removal cancel) returns to the rewards screen with the reward
    /// still there. Before anything changed on the screen, that is the identical rewards screen: the back-out declines
    /// the reward instead (no longer offered; left behind at proceed through SkipLocalRewardsSet -> Reward.OnSkipped,
    /// exactly as when it is never opened). After a change (REROLL, a potion) it is a plain back-out, and the reward is
    /// held back until the game changes again (RewardsPrompt.Held). On a must-take-all screen backing out can only
    /// return to the screen that has to be completed, so it is not offered (null).
    /// </summary>
    private static Action? BackOut(Prompt screen, RewardOrigin o, Action back)
    {
        if (o.Screen.Set.DisallowSkipping) return null;
        if (screen.Changes == 0) return () => { o.Screen.Declined.Add(o.Reward); back(); };
        // The game changed on this screen, so rewards held back earlier would no longer reopen identically.
        return () => { o.Screen.Held.Clear(); o.Screen.Held.Add(o.Reward); back(); };
    }

    /// <summary>
    /// NRewardsScreen: every unclaimed reward button (except declined ones, see RewardsPrompt.Declined), and the proceed /
    /// skip button unless skipping is disallowed.
    /// </summary>
    private void RewardsCands(Player player, RewardsPrompt rp, Action<Dictionary<string, object?>, Action> Add)
    {
        var rewards = rp.Set.Rewards;
        bool allClaimed = true;
        var held = new List<int>();
        int offered = 0;
        for (int i = 0; i < rewards.Count; i++)
        {
            var r = rewards[i];
            if (r.SuccessfullySelected) continue;
            allClaimed = false;
            if (rp.Declined.Contains(r)) continue;
            if (r is PotionReward pr && pr.Potion != null && !ObsShared.CanReceivePotion(player, pr.Potion)) continue; // TooFull: the click does nothing
            if (rp.Held.Contains(r)) { held.Add(i); continue; }
            int ri = i;
            Add(ObsShared.RewardDesc(r, i, _obs.PlayedThisTurn), () => rp.Tcs!.TrySetResult(new RewardChoice(false, ri)));
            offered++;
        }
        // NRewardsScreen.TryEnableProceedButton: not while skipping is disallowed (unless everything is taken), and a
        // terminal screen also needs Hook.ShouldProceedToNextMapPoint.
        bool proceed = (!rp.Set.DisallowSkipping || allClaimed) && (!rp.Terminal || Hook.ShouldProceedToNextMapPoint(_runState!));
        // A held-back reward is still the way forward when nothing else is.
        if (offered == 0 && !proceed)
            foreach (int i in held)
            {
                int ri = i;
                Add(ObsShared.RewardDesc(rewards[i], i, _obs.PlayedThisTurn), () => rp.Tcs!.TrySetResult(new RewardChoice(false, ri)));
            }
        if (proceed) Add(ObsShared.Simple("proceed"), () => rp.Tcs!.TrySetResult(new RewardChoice(true, -1)));
    }

    private void SphereCands(SpherePrompt sp, Action<Dictionary<string, object?>, Action> Add)
    {
        var g = sp.Game;
        foreach (var tool in new[] { "small", "big" })
            for (int x = 0; x < g.GridSize.X; x++)
                for (int y = 0; y < g.GridSize.Y; y++)
                    if (g.cells[x, y].IsHidden)
                    {
                        var cell = g.cells[x, y];
                        var t = tool == "small" ? CrystalSphereMinigame.CrystalSphereToolType.Small : CrystalSphereMinigame.CrystalSphereToolType.Big;
                        Add(ObsShared.SphereDesc(tool, x, y), () => TaskHelper.RunSafely(SphereClick(sp, t, cell)));
                    }
    }

    /// <summary>NCrystalSphereScreen: the tool buttons call SetTool, a hidden cell's release calls CellClicked.</summary>
    private static async Task SphereClick(SpherePrompt sp, CrystalSphereMinigame.CrystalSphereToolType tool, CrystalSphereCell cell)
    {
        sp.Busy = true;
        try
        {
            sp.Game.SetTool(tool);
            await sp.Game.CellClicked(cell);
        }
        finally { sp.Busy = false; }
    }

    /// <summary>Use (targeting per NPotionHolder.UsePotion) and discard, for every filled slot the popup would enable.</summary>
    private void PotionCands(Player player, Action<Dictionary<string, object?>, Action> Add)
    {
        foreach (var p in _pendingDiscards.Where(p => !player.PotionSlots.Contains(p)).ToList()) _pendingDiscards.Remove(p);
        var slots = player.PotionSlots;
        bool inCombat = CombatManager.Instance.IsInProgress;
        for (int s = 0; s < slots.Count; s++)
        {
            var p = slots[s];
            if (p == null) continue;
            int slot = s;
            if (CanUsePotion(p))
            {
                if (p.TargetType == TargetType.AnyEnemy)
                {
                    if (inCombat)
                    {
                        var targets = CombatEnemies();
                        for (int t = 0; t < targets.Count; t++)
                        {
                            var target = targets[t];
                            if (Targetable(target) && p.IsValidTarget(target))
                                Add(ObsShared.PotionDesc(slot, t), () => UsePotion(p, target));
                        }
                    }
                }
                else if (p.TargetType == TargetType.AnyAlly)
                {
                    // Another player: none in single player.
                }
                else
                    Add(ObsShared.PotionDesc(slot, -1), () => UsePotion(p, null));
            }
            if (CanDiscardPotion(p))
                Add(ObsShared.DiscardPotionDesc(slot), () => DiscardPotion(p));
        }
    }

    #endregion
}
