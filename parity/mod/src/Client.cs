using System.Reflection;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Entities.Rewards;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Events.Custom;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Nodes.TreasureRooms;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using Sts2Obs;

namespace ParityOracle;

/// <summary>A legal action in the client: the descriptor (same schema as the headless `cands[i]`) plus how to do it.</summary>
public sealed record Cand(Dictionary<string, object?> Desc, Func<Task> Exec);

/// <summary>A stable decision point in the client.</summary>
public sealed class DecisionPoint
{
    public string Decision = "unknown";
    public Dictionary<string, object?> Obs = new();
    public List<Cand> Cands = new();
    public string? Note;
    /// <summary>The screen the decision is on, for the navigation bookkeeping (Nav.BeforeInput).</summary>
    public object? Screen;
}

/// <summary>
/// Reads the real client's state into the v1 RL protocol (engine/RL_PROTOCOL.md): which decision the screen shows,
/// the observation and the candidate list, and how each candidate is performed through the client's own UI path.
///
/// Rules for this file: a decision is only reported when the client actually shows it (screen up, buttons present,
/// executor idle, stable for a few frames); candidates come from the UI elements that are enabled (map points, reward
/// buttons, proceed buttons, option buttons) where the client has them and from the engine's legality code (ported
/// verbatim) where the UI computes legality the same way from model state (card play, potions); every exec is the
/// UI's click (ForceClick / the signal the node connects) or the exact call the node makes. Steps the engine has no
/// decision for (the synthetic PROCEED of a finished event, a rest site / chest with nothing left, ancient dialogue,
/// the abandon confirmation) are "auto" steps.
/// </summary>
public static class Client
{
    public static readonly OracleSelector Selector = new();
    private static IDisposable? _selectorScope;
    private static int _autoCooldownFrames;
    private static ulong _treasurePickedRoom;
    /// <summary>Potions whose discard is queued: NPotionHolder.DisableUntilPotionRemoved greys them out (engine _pendingDiscards).</summary>
    private static readonly HashSet<PotionModel> _pendingDiscards = new();

    const BindingFlags NP = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;

    public static object? FieldObj(object o, string name)
    {
        for (var t = o.GetType(); t != null; t = t.BaseType)
        {
            var f = t.GetField(name, NP | BindingFlags.DeclaredOnly);
            if (f != null) return f.GetValue(o);
        }
        return null;
    }

    public static T? Field<T>(object o, string name) where T : class => FieldObj(o, name) as T;

    private static object? Invoke(object o, string method, params object?[] args)
    {
        for (var t = o.GetType(); t != null; t = t.BaseType)
        {
            var m = t.GetMethod(method, NP | BindingFlags.DeclaredOnly);
            if (m != null) return m.Invoke(o, args);
        }
        throw new MissingMethodException(o.GetType().Name, method);
    }

    public static void InstallSelector()
    {
        try { _selectorScope?.Dispose(); } catch { }
        _selectorScope = null;
        Selector.Abort();
        if (RunManager.Instance.CardSelectorStack.LocalSelector == null)
            _selectorScope = RunManager.Instance.CardSelectorStack.UseLocalSelector(Selector);
        _treasurePickedRoom = 0;
        _pendingDiscards.Clear();
        Nav.Reset();
    }

    public static void UninstallSelector()
    {
        Selector.Abort();
        try { _selectorScope?.Dispose(); } catch { }
        _selectorScope = null;
    }

    public static Player? Me => Obs.Rs is { } rs ? LocalContext.GetMe(rs) ?? rs.Players[0] : null;

    public static bool ExecutorIdle(bool allowChoice = false)
    {
        try
        {
            var rm = RunManager.Instance;
            var running = rm.ActionExecutor.CurrentlyRunningAction;
            if (running != null && !(allowChoice && running.State == MegaCrit.Sts2.Core.Entities.Actions.GameActionState.GatheringPlayerChoice))
                return false;
            if (rm.ActionQueueSet.GetReadyAction() != null) return false;
            return true;
        }
        catch { return false; }
    }

    // ------------------------------------------------------------------ classification

    private enum Kind { Wait, Auto, Decision }

    private sealed class Cls
    {
        public Kind K = Kind.Wait;
        public string Decision = "";
        public Func<Task>? Auto;
        public string Note = "";
    }

    private static Cls Wait(string note) => new() { K = Kind.Wait, Note = note };
    private static Cls Auto(string note, Func<Task> f) => new() { K = Kind.Auto, Note = note, Auto = f };
    private static Cls Dec(string d) => new() { K = Kind.Decision, Decision = d };
    private static Task Done() => Task.CompletedTask;

    public static bool ReadyForRewind()
    {
        if (Obs.Rs == null || RunManager.Instance.NetService.IsGameLoading
            || NMapScreen.Instance?.IsTraveling == true || !ExecutorIdle(allowChoice: true)) return false;
        return Classify().K != Kind.Wait;
    }

    private static Cls Classify()
    {
        var rs = Obs.Rs;
        if (rs == null) return Wait("no run state");
        var player = Me!;
        // A counter relic's activation flash (Kunai, Pen Nib, ... DoActivateVisuals: _isActivating for Cmd.Wait(1f))
        // changes its DisplayAmount and Status for a second; headless (and in Instant mode) it takes no time. The
        // decision is read once it is over.
        if (player.Relics?.Any(ObsShared.IsActivating) == true) return Wait("relic activating");
        var top = NOverlayStack.Instance?.Peek();
        if (top is NGameOverScreen || player.Creature == null || player.Creature.IsDead || RunManager.Instance.IsAbandoned)
            return Dec("game_over");

        // Trial's Double Down: the confirmation popup's yes is the effective choice (engine: RunManager.Abandon()).
        if (NModalContainer.Instance?.OpenModal is NAbandonRunConfirmPopup pop)
            return Auto("abandon confirm (yes)", () => { Invoke(pop, "OnYesButtonPressed", new object?[] { null }); return Done(); });

        if (Selector.HasPending) return Dec("card_select");
        if (top is NChooseABundleSelectionScreen bs)
            return FieldObj(bs, "_bundles") != null && UiHelper.FindAll<NCardBundle>(bs).Count > 0 ? Dec("bundle_select") : Wait("bundle screen loading");
        if (top is NCardRewardSelectionScreen crs)
            return FieldObj(crs, "_completionSource") is TaskCompletionSource<int?> tcs && !tcs.Task.IsCompleted && ExecutorIdle()
                ? Dec("card_reward") : Wait("card reward screen not waiting");
        // A terminal rewards screen (and a finished Crystal Sphere screen) stays on the overlay stack after its proceed
        // opened the map (AutoSlay notes the same).
        var mapEarly = NMapScreen.Instance;
        bool underMap = top is NRewardsScreen or null || (top is NCrystalSphereScreen sph && Sphere(sph)?.IsFinished == true);
        if (mapEarly != null && mapEarly.IsOpen && underMap)
        {
            if (mapEarly.IsTravelEnabled && TravelablePoints(mapEarly).Count > 0 && ExecutorIdle()) return Dec("map_select");
            if (top == null) return Wait("map open, not travelable");
        }
        if (top is NRewardsScreen rws)
        {
            if (!rws.IsNodeReady() || FieldObj(rws, "_rewardsSet") is not RewardsSet) return Wait("rewards screen loading");
            return ExecutorIdle() ? Dec("rewards") : Wait("rewards: executor busy");
        }
        if (top is NCrystalSphereScreen css) return ClassifySphere(css);
        if (top != null) return Wait("overlay " + top.GetType().Name);

        if (CombatManager.Instance.IsInProgress)
        {
            var pcs = player.PlayerCombatState;
            if (pcs != null && pcs.Phase == PlayerTurnPhase.Play && !CombatManager.Instance.IsPlayerReadyToEndTurn(player)
                && !CombatManager.Instance.PlayerActionsDisabled && ExecutorIdle())
                return Dec("combat_play");
            return Wait("combat busy");
        }

        var map = NMapScreen.Instance;
        if (map != null && map.IsOpen)
        {
            if (map.IsTravelEnabled && TravelablePoints(map).Count > 0 && ExecutorIdle()) return Dec("map_select");
            return Wait("map open, not travelable");
        }

        if (NModalContainer.Instance?.OpenModal is { } modal) return Wait("modal " + modal.GetType().Name);

        var room = rs.CurrentRoom;
        switch (room)
        {
            case EventRoom:
                return ClassifyEvent();
            case RestSiteRoom:
            {
                var nroom = NRestSiteRoom.Instance;
                if (nroom == null || !nroom.IsNodeReady()) return Wait("rest node missing");
                if (!ExecutorIdle()) return Wait("rest busy");
                if (nroom.Options.Count == 0)
                    return nroom.ProceedButton.IsEnabled
                        ? Auto("rest proceed (no options left)", () => { nroom.ProceedButton.ForceClick(); return Done(); })
                        : Wait("rest: no options, proceed disabled");
                var buttons = UiHelper.FindAll<NRestSiteButton>(nroom).Where(b => b.IsVisibleInTree()).ToList();
                if (buttons.Count == 0) return Wait("rest buttons not shown");
                return Dec("rest_site");
            }
            case MerchantRoom:
            {
                var nroom = NMerchantRoom.Instance;
                if (nroom == null || !nroom.IsNodeReady()) return Wait("shop node missing");
                if (Watch.Enabled)
                {
                    if (!nroom.Inventory.IsOpen)
                        return Auto("open shop inventory", () => { nroom.OpenInventory(); return Done(); });
                    if (InventoryMoving(nroom.Inventory)) return Wait("shop inventory opening");
                    return ExecutorIdle() ? Dec("shop") : Wait("shop busy");
                }
                if (nroom.Inventory.IsOpen)
                    return Auto("close shop inventory", () => { CloseInventory(nroom.Inventory); return Done(); });
                return ExecutorIdle() ? Dec("shop") : Wait("shop busy");
            }
            case TreasureRoom:
                return ClassifyTreasure();
        }
        return Wait("room " + room?.GetType().Name);
    }

    private static void CloseInventory(MegaCrit.Sts2.Core.Nodes.Screens.Shops.NMerchantInventory inv)
    {
        var back = Field<NBackButton>(inv, "_backButton");
        if (back != null && back.IsEnabled) back.ForceClick();
        else Invoke(inv, "Close");
    }

    private static bool InventoryMoving(NMerchantInventory inv) =>
        Field<Tween>(inv, "_inventoryTween")?.IsRunning() == true;

    private static async Task LeaveInventory(NMerchantInventory inv, NProceedButton proceed)
    {
        if (inv.IsOpen) CloseInventory(inv);
        while (InventoryMoving(inv) || !proceed.IsEnabled) await Driver.Frames(Driver.Tree!, 1);
        proceed.ForceClick();
    }

    private static CrystalSphereMinigame? Sphere(NCrystalSphereScreen s) => FieldObj(s, "_entity") as CrystalSphereMinigame;

    private static Cls ClassifySphere(NCrystalSphereScreen s)
    {
        var g = Sphere(s);
        if (g != null && !g.IsFinished && g.DivinationCount > 0) return ExecutorIdle() ? Dec("crystal_sphere") : Wait("sphere busy");
        var proceed = s.GetNodeOrNull<NProceedButton>("%ProceedButton");
        if (proceed != null && proceed.IsEnabled) return Auto("sphere proceed", () => { proceed.ForceClick(); return Done(); });
        return Wait("sphere finishing");
    }

    private static Cls ClassifyEvent()
    {
        var ev = RunManager.Instance.EventSynchronizer?.GetLocalEvent();
        if (ev == null) return Wait("event loading");
        if (ev is FakeMerchant fm && !fm.StartedFight)
        {
            if (ev.Node is not NFakeMerchant nfm || !nfm.IsNodeReady()) return Wait("fake merchant loading");
            if (Watch.Enabled)
            {
                if (!nfm.Inventory.IsOpen)
                    return Auto("open fake merchant inventory", () => { Invoke(nfm, "OpenInventory"); return Done(); });
                if (InventoryMoving(nfm.Inventory)) return Wait("fake merchant inventory opening");
                return ExecutorIdle() ? Dec("fake_merchant") : Wait("fake merchant busy");
            }
            if (nfm.Inventory.IsOpen) return Auto("close fake merchant inventory", () => { CloseInventory(nfm.Inventory); return Done(); });
            return ExecutorIdle() ? Dec("fake_merchant") : Wait("fake merchant busy");
        }
        var nroom = NRun.Instance?.EventRoom;
        if (nroom == null) return Wait("event room loading");
        var buttons = UiHelper.FindAll<NEventOptionButton>(nroom).Where(b => b.IsVisibleInTree()).ToList();
        if (ev.IsFinished)
        {
            // The synthetic PROCEED of a finished event only opens the map (engine: map_select directly).
            var proceed = buttons.FirstOrDefault(b => b.Option.IsProceed && b.IsEnabled);
            if (proceed != null && ExecutorIdle()) return Auto("event proceed", () => { proceed.ForceClick(); return Done(); });
            return Wait("event finished, waiting for proceed");
        }
        var enabled = buttons.Where(b => b.IsEnabled && !b.Option.IsLocked).ToList();
        if (enabled.Count > 0 && ExecutorIdle()) return Dec("event_choice");
        var ancient = UiHelper.FindFirst<NAncientEventLayout>(nroom);
        if (ancient != null)
        {
            var hit = ancient.GetNodeOrNull<NButton>("%DialogueHitbox");
            if (hit != null && hit.Visible && hit.IsEnabled)
                return Auto("ancient dialogue", () => { hit.EmitSignal(NClickableControl.SignalName.Released, hit); return Done(); });
        }
        return Wait("event options not ready");
    }

    private static NTreasureRoom? TreasureNode() => NRun.Instance?.TreasureRoom;

    private static Cls ClassifyTreasure()
    {
        var nroom = TreasureNode();
        if (nroom == null || !nroom.IsNodeReady()) return Wait("treasure loading");
        if (!ExecutorIdle()) return Wait("treasure busy");
        bool opened = ChestOpened(nroom);
        bool collectionOpen = FieldObj(nroom, "_isRelicCollectionOpen") is true;
        var chest = Field<NTreasureButton>(nroom, "_chestButton");
        if (!opened)
        {
            if (chest != null && chest.IsEnabled && !collectionOpen) return Dec("treasure");
            return Wait("chest opening");
        }
        if (collectionOpen && _treasurePickedRoom != nroom.GetInstanceId() && TreasureHolders(nroom).Count > 0) return Dec("treasure");
        if (!collectionOpen && nroom.ProceedButton.IsEnabled && !nroom.ProceedButton.IsSkip)
            return Auto("treasure proceed", () => { nroom.ProceedButton.ForceClick(); return Done(); });
        return Wait("treasure: relic picking");
    }

    /// <summary>Whether the chest has been opened. v0.111 kept NTreasureRoom._hasChestBeenOpened; v0.113 dropped it:
    /// OpenChest sets the chest's mouse filter to Ignore at once, _isRelicCollectionOpen during the relic pick, and the
    /// proceed state to Skip / Proceed (enum None, Hidden, Skip, Proceed) afterwards.</summary>
    private static bool ChestOpened(NTreasureRoom nroom)
    {
        if (FieldObj(nroom, "_hasChestBeenOpened") is bool b) return b;
        if (FieldObj(nroom, "_isRelicCollectionOpen") is true) return true;
        var st = FieldObj(nroom, "_proceedButtonState");
        if (st != null && Convert.ToInt32(st) >= 2) return true;
        var chest = Field<NTreasureButton>(nroom, "_chestButton");
        return chest != null && chest.MouseFilter == Control.MouseFilterEnum.Ignore;
    }

    private static List<NTreasureRoomRelicHolder> TreasureHolders(NTreasureRoom nroom)
    {
        var coll = Field<NTreasureRoomRelicCollection>(nroom, "_relicCollection");
        if (coll == null) return new();
        var inUse = FieldObj(coll, "_holdersInUse") as List<NTreasureRoomRelicHolder> ?? new();
        return inUse.Where(h => h.Visible && h.IsEnabled && h.IsVisibleInTree()).ToList();
    }

    public static List<NMapPoint> TravelablePoints(NMapScreen map) =>
        UiHelper.FindAll<NMapPoint>(map).Where(p => p.State == MapPointState.Travelable && p.IsEnabled).ToList();

    // ------------------------------------------------------------------ waiting for a stable decision

    public static async Task<DecisionPoint> WaitForDecision(SceneTree tree, double timeoutSec, int stableFrames = 3)
    {
        var t0 = DateTime.UtcNow;
        string? last = null;
        int stable = 0;
        string lastNote = "";
        while ((DateTime.UtcNow - t0).TotalSeconds < timeoutSec)
        {
            await Driver.Frames(tree, 2);
            if (Watch.Enabled)
            {
                var held = DateTime.UtcNow;
                await Watch.WaitForPlayback(tree);
                t0 += DateTime.UtcNow - held;
            }
            Cls c;
            try { c = Classify(); }
            catch (Exception ex) { c = Wait("classify threw " + ex.GetType().Name + ": " + ex.Message); }
            if (c.Note != lastNote && c.Note != "") { Entry.Log("  [" + c.Note + "]"); lastNote = c.Note; }
            if (c.K == Kind.Auto)
            {
                if (_autoCooldownFrames > 0) { _autoCooldownFrames--; continue; }
                try { await c.Auto!(); }
                catch (Exception ex) { Entry.Log($"auto '{c.Note}' threw {ex}"); }
                _autoCooldownFrames = 6;
                last = null; stable = 0;
                t0 = DateTime.UtcNow; // progress made
                continue;
            }
            if (c.K == Kind.Wait) { last = null; stable = 0; continue; }
            DecisionPoint dp;
            try { dp = Build(c.Decision); }
            catch (Exception ex) { Entry.Log("build threw " + ex); last = null; stable = 0; continue; }
            var dig = Json.Canon(new Dictionary<string, object?> { ["d"] = dp.Decision, ["o"] = dp.Obs, ["c"] = dp.Cands.Select(x => x.Desc).ToList() });
            if (dig == last)
            {
                if (++stable >= stableFrames)
                {
                    Obs.CommitKnownTop(Me!);  // the engine records the known draw-pile top once per decision
                    return dp;
                }
            }
            else { last = dig; stable = 0; }
        }
        var stuck = new DecisionPoint { Decision = "stuck", Note = lastNote };
        try { if (Me != null) stuck.Obs = Obs.Build(Me, () => new()); } catch { }
        return stuck;
    }

    // ------------------------------------------------------------------ decision → obs + candidates

    private static DecisionPoint Build(string decision)
    {
        var player = Me!;
        var dp = new DecisionPoint { Decision = decision };
        // ctx is built inside the observation, after its combat part (as in the engine's RlObs)
        dp.Obs = Obs.Build(player, () => Ctx(player, decision));
        if (decision != "game_over") dp.Cands = Candidates(player, decision, dp);
        return dp;
    }

    private static RewardsSet? CurrentRewardsSet(out NRewardsScreen? screen, out bool terminal)
    {
        screen = NOverlayStack.Instance?.Peek() as NRewardsScreen;
        terminal = screen != null && FieldObj(screen, "_isTerminal") is true;
        return screen == null ? null : FieldObj(screen, "_rewardsSet") as RewardsSet;
    }

    private static Dictionary<string, object?> Ctx(Player player, string decision)
    {
        switch (decision)
        {
            case "card_select":
            {
                var cs = Selector.Pending!;
                return ObsShared.SelectionCtx(cs.Min, cs.Max, cs.Picked.Select(i => cs.Options[i]), cs.Ctx.Kind, cs.Cancelable,
                                              cs.Ctx.Source, cs.Ctx.Pile, cs.Ctx.PromptKey, Obs.Played);
            }
            case "card_reward":
            {
                var screen = (NCardRewardSelectionScreen)NOverlayStack.Instance!.Peek()!;
                return ObsShared.CardRewardCtx(FieldObj(screen, "_extraOptions") as IReadOnlyList<CardRewardAlternative> ?? Array.Empty<CardRewardAlternative>());
            }
            case "rewards":
            {
                var set = CurrentRewardsSet(out var screen, out bool terminal);
                if (set == null || screen == null) return new() { ["terminal"] = terminal };
                var st = Nav.Rewards(screen, set);   // declined rewards are left out of ctx.blocked (engine RewardsPrompt.Declined)
                return ObsShared.RewardsCtx(player, set, terminal, r => st.Declined.Contains(r));
            }
            case "treasure":
            {
                var nroom = TreasureNode();
                return ObsShared.TreasureCtx(nroom != null && ChestOpened(nroom));
            }
            case "event_choice":
                return ObsShared.EventCtx(RunManager.Instance.EventSynchronizer?.GetLocalEvent());
            case "fake_merchant":
                return ObsShared.FakeMerchantCtx(RunManager.Instance.EventSynchronizer?.GetLocalEvent(), player, Obs.Played);
            case "rest_site":
                return ObsShared.RestCtx(NRestSiteRoom.Instance is { } nr && nr.ProceedButton.IsEnabled);
            case "crystal_sphere":
                if (NOverlayStack.Instance?.Peek() is NCrystalSphereScreen s && Sphere(s) is { } g) return ObsShared.SphereCtx(g);
                break;
            case "shop":
                return ObsShared.ShopCtx(Obs.Rs?.CurrentRoom, player, Obs.Played);
        }
        return new Dictionary<string, object?>();
    }

    private static List<Cand> Candidates(Player player, string decision, DecisionPoint dp)
    {
        var cands = new List<Cand>();
        void Add(Dictionary<string, object?> desc, Func<Task> exec) => cands.Add(new Cand(desc, exec));

        switch (decision)
        {
            case "combat_play":
            {
                var pcs = player.PlayerCombatState!;
                var targets = ObsShared.CombatEnemies();   // dead enemies that stay to revive keep their index
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
                                if (ObsShared.Targetable(target) && c.CanPlayTargeting(target))
                                    Add(ObsShared.PlayDesc(c, hi, t, target), () => PlayCard(c, target));
                            }
                            break;
                        case TargetType.AnyAlly:
                            break;
                        default:
                            if (c.CanPlayTargeting(null))
                                Add(ObsShared.PlayDesc(c, hi, -1, null), () => PlayCard(c, null));
                            break;
                    }
                }
                PotionCands(player, Add);
                Add(ObsShared.Simple("end_turn"), () => EndTurn(player));
                return cands;
            }
            case "map_select":
            {
                var map = NMapScreen.Instance!;
                var order = ObsShared.TravelablePoints(Obs.Rs!).Select(p => p.coord).ToList();   // the engine's order
                int Rank(NMapPoint p) { int r = order.IndexOf(p.Point.coord); return r < 0 ? 1000 + p.Point.coord.row * 20 + p.Point.coord.col : r; }
                foreach (var np in TravelablePoints(map).OrderBy(Rank))
                {
                    var pt = np;
                    Add(ObsShared.MapDesc(Obs.NodeIndex(pt.Point)), () => { pt.ForceClick(); return Done(); });
                }
                break;
            }
            case "card_reward":
            {
                var screen = (NCardRewardSelectionScreen)NOverlayStack.Instance!.Peek()!;
                var st = Nav.CardRewardState(screen);
                dp.Screen = st;
                var options = FieldObj(screen, "_options") as IReadOnlyList<CardCreationResult> ?? Array.Empty<CardCreationResult>();
                var alts = FieldObj(screen, "_extraOptions") as IReadOnlyList<CardRewardAlternative> ?? Array.Empty<CardRewardAlternative>();
                for (int i = 0; i < options.Count; i++)
                {
                    var card = options[i].Card;
                    Add(ObsShared.RewardCardDesc(card, Obs.Played), () =>
                    {
                        var holder = UiHelper.FindAll<NGridCardHolder>(screen).FirstOrDefault(h => h.CardModel == card && !h.IsQueuedForDeletion())
                                     ?? throw new InvalidOperationException("card holder not found");
                        holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);   // NCardRewardSelectionScreen.SelectCard
                        return Done();
                    });
                }
                for (int a = 0; a < alts.Count; a++)
                {
                    int ai = a;
                    var desc = ObsShared.AltDesc(alts[a].OptionId);
                    Func<Task> click = () =>
                    {
                        var container = Field<Control>(screen, "_rewardAlternativesContainer");
                        var btns = container?.GetChildren().OfType<NCardRewardAlternativeButton>().Where(b => !b.IsQueuedForDeletion()).ToList() ?? new();
                        if (ai < btns.Count) btns[ai].ForceClick();       // Released → OnAlternateRewardSelected(ai)
                        else Invoke(screen, "OnAlternateRewardSelected", ai);
                        return Done();
                    };
                    // Skip (EndSelectionAndDoNotCompleteReward: the reward stays) on a screen opened from a rewards
                    // screen: decline, or back out with the reward held back, or nothing on a must-take-all screen.
                    if (st.FromReward is { } o && alts[a].AfterSelected == PostAlternateCardRewardAction.EndSelectionAndDoNotCompleteReward)
                    {
                        var back = Nav.BackOut(st.Changes, o, click);
                        if (back == null) continue;
                        click = back;
                    }
                    Add(desc, click);
                }
                break;
            }
            case "rewards":
            {
                var set = CurrentRewardsSet(out var screen, out bool terminal)!;
                var st = Nav.Rewards(screen!, set);
                st.Settle();
                dp.Screen = st;
                var buttons = (FieldObj(screen!, "_rewardButtons") as List<Control> ?? new())
                    .OfType<NRewardButton>().Where(b => b.IsEnabled && b.Reward != null && !b.IsQueuedForDeletion()).ToList();
                var rewards = set.Rewards;
                var held = new List<(Dictionary<string, object?> desc, Func<Task> exec)>();
                int offered = 0;
                for (int i = 0; i < rewards.Count; i++)
                {
                    var r = rewards[i];
                    var btn = buttons.FirstOrDefault(b => ReferenceEquals(b.Reward, r));
                    if (btn == null) continue;
                    // Declined rewards are left behind like untaken ones (engine RewardsPrompt.Declined).
                    if (st.Declined.Contains(r)) continue;
                    // The engine drops a potion reward the belt cannot take (the click only shows "full").
                    if (r is PotionReward pr && pr.Potion != null && !ObsShared.CanReceivePotion(player, pr.Potion)) continue;
                    var b = btn;
                    Func<Task> exec = () => { Nav.OpenReward(st, r, b.ForceClick); return Done(); };
                    if (st.Held.Contains(r)) { held.Add((ObsShared.RewardDesc(r, i, Obs.Played), exec)); continue; }
                    Add(ObsShared.RewardDesc(r, i, Obs.Played), exec);
                    offered++;
                }
                var proceed = Field<NProceedButton>(screen!, "_proceedButton");
                bool canProceed = proceed != null && proceed.IsEnabled;
                // A held-back reward is still the way forward when nothing else is (engine RewardsCands).
                if (offered == 0 && !canProceed)
                    foreach (var h in held) Add(h.desc, h.exec);
                if (canProceed)
                    Add(ObsShared.Simple("proceed"), () => { proceed!.ForceClick(); return Done(); });
                break;
            }
            case "card_select":
            {
                var cs = Selector.Pending!;
                dp.Screen = cs;
                bool chooseACard = cs.Ctx.Kind == "choose_a_card";
                if (cs.Picked.Count < cs.Max)
                    for (int i = 0; i < cs.Options.Count; i++)
                    {
                        if (cs.Picked.Contains(i)) continue;
                        int si = i;
                        Add(ObsShared.PickDesc(i, cs.Options[i], Obs.Played), () =>
                        {
                            cs.Pick(si);
                            if (cs.Picked.Count >= cs.Max) cs.Submit();
                            return Done();
                        });
                    }
                if (chooseACard)
                {
                    if (cs.Min == 0) Add(new() { ["k"] = "skip" }, () => { cs.Cancel(); return Done(); });
                }
                else
                {
                    if (cs.Picked.Count >= cs.Min && (cs.Picked.Count < cs.Max || cs.Max == 0) && !(cs.Min == 0 && cs.Picked.Count == 0 && cs.Cancelable))
                        Add(new() { ["k"] = "confirm" }, () => { cs.Submit(); return Done(); });
                    // Engine CardSelectCands: no cancel for Smith / Cook / the merchant's removal; for a removal reward
                    // opened from a rewards screen the cancel declines it (or backs out with it held back).
                    Func<Task> cancel = () => { cs.Cancel(); return Done(); };
                    if (!cs.Cancelable || cs.RoomService != null) { }
                    else if (cs.FromReward is { } o)
                    {
                        var back = Nav.BackOut(cs.Changes, o, cancel);
                        if (back != null) Add(new() { ["k"] = "skip" }, back);
                    }
                    else Add(new() { ["k"] = "skip" }, cancel);
                }
                break;
            }
            case "bundle_select":
            {
                var screen = (NChooseABundleSelectionScreen)NOverlayStack.Instance!.Peek()!;
                var nodes = UiHelper.FindAll<NCardBundle>(screen);
                var bundles = FieldObj(screen, "_bundles") as IReadOnlyList<IReadOnlyList<CardModel>> ?? Array.Empty<IReadOnlyList<CardModel>>();
                for (int i = 0; i < bundles.Count; i++)
                {
                    var b = bundles[i];
                    Add(ObsShared.BundleDesc(b, Obs.Played), async () =>
                    {
                        var node = nodes.FirstOrDefault(n => ReferenceEquals(n.Bundle, b)) ?? throw new InvalidOperationException("bundle node not found");
                        node.Hitbox.ForceClick();
                        await Driver.Frames(Driver.Tree!, 10);
                        var confirm = UiHelper.FindAll<NConfirmButton>(screen).FirstOrDefault(x => x.IsEnabled && x.IsVisibleInTree())
                                      ?? throw new InvalidOperationException("bundle confirm not found");
                        confirm.ForceClick();
                    });
                }
                break;
            }
            case "event_choice":
            {
                var ev = RunManager.Instance.EventSynchronizer!.GetLocalEvent()!;
                var options = ev.CurrentOptions;
                var nroom = NRun.Instance?.EventRoom;
                for (int i = 0; i < options.Count; i++)
                {
                    var o = options[i];
                    if (o.IsLocked || ObsShared.IsSpentOption(o)) continue;
                    Add(ObsShared.OptionDesc(ev, o), () =>
                    {
                        var btn = nroom == null ? null : UiHelper.FindAll<NEventOptionButton>(nroom).FirstOrDefault(b => ReferenceEquals(b.Option, o));
                        if (btn == null) throw new InvalidOperationException("event option button not found");
                        btn.ForceClick();
                        return Done();
                    });
                }
                break;
            }
            case "rest_site":
            {
                var nroom = NRestSiteRoom.Instance!;
                var buttons = UiHelper.FindAll<NRestSiteButton>(nroom);
                foreach (var o in nroom.Options)
                {
                    if (!o.IsEnabled) continue;
                    var opt = o;
                    Add(ObsShared.RestDesc(o, player), () =>
                    {
                        var btn = buttons.FirstOrDefault(b => ReferenceEquals(b.Option, opt)) ?? throw new InvalidOperationException("rest button not found");
                        // NRestSiteButton.OnRelease → RestSiteSynchronizer.ChooseLocalOption reaches the Smith / Cook
                        // selection synchronously (engine RestChoice's _openingService window).
                        var svc = opt is SmithRestSiteOption ? "rest:SMITH" : opt is CookRestSiteOption ? "rest:COOK" : null;
                        Nav.OpenService(svc, btn.ForceClick);
                        return Done();
                    });
                }
                if (nroom.ProceedButton.IsEnabled) Add(ObsShared.Simple("leave"), () => { nroom.ProceedButton.ForceClick(); return Done(); });
                break;
            }
            case "shop":
            {
                var nroom = NMerchantRoom.Instance!;
                if (Obs.Rs?.CurrentRoom is MerchantRoom mr && RunManager.Instance.MerchantSynchronizer.GetLocalInventory() is { } inv)
                {
                    foreach (var e in inv.CharacterCardEntries.Concat(inv.ColorlessCardEntries))
                        if (e.IsStocked && e.EnoughGold && e.CreationResult?.Card is { } card)
                        {
                            var entry = e;
                            Add(ObsShared.BuyCardDesc(card, e.Cost, Obs.Played), () => Purchase(entry, inv));
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
                if (nroom.ProceedButton.IsEnabled || (Watch.Enabled && nroom.Inventory.IsOpen))
                    Add(ObsShared.Simple("leave"), () => Watch.Enabled
                        ? LeaveInventory(nroom.Inventory, nroom.ProceedButton)
                        : ClickProceed(nroom.ProceedButton));
                break;
            }
            case "fake_merchant":
            {
                var fm = (FakeMerchant)RunManager.Instance.EventSynchronizer!.GetLocalEvent()!;
                var inv = fm.Inventory;
                foreach (var e in inv.RelicEntries)
                    if (e.IsStocked && e.EnoughGold && e.Model != null)
                    {
                        var entry = e;
                        Add(ObsShared.BuyRelicDesc(e.Model, e.Cost), () => Purchase(entry, inv));
                    }
                var node = (NFakeMerchant)fm.Node!;
                var proceed = Field<NProceedButton>(node, "_proceedButton");
                if (proceed != null && (proceed.IsEnabled || (Watch.Enabled && node.Inventory.IsOpen)))
                    Add(ObsShared.Simple("leave"), () => Watch.Enabled
                        ? LeaveInventory(node.Inventory, proceed)
                        : ClickProceed(proceed));
                break;
            }
            case "treasure":
            {
                var nroom = TreasureNode()!;
                bool opened = ChestOpened(nroom);
                if (!opened)
                {
                    var chest = Field<NTreasureButton>(nroom, "_chestButton")!;
                    Add(ObsShared.Simple("open"), () => { chest.ForceClick(); return Done(); });   // OnChestButtonReleased → OpenChest
                }
                else
                {
                    ulong roomId = nroom.GetInstanceId();
                    foreach (var h in TreasureHolders(nroom))
                    {
                        var holder = h;
                        Add(ObsShared.TreasureDesc(holder.Relic.Model), async () =>
                        {
                            // NTreasureRoomRelicCollection.PickRelic ignores clicks within 200 ms of opening.
                            await Task.Delay(300);
                            _treasurePickedRoom = roomId;
                            holder.ForceClick();
                        });
                    }
                    if (nroom.ProceedButton.IsEnabled && nroom.ProceedButton.IsSkip)
                        Add(ObsShared.Simple("skip"), () => { _treasurePickedRoom = roomId; nroom.ProceedButton.ForceClick(); return Done(); });
                }
                break;
            }
            case "crystal_sphere":
            {
                var s = (NCrystalSphereScreen)NOverlayStack.Instance!.Peek()!;
                var g = Sphere(s)!;
                foreach (var tool in new[] { "small", "big" })
                    for (int x = 0; x < g.GridSize.X; x++)
                        for (int y = 0; y < g.GridSize.Y; y++)
                            if (g.cells[x, y].IsHidden)
                            {
                                var cell = g.cells[x, y];
                                var tt = tool == "small" ? CrystalSphereMinigame.CrystalSphereToolType.Small : CrystalSphereMinigame.CrystalSphereToolType.Big;
                                Add(ObsShared.SphereDesc(tool, x, y), () =>
                                {
                                    // NCrystalSphereScreen: the tool buttons call SetTool, a hidden cell's release calls CellClicked.
                                    g.SetTool(tt);
                                    TaskHelper.RunSafely(g.CellClicked(cell));
                                    return Done();
                                });
                            }
                break;
            }
        }
        PotionCands(player, Add);
        return cands;
    }

    /// <summary>Engine definition of "a card selection screen is open" (NPlayerHand.IsInCardSelection / ICardSelector overlay).</summary>
    private static bool InCardSelectScreen =>
        Selector.HasPending || NOverlayStack.Instance?.Peek() is NChooseABundleSelectionScreen or NCardRewardSelectionScreen;

    /// <summary>NPotionPopup's use-button rules (engine Harness.CanUsePotion, shared).</summary>
    private static bool CanUsePotion(Player me, PotionModel p) =>
        ObsShared.CanUsePotion(me, p, _pendingDiscards.Contains(p), () => InCardSelectScreen);

    private static bool CanDiscardPotion(Player me, PotionModel p) => ObsShared.CanDiscardPotion(me, p, _pendingDiscards.Contains(p));

    private static void PotionCands(Player player, Action<Dictionary<string, object?>, Func<Task>> Add)
    {
        _pendingDiscards.RemoveWhere(p => !player.PotionSlots.Contains(p));
        var slots = player.PotionSlots;
        bool inCombat = CombatManager.Instance.IsInProgress;
        for (int s = 0; s < slots.Count; s++)
        {
            var p = slots[s];
            if (p == null) continue;
            int slot = s;
            if (CanUsePotion(player, p))
            {
                if (p.TargetType == TargetType.AnyEnemy)
                {
                    if (inCombat)
                    {
                        var targets = ObsShared.CombatEnemies();
                        for (int t = 0; t < targets.Count; t++)
                        {
                            var target = targets[t];
                            if (ObsShared.Targetable(target) && p.IsValidTarget(target))
                                Add(ObsShared.PotionDesc(slot, t), () => UsePotion(p, target));
                        }
                    }
                }
                else if (p.TargetType == TargetType.AnyAlly) { }
                else
                    Add(ObsShared.PotionDesc(slot, -1), () => UsePotion(p, null));
            }
            if (CanDiscardPotion(player, p))
                Add(ObsShared.DiscardPotionDesc(slot), () => DiscardPotion(p));
        }
    }

    // ------------------------------------------------------------------ UI-path actions

    private static Task ClickProceed(NProceedButton proceed)
    {
        proceed.ForceClick();
        return Done();
    }

    public static void PickVisualCard(NDeckCardSelectScreen screen, CardModel card) => Invoke(screen, "OnCardClicked", card);
    public static void SubmitVisualCards(NDeckCardSelectScreen screen) => Invoke(screen, "CheckIfSelectionComplete");

    private static Task PlayCard(CardModel card, Creature? target)
    {
        // NCardPlay → CardModel.TryManualPlay → ActionQueueSynchronizer.RequestEnqueue(PlayCardAction)
        if (!card.TryManualPlay(target)) throw new InvalidOperationException($"TryManualPlay refused {card.Id} target={target?.Monster?.Id}");
        return Done();
    }

    private static Task UsePotion(PotionModel p, Creature? target)
    {
        // NPotionHolder.UsePotion: targeted potions get the chosen creature, Self gets the owner, others null.
        if (target == null && p.TargetType == TargetType.Self) target = p.Owner.Creature;
        p.EnqueueManualUse(target);
        return Done();
    }

    private static Task DiscardPotion(PotionModel p)
    {
        // NPotionPopup.OnDiscardButtonPressed
        var owner = p.Owner;
        int idx = owner.PotionSlots.ToList().IndexOf(p);
        if (idx < 0) throw new InvalidOperationException("potion not in belt");
        _pendingDiscards.Add(p);
        RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new DiscardPotionGameAction(owner, (uint)idx, CombatManager.Instance.IsInProgress));
        return Done();
    }

    private static Task EndTurn(Player player)
    {
        var btn = NCombatRoom.Instance?.Ui?.EndTurnButton;
        if (btn != null) btn.CallReleaseLogic();
        else
            RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(player, player.PlayerCombatState!.TurnNumber));
        return Done();
    }

    private static Task Purchase(MerchantEntry e, MerchantInventory inv)
    {
        // NMerchantSlot.OnTryPurchase → entry.OnTryPurchaseWrapper(inventory); the removal opens its selection
        // synchronously (engine Purchase's _openingService window: no cancel).
        if (e is MerchantCardRemovalEntry rem)
        {
            Task<bool>? purchase = null;
            Nav.OpenService("shop:REMOVE", () => purchase = RunManager.Instance.MerchantSynchronizer.PurchaseLocalEntry(rem));
            TaskHelper.RunSafely(purchase!);
        }
        else TaskHelper.RunSafely(RunManager.Instance.MerchantSynchronizer.PurchaseLocalEntry(e));
        return Done();
    }
}
