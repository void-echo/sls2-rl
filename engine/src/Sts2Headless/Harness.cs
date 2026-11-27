using System.Reflection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Ascension;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Entities.TreasureRelicPicking;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Odds;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Unlocks;
using Sts2Obs;

namespace Sts2Headless;

/// <summary>What the game is waiting for right now.</summary>
internal sealed class Decision
{
    public string Kind = "unknown";
    public Prompt? Prompt;
    public string? Error;
    public bool Done;
    public bool Victory;
    public bool Abandoned;
}

/// <summary>
/// Harness v1. The real sts2.dll runs on one thread (the process main thread) under <see cref="GameLoop"/>.
/// The harness never decides for the player and never reimplements a flow: each player input is the call
/// the client's UI (or AutoSlay) makes, and every place the client would wait on a screen is a
/// harness-owned TaskCompletionSource (<see cref="Prompt"/>). After each input the loop is drained and
/// the resulting stable state is read off the game (<see cref="Decide"/>).
/// </summary>
public partial class RunSimulator
{
    internal static RunSimulator? Current;
    private static bool _inited;
    private static IDisposable? _selectorScope;
    private static readonly HeadlessCardSelector _selector = new();

    private RunState? _runState;
    private readonly List<Prompt> _prompts = new();
    private long _seq;
    private string? _fault;
    private bool _won;
    private bool _suppressFaults;
    /// <summary>The room whose exit to the map the player has unlocked (rewards proceed, shop leave, ...).</summary>
    private AbstractRoom? _travelRoom;
    /// <summary>Rest site room in which an option succeeded (its proceed button is enabled).</summary>
    private AbstractRoom? _restOkRoom;
    private TreasureRoomRelicSynchronizer? _relicSync;
    /// <summary>Potions whose discard action is queued: NPotionHolder.DisableUntilPotionRemoved greys them out.</summary>
    private readonly HashSet<PotionModel> _pendingDiscards = new();
    /// <summary>
    /// Where the screen being opened right now comes from, set only for the synchronous part of the input that opens it
    /// (SelectLocalReward, RestSiteSynchronizer.ChooseLocalOption, the removal entry's OnTryPurchaseWrapper all reach
    /// ShowScreen / the selector before their first await). The prompt created inside the window takes it over.
    /// </summary>
    private RewardOrigin? _openingReward;
    private string? _openingService;

    private Player Me => _runState!.Players[0];

    // ───────────────────────────── one-time process setup

    public static void Init()
    {
        if (_inited) return;
        _inited = true;
        TestMode.IsOn = true;
        // Mod discovery and assembly metadata are required even unmodded; TestMode skips loading mods.
        // Done before the game loop is installed: GetResult must not block the loop thread on its own queue.
        MegaCrit.Sts2.Core.Modding.ModManager.Initialize(new MegaCrit.Sts2.Core.Modding.ModManagerFileIo(), null, null).GetAwaiter().GetResult();
        MegaCrit.Sts2.Core.Modding.AssemblyInfo.Init();
        SynchronizationContext.SetSynchronizationContext(GameLoop.Instance);
        try { var _ = MegaCrit.Sts2.Core.Platform.PlatformUtil.PrimaryPlatform; }
        catch (Exception ex) { Console.Error.WriteLine($"[WARN] PlatformUtil init: {ex.Message}"); }
        SaveManager.Instance.InitProfileId(0);
        SaveManager.Instance.InitPrefsDataForTest();

        Patches.ApplyAll();
        Revert.PatchRevert.Apply();
        RemoveMonoModHelper();
        foreach (var l in Patches.Log.Where(l => l.StartsWith("[WARN]"))) Console.Error.WriteLine(l);
        Localization.Init();

        var subtypes = AbstractModelSubtypes.All;
        var failures = new List<string>();
        for (int i = 0; i < subtypes.Count; i++)
        {
            try { ModelDb.Inject(subtypes[i]); }
            catch (Exception ex) { failures.Add($"{subtypes[i].Name}: {ex.GetType().Name}: {ex.Message}"); }
        }
        if (failures.Count > 0)
            throw new InvalidOperationException($"ModelDb.Inject failed for {failures.Count} models (content would be missing): "
                                                + string.Join("; ", failures.Take(5)));
        SaveManager.Instance.InitProgressData();
        ModelIdSerializationCache.Init();

        RewardsSet.testSelector = set => Current != null ? Current.RewardsLoop(set) : Task.CompletedTask;
        CombatManager.Instance.CombatWon += room => Current?.OnCombatWon(room);
        RunManager.Instance.RoomEntered += () => Current?.OnRoomEntered();
        TaskHelper.UnobservedFault += e => Current?.OnFault(e);
        GameLoop.Instance.CallbackFaulted += e => Current?.OnFault(e);
        Diag.AttachGlobal();
    }

    // ───────────────────────────── run lifecycle

    /// <summary>
    /// New run, following StartRunLobby.BeginRunLocally + NGame.StartNewSingleplayerRun + NGame.StartRun:
    /// acts from ActModel.GetRandomList with the "act_selection" RNG, RunState.CreateForNewRun with an
    /// all-unlocked profile, SetUpNewSingleplayer, FinalizeStartingRelics, Launch, EnterAct(0).
    /// </summary>
    internal Decision StartRun(string character, int ascension, string? seed)
    {
        CleanUpRun();
        _saves.Clear();
        _saveId = 0;
        _latestSave = null;
        // A previous run must leave nothing behind; otherwise the caller has to use a fresh process.
        // (Before the first run of a process the executor and queues do not exist yet.)
        if (RunManager.Instance.ActionExecutor is { IsRunning: true } || RunManager.Instance.ActionQueueSet is { IsEmpty: false })
            throw new InvalidOperationException("dirty process: the previous run left game actions running or queued");
        Current = this;
        seed ??= "headless_" + DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        var charModel = ModelDb.AllCharacters.FirstOrDefault(c =>
            string.Equals(c.Id.Entry, character, StringComparison.OrdinalIgnoreCase)
            || string.Equals(c.GetType().Name, character, StringComparison.OrdinalIgnoreCase));
        if (charModel == null) return Err($"Unknown character: {character}");
        if (ascension < 0 || ascension > 10) return Err($"ascension must be 0..10, got {ascension}");

        // Process-wide state the real game keeps per profile / per launch, made a function of the seed.
        ulong h = StringHelper.GetDeterministicHashCode(seed);
        Rng.Chaotic.LoadFromSerializable(new Rng(h, "headless_chaotic").ToSerializable());
        ResetProfile();

        var unlock = UnlockState.all;
        var player = Player.CreateForNewRun(charModel, unlock, 1uL);
        var actRng = new Rng(h, "act_selection");
        var acts = ActModel.GetRandomList(actRng, unlock, isMultiplayer: false).ToList();
        _runState = RunState.CreateForNewRun(new[] { player }, acts.Select(a => a.ToMutable()).ToList(),
            Array.Empty<ModifierModel>(), GameMode.Standard, false, ascension, seed);
        RunManager.Instance.SetUpNewSingleplayer(_runState, shouldSave: false);
        Diag.Attach();
        ResetAscensionStatics();
        // localOnly: the selector sits where the UI screens are (CardSelectCmd.LocalSelector), so the real single-player
        // path still runs around it: ReserveChoiceId, SignalPlayerChoiceBegun/Ended (a choice inside a hook becomes a
        // queued hook action and pauses the running action), exactly as when a screen is shown. The non-local Selector
        // is the AutoSlay/test seam that skips all of that and changes the order of turn-start effects.
        _selectorScope = RunManager.Instance.CardSelectorStack.UseLocalSelector(_selector);
        _relicSync = RunManager.Instance.TreasureRoomRelicSynchronizer;
        _relicSync.RelicsAwarded += OnRelicsAwarded;

        try
        {
            RunTask(RunManager.Instance.FinalizeStartingRelics(), "FinalizeStartingRelics");
            RunManager.Instance.Launch();
            RunTask(RunManager.Instance.EnterAct(0, doTransition: false), "EnterAct(0)");
        }
        catch (Exception e) { return Err("run start failed: " + FaultText(e)); }
        return Decide();
    }

    private static readonly PropertyInfo? _uniqueId = typeof(ProgressState).GetProperty("UniqueId");

    /// <summary>
    /// Fresh progress each run. Gameplay reads a few progress fields (Looming Fruit's per-profile cornucopia
    /// via UniqueId, Wongo points, ancient visit counts); a fixed profile keeps runs independent and
    /// reproducible. STS2_PROFILE_ID overrides the profile id.
    /// </summary>
    private static void ResetProfile()
    {
        var progress = ProgressState.CreateDefault();
        try { _uniqueId?.SetValue(progress, Environment.GetEnvironmentVariable("STS2_PROFILE_ID") ?? "SLS2RL1"); }
        catch (Exception e) { Console.Error.WriteLine($"[WARN] profile id: {e.Message}"); }
        SaveManager.Instance.Progress = progress;
    }

    /// <summary>CardRarityOdds.regularCommonOdds is a static initialised once per process from the first run's ascension.</summary>
    private static void ResetAscensionStatics()
    {
        typeof(CardRarityOdds).GetField("regularCommonOdds", BindingFlags.Public | BindingFlags.Static)!
            .SetValue(null, AscensionHelper.GetValueIfAscension(AscensionLevel.Scarcity, 0.615f, 0.6f));
    }

    /// <summary>
    /// Tear down the previous run (audit §7): drop every pending prompt (the old run's tasks are abandoned, never
    /// resumed), RunManager.CleanUp (action queue, CombatManager.Reset, synchronizers, LocalContext), let the
    /// cancellation settle, then empty the loop. Asserts that nothing of the old run is still live.
    /// </summary>
    internal void CleanUpRun()
    {
        KnownTop.Reset();
        _suppressFaults = true;
        try
        {
            // Cancel every wait first so the old run's tasks (turn loop, event options, reward loops) unwind now;
            // a turn loop left suspended would make the next combat wait for it (CombatManager.WaitForPreviousTurnLoopToFinish).
            foreach (var p in _prompts.ToList()) p.Abort();
            _prompts.Clear();
            DrainForCleanup();
            if (_relicSync != null) _relicSync.RelicsAwarded -= OnRelicsAwarded;
            _relicSync = null;
            if (RunManager.Instance.DebugOnlyGetState() != null) RunManager.Instance.CleanUp(graceful: true);
            DrainForCleanup();
        }
        catch (Exception e) { Console.Error.WriteLine($"[WARN] cleanup: {e.GetType().Name}: {e.Message}"); }
        finally
        {
            GameLoop.Instance.Clear();
            GameLoop.Instance.ResetForeignPosts();
            _suppressFaults = false;
        }
        Patches.ResetRunScoped();
        Diag.Reset();
        _selectorScope?.Dispose();
        _selectorScope = null;
        SelectionCapture.Depth = 0;
        SelectionCapture.Current = null;
        _runState = null;
        _fault = null;
        _won = false;
        _travelRoom = _restOkRoom = null;
        _pendingDiscards.Clear();
        _openingReward = null;
        _openingService = null;
        _seq = 0;
        ResetRlState();
        if (Current == this) Current = null;
        if (CombatManager.Instance.IsInProgress)
            throw new InvalidOperationException("CombatManager still in progress after run cleanup");
        if (_turnLoopTask.GetValue(CombatManager.Instance) is Task loop && !loop.IsCompleted)
            throw new InvalidOperationException("the previous run's combat turn loop is still suspended after cleanup");
    }

    private static readonly FieldInfo _turnLoopTask = typeof(CombatManager).GetField("_turnLoopTask", BindingFlags.Instance | BindingFlags.NonPublic)!;

    private static void DrainForCleanup()
    {
        for (int i = 0; i < 100_000 && GameLoop.Instance.RunUntilIdle() > 0; i++) { }
    }

    // ───────────────────────────── prompts

    internal static void AddPrompt(Prompt p)
    {
        var self = Current;
        if (self == null) throw new InvalidOperationException("prompt outside a run");
        p.Seq = ++self._seq;
        if (!self._prompts.Contains(p)) self._prompts.Add(p);
        Diag.Note($"prompt: {DescribePrompt(p)}");
    }

    /// <summary>
    /// Tags a card selection with the screen it was opened from, when backing out of it (Cancel) would only return to
    /// that screen unchanged: the removal of a CardRemovalReward (RewardSynchronizer.DoUnsyncedCardRemoval returns false,
    /// the reward stays), Smith and Cook (RestSiteSynchronizer.ChooseOption returns false, the options stay), and the
    /// merchant's card removal (OneOffSynchronizer.DoMerchantCardRemoval returns false, no gold is spent, Used stays false).
    /// </summary>
    internal static void TagOrigin(CardSelectPrompt p)
    {
        var self = Current;
        if (self == null || !p.Cancelable) return;
        if (self._openingReward is { Reward: CardRemovalReward } o && p.Ctx.Kind == "deck_remove")
        {
            p.FromReward = o;
            self._openingReward = null;
        }
        else if (self._openingService is { } svc && p.Ctx.Kind == (svc == "rest:SMITH" ? "deck_upgrade" : "deck_remove"))
        {
            p.RoomService = svc;
            self._openingService = null;
        }
    }

    private static readonly FieldInfo _cardRewardCards = typeof(CardReward).GetField("_cards", BindingFlags.Instance | BindingFlags.NonPublic)!;

    /// <summary>The rewards screen a card reward screen is being opened from (CardReward.OnSelect passes its own _cards).</summary>
    internal static RewardOrigin? TakeCardRewardOrigin(object options)
    {
        var self = Current;
        if (self?._openingReward is { Reward: CardReward cr } o && ReferenceEquals(_cardRewardCards.GetValue(cr), options))
        {
            self._openingReward = null;
            return o;
        }
        return null;
    }

    private static string DescribePrompt(Prompt p) => p switch
    {
        CardSelectPrompt cs => $"card_select#{cs.Seq} {cs.Ctx.Kind} src={cs.Ctx.Source ?? "-"} n={cs.Options.Count} min={cs.Min} max={cs.Max}",
        _ => $"{p.Decision}#{p.Seq} {p.GetType().Name}",
    };

    private Prompt? TopPrompt()
    {
        // Card reward and rewards prompts re-register each time the game waits on them again; a sphere
        // prompt is only momentarily not waiting while a click resolves.
        _prompts.RemoveAll(p => !p.IsWaiting && !(p is SpherePrompt sp && !sp.Game.IsFinished));
        Prompt? top = null;
        foreach (var p in _prompts)
            if (p.IsWaiting && (top == null || p.Seq > top.Seq)) top = p;
        return top;
    }

    // ───────────────────────────── observation of game events

    private void OnFault(Exception e)
    {
        if (_suppressFaults || e is OperationCanceledException) return;
        _fault ??= FaultText(e);
    }

    private static string Shorten(string s) => s.Length > 3000 ? s[..3000] : s;

    /// <summary>Exception message, then the compact game context, then the stack (log lines get cut from the end).</summary>
    internal string FaultText(Exception e) => $"{e.GetType().Name}: {e.Message} | context: {ErrorContext()} | {Shorten(e.ToString())}";

    /// <summary>
    /// Compact context for error reports: where the run is, what the game and the harness are waiting on, and the
    /// last game events (actions with their source models, choices, combat and room transitions, player inputs).
    /// </summary>
    internal string ErrorContext()
    {
        var sb = new System.Text.StringBuilder();
        try
        {
            var rs = _runState;
            var cs = CombatManager.Instance.DebugOnlyGetState();
            sb.Append($"room={rs?.CurrentRoom?.GetType().Name}");
            if (rs?.CurrentRoom is CombatRoom cr) sb.Append($"({cr.Encounter?.Id.Entry})");
            sb.Append($" act={rs?.CurrentActIndex + 1} floor={rs?.TotalFloor} combat={CombatManager.Instance.IsInProgress}");
            if (cs != null) sb.Append($" round={cs.RoundNumber} side={cs.CurrentSide}");
            sb.Append($" phase={rs?.Players[0].PlayerCombatState?.Phase} sync={RunManager.Instance.ActionQueueSynchronizer?.CombatState}");
            var running = RunManager.Instance.ActionExecutor?.CurrentlyRunningAction;
            sb.Append($" running={(running == null ? "-" : Diag.Describe(running))}");
            var waiting = _prompts.Where(p => p.IsWaiting).Select(DescribePrompt).ToList();
            sb.Append($" prompts=[{string.Join("; ", waiting)}]");
            sb.Append(" last: ").Append(string.Join(" / ", Diag.Last(20)));
        }
        catch (Exception x) { sb.Append($" (context failed: {x.GetType().Name})"); }
        return sb.ToString();
    }

    internal static void OnWinRun() { if (Current != null) Current._won = true; }

    /// <summary>An exception surfaced outside the game loop (unobserved task, finalizer thread).</summary>
    internal static void ReportBackgroundFault(Exception e) => Current?.OnFault(e.GetBaseException());

    /// <summary>NEventRoom.Proceed: the map opens; the room is left at the next map move.</summary>
    internal static void OnMapOpened()
    {
        var self = Current;
        if (self?._runState != null) self._travelRoom = self._runState.CurrentRoom;
    }

    /// <summary>
    /// NCombatUi.OnCombatWon: rewards if the encounter gives any, otherwise ProceedWithoutRewards. The client
    /// does this after the death animations / a timed wait, i.e. once everything EndCombatInternal started has
    /// settled; PostWhenIdle gives the same ordering.
    /// </summary>
    private void OnCombatWon(CombatRoom room)
    {
        GameLoop.Instance.PostWhenIdle(() =>
        {
            if (room.Encounter.ShouldGiveRewards) TaskHelper.RunSafely(room.OfferRoomEndRewards());
            else TaskHelper.RunSafely(ProceedWithoutRewards(room));
        });
    }

    private async Task ProceedWithoutRewards(CombatRoom room)
    {
        await RunManager.Instance.ProceedFromTerminalRewardsScreenAsync();
        AfterTerminalProceed(room);
    }

    private void AfterTerminalProceed(AbstractRoom room)
    {
        // ProceedFromTerminalRewardsScreen either resumed the parent event (room changed) or opened the map.
        if (_runState != null && ReferenceEquals(_runState.CurrentRoom, room)) _travelRoom = room;
    }

    private void OnRoomEntered()
    {
        // NTreasureRoom: the chest button waits for the player.
        if (_runState?.CurrentRoom is TreasureRoom tr)
            GameLoop.Instance.PostWhenIdle(() => TaskHelper.RunSafely(ChestRoom(tr)));
    }

    /// <summary>
    /// RewardsSet.Offer in TestMode calls this instead of NRewardsScreen.ShowScreen. The prompt lets the player
    /// take any reward (RewardsSetSynchronizer.SelectLocalReward, as NRewardButton does) or leave. Leaving
    /// skips the rest (SkipLocalRewardsSet, which Offer requires before returning in TestMode); for the
    /// terminal screen of a combat room the proceed button's logic then runs (NRewardsScreen.OnProceedButtonPressed).
    /// </summary>
    internal async Task RewardsLoop(RewardsSet set)
    {
        var sync = RunManager.Instance.RewardsSetSynchronizer;
        var p = new RewardsPrompt(set, set.Room is CombatRoom);
        while (true)
        {
            // A non-terminal screen closes itself once everything is taken (NRewardsScreen.UpdateScreenState).
            if (!p.Terminal && sync.IsRewardsSetCompleted(set)) break;
            p.Tcs = new TaskCompletionSource<RewardChoice>();
            AddPrompt(p);
            var c = await p.Tcs.Task;
            if (c.Proceed) break;
            var reward = set.Rewards[c.Index];
            var outer = _openingReward;
            _openingReward = new RewardOrigin(p, reward);
            Task<bool> select;
            try { select = sync.SelectLocalReward(reward); }
            finally { _openingReward = outer; }
            await select;
            if (reward.SuccessfullySelected) p.Held.Clear();
        }
        if (!sync.IsRewardsSetCompleted(set)) sync.SkipLocalRewardsSet();
        if (p.Terminal && set.Room != null)
        {
            var room = set.Room;
            GameLoop.Instance.Post(_ => TaskHelper.RunSafely(TerminalProceed(room)), null);
        }
    }

    /// <summary>NRewardsScreen.OnProceedButtonPressed for a terminal (combat room) rewards screen.</summary>
    private async Task TerminalProceed(AbstractRoom room)
    {
        var rs = _runState;
        if (rs == null) return;
        if (room.RoomType == RoomType.Boss || room.IsVictoryRoom)
        {
            if (rs.Map.SecondBossMapPoint != null && rs.CurrentMapCoord == rs.Map.BossMapPoint.coord)
            {
                await RunManager.Instance.ProceedFromTerminalRewardsScreenAsync();
                AfterTerminalProceed(room);
            }
            else RunManager.Instance.ActChangeSynchronizer.SetLocalPlayerReady();
            return;
        }
        await RunManager.Instance.ProceedFromTerminalRewardsScreenAsync();
        AfterTerminalProceed(room);
    }

    /// <summary>
    /// NTreasureRoom: wait for the chest button, then OpenChest: gold (DoNormalRewards), extra rewards, and the relic
    /// collection, where the relic is taken (PickRelicLocally) or skipped (SkipRelicLocally + proceed).
    /// </summary>
    private async Task ChestRoom(TreasureRoom room)
    {
        var chest = new ChestPrompt(room);
        AddPrompt(chest);
        await chest.Tcs.Task;
        await room.DoNormalRewards();
        await room.DoExtraRewardsIfNeeded();
        var sync = RunManager.Instance.TreasureRoomRelicSynchronizer;
        var relics = sync.CurrentRelics?.ToList();
        if (relics == null || relics.Count == 0)
        {
            _travelRoom = room;
            return;
        }
        var p = new TreasurePrompt(relics);
        AddPrompt(p);
        var pick = await p.Tcs.Task;
        if (pick == null)
        {
            sync.SkipRelicLocally();
            await RunManager.Instance.ProceedFromTerminalRewardsScreenAsync();
            _travelRoom = room;
        }
        else sync.PickRelicLocally(pick.Value);   // award arrives through RelicsAwarded
    }

    /// <summary>NTreasureRoomRelicCollection.AnimateRelicAwards: the UI grants the awarded relics after its animation.</summary>
    private void OnRelicsAwarded(List<RelicPickingResult> results, bool isFinished)
    {
        var room = _runState?.CurrentRoom;
        GameLoop.Instance.Post(_ =>
        {
            foreach (var r in results)
                if (r.type != RelicPickingResultType.Skipped && r.player != null)
                    TaskHelper.RunSafely(RelicCmd.Obtain(r.relic.ToMutable(), r.player));
            if (room != null) _travelRoom = room;
        }, null);
    }

    // ───────────────────────────── stable-state detection

    private void Pump()
    {
        int guard = 0;
        while (GameLoop.Instance.RunUntilIdle() > 0)
            if (++guard > 1_000_000) { _fault ??= "game loop did not settle"; break; }
    }

    /// <summary>Drive a game task that runs before the first decision (setup steps).</summary>
    private void RunTask(Task t, string what)
    {
        int idle = 0;
        while (!t.IsCompleted)
        {
            if (GameLoop.Instance.RunUntilIdle() > 0) { idle = 0; continue; }
            if (t.IsCompleted || TopPrompt() != null) break;
            if (!GameLoop.Instance.WaitForPost(50) && ++idle > 40)
                throw new InvalidOperationException($"{what} did not complete");
        }
        if (t.IsFaulted) throw t.Exception!.GetBaseException();
    }

    internal Decision Decide()
    {
        for (int attempt = 0; ; attempt++)
        {
            Pump();
            var d = TryDecide();
            if (d != null) return d;
            if (attempt >= 40) return Err("stalled: no decision reachable: " + DescribeState() + " | context: " + ErrorContext());
            GameLoop.Instance.WaitForPost(attempt < 5 ? 1 : 50);
        }
    }

    /// <summary>
    /// The stable state, or null while the game is still working. Terminal integrity: a run only ends as a death
    /// by the game's rules (every player creature at 0 HP, RunState.IsGameOver), a victory (RunManager.WinRun ran
    /// in the last act), or an abandon (RunManager.IsAbandoned, i.e. Trial's Double Down). Anything else that
    /// looks terminal is reported as an error with diagnostics, never as game_over.
    /// </summary>
    private Decision? TryDecide()
    {
        if (_fault != null) return Err(_fault);
        if (_runState == null) return Err("No run in progress");
        // Single-thread invariant: a post from another thread means game code ran off the game thread.
        if (GameLoop.Instance.ForeignPosts > 0)
            return Err("integrity: game code ran on a foreign thread (posted to the game loop from it): " + DescribeState()
                       + " | first foreign post: " + Shorten(GameLoop.Instance.FirstForeignPost ?? "?"));
        var me = Me;
        var rs = _runState;
        bool dead = me.Creature == null || me.Creature.IsDead;
        if (_won)
        {
            if (rs.CurrentActIndex != rs.Acts.Count - 1)
                return Err($"integrity: WinRun outside the last act (act {rs.CurrentActIndex + 1}/{rs.Acts.Count}): " + DescribeState());
            return new Decision { Kind = "game_over", Done = true, Victory = true };
        }
        if (RunManager.Instance.IsAbandoned)
        {
            if (!dead) return Err("integrity: run abandoned but the player is alive: " + DescribeState());
            return new Decision { Kind = "game_over", Done = true, Victory = false, Abandoned = true };
        }
        if (dead || rs.IsGameOver)
        {
            if (!(dead && rs.IsGameOver && (me.Creature?.CurrentHp ?? 0) <= 0))
                return Err($"integrity: inconsistent defeat (dead={dead} gameOver={rs.IsGameOver} hp={me.Creature?.CurrentHp}): " + DescribeState());
            // The death only ends the run once the kill has finished (LoseCombat, death hooks).
            if (!GameLoop.Instance.IsEmpty) return null;
            return new Decision { Kind = "game_over", Done = true, Victory = false };
        }

        var top = TopPrompt();
        if (top != null) return new Decision { Kind = top.Decision, Prompt = top };

        var executor = RunManager.Instance.ActionExecutor;
        if (CombatManager.Instance.IsInProgress)
        {
            if (me.PlayerCombatState?.Phase == PlayerTurnPhase.Play && !executor.IsRunning && GameLoop.Instance.IsEmpty
                && !CombatManager.Instance.PlayerActionsDisabled)
                return new Decision { Kind = "combat_play" };
            return null;
        }
        if (executor.IsRunning || !GameLoop.Instance.IsEmpty) return null;
        var room = rs.CurrentRoom;
        if (room == null) return null;
        if (room is MapRoom || ReferenceEquals(room, _travelRoom)) return new Decision { Kind = "map_select" };
        switch (room)
        {
            case EventRoom:
            {
                var ev = LocalEvent();
                if (ev == null) return null;
                if (ev is FakeMerchant fm && !fm.StartedFight) return new Decision { Kind = "fake_merchant" };
                // A finished event shows a synthetic PROCEED whose only effect is opening the map.
                if (ev.IsFinished) return new Decision { Kind = "map_select" };
                if (ev.CurrentOptions.Count > 0) return new Decision { Kind = "event_choice" };
                return null;
            }
            case RestSiteRoom rsr:
                // NRestSiteRoom: with no options left the proceed button is the only control (it opens the map).
                if (rsr.Options.Count == 0) return new Decision { Kind = "map_select" };
                return new Decision { Kind = "rest_site" };
            case MerchantRoom: return new Decision { Kind = "shop" };
        }
        return null;
    }

    internal EventModel? LocalEvent()
    {
        try { return RunManager.Instance.EventSynchronizer.GetLocalEvent(); }
        catch { return null; }
    }

    private string DescribeState()
    {
        var rs = _runState;
        var me = rs?.Players[0];
        string ev = "";
        try { ev = rs?.CurrentRoom is EventRoom ? $" event={LocalEvent()?.Id} finished={LocalEvent()?.IsFinished} opts={LocalEvent()?.CurrentOptions.Count}" : ""; } catch { }
        return $"act={rs?.CurrentActIndex} floor={rs?.ActFloor} room={rs?.CurrentRoom?.GetType().Name} rooms={rs?.CurrentRoomCount}{ev} " +
               $"combat={CombatManager.Instance.IsInProgress} phase={me?.PlayerCombatState?.Phase} hp={me?.Creature?.CurrentHp} " +
               $"executor={RunManager.Instance.ActionExecutor?.IsRunning} action={RunManager.Instance.ActionExecutor?.CurrentlyRunningAction} " +
               $"prompts={_prompts.Count} loopEmpty={GameLoop.Instance.IsEmpty} foreignPosts={GameLoop.Instance.ForeignPosts} " +
               $"travel={_travelRoom != null} gameOver={rs?.IsGameOver} abandoned={RunManager.Instance.IsAbandoned}";
    }

    private static Decision Err(string msg) => new() { Kind = "error", Error = msg };

    // ───────────────────────────── legality helpers (mirroring the UI's enable rules)

    /// <summary>Enemies still in the fight (dead ones that stay to revive included): the targets' indices.</summary>
    /// <summary>MonoMod (inside Harmony) extracts a native helper to /tmp/mm-exhelper.so.XXXXXX on Linux and never
    /// deletes it: one file per engine start, hundreds of thousands over a training run on a shared /tmp. Once the
    /// library is mapped the file can go; the mapping stays valid.</summary>
    private static void RemoveMonoModHelper()
    {
        try
        {
            if (!File.Exists("/proc/self/maps")) return;
            foreach (var line in File.ReadLines("/proc/self/maps"))
            {
                int i = line.IndexOf("/tmp/mm-exhelper", StringComparison.Ordinal);
                if (i < 0) continue;
                var path = line[i..].Trim();
                if (path.EndsWith(" (deleted)")) continue;
                File.Delete(path);
                break;
            }
        }
        catch { }
    }

    internal static List<Creature> CombatEnemies() => ObsShared.CombatEnemies();

    /// <summary>NTargetManager: an enemy can be targeted while alive, hittable and no hook forbids it.</summary>
    internal static bool Targetable(Creature c) => ObsShared.Targetable(c);

    /// <summary>NPlayerHand.IsInCardSelection / an ICardSelector overlay: a card selection prompt is open.</summary>
    private bool InCardSelectScreen => TopPrompt() is CardSelectPrompt or BundlePrompt or CardRewardPrompt;

    /// <summary>NPotionPopup's use-button rules (plus NPotionHolder.DisableUntilPotionRemoved).</summary>
    internal bool CanUsePotion(PotionModel p) => ObsShared.CanUsePotion(Me, p, _pendingDiscards.Contains(p), () => InCardSelectScreen);

    internal bool CanDiscardPotion(PotionModel p) => ObsShared.CanDiscardPotion(Me, p, _pendingDiscards.Contains(p));

    /// <summary>NMapScreen.RecalculateTravelability + MapTravel.GetTravelablePointsFrom (free travel: Winged Boots, Flight).</summary>
    internal List<MapPoint> TravelablePoints() => ObsShared.TravelablePoints(_runState!);

    // ───────────────────────────── player inputs (the calls the UI makes)

    internal void PlayCard(CardModel card, Creature? target)
    {
        if (!card.TryManualPlay(target)) _fault ??= $"TryManualPlay refused {card.Id} -> {target?.Monster?.Id}";
    }

    internal void EndTurn() =>
        RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(Me, Me.PlayerCombatState!.TurnNumber));

    /// <summary>NPotionHolder.UsePotion: targeted potions get the chosen creature, Self gets the owner, others null.</summary>
    internal void UsePotion(PotionModel p, Creature? target)
    {
        if (target == null && p.TargetType == TargetType.Self) target = p.Owner.Creature;
        p.EnqueueManualUse(target);
    }

    internal void DiscardPotion(PotionModel p)
    {
        int slot = Me.PotionSlots.ToList().IndexOf(p);
        if (slot < 0) { _fault ??= $"discard: {p.Id} not in the belt"; return; }
        _pendingDiscards.Add(p);
        RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new DiscardPotionGameAction(Me, (uint)slot, CombatManager.Instance.IsInProgress));
    }

    internal void TravelTo(MapPoint point)
    {
        var rs = _runState!;
        var sync = RunManager.Instance.MapSelectionSynchronizer;
        var source = new MapLocation(rs.CurrentMapCoord, rs.CurrentActIndex);
        // MapSelectionSynchronizer silently drops a vote from any other source: the click would do nothing.
        if (_acceptingVotesFrom.GetValue(sync) is MapLocation accepting && accepting != source)
        {
            _fault ??= $"integrity: map vote source {source} is not the one being accepted ({accepting})";
            return;
        }
        var vote = new MapVote { coord = point.coord, mapGenerationCount = sync.MapGenerationCount };
        RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new VoteForMapCoordAction(Me, source, vote));
    }

    private static readonly FieldInfo _acceptingVotesFrom =
        typeof(MapSelectionSynchronizer).GetField("_acceptingVotesFromSource", BindingFlags.Instance | BindingFlags.NonPublic)!;

    internal void ChooseEventOption(int index)
    {
        var ev = LocalEvent()!;
        var opt = ev.CurrentOptions[index];
        if (opt.IsProceed) TaskHelper.RunSafely(opt.Chosen());     // NEventRoom.OptionButtonClicked
        else RunManager.Instance.EventSynchronizer.ChooseLocalOption(index);
    }

    internal void ChooseRestOption(int index)
    {
        var room = _runState!.CurrentRoom!;
        TaskHelper.RunSafely(RestChoice(index, room));
    }

    private async Task RestChoice(int index, AbstractRoom room)
    {
        var option = ((RestSiteRoom)room).Options[index];
        _openingService = option is SmithRestSiteOption ? "rest:SMITH" : option is CookRestSiteOption ? "rest:COOK" : null;
        Task<bool> choose;
        try { choose = RunManager.Instance.RestSiteSynchronizer.ChooseLocalOption(index); }
        finally { _openingService = null; }
        if (await choose) _restOkRoom = room;
    }

    internal bool RestSucceeded => ReferenceEquals(_restOkRoom, _runState?.CurrentRoom);

    internal void LeaveRoom() => _travelRoom = _runState!.CurrentRoom;

    /// <summary>NMerchantSlot / NFakeMerchant: OnTryPurchaseWrapper(inventory), as AutoSlay's ShopRoomHandler does.</summary>
    internal void Purchase(MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry entry, MegaCrit.Sts2.Core.Entities.Merchant.MerchantInventory inv)
    {
        if (entry is MegaCrit.Sts2.Core.Entities.Merchant.MerchantCardRemovalEntry rem)
        {
            _openingService = "shop:REMOVE";
            Task<bool> purchase;
            try { purchase = RunManager.Instance.MerchantSynchronizer.PurchaseLocalEntry(rem); }
            finally { _openingService = null; }
            TaskHelper.RunSafely(purchase);
        }
        else
            TaskHelper.RunSafely(RunManager.Instance.MerchantSynchronizer.PurchaseLocalEntry(entry));
    }
}
