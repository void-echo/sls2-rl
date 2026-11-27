using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using HarmonyLib;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Models.Monsters;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Models.Powers;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Audio;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves.Managers;
using MegaCrit.Sts2.Core.TestSupport;
using Sts2Obs;

namespace Sts2Headless;

/// <summary>
/// Every Harmony patch the harness applies, grouped by purpose. The rule for this file: a patch may
/// (a) stand in for a UI node by doing exactly what the node does to game objects, (b) skip pure
/// presentation (audio, VFX, screen shake) whose singleton does not exist headless, (c) put a
/// TestMode-only gameplay branch back to the shipping branch, (d) keep work on the game thread, or
/// (e) keep the timing of client I/O the harness skips (the run save), or (f) record, without changing anything,
/// what the player sees happen for the observation (cards put on top of the draw pile). It must never decide for
/// the player or invent game rules. Per-item fixes live in <see cref="ItemFixes"/> so new findings are one line.
/// </summary>
internal static class Patches
{
    private static Harmony _h = null!;
    private static readonly List<string> _log = new();

    public static IReadOnlyList<string> Log => _log;

    public static void ApplyAll()
    {
        _h = new Harmony("sts2headless.v1");
        PatchSelectionCapture();
        PatchScreens();
        PatchAudioSinks();
        PatchTestModeBranches();
        PatchUiResidentLogic();
        PatchScheduling();
        PatchSaving();
        PatchObservationRecords();
        foreach (var fix in ItemFixes.All) Apply(fix);
    }

    // ───────────────────────────── helpers

    private static MethodInfo M(Type t, string name, params Type[]? args)
    {
        var m = args == null || args.Length == 0
            ? AccessTools.Method(t, name)
            : AccessTools.Method(t, name, args);
        return m ?? throw new MissingMethodException(t.FullName, name);
    }

    private static HarmonyMethod H(string name) =>
        new(typeof(Patches).GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public)
            ?? throw new MissingMethodException(nameof(Patches), name));

    private static void Patch(MethodBase target, string? prefix = null, string? postfix = null, string? transpiler = null, string? finalizer = null)
    {
        _h.Patch(target,
            prefix == null ? null : H(prefix),
            postfix == null ? null : H(postfix),
            transpiler == null ? null : H(transpiler),
            finalizer == null ? null : H(finalizer));
        _log.Add($"{target.DeclaringType?.Name}.{target.Name}: {prefix ?? postfix ?? transpiler ?? finalizer}");
    }

    // ───────────────────────────── selection semantics

    private static readonly Dictionary<string, string> _selectionKinds = new()
    {
        ["FromChooseACardScreen"] = "choose_a_card",
        ["FromSimpleGridForRewards"] = "grid_rewards",
        ["FromSimpleGrid"] = "grid",
        ["FromCombatPile"] = "combat_pile",
        ["FromDeckForUpgrade"] = "deck_upgrade",
        ["FromDeckForTransformation"] = "deck_transform",
        ["FromDeckForEnchantment"] = "deck_enchant",
        ["FromDeckForRemoval"] = "deck_remove",
        ["FromDeckGeneric"] = "deck",
        ["FromHand"] = "hand",
        ["FromHandForDiscard"] = "hand_discard",
        ["FromHandForUpgrade"] = "hand_upgrade",
    };

    private static void PatchSelectionCapture()
    {
        foreach (var m in typeof(CardSelectCmd).GetMethods(BindingFlags.Public | BindingFlags.Static))
            if (_selectionKinds.ContainsKey(m.Name))
                Patch(m, prefix: nameof(SelectionPrefix), finalizer: nameof(SelectionFinalizer));
    }

    /// <summary>
    /// Runs before the synchronous part of a CardSelectCmd entry. With a selector installed, every entry calls
    /// Selector.GetSelectedCards before its first await, so the context is still current when the selector reads it.
    /// Only the outermost entry sets it (FromDeckForRemoval forwards to FromDeckGeneric).
    /// </summary>
    private static void SelectionPrefix(MethodBase __originalMethod, object[] __args, out bool __state)
    {
        __state = false;
        if (SelectionCapture.Depth++ > 0) return;
        var ctx = new SelectionContext { Kind = _selectionKinds[__originalMethod.Name] };
        foreach (var a in __args)
        {
            switch (a)
            {
                case CardSelectorPrefs prefs:
                    ctx.Min = prefs.MinSelect;
                    ctx.Max = prefs.MaxSelect;
                    ctx.Cancelable = prefs.Cancelable;
                    ctx.RequireConfirm = prefs.RequireManualConfirmation;
                    ctx.PromptKey = SafeKey(prefs.Prompt);
                    break;
                case CardPile pile:
                    ctx.Pile = pile.Type.ToString();
                    break;
                case AbstractModel src:
                    ctx.Source = src.Id.ToString();
                    break;
            }
        }
        if (__originalMethod.Name == "FromChooseACardScreen")
        {
            var ps = __originalMethod.GetParameters();
            for (int i = 0; i < ps.Length; i++)
                if (ps[i].Name == "canSkip") ctx.CanSkip = __args[i] is bool b && b;
        }
        if (__originalMethod.Name == "FromHandForUpgrade") { ctx.Min = 1; ctx.Max = 1; }
        SelectionCapture.Current = ctx;
        __state = true;
    }

    private static Exception? SelectionFinalizer(Exception? __exception, bool __state)
    {
        SelectionCapture.Depth = Math.Max(0, SelectionCapture.Depth - 1);
        // Only the entry that set the context clears it, and only in the caller's flow (see SelectionCapture).
        if (__state) SelectionCapture.Current = null;
        return __exception;
    }

    private static string? SafeKey(MegaCrit.Sts2.Core.Localization.LocString? s)
    {
        try { return s == null ? null : s.LocTable + "." + s.LocEntryKey; } catch { return null; }
    }

    // ───────────────────────────── screen stand-ins

    private static readonly ConditionalWeakTable<object, BundlePrompt> _bundleByScreen = new();

    private static void PatchScreens()
    {
        // Card reward screen: CardReward.OnSelect drives it (ShowScreen, OptionSelected loop, RefreshOptions on reroll).
        var crs = typeof(NCardRewardSelectionScreen);
        Patch(M(crs, "ShowScreen"), prefix: nameof(CardRewardShowPrefix));
        Patch(M(crs, "OptionSelected"), prefix: nameof(CardRewardOptionSelectedPrefix));
        Patch(M(crs, "RefreshOptions"), prefix: nameof(CardRewardRefreshPrefix));
        Patch(M(crs, "GetCardHolder"), prefix: nameof(CardHolderNullPrefix));

        // Choose-a-bundle: the TestMode branch picks bundles[0]; run the shipping branch against a stand-in screen.
        Patch(M(typeof(CardSelectCmd), "FromChooseABundleScreen"), prefix: nameof(TestModeOffPrefix), finalizer: nameof(TestModeRestoreFinalizer));
        var bs = typeof(NChooseABundleSelectionScreen);
        Patch(M(bs, "ShowScreen"), prefix: nameof(BundleShowPrefix));
        Patch(M(bs, "CardsSelected"), prefix: nameof(BundleCardsSelectedPrefix));

        // Crystal Sphere: the minigame model is complete; only the screen is missing.
        Patch(M(typeof(NCrystalSphereScreen), "ShowScreen"), prefix: nameof(SphereShowPrefix));
    }

    private static T NewNode<T>() where T : class => (T)RuntimeHelpers.GetUninitializedObject(typeof(T));

    private static bool CardRewardShowPrefix(IReadOnlyList<CardCreationResult> options,
        IReadOnlyList<MegaCrit.Sts2.Core.Entities.CardRewardAlternatives.CardRewardAlternative> extraOptions,
        ref NCardRewardSelectionScreen? __result)
    {
        var screen = NewNode<NCardRewardSelectionScreen>();
        CardRewardPrompt.ByScreen.Add(screen, new CardRewardPrompt(options, extraOptions) { FromReward = RunSimulator.TakeCardRewardOrigin(options) });
        __result = screen;
        return false;
    }

    private static bool CardRewardOptionSelectedPrefix(NCardRewardSelectionScreen __instance, ref Task<int?> __result)
    {
        if (!CardRewardPrompt.ByScreen.TryGetValue(__instance, out var p))
            throw new InvalidOperationException("card reward screen without prompt");
        p.Tcs = new TaskCompletionSource<int?>();
        RunSimulator.AddPrompt(p);
        __result = p.Tcs.Task;
        return false;
    }

    private static bool CardRewardRefreshPrefix(NCardRewardSelectionScreen __instance, IReadOnlyList<CardCreationResult> options,
        IReadOnlyList<MegaCrit.Sts2.Core.Entities.CardRewardAlternatives.CardRewardAlternative> extraOptions)
    {
        if (CardRewardPrompt.ByScreen.TryGetValue(__instance, out var p))
        {
            p.Options = options;
            p.Alts = extraOptions;
        }
        return false;
    }

    private static bool CardHolderNullPrefix(ref MegaCrit.Sts2.Core.Nodes.Cards.Holders.NCardHolder? __result)
    {
        __result = null;
        return false;
    }

    private static bool TalkNullPrefix(ref MegaCrit.Sts2.Core.Nodes.Vfx.NSpeechBubbleVfx? __result)
    {
        __result = null;
        return false;
    }

    private static bool BundleShowPrefix(IReadOnlyList<IReadOnlyList<CardModel>> bundles, ref NChooseABundleSelectionScreen __result)
    {
        var screen = NewNode<NChooseABundleSelectionScreen>();
        var p = new BundlePrompt(bundles);
        _bundleByScreen.Add(screen, p);
        RunSimulator.AddPrompt(p);
        __result = screen;
        return false;
    }

    private static bool BundleCardsSelectedPrefix(NChooseABundleSelectionScreen __instance, ref Task<IEnumerable<IReadOnlyList<CardModel>>> __result)
    {
        if (!_bundleByScreen.TryGetValue(__instance, out var p))
            throw new InvalidOperationException("bundle screen without prompt");
        __result = p.Tcs.Task;
        return false;
    }

    private static bool SphereShowPrefix(CrystalSphereMinigame grid, ref NCrystalSphereScreen? __result)
    {
        RunSimulator.AddPrompt(new SpherePrompt(grid));
        __result = null;
        return false;
    }

    // ───────────────────────────── presentation singletons that do not exist headless

    private static NAudioManager? _audio;
    private static NDebugAudioManager? _debugAudio;

    /// <summary>
    /// NAudioManager.Instance and NDebugAudioManager.Instance resolve through NGame.Instance, which is null
    /// headless; several gameplay methods call them without a null check (Crusher/Rocket.BeforeDeath,
    /// DenseVegetation.Rest, JungleMazeAdventure.SafetyInNumbers). Both return an inert object whose
    /// public methods do nothing. NGame itself is deliberately NOT stubbed: many `NGame.Instance?.…`
    /// paths would wake up.
    /// </summary>
    private static void PatchAudioSinks()
    {
        _audio = NewNode<NAudioManager>();
        _debugAudio = NewNode<NDebugAudioManager>();
        Patch(AccessTools.PropertyGetter(typeof(NAudioManager), "Instance"), prefix: nameof(AudioInstancePrefix));
        Patch(AccessTools.PropertyGetter(typeof(NDebugAudioManager), "Instance"), prefix: nameof(DebugAudioInstancePrefix));
        foreach (var t in new[] { typeof(NAudioManager), typeof(NDebugAudioManager) })
            foreach (var m in t.GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.DeclaredOnly))
            {
                if (m.IsGenericMethodDefinition || m.IsAbstract || m.Name.StartsWith("get_") || m.Name.StartsWith("set_")) continue;
                if (typeof(Task).IsAssignableFrom(m.ReturnType)) continue;
                try { _h.Patch(m, H(nameof(SkipOriginal))); } catch { }
            }
        _log.Add("audio sinks installed");
    }

    private static bool AudioInstancePrefix(ref NAudioManager? __result) { __result = _audio; return false; }
    private static bool DebugAudioInstancePrefix(ref NDebugAudioManager? __result) { __result = _debugAudio; return false; }
    private static bool SkipOriginal() => false;

    // ───────────────────────────── TestMode gameplay branches put back to the shipping behaviour

    private static void PatchTestModeBranches()
    {
        // Potion price jitter draws from PlayerRng.Shops; skipping it desyncs every later shop.
        Patch(M(typeof(MerchantPotionEntry), "CalcCost"), prefix: nameof(TestModeOffPrefix), finalizer: nameof(TestModeRestoreFinalizer));
        // Fixed test relics / potions instead of random ones.
        Patch(M(typeof(CallingBell), "GenerateRewards"), prefix: nameof(TestModeOffPrefix), finalizer: nameof(TestModeRestoreFinalizer));
        Patch(M(typeof(Cauldron), "GenerateRewards"), prefix: nameof(TestModeOffPrefix), finalizer: nameof(TestModeRestoreFinalizer));
        // One multi-target hit instead of the per-creature hits the VFX drives in the client.
        Patch(M(typeof(RollingBoulderPower), "AfterPlayerTurnStart"), prefix: nameof(RollingBoulderPrefix));
    }

    private static void TestModeOffPrefix(out bool __state)
    {
        __state = TestMode.IsOn;
        TestMode.IsOn = false;
    }

    private static Exception? TestModeRestoreFinalizer(bool __state, Exception? __exception)
    {
        TestMode.IsOn = __state;
        return __exception;
    }

    private static readonly MethodInfo _boulderDamage = AccessTools.Method(typeof(RollingBoulderPower), "DoDamage");
    private static readonly MethodInfo _powerFlash = AccessTools.Method(typeof(PowerModel), "Flash");

    private static bool RollingBoulderPrefix(RollingBoulderPower __instance, PlayerChoiceContext choiceContext, Player player, ref Task __result)
    {
        __result = RollingBoulder(__instance, choiceContext, player);
        return false;
    }

    /// <summary>
    /// Shipping branch of RollingBoulderPower.AfterPlayerTurnStart: the boulder VFX emits HitCreature for
    /// each hittable enemy as it rolls past (left to right, i.e. slot order) and each hit calls DoDamage on
    /// that single creature. Reproduced as single-target DoDamage calls in enemy order.
    /// </summary>
    private static async Task RollingBoulder(RollingBoulderPower p, PlayerChoiceContext ctx, Player player)
    {
        if (player != p.Owner.Player) return;
        _powerFlash.Invoke(p, null);
        var tasks = new List<Task>();
        foreach (var c in p.Owner.CombatState!.HittableEnemies.ToList())
            tasks.Add((Task)_boulderDamage.Invoke(p, new object[] { ctx, new[] { c } })!);
        await Task.WhenAll(tasks);
        p.SetAmount(p.Amount + p.DynamicVars.Damage.IntValue);
    }

    // ───────────────────────────── game logic that lives in UI nodes

    private static void PatchUiResidentLogic()
    {
        // NMerchantCardRemoval sets Used / applies ShouldAllowMerchantCardRemoval when it fills its slot.
        Patch(M(typeof(MerchantInventory), "CreateForNormalMerchant"), postfix: nameof(MerchantInventoryPostfix));
        // NEventRoom.Proceed (synthetic PROCEED option, blocked ancients) only opens the map.
        Patch(M(typeof(NEventRoom), "Proceed"), prefix: nameof(EventProceedPrefix));
        // Foul Potion targets the merchant button: usable in a shop / Fake Merchant room when the inventory is closed.
        Patch(AccessTools.PropertyGetter(typeof(FoulPotion), "PassesCustomUsabilityCheck"), prefix: nameof(FoulPotionUsablePrefix));
        Patch(M(typeof(FoulPotion), "OnUse"), prefix: nameof(FoulPotionOnUsePrefix));
        // Victory and abandon are observed, not reimplemented.
        Patch(M(typeof(RunManager), "WinRun"), prefix: nameof(WinRunPrefix));
        Patch(M(typeof(Trial), "DoubleDown"), prefix: nameof(TrialDoubleDownPrefix));
    }

    private static void MerchantInventoryPostfix(MerchantInventory __result, Player player)
    {
        var rem = __result.CardRemovalEntry;
        if (rem == null) return;
        rem.PurchaseCompleted += (_, _) => rem.SetUsed();                       // NMerchantCardRemoval.OnCardRemovalUsed
        if (!Hook.ShouldAllowMerchantCardRemoval(player.RunState, player)) rem.SetUsed(); // NMerchantCardRemoval.FillSlot
    }

    private static bool EventProceedPrefix(ref Task __result)
    {
        RunSimulator.OnMapOpened();
        __result = Task.CompletedTask;
        return false;
    }

    private static bool InMerchantLikeRoom(Player owner)
    {
        var room = owner.RunState.CurrentRoom;
        return room is MerchantRoom || (room is EventRoom er && er.CanonicalEvent is FakeMerchant);
    }

    private static bool FoulPotionUsablePrefix(FoulPotion __instance, ref bool __result)
    {
        if (CombatManager.Instance.IsInProgress) return true;
        if (InMerchantLikeRoom(__instance.Owner))
        {
            __result = true;
            return false;
        }
        return true;
    }

    private static bool FoulPotionOnUsePrefix(FoulPotion __instance, ref Task __result)
    {
        if (CombatManager.Instance.IsInProgress) return true;
        if (__instance.Owner.RunState.CurrentRoom is EventRoom er && er.CanonicalEvent is FakeMerchant)
        {
            __result = FoulPotionAtFakeMerchant(__instance);
            return false;
        }
        return true;
    }

    /// <summary>FoulPotion.OnUse's Fake Merchant branch without its `Node is NFakeMerchant` guard (VFX only).</summary>
    private static async Task FoulPotionAtFakeMerchant(FoulPotion potion)
    {
        var tasks = new List<Task>();
        foreach (var player in potion.Owner.RunState.Players)
        {
            var fm = (FakeMerchant)RunManager.Instance.EventSynchronizer.GetEventForPlayer(player);
            tasks.Add(fm.FoulPotionThrown(potion));
        }
        await Task.WhenAll(tasks);
    }

    private static void WinRunPrefix() => RunSimulator.OnWinRun();

    /// <summary>
    /// Trial's Double Down opens NAbandonRunConfirmPopup (the option stays clickable if the popup is cancelled, so the
    /// only effective choice is confirming). Confirming calls RunManager.Abandon (NAbandonRunConfirmPopup.OnYesButtonPressed):
    /// IsAbandoned is set and every player is killed, a loss by the game's own rules.
    /// </summary>
    private static bool TrialDoubleDownPrefix(ref Task __result)
    {
        RunManager.Instance.Abandon();
        __result = Task.CompletedTask;
        return false;
    }

    // ───────────────────────────── scheduling

    private static readonly AccessTools.FieldRef<ActionExecutor, bool> _isPaused = AccessTools.FieldRefAccess<ActionExecutor, bool>("_isPaused");
    private static TaskCompletionSource _unpaused = new(TaskCreationOptions.RunContinuationsAsynchronously);

    private static void PatchScheduling()
    {
        // TestMode makes ActionExecutor.WaitForUnpause poll with Task.Delay(50) on a timer thread. Resume on
        // the game loop instead, after the call that unpaused it (the client resumes on the next frame).
        Patch(M(typeof(ActionExecutor), "WaitForUnpause"), prefix: nameof(WaitForUnpausePrefix));
        Patch(M(typeof(ActionExecutor), "Unpause"), postfix: nameof(UnpausePostfix));
        // TalkCmd.Play builds a speech bubble node; its return value is presentation only.
        Patch(M(typeof(TalkCmd), "Play"), prefix: nameof(TalkNullPrefix));
    }

    private static bool WaitForUnpausePrefix(ActionExecutor __instance, ref Task __result)
    {
        __result = WaitUnpaused(__instance);
        return false;
    }

    private static async Task WaitUnpaused(ActionExecutor ex)
    {
        if (NonInteractiveMode.AutoSlayerCheck()) return;
        while (_isPaused(ex))
            await _unpaused.Task;
    }

    /// <summary>Called between runs: waiters of the old run must never be resumed by the new one.</summary>
    internal static void ResetRunScoped()
    {
        _unpaused = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
    }

    private static void UnpausePostfix()
    {
        var t = _unpaused;
        _unpaused = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        t.TrySetResult();
    }

    // ───────────────────────────── saving: no file, same timing

    /// <summary>
    /// The client always saves a single-player run (NCharacterSelectScreen.StartNewSingleplayerRun passes shouldSave:
    /// true), so RunSaveManager.SaveRun(room) serializes the run and writes it through GodotFileIo.WriteFileAsync,
    /// whose FileAccessStream.WriteAsync (the base Stream implementation) runs the write on the thread pool: every
    /// awaited save yields, and its caller resumes on the main thread in a later frame. The harness starts runs with
    /// shouldSave: false, where the method returns a completed task and its callers run straight on. That changed
    /// game behaviour: CombatManager.EndCombatInternal awaits the save before SetCombatState(NotInCombat), and when
    /// the combat ends inside the inline continuation of a hook action's completion (GameAction.Execute's finally,
    /// _completionSource.TrySetResult -> HookPlayerChoiceContext.WaitForCompletion -> the turn loop ->
    /// CheckWinCondition), ActionQueueSet.CombatEnded cancelled the already-Finished hook action before
    /// ActionQueueSet.PopAction ran ("Tried to pop action GenericHookGameAction ... didn't find it in any queue").
    /// The stand-in writes nothing and completes like the write: once the work queued now has settled.
    /// </summary>
    private static void PatchSaving()
    {
        Patch(M(typeof(RunSaveManager), "SaveRun", typeof(AbstractRoom)), prefix: nameof(RunSavePrefix));
    }

    private static bool RunSavePrefix(AbstractRoom? preFinishedRoom, ref Task __result)
    {
        try { RunSimulator.Current?.CaptureSave(preFinishedRoom); }
        catch (Exception e) { RunSimulator.ReportBackgroundFault(e); throw; }
        var write = new TaskCompletionSource();
        GameLoop.Instance.PostWhenIdle(() => write.TrySetResult());
        __result = write.Task;
        return false;
    }

    // ───────────────────────────── (f) observation records: what the player sees happen, no effect on the game

    private static void PatchObservationRecords()
    {
        // KnownTop (engine/src/Shared/KnownTop.cs): CardPileCmd.Add, CardPileCmd.Shuffle, CardPile.RandomizeOrderInternal.
        foreach (var (target, prefix) in KnownTop.PatchTargets())
        {
            _h.Patch(target, prefix: new HarmonyMethod(prefix));
            _log.Add($"{target.DeclaringType?.Name}.{target.Name}: KnownTop.{prefix.Name}");
        }
    }

    // ───────────────────────────── per-item fixes (event / relic / monster code that touches UI singletons)

    internal enum FixKind
    {
        /// <summary>Replace LocalContext.IsMe(...) with false: the guarded blocks are presentation only.</summary>
        IsMeFalse,
        /// <summary>Make listed UI calls null-safe no-ops (screen shake, creature node lookups, Task.Delay).</summary>
        SafeUiCalls,
        /// <summary>
        /// Task.ContinueWith without a scheduler runs on the thread pool when called from an await continuation
        /// (TaskScheduler.Current is Default there): run it on the game loop instead (deterministic, single thread).
        /// </summary>
        ContinueOnGameLoop,
    }

    internal sealed record ItemFix(Type Type, string Method, FixKind Kind, string Why);

    private static void Apply(ItemFix fix)
    {
        // Declared methods only: hooks such as AfterDeath also exist on the model base classes.
        var declared = AccessTools.GetDeclaredMethods(fix.Type).Where(x => x.Name == fix.Method).ToList();
        if (declared.Count > 1) throw new AmbiguousMatchException($"item fix {fix.Type.Name}.{fix.Method}: {declared.Count} overloads");
        var m = declared.FirstOrDefault();
        if (m == null)
        {
            _log.Add($"[WARN] item fix target missing: {fix.Type.Name}.{fix.Method}");
            return;
        }
        MethodBase target = m.GetCustomAttribute<AsyncStateMachineAttribute>() != null ? AccessTools.AsyncMoveNext(m) : m;
        _currentFix = fix;
        _currentFixHits = 0;
        Patch(target, transpiler: fix.Kind switch
        {
            FixKind.IsMeFalse => nameof(IsMeFalseTranspiler),
            FixKind.ContinueOnGameLoop => nameof(ContinueOnGameLoopTranspiler),
            _ => nameof(SafeUiCallsTranspiler),
        });
        if (_currentFixHits == 0) _log.Add($"[WARN] item fix {fix.Type.Name}.{fix.Method} changed nothing (game update?)");
    }

    private static ItemFix? _currentFix;
    private static int _currentFixHits;

    private static IEnumerable<CodeInstruction> IsMeFalseTranspiler(IEnumerable<CodeInstruction> instructions)
    {
        foreach (var ins in instructions)
        {
            if ((ins.opcode == OpCodes.Call || ins.opcode == OpCodes.Callvirt) && ins.operand is MethodInfo mi
                && mi.DeclaringType == typeof(LocalContext) && mi.Name == "IsMe")
            {
                var p = mi.GetParameters()[0].ParameterType;
                ins.opcode = OpCodes.Call;
                ins.operand = p == typeof(Player) ? AccessTools.Method(typeof(UiShims), nameof(UiShims.IsMeFalse), new[] { typeof(Player) })
                                                  : AccessTools.Method(typeof(UiShims), nameof(UiShims.IsMeFalse), new[] { typeof(Creature) });
                _currentFixHits++;
            }
            yield return ins;
        }
    }

    private static IEnumerable<CodeInstruction> ContinueOnGameLoopTranspiler(IEnumerable<CodeInstruction> instructions)
    {
        foreach (var ins in instructions)
        {
            if ((ins.opcode == OpCodes.Call || ins.opcode == OpCodes.Callvirt) && ins.operand is MethodInfo mi
                && mi.DeclaringType == typeof(Task) && mi.Name == "ContinueWith" && !mi.IsGenericMethod)
            {
                var ps = mi.GetParameters();
                if (ps.Length == 1 && ps[0].ParameterType == typeof(Action<Task>))
                {
                    ins.opcode = OpCodes.Call;
                    ins.operand = AccessTools.Method(typeof(UiShims), nameof(UiShims.ContinueWithOnLoop));
                    _currentFixHits++;
                }
            }
            yield return ins;
        }
    }

    private static IEnumerable<CodeInstruction> SafeUiCallsTranspiler(IEnumerable<CodeInstruction> instructions)
    {
        foreach (var ins in instructions)
        {
            if ((ins.opcode == OpCodes.Call || ins.opcode == OpCodes.Callvirt) && ins.operand is MethodInfo mi)
            {
                MethodInfo? shim = null;
                if (mi.DeclaringType == typeof(NGame) && (mi.Name == "ScreenShakeTrauma" || mi.Name == "ScreenShake" || mi.Name == "ScreenRumble"))
                    shim = AccessTools.Method(typeof(UiShims), mi.Name);
                else if (mi.DeclaringType == typeof(NCombatRoom) && mi.Name == "GetCreatureNode")
                    shim = AccessTools.Method(typeof(UiShims), nameof(UiShims.GetCreatureNode));
                else if (mi.DeclaringType == typeof(Task) && mi.Name == "Delay" && mi.GetParameters().Length == 1)
                    shim = mi.GetParameters()[0].ParameterType == typeof(int)
                        ? AccessTools.Method(typeof(UiShims), nameof(UiShims.Delay), new[] { typeof(int) })
                        : AccessTools.Method(typeof(UiShims), nameof(UiShims.Delay), new[] { typeof(TimeSpan) });
                if (shim != null)
                {
                    ins.opcode = OpCodes.Call;
                    ins.operand = shim;
                    _currentFixHits++;
                }
            }
            yield return ins;
        }
    }
}

/// <summary>Static, null-safe replacements for instance calls on UI nodes (same stack shape: instance first).</summary>
public static class UiShims
{
    public static bool IsMeFalse(Player? player) => false;
    public static bool IsMeFalse(Creature? creature) => false;
    public static void ScreenShakeTrauma(NGame? game, ShakeStrength strength) { }
    public static void ScreenShake(NGame? game, ShakeStrength strength, ShakeDuration duration, float degAngle) { }
    public static void ScreenRumble(NGame? game, ShakeStrength strength, ShakeDuration duration, RumbleStyle style) { }
    public static NCreature? GetCreatureNode(NCombatRoom? room, Creature? creature) => room?.GetCreatureNode(creature);
    public static Task ContinueWithOnLoop(Task task, Action<Task> continuation) => task.ContinueWith(continuation, GameLoop.Scheduler);
    public static Task Delay(int ms) => Task.CompletedTask;
    public static Task Delay(TimeSpan t) => Task.CompletedTask;
}

/// <summary>
/// Per-item fixes. Each row names a game method that dereferences a UI singleton in a gameplay path and
/// how it is neutralised. Add rows here when the content review finds more.
/// </summary>
internal static class ItemFixes
{
    public static readonly Patches.ItemFix[] All =
    {
        new(typeof(WaterfallOfObviousSecrets), "EnterWaterfall", Patches.FixKind.IsMeFalse,
            "NEventRoom.Instance.SetPortrait only changes the event portrait"),
        new(typeof(DenseVegetation), "Rest", Patches.FixKind.IsMeFalse,
            "IsMe-guarded block plays audio and NGame.ScreenRumble; crashed before showing FIGHT"),
        new(typeof(PunchOff), "Nab", Patches.FixKind.SafeUiCalls, "NGame.Instance.ScreenShakeTrauma after the Injury curse"),
        new(typeof(Amalgamator), "CombineStrikes", Patches.FixKind.SafeUiCalls, "NGame.Instance.ScreenShakeTrauma + Task.Delay(300)"),
        new(typeof(Amalgamator), "CombineDefends", Patches.FixKind.SafeUiCalls, "NGame.Instance.ScreenShakeTrauma + Task.Delay(300)"),
        new(typeof(Trial), "Accept", Patches.FixKind.IsMeFalse, "NEventRoom.Instance.Layout before the trial roll"),
        new(typeof(Trial), "AddVfxAnchoredToPortrait", Patches.FixKind.IsMeFalse, "portrait VFX scene instantiation"),
        new(typeof(SoulNexus), "AfterDeath", Patches.FixKind.SafeUiCalls, "NCombatRoom.Instance.GetCreatureNode in the Died handler"),
        new(typeof(ActionQueueSynchronizer), "GetHookActionForId", Patches.FixKind.ContinueOnGameLoop,
            "ExecutionStartedTask.ContinueWith(HookActionStarted) ran on a pool thread, racing _hookActions (List) with the game thread"),
    };
}
