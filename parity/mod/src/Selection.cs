using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;

namespace ParityOracle;

/// <summary>What the selection entry point (CardSelectCmd.From*) was asked for; same capture as the engine's Patches.SelectionPrefix.</summary>
public sealed class SelectionContext
{
    public string Kind = "unknown";
    public int? Min, Max;
    public bool Cancelable;
    public bool RequireConfirm;
    public bool? CanSkip;
    public string? PromptKey;
    public string? Source;
    public string? Pile;
    public CardSelectorPrefs? Prefs;
}

/// <summary>
/// Harmony prefix/finalizer on every CardSelectCmd.From* entry (engine/src/Sts2Headless/Patches.cs, PatchSelectionCapture),
/// so the client digest carries the same ctx.sel / cancelable / source / pile / prompt and the same skip/confirm rules.
/// Applied only while a job runs.
/// </summary>
public static class SelectionCapture
{
    // AsyncLocal, not a static: the context must flow with the From* entry's own async continuation. A selection
    // started inside a turn-start or combat-start hook (HookPlayerChoiceContext) reaches the selector only after
    // SignalPlayerChoiceBegun resumed, i.e. after the entry returned to its caller; a static cleared by the
    // finalizer is gone by then (engine finding, see docs/survey/parity_report.md, v1 section).
    private static readonly AsyncLocal<SelectionContext?> _current = new();
    public static SelectionContext? Current { get => _current.Value; set => _current.Value = value; }

    private static readonly Dictionary<string, string> Kinds = new()
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

    public static void Apply(Harmony h)
    {
        var pre = new HarmonyMethod(typeof(SelectionCapture).GetMethod(nameof(Prefix), BindingFlags.Static | BindingFlags.NonPublic));
        var fin = new HarmonyMethod(typeof(SelectionCapture).GetMethod(nameof(Finalizer), BindingFlags.Static | BindingFlags.NonPublic));
        int n = 0;
        foreach (var m in typeof(CardSelectCmd).GetMethods(BindingFlags.Public | BindingFlags.Static))
            if (Kinds.ContainsKey(m.Name)) { h.Patch(m, prefix: pre, finalizer: fin); n++; }
        Entry.Log($"selection capture patches applied: {n}");
    }

    private static void Prefix(MethodBase __originalMethod, object[] __args, out bool __state)
    {
        __state = false;
        if (_current.Value != null) return;   // nested entry (FromDeckForRemoval → FromDeckGeneric): the outer one owns it
        var ctx = new SelectionContext { Kind = Kinds[__originalMethod.Name] };
        foreach (var a in __args)
        {
            switch (a)
            {
                case CardSelectorPrefs prefs:
                    ctx.Prefs = prefs;
                    ctx.Min = prefs.MinSelect;
                    ctx.Max = prefs.MaxSelect;
                    ctx.Cancelable = prefs.Cancelable;
                    ctx.RequireConfirm = prefs.RequireManualConfirmation;
                    try { ctx.PromptKey = prefs.Prompt == null ? null : prefs.Prompt.LocTable + "." + prefs.Prompt.LocEntryKey; } catch { }
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
        _current.Value = ctx;
        __state = true;
    }

    /// <summary>Clears the caller's flow only; the entry's async continuation keeps its captured copy.</summary>
    private static Exception? Finalizer(bool __state, Exception? __exception)
    {
        if (__state) _current.Value = null;
        return __exception;
    }
}

/// <summary>
/// The pending selection, with the engine's CardSelectPrompt semantics (Max clamped to the option count, Min to Max).
/// </summary>
public sealed class PendingSelection
{
    public readonly List<CardModel> Options;
    public readonly int Min, Max;
    public readonly bool Cancelable;
    public readonly SelectionContext Ctx;
    public readonly List<int> Picked = new();
    public readonly TaskCompletionSource<IEnumerable<CardModel>> Tcs = new(TaskCreationOptions.RunContinuationsAsynchronously);
    /// <summary>The removal reward this selection belongs to, when opened from a rewards screen (engine CardSelectPrompt.FromReward).</summary>
    public Nav.RewardOrigin? FromReward;
    /// <summary>Smith, Cook or the merchant's removal (engine CardSelectPrompt.RoomService): no cancel.</summary>
    public string? RoomService;
    /// <summary>Inputs other than a pick given while this selection was on top (engine Prompt.Changes).</summary>
    public int Changes;
    public NDeckCardSelectScreen? Visual;

    public PendingSelection(List<CardModel> options, int min, int max, bool cancelable, SelectionContext ctx)
    {
        Options = options;
        Max = Math.Max(0, Math.Min(max, options.Count));
        Min = Math.Max(0, Math.Min(min, Max));
        Cancelable = cancelable;
        Ctx = ctx;
    }

    public bool IsWaiting => !Tcs.Task.IsCompleted;
    public void Pick(int index)
    {
        Picked.Add(index);
        if (Visual != null) Client.PickVisualCard(Visual, Options[index]);
    }
    public void Submit()
    {
        if (Visual != null) Client.SubmitVisualCards(Visual);
        Tcs.TrySetResult(Picked.Select(i => Options[i]).ToList());
    }
    public void Cancel() => Tcs.TrySetResult(Array.Empty<CardModel>());
}

/// <summary>
/// Local-only card selector (CardSelectCmd.UseSelector(localOnly: true)), the same seam the v1 engine uses, so the
/// real single-player path (ReserveChoiceId, SignalPlayerChoiceBegun/Ended, queued hook choices) runs around it and
/// only the selection screen is replaced. Same rules as engine HeadlessCardSelector: no auto-selection; an empty
/// option list returns nothing; choose-a-card has min 0 only when canSkip.
/// </summary>
public sealed class OracleSelector : MegaCrit.Sts2.Core.TestSupport.ICardSelector
{
    public PendingSelection? Pending;
    public bool HasPending => Pending != null && Pending.IsWaiting;

    public Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
    {
        var ctx = SelectionCapture.Current;
        // The engine refuses a selection without a captured CardSelectCmd context (an error, so such a run is never
        // recorded); the client keeps going with an "unknown" context and logs it.
        if (ctx == null) { Entry.Log("WARNING: selector reached without a captured CardSelectCmd context"); ctx = new SelectionContext(); }
        SelectionCapture.Current = null;
        var list = options.ToList();
        if (list.Count == 0) return Task.FromResult<IEnumerable<CardModel>>(Array.Empty<CardModel>());
        int min = minSelect, max = maxSelect;
        if (ctx.Kind == "choose_a_card")
        {
            min = ctx.CanSkip == true ? 0 : 1;
            max = 1;
        }
        var p = new PendingSelection(list, min, max, ctx.Cancelable, ctx);
        Nav.TagOrigin(p);
        Pending = p;
        if (Watch.Enabled && p.RoomService == "shop:REMOVE" && ctx.Prefs is { } prefs)
        {
            p.Visual = NDeckCardSelectScreen.Create(list, prefs with { Cancelable = false });
            NOverlayStack.Instance!.Push(p.Visual);
        }
        Entry.Log($"selector: pending {ctx.Kind} {list.Count} options, select {p.Min}-{p.Max}{(p.Cancelable ? " cancelable" : "")}"
                  + (p.FromReward != null ? " (from rewards screen)" : "") + (p.RoomService != null ? $" ({p.RoomService}, no cancel)" : ""));
        return p.Tcs.Task;
    }

    public void Abort()
    {
        var p = Pending;
        Pending = null;
        p?.Tcs.TrySetResult(Array.Empty<CardModel>());
    }

    public MegaCrit.Sts2.Core.TestSupport.CardRewardSelection GetSelectedCardReward(
        IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
        => throw new InvalidOperationException("GetSelectedCardReward reached: the client shows the real card reward screen");
}
