using System.Threading;
using System.Runtime.CompilerServices;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.TestSupport;

namespace Sts2Headless;

/// <summary>
/// A point where a game task is suspended waiting for the player, i.e. where the real client would
/// show an overlay screen. Every prompt owns the TaskCompletionSource the game is awaiting; the
/// harness answers it with exactly the value the UI screen would produce. The newest waiting prompt
/// is the one on top (the client's overlay stack is LIFO too).
/// </summary>
internal abstract class Prompt
{
    public long Seq;
    /// <summary>
    /// Inputs given while this prompt was on top that changed game state (everything except a pick that only extends a
    /// card selection). While it is 0, backing out of the screen returns to exactly the decision it was opened from.
    /// </summary>
    public int Changes;
    public abstract string Decision { get; }
    public abstract bool IsWaiting { get; }
    /// <summary>Run teardown: cancel the wait so the game task awaiting it unwinds (the client closes its screens).</summary>
    public abstract void Abort();
}

/// <summary>What the selection entry point (CardSelectCmd.From*) was asked for. Captured by a Harmony prefix.</summary>
internal sealed class SelectionContext
{
    public string Kind = "unknown";
    public int? Min, Max;
    public bool Cancelable;
    public bool RequireConfirm;
    public bool? CanSkip;        // FromChooseACardScreen only
    public string? PromptKey;
    public string? Source;
    public string? Pile;
}

/// <summary>Brackets the synchronous part of each CardSelectCmd.From* call; see Patches.SelectionPrefix.</summary>
internal static class SelectionCapture
{
    // Nesting of the synchronous part of CardSelectCmd entries (FromDeckForRemoval forwards to FromDeckGeneric).
    [ThreadStatic] public static int Depth;

    // Async-flow local: an entry may await (hooks, turn-start effects) before it reaches the selector. The
    // finalizer clears the value in the caller's flow, while the entry's continuation keeps the copy it captured.
    private static readonly AsyncLocal<SelectionContext?> _current = new();

    public static SelectionContext? Current
    {
        get => _current.Value;
        set => _current.Value = value;
    }
}

/// <summary>Hand / pile / deck / grid / choose-a-card selection (any CardSelectCmd entry that reaches the selector).</summary>
internal sealed class CardSelectPrompt : Prompt
{
    public readonly List<CardModel> Options;
    public readonly int Min, Max;
    public readonly bool Cancelable;
    public readonly SelectionContext Ctx;
    public readonly List<int> Picked = new();
    public readonly TaskCompletionSource<IEnumerable<CardModel>> Tcs = new();
    /// <summary>The removal reward this selection belongs to, when it was opened from a rewards screen.</summary>
    public RewardOrigin? FromReward;
    /// <summary>The room service (Smith, Cook, merchant removal) this selection belongs to; see RunSimulator.TagOrigin.</summary>
    public string? RoomService;

    public CardSelectPrompt(List<CardModel> options, int min, int max, bool cancelable, SelectionContext ctx)
    {
        Options = options;
        Max = Math.Max(0, Math.Min(max, options.Count));
        // With manual confirmation the entry does not auto-resolve when there are fewer options than
        // MinSelect; the screen then accepts all of them.
        Min = Math.Max(0, Math.Min(min, Max));
        Cancelable = cancelable;
        Ctx = ctx;
    }

    public override string Decision => "card_select";
    public override bool IsWaiting => !Tcs.Task.IsCompleted;
    public void Submit() => Tcs.TrySetResult(Picked.Select(i => Options[i]).ToList());
    public void Cancel() => Tcs.TrySetResult(Array.Empty<CardModel>());
    public override void Abort() => Tcs.TrySetCanceled();
}

/// <summary>NChooseABundleSelectionScreen stand-in (Scroll Boxes): pick one pack.</summary>
internal sealed class BundlePrompt : Prompt
{
    public readonly IReadOnlyList<IReadOnlyList<CardModel>> Bundles;
    public readonly TaskCompletionSource<IEnumerable<IReadOnlyList<CardModel>>> Tcs = new();
    public BundlePrompt(IReadOnlyList<IReadOnlyList<CardModel>> bundles) { Bundles = bundles; }
    public override string Decision => "bundle_select";
    public override bool IsWaiting => !Tcs.Task.IsCompleted;
    public void Choose(int i) => Tcs.TrySetResult(new[] { Bundles[i] });
    public override void Abort() => Tcs.TrySetCanceled();
}

/// <summary>
/// NCardRewardSelectionScreen stand-in. CardReward.OnSelect awaits OptionSelected() in a loop;
/// the answer is a card index, or Options.Count + alternative index (Skip / REROLL / SACRIFICE),
/// exactly as the screen's SelectCard / OnAlternateRewardSelected produce it.
/// </summary>
internal sealed class CardRewardPrompt : Prompt
{
    public IReadOnlyList<CardCreationResult> Options;
    public IReadOnlyList<CardRewardAlternative> Alts;
    public TaskCompletionSource<int?>? Tcs;
    /// <summary>The rewards screen and reward this screen was opened from (null for a card reward selected elsewhere).</summary>
    public RewardOrigin? FromReward;
    public CardRewardPrompt(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alts)
    {
        Options = options;
        Alts = alts;
    }
    public override string Decision => "card_reward";
    public override bool IsWaiting => Tcs != null && !Tcs.Task.IsCompleted;
    public void Choose(int index) => Tcs?.TrySetResult(index);
    public override void Abort() => Tcs?.TrySetCanceled();

    public static readonly ConditionalWeakTable<object, CardRewardPrompt> ByScreen = new();
}

/// <summary>What the player does on a rewards screen.</summary>
internal readonly record struct RewardChoice(bool Proceed, int Index);

/// <summary>A reward opened from a rewards screen (the screen the player returns to when backing out).</summary>
internal sealed record RewardOrigin(RewardsPrompt Screen, Reward Reward);

/// <summary>NRewardsScreen stand-in, opened from RewardsSet.Offer through RewardsSet.testSelector.</summary>
internal sealed class RewardsPrompt : Prompt
{
    public readonly RewardsSet Set;
    public readonly bool Terminal;
    public TaskCompletionSource<RewardChoice>? Tcs;
    /// <summary>
    /// Rewards the player opened and backed out of without changing anything (card reward Skip, removal cancel): they
    /// are no longer offered and are left behind like any untaken reward (SkipLocalRewardsSet -> Reward.OnSkipped at
    /// proceed). Re-opening one would only return to the identical screen (RL_PROTOCOL.md, no navigation cycles).
    /// </summary>
    public readonly HashSet<Reward> Declined = new(ReferenceEqualityComparer.Instance);
    /// <summary>
    /// Rewards backed out of after something changed on their screen (Driftwood's REROLL, a potion): they stay, but
    /// re-opening one before the game changes again would reproduce the screen just left, so they are held back until
    /// an input on this rewards screen changes the game (a reward taken, a potion used or discarded).
    /// </summary>
    public readonly HashSet<Reward> Held = new(ReferenceEqualityComparer.Instance);
    public RewardsPrompt(RewardsSet set, bool terminal) { Set = set; Terminal = terminal; }
    public override string Decision => "rewards";
    public override bool IsWaiting => Tcs != null && !Tcs.Task.IsCompleted;
    public override void Abort() => Tcs?.TrySetCanceled();
}

/// <summary>NTreasureRoom before the chest is opened: the chest button is the only way forward (potions stay usable).</summary>
internal sealed class ChestPrompt : Prompt
{
    public readonly MegaCrit.Sts2.Core.Rooms.TreasureRoom Room;
    public readonly TaskCompletionSource Tcs = new();
    public ChestPrompt(MegaCrit.Sts2.Core.Rooms.TreasureRoom room) { Room = room; }
    public override string Decision => "treasure";
    public override bool IsWaiting => !Tcs.Task.IsCompleted;
    public override void Abort() => Tcs.TrySetCanceled();
}

/// <summary>NTreasureRoomRelicCollection stand-in: take the offered relic or skip it.</summary>
internal sealed class TreasurePrompt : Prompt
{
    public readonly IReadOnlyList<RelicModel> Relics;
    public readonly TaskCompletionSource<int?> Tcs = new();
    public TreasurePrompt(IReadOnlyList<RelicModel> relics) { Relics = relics; }
    public override string Decision => "treasure";
    public override bool IsWaiting => !Tcs.Task.IsCompleted;
    public override void Abort() => Tcs.TrySetCanceled();
}

/// <summary>NCrystalSphereScreen stand-in. The minigame model is driven directly (SetTool + CellClicked).</summary>
internal sealed class SpherePrompt : Prompt
{
    public readonly CrystalSphereMinigame Game;
    public bool Busy;
    public SpherePrompt(CrystalSphereMinigame game) { Game = game; }
    public override string Decision => "crystal_sphere";
    public override bool IsWaiting => !Busy && !Game.IsFinished;
    public override void Abort() => Game.ForceMinigameEnd();
}

/// <summary>
/// The card selector registered with CardSelectCmd.UseSelector(localOnly: true) (the seam the game provides for
/// tests and AutoSlay). It never decides anything: it turns every request into a CardSelectPrompt and
/// returns the prompt's task. The selection semantics (min/max/cancelable/can-skip) come from the
/// SelectionContext captured at the CardSelectCmd entry, because ICardSelector itself does not carry them.
/// </summary>
internal sealed class HeadlessCardSelector : ICardSelector
{
    public Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
    {
        // Every selection reaches the selector through a patched CardSelectCmd entry; a missing context means an
        // unknown path whose min/max/cancel semantics we would be guessing, so the run is invalidated instead.
        var ctx = SelectionCapture.Current
                  ?? throw new InvalidOperationException("card selection reached the selector without a captured CardSelectCmd context");
        SelectionCapture.Current = null;
        var list = options.ToList();
        if (list.Count == 0)
            return Task.FromResult<IEnumerable<CardModel>>(Array.Empty<CardModel>());
        int min = minSelect, max = maxSelect;
        if (ctx.Kind == "choose_a_card")
        {
            // CardSelectCmd passes (cards, 0, 1) to selectors regardless of canSkip; the screen only
            // offers its Skip button when canSkip is true.
            min = ctx.CanSkip == true ? 0 : 1;
            max = 1;
        }
        var p = new CardSelectPrompt(list, min, max, ctx.Cancelable, ctx);
        RunSimulator.TagOrigin(p);
        RunSimulator.AddPrompt(p);
        return p.Tcs.Task;
    }

    public CardRewardSelection GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
    {
        // CardReward.OnSelect only asks the selector when no screen was shown. The harness always
        // provides the NCardRewardSelectionScreen stand-in, so reaching this is a harness bug.
        throw new InvalidOperationException("GetSelectedCardReward reached: the card reward screen shim is not installed");
    }
}
