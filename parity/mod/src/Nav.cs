using System.Reflection;
using System.Runtime.CompilerServices;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Rewards;

namespace ParityOracle;

/// <summary>
/// The engine's "no navigation cycles" bookkeeping (engine/RL_PROTOCOL.md, "No navigation cycles"; engine Prompts.cs
/// RewardsPrompt.Declined/Held, Prompt.Changes, CardSelectPrompt.FromReward/RoomService, RunSimulator.TagOrigin /
/// BackOut / TakeCardRewardOrigin), kept per client screen so the client offers exactly the engine's candidates:
///   - a rewards screen (NRewardsScreen) remembers the rewards declined there (card reward Skip / removal cancel with
///     nothing changed on the sub-screen) and the ones held back (the same back-out after a reroll or a potion) until
///     the game changes again on the rewards screen (a reward taken, a potion used or discarded there);
///   - a card reward screen (NCardRewardSelectionScreen) and a pending selection count the inputs that changed the
///     game while they were on top (everything except a pick), and know the rewards screen they were opened from;
///   - Smith, Cook and the merchant's removal selections are tagged with their room service (no cancel).
/// The origin of a screen is set only for the synchronous part of the click that opens it (the reward button's
/// ForceClick, the rest button's ForceClick, the removal entry's OnTryPurchaseWrapper), exactly like the engine's
/// _openingReward / _openingService window: the game reaches ShowScreen / the selector before its first await.
/// </summary>
public static class Nav
{
    public sealed class RewardsScreenState
    {
        public readonly RewardsSet Set;
        public readonly HashSet<Reward> Declined = new(ReferenceEqualityComparer.Instance);
        public readonly HashSet<Reward> Held = new(ReferenceEqualityComparer.Instance);
        /// <summary>The reward selected last from this screen (engine RewardsLoop: Held is cleared once it was taken).</summary>
        public Reward? Opened;
        public RewardsScreenState(RewardsSet set) { Set = set; }

        /// <summary>Engine RewardsLoop after `await select`: a reward taken releases every held-back reward.</summary>
        public void Settle()
        {
            if (Opened != null && Opened.SuccessfullySelected)
            {
                Held.Clear();
                Opened = null;
            }
        }
    }

    public sealed record RewardOrigin(RewardsScreenState Screen, Reward Reward);

    public sealed class CardRewardScreenState
    {
        public RewardOrigin? FromReward;
        public int Changes;
    }

    private static readonly ConditionalWeakTable<NRewardsScreen, RewardsScreenState> _rewards = new();
    private static readonly ConditionalWeakTable<NCardRewardSelectionScreen, CardRewardScreenState> _cardRewards = new();

    /// <summary>Set only while the click that may open a reward's screen runs synchronously (engine _openingReward).</summary>
    public static RewardOrigin? OpeningReward;
    /// <summary>Set only while a Smith / Cook / merchant removal input runs synchronously (engine _openingService).</summary>
    public static string? OpeningService;

    public static void Reset()
    {
        OpeningReward = null;
        OpeningService = null;
        _rewards.Clear();
        _cardRewards.Clear();
    }

    public static RewardsScreenState Rewards(NRewardsScreen screen, RewardsSet set)
    {
        if (_rewards.TryGetValue(screen, out var st) && ReferenceEquals(st.Set, set)) return st;
        st = new RewardsScreenState(set);
        _rewards.AddOrUpdate(screen, st);
        return st;
    }

    public static CardRewardScreenState CardRewardState(NCardRewardSelectionScreen screen) => _cardRewards.GetValue(screen, _ => new CardRewardScreenState());

    private static readonly FieldInfo _shownScreen = typeof(CardReward).GetField("_currentlyShownScreen", BindingFlags.Instance | BindingFlags.NonPublic)!;

    /// <summary>
    /// The reward button's click (NRewardButton.OnRelease → GetReward → RewardsSetSynchronizer.SelectLocalReward), with
    /// the opening window around its synchronous part. A card reward shows its screen there (CardReward.OnSelect →
    /// NCardRewardSelectionScreen.ShowScreen(_cards, ...), engine TakeCardRewardOrigin); a removal reward reaches the
    /// selector there (RewardSynchronizer.DoUnsyncedCardRemoval, engine TagOrigin).
    /// </summary>
    public static void OpenReward(RewardsScreenState st, Reward r, Action click)
    {
        st.Opened = r;
        var outer = OpeningReward;
        var origin = new RewardOrigin(st, r);
        OpeningReward = origin;
        try { click(); }
        finally { OpeningReward = outer; }
        if (r is CardReward cr && _shownScreen.GetValue(cr) is NCardRewardSelectionScreen screen)
            CardRewardState(screen).FromReward = origin;
    }

    /// <summary>A Smith / Cook rest option or the merchant removal, with the opening window around its synchronous part.</summary>
    public static void OpenService(string? service, Action click)
    {
        OpeningService = service;
        try { click(); }
        finally { OpeningService = null; }
    }

    /// <summary>Engine RunSimulator.TagOrigin: only cancelable selections whose cancel would return to the screen unchanged.</summary>
    public static void TagOrigin(PendingSelection p)
    {
        if (!p.Cancelable) return;
        if (OpeningReward is { Reward: CardRemovalReward } o && p.Ctx.Kind == "deck_remove")
        {
            p.FromReward = o;
            OpeningReward = null;
        }
        else if (OpeningService is { } svc && p.Ctx.Kind == (svc == "rest:SMITH" ? "deck_upgrade" : "deck_remove"))
        {
            p.RoomService = svc;
            OpeningService = null;
        }
    }

    /// <summary>
    /// Engine RunSimulator.BackOut, decided when the candidates are built: backing out of a reward's screen before
    /// anything changed on it declines the reward; after a change it is a plain back-out and the reward is held back;
    /// on a must-take-all screen there is no back-out (null).
    /// </summary>
    public static Func<Task>? BackOut(int changes, RewardOrigin o, Func<Task> back)
    {
        if (o.Screen.Set.DisallowSkipping) return null;
        if (changes == 0) return () => { o.Screen.Declined.Add(o.Reward); return back(); };
        return () => { o.Screen.Held.Clear(); o.Screen.Held.Add(o.Reward); return back(); };
    }

    /// <summary>
    /// Engine RunSimulator.RlAct before executing a candidate: every input except a pick changes the game on the screen
    /// it is given on, and a potion used or discarded on a rewards screen releases its held-back rewards.
    /// </summary>
    public static void BeforeInput(object? screen, string? kind)
    {
        switch (screen)
        {
            case CardRewardScreenState cr when kind != "pick": cr.Changes++; break;
            case PendingSelection ps when kind != "pick": ps.Changes++; break;
            case RewardsScreenState rs:
                // The previous reward's selection is over by the time the next input is given on its screen.
                rs.Settle();
                rs.Opened = null;
                if (kind is "potion" or "discard_potion") rs.Held.Clear();
                break;
        }
    }
}
