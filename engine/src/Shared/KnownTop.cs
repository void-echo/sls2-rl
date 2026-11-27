using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Enchantments;

namespace Sts2Obs;

/// <summary>
/// The cards at the top of the draw pile whose order the player knows (docs/02_观测与编码方法论.md: memory). The
/// pile is exported sorted, so this is the only order information in the observation. Known: cards put on top in
/// plain sight (Headbutt, Thinking Ahead, Shining Strike, Rebound, Nostalgia, ...), and a single Perfect Fit card on
/// top after a shuffle into an empty pile or the opening shuffle. A card shuffled into the draw pile at a random
/// position makes the order unknown again (the player does not see where it went). Insertions are recorded from the
/// game's own calls (Harmony prefixes, <see cref="PatchTargets"/>: no effect on the game) and the list is checked
/// against the real pile at every observation, so it can only under-report.
///
/// Shared by the headless engine (Patches.PatchObservationRecords, reset in Harness.CleanUpRun) and the parity mod
/// (Entry.ApplyPatches, reset when a run starts). <see cref="Current"/> records the reading (a card no longer on
/// top is forgotten, the Perfect Fit check is consumed), so it must run once per decision; <see cref="Peek"/> reads
/// without recording, for observations of a screen that is still settling.
/// </summary>
public static class KnownTop
{
    private static readonly List<CardModel> _known = new();
    private static bool _shuffled;

    public static void Reset()
    {
        _known.Clear();
        _shuffled = false;
    }

    /// <summary>CardPileCmd.Add (the overload every Add goes through), before the cards move.</summary>
    public static void OnAdd(IEnumerable<CardModel> cards, CardPile pile, CardPilePosition position)
    {
        if (pile.Type != PileType.Draw) return;
        if (position == CardPilePosition.Top)
            foreach (var c in cards) _known.Insert(0, c);  // each one lands on top of the previous one
        else if (position == CardPilePosition.Random)
            _known.Clear();
    }

    /// <summary>CardPileCmd.Shuffle: the discard pile goes under the draw pile; into an empty pile it is the new pile.</summary>
    public static void OnShuffle(CardPile drawPile)
    {
        if (drawPile.Cards.Count > 0) return;  // shuffled cards go to the bottom: the known top stays known
        _known.Clear();
        _shuffled = true;
    }

    /// <summary>CardPile.RandomizeOrderInternal: the opening shuffle of a combat.</summary>
    public static void OnOpeningShuffle()
    {
        _known.Clear();
        _shuffled = true;
    }

    /// <summary>The known top cards, top first, after checking them against the pile; records the reading.</summary>
    public static List<CardModel> Current(CardPile drawPile) => Read(drawPile, record: true);

    /// <summary>What <see cref="Current"/> would return now, without recording it.</summary>
    public static List<CardModel> Peek(CardPile drawPile) => Read(drawPile, record: false);

    private static List<CardModel> Read(CardPile drawPile, bool record)
    {
        var cards = drawPile.Cards;
        var known = _known;
        if (_shuffled && _known.Count == 0)
        {
            int n = 0;  // Perfect Fit moves its card to the top after a shuffle; the order of several is random
            while (n < cards.Count && cards[n].Enchantment is PerfectFit) n++;
            if (n == 1) known = new List<CardModel> { cards[0] };
        }
        var kept = new List<CardModel>();
        foreach (var c in known)
        {
            if (!cards.Contains(c)) continue;  // drawn, played or taken out of the pile
            if (kept.Count < cards.Count && ReferenceEquals(cards[kept.Count], c)) kept.Add(c);
            else break;  // something the player did not see went above it
        }
        if (record)
        {
            _shuffled = false;
            _known.Clear();
            _known.AddRange(kept);
        }
        return kept;
    }

    // ───────────────────────────── the recorded calls (Harmony prefixes; they only record, never change the game)

    /// <summary>Each game method whose calls are recorded, with its prefix.</summary>
    public static IEnumerable<(MethodBase Target, MethodInfo Prefix)> PatchTargets()
    {
        yield return (Method(typeof(CardPileCmd), "Add", typeof(IEnumerable<CardModel>), typeof(CardPile), typeof(CardPilePosition),
                             typeof(AbstractModel), typeof(bool), typeof(bool)), Prefix(nameof(PileAddPrefix)));
        yield return (Method(typeof(CardPileCmd), "Shuffle", typeof(PlayerChoiceContext), typeof(Player)), Prefix(nameof(ShufflePrefix)));
        yield return (Method(typeof(CardPile), "RandomizeOrderInternal"), Prefix(nameof(OpeningShufflePrefix)));
    }

    private static MethodInfo Method(Type t, string name, params Type[] args) =>
        (args.Length == 0 ? AccessTools.Method(t, name) : AccessTools.Method(t, name, args))
        ?? throw new MissingMethodException(t.FullName, name);

    private static MethodInfo Prefix(string name) =>
        typeof(KnownTop).GetMethod(name, BindingFlags.Static | BindingFlags.Public) ?? throw new MissingMethodException(nameof(KnownTop), name);

    public static void PileAddPrefix(IEnumerable<CardModel> cards, CardPile newPile, CardPilePosition position) =>
        OnAdd(cards, newPile, position);

    public static void ShufflePrefix(Player player)
    {
        var draw = player.PlayerCombatState?.DrawPile;
        if (draw != null) OnShuffle(draw);
    }

    public static void OpeningShufflePrefix() => OnOpeningShuffle();
}
