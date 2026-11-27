using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using Sts2Obs;

namespace ParityOracle;

/// <summary>
/// The RL observation in the client. The observation and the data parts of the candidate descriptors are the
/// engine's own code, <c>engine/src/Shared/ObsShared.cs</c> and <c>KnownTop.cs</c>, compiled into this mod
/// (ParityOracle.csproj), so the two sides cannot drift apart; this class only holds the client's observation state
/// (<see cref="ObsState"/>: cards played this turn, the act's map nodes) and the one client-specific rule about the
/// known draw-pile top (<see cref="CommitKnownTop"/>). ctx is built by Client (it depends on the decision).
/// </summary>
public static class Obs
{
    public static RunState? Rs => RunManager.Instance?.DebugOnlyGetState();

    /// <summary>The client's observation state. The known draw-pile top is only peeked while building observations:
    /// the client reads a screen many times while it settles, the engine reads each decision once.</summary>
    public static readonly ObsState State = new() { PeekKnownTop = true };

    /// <summary>Cards played this turn as of the last combat observation (tags cards with <c>played_turn</c>).</summary>
    public static ISet<object>? Played => State.PlayedThisTurn;

    /// <summary>A new run: the engine resets the same things in Harness.CleanUpRun / ResetRlState.</summary>
    public static void ResetRun()
    {
        State.ResetMap();
        State.PlayedThisTurn = null;
        KnownTop.Reset();
    }

    public static Dictionary<string, object?> Build(Player player, Func<Dictionary<string, object?>> ctx) =>
        ObsShared.Observation(Rs!, player, State, ctx);

    public static Dictionary<string, object?> Card(CardModel c, bool inHand = false) => ObsShared.Card(c, State.PlayedThisTurn, inHand);

    public static List<MapPoint> MapNodes() => ObsShared.MapNodes(Rs, State);

    public static int NodeIndex(MapPoint p) => ObsShared.NodeIndex(Rs, State, p);

    /// <summary>
    /// Records the known draw-pile top for the decision being reported, exactly where the engine's observation reads
    /// it (KnownTop.Current inside the combat part, once per decision). Called once the decision is stable, so the
    /// value recorded is the one the reported observation peeked.
    /// </summary>
    public static void CommitKnownTop(Player player)
    {
        if (!CombatManager.Instance.IsInProgress) return;
        var pcs = player.PlayerCombatState;
        if (pcs == null || CombatManager.Instance.DebugOnlyGetState() == null || pcs.DrawPile == null) return;
        KnownTop.Current(pcs.DrawPile);
    }
}
