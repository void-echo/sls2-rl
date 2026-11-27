using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

/// <summary>
/// Error context for <c>type:error</c> reports. A ring buffer of what happened recently, fed only by the game's own
/// events (action queue, executor, per-action lifecycle, combat and room events) plus the harness's prompts and the
/// player's inputs; nothing here changes game behaviour. <see cref="Context"/> renders the compact block that
/// <see cref="RunSimulator"/> puts in front of a stack trace.
/// </summary>
internal static class Diag
{
    private const int Cap = 64;
    private static readonly string[] _ring = new string[Cap];
    private static long _n;
    private static ActionQueueSet? _set;
    private static ActionExecutor? _exec;

    /// <summary>Who asked for the choice behind each hook action (captured when it is enqueued).</summary>
    private static readonly ConditionalWeakTable<GameAction, string> _what = new();

    /// <summary>STS2_DIAG_STACKS=1: also print the full stack of every action cancel to stderr (investigations).</summary>
    private static readonly bool _stacks = Environment.GetEnvironmentVariable("STS2_DIAG_STACKS") == "1";

    public static void Reset()
    {
        Array.Clear(_ring);
        _n = 0;
        if (_set != null) _set.ActionEnqueued -= OnEnqueued;
        if (_exec != null)
        {
            _exec.BeforeActionExecuted -= OnBeforeExecuted;
            _exec.AfterActionExecuted -= OnAfterExecuted;
        }
        _set = null;
        _exec = null;
    }

    /// <summary>Subscribe to the run's queue and executor (they are created per run by RunManager).</summary>
    public static void Attach()
    {
        Reset();
        _set = RunManager.Instance.ActionQueueSet;
        _exec = RunManager.Instance.ActionExecutor;
        _set.ActionEnqueued += OnEnqueued;
        _exec.BeforeActionExecuted += OnBeforeExecuted;
        _exec.AfterActionExecuted += OnAfterExecuted;
    }

    /// <summary>Process-wide game events (subscribed once).</summary>
    public static void AttachGlobal()
    {
        var cm = CombatManager.Instance;
        cm.CombatSetUp += s => Note("combat set up: " + string.Join(",", s.Enemies.Select(e => e.Monster?.Id.Entry ?? "?")));
        cm.TurnStarted += s => Note($"turn started: round {s.RoundNumber} side {s.CurrentSide}");
        cm.CombatWon += _ => Note("combat won");
        cm.CombatEnded += _ => Note("combat ended");
        cm.PlayerEndedTurn += (_, _) => Note("player ended turn");
        cm.PlayerUnendedTurn += _ => Note("player un-ended turn");
        RunManager.Instance.RoomEntered += () =>
        {
            var rs = RunManager.Instance.DebugOnlyGetState();
            Note($"room entered: {rs?.CurrentRoom?.GetType().Name} act {rs?.CurrentActIndex + 1} floor {rs?.TotalFloor}");
        };
    }

    public static void Note(string what)
    {
        string phase;
        try
        {
            var sync = RunManager.Instance.ActionQueueSynchronizer?.CombatState.ToString() ?? "-";
            phase = CombatManager.Instance.IsInProgress
                ? $"r{CombatManager.Instance.DebugOnlyGetState()?.RoundNumber} {sync}"
                : sync;
        }
        catch { phase = "?"; }
        _ring[_n % Cap] = $"#{_n} [{phase}] {what}";
        _n++;
    }

    private static string Model(AbstractModel? m) => m == null ? "-" : m.Id.Entry;

    /// <summary>Short description of an action: type, queue id, owner-side source model.</summary>
    public static string Describe(GameAction a)
    {
        var sb = new StringBuilder(a.GetType().Name.Replace("GameAction", "").Replace("Action", ""));
        sb.Append(" id=").Append(a.Id?.ToString() ?? "-");
        switch (a)
        {
            case PlayCardAction p: sb.Append(' ').Append(p.CardModelId.Entry); break;
            case GenericHookGameAction h:
                sb.Append(" hook=").Append(h.HookId).Append(' ').Append(h.ActionType);
                if (_what.TryGetValue(a, out var w)) sb.Append(' ').Append(w);
                break;
            default:
                if (_what.TryGetValue(a, out var w2)) sb.Append(' ').Append(w2);
                break;
        }
        return sb.ToString();
    }

    private static void OnEnqueued(GameAction a)
    {
        try
        {
            switch (a)
            {
                case GenericHookGameAction h when h.ChoiceContext is { } ctx:
                    // The context's model stack names the hook that asked for the choice (e.g. a power's
                    // AfterPlayerTurnStart pushed onto the turn-start context).
                    var stack = ctx.ModelStack?.Select(Model).ToList() ?? new List<string>();
                    _what.AddOrUpdate(a, $"src={Model(ctx.Source)} models=[{string.Join(">", stack.AsEnumerable().Reverse())}]");
                    break;
                case UsePotionAction u:
                    _what.AddOrUpdate(a, u.Player.GetPotionAtSlotIndex((int)u.PotionIndex)?.Id.Entry ?? "?");
                    break;
            }
            a.BeforeCancelled += OnCancelled;
            a.BeforePausedForPlayerChoice += x => Note("paused for choice: " + Describe(x));
            a.BeforeReadyToResumeAfterPlayerChoice += x => Note("ready to resume: " + Describe(x));
            // JustBeforeFinished runs after State = Finished and before the completion source and AfterFinished
            // (whose first subscriber is ActionQueueSet.PopAction): what happens in between shows up as later entries.
            a.JustBeforeFinished += x => Note("finishing: " + Describe(x));
            a.AfterFinished += x => Note("popped: " + Describe(x));
        }
        catch { }
        Note($"enqueued: {Describe(a)} state={a.State}");
    }

    private static void OnCancelled(GameAction a)
    {
        // Cancels are rare; name the game method that cancelled (CombatEnded, StartCancelling..., Enqueue...).
        string by = "?";
        try
        {
            var st = new StackTrace(2, false);
            for (int i = 0; i < st.FrameCount; i++)
            {
                var m = st.GetFrame(i)?.GetMethod();
                if (m?.DeclaringType is { } t && (t == typeof(ActionQueueSet) || t == typeof(ActionQueueSynchronizer)))
                {
                    by = $"{t.Name}.{m.Name}";
                    break;
                }
            }
        }
        catch { }
        if (_stacks) Console.Error.WriteLine($"[DIAG] cancel of {Describe(a)}:\n{Environment.StackTrace}");
        // Cancel sets State = Canceled before raising BeforeCancelled; the last state we saw is the useful one.
        Note($"cancelled: {Describe(a)} by {by}");
    }

    private static void OnBeforeExecuted(GameAction a) => Note($"execute: {Describe(a)} from {a.State}");
    private static void OnAfterExecuted(GameAction a) => Note($"executed: {Describe(a)}");

    /// <summary>The last <paramref name="k"/> entries, oldest first.</summary>
    public static IEnumerable<string> Last(int k)
    {
        long from = Math.Max(0, _n - Math.Min(k, Cap));
        for (long i = from; i < _n; i++) yield return _ring[i % Cap];
    }
}
