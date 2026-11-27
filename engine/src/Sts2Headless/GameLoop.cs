using System.Collections.Concurrent;

namespace Sts2Headless;

/// <summary>
/// The single game thread's message queue. It plays the role of Godot's main-thread
/// SynchronizationContext: Post only enqueues, nothing runs inline on the posting thread, and the
/// owner drains the queue with <see cref="RunUntilIdle"/>. `await Task.Yield()` and continuations
/// that were deliberately deferred therefore run after the current synchronous work, the same
/// "next frame" ordering the real client has. All game code runs on the thread that created the
/// loop; posts from any other thread (there should be none) are queued and counted as foreign.
///
/// A second, lower-priority queue (<see cref="PostWhenIdle"/>) holds work the client only does after
/// an animation or a timed wait (rewards after the death animation, the chest opening): such a
/// callback runs only once everything else has settled, one at a time.
///
/// Every callback (and every harness command, see <see cref="Run"/>) runs inside a task on
/// <see cref="Scheduler"/>, so TaskScheduler.Current is the game loop too: Task.ContinueWith without an explicit
/// scheduler (ActionQueueSynchronizer.GetHookActionForId) continues on the game thread instead of the thread pool,
/// where in the client it races the main thread.
/// </summary>
internal sealed class GameLoop : SynchronizationContext
{
    private readonly ConcurrentQueue<(SendOrPostCallback cb, object? state)> _queue = new();
    private readonly Queue<Action> _idle = new();
    private readonly AutoResetEvent _signal = new(false);
    private readonly int _threadId = Environment.CurrentManagedThreadId;

    public static GameLoop Instance { get; } = new();

    /// <summary>Posts that arrived from a thread other than the game thread (diagnostic; should stay 0).</summary>
    public int ForeignPosts;

    /// <summary>Raised when a queued callback throws (async void bodies, broken continuations).</summary>
    public event Action<Exception>? CallbackFaulted;

    public bool IsLoopThread => Environment.CurrentManagedThreadId == _threadId;

    public static readonly TaskScheduler Scheduler = new LoopScheduler();

    private sealed class LoopScheduler : TaskScheduler
    {
        protected override void QueueTask(Task task) => Instance.Post(_ => TryExecuteTask(task), null);
        protected override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued) => Instance.IsLoopThread && TryExecuteTask(task);
        protected override IEnumerable<Task>? GetScheduledTasks() => null;
        public override int MaximumConcurrencyLevel => 1;
    }

    /// <summary>Run <paramref name="a"/> on the game thread now, with <see cref="Scheduler"/> as TaskScheduler.Current.</summary>
    public void Run(Action a)
    {
        if (TaskScheduler.Current == Scheduler) { a(); return; }
        var t = new Task(a);
        t.RunSynchronously(Scheduler);
        if (t.IsFaulted) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(t.Exception!.InnerException!).Throw();
    }

    /// <summary>Stack of the first foreign post since the last reset (diagnostics for the integrity error).</summary>
    public string? FirstForeignPost;

    public void ResetForeignPosts()
    {
        ForeignPosts = 0;
        FirstForeignPost = null;
    }

    public override void Post(SendOrPostCallback d, object? state)
    {
        if (!IsLoopThread)
        {
            string cont = state is Delegate del ? $"{del.Method.DeclaringType?.FullName}.{del.Method.Name} target {del.Target?.GetType().FullName}" : state?.GetType().FullName ?? "-";
            var where = $"thread {Environment.CurrentManagedThreadId} continuation {cont} via {d.Method.DeclaringType?.Name}.{d.Method.Name}: {Environment.StackTrace}";
            Interlocked.CompareExchange(ref FirstForeignPost, where, null);
            Interlocked.Increment(ref ForeignPosts);
        }
        _queue.Enqueue((d, state));
        _signal.Set();
    }

    public override void Send(SendOrPostCallback d, object? state)
    {
        if (IsLoopThread) d(state);
        else
        {
            using var done = new ManualResetEventSlim(false);
            Exception? err = null;
            Post(_ => { try { d(state); } catch (Exception e) { err = e; } finally { done.Set(); } }, null);
            done.Wait();
            if (err != null) throw err;
        }
    }

    public override SynchronizationContext CreateCopy() => this;

    /// <summary>Run <paramref name="a"/> once the regular queue is empty (game thread only).</summary>
    public void PostWhenIdle(Action a)
    {
        if (!IsLoopThread) throw new InvalidOperationException("PostWhenIdle off the game thread");
        _idle.Enqueue(a);
    }

    public bool IsEmpty => _queue.IsEmpty && _idle.Count == 0;

    /// <summary>Run queued callbacks (then idle callbacks) until both queues are empty. Returns how many ran.</summary>
    public int RunUntilIdle()
    {
        int n = 0;
        while (true)
        {
            if (_queue.TryDequeue(out var w))
            {
                n++;
                try { Run(() => w.cb(w.state)); }
                catch (Exception e) { CallbackFaulted?.Invoke(e); }
                continue;
            }
            if (_idle.Count == 0) break;
            var a = _idle.Dequeue();
            n++;
            try { Run(a); }
            catch (Exception e) { CallbackFaulted?.Invoke(e); }
        }
        return n;
    }

    /// <summary>Block until a post arrives (true) or the timeout passes (false).</summary>
    public bool WaitForPost(int timeoutMs) => !_queue.IsEmpty || _signal.WaitOne(timeoutMs);

    /// <summary>Drop everything that is queued (used between runs; the old run's tasks are abandoned).</summary>
    public void Clear()
    {
        while (_queue.TryDequeue(out _)) { }
        _idle.Clear();
    }
}
