using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace Godot
{
    /// <summary>
    /// Root of Godot's object model. Headless no native object exists: construction does nothing, and every other
    /// member (signals, properties, notifications) is an inert generated declaration (Generated/Generated.cs).
    /// </summary>
    public partial class GodotObject : IDisposable
    {
        public GodotObject() { }

        public static bool IsInstanceValid([NotNullWhen(true)] GodotObject? instance) => instance != null;

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing) { }

        public SignalAwaiter ToSignal(GodotObject source, StringName signal) => new SignalAwaiter(source, signal, this);
    }

    public interface IAwaiter : INotifyCompletion
    {
        bool IsCompleted { get; }
        void GetResult();
    }

    public interface IAwaiter<out TResult> : INotifyCompletion
    {
        bool IsCompleted { get; }
        TResult GetResult();
    }

    public interface IAwaitable
    {
        IAwaiter GetAwaiter();
    }

    public interface IAwaitable<out TResult>
    {
        IAwaiter<TResult> GetAwaiter();
    }

    /// <summary>
    /// Awaiting a Godot signal. Headless no signal is ever emitted (frames, timers, animations do not exist), so the
    /// awaiter is already completed: code waiting for "the next frame" or an animation continues immediately, which
    /// is what the game's own NonInteractiveMode does for its timed waits (Cmd.Wait).
    /// </summary>
    public partial class SignalAwaiter : IAwaiter<Variant[]>, INotifyCompletion, IAwaitable<Variant[]>
    {
        public SignalAwaiter() { }
        public SignalAwaiter(GodotObject source, StringName signal, GodotObject target) { }
        public bool IsCompleted => true;
        public void OnCompleted(Action continuation) => continuation();
        public Variant[] GetResult() => Array.Empty<Variant>();
        public IAwaiter<Variant[]> GetAwaiter() => this;
    }

    public readonly partial struct Signal : IAwaitable<Variant[]>
    {
        private readonly GodotObject _owner;
        private readonly StringName _signalName;
        public GodotObject Owner => _owner;
        public StringName Name => _signalName;
        public Signal(GodotObject owner, StringName name) { _owner = owner; _signalName = name; }
        public IAwaiter<Variant[]> GetAwaiter() => new SignalAwaiter(_owner, _signalName, _owner);
    }

    /// <summary>GD: logging goes to stderr; loading and the engine RNG do not exist headless (only UI code uses them).</summary>
    public static partial class GD
    {
        public static T Load<T>(string path) where T : class => default!;
        public static void Print(string what) => Console.Error.WriteLine(what);
        public static void Print(params object[] what) => Console.Error.WriteLine(string.Concat(what));
        public static void PrintRich(string what) { }
        public static void PrintRich(params object[] what) { }
        public static void PrintErr(string what) => Console.Error.WriteLine("[GD.PrintErr] " + what);
        public static void PrintErr(params object[] what) => Console.Error.WriteLine("[GD.PrintErr] " + string.Concat(what));
        public static void PushError(string message) => Console.Error.WriteLine("[GD.PushError] " + message);
        public static void PushError(params object[] what) => Console.Error.WriteLine("[GD.PushError] " + string.Concat(what));
        public static void PushWarning(string message) { }
        public static void PushWarning(params object[] what) { }
        // Engine RNG: only debug VFX (NCombatVfxSpawner) call these; a constant keeps runs reproducible.
        public static float Randf() => 0.5f;
        public static double RandRange(double from, double to) => (from + to) / 2;
        public static int RandRange(int from, int to) => from;
    }
}
