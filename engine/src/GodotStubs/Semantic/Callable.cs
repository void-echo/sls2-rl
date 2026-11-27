using System;
using System.Linq;
using System.Threading;

namespace Godot
{
    /// <summary>
    /// Godot.Callable headless. A delegate callable invokes its delegate (arguments converted from Variant like
    /// Godot's trampolines do); CallDeferred posts the call to the current SynchronizationContext (the harness game
    /// loop), which is where Godot runs deferred calls: after the current frame's work. Game logic depends on this:
    /// GameAction.Cancel completes the action through Callable.From(...).CallDeferred().
    /// A (target, method) callable has no script runtime to dispatch to and does nothing.
    /// </summary>
    public readonly partial struct Callable
    {
        private readonly GodotObject? _target;
        private readonly StringName? _method;
        private readonly Delegate? _delegate;

        public GodotObject Target => _target!;
        public StringName Method => _method!;
        public Delegate Delegate => _delegate!;

        private Callable(Delegate d) { _target = null; _method = null; _delegate = d; }

        public Callable(GodotObject target, StringName method)
        {
            _target = target;
            _method = method;
            _delegate = null;
        }

        public static Callable From(Action action) => new(action);
        public static Callable From<T0>(Action<T0> action) => new(action);
        public static Callable From<T0, T1>(Action<T0, T1> action) => new(action);
        public static Callable From<T0, T1, T2>(Action<T0, T1, T2> action) => new(action);
        public static Callable From<T0, T1, T2, T3>(Action<T0, T1, T2, T3> action) => new(action);
        public static Callable From<T0, T1, T2, T3, T4>(Action<T0, T1, T2, T3, T4> action) => new(action);
        public static Callable From<T0, T1, T2, T3, T4, T5>(Action<T0, T1, T2, T3, T4, T5> action) => new(action);
        public static Callable From<T0, T1, T2, T3, T4, T5, T6>(Action<T0, T1, T2, T3, T4, T5, T6> action) => new(action);
        public static Callable From<T0, T1, T2, T3, T4, T5, T6, T7>(Action<T0, T1, T2, T3, T4, T5, T6, T7> action) => new(action);
        public static Callable From<T0, T1, T2, T3, T4, T5, T6, T7, T8>(Action<T0, T1, T2, T3, T4, T5, T6, T7, T8> action) => new(action);
        public static Callable From<TResult>(Func<TResult> func) => new(func);
        public static Callable From<T0, TResult>(Func<T0, TResult> func) => new(func);
        public static Callable From<T0, T1, TResult>(Func<T0, T1, TResult> func) => new(func);
        public static Callable From<T0, T1, T2, TResult>(Func<T0, T1, T2, TResult> func) => new(func);
        public static Callable From<T0, T1, T2, T3, TResult>(Func<T0, T1, T2, T3, TResult> func) => new(func);
        public static Callable From<T0, T1, T2, T3, T4, TResult>(Func<T0, T1, T2, T3, T4, TResult> func) => new(func);
        public static Callable From<T0, T1, T2, T3, T4, T5, TResult>(Func<T0, T1, T2, T3, T4, T5, TResult> func) => new(func);
        public static Callable From<T0, T1, T2, T3, T4, T5, T6, TResult>(Func<T0, T1, T2, T3, T4, T5, T6, TResult> func) => new(func);
        public static Callable From<T0, T1, T2, T3, T4, T5, T6, T7, TResult>(Func<T0, T1, T2, T3, T4, T5, T6, T7, TResult> func) => new(func);
        public static Callable From<T0, T1, T2, T3, T4, T5, T6, T7, T8, TResult>(Func<T0, T1, T2, T3, T4, T5, T6, T7, T8, TResult> func) => new(func);

        public Variant Call(params Variant[] args)
        {
            if (_delegate == null) return default;
            var ps = _delegate.Method.GetParameters();
            var conv = new object?[ps.Length];
            for (int i = 0; i < ps.Length; i++)
                conv[i] = i < args.Length ? args[i].AsType(ps[i].ParameterType) : null;
            object? r;
            try { r = _delegate.DynamicInvoke(conv); }
            catch (System.Reflection.TargetInvocationException e) when (e.InnerException != null)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerException).Throw();
                throw;
            }
            return Variant.FromObject(r);
        }

        public void CallDeferred(params Variant[] args)
        {
            if (_delegate == null) return;
            var self = this;
            var ctx = SynchronizationContext.Current;
            if (ctx != null) ctx.Post(_ => self.Call(args), null);
            else self.Call(args);
        }
    }
}
