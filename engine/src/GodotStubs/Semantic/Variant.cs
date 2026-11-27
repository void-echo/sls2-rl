using System;
using System.Collections.Generic;

namespace Godot
{
    /// <summary>
    /// Godot.Variant headless: a boxed managed value with Godot's conversion surface. Conversions follow Godot's
    /// rules for the cases game code can hit (numeric widening/narrowing between int/float kinds, bool from numbers,
    /// object identity for GodotObject, strings); anything else yields default(T), as Godot does for a nil variant.
    /// </summary>
    public partial struct Variant : IDisposable
    {
        public enum Type : long
        {
            Nil, Bool, Int, Float, String, Vector2, Vector2I, Rect2, Rect2I, Vector3, Vector3I, Transform2D, Vector4, Vector4I,
            Plane, Quaternion, Aabb, Basis, Transform3D, Projection, Color, StringName, NodePath, Rid, Object, Callable,
            Signal, Dictionary, Array, PackedByteArray, PackedInt32Array, PackedInt64Array, PackedFloat32Array,
            PackedFloat64Array, PackedStringArray, PackedVector2Array, PackedVector3Array, PackedColorArray,
            PackedVector4Array, Max,
        }

        private readonly object? _obj;

        private Variant(object? obj) { _obj = obj; }

        internal static Variant FromObject(object? o) => new(o);
        internal object? Obj => _obj;

        public void Dispose() { }

        public Type VariantType => _obj switch
        {
            null => Type.Nil,
            bool => Type.Bool,
            sbyte or byte or short or ushort or int or uint or long or ulong or char => Type.Int,
            float or double => Type.Float,
            string => Type.String,
            Vector2 => Type.Vector2, Vector2I => Type.Vector2I, Rect2 => Type.Rect2, Rect2I => Type.Rect2I,
            Vector4 => Type.Vector4, Vector3 => Type.Vector3, Vector3I => Type.Vector3I, Transform2D => Type.Transform2D, Quaternion => Type.Quaternion,
            Basis => Type.Basis, Color => Type.Color, StringName => Type.StringName, NodePath => Type.NodePath,
            GodotObject => Type.Object, Callable => Type.Callable, Signal => Type.Signal,
            Godot.Collections.Dictionary => Type.Dictionary, Godot.Collections.Array => Type.Array,
            byte[] => Type.PackedByteArray, int[] => Type.PackedInt32Array, long[] => Type.PackedInt64Array,
            float[] => Type.PackedFloat32Array, double[] => Type.PackedFloat64Array, string[] => Type.PackedStringArray,
            Vector2[] => Type.PackedVector2Array, Vector3[] => Type.PackedVector3Array, Color[] => Type.PackedColorArray,
            _ => Type.Object,
        };

        public override string ToString() => _obj switch { null => "<null>", bool b => b ? "true" : "false", _ => _obj.ToString() ?? "" };

        /// <summary>Godot's Variant-to-T conversion for the managed value.</summary>
        internal object? AsType(System.Type t)
        {
            if (_obj == null) return t.IsValueType ? Activator.CreateInstance(t) : null;
            if (t.IsInstanceOfType(_obj)) return _obj;
            if (t == typeof(Variant)) return this;
            if (t == typeof(string)) return ToString();
            if (t == typeof(StringName)) return new StringName(_obj.ToString() ?? "");
            if (t == typeof(NodePath)) return new NodePath(_obj.ToString() ?? "");
            try
            {
                if (t == typeof(bool)) return Convert.ToDouble(_obj) != 0;
                if (t.IsPrimitive || t.IsEnum)
                {
                    var target = t.IsEnum ? Enum.GetUnderlyingType(t) : t;
                    object v = _obj is float or double && target != typeof(float) && target != typeof(double)
                        ? Convert.ChangeType(Math.Truncate(Convert.ToDouble(_obj)), target)
                        : Convert.ChangeType(_obj is bool bb ? (bb ? 1 : 0) : _obj, target);
                    return t.IsEnum ? Enum.ToObject(t, v) : v;
                }
            }
            catch (Exception e) when (e is InvalidCastException or FormatException or OverflowException) { }
            return t.IsValueType ? Activator.CreateInstance(t) : null;
        }

        public T As<T>() => AsType(typeof(T)) is T v ? v : default!;
        public bool AsBool() => As<bool>();
        public char AsChar() => As<char>();
        public sbyte AsSByte() => As<sbyte>();
        public short AsInt16() => As<short>();
        public int AsInt32() => As<int>();
        public long AsInt64() => As<long>();
        public byte AsByte() => As<byte>();
        public ushort AsUInt16() => As<ushort>();
        public uint AsUInt32() => As<uint>();
        public ulong AsUInt64() => As<ulong>();
        public float AsSingle() => As<float>();
        public double AsDouble() => As<double>();
        public string AsString() => ToString();
        public Vector2 AsVector2() => As<Vector2>();
        public Vector2I AsVector2I() => As<Vector2I>();
        public Vector3 AsVector3() => As<Vector3>();
        public Color AsColor() => As<Color>();
        public StringName AsStringName() => As<StringName>();
        public NodePath AsNodePath() => As<NodePath>();
        public Callable AsCallable() => As<Callable>();
        public Signal AsSignal() => As<Signal>();
        public byte[] AsByteArray() => As<byte[]>() ?? Array.Empty<byte>();
        public GodotObject AsGodotObject() => As<GodotObject>();
        public T[] AsGodotObjectArray<T>() where T : GodotObject => As<T[]>() ?? Array.Empty<T>();
        public Godot.Collections.Array<T> AsGodotArray<T>() => _obj as Godot.Collections.Array<T> ?? new Godot.Collections.Array<T>();
        public Godot.Collections.Dictionary<TKey, TValue> AsGodotDictionary<TKey, TValue>() =>
            _obj as Godot.Collections.Dictionary<TKey, TValue> ?? new Godot.Collections.Dictionary<TKey, TValue>();
        public Godot.Collections.Dictionary AsGodotDictionary() => _obj as Godot.Collections.Dictionary ?? new Godot.Collections.Dictionary();
        public Godot.Collections.Array AsGodotArray() => _obj as Godot.Collections.Array ?? new Godot.Collections.Array();

        public static Variant From<T>(in T from) => new(from);
        public static Variant CreateFrom(bool from) => new(from);
        public static Variant CreateFrom(int from) => new(from);
        public static Variant CreateFrom(long from) => new(from);
        public static Variant CreateFrom(float from) => new(from);
        public static Variant CreateFrom(double from) => new(from);
        public static Variant CreateFrom(string from) => new(from);
        public static Variant CreateFrom(GodotObject[] from) => new(from);
        public static Variant CreateFrom(GodotObject from) => new(from);
        public static Variant CreateFrom(StringName from) => new(from);
        public static Variant CreateFrom<T>(Godot.Collections.Array<T> from) => new(from);
        public static Variant CreateFrom<TKey, TValue>(Godot.Collections.Dictionary<TKey, TValue> from) => new(from);

        public static explicit operator bool(Variant from) => from.AsBool();
        public static explicit operator int(Variant from) => from.AsInt32();
        public static explicit operator long(Variant from) => from.AsInt64();
        public static explicit operator ulong(Variant from) => from.AsUInt64();
        public static explicit operator float(Variant from) => from.AsSingle();
        public static explicit operator double(Variant from) => from.AsDouble();
        public static explicit operator string(Variant from) => from.AsString();
        public static explicit operator Vector2(Variant from) => from.AsVector2();
        public static explicit operator Color(Variant from) => from.AsColor();
        public static explicit operator StringName(Variant from) => from.AsStringName();
        public static explicit operator GodotObject(Variant from) => from.AsGodotObject();

        public static implicit operator Variant(bool from) => new(from);
        public static implicit operator Variant(int from) => new(from);
        public static implicit operator Variant(long from) => new(from);
        public static implicit operator Variant(ulong from) => new(from);
        public static implicit operator Variant(float from) => new(from);
        public static implicit operator Variant(double from) => new(from);
        public static implicit operator Variant(string from) => new(from);
        public static implicit operator Variant(StringName from) => new(from);
        public static implicit operator Variant(NodePath from) => new(from);
        public static implicit operator Variant(GodotObject from) => new(from);
        public static implicit operator Variant(Vector2 from) => new(from);
        public static implicit operator Variant(Vector2I from) => new(from);
        public static implicit operator Variant(Vector3 from) => new(from);
        public static implicit operator Variant(Vector4 from) => new(from);
        public static implicit operator Variant(Color from) => new(from);
        public static implicit operator Variant(Callable from) => new(from);
        public static implicit operator Variant(Godot.Collections.Dictionary from) => new(from);
        public static implicit operator Variant(Godot.Collections.Array from) => new(from);
    }
}
