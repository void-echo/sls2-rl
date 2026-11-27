// StringName, NodePath and the string helpers sts2.dll uses. In Godot these wrap interned native strings; headless
// they wrap the managed string, with the same equality (by value), implicit conversions and null behaviour.
using System;
using System.Diagnostics.CodeAnalysis;
using Godot.NativeInterop;

namespace Godot
{
    public sealed partial class StringName : IDisposable, IEquatable<StringName?>
    {
        private readonly string _value;

        public StringName() { _value = ""; }
        public StringName(string name) { _value = name ?? ""; }

        public bool IsEmpty => _value.Length == 0;
        public void Dispose() { }
        public void Dispose(bool disposing) { }

        public static implicit operator StringName(string from) => new(from);
        public static implicit operator string?(StringName? from) => from?._value;
        public override string ToString() => _value;

        public static bool operator ==(StringName? left, StringName? right) =>
            left is null ? right is null : left.Equals(right);
        public static bool operator !=(StringName? left, StringName? right) => !(left == right);
        public bool Equals([NotNullWhen(true)] StringName? other) => other is not null && other._value == _value;

        // godot_string_name is the native handle; headless it carries no value, so it only equals an empty name.
        public static bool operator ==(StringName? left, in godot_string_name right) => left is null || left.IsEmpty;
        public static bool operator !=(StringName? left, in godot_string_name right) => !(left == right);
        public static bool operator ==(in godot_string_name left, StringName? right) => right is null || right.IsEmpty;
        public static bool operator !=(in godot_string_name left, StringName? right) => !(left == right);
        public bool Equals(in godot_string_name other) => IsEmpty;

        public override bool Equals([NotNullWhen(true)] object? obj) => obj is StringName s && Equals(s);
        public override int GetHashCode() => _value.GetHashCode();
    }

    public sealed partial class NodePath : IDisposable, IEquatable<NodePath?>
    {
        private readonly string _path;

        public NodePath() { _path = ""; }
        public NodePath(string path) { _path = path ?? ""; }

        public bool IsEmpty => _path.Length == 0;
        public void Dispose() { }
        public void Dispose(bool disposing) { }

        public static implicit operator NodePath(string from) => new(from);
        public static implicit operator string?(NodePath? from) => from?._path;
        public override string ToString() => _path;

        public static bool operator ==(NodePath? left, NodePath? right) => left is null ? right is null : left.Equals(right);
        public static bool operator !=(NodePath? left, NodePath? right) => !(left == right);
        public bool Equals([NotNullWhen(true)] NodePath? other) => other is not null && other._path == _path;
        public override bool Equals([NotNullWhen(true)] object? obj) => obj is NodePath p && Equals(p);
        public override int GetHashCode() => _path.GetHashCode();
        public bool IsAbsolute() => _path.StartsWith('/');
    }

    /// <summary>
    /// The Godot string helpers sts2.dll calls. GetBaseDir / GetFile / PathJoin / HexEncode are Godot's managed
    /// implementations (GodotSharp/Core/StringExtensions.cs, MIT); Capitalize and ToSnakeCase call into the engine in
    /// Godot and are ports of String::capitalize / String::to_snake_case (core/string/ustring.cpp, MIT).
    /// </summary>
    public static partial class StringExtensions
    {
        public static string Capitalize(this string instance)
        {
            var aux = CamelcaseToUnderscore(instance).Replace("_", " ").Trim();
            var cap = new System.Text.StringBuilder();
            var slices = aux.Split(' ');
            for (int i = 0; i < slices.Length; i++)
            {
                var slice = slices[i];
                if (slice.Length == 0) continue;
                if (cap.Length > 0) cap.Append(' ');
                cap.Append(char.ToUpperInvariant(slice[0])).Append(slice, 1, slice.Length - 1);
            }
            return cap.ToString();
        }

        public static string ToSnakeCase(this string instance) => CamelcaseToUnderscore(instance).Replace(" ", "_").Trim();

        /// <summary>String::_camelcase_to_underscore: underscores at aA, AAa, 2Aa, 2aa, A2 and a2 boundaries, then lower case.</summary>
        private static string CamelcaseToUnderscore(string s)
        {
            var sb = new System.Text.StringBuilder();
            int start = 0;
            for (int i = 1; i < s.Length; i++)
            {
                bool prevUpper = char.IsUpper(s[i - 1]), prevLower = char.IsLower(s[i - 1]), prevDigit = char.IsAsciiDigit(s[i - 1]);
                bool curUpper = char.IsUpper(s[i]), curLower = char.IsLower(s[i]), curDigit = char.IsAsciiDigit(s[i]);
                bool nextLower = i + 1 < s.Length && char.IsLower(s[i + 1]);
                bool a = prevLower && curUpper;
                bool b = (prevUpper || prevDigit) && curUpper && nextLower;
                bool c = prevDigit && curLower && nextLower;
                bool d = (prevUpper || prevLower) && curDigit;
                if (a || b || c || d)
                {
                    sb.Append(s, start, i - start).Append('_');
                    start = i;
                }
            }
            sb.Append(s, start, s.Length - start);
            return sb.ToString().ToLowerInvariant();
        }

        public static string GetBaseDir(this string instance)
        {
            int num = instance.IndexOf("://", StringComparison.Ordinal);
            string text = string.Empty;
            string rest;
            if (num != -1)
            {
                int n = num + 3;
                rest = instance.Substring(n);
                text = instance.Substring(0, n);
            }
            else if (instance.StartsWith('/'))
            {
                rest = instance.Substring(1);
                text = "/";
            }
            else rest = instance;
            int sep = Mathf.Max(rest.LastIndexOf('/'), rest.LastIndexOf('\\'));
            return sep == -1 ? text : text + rest.Substring(0, sep);
        }

        public static string GetFile(this string instance)
        {
            int sep = Mathf.Max(instance.LastIndexOf('/'), instance.LastIndexOf('\\'));
            return sep == -1 ? instance : instance.Substring(sep + 1);
        }

        public static string PathJoin(this string instance, string file)
        {
            if (instance.Length == 0) return file;
            if (instance[instance.Length - 1] == '/' || (file.Length > 0 && file[0] == '/')) return instance + file;
            return instance + "/" + file;
        }

        internal static string HexEncode(this byte b)
        {
            string text = string.Empty;
            for (int i = 0; i < 2; i++)
            {
                int num = b & 0xF;
                char c = (num >= 10) ? (char)(97 + num - 10) : (char)(48 + num);
                b >>= 4;
                text = c + text;
            }
            return text;
        }
    }
}

namespace Godot.NativeInterop
{
    /// <summary>Native interned-string handle (opaque; headless it is always empty).</summary>
    public partial struct godot_string_name
    {
    }
}
