using System.Collections;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace ParityOracle;

/// <summary>
/// Minimal JSON writer for Dictionary/List/primitive trees (no reflection), plus a canonical form used to match
/// candidate descriptors from the headless action log against candidates computed in the client.
/// </summary>
public static class Json
{
    public static string Write(object? v, bool sortKeys = false)
    {
        var sb = new StringBuilder();
        WriteTo(sb, v, sortKeys);
        return sb.ToString();
    }

    private static void WriteTo(StringBuilder sb, object? v, bool sortKeys)
    {
        switch (v)
        {
            case null: sb.Append("null"); break;
            case string s: Str(sb, s); break;
            case bool b: sb.Append(b ? "true" : "false"); break;
            case int or long or short or byte or uint or ulong or sbyte or ushort:
                sb.Append(Convert.ToString(v, CultureInfo.InvariantCulture)); break;
            case double or float or decimal: Num(sb, Convert.ToDouble(v, CultureInfo.InvariantCulture)); break;
            case Enum e: Str(sb, e.ToString()); break;
            case JsonElement je: WriteElement(sb, je, sortKeys); break;
            case IDictionary d:
            {
                sb.Append('{');
                var keys = d.Keys.Cast<object>().Select(k => k.ToString()!).ToList();
                if (sortKeys) keys.Sort(StringComparer.Ordinal);
                bool first = true;
                foreach (var k in keys)
                {
                    if (!first) sb.Append(',');
                    first = false;
                    Str(sb, k);
                    sb.Append(':');
                    WriteTo(sb, d[k], sortKeys);
                }
                sb.Append('}');
                break;
            }
            case IEnumerable en:
            {
                sb.Append('[');
                bool first = true;
                foreach (var x in en)
                {
                    if (!first) sb.Append(',');
                    first = false;
                    WriteTo(sb, x, sortKeys);
                }
                sb.Append(']');
                break;
            }
            default: Str(sb, v.ToString() ?? ""); break;
        }
    }

    private static void WriteElement(StringBuilder sb, JsonElement e, bool sortKeys)
    {
        switch (e.ValueKind)
        {
            case JsonValueKind.Object:
            {
                var props = e.EnumerateObject().ToList();
                if (sortKeys) props.Sort((a, b) => string.CompareOrdinal(a.Name, b.Name));
                sb.Append('{');
                for (int i = 0; i < props.Count; i++)
                {
                    if (i > 0) sb.Append(',');
                    Str(sb, props[i].Name);
                    sb.Append(':');
                    WriteElement(sb, props[i].Value, sortKeys);
                }
                sb.Append('}');
                break;
            }
            case JsonValueKind.Array:
            {
                sb.Append('[');
                bool first = true;
                foreach (var x in e.EnumerateArray())
                {
                    if (!first) sb.Append(',');
                    first = false;
                    WriteElement(sb, x, sortKeys);
                }
                sb.Append(']');
                break;
            }
            case JsonValueKind.String: Str(sb, e.GetString() ?? ""); break;
            case JsonValueKind.Number: Num(sb, e.GetDouble()); break;
            case JsonValueKind.True: sb.Append("true"); break;
            case JsonValueKind.False: sb.Append("false"); break;
            default: sb.Append("null"); break;
        }
    }

    private static void Num(StringBuilder sb, double d)
    {
        if (double.IsNaN(d) || double.IsInfinity(d)) { sb.Append("null"); return; }
        if (Math.Abs(d) < 1e15 && d == Math.Floor(d)) sb.Append(((long)d).ToString(CultureInfo.InvariantCulture));
        else sb.Append(d.ToString("R", CultureInfo.InvariantCulture));
    }

    private static void Str(StringBuilder sb, string s)
    {
        sb.Append('"');
        foreach (var c in s)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                default:
                    if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4"));
                    else sb.Append(c);
                    break;
            }
        }
        sb.Append('"');
    }

    /// <summary>Canonical text for matching: sorted keys, integral numbers without decimals.</summary>
    public static string Canon(object? v) => Write(v, sortKeys: true);
}
