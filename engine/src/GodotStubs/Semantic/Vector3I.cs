using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;

namespace Godot;

[Serializable]
public partial struct Vector3I : IEquatable<Vector3I>
{
	public enum Axis
	{
		X,
		Y,
		Z
	}

	public int X;

	public int Y;

	public int Z;

	private static readonly Vector3I _minValue = new Vector3I(int.MinValue, int.MinValue, int.MinValue);

	private static readonly Vector3I _maxValue = new Vector3I(int.MaxValue, int.MaxValue, int.MaxValue);

	private static readonly Vector3I _zero = new Vector3I(0, 0, 0);

	private static readonly Vector3I _one = new Vector3I(1, 1, 1);

	private static readonly Vector3I _up = new Vector3I(0, 1, 0);

	private static readonly Vector3I _down = new Vector3I(0, -1, 0);

	private static readonly Vector3I _right = new Vector3I(1, 0, 0);

	private static readonly Vector3I _left = new Vector3I(-1, 0, 0);

	private static readonly Vector3I _forward = new Vector3I(0, 0, -1);

	private static readonly Vector3I _back = new Vector3I(0, 0, 1);

	public int this[int index]
	{
		readonly get
		{
			return index switch
			{
				0 => X, 
				1 => Y, 
				2 => Z, 
				_ => throw new ArgumentOutOfRangeException("index"), 
			};
		}
		set
		{
			switch (index)
			{
			case 0:
				X = value;
				break;
			case 1:
				Y = value;
				break;
			case 2:
				Z = value;
				break;
			default:
				throw new ArgumentOutOfRangeException("index");
			}
		}
	}

	public static Vector3I MinValue => _minValue;

	public static Vector3I MaxValue => _maxValue;

	public static Vector3I Zero => _zero;

	public static Vector3I One => _one;

	public static Vector3I Up => _up;

	public static Vector3I Down => _down;

	public static Vector3I Right => _right;

	public static Vector3I Left => _left;

	public static Vector3I Forward => _forward;

	public static Vector3I Back => _back;

	public readonly void Deconstruct(out int x, out int y, out int z)
	{
		x = X;
		y = Y;
		z = Z;
	}

	public readonly Vector3I Abs()
	{
		return new Vector3I(Mathf.Abs(X), Mathf.Abs(Y), Mathf.Abs(Z));
	}

	public readonly Vector3I Clamp(Vector3I min, Vector3I max)
	{
		return new Vector3I(Mathf.Clamp(X, min.X, max.X), Mathf.Clamp(Y, min.Y, max.Y), Mathf.Clamp(Z, min.Z, max.Z));
	}

	public readonly Vector3I Clamp(int min, int max)
	{
		return new Vector3I(Mathf.Clamp(X, min, max), Mathf.Clamp(Y, min, max), Mathf.Clamp(Z, min, max));
	}

	public readonly int DistanceSquaredTo(Vector3I to)
	{
		return (to - this).LengthSquared();
	}

	public readonly float DistanceTo(Vector3I to)
	{
		return (to - this).Length();
	}

	public readonly float Length()
	{
		int num = X * X;
		int num2 = Y * Y;
		int num3 = Z * Z;
		return Mathf.Sqrt(num + num2 + num3);
	}

	public readonly int LengthSquared()
	{
		int num = X * X;
		int num2 = Y * Y;
		int num3 = Z * Z;
		return num + num2 + num3;
	}

	public readonly Vector3I Max(Vector3I with)
	{
		return new Vector3I(Mathf.Max(X, with.X), Mathf.Max(Y, with.Y), Mathf.Max(Z, with.Z));
	}

	public readonly Vector3I Max(int with)
	{
		return new Vector3I(Mathf.Max(X, with), Mathf.Max(Y, with), Mathf.Max(Z, with));
	}

	public readonly Vector3I Min(Vector3I with)
	{
		return new Vector3I(Mathf.Min(X, with.X), Mathf.Min(Y, with.Y), Mathf.Min(Z, with.Z));
	}

	public readonly Vector3I Min(int with)
	{
		return new Vector3I(Mathf.Min(X, with), Mathf.Min(Y, with), Mathf.Min(Z, with));
	}

	public readonly Axis MaxAxisIndex()
	{
		if (X >= Y)
		{
			if (X >= Z)
			{
				return Axis.X;
			}
			return Axis.Z;
		}
		if (Y >= Z)
		{
			return Axis.Y;
		}
		return Axis.Z;
	}

	public readonly Axis MinAxisIndex()
	{
		if (X >= Y)
		{
			if (Y >= Z)
			{
				return Axis.Z;
			}
			return Axis.Y;
		}
		if (X >= Z)
		{
			return Axis.Z;
		}
		return Axis.X;
	}

	public readonly Vector3I Sign()
	{
		Vector3I result = this;
		result.X = Mathf.Sign(result.X);
		result.Y = Mathf.Sign(result.Y);
		result.Z = Mathf.Sign(result.Z);
		return result;
	}

	public readonly Vector3I Snapped(Vector3I step)
	{
		return new Vector3I((int)Mathf.Snapped((double)X, (double)step.X), (int)Mathf.Snapped((double)Y, (double)step.Y), (int)Mathf.Snapped((double)Z, (double)step.Z));
	}

	public readonly Vector3I Snapped(int step)
	{
		return new Vector3I((int)Mathf.Snapped((double)X, (double)step), (int)Mathf.Snapped((double)Y, (double)step), (int)Mathf.Snapped((double)Z, (double)step));
	}

	public Vector3I(int x, int y, int z)
	{
		X = x;
		Y = y;
		Z = z;
	}

	public static Vector3I operator +(Vector3I left, Vector3I right)
	{
		left.X += right.X;
		left.Y += right.Y;
		left.Z += right.Z;
		return left;
	}

	public static Vector3I operator -(Vector3I left, Vector3I right)
	{
		left.X -= right.X;
		left.Y -= right.Y;
		left.Z -= right.Z;
		return left;
	}

	public static Vector3I operator -(Vector3I vec)
	{
		vec.X = -vec.X;
		vec.Y = -vec.Y;
		vec.Z = -vec.Z;
		return vec;
	}

	public static Vector3I operator *(Vector3I vec, int scale)
	{
		vec.X *= scale;
		vec.Y *= scale;
		vec.Z *= scale;
		return vec;
	}

	public static Vector3I operator *(int scale, Vector3I vec)
	{
		vec.X *= scale;
		vec.Y *= scale;
		vec.Z *= scale;
		return vec;
	}

	public static Vector3I operator *(Vector3I left, Vector3I right)
	{
		left.X *= right.X;
		left.Y *= right.Y;
		left.Z *= right.Z;
		return left;
	}

	public static Vector3I operator /(Vector3I vec, int divisor)
	{
		vec.X /= divisor;
		vec.Y /= divisor;
		vec.Z /= divisor;
		return vec;
	}

	public static Vector3I operator /(Vector3I vec, Vector3I divisorv)
	{
		vec.X /= divisorv.X;
		vec.Y /= divisorv.Y;
		vec.Z /= divisorv.Z;
		return vec;
	}

	public static Vector3I operator %(Vector3I vec, int divisor)
	{
		vec.X %= divisor;
		vec.Y %= divisor;
		vec.Z %= divisor;
		return vec;
	}

	public static Vector3I operator %(Vector3I vec, Vector3I divisorv)
	{
		vec.X %= divisorv.X;
		vec.Y %= divisorv.Y;
		vec.Z %= divisorv.Z;
		return vec;
	}

	public static bool operator ==(Vector3I left, Vector3I right)
	{
		return left.Equals(right);
	}

	public static bool operator !=(Vector3I left, Vector3I right)
	{
		return !left.Equals(right);
	}

	public static bool operator <(Vector3I left, Vector3I right)
	{
		if (left.X == right.X)
		{
			if (left.Y == right.Y)
			{
				return left.Z < right.Z;
			}
			return left.Y < right.Y;
		}
		return left.X < right.X;
	}

	public static bool operator >(Vector3I left, Vector3I right)
	{
		if (left.X == right.X)
		{
			if (left.Y == right.Y)
			{
				return left.Z > right.Z;
			}
			return left.Y > right.Y;
		}
		return left.X > right.X;
	}

	public static bool operator <=(Vector3I left, Vector3I right)
	{
		if (left.X == right.X)
		{
			if (left.Y == right.Y)
			{
				return left.Z <= right.Z;
			}
			return left.Y < right.Y;
		}
		return left.X < right.X;
	}

	public static bool operator >=(Vector3I left, Vector3I right)
	{
		if (left.X == right.X)
		{
			if (left.Y == right.Y)
			{
				return left.Z >= right.Z;
			}
			return left.Y > right.Y;
		}
		return left.X > right.X;
	}

	public static implicit operator Vector3(Vector3I value)
	{
		return new Vector3(value.X, value.Y, value.Z);
	}

	public static explicit operator Vector3I(Vector3 value)
	{
		return new Vector3I((int)value.X, (int)value.Y, (int)value.Z);
	}

	public override readonly bool Equals([NotNullWhen(true)] object? obj)
	{
		if (obj is Vector3I other)
		{
			return Equals(other);
		}
		return false;
	}

	public readonly bool Equals(Vector3I other)
	{
		if (X == other.X && Y == other.Y)
		{
			return Z == other.Z;
		}
		return false;
	}

	public override readonly int GetHashCode()
	{
		return HashCode.Combine(X, Y, Z);
	}

	public override readonly string ToString()
	{
		return ToString(null);
	}

	public readonly string ToString(string? format)
	{
		return $"({X.ToString(format, CultureInfo.InvariantCulture)}, {Y.ToString(format, CultureInfo.InvariantCulture)}, {Z.ToString(format, CultureInfo.InvariantCulture)})";
	}
}
