using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace Godot;

[Serializable]
public partial struct Transform2D : IEquatable<Transform2D>
{
	public Vector2 X;

	public Vector2 Y;

	public Vector2 Origin;

	private static readonly Transform2D _identity = new Transform2D(1f, 0f, 0f, 1f, 0f, 0f);

	private static readonly Transform2D _flipX = new Transform2D(-1f, 0f, 0f, 1f, 0f, 0f);

	private static readonly Transform2D _flipY = new Transform2D(1f, 0f, 0f, -1f, 0f, 0f);

	public readonly float Rotation => Mathf.Atan2(X.Y, X.X);

	public readonly Vector2 Scale
	{
		get
		{
			float num = Mathf.Sign(Determinant());
			return new Vector2(X.Length(), num * Y.Length());
		}
	}

	public readonly float Skew
	{
		get
		{
			float num = Mathf.Sign(Determinant());
			return Mathf.Acos(X.Normalized().Dot(num * Y.Normalized())) - (float)Math.PI / 2f;
		}
	}

	public Vector2 this[int column]
	{
		readonly get
		{
			return column switch
			{
				0 => X, 
				1 => Y, 
				2 => Origin, 
				_ => throw new ArgumentOutOfRangeException("column"), 
			};
		}
		set
		{
			switch (column)
			{
			case 0:
				X = value;
				break;
			case 1:
				Y = value;
				break;
			case 2:
				Origin = value;
				break;
			default:
				throw new ArgumentOutOfRangeException("column");
			}
		}
	}

	public float this[int column, int row]
	{
		readonly get
		{
			return this[column][row];
		}
		set
		{
			Vector2 value2 = this[column];
			value2[row] = value;
			this[column] = value2;
		}
	}

	public static Transform2D Identity => _identity;

	public static Transform2D FlipX => _flipX;

	public static Transform2D FlipY => _flipY;

	public readonly Transform2D AffineInverse()
	{
		float num = Determinant();
		if (num == 0f)
		{
			throw new InvalidOperationException("Matrix determinant is zero and cannot be inverted.");
		}
		Transform2D result = this;
		result[0, 0] = this[1, 1];
		result[1, 1] = this[0, 0];
		float num2 = 1f / num;
		result[0] *= new Vector2(num2, 0f - num2);
		result[1] *= new Vector2(0f - num2, num2);
		result[2] = result.BasisXform(-result[2]);
		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public readonly float Determinant()
	{
		return X.X * Y.Y - X.Y * Y.X;
	}

	public readonly Vector2 BasisXform(Vector2 v)
	{
		return new Vector2(Tdotx(v), Tdoty(v));
	}

	public readonly Vector2 BasisXformInv(Vector2 v)
	{
		return new Vector2(X.Dot(v), Y.Dot(v));
	}

	public readonly Transform2D InterpolateWith(Transform2D transform, float weight)
	{
		return new Transform2D(Mathf.LerpAngle(Rotation, transform.Rotation, weight), Scale.Lerp(transform.Scale, weight), Mathf.LerpAngle(Skew, transform.Skew, weight), Origin.Lerp(transform.Origin, weight));
	}

	public readonly Transform2D Inverse()
	{
		Transform2D result = this;
		result.X.Y = Y.X;
		result.Y.X = X.Y;
		result.Origin = result.BasisXform(-result.Origin);
		return result;
	}

	public readonly bool IsFinite()
	{
		if (X.IsFinite() && Y.IsFinite())
		{
			return Origin.IsFinite();
		}
		return false;
	}

	public readonly Transform2D Orthonormalized()
	{
		Transform2D result = this;
		Vector2 x = result.X;
		Vector2 y = result.Y;
		x.Normalize();
		y -= x * x.Dot(y);
		y.Normalize();
		result.X = x;
		result.Y = y;
		return result;
	}

	public readonly Transform2D Rotated(float angle)
	{
		return new Transform2D(angle, default(Vector2)) * this;
	}

	public readonly Transform2D RotatedLocal(float angle)
	{
		return this * new Transform2D(angle, default(Vector2));
	}

	public readonly Transform2D Scaled(Vector2 scale)
	{
		Transform2D result = this;
		result.X *= scale;
		result.Y *= scale;
		result.Origin *= scale;
		return result;
	}

	public readonly Transform2D ScaledLocal(Vector2 scale)
	{
		Transform2D result = this;
		result.X *= scale;
		result.Y *= scale;
		return result;
	}

	private readonly float Tdotx(Vector2 with)
	{
		return this[0, 0] * with[0] + this[1, 0] * with[1];
	}

	private readonly float Tdoty(Vector2 with)
	{
		return this[0, 1] * with[0] + this[1, 1] * with[1];
	}

	public readonly Transform2D Translated(Vector2 offset)
	{
		Transform2D result = this;
		result.Origin += offset;
		return result;
	}

	public readonly Transform2D TranslatedLocal(Vector2 offset)
	{
		Transform2D result = this;
		result.Origin += result.BasisXform(offset);
		return result;
	}

	public Transform2D(Vector2 xAxis, Vector2 yAxis, Vector2 originPos)
	{
		X = xAxis;
		Y = yAxis;
		Origin = originPos;
	}

	public Transform2D(float xx, float xy, float yx, float yy, float ox, float oy)
	{
		X = new Vector2(xx, xy);
		Y = new Vector2(yx, yy);
		Origin = new Vector2(ox, oy);
	}

	public Transform2D(float rotation, Vector2 origin)
	{
		(float Sin, float Cos) tuple = Mathf.SinCos(rotation);
		float item = tuple.Sin;
		float item2 = tuple.Cos;
		X.X = (Y.Y = item2);
		X.Y = (Y.X = item);
		Y.X *= -1f;
		Origin = origin;
	}

	public Transform2D(float rotation, Vector2 scale, float skew, Vector2 origin)
	{
		(float Sin, float Cos) tuple = Mathf.SinCos(rotation);
		float item = tuple.Sin;
		float item2 = tuple.Cos;
		(float Sin, float Cos) tuple2 = Mathf.SinCos(rotation + skew);
		float item3 = tuple2.Sin;
		float item4 = tuple2.Cos;
		X.X = item2 * scale.X;
		Y.Y = item4 * scale.Y;
		Y.X = (0f - item3) * scale.Y;
		X.Y = item * scale.X;
		Origin = origin;
	}

	public static Transform2D operator *(Transform2D left, Transform2D right)
	{
		left.Origin = left * right.Origin;
		float x = left.Tdotx(right.X);
		float y = left.Tdoty(right.X);
		float x2 = left.Tdotx(right.Y);
		float y2 = left.Tdoty(right.Y);
		left.X.X = x;
		left.X.Y = y;
		left.Y.X = x2;
		left.Y.Y = y2;
		return left;
	}

	public static Vector2 operator *(Transform2D transform, Vector2 vector)
	{
		return new Vector2(transform.Tdotx(vector), transform.Tdoty(vector)) + transform.Origin;
	}

	public static Vector2 operator *(Vector2 vector, Transform2D transform)
	{
		Vector2 with = vector - transform.Origin;
		return new Vector2(transform.X.Dot(with), transform.Y.Dot(with));
	}

	public static Rect2 operator *(Transform2D transform, Rect2 rect)
	{
		Vector2 vector = transform * rect.Position;
		Vector2 vector2 = transform.X * rect.Size.X;
		Vector2 vector3 = transform.Y * rect.Size.Y;
		return new Rect2(vector, default(Vector2)).Expand(vector + vector2).Expand(vector + vector3).Expand(vector + vector2 + vector3);
	}

	public static Rect2 operator *(Rect2 rect, Transform2D transform)
	{
		Vector2 position = rect.Position * transform;
		Vector2 to = new Vector2(rect.Position.X, rect.Position.Y + rect.Size.Y) * transform;
		Vector2 to2 = new Vector2(rect.Position.X + rect.Size.X, rect.Position.Y + rect.Size.Y) * transform;
		Vector2 to3 = new Vector2(rect.Position.X + rect.Size.X, rect.Position.Y) * transform;
		return new Rect2(position, default(Vector2)).Expand(to).Expand(to2).Expand(to3);
	}

	public static Vector2[] operator *(Transform2D transform, Vector2[] array)
	{
		Vector2[] array2 = new Vector2[array.Length];
		for (int i = 0; i < array.Length; i++)
		{
			array2[i] = transform * array[i];
		}
		return array2;
	}

	public static Vector2[] operator *(Vector2[] array, Transform2D transform)
	{
		Vector2[] array2 = new Vector2[array.Length];
		for (int i = 0; i < array.Length; i++)
		{
			array2[i] = array[i] * transform;
		}
		return array2;
	}

	public static bool operator ==(Transform2D left, Transform2D right)
	{
		return left.Equals(right);
	}

	public static bool operator !=(Transform2D left, Transform2D right)
	{
		return !left.Equals(right);
	}

	public override readonly bool Equals([NotNullWhen(true)] object? obj)
	{
		if (obj is Transform2D other)
		{
			return Equals(other);
		}
		return false;
	}

	public readonly bool Equals(Transform2D other)
	{
		if (X.Equals(other.X) && Y.Equals(other.Y))
		{
			return Origin.Equals(other.Origin);
		}
		return false;
	}

	public readonly bool IsEqualApprox(Transform2D other)
	{
		if (X.IsEqualApprox(other.X) && Y.IsEqualApprox(other.Y))
		{
			return Origin.IsEqualApprox(other.Origin);
		}
		return false;
	}

	public override readonly int GetHashCode()
	{
		return HashCode.Combine(X, Y, Origin);
	}

	public override readonly string ToString()
	{
		return ToString(null);
	}

	public readonly string ToString(string? format)
	{
		return $"[X: {X.ToString(format)}, Y: {Y.ToString(format)}, O: {Origin.ToString(format)}]";
	}
}
