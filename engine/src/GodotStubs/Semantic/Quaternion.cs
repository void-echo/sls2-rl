using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;

namespace Godot;

[Serializable]
public partial struct Quaternion : IEquatable<Quaternion>
{
	public float X;

	public float Y;

	public float Z;

	public float W;

	private static readonly Quaternion _identity = new Quaternion(0f, 0f, 0f, 1f);

	public float this[int index]
	{
		readonly get
		{
			return index switch
			{
				0 => X, 
				1 => Y, 
				2 => Z, 
				3 => W, 
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
			case 3:
				W = value;
				break;
			default:
				throw new ArgumentOutOfRangeException("index");
			}
		}
	}

	public static Quaternion Identity => _identity;

	public readonly float AngleTo(Quaternion to)
	{
		float num = Dot(to);
		return Mathf.Acos(Mathf.Clamp(num * num * 2f - 1f, -1f, 1f));
	}

	public readonly Quaternion SphericalCubicInterpolate(Quaternion b, Quaternion preA, Quaternion postB, float weight)
	{
		Quaternion rotationQuaternion = new Basis(this).GetRotationQuaternion();
		Quaternion rotationQuaternion2 = new Basis(preA).GetRotationQuaternion();
		Quaternion rotationQuaternion3 = new Basis(b).GetRotationQuaternion();
		Quaternion rotationQuaternion4 = new Basis(postB).GetRotationQuaternion();
		rotationQuaternion2 = ((Math.Sign(rotationQuaternion.Dot(rotationQuaternion2)) < 0) ? (-rotationQuaternion2) : rotationQuaternion2);
		bool num = Math.Sign(rotationQuaternion.Dot(rotationQuaternion3)) < 0;
		rotationQuaternion3 = (num ? (-rotationQuaternion3) : rotationQuaternion3);
		rotationQuaternion4 = ((num ? (rotationQuaternion3.Dot(rotationQuaternion4) <= 0f) : (Math.Sign(rotationQuaternion3.Dot(rotationQuaternion4)) < 0)) ? (-rotationQuaternion4) : rotationQuaternion4);
		Quaternion quaternion = new Quaternion(0f, 0f, 0f, 0f);
		Quaternion quaternion2 = (rotationQuaternion.Inverse() * rotationQuaternion3).Log();
		Quaternion quaternion3 = (rotationQuaternion.Inverse() * rotationQuaternion2).Log();
		Quaternion quaternion4 = (rotationQuaternion.Inverse() * rotationQuaternion4).Log();
		Quaternion quaternion5 = new Quaternion(Mathf.CubicInterpolate(quaternion.X, quaternion2.X, quaternion3.X, quaternion4.X, weight), Mathf.CubicInterpolate(quaternion.Y, quaternion2.Y, quaternion3.Y, quaternion4.Y, weight), Mathf.CubicInterpolate(quaternion.Z, quaternion2.Z, quaternion3.Z, quaternion4.Z, weight), 0f);
		Quaternion quaternion6 = rotationQuaternion * quaternion5.Exp();
		quaternion = (rotationQuaternion3.Inverse() * rotationQuaternion).Log();
		quaternion2 = new Quaternion(0f, 0f, 0f, 0f);
		quaternion3 = (rotationQuaternion3.Inverse() * rotationQuaternion2).Log();
		quaternion4 = (rotationQuaternion3.Inverse() * rotationQuaternion4).Log();
		quaternion5 = new Quaternion(Mathf.CubicInterpolate(quaternion.X, quaternion2.X, quaternion3.X, quaternion4.X, weight), Mathf.CubicInterpolate(quaternion.Y, quaternion2.Y, quaternion3.Y, quaternion4.Y, weight), Mathf.CubicInterpolate(quaternion.Z, quaternion2.Z, quaternion3.Z, quaternion4.Z, weight), 0f);
		Quaternion to = rotationQuaternion3 * quaternion5.Exp();
		return quaternion6.Slerp(to, weight);
	}

	public readonly Quaternion SphericalCubicInterpolateInTime(Quaternion b, Quaternion preA, Quaternion postB, float weight, float bT, float preAT, float postBT)
	{
		Quaternion rotationQuaternion = new Basis(this).GetRotationQuaternion();
		Quaternion rotationQuaternion2 = new Basis(preA).GetRotationQuaternion();
		Quaternion rotationQuaternion3 = new Basis(b).GetRotationQuaternion();
		Quaternion rotationQuaternion4 = new Basis(postB).GetRotationQuaternion();
		rotationQuaternion2 = ((Math.Sign(rotationQuaternion.Dot(rotationQuaternion2)) < 0) ? (-rotationQuaternion2) : rotationQuaternion2);
		bool num = Math.Sign(rotationQuaternion.Dot(rotationQuaternion3)) < 0;
		rotationQuaternion3 = (num ? (-rotationQuaternion3) : rotationQuaternion3);
		rotationQuaternion4 = ((num ? (rotationQuaternion3.Dot(rotationQuaternion4) <= 0f) : (Math.Sign(rotationQuaternion3.Dot(rotationQuaternion4)) < 0)) ? (-rotationQuaternion4) : rotationQuaternion4);
		Quaternion quaternion = new Quaternion(0f, 0f, 0f, 0f);
		Quaternion quaternion2 = (rotationQuaternion.Inverse() * rotationQuaternion3).Log();
		Quaternion quaternion3 = (rotationQuaternion.Inverse() * rotationQuaternion2).Log();
		Quaternion quaternion4 = (rotationQuaternion.Inverse() * rotationQuaternion4).Log();
		Quaternion quaternion5 = new Quaternion(Mathf.CubicInterpolateInTime(quaternion.X, quaternion2.X, quaternion3.X, quaternion4.X, weight, bT, preAT, postBT), Mathf.CubicInterpolateInTime(quaternion.Y, quaternion2.Y, quaternion3.Y, quaternion4.Y, weight, bT, preAT, postBT), Mathf.CubicInterpolateInTime(quaternion.Z, quaternion2.Z, quaternion3.Z, quaternion4.Z, weight, bT, preAT, postBT), 0f);
		Quaternion quaternion6 = rotationQuaternion * quaternion5.Exp();
		quaternion = (rotationQuaternion3.Inverse() * rotationQuaternion).Log();
		quaternion2 = new Quaternion(0f, 0f, 0f, 0f);
		quaternion3 = (rotationQuaternion3.Inverse() * rotationQuaternion2).Log();
		quaternion4 = (rotationQuaternion3.Inverse() * rotationQuaternion4).Log();
		quaternion5 = new Quaternion(Mathf.CubicInterpolateInTime(quaternion.X, quaternion2.X, quaternion3.X, quaternion4.X, weight, bT, preAT, postBT), Mathf.CubicInterpolateInTime(quaternion.Y, quaternion2.Y, quaternion3.Y, quaternion4.Y, weight, bT, preAT, postBT), Mathf.CubicInterpolateInTime(quaternion.Z, quaternion2.Z, quaternion3.Z, quaternion4.Z, weight, bT, preAT, postBT), 0f);
		Quaternion to = rotationQuaternion3 * quaternion5.Exp();
		return quaternion6.Slerp(to, weight);
	}

	public readonly float Dot(Quaternion b)
	{
		return X * b.X + Y * b.Y + Z * b.Z + W * b.W;
	}

	public readonly Quaternion Exp()
	{
		Vector3 vector = new Vector3(X, Y, Z);
		float num = vector.Length();
		vector = vector.Normalized();
		if (num < 1E-06f || !vector.IsNormalized())
		{
			return new Quaternion(0f, 0f, 0f, 1f);
		}
		return new Quaternion(vector, num);
	}

	public readonly float GetAngle()
	{
		return 2f * Mathf.Acos(W);
	}

	public readonly Vector3 GetAxis()
	{
		if (Mathf.Abs(W) > 0.999999f)
		{
			return new Vector3(X, Y, Z);
		}
		float num = 1f / Mathf.Sqrt(1f - W * W);
		return new Vector3(X * num, Y * num, Z * num);
	}

	public readonly Vector3 GetEuler(EulerOrder order = EulerOrder.Yxz)
	{
		return new Basis(this).GetEuler(order);
	}

	public readonly Quaternion Inverse()
	{
		return new Quaternion(0f - X, 0f - Y, 0f - Z, W);
	}

	public readonly bool IsFinite()
	{
		if (Mathf.IsFinite(X) && Mathf.IsFinite(Y) && Mathf.IsFinite(Z))
		{
			return Mathf.IsFinite(W);
		}
		return false;
	}

	public readonly bool IsNormalized()
	{
		return Mathf.Abs(LengthSquared() - 1f) <= 1E-06f;
	}

	public readonly Quaternion Log()
	{
		Vector3 vector = GetAxis() * GetAngle();
		return new Quaternion(vector.X, vector.Y, vector.Z, 0f);
	}

	public readonly float Length()
	{
		return Mathf.Sqrt(LengthSquared());
	}

	public readonly float LengthSquared()
	{
		return Dot(this);
	}

	public readonly Quaternion Normalized()
	{
		return this / Length();
	}

	public readonly Quaternion Slerp(Quaternion to, float weight)
	{
		float num = Dot(to);
		Quaternion quaternion = default(Quaternion);
		if ((double)num < 0.0)
		{
			num = 0f - num;
			quaternion = -to;
		}
		else
		{
			quaternion = to;
		}
		float num4;
		float num5;
		if (1.0 - (double)num > 9.999999974752427E-07)
		{
			float num2 = Mathf.Acos(num);
			float num3 = Mathf.Sin(num2);
			num4 = Mathf.Sin((1f - weight) * num2) / num3;
			num5 = Mathf.Sin(weight * num2) / num3;
		}
		else
		{
			num4 = 1f - weight;
			num5 = weight;
		}
		return new Quaternion(num4 * X + num5 * quaternion.X, num4 * Y + num5 * quaternion.Y, num4 * Z + num5 * quaternion.Z, num4 * W + num5 * quaternion.W);
	}

	public readonly Quaternion Slerpni(Quaternion to, float weight)
	{
		float s = Dot(to);
		if (Mathf.Abs(s) > 0.9999f)
		{
			return this;
		}
		float num = Mathf.Acos(s);
		float num2 = 1f / Mathf.Sin(num);
		float num3 = Mathf.Sin(weight * num) * num2;
		float num4 = Mathf.Sin((1f - weight) * num) * num2;
		return new Quaternion(num4 * X + num3 * to.X, num4 * Y + num3 * to.Y, num4 * Z + num3 * to.Z, num4 * W + num3 * to.W);
	}

	public Quaternion(float x, float y, float z, float w)
	{
		X = x;
		Y = y;
		Z = z;
		W = w;
	}

	public Quaternion(Basis basis)
	{
		this = basis.GetQuaternion();
	}

	public Quaternion(Vector3 axis, float angle)
	{
		float num = axis.Length();
		if (num == 0f)
		{
			X = 0f;
			Y = 0f;
			Z = 0f;
			W = 0f;
			return;
		}
		(float Sin, float Cos) tuple = Mathf.SinCos(angle * 0.5f);
		float item = tuple.Sin;
		float item2 = tuple.Cos;
		float num2 = item / num;
		X = axis.X * num2;
		Y = axis.Y * num2;
		Z = axis.Z * num2;
		W = item2;
	}

	public Quaternion(Vector3 arcFrom, Vector3 arcTo)
	{
		Vector3 vector = arcFrom.Normalized();
		Vector3 with = arcTo.Normalized();
		float num = vector.Dot(with);
		if (Mathf.Abs(num) > 0.99999976f)
		{
			if (num >= 0f)
			{
				X = 0f;
				Y = 0f;
				Z = 0f;
				W = 1f;
				return;
			}
			Vector3 anyPerpendicular = vector.GetAnyPerpendicular();
			X = anyPerpendicular.X;
			Y = anyPerpendicular.Y;
			Z = anyPerpendicular.Z;
			W = 0f;
		}
		else
		{
			Vector3 vector2 = vector.Cross(with);
			float num2 = Mathf.Sqrt((1f + num) * 2f);
			float num3 = 1f / num2;
			X = vector2.X * num3;
			Y = vector2.Y * num3;
			Z = vector2.Z * num3;
			W = num2 * 0.5f;
		}
		this = Normalized();
	}

	public static Quaternion FromEuler(Vector3 eulerYXZ)
	{
		float s = eulerYXZ.Y * 0.5f;
		float s2 = eulerYXZ.X * 0.5f;
		float s3 = eulerYXZ.Z * 0.5f;
		var (num, num2) = Mathf.SinCos(s);
		var (num3, num4) = Mathf.SinCos(s2);
		var (num5, num6) = Mathf.SinCos(s3);
		return new Quaternion(num * num4 * num5 + num2 * num3 * num6, num * num4 * num6 - num2 * num3 * num5, num2 * num4 * num5 - num * num3 * num6, num * num3 * num5 + num2 * num4 * num6);
	}

	public static Quaternion operator *(Quaternion left, Quaternion right)
	{
		return new Quaternion(left.W * right.X + left.X * right.W + left.Y * right.Z - left.Z * right.Y, left.W * right.Y + left.Y * right.W + left.Z * right.X - left.X * right.Z, left.W * right.Z + left.Z * right.W + left.X * right.Y - left.Y * right.X, left.W * right.W - left.X * right.X - left.Y * right.Y - left.Z * right.Z);
	}

	public static Vector3 operator *(Quaternion quaternion, Vector3 vector)
	{
		Vector3 vector2 = new Vector3(quaternion.X, quaternion.Y, quaternion.Z);
		Vector3 vector3 = vector2.Cross(vector);
		return vector + (vector3 * quaternion.W + vector2.Cross(vector3)) * 2f;
	}

	public static Vector3 operator *(Vector3 vector, Quaternion quaternion)
	{
		return quaternion.Inverse() * vector;
	}

	public static Quaternion operator +(Quaternion left, Quaternion right)
	{
		return new Quaternion(left.X + right.X, left.Y + right.Y, left.Z + right.Z, left.W + right.W);
	}

	public static Quaternion operator -(Quaternion left, Quaternion right)
	{
		return new Quaternion(left.X - right.X, left.Y - right.Y, left.Z - right.Z, left.W - right.W);
	}

	public static Quaternion operator -(Quaternion quat)
	{
		return new Quaternion(0f - quat.X, 0f - quat.Y, 0f - quat.Z, 0f - quat.W);
	}

	public static Quaternion operator *(Quaternion left, float right)
	{
		return new Quaternion(left.X * right, left.Y * right, left.Z * right, left.W * right);
	}

	public static Quaternion operator *(float left, Quaternion right)
	{
		return new Quaternion(right.X * left, right.Y * left, right.Z * left, right.W * left);
	}

	public static Quaternion operator /(Quaternion left, float right)
	{
		return left * (1f / right);
	}

	public static bool operator ==(Quaternion left, Quaternion right)
	{
		return left.Equals(right);
	}

	public static bool operator !=(Quaternion left, Quaternion right)
	{
		return !left.Equals(right);
	}

	public override readonly bool Equals([NotNullWhen(true)] object? obj)
	{
		if (obj is Quaternion other)
		{
			return Equals(other);
		}
		return false;
	}

	public readonly bool Equals(Quaternion other)
	{
		if (X == other.X && Y == other.Y && Z == other.Z)
		{
			return W == other.W;
		}
		return false;
	}

	public readonly bool IsEqualApprox(Quaternion other)
	{
		if (Mathf.IsEqualApprox(X, other.X) && Mathf.IsEqualApprox(Y, other.Y) && Mathf.IsEqualApprox(Z, other.Z))
		{
			return Mathf.IsEqualApprox(W, other.W);
		}
		return false;
	}

	public override readonly int GetHashCode()
	{
		return HashCode.Combine(X, Y, Z, W);
	}

	public override readonly string ToString()
	{
		return ToString(null);
	}

	public readonly string ToString(string? format)
	{
		return $"({X.ToString(format, CultureInfo.InvariantCulture)}, {Y.ToString(format, CultureInfo.InvariantCulture)}, {Z.ToString(format, CultureInfo.InvariantCulture)}, {W.ToString(format, CultureInfo.InvariantCulture)})";
	}
}
