using System;
using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;

namespace Godot;

[Serializable]
public partial struct Basis : IEquatable<Basis>
{
	public Vector3 Row0;

	public Vector3 Row1;

	public Vector3 Row2;

	private static readonly Basis[] _orthoBases = new Basis[24]
	{
		new Basis(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f),
		new Basis(0f, -1f, 0f, 1f, 0f, 0f, 0f, 0f, 1f),
		new Basis(-1f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, 1f),
		new Basis(0f, 1f, 0f, -1f, 0f, 0f, 0f, 0f, 1f),
		new Basis(1f, 0f, 0f, 0f, 0f, -1f, 0f, 1f, 0f),
		new Basis(0f, 0f, 1f, 1f, 0f, 0f, 0f, 1f, 0f),
		new Basis(-1f, 0f, 0f, 0f, 0f, 1f, 0f, 1f, 0f),
		new Basis(0f, 0f, -1f, -1f, 0f, 0f, 0f, 1f, 0f),
		new Basis(1f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, -1f),
		new Basis(0f, 1f, 0f, 1f, 0f, 0f, 0f, 0f, -1f),
		new Basis(-1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, -1f),
		new Basis(0f, -1f, 0f, -1f, 0f, 0f, 0f, 0f, -1f),
		new Basis(1f, 0f, 0f, 0f, 0f, 1f, 0f, -1f, 0f),
		new Basis(0f, 0f, -1f, 1f, 0f, 0f, 0f, -1f, 0f),
		new Basis(-1f, 0f, 0f, 0f, 0f, -1f, 0f, -1f, 0f),
		new Basis(0f, 0f, 1f, -1f, 0f, 0f, 0f, -1f, 0f),
		new Basis(0f, 0f, 1f, 0f, 1f, 0f, -1f, 0f, 0f),
		new Basis(0f, -1f, 0f, 0f, 0f, 1f, -1f, 0f, 0f),
		new Basis(0f, 0f, -1f, 0f, -1f, 0f, -1f, 0f, 0f),
		new Basis(0f, 1f, 0f, 0f, 0f, -1f, -1f, 0f, 0f),
		new Basis(0f, 0f, 1f, 0f, -1f, 0f, 1f, 0f, 0f),
		new Basis(0f, 1f, 0f, 0f, 0f, 1f, 1f, 0f, 0f),
		new Basis(0f, 0f, -1f, 0f, 1f, 0f, 1f, 0f, 0f),
		new Basis(0f, -1f, 0f, 0f, 0f, -1f, 1f, 0f, 0f)
	};

	private static readonly Basis _identity = new Basis(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f);

	private static readonly Basis _flipX = new Basis(-1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f);

	private static readonly Basis _flipY = new Basis(1f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, 1f);

	private static readonly Basis _flipZ = new Basis(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, -1f);

	public Vector3 X
	{
		readonly get
		{
			return Column0;
		}
		set
		{
			Column0 = value;
		}
	}

	public Vector3 Y
	{
		readonly get
		{
			return Column1;
		}
		set
		{
			Column1 = value;
		}
	}

	public Vector3 Z
	{
		readonly get
		{
			return Column2;
		}
		set
		{
			Column2 = value;
		}
	}

	public Vector3 Column0
	{
		readonly get
		{
			return new Vector3(Row0.X, Row1.X, Row2.X);
		}
		set
		{
			Row0.X = value.X;
			Row1.X = value.Y;
			Row2.X = value.Z;
		}
	}

	public Vector3 Column1
	{
		readonly get
		{
			return new Vector3(Row0.Y, Row1.Y, Row2.Y);
		}
		set
		{
			Row0.Y = value.X;
			Row1.Y = value.Y;
			Row2.Y = value.Z;
		}
	}

	public Vector3 Column2
	{
		readonly get
		{
			return new Vector3(Row0.Z, Row1.Z, Row2.Z);
		}
		set
		{
			Row0.Z = value.X;
			Row1.Z = value.Y;
			Row2.Z = value.Z;
		}
	}

	public readonly Vector3 Scale => Mathf.Sign(Determinant()) * new Vector3(Column0.Length(), Column1.Length(), Column2.Length());

	public Vector3 this[int column]
	{
		readonly get
		{
			return column switch
			{
				0 => Column0, 
				1 => Column1, 
				2 => Column2, 
				_ => throw new ArgumentOutOfRangeException("column"), 
			};
		}
		set
		{
			switch (column)
			{
			case 0:
				Column0 = value;
				break;
			case 1:
				Column1 = value;
				break;
			case 2:
				Column2 = value;
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
			Vector3 value2 = this[column];
			value2[row] = value;
			this[column] = value2;
		}
	}

	public static Basis Identity => _identity;

	public static Basis FlipX => _flipX;

	public static Basis FlipY => _flipY;

	public static Basis FlipZ => _flipZ;

	internal void SetQuaternionScale(Quaternion quaternion, Vector3 scale)
	{
		SetDiagonal(scale);
		Rotate(quaternion);
	}

	private void Rotate(Quaternion quaternion)
	{
		this = new Basis(quaternion) * this;
	}

	private void SetDiagonal(Vector3 diagonal)
	{
		Row0 = new Vector3(diagonal.X, 0f, 0f);
		Row1 = new Vector3(0f, diagonal.Y, 0f);
		Row2 = new Vector3(0f, 0f, diagonal.Z);
	}

	public readonly float Determinant()
	{
		float num = Row1[1] * Row2[2] - Row1[2] * Row2[1];
		float num2 = Row1[2] * Row2[0] - Row1[0] * Row2[2];
		float num3 = Row1[0] * Row2[1] - Row1[1] * Row2[0];
		return Row0[0] * num + Row0[1] * num2 + Row0[2] * num3;
	}

	public readonly Vector3 GetEuler(EulerOrder order = EulerOrder.Yxz)
	{
		if ((ulong)order <= 5uL)
		{
			switch (order)
			{
			case EulerOrder.Xyz:
			{
				float num5 = Row0[2];
				Vector3 result5 = default(Vector3);
				if (num5 < 0.999999f)
				{
					if (num5 > -0.999999f)
					{
						if (Row1[0] == 0f && Row0[1] == 0f && Row1[2] == 0f && Row2[1] == 0f && Row1[1] == 1f)
						{
							result5.X = 0f;
							result5.Y = Mathf.Atan2(Row0[2], Row0[0]);
							result5.Z = 0f;
						}
						else
						{
							result5.X = Mathf.Atan2(0f - Row1[2], Row2[2]);
							result5.Y = Mathf.Asin(num5);
							result5.Z = Mathf.Atan2(0f - Row0[1], Row0[0]);
						}
					}
					else
					{
						result5.X = Mathf.Atan2(Row2[1], Row1[1]);
						result5.Y = -(float)Math.PI / 2f;
						result5.Z = 0f;
					}
				}
				else
				{
					result5.X = Mathf.Atan2(Row2[1], Row1[1]);
					result5.Y = (float)Math.PI / 2f;
					result5.Z = 0f;
				}
				return result5;
			}
			case EulerOrder.Xzy:
			{
				float num3 = Row0[1];
				Vector3 result3 = default(Vector3);
				if (num3 < 0.999999f)
				{
					if (num3 > -0.999999f)
					{
						result3.X = Mathf.Atan2(Row2[1], Row1[1]);
						result3.Y = Mathf.Atan2(Row0[2], Row0[0]);
						result3.Z = Mathf.Asin(0f - num3);
					}
					else
					{
						result3.X = 0f - Mathf.Atan2(Row1[2], Row2[2]);
						result3.Y = 0f;
						result3.Z = (float)Math.PI / 2f;
					}
				}
				else
				{
					result3.X = 0f - Mathf.Atan2(Row1[2], Row2[2]);
					result3.Y = 0f;
					result3.Z = -(float)Math.PI / 2f;
				}
				return result3;
			}
			case EulerOrder.Yxz:
			{
				float num4 = Row1[2];
				Vector3 result4 = default(Vector3);
				if (num4 < 0.999999f)
				{
					if (num4 > -0.999999f)
					{
						if (Row1[0] == 0f && Row0[1] == 0f && Row0[2] == 0f && Row2[0] == 0f && Row0[0] == 1f)
						{
							result4.X = Mathf.Atan2(0f - num4, Row1[1]);
							result4.Y = 0f;
							result4.Z = 0f;
						}
						else
						{
							result4.X = Mathf.Asin(0f - num4);
							result4.Y = Mathf.Atan2(Row0[2], Row2[2]);
							result4.Z = Mathf.Atan2(Row1[0], Row1[1]);
						}
					}
					else
					{
						result4.X = (float)Math.PI / 2f;
						result4.Y = Mathf.Atan2(Row0[1], Row0[0]);
						result4.Z = 0f;
					}
				}
				else
				{
					result4.X = -(float)Math.PI / 2f;
					result4.Y = 0f - Mathf.Atan2(Row0[1], Row0[0]);
					result4.Z = 0f;
				}
				return result4;
			}
			case EulerOrder.Yzx:
			{
				float num2 = Row1[0];
				Vector3 result2 = default(Vector3);
				if (num2 < 0.999999f)
				{
					if (num2 > -0.999999f)
					{
						result2.X = Mathf.Atan2(0f - Row1[2], Row1[1]);
						result2.Y = Mathf.Atan2(0f - Row2[0], Row0[0]);
						result2.Z = Mathf.Asin(num2);
					}
					else
					{
						result2.X = Mathf.Atan2(Row2[1], Row2[2]);
						result2.Y = 0f;
						result2.Z = -(float)Math.PI / 2f;
					}
				}
				else
				{
					result2.X = Mathf.Atan2(Row2[1], Row2[2]);
					result2.Y = 0f;
					result2.Z = (float)Math.PI / 2f;
				}
				return result2;
			}
			case EulerOrder.Zxy:
			{
				float num6 = Row2[1];
				Vector3 result6 = default(Vector3);
				if (num6 < 0.999999f)
				{
					if (num6 > -0.999999f)
					{
						result6.X = Mathf.Asin(num6);
						result6.Y = Mathf.Atan2(0f - Row2[0], Row2[2]);
						result6.Z = Mathf.Atan2(0f - Row0[1], Row1[1]);
					}
					else
					{
						result6.X = -(float)Math.PI / 2f;
						result6.Y = Mathf.Atan2(Row0[2], Row0[0]);
						result6.Z = 0f;
					}
				}
				else
				{
					result6.X = (float)Math.PI / 2f;
					result6.Y = Mathf.Atan2(Row0[2], Row0[0]);
					result6.Z = 0f;
				}
				return result6;
			}
			case EulerOrder.Zyx:
			{
				float num = Row2[0];
				Vector3 result = default(Vector3);
				if (num < 0.999999f)
				{
					if (num > -0.999999f)
					{
						result.X = Mathf.Atan2(Row2[1], Row2[2]);
						result.Y = Mathf.Asin(0f - num);
						result.Z = Mathf.Atan2(Row1[0], Row0[0]);
					}
					else
					{
						result.X = 0f;
						result.Y = (float)Math.PI / 2f;
						result.Z = 0f - Mathf.Atan2(Row0[1], Row1[1]);
					}
				}
				else
				{
					result.X = 0f;
					result.Y = -(float)Math.PI / 2f;
					result.Z = 0f - Mathf.Atan2(Row0[1], Row1[1]);
				}
				return result;
			}
			}
		}
		throw new ArgumentOutOfRangeException("order");
	}

	internal readonly Quaternion GetQuaternion()
	{
		float num = Row0[0] + Row1[1] + Row2[2];
		if (num > 0f)
		{
			float num2 = Mathf.Sqrt(num + 1f) * 2f;
			float num3 = 1f / num2;
			return new Quaternion((Row2[1] - Row1[2]) * num3, (Row0[2] - Row2[0]) * num3, (Row1[0] - Row0[1]) * num3, num2 * 0.25f);
		}
		if (Row0[0] > Row1[1] && Row0[0] > Row2[2])
		{
			float num4 = Mathf.Sqrt(Row0[0] - Row1[1] - Row2[2] + 1f) * 2f;
			float num5 = 1f / num4;
			return new Quaternion(num4 * 0.25f, (Row0[1] + Row1[0]) * num5, (Row0[2] + Row2[0]) * num5, (Row2[1] - Row1[2]) * num5);
		}
		if (Row1[1] > Row2[2])
		{
			float num6 = Mathf.Sqrt(0f - Row0[0] + Row1[1] - Row2[2] + 1f) * 2f;
			float num7 = 1f / num6;
			return new Quaternion((Row0[1] + Row1[0]) * num7, num6 * 0.25f, (Row1[2] + Row2[1]) * num7, (Row0[2] - Row2[0]) * num7);
		}
		float num8 = Mathf.Sqrt(0f - Row0[0] - Row1[1] + Row2[2] + 1f) * 2f;
		float num9 = 1f / num8;
		return new Quaternion((Row0[2] + Row2[0]) * num9, (Row1[2] + Row2[1]) * num9, num8 * 0.25f, (Row1[0] - Row0[1]) * num9);
	}

	public readonly Quaternion GetRotationQuaternion()
	{
		Basis basis = Orthonormalized();
		if (basis.Determinant() < 0f)
		{
			basis = basis.Scaled(-Vector3.One);
		}
		return basis.GetQuaternion();
	}

	public readonly Basis Inverse()
	{
		float num = Row1[1] * Row2[2] - Row1[2] * Row2[1];
		float num2 = Row1[2] * Row2[0] - Row1[0] * Row2[2];
		float num3 = Row1[0] * Row2[1] - Row1[1] * Row2[0];
		float num4 = Row0[0] * num + Row0[1] * num2 + Row0[2] * num3;
		if (num4 == 0f)
		{
			throw new InvalidOperationException("Matrix determinant is zero and cannot be inverted.");
		}
		float num5 = 1f / num4;
		float num6 = Row0[2] * Row2[1] - Row0[1] * Row2[2];
		float num7 = Row0[1] * Row1[2] - Row0[2] * Row1[1];
		float num8 = Row0[0] * Row2[2] - Row0[2] * Row2[0];
		float num9 = Row0[2] * Row1[0] - Row0[0] * Row1[2];
		float num10 = Row0[1] * Row2[0] - Row0[0] * Row2[1];
		float num11 = Row0[0] * Row1[1] - Row0[1] * Row1[0];
		return new Basis(num * num5, num6 * num5, num7 * num5, num2 * num5, num8 * num5, num9 * num5, num3 * num5, num10 * num5, num11 * num5);
	}

	public readonly bool IsFinite()
	{
		if (Row0.IsFinite() && Row1.IsFinite())
		{
			return Row2.IsFinite();
		}
		return false;
	}

	internal readonly Basis Lerp(Basis to, float weight)
	{
		Basis result = this;
		result.Row0 = Row0.Lerp(to.Row0, weight);
		result.Row1 = Row1.Lerp(to.Row1, weight);
		result.Row2 = Row2.Lerp(to.Row2, weight);
		return result;
	}

	public static Basis LookingAt(Vector3 target, Vector3? up = null, bool useModelFront = false)
	{
		Vector3 valueOrDefault = up.GetValueOrDefault();
		if (!up.HasValue)
		{
			valueOrDefault = Vector3.Up;
			up = valueOrDefault;
		}
		Vector3 vector = target.Normalized();
		if (!useModelFront)
		{
			vector = -vector;
		}
		Vector3 vector2 = up.Value.Cross(vector);
		if (vector2.IsZeroApprox())
		{
			throw new ArgumentException("Target and up vectors are colinear. This is not advised as it may cause unwanted rotation around local Z axis.");
		}
		vector2.Normalize();
		Vector3 column = vector.Cross(vector2);
		return new Basis(vector2, column, vector);
	}

	[EditorBrowsable(EditorBrowsableState.Never)]
	public static Basis LookingAt(Vector3 target, Vector3 up)
	{
		return LookingAt(target, up, useModelFront: false);
	}

	public readonly Basis Orthonormalized()
	{
		Vector3 vector = this[0];
		Vector3 vector2 = this[1];
		Vector3 vector3 = this[2];
		vector.Normalize();
		vector2 -= vector * vector.Dot(vector2);
		vector2.Normalize();
		vector3 = vector3 - vector * vector.Dot(vector3) - vector2 * vector2.Dot(vector3);
		vector3.Normalize();
		return new Basis(vector, vector2, vector3);
	}

	public readonly Basis Rotated(Vector3 axis, float angle)
	{
		return new Basis(axis, angle) * this;
	}

	public readonly Basis Scaled(Vector3 scale)
	{
		Basis result = this;
		result.Row0 *= scale.X;
		result.Row1 *= scale.Y;
		result.Row2 *= scale.Z;
		return result;
	}

	public readonly Basis ScaledLocal(Vector3 scale)
	{
		Basis result = this;
		result.Row0 *= scale;
		result.Row1 *= scale;
		result.Row2 *= scale;
		return result;
	}

	public readonly Basis Slerp(Basis target, float weight)
	{
		Quaternion quaternion = new Quaternion(this);
		Quaternion to = new Quaternion(target);
		Basis result = new Basis(quaternion.Slerp(to, weight));
		result.Row0 *= Mathf.Lerp(Row0.Length(), target.Row0.Length(), weight);
		result.Row1 *= Mathf.Lerp(Row1.Length(), target.Row1.Length(), weight);
		result.Row2 *= Mathf.Lerp(Row2.Length(), target.Row2.Length(), weight);
		return result;
	}

	public readonly float Tdotx(Vector3 with)
	{
		return Row0[0] * with[0] + Row1[0] * with[1] + Row2[0] * with[2];
	}

	public readonly float Tdoty(Vector3 with)
	{
		return Row0[1] * with[0] + Row1[1] * with[1] + Row2[1] * with[2];
	}

	public readonly float Tdotz(Vector3 with)
	{
		return Row0[2] * with[0] + Row1[2] * with[1] + Row2[2] * with[2];
	}

	public readonly Basis Transposed()
	{
		Basis result = this;
		result.Row0[1] = Row1[0];
		result.Row1[0] = Row0[1];
		result.Row0[2] = Row2[0];
		result.Row2[0] = Row0[2];
		result.Row1[2] = Row2[1];
		result.Row2[1] = Row1[2];
		return result;
	}

	public Basis(Quaternion quaternion)
	{
		float num = 2f / quaternion.LengthSquared();
		float num2 = quaternion.X * num;
		float num3 = quaternion.Y * num;
		float num4 = quaternion.Z * num;
		float num5 = quaternion.W * num2;
		float num6 = quaternion.W * num3;
		float num7 = quaternion.W * num4;
		float num8 = quaternion.X * num2;
		float num9 = quaternion.X * num3;
		float num10 = quaternion.X * num4;
		float num11 = quaternion.Y * num3;
		float num12 = quaternion.Y * num4;
		float num13 = quaternion.Z * num4;
		Row0 = new Vector3(1f - (num11 + num13), num9 - num7, num10 + num6);
		Row1 = new Vector3(num9 + num7, 1f - (num8 + num13), num12 - num5);
		Row2 = new Vector3(num10 - num6, num12 + num5, 1f - (num8 + num11));
	}

	public Basis(Vector3 axis, float angle)
	{
		Vector3 vector = new Vector3(axis.X * axis.X, axis.Y * axis.Y, axis.Z * axis.Z);
		(float Sin, float Cos) tuple = Mathf.SinCos(angle);
		float item = tuple.Sin;
		float item2 = tuple.Cos;
		Row0.X = vector.X + item2 * (1f - vector.X);
		Row1.Y = vector.Y + item2 * (1f - vector.Y);
		Row2.Z = vector.Z + item2 * (1f - vector.Z);
		float num = 1f - item2;
		float num2 = axis.X * axis.Y * num;
		float num3 = axis.Z * item;
		Row0.Y = num2 - num3;
		Row1.X = num2 + num3;
		num2 = axis.X * axis.Z * num;
		num3 = axis.Y * item;
		Row0.Z = num2 + num3;
		Row2.X = num2 - num3;
		num2 = axis.Y * axis.Z * num;
		num3 = axis.X * item;
		Row1.Z = num2 - num3;
		Row2.Y = num2 + num3;
	}

	public Basis(Vector3 column0, Vector3 column1, Vector3 column2)
	{
		Row0 = new Vector3(column0.X, column1.X, column2.X);
		Row1 = new Vector3(column0.Y, column1.Y, column2.Y);
		Row2 = new Vector3(column0.Z, column1.Z, column2.Z);
	}

	public Basis(float xx, float yx, float zx, float xy, float yy, float zy, float xz, float yz, float zz)
	{
		Row0 = new Vector3(xx, yx, zx);
		Row1 = new Vector3(xy, yy, zy);
		Row2 = new Vector3(xz, yz, zz);
	}

	public static Basis FromEuler(Vector3 euler, EulerOrder order = EulerOrder.Yxz)
	{
		(float Sin, float Cos) tuple = Mathf.SinCos(euler.X);
		float item = tuple.Sin;
		float item2 = tuple.Cos;
		Basis basis = new Basis(new Vector3(1f, 0f, 0f), new Vector3(0f, item2, item), new Vector3(0f, 0f - item, item2));
		(float Sin, float Cos) tuple2 = Mathf.SinCos(euler.Y);
		item = tuple2.Sin;
		item2 = tuple2.Cos;
		Basis basis2 = new Basis(new Vector3(item2, 0f, 0f - item), new Vector3(0f, 1f, 0f), new Vector3(item, 0f, item2));
		(float Sin, float Cos) tuple3 = Mathf.SinCos(euler.Z);
		item = tuple3.Sin;
		item2 = tuple3.Cos;
		Basis basis3 = new Basis(new Vector3(item2, item, 0f), new Vector3(0f - item, item2, 0f), new Vector3(0f, 0f, 1f));
		if ((ulong)order <= 5uL)
		{
			switch (order)
			{
			case EulerOrder.Xyz:
				return basis * basis2 * basis3;
			case EulerOrder.Xzy:
				return basis * basis3 * basis2;
			case EulerOrder.Yxz:
				return basis2 * basis * basis3;
			case EulerOrder.Yzx:
				return basis2 * basis3 * basis;
			case EulerOrder.Zxy:
				return basis3 * basis * basis2;
			case EulerOrder.Zyx:
				return basis3 * basis2 * basis;
			}
		}
		throw new ArgumentOutOfRangeException("order");
	}

	public static Basis FromScale(Vector3 scale)
	{
		return new Basis(scale.X, 0f, 0f, 0f, scale.Y, 0f, 0f, 0f, scale.Z);
	}

	public static Basis operator *(Basis left, Basis right)
	{
		return new Basis(right.Tdotx(left.Row0), right.Tdoty(left.Row0), right.Tdotz(left.Row0), right.Tdotx(left.Row1), right.Tdoty(left.Row1), right.Tdotz(left.Row1), right.Tdotx(left.Row2), right.Tdoty(left.Row2), right.Tdotz(left.Row2));
	}

	public static Vector3 operator *(Basis basis, Vector3 vector)
	{
		return new Vector3(basis.Row0.Dot(vector), basis.Row1.Dot(vector), basis.Row2.Dot(vector));
	}

	public static Vector3 operator *(Vector3 vector, Basis basis)
	{
		return new Vector3(basis.Row0[0] * vector.X + basis.Row1[0] * vector.Y + basis.Row2[0] * vector.Z, basis.Row0[1] * vector.X + basis.Row1[1] * vector.Y + basis.Row2[1] * vector.Z, basis.Row0[2] * vector.X + basis.Row1[2] * vector.Y + basis.Row2[2] * vector.Z);
	}

	public static bool operator ==(Basis left, Basis right)
	{
		return left.Equals(right);
	}

	public static bool operator !=(Basis left, Basis right)
	{
		return !left.Equals(right);
	}

	public override readonly bool Equals([NotNullWhen(true)] object? obj)
	{
		if (obj is Basis other)
		{
			return Equals(other);
		}
		return false;
	}

	public readonly bool Equals(Basis other)
	{
		if (Row0.Equals(other.Row0) && Row1.Equals(other.Row1))
		{
			return Row2.Equals(other.Row2);
		}
		return false;
	}

	public readonly bool IsEqualApprox(Basis other)
	{
		if (Row0.IsEqualApprox(other.Row0) && Row1.IsEqualApprox(other.Row1))
		{
			return Row2.IsEqualApprox(other.Row2);
		}
		return false;
	}

	public override readonly int GetHashCode()
	{
		return HashCode.Combine(Row0, Row1, Row2);
	}

	public override readonly string ToString()
	{
		return ToString(null);
	}

	public readonly string ToString(string? format)
	{
		return $"[X: {X.ToString(format)}, Y: {Y.ToString(format)}, Z: {Z.ToString(format)}]";
	}
}
