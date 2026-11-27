# Semantic part of the headless GodotSharp stub

Types game logic computes with, with Godot's semantics (everything else is generated, inert, in `../Generated/`).

- `Mathf.cs`, `Vector2.cs`, `Vector2I.cs`, `Vector3.cs`, `Vector3I.cs`, `Rect2.cs`, `Rect2I.cs`, `Color.cs`, `Colors.cs`,
  `Transform2D.cs`, `Basis.cs`, `Quaternion.cs`, `Side.cs`, `Corner.cs`, `EulerOrder.cs`, `ClockDirection.cs` are the
  C# sources of Godot 4.5's GodotSharp (as shipped with the game), unchanged except that the types are declared
  `partial` and `Color`'s four OK-HSL members, which call into the native engine, are removed.
  Godot Engine is Copyright (c) 2014-present Godot Engine contributors and Copyright (c) 2007-2014 Juan Linietsky,
  Ariel Manzur, released under the MIT license (https://godotengine.org/license).
- `Strings.cs` (StringName, NodePath, StringExtensions), `Callable.cs`, `Variant.cs`, `Collections.cs`, `Objects.cs`
  (GodotObject root, signals and awaiters, GD) are written for the headless engine with the same API shape.

A member of one of these types that `sts2.dll` starts to reference after a game update makes
`engine/tools/gen_stubs.sh` fail with `SEMANTIC ... missing`: add it here from Godot's source, keeping Godot's behaviour.
