using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;

namespace Godot.Collections
{
    // Godot's variant collections, backed by managed List / Dictionary with the same interface set, so game code
    // (and LINQ over GetChildren() and friends) can iterate, count and index them.

    public sealed partial class Array : IList<Variant>, ICollection<Variant>, IEnumerable<Variant>, IEnumerable, IReadOnlyList<Variant>,
        IReadOnlyCollection<Variant>, ICollection, IDisposable
    {
        private readonly List<Variant> _l = new();
        public Array() { }
        public Array(IEnumerable<Variant> collection) { _l.AddRange(collection); }
        public void Dispose() { }
        public Variant this[int index] { get => _l[index]; set => _l[index] = value; }
        public int Count => _l.Count;
        public bool IsReadOnly => false;
        public bool IsSynchronized => false;
        public object SyncRoot => _l;
        public void Add(Variant item) => _l.Add(item);
        public void Clear() => _l.Clear();
        public bool Contains(Variant item) => _l.Contains(item);
        public void CopyTo(Variant[] array, int arrayIndex) => _l.CopyTo(array, arrayIndex);
        void ICollection.CopyTo(System.Array array, int index) => ((ICollection)_l).CopyTo(array, index);
        public int IndexOf(Variant item) => _l.IndexOf(item);
        public void Insert(int index, Variant item) => _l.Insert(index, item);
        public bool Remove(Variant item) => _l.Remove(item);
        public void RemoveAt(int index) => _l.RemoveAt(index);
        public IEnumerator<Variant> GetEnumerator() => _l.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    public sealed partial class Array<T> : IList<T>, ICollection<T>, IEnumerable<T>, IEnumerable, IReadOnlyList<T>, IReadOnlyCollection<T>
    {
        private readonly List<T> _l = new();
        public Array() { }
        public Array(IEnumerable<T> collection) { _l.AddRange(collection); }
        public T this[int index] { get => _l[index]; set => _l[index] = value; }
        public int Count => _l.Count;
        public bool IsReadOnly => false;
        public void Add(T item) => _l.Add(item);
        public void Clear() => _l.Clear();
        public bool Contains(T item) => _l.Contains(item);
        public void CopyTo(T[] array, int arrayIndex) => _l.CopyTo(array, arrayIndex);
        public int IndexOf(T item) => _l.IndexOf(item);
        public void Insert(int index, T item) => _l.Insert(index, item);
        public bool Remove(T item) => _l.Remove(item);
        public void RemoveAt(int index) => _l.RemoveAt(index);
        public IEnumerator<T> GetEnumerator() => _l.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
        public static explicit operator Array<T>(Variant from) => from.AsGodotArray<T>();
        public static implicit operator Variant(Array<T> from) => Variant.CreateFrom(from);
    }

    public sealed partial class Dictionary : IDictionary<Variant, Variant>, ICollection<KeyValuePair<Variant, Variant>>,
        IEnumerable<KeyValuePair<Variant, Variant>>, IEnumerable, IReadOnlyDictionary<Variant, Variant>,
        IReadOnlyCollection<KeyValuePair<Variant, Variant>>, IDisposable
    {
        private readonly System.Collections.Generic.Dictionary<Variant, Variant> _d = new();
        public Dictionary() { }
        public void Dispose() { }
        public Variant this[Variant key] { get => _d[key]; set => _d[key] = value; }
        public ICollection<Variant> Keys => _d.Keys;
        public ICollection<Variant> Values => _d.Values;
        IEnumerable<Variant> IReadOnlyDictionary<Variant, Variant>.Keys => _d.Keys;
        IEnumerable<Variant> IReadOnlyDictionary<Variant, Variant>.Values => _d.Values;
        public int Count => _d.Count;
        public bool IsReadOnly => false;
        public void Add(Variant key, Variant value) => _d.Add(key, value);
        public void Add(KeyValuePair<Variant, Variant> item) => _d.Add(item.Key, item.Value);
        public void Clear() => _d.Clear();
        public bool Contains(KeyValuePair<Variant, Variant> item) => ((ICollection<KeyValuePair<Variant, Variant>>)_d).Contains(item);
        public bool ContainsKey(Variant key) => _d.ContainsKey(key);
        public void CopyTo(KeyValuePair<Variant, Variant>[] array, int arrayIndex) => ((ICollection<KeyValuePair<Variant, Variant>>)_d).CopyTo(array, arrayIndex);
        public bool Remove(Variant key) => _d.Remove(key);
        public bool Remove(KeyValuePair<Variant, Variant> item) => ((ICollection<KeyValuePair<Variant, Variant>>)_d).Remove(item);
        public bool TryGetValue(Variant key, out Variant value) => _d.TryGetValue(key, out value);
        public IEnumerator<KeyValuePair<Variant, Variant>> GetEnumerator() => _d.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    public sealed partial class Dictionary<TKey, TValue> : IDictionary<TKey, TValue>, ICollection<KeyValuePair<TKey, TValue>>,
        IEnumerable<KeyValuePair<TKey, TValue>>, IEnumerable, IReadOnlyDictionary<TKey, TValue>, IReadOnlyCollection<KeyValuePair<TKey, TValue>>
    {
        private readonly System.Collections.Generic.Dictionary<TKey, TValue> _d = new();
        public Dictionary() { }
        public TValue this[TKey key] { get => _d[key]; set => _d[key] = value; }
        public ICollection<TKey> Keys => _d.Keys;
        public ICollection<TValue> Values => _d.Values;
        IEnumerable<TKey> IReadOnlyDictionary<TKey, TValue>.Keys => _d.Keys;
        IEnumerable<TValue> IReadOnlyDictionary<TKey, TValue>.Values => _d.Values;
        public int Count => _d.Count;
        public bool IsReadOnly => false;
        public void Add(TKey key, TValue value) => _d.Add(key, value);
        public void Add(KeyValuePair<TKey, TValue> item) => _d.Add(item.Key, item.Value);
        public void Clear() => _d.Clear();
        public bool Contains(KeyValuePair<TKey, TValue> item) => ((ICollection<KeyValuePair<TKey, TValue>>)_d).Contains(item);
        public bool ContainsKey(TKey key) => _d.ContainsKey(key);
        public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex) => ((ICollection<KeyValuePair<TKey, TValue>>)_d).CopyTo(array, arrayIndex);
        public bool Remove(TKey key) => _d.Remove(key);
        public bool Remove(KeyValuePair<TKey, TValue> item) => ((ICollection<KeyValuePair<TKey, TValue>>)_d).Remove(item);
        public bool TryGetValue(TKey key, [MaybeNullWhen(false)] out TValue value) => _d.TryGetValue(key, out value!);
        public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator() => _d.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
