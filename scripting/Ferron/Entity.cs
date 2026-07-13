using System.Runtime.InteropServices;

namespace Ferron;

// Layout must match Rust CEntity.
[StructLayout(LayoutKind.Sequential)]
public readonly struct Entity
{
    public readonly uint Index;
    public readonly uint Generation;

    public Entity(uint index, uint generation)
    {
        Index = index;
        Generation = generation;
    }

    /// True while this handle is live and the entity has a T component.
    public bool HasComponent<T>() where T : struct =>
        Native.HasComponent(this, ComponentKinds.Of<T>());

    /// Typed read of an engine component, or null if the entity lacks it (or
    /// the handle is stale). Components come back by value; write back through
    /// the corresponding setter (Native.SetTransform, World.SetTag). v1
    /// supports Transform and Tag — script-defined components arrive with the
    /// byte-blob milestone.
    public T? GetComponent<T>() where T : struct
    {
        // typeof dispatch per supported component; the (T)(object) round-trip
        // boxes, which is fine at scripting call rates.
        if (typeof(T) == typeof(Transform))
            return HasComponent<Transform>() ? (T)(object)Native.GetTransform(this) : null;
        if (typeof(T) == typeof(Tag))
            return Native.GetTag(this) is { } tag ? (T)(object)new Tag(tag) : null;
        throw new ArgumentException($"{typeof(T).Name} is not a script-visible engine component");
    }

    public override string ToString() => $"Entity({Index}v{Generation})";
}
