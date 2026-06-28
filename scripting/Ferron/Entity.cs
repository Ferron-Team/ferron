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

    public override string ToString() => $"Entity({Index}v{Generation})";
}
