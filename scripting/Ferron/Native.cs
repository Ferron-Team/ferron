using System.Runtime.InteropServices;
using System.Text;

namespace Ferron;

// Field order and types must match the Rust FerronApi struct.
[StructLayout(LayoutKind.Sequential)]
public unsafe struct FerronApi
{
    public delegate* unmanaged<byte*, void> Log;
    public delegate* unmanaged<Entity> Spawn;
    public delegate* unmanaged<Entity, Transform*, byte> GetTransform;
    public delegate* unmanaged<Entity, Transform*, byte> SetTransform;
}

public static unsafe class Native
{
    private static FerronApi _api;

    internal static void Initialize(FerronApi* api) => _api = *api;

    public static void Log(string message)
    {
        var bytes = Encoding.UTF8.GetBytes(message);
        Span<byte> buffer = stackalloc byte[bytes.Length + 1];
        bytes.CopyTo(buffer);
        buffer[bytes.Length] = 0;
        fixed (byte* p = buffer)
            _api.Log(p);
    }

    public static Entity Spawn() => _api.Spawn();

    public static Transform GetTransform(Entity entity)
    {
        Transform transform = default;
        _api.GetTransform(entity, &transform);
        return transform;
    }

    public static void SetTransform(Entity entity, Transform value) =>
        _api.SetTransform(entity, &value);
}
